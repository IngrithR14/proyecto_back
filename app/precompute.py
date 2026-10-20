"""Precomputa los resultados pesados y los guarda en `cache/*.json` (FASE 3).

La API (`app/main.py`) sirve desde la cache para ser instantánea. Este módulo es
idempotente y reanudable: si un K ya está cacheado, se omite. Uso:

    python -m app.precompute                 # todo (EDA + ruteo + MC + protocolo)
    python -m app.precompute --rapido        # sólo EDA + ruteo + MC en K_ACTUAL
    python -m app.precompute --k 40 50 58    # sólo esos K del MC
    python -m app.precompute --rehacer       # ignora todas las caches

Fases del protocolo de evaluación (secciones 1-3), cada una con su cache:

    seleccion.json       λ, K y β por K-Fold sobre el 80 % de train (test sellado)
    pliegues.json        K-Fold con test por inserción + arrepentimiento
    evaluacion.json      estrategias contra ACTUAL: 8 dimensiones + regla
    comparacion_rutas.json  NN vs ACO por pares de ruta (REANUDABLE por ruta)

El MonteCarlo completo (36 valores de K con rutas diarias) tarda ~20 min; el
experimento por ruta ~15 min en paralelo. Se corren una vez y quedan en cache.
Nada de cifras inventadas: todo sale del núcleo.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from . import config as cfg
from .core import comparacion, eda, pliegues
from .core.calendario import construir_calendario, resumen_carga
from .core.estructura import (asignacion_capacidad, construir_actual,
                              construir_rutas_diarias, rutas_primer_ciclo,
                              validar_cubierta)
from .core.montecarlo import (carga_por_territorio_fecha, comparar_pareada,
                              costo_escenario, evaluar_estrategia,
                              factibilidad, generar_entradas,
                              horas_por_escenario, resumen_estrategia)
from .core.universo import cargar_universo, resumen_universos


def _escribe(nombre: str, datos: Any) -> None:
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    ruta = os.path.join(cfg.CACHE_DIR, nombre)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, default=_np)
    print(f"  ✓ cache/{nombre} ({os.path.getsize(ruta)/1024:.0f} KB)")


def _np(o):
    """Convierte numpy→python para json."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def _lee(nombre: str) -> Any | None:
    ruta = os.path.join(cfg.CACHE_DIR, nombre)
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Evaluación de una estrategia (ACTUAL o K) con MonteCarlo
# ---------------------------------------------------------------------------
def _km_por_territorio(rutas: list[dict]) -> dict:
    acc: dict = {}
    for r in rutas:
        acc[r["territorio"]] = acc.get(r["territorio"], 0.0) + r["km_base"]
    return acc


def evaluar(clientes: pd.DataFrame, labels: np.ndarray | None,
            asistencia: np.ndarray, trafico: np.ndarray,
            nombre: str) -> tuple[dict, np.ndarray]:
    """Rutas diarias → factibilidad + costo (D3) + resumen. Devuelve (resumen, costo_total)."""
    rutas = construir_rutas_diarias(clientes, labels=labels)

    ctf = carga_por_territorio_fecha(clientes, labels)
    fact = factibilidad(ctf)
    ev = evaluar_estrategia(rutas, asistencia, trafico)
    n_asesores = len(ev["territorios"])
    costo = costo_escenario(ev, trafico, n_asesores)
    horas = horas_por_escenario(ctf, _km_por_territorio(rutas), trafico)
    visitas_prog = int(ctf["visitas"].sum())
    res = resumen_estrategia(nombre, ev, costo, n_asesores, fact, horas, visitas_prog)
    res["n_rutas_dia"] = len(rutas)
    res["cobertura_exacta"] = validar_cubierta(rutas, len(clientes))
    return res, np.asarray(costo["total"])


def _labels_de(df: pd.DataFrame, K: int | None) -> np.ndarray | None:
    if K is None:
        return None
    return asignacion_capacidad(df, K)


def precompute_mc(ks: list[int], n_sim: int | None = None,
                  rehacer: bool = False) -> None:
    """MonteCarlo para cada K (y ACTUAL como referencia) → cache/montecarlo.json."""
    n_sim = n_sim or cfg.N_SIM
    cache = {} if rehacer else (_lee("montecarlo.json") or {})
    actual_cache = cache.get("actual")
    estrategias = cache.get("estrategias", {})
    comparaciones = cache.get("comparaciones", {})

    u = cargar_universo()
    df = u.clientes.copy().reset_index(drop=True)
    df["_uid"] = np.arange(len(df))
    from .core.geo import proyectar
    proyectar(df, col_grupo=None)
    n_cli = len(df)

    # números aleatorios comunes (paired comparison) — una sola vez
    asistencia, trafico = generar_entradas(n_sim, n_cli, cfg.SEMILLA)

    t0 = time.time()
    if actual_cache is None or rehacer:
        print("• ACTUAL (K=None → ZON_REP)…")
        res, costo_actual = evaluar(df, None, asistencia, trafico, "ACTUAL")
        actual_cache = res
        costo_actual_ref = costo_actual
    else:
        # regenerar costo_actual para la comparación pareada si hace falta
        print("• ACTUAL desde cache (regenerando costo para comparación)…")
        rutas_a = construir_rutas_diarias(df, labels=None)
        ev_a = evaluar_estrategia(rutas_a, asistencia, trafico)
        costo_actual_ref = np.asarray(
            costo_escenario(ev_a, trafico, len(ev_a["territorios"]))["total"])
        res = actual_cache

    pendientes = [k for k in ks if str(k) not in estrategias or rehacer]
    print(f"• K pendientes: {pendientes if pendientes else '(ninguno)'}")
    for K in pendientes:
        t1 = time.time()
        labels = _labels_de(df, K)
        res_k, costo_k = evaluar(df, labels, asistencia, trafico, f"K={K}")
        estrategias[str(K)] = res_k
        comparaciones[str(K)] = comparar_pareada(costo_k, costo_actual_ref)
        print(f"  K={K:>2}: pico={res_k['pico_carga']} factible={res_k['factible']} "
              f"costo={res_k['costo_medio_M']:.1f}M cobertura={res_k['cobertura']:.3f} "
              f"({time.time()-t1:.0f}s)")
        # guardado incremental (reanudable)
        _escribe("montecarlo.json", {"actual": actual_cache, "n_sim": n_sim,
                                     "estrategias": estrategias,
                                     "comparaciones": comparaciones})
    _escribe("montecarlo.json", {"actual": actual_cache, "n_sim": n_sim,
                                 "estrategias": estrategias,
                                 "comparaciones": comparaciones})
    print(f"MonteCarlo listo en {time.time()-t0:.0f}s")


def precompute_general() -> None:
    """EDA + universo + parámetros + experimento NN vs ACO (rápido)."""
    print("• params…")
    _escribe("params.json", {"params": cfg.params(),
                             "cap_dinamico": cfg.cap_dinamico(),
                             "costo_fijo_ciclo": cfg.costo_fijo_ciclo(),
                             "k_min": cfg.K_MIN, "k_max": cfg.K_MAX,
                             "k_actual": cfg.K_ACTUAL})

    print("• universos…")
    u = cargar_universo()
    _escribe("universos.json", resumen_universos(u))

    print("• calendario (D2)…")
    va, meta_cal = construir_calendario(u.clientes)
    rc = resumen_carga(va)
    _escribe("calendario.json", {"meta": meta_cal, "resumen": rc})

    print("• EDA…")
    base = eda.cargar_base()
    _escribe("eda.json", eda.resumen_eda(base))

    print("• experimento NN vs ACO…")
    exp = comparacion.experimento_nn_vs_aco(u.clientes)
    _escribe("ruteo.json", exp)


def precompute_geo(ks: list[int]) -> None:
    """Geografía por K para el mapa (Leaflet): centroide + clientes por territorio."""
    cache = _lee("geo.json") or {}
    u = cargar_universo()
    df = u.clientes.copy().reset_index(drop=True)
    df["_uid"] = np.arange(len(df))
    from .core.geo import proyectar
    proyectar(df, col_grupo=None)
    for K in ks:
        key = "actual" if K is None else str(K)
        if key in cache:
            continue
        labels = _labels_de(df, K)
        d = df.copy()
        d["_t"] = labels if labels is not None else d[cfg.COL_ASESOR].values
        # muestrea puntos para no mandar 14k al navegador por cada K
        pts = d[["Latitud", "Longitud", "_t"]].dropna()
        if len(pts) > 4000:
            pts = pts.sample(4000, random_state=cfg.SEMILLA)
        terr = (d.groupby("_t")
                .agg(n=("CodigoCliente", "size"),
                     visitas=("visitas_mes_rep", "sum"),
                     lat=("Latitud", "mean"), lon=("Longitud", "mean"))
                .reset_index())
        cache[key] = {
            "territorios": [{
                "id": _np(r["_t"]), "n": int(r["n"]), "visitas": int(r["visitas"]),
                "lat": round(float(r["lat"]), 5), "lon": round(float(r["lon"]), 5),
            } for _, r in terr.iterrows()],
            "puntos": [{"lat": round(float(r["Latitud"]), 5),
                        "lon": round(float(r["Longitud"]), 5),
                        "t": _np(r["_t"])} for _, r in pts.iterrows()],
        }
        _escribe("geo.json", cache)
        print(f"  geo K={key}: {len(terr)} territorios, {len(pts)} puntos")


def _particion() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partición 80/20 estratificada por ZON_REP + universo con `_uid` y proyección.

    Es LA MISMA partición para selección, comparación NN-ACO y evaluación de
    estrategias: `SEMILLA_MODELO` fija el sorteo y el test queda sellado.
    """
    from .core.ajuste import particionar
    from .core.geo import proyectar

    u = cargar_universo()
    df = u.clientes.copy().reset_index(drop=True)
    df["_uid"] = np.arange(len(df))
    if "x_km" not in df.columns:
        proyectar(df, col_grupo=None)
    train, test = particionar(df)
    return train.reset_index(drop=True), test.reset_index(drop=True), df


def precompute_seleccion(rehacer: bool = False) -> dict | None:
    """Selección de λ, K y β por K-Fold sobre el 80 % de train → cache/seleccion.json.

    Sólo train: el test del 20 % NUNCA se usa aquí (queda sellado para la
    evaluación final). La rejilla está declarada en `config` (LAM_GRID, K_CANDIDATOS).
    """
    if not rehacer and (prev := _lee("seleccion.json")):
        print("• selección (λ, K, β) ya en cache")
        return prev
    print("• Selección de λ, K y β (K-Fold %d pliegues, sólo train)…" % cfg.K_PLIEGUES)
    t0 = time.time()
    train, _test, _uni = _particion()
    sel = pliegues.seleccionar_config(train)
    sel["tiempo_s"] = round(time.time() - t0, 1)
    _escribe("seleccion.json", sel)
    print(f"  λ={sel['lambda_elegida']}  K={sel['K_elegido']}  "
          f"β={sel['beta_elegido']}  ({sel['tiempo_s']} s)")
    return sel


def precompute_pliegues(rehacer: bool = False) -> None:
    """K-Fold del protocolo (con test insertado) sobre la estrategia elegida
    → cache/pliegues.json (estabilidad entre pliegues + arrepentimiento)."""
    if not rehacer and _lee("pliegues.json"):
        print("• pliegues (K-Fold) ya en cache")
        return
    print("• K-Fold del protocolo (pliegues con test por inserción)…")
    t0 = time.time()
    train, _test, _uni = _particion()
    sel = _lee("seleccion.json") or {}
    lam = sel.get("lambda_elegida")
    beta = sel.get("beta_elegido")
    estrategia = "rebalanceo" if cfg.REBAL_POBLACION else "actual"
    kf = pliegues.pliegues_ajuste(train, estrategia=estrategia, lam_norm=lam,
                                  beta=beta, con_arrepentimiento=True)
    kf["estrategia"] = estrategia
    kf["tiempo_s"] = round(time.time() - t0, 1)
    _escribe("pliegues.json", kf)
    print(f"  {cfg.K_PLIEGUES} pliegues en {kf['tiempo_s']} s")


def precompute_comparacion_rutas(rehacer: bool = False,
                                 n_jobs: int | None = None) -> None:
    """NN vs ACO por pares de ruta → cache/comparacion_rutas.json (REANUDABLE).

    Si el proceso se interrumpe, la siguiente corrida sólo ejecuta las rutas
    faltantes (está en `cache['rutas']` del propio JSON).
    """
    from .core import comparacion_rutas
    from .core.ajuste import ajustar

    if rehacer:
        prev = None
    else:
        prev = _lee("comparacion_rutas.json")
    if prev and prev.get("meta", {}).get("completadas") == prev.get("meta", {}).get("n_rutas"):
        print("• comparación NN vs ACO ya en cache (%s rutas)" % prev["meta"]["n_rutas"])
        return

    print("• Comparación NN vs ACO por pares de ruta (reanudable)…")
    t0 = time.time()
    train, test, universo = _particion()
    sel = _lee("seleccion.json") or {}
    beta = sel.get("beta_elegido", cfg.ACO_BETA)
    # estrategia recomendada (regla de la sección 3); si no, ACTUAL
    estrategia, K = "actual", None
    rec = ((_lee("evaluacion.json") or {}).get("informe") or {}).get("regla", {}).get("recomendada")
    if rec and str(rec).startswith("REBAL_"):
        estrategia = "rebalanceo"
    elif rec and str(rec).startswith("K"):
        estrategia, K = "k", int("".join(c for c in str(rec) if c.isdigit()))
    mod = ajustar(train, estrategia=estrategia, K=K, beta=beta,
                  universo_rebalanceo=universo if estrategia == "rebalanceo" else None)

    cache = (prev or {}).get("rutas") and {"rutas": prev["rutas"],
                                           "meta_rutas": prev.get("meta_rutas", {}),
                                           "meta": prev.get("meta", {})}
    exp = comparacion_rutas.experimento_nn_vs_aco(mod, train, test, cache=cache,
                                                  beta=beta, n_jobs=n_jobs)
    _escribe("comparacion_rutas.json",
             {k: v for k, v in exp.items() if k != "cache"} | {"cache": exp["cache"]})
    print(f"  {exp['meta']['n_rutas']} rutas · ahorro medio "
          f"{exp['global']['ahorro_medio_pct']} % en {time.time() - t0:.0f} s")


def precompute_evaluacion(rehacer: bool = False) -> None:
    """Estrategias contra ACTUAL (8 dimensiones + regla) → cache/evaluacion.json."""
    from .core.evaluacion import evaluacion_completa

    if not rehacer and _lee("evaluacion.json"):
        print("• evaluación de estrategias ya en cache")
        return
    print("• Evaluación de estrategias contra ACTUAL (8 dimensiones + regla)…")
    t0 = time.time()
    train, test, universo = _particion()
    sel = _lee("seleccion.json") or {}
    est = evaluacion_completa(train, test, universo,
                              lam_norm=sel.get("lambda_elegida"),
                              beta=sel.get("beta_elegido"))
    est["tiempo_s"] = round(time.time() - t0, 1)
    _escribe("evaluacion.json", est)
    regla = est["informe"]["regla"]
    print(f"  regla: {regla['decision']} → {regla['recomendada'] or 'ninguna'} "
          f"({est['tiempo_s']} s)")


def precompute_rutas() -> None:
    """Rutas del 1er ciclo (Q1 y Q2) para ACTUAL y la recomendada → cache/rutas.json.

    Son la VISUALIZACIÓN del plan desplegado: la estrategia se ajusta con el
    protocolo (`ajustar(train)`) y los territorios de los clientes de test salen
    de la misma regla del modelo. Ninguna métrica se calcula aquí.
    """
    print("• Rutas del primer ciclo (por día y asesor)…")
    from .core.ajuste import ajustar

    train, test, universo = _particion()
    regla = ((_lee("evaluacion.json") or {}).get("informe") or {}).get("regla") or {}
    rec = regla.get("recomendada")
    es_recomendada = bool(rec)
    estrategia, K = "rebalanceo", None
    if rec and str(rec).startswith("K"):
        estrategia, K = "k", int("".join(c for c in str(rec) if c.isdigit()))

    cache: dict[str, Any] = {
        "actual": {
            "k_territorios": cfg.K_ACTUAL,
            "es_recomendada": True,
            "etiqueta": "ACTUAL (zonas reales, línea base)",
            "1": rutas_primer_ciclo(universo, labels=None, quincena=1),
            "2": rutas_primer_ciclo(universo, labels=None, quincena=2),
        },
    }

    if estrategia == "actual":
        labels_rec, k_rec = None, cfg.K_ACTUAL
    else:
        mod = ajustar(train, estrategia=estrategia, K=K,
                      universo_rebalanceo=universo if estrategia == "rebalanceo" else None)
        labels_rec = np.array([
            mod.territorio_de[int(u)] if int(u) in mod.vistos
            else mod.territorio_para_cliente(z, float(x), float(y))
            for u, z, x, y in zip(universo["_uid"], universo[cfg.COL_ASESOR],
                                  universo["x_km"], universo["y_km"])], dtype=int)
        k_rec = int(mod.K)
    cache["recomendada"] = {
        "k_territorios": k_rec,
        "es_recomendada": es_recomendada,
        "etiqueta": (f"Recomendada ({rec})" if es_recomendada else
                     "REBAL_58 (plan alternativo, NO recomendado por la regla)"),
        "1": rutas_primer_ciclo(universo, labels=labels_rec, quincena=1),
        "2": rutas_primer_ciclo(universo, labels=labels_rec, quincena=2),
    }
    cache["nota"] = ("Plan del primer ciclo: ACTUAL usa ZON_REP; la otra vista usa la "
                     "estrategia de la regla de decisión (sección 3) y, si la regla no "
                     "aprobó ninguna, el re-balanceo F2 como plan alternativo NO "
                     "recomendado. Sólo visualización: las métricas están en "
                     "evaluacion.json.")
    _escribe("rutas.json", cache)


def main() -> None:
    ap = argparse.ArgumentParser(description="Precomputa la cache de la API")
    ap.add_argument("--rapido", action="store_true", help="sólo EDA+ruteo+MC en K_ACTUAL")
    ap.add_argument("--k", nargs="*", type=int, help="K específicos del MonteCarlo")
    ap.add_argument("--n-sim", type=int, default=None, help="escenarios (default cfg.N_SIM)")
    ap.add_argument("--rehacer", action="store_true", help="ignora cache existente")
    ap.add_argument("--sin-mc", action="store_true", help="omite el MonteCarlo")
    ap.add_argument("--sin-seleccion", action="store_true", help="omite λ/K/β por K-Fold")
    ap.add_argument("--sin-comparacion", action="store_true",
                    help="omite el experimento NN vs ACO por ruta")
    ap.add_argument("--sin-evaluacion", action="store_true",
                    help="omite la evaluación de estrategias (8 dimensiones)")
    args = ap.parse_args()

    precompute_general()
    if args.sin_mc:
        return
    if args.k:
        ks = args.k
    elif args.rapido:
        ks = [cfg.K_ACTUAL, cfg.K_MIN]
    else:
        ks = list(range(cfg.K_MIN, cfg.K_MAX + 1, cfg.K_PASO))
    precompute_mc(ks, n_sim=args.n_sim, rehacer=args.rehacer)
    # geo: ACTUAL + algunos K representativos
    geo_ks = [None, cfg.K_ACTUAL] + ([cfg.K_MIN] if not args.rapido else [])
    precompute_geo(sorted(set(geo_ks), key=lambda x: (x is not None, x)))

    # --- protocolo de evaluación (secciones 1-3) ---
    if not args.sin_seleccion:
        precompute_seleccion(rehacer=args.rehacer)
    precompute_pliegues(rehacer=args.rehacer)
    if not args.sin_evaluacion:
        precompute_evaluacion(rehacer=args.rehacer)
    if not args.sin_comparacion:
        precompute_comparacion_rutas(rehacer=args.rehacer)
    # rutas del primer ciclo (usa la regla de la sección 3)
    precompute_rutas()


if __name__ == "__main__":
    main()
