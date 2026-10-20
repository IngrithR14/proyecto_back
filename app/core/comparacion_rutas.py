"""Comparación NN vs ACO por pares de ruta (sección 2 del protocolo).

Población: TODAS las rutas del plan (una por territorio y fecha, ≤ tope de
visitas), construidas con los clientes de train y completadas con los de test que
cupieron por inserción. Ambos métodos ven exactamente los MISMOS territorios,
depósitos, rutas y ventanas; lo único que cambia es el orden de las paradas.

- ACO corre con `ACO_SEEDS_RUTA` semillas por ruta; el valor ACO de la ruta es la
  MEDIANA de sus semillas (menos sensible a la casualidad que el mínimo).
- IC95 del ahorro: bootstrap de `BOOTSTRAP_N` remuestreos con semilla fija
  (`BOOTSTRAP_SEMILLA`) — reproducible.
- Ahorro por tamaño de ruta (`RANGOS_TAMANO_AHORRO`), % de rutas mejor / igual
  (tolerancia `TOL_EMPATE_ACO`) / peor, y violaciones de ventana (0 esperadas).
- Wilcoxon pareado por ruta + tamaño del efecto r = Z/√n.
- Dispersión (desviación entre semillas) reportada SOLO para ACO.
- Robustez: con los clientes de cada ruta se ejecutan los escenarios de MonteCarlo
  (previo estricto) en P_ASIST = 0,70 / 0,85 / 1,00.

La regla de decisión `regla_nn_vs_aco` está escrita ANTES de ver resultados
(`CRITERIO_NN_VS_ACO`).

Reanudabilidad: los resultados por ruta se guardan en `cache/comparacion_rutas.json`;
si el proceso se interrumpe, al reanudar sólo se ejecutan las rutas faltantes.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from .. import config as cfg


# ---------------------------------------------------------------------------
# Población: rutas del plan (train + test insertado)
# ---------------------------------------------------------------------------
def ruta_id(territorio: Any, fecha: Any) -> str:
    return f"t{territorio}_{str(fecha)[:10]}"


def poblacion(modelo, train: pd.DataFrame, test: pd.DataFrame | None = None
              ) -> tuple[list[dict], dict[str, Any]]:
    """Rutas del plan: territorios/depósitos de `ajustar(train)` + test por inserción."""
    from .ajuste import evaluar

    datos = train if test is None else pd.concat([train, test], ignore_index=True)
    ev = evaluar(modelo, datos)
    return ev["rutas"], {k: v for k, v in ev.items() if k != "rutas"}


def _serializar(rutas: list[dict]) -> dict[str, dict]:
    """Rutas → dict reanudable (sin la matriz D: se reconstruye desde lat/lon)."""
    out: dict[str, dict] = {}
    for r in rutas:
        rid = ruta_id(r["territorio"], r["fecha"])
        out[rid] = {
            "territorio": str(r["territorio"]),
            "fecha": str(r["fecha"])[:10],
            "n_clientes": int(r["n_clientes"]),
            "uids": [int(u) for u in r["uid"]],
            "lats": [float(x) for x in r["lats"]],
            "lons": [float(x) for x in r["lons"]],
            "slots": [[int(s) for s in sorted(sl)] for sl in r["slots"]],
        }
    return out


# ---------------------------------------------------------------------------
# Corrida por ruta (NN+2-opt vs ACO con semillas) — función de proceso hijo
# ---------------------------------------------------------------------------
def _correr_ruta(arg: dict) -> dict:
    """NN+2-opt vs ACO (`seeds` semillas) en UNA ruta. Corre en proceso hijo."""
    from .geo import haversine_mat
    from .ruteo import aco, dist_ruta, dos_opt, factible_orden, nearest_neighbor

    lats = np.asarray(arg["lats"], float)
    lons = np.asarray(arg["lons"], float)
    slots = [set(s) for s in arg["slots"]]
    D = haversine_mat(lats, lons)
    semilla = int(arg["semilla"])

    t0 = time.perf_counter()
    orden_nn = dos_opt(nearest_neighbor(D, slots), D, slots)
    t_nn = time.perf_counter() - t0

    corridas: list[tuple[float, list[int]]] = []
    t_aco = 0.0
    for s in range(int(arg["seeds"])):
        t1 = time.perf_counter()
        orden, d_aco, _hist = aco(D, slots, m=cfg.ACO_M, alfa=cfg.ACO_ALFA,
                                  beta=float(arg["beta"]), rho=cfg.ACO_RHO,
                                  Q=cfg.ACO_Q, iteraciones=cfg.ACO_ITER,
                                  semilla=semilla + 1000 * int(arg["i"]) + s)
        t_aco += time.perf_counter() - t1
        corridas.append((float(d_aco), list(orden)))

    kms = np.asarray([c[0] for c in corridas], float)
    mediana = float(np.median(kms))
    # orden ACO representante = el de la semilla más cercana a la mediana
    i_med = int(np.argmin(np.abs(kms - mediana)))
    orden_aco = corridas[i_med][1]

    viol = int(not factible_orden(orden_nn, slots)) + int(not factible_orden(orden_aco, slots))
    return {
        "km_nn": float(dist_ruta(D, orden_nn)),
        "orden_nn": [int(x) for x in orden_nn],
        "t_nn_s": round(t_nn, 4),
        "km_aco_seeds": [round(float(x), 4) for x in kms],
        "km_aco_mediana": mediana,
        "orden_aco": [int(x) for x in orden_aco],
        "t_aco_s": round(t_aco, 4),
        "violaciones_ventana": viol,
        "n_clientes": int(arg["n_clientes"]),
    }


def _correr_todo(cache: dict[str, Any], serial: dict[str, dict], beta: float,
                 semilla: int, n_jobs: int | None = None) -> None:
    """Ejecuta las rutas pendientes (reanudable) y las añade a `cache['rutas']`."""
    hechas: dict[str, Any] = cache.setdefault("rutas", {})
    args = [{"rid": rid, "i": i, "semilla": semilla, "seeds": cfg.ACO_SEEDS_RUTA,
             "beta": beta, **meta}
            for i, (rid, meta) in enumerate(sorted(serial.items()))]
    pendientes = [a for a in args if a["rid"] not in hechas]
    if not pendientes:
        return
    n_jobs = int(n_jobs or cfg.N_JOBS or (os.cpu_count() or 2))
    t0 = time.perf_counter()
    if n_jobs > 1 and len(pendientes) > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(pendientes))) as ex:
            res = list(ex.map(_correr_ruta, pendientes, chunksize=8))
    else:
        res = [_correr_ruta(a) for a in pendientes]
    for a, r in zip(pendientes, res):
        hechas[a["rid"]] = r
    # tiempo en pared del secuenciación del ciclo (se acumula si se corre por partes)
    meta = cache.setdefault("meta", {})
    meta["n_jobs"] = n_jobs
    meta["t_ciclo_min"] = round(float(meta.get("t_ciclo_min", 0.0))
                                + (time.perf_counter() - t0) / 60.0, 2)


# ---------------------------------------------------------------------------
# Métricas por ruta y resúmenes
# ---------------------------------------------------------------------------
def filas_por_ruta(cache: dict[str, Any]) -> pd.DataFrame:
    meta = cache.get("meta_rutas", {})
    filas = []
    for rid, m in sorted(cache["rutas"].items()):
        km_nn = float(m["km_nn"])
        filas.append({
            "ruta": rid,
            "territorio": meta.get(rid, {}).get("territorio"),
            "fecha": meta.get(rid, {}).get("fecha"),
            "n_clientes": int(m["n_clientes"]),
            "km_nn": km_nn,
            "km_aco": float(m["km_aco_mediana"]),
            "ahorro_pct": 100.0 * (km_nn - float(m["km_aco_mediana"])) / km_nn if km_nn else 0.0,
            "t_nn_s": float(m["t_nn_s"]),
            "t_aco_s": float(m["t_aco_s"]),
            "violaciones_ventana": int(m.get("violaciones_ventana", 0)),
        })
    df = pd.DataFrame(filas)
    if len(df):
        tol = cfg.TOL_EMPATE_ACO
        df["resultado"] = np.where(df["ahorro_pct"] > tol, "aco_mejor",
                                   np.where(df["ahorro_pct"] < -tol, "aco_peor", "empate"))
    return df


def _bootstrap_media(vals: np.ndarray, n: int, semilla: int) -> tuple[float, float, float]:
    """IC95 por bootstrap de la media (semilla fija → reproducible)."""
    if not len(vals):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(vals), size=(int(n), len(vals)))
    medias = vals[idx].mean(axis=1)
    inf, sup = np.percentile(medias, [2.5, 97.5])
    return float(vals.mean()), float(inf), float(sup)


def resumen_global(df: pd.DataFrame, cache: dict[str, Any]) -> dict[str, Any]:
    """Ahorro medio con IC95 (bootstrap por ruta), % mejor/igual/peor y tiempos."""
    n = len(df)
    vals = df["ahorro_pct"].to_numpy(float) if n else np.array([])
    media, inf, sup = _bootstrap_media(vals, cfg.BOOTSTRAP_N, cfg.BOOTSTRAP_SEMILLA)
    seeds = [m["km_aco_seeds"] for m in cache["rutas"].values()]
    desv = [float(np.std(s, ddof=1)) if len(s) > 1 else 0.0 for s in seeds]
    return {
        "n_rutas": int(n),
        "n_clientes": int(df["n_clientes"].sum()) if n else 0,
        "km_nn_total": round(float(df["km_nn"].sum()), 1) if n else 0.0,
        "km_aco_total": round(float(df["km_aco"].sum()), 1) if n else 0.0,
        "ahorro_medio_pct": round(media, 3),
        "ic95_inf_pct": round(inf, 3),
        "ic95_sup_pct": round(sup, 3),
        "ahorro_mediano_pct": round(float(np.median(vals)), 3) if n else 0.0,
        "pct_mejor": round(100.0 * (df["resultado"] == "aco_mejor").mean(), 2) if n else 0.0,
        "pct_empate": round(100.0 * (df["resultado"] == "empate").mean(), 2) if n else 0.0,
        "pct_peor": round(100.0 * (df["resultado"] == "aco_peor").mean(), 2) if n else 0.0,
        "violaciones_ventana": int(df["violaciones_ventana"].sum()) if n else 0,
        "t_nn_total_min": round(float(df["t_nn_s"].sum() / 60.0), 2) if n else 0.0,
        "t_aco_total_min": round(float(df["t_aco_s"].sum() / 60.0), 2) if n else 0.0,
        # tiempo en pared del secuenciación del ciclo con la paralelización declarada
        "t_ciclo_min": float(cache.get("meta", {}).get("t_ciclo_min", 0.0) or 0.0),
        "n_jobs": int(cache.get("meta", {}).get("n_jobs", 1) or 1),
        "tolerancia_empate_pct": cfg.TOL_EMPATE_ACO,
        # dispersión reportada SOLO para ACO (NN es determinista)
        "dispersion_aco": {
            "semillas_por_ruta": cfg.ACO_SEEDS_RUTA,
            "desv_media_seeds_km": round(float(np.mean(desv)), 4) if desv else 0.0,
            "desv_max_seeds_km": round(float(np.max(desv)), 4) if desv else 0.0,
            "nota": "Dispersión entre semillas reportada SOLO para ACO; NN no tiene.",
        },
        "nota": (f"El valor ACO de cada ruta es la MEDIANA de {cfg.ACO_SEEDS_RUTA} "
                 "semillas; NN es determinista. ahorro_pct = 100·(km_NN − km_ACO)/km_NN "
                 "por ruta; el IC95 es un bootstrap de remuestreo de rutas. "
                 "`t_ciclo_min` es el tiempo en pared de secuenciar el ciclo entero "
                 f"con {int(cache.get('meta', {}).get('n_jobs', 1) or 1)} procesos; "
                 "`t_aco_total_min` es la suma de tiempos por ruta (CPU acumulada)."),
    }


def resumen_por_tamano(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Ahorro por tamaño de ruta con su propio IC95 (bootstrap por ruta)."""
    out = []
    for lo, hi in cfg.RANGOS_TAMANO_AHORRO:
        sub = df[(df["n_clientes"] >= lo) & (df["n_clientes"] <= hi)] if len(df) else df
        n = len(sub)
        vals = sub["ahorro_pct"].to_numpy(float) if n else np.array([])
        media, inf, sup = _bootstrap_media(vals, cfg.BOOTSTRAP_N, cfg.BOOTSTRAP_SEMILLA)
        out.append({
            "rango": f"{lo}–{hi} clientes",
            "lo": lo, "hi": hi,
            "n_rutas": int(n),
            "ahorro_medio_pct": round(media, 3) if n else None,
            "ic95_inf_pct": round(inf, 3) if n else None,
            "ic95_sup_pct": round(sup, 3) if n else None,
            "pct_mejor": round(100.0 * (sub["resultado"] == "aco_mejor").mean(), 2) if n else None,
            "pct_empate": round(100.0 * (sub["resultado"] == "empate").mean(), 2) if n else None,
            "pct_peor": round(100.0 * (sub["resultado"] == "aco_peor").mean(), 2) if n else None,
            "km_nn_total": round(float(sub["km_nn"].sum()), 1) if n else 0.0,
            "km_aco_total": round(float(sub["km_aco"].sum()), 1) if n else 0.0,
        })
    return out


def wilcoxon_por_ruta(df: pd.DataFrame) -> dict[str, Any]:
    """Wilcoxon pareado por ruta (km_NN vs km_ACO) + tamaño del efecto r = Z/√n."""
    from scipy.stats import norm, wilcoxon

    if len(df) < 5:
        return {"n": int(len(df)), "aplicable": False,
                "nota": "Se necesitan al menos 5 rutas pareadas."}
    try:
        stat, p = wilcoxon(df["km_nn"], df["km_aco"])
    except ValueError as e:                     # todas las diferencias son 0
        return {"n": int(len(df)), "estadistico": 0.0, "p_value": 1.0,
                "efecto_r": 0.0, "aplicable": True, "nota": str(e)}
    p = float(p)
    # `norm.isf(p/2)` en vez de `norm.ppf(1 - p/2)`: con p muy pequeñas,
    # `1 - p/2` es exactamente 1.0 en coma flotante y `ppf(1) = inf` (rompería
    # el JSON del navegador y el tamaño del efecto).
    z = float(norm.isf(p / 2)) if 0 < p < 1 else 0.0
    r = (z / np.sqrt(len(df))) if np.isfinite(z) else None
    return {
        "n": int(len(df)),
        "estadistico": float(stat),
        "p_value": p,
        "significativo_005": bool(p < 0.05),
        "efecto_r": round(float(r), 4) if r is not None else None,
        "interpretacion_efecto": (None if r is None else
                                  "despreciable" if r < 0.1 else "pequeño" if r < 0.3
                                  else "mediano" if r < 0.5 else "grande"),
        "aplicable": True,
        "nota": ("Prueba pareada por ruta: H0 = la mediana de las diferencias de km es 0. "
                 "El tamaño del efecto r = Z/√n mide cuánto importa en la práctica aunque "
                 "el p-valor sea pequeño."),
    }


# ---------------------------------------------------------------------------
# Robustez bajo ausencias (MonteCarlo con previo estricto)
# ---------------------------------------------------------------------------
def _km_bajo_ausencias(D: np.ndarray, orden: list[int], A: np.ndarray) -> np.ndarray:
    """km por escenario de una ruta con previo estricto (salta ausentes)."""
    n = len(orden)
    pos = np.arange(n)
    Ao = A[:, orden]                                    # depósito primero
    idx = np.where(Ao, pos, -1)
    prev = np.maximum.accumulate(idx, axis=1)
    prev = np.concatenate([np.full((len(A), 1), -1), prev[:, :-1]], axis=1)
    Do = D[np.ix_(orden, orden)]
    contrib = Do[np.clip(prev, 0, None), pos] * Ao
    return contrib.sum(axis=1)


def robustez_montecarlo(serial: dict[str, dict], cache: dict[str, Any],
                        n_sim: int | None = None, semilla: int | None = None
                        ) -> list[dict[str, Any]]:
    """Cada ruta se ejecuta con TODOS sus clientes bajo P_ASIST 0,70 / 0,85 / 1,00.

    `serial` aporta los clientes/geometría de cada ruta y `cache` los órdenes ya
    calculados (NN y ACO). Devuelve, por nivel de asistencia, km medio por ruta (NN
    y ACO) y el ahorro de ACO con IC95 PAREADO POR ESCENARIO (cada escenario compara
    el mismo patrón de ausencias en ambas rutas).
    """
    from .geo import haversine_mat
    from .montecarlo import generar_entradas

    n_sim = cfg.N_SIM if n_sim is None else int(n_sim)
    semilla = cfg.SEMILLA if semilla is None else int(semilla)
    corridas = cache.get("rutas", {})
    rutas = {rid: m for rid, m in serial.items() if rid in corridas}
    if not rutas:
        return []
    faltantes = [rid for rid in rutas
                 if "orden_nn" not in corridas[rid] or "orden_aco" not in corridas[rid]]
    if faltantes:
        return []

    uids = sorted({u for m in rutas.values() for u in m["uids"]})
    pos = {u: i for i, u in enumerate(uids)}
    matrices = {rid: haversine_mat(np.asarray(m["lats"], float),
                                   np.asarray(m["lons"], float))
                for rid, m in rutas.items()}

    salidas = []
    for p in cfg.P_ASIST_EVAL:
        asistencia, _trf = generar_entradas(n_sim, len(uids), semilla,
                                            n_reserva=0, p_asist=p)
        km_nn_tot = np.zeros(n_sim)
        km_aco_tot = np.zeros(n_sim)
        km_nn_ruta, km_aco_ruta = [], []
        for rid, m in rutas.items():
            idxs = [pos[u] for u in m["uids"]]
            base = np.concatenate(
                [np.ones((len(asistencia), 1), bool), asistencia[:, idxs]], axis=1)
            km_nn = _km_bajo_ausencias(matrices[rid], corridas[rid]["orden_nn"], base)
            km_aco = _km_bajo_ausencias(matrices[rid], corridas[rid]["orden_aco"], base)
            km_nn_tot += km_nn
            km_aco_tot += km_aco
            km_nn_ruta.append(km_nn.mean())
            km_aco_ruta.append(km_aco.mean())
        ahorro_esc = 100.0 * (km_nn_tot - km_aco_tot) / km_nn_tot
        media, inf, sup = _bootstrap_media(ahorro_esc, cfg.BOOTSTRAP_N,
                                           cfg.BOOTSTRAP_SEMILLA)
        salidas.append({
            "p_asist": float(p),
            "n_sim": n_sim,
            "n_rutas": len(rutas),
            "km_medio_por_ruta_nn": round(float(np.mean(km_nn_ruta)), 3),
            "km_medio_por_ruta_aco": round(float(np.mean(km_aco_ruta)), 3),
            "ahorro_medio_pct": round(media, 3),
            "ic95_inf_pct": round(inf, 3),
            "ic95_sup_pct": round(sup, 3),
            "ahorro_medio_escenario_pct": round(float(ahorro_esc.mean()), 3),
            "nota": ("IC95 pareado por escenario: en cada escenario se compara el mismo "
                     "patrón de ausencias en las rutas NN y ACO."),
        })
    return salidas


# ---------------------------------------------------------------------------
# Regla de decisión (escrita ANTES de ver resultados)
# ---------------------------------------------------------------------------
CRITERIO_NN_VS_ACO: str = (
    "Se adopta ACO como método de secuenciación SÓLO si se cumplen las cinco "
    "condiciones siguientes: (1) el ahorro medio por ruta es ≥ "
    "UMBRALES_DECISION['ahorro_aco_pct'] (2 %); (2) el límite inferior del IC95 del "
    "ahorro es > 0; (3) el ahorro ≥ 2 % persiste con P_ASIST = 0,85 (robustez bajo "
    "ausencias); (4) ACO tiene 0 violaciones de ventana; (5) el cómputo de todas las "
    "rutas con ACO tarda ≤ UMBRALES_DECISION['tiempo_ciclo_min'] (30 min) en tiempo "
    "en pared con la paralelización declarada (`N_JOBS` procesos). Si ACO sólo "
    "gana en un rango de tamaño de ruta (ahorro ≥ 2 % e IC95 inferior > 0 en ese rango, "
    "pero no en el global) se adopta ACO en ese rango y NN en el resto. En ningún caso "
    "se ajusta un umbral para cambiar el veredicto.")


def regla_nn_vs_aco(global_: dict, por_tamano: list[dict],
                    robustez: list[dict]) -> dict[str, Any]:
    """Aplica CRITERIO_NN_VS_ACO. Escrita antes de ver cualquier resultado."""
    umb = cfg.UMBRALES_DECISION
    robus_085 = next((r for r in robustez if abs(r["p_asist"] - 0.85) < 1e-9), None)
    c1 = bool(global_["ahorro_medio_pct"] >= umb["ahorro_aco_pct"])
    c2 = bool(global_["ic95_inf_pct"] > 0)
    c3 = bool(robus_085 is not None
              and robus_085["ahorro_medio_pct"] >= umb["ahorro_aco_pct"]
              and robus_085["ic95_inf_pct"] > 0)
    c4 = bool(global_["violaciones_ventana"] == 0)
    c5 = bool(global_.get("t_ciclo_min", global_["t_aco_total_min"])
              <= umb["tiempo_ciclo_min"])
    cumple = {"ahorro_minimo": c1, "ic95_positivo": c2, "persiste_con_085": c3,
              "sin_violaciones_ventana": c4, "tiempo_ciclo": c5}

    rangos = [r["rango"] for r in por_tamano
              if r["n_rutas"] > 0 and (r["ahorro_medio_pct"] or 0) >= umb["ahorro_aco_pct"]
              and (r["ic95_inf_pct"] or 0) > 0]

    if all(cumple.values()):
        decision = "aco"
        veredicto = ("ACO pasa las cinco condiciones: se adopta ACO en todas las rutas "
                     "del ciclo.")
    elif rangos and not (c1 and c2):
        decision = "mixta"
        veredicto = (f"ACO sólo gana en {', '.join(rangos)} (ahorro ≥ "
                     f"{umb['ahorro_aco_pct']} % e IC95 inferior > 0): se usa ACO en ese "
                     "rango de tamaño y NN en el resto.")
    else:
        decision = "nn"
        faltan = ", ".join(k for k, v in cumple.items() if not v)
        veredicto = (f"NN sigue siendo el método por defecto: no se cumplen todas las "
                     f"condiciones ({faltan}).")
    return {
        "decision": decision,
        "cumple": cumple,
        "rangos_con_aco": rangos,
        "veredicto": veredicto,
        "criterio": CRITERIO_NN_VS_ACO,
        "umbrales": {k: umb[k] for k in ("ahorro_aco_pct", "tiempo_ciclo_min")},
    }


# ---------------------------------------------------------------------------
# Experimento completo (reanudable)
# ---------------------------------------------------------------------------
def experimento_nn_vs_aco(modelo, train: pd.DataFrame, test: pd.DataFrame | None,
                          cache: dict[str, Any] | None = None,
                          beta: float | None = None, semilla: int | None = None,
                          n_jobs: int | None = None,
                          con_robustez: bool = True) -> dict[str, Any]:
    """Población + corrida por ruta (reanudable) + métricas + regla."""
    beta = cfg.ACO_BETA if beta is None else float(beta)
    semilla = cfg.SEMILLA_MODELO if semilla is None else int(semilla)

    rutas, _ev = poblacion(modelo, train, test)
    serial = _serializar(rutas)

    cache = {"rutas": {}} if cache is None else cache
    cache.setdefault("rutas", {})
    cache.setdefault("meta_rutas", {})
    for rid, meta in serial.items():
        cache["meta_rutas"][rid] = {"territorio": meta["territorio"],
                                    "fecha": meta["fecha"],
                                    "n_clientes": meta["n_clientes"]}
    cache.setdefault("meta", {}).update(
        {"beta": beta, "semilla": semilla, "seeds_por_ruta": cfg.ACO_SEEDS_RUTA,
         "n_rutas": len(serial), "completadas": len(cache["rutas"]), "reanudable": True})

    _correr_todo(cache, serial, beta, semilla, n_jobs=n_jobs)
    cache["meta"]["completadas"] = len(cache["rutas"])

    df = filas_por_ruta(cache)
    glob = resumen_global(df, cache)
    por_tam = resumen_por_tamano(df)
    wilcox = wilcoxon_por_ruta(df)
    robus = robustez_montecarlo(serial, cache) if con_robustez else []
    regla = regla_nn_vs_aco(glob, por_tam, robus)

    return {
        "meta": cache["meta"],
        "global": glob,
        "por_ruta": df.round(4).to_dict(orient="records"),
        "por_tamano": por_tam,
        "wilcoxon": wilcox,
        "robustez": robus,
        "regla": regla,
        "criterio_regla": CRITERIO_NN_VS_ACO,
        "cache": cache,
    }
