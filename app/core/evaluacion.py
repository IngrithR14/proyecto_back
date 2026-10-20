"""Comparación de estrategias contra ACTUAL (sección 3 del protocolo).

Estrategias evaluadas, todas con el mismo protocolo (`ajustar(train)` + test que
entra SOLO por inserción; factibilidad y costos sobre TODAS las visitas de la
población evaluada):

  ACTUAL       58 zonas reales — línea base; NO entra a la regla.
  REBAL_58     F2: re-balanceo mínimo de ACTUAL — única candidata recomendable.
  K43..K58     reagrupación global con restricción de capacidad — DIAGNÓSTICA,
               jamás recomendable (ver diagnóstico: reagrupar dispara los km).

OCHO dimensiones (umbrales en `cfg.UMBRALES_DECISION`, etiquetados "propuesta, no
calibrada"): costo, eficiencia, factibilidad, cumplimiento, balance, generalización,
estabilidad y compacidad. Seis de ellas son condiciones de la regla (`condiciones`);
balance y compacidad se reportan como contexto.

La ELECCIÓN se hace con `SEMILLA_MODELO` y los números FINALES del informe con
`SEMILLA_REPORTE` (distintas); la estabilidad del ranking de K se mide con
`SEMILLAS_ESTABILIDAD` (3 semillas).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import config as cfg

DIMENSIONES: tuple[str, ...] = ("costo", "eficiencia", "factibilidad", "cumplimiento",
                                "balance", "generalizacion", "estabilidad", "compacidad")

CONDICIONES: tuple[str, ...] = ("factibilidad", "costo", "cumplimiento", "eficiencia",
                                "generalizacion", "estabilidad")

CRITERIO_ESTRATEGIAS: str = (
    "Una estrategia CANDIDATA se recomienda SÓLO si cumple las seis condiciones: "
    "(1) factibilidad: 0 días sobre el tope (cap dinámico); (2) costo: el límite "
    "superior del IC95 de su Δcosto frente a ACTUAL es < 0 (pareado por los mismos "
    "escenarios); (3) cumplimiento: no pierde más de "
    "`cobertura_pp` puntos de cobertura frente a ACTUAL; (4) eficiencia: sus km del "
    "ciclo no superan `ratio_km_vs_actual` × los km de ACTUAL; (5) generalización: el km "
    "extra al insertar los clientes de test no supera `km_extra_test_pct` % y el "
    "arrepentimiento no supera `arrepentimiento_pct` %; (6) estabilidad: el mismo K (o "
    "±1) se elige en al menos `estabilidad_semillas` de las 3 semillas. ACTUAL es la "
    "línea base y no entra a la regla; la reagrupación global (K) es diagnóstica y "
    "nunca puede ser recomendada. Si ninguna candidata cumple las seis condiciones → "
    "\"ninguna estrategia cumple las condiciones\". En ningún caso se ajusta un umbral "
    "para cambiar el veredicto.")


# ---------------------------------------------------------------------------
# Nombres y etiquetas
# ---------------------------------------------------------------------------
def nombre_de(modelo) -> str:
    if modelo.estrategia == "actual":
        return "ACTUAL"
    if modelo.estrategia == "rebalanceo":
        return f"REBAL_{modelo.K}"
    return f"K{modelo.K}"


def etiqueta_de(nombre: str) -> str:
    if nombre == "ACTUAL":
        return "ACTUAL (línea base: 58 zonas reales)"
    if nombre.startswith("REBAL_"):
        return f"{nombre} (re-balanceo mínimo de ACTUAL)"
    return f"{nombre} (reagrupación global, diagnóstica)"


def es_diagnostica(nombre: str) -> bool:
    """Sólo la re-agrupación global es diagnóstica; ACTUAL es base y REBAL es candidata."""
    return nombre.startswith("K")


# ---------------------------------------------------------------------------
# Arrepentimiento del plan final
# ---------------------------------------------------------------------------
def arrepentimiento(modelo, datos: pd.DataFrame, km_plan: float) -> dict[str, Any]:
    """¿Cuánto km cuesta haber planificado sin ver a los clientes de test?

    Se reconstruyen las rutas con los MISMOS territorios y depósitos del modelo y
    TODOS los clientes (sin re-agrupar ni re-balancear).
    """
    from .ajuste import km_de_rutas, rutas_diarias
    from .geo import proyectar

    d = datos.copy().reset_index(drop=True)
    if "x_km" not in d.columns:
        proyectar(d, col_grupo=None)
    labels = np.array([
        modelo.territorio_de[int(u)] if int(u) in modelo.vistos
        else modelo.territorio_para_cliente(z, float(x), float(y))
        for u, z, x, y in zip(d["_uid"], d[cfg.COL_ASESOR], d["x_km"], d["y_km"])],
        dtype=int)
    km_refit = km_de_rutas(rutas_diarias(d, labels))
    pct = round(100.0 * (km_plan - km_refit) / km_refit, 3) if km_refit else None
    return {"km_replan": round(km_refit, 1), "arrepentimiento_pct": pct,
            "nota": "km(plan) − km(rutas reconstruidas con los mismos territorios)"}


# ---------------------------------------------------------------------------
# Ocho dimensiones
# ---------------------------------------------------------------------------
def _mc(ev: dict) -> dict:
    return ev.get("montecarlo") or {}


def _ic95_pareado(a, b, escala: float = 1.0) -> list[float] | None:
    """IC95 (media ± 1.96·EE) del Δ = a − b emparejado por el MISMO escenario.

    Todos los escenarios comparten la misma matriz de asistencia y tráfico, así
    que el Δ se resta por índice de escenario (comparación pareada, como pide el
    protocolo). Devuelve `None` si no hay escenarios emparejados.
    """
    if a is None or b is None:
        return None
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size < 2:
        return None
    d = (a - b) * escala
    ee = d.std(ddof=1) / np.sqrt(len(d))
    return [round(float(d.mean() - 1.96 * ee), 4), round(float(d.mean() + 1.96 * ee), 4)]


def dimensiones(ev_base: dict, ev: dict, arrep: dict, arrep_base: dict,
                estabilidad_ok: bool | None, estabilidad_nota: str) -> dict[str, Any]:
    """Las 8 dimensiones de `ev` frente a la línea base `ev_base` (ACTUAL)."""
    mc, mcb = _mc(ev), _mc(ev_base)
    umb = cfg.UMBRALES_DECISION
    fact = ev["factibilidad"]
    ins = ev["insercion"]

    costo = mc.get("costo_medio_M")
    costo_b = mcb.get("costo_medio_M")
    delta_costo = (100.0 * (costo - costo_b) / costo_b
                   if (costo is not None and costo_b) else None)
    km = mc.get("km_medio", ev["km_total"])
    km_b = mcb.get("km_medio", ev_base["km_total"])
    cob = mc.get("cobertura")
    cob_b = mcb.get("cobertura")
    cv = fact.get("cv_carga_asesor")
    cv_b = ev_base["factibilidad"].get("cv_carga_asesor")
    km_ins = ins.get("km_plan_vs_antes_pct")
    arb = arrep.get("arrepentimiento_pct")

    # Δ pareado por escenario: costo (M COP) y km, con su IC95 (sección 3: la
    # condición de costo es el IC95 SUPERIOR del Δcosto < 0, no la media).
    ic95_delta_costo = _ic95_pareado(ev.get("costo_total"),
                                    ev_base.get("costo_total"), escala=1e-6)
    ic95_delta_km = _ic95_pareado(ev.get("km_escenario"),
                                 ev_base.get("km_escenario"))
    delta_costo_M = (round((ic95_delta_costo[0] + ic95_delta_costo[1]) / 2, 3)
                     if ic95_delta_costo else None)

    return {
        "costo": {
            "costo_medio_M": costo, "delta_vs_actual_pct": delta_costo,
            "ic95_M": [mc.get("ic95_inf"), mc.get("ic95_sup")],
            "delta_vs_actual_M": delta_costo_M,
            "ic95_delta_M": ic95_delta_costo,
            "umbral": "IC95 superior del Δcosto < 0 vs ACTUAL",
            "cumple": bool(ic95_delta_costo is not None and ic95_delta_costo[1] < 0),
            "nota": ("Costo total del ciclo con simulador F3 (fijo + combustible + "
                     "horas extra); Δ pareado por los mismos escenarios aleatorios."),
        },
        "eficiencia": {
            "km_medio": km, "ratio_vs_actual": round(km / km_b, 4) if km_b else None,
            "ic95_delta_km": ic95_delta_km,
            "umbral": f"≤ {umb['ratio_km_vs_actual']} × ACTUAL",
            "cumple": bool(km_b and km / km_b <= umb["ratio_km_vs_actual"]),
        },
        "factibilidad": {
            "dias_sobre_cap": int(fact.get("dias_sobre_cap", 0)),
            "pico_carga": int(fact.get("pico_global", 0)),
            "cap_dinamico": int(fact.get("cap_dinamico", cfg.cap_dinamico())),
            "umbral": "0 días sobre el tope",
            "cumple": bool(fact.get("dias_sobre_cap", 0) == 0),
        },
        "cumplimiento": {
            "cobertura": cob, "cobertura_base": cob_b,
            "delta_pp_vs_actual": round(100.0 * (cob - cob_b), 3)
            if (cob is not None and cob_b is not None) else None,
            "visitas_perdidas": mc.get("visitas_perdidas"),
            "umbral": f"pérdida ≤ {umb['cobertura_pp']} puntos vs ACTUAL",
            "cumple": bool(cob is not None and cob_b is not None
                           and 100.0 * (cob - cob_b) >= -umb["cobertura_pp"]),
        },
        "balance": {
            "cv_carga": cv, "cv_base": cv_b,
            "delta_vs_actual": round(cv - cv_b, 4) if (cv is not None and cv_b is not None) else None,
            "umbral": "no empeora el CV de carga vs ACTUAL",
            "cumple": bool(cv is not None and cv_b is not None and cv <= cv_b),
            "nota": "Dimensión informativa (no es condición de la regla).",
        },
        "generalizacion": {
            "km_extra_test_pct": km_ins, "arrepentimiento_pct": arb,
            "km_extra_cliente_p90": ins.get("km_extra_cliente_p90"),
            "pct_no_insertados": round(100.0 - ins.get("pct_insertados_completos", 100.0), 2)
            if ins.get("n_test") else 0.0,
            "umbral": (f"km extra ≤ {umb['km_extra_test_pct']} % y arrepentimiento "
                       f"< {umb['arrepentimiento_pct']} %"),
            "cumple": bool(km_ins is not None and km_ins <= umb["km_extra_test_pct"]
                           and arb is not None and arb < umb["arrepentimiento_pct"]),
        },
        "estabilidad": {
            "cumple": bool(estabilidad_ok) if estabilidad_ok is not None else True,
            "umbral": (f"mismo K (o ±1) en ≥ {int(umb['estabilidad_semillas'])} de 3 "
                       "semillas"),
            "nota": estabilidad_nota,
        },
        "compacidad": {
            "visitas_por_asesor_dia": round(
                ev["visitas_programadas"] / (ev["K"] * cfg.DIAS_HABILES_CICLO), 3)
            if ev.get("visitas_programadas") and ev.get("K") else None,
            "n_asesores": int(ev["K"]),
            "umbral": "informativa",
            "nota": "Visitas programadas por asesor-día (más alto = territorios más densos).",
        },
    }


# ---------------------------------------------------------------------------
# Estabilidad del ranking de K (3 semillas distintas)
# ---------------------------------------------------------------------------
def estabilidad_k(train: pd.DataFrame, K_grid: tuple[int, ...] | None = None,
                  lam_norm: float | None = None) -> dict[str, Any]:
    """¿Se elige la misma K (o ±1) con 3 semillas distintas? (Spearman entre rankings)."""
    from scipy.stats import spearmanr

    from .ajuste import ajustar, evaluar

    K_grid = tuple(K_grid or cfg.K_CANDIDATOS)
    lam_norm = cfg.LAM_GRID[1] if lam_norm is None else float(lam_norm)
    filas: list[dict[str, Any]] = []
    rankings: list[list[int]] = []
    elegidas: list[int] = []
    for s in cfg.SEMILLAS_ESTABILIDAD:
        costos: dict[int, dict] = {}
        for K in K_grid:
            mod = ajustar(train, estrategia="k", K=int(K), lam_norm=lam_norm, semilla=s)
            ev = evaluar(mod, train)
            km = ev["km_total"]
            costos[int(K)] = {
                "costo_det_M": round((cfg.costo_fijo_ciclo() * mod.K
                                      + km * cfg.COSTO_KM) / 1e6, 2),
                "km_ciclo": round(km, 1),
                "dias_sobre_cap": int(ev["factibilidad"].get("dias_sobre_cap", 0)),
                "factible": bool(ev["factibilidad"].get("factible", False)),
            }
        orden = sorted(costos, key=lambda k: (not costos[k]["factible"],
                                              costos[k]["costo_det_M"]))
        rankings.append(orden)
        elegidas.append(orden[0])
        filas.append({"semilla": s, "elegida": orden[0],
                      "costos": {str(k): v for k, v in costos.items()}})

    # Spearman entre los rankings (pares de semillas)
    pares = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            a = pd.Series({k: rankings[i].index(k) for k in K_grid})
            b = pd.Series({k: rankings[j].index(k) for k in K_grid})
            rho, p = spearmanr(a, b)
            pares.append({"semillas": [cfg.SEMILLAS_ESTABILIDAD[i],
                                       cfg.SEMILLAS_ESTABILIDAD[j]],
                          "rho": round(float(rho), 4), "p_value": round(float(p), 4)})
    referencias = pd.Series({k: rankings[0].index(k) for k in K_grid})
    acuerdo = sum(1 for e in elegidas if abs(e - elegidas[0]) <= 1)
    umbral = int(cfg.UMBRALES_DECISION["estabilidad_semillas"])
    return {
        "semillas": list(cfg.SEMILLAS_ESTABILIDAD),
        "K_grid": list(K_grid),
        "lam_norm": lam_norm,
        "K_elegidas": elegidas,
        "correlaciones": pares,
        "rho_minimo": round(min((p["rho"] for p in pares), default=1.0), 4),
        "acuerdo_misma_K_o_mas_menos_1": acuerdo,
        "umbral": umbral,
        "cumple": bool(acuerdo >= umbral),
        "filas": filas,
        "referencia_semilla": int(cfg.SEMILLAS_ESTABILIDAD[0]),
        "nota": ("Ranking por costo determinista (fijo + combustible) con la estrategia "
                 "'k' y la λ elegida por K-Fold; se evalúa sobre train (diagnóstico de "
                 "estabilidad, no evaluación de test)."),
        "referencia_ranking": [int(k) for k in rankings[0]] if rankings else referencias,
    }


# ---------------------------------------------------------------------------
# Regla de decisión (escrita ANTES de ver resultados)
# ---------------------------------------------------------------------------
def regla_estrategias(estrategias: dict[str, Any]) -> dict[str, Any]:
    """Aplica CRITERIO_ESTRATEGIAS. Sólo REBAL_* puede ser recomendada."""
    candidatas = {n: e for n, e in estrategias.items()
                  if n != "ACTUAL" and not e["es_diagnostica"]}
    cumplen = [n for n, e in candidatas.items() if all(e["condiciones"].values())]
    if cumplen:
        # menor costo entre las que cumplen todo
        elegida = min(cumplen,
                      key=lambda n: estrategias[n]["dimensiones"]["costo"]["costo_medio_M"])
        decision = "recomendada"
        veredicto = (f"{elegida} cumple las seis condiciones: es la estrategia "
                     "recomendada para el próximo ciclo.")
    else:
        elegida = None
        decision = "ninguna"
        veredicto = ("ninguna estrategia cumple las condiciones; se mantiene el método "
                     "actual y se reporta qué condición le falta a cada candidata.")
    detalle = {n: {c: bool(e["condiciones"][c]) for c in CONDICIONES}
               for n, e in estrategias.items()}
    return {
        "decision": decision,
        "recomendada": elegida,
        "veredicto": veredicto,
        "criterio": CRITERIO_ESTRATEGIAS,
        "pool_candidatas": sorted(candidatas),
        "condiciones_por_estrategia": detalle,
        "diagnosticas": sorted(n for n, e in estrategias.items() if e["es_diagnostica"]),
        "nota": ("ACTUAL es la línea base (no entra a la regla). La reagrupación global "
                 "(K) es diagnóstica: se reporta pero jamás se recomienda."),
    }


# ---------------------------------------------------------------------------
# Reporte F2 del re-balanceo
# ---------------------------------------------------------------------------
def reporte_f2(mod_rebal, ev_rebal: dict, ev_actual: dict) -> dict[str, Any]:
    r = dict(mod_rebal.reporte or {})
    return {
        "poblacion_del_rebalanceo": cfg.REBAL_POBLACION,
        "clientes_totales": r.get("n_clientes"),
        "clientes_movidos": r.get("clientes_movidos"),
        "clientes_movidos_pct": r.get("clientes_movidos_pct"),
        "dias_sobre_tope_antes": r.get("dias_sobre_tope_antes"),
        "dias_sobre_tope_despues": r.get("dias_sobre_tope_despues"),
        "pico_antes": r.get("pico_antes"),
        "pico_despues": r.get("pico_despues"),
        "factible_antes": r.get("factible_antes"),
        "factible_despues": r.get("factible_despues"),
        "logro_factibilidad": r.get("logro_factibilidad"),
        "km_antes": r.get("km_antes"),
        "km_despues": r.get("km_despues"),
        "delta_km": r.get("delta_km"),
        "delta_km_pct": r.get("delta_km_pct"),
        "movimientos": (r.get("movimientos") or [])[:20],
        "plan_evaluado": {
            "km_actual": ev_actual["km_total"],
            "km_rebal": ev_rebal["km_total"],
            "delta_km_plan_pct": round(
                100.0 * (ev_rebal["km_total"] - ev_actual["km_total"])
                / ev_actual["km_total"], 3) if ev_actual["km_total"] else None,
            "cobertura_actual": _mc(ev_actual).get("cobertura"),
            "cobertura_rebal": _mc(ev_rebal).get("cobertura"),
        },
        "verificaciones": {k: r.get(k) for k in ("cada_cliente_una_vez",
                                                 "sin_movimientos_a_si_mismo")},
        "nota": ("El re-balanceo mueve el mínimo de clientes para eliminar los días "
                 "sobre el tope, eligiendo en cada día la fecha sobrecargada del "
                 "cliente cuyo destino menos aumenta la distancia a su centroide. "
                 "Los umbrales y el tope no se tocan: si no se puede, se reporta como "
                 "infactible."),
    }


# ---------------------------------------------------------------------------
# Evaluación de todas las estrategias
# ---------------------------------------------------------------------------
def _resumen_eval(ev: dict) -> dict[str, Any]:
    """`ev` sin las rutas (para la caché y la API)."""
    return {k: v for k, v in ev.items() if k != "rutas"}


def evaluacion(train: pd.DataFrame, test: pd.DataFrame, universo: pd.DataFrame,
               lam_norm: float | None = None, beta: float | None = None,
               semilla: int | None = None, K_grid: tuple[int, ...] | None = None,
               estab: dict | None = None, con_mc: bool = True) -> dict[str, Any]:
    """Estrategias contra ACTUAL: 8 dimensiones, condiciones y regla."""
    from .ajuste import ajustar, evaluar
    from .montecarlo import generar_entradas

    semilla = cfg.SEMILLA_MODELO if semilla is None else int(semilla)
    lam_norm = cfg.LAM_GRID[1] if lam_norm is None else float(lam_norm)
    K_grid = tuple(K_grid or cfg.K_CANDIDATOS)

    asistencia = trafico = None
    if con_mc:
        # las columnas se indexan por `_uid` GLOBAL (puede superar len(universo) si se
        # muestrea), así que se dimensionan por el uid máximo
        n_cols = max(len(universo), int(universo["_uid"].max()) + 1)
        asistencia, trafico = generar_entradas(cfg.N_SIM, n_cols, semilla)

    modelos: dict[str, Any] = {}
    evals: dict[str, dict] = {}
    orden = [("ACTUAL", "actual", None), ("REBAL_58", "rebalanceo", None)]
    orden += [(f"K{K}", "k", int(K)) for K in K_grid]

    for nombre, est, K in orden:
        # el reporte F2 necesita los km antes/después: se piden con `con_km`
        mod = ajustar(train, estrategia=est, K=K, lam_norm=lam_norm, beta=beta,
                      semilla=semilla, con_km=(est == "rebalanceo"),
                      universo_rebalanceo=universo if est == "rebalanceo" else None)
        ev = evaluar(mod, universo, asistencia=asistencia, trafico=trafico)
        modelos[nombre] = mod
        evals[nombre] = ev

    ev_base = evals["ACTUAL"]
    arrep_base = arrepentimiento(modelos["ACTUAL"], universo, ev_base["km_total"])

    estrategias: dict[str, Any] = {}
    for nombre, _est, _K in orden:
        ev = evals[nombre]
        arb = arrepentimiento(modelos[nombre], universo, ev["km_total"])
        est_ok, est_nota = True, "Estrategia determinista (no depende de semilla)."
        if es_diagnostica(nombre):
            if estab is not None:
                est_ok = bool(estab["cumple"])
                est_nota = (f"Ranking de K entre 3 semillas: acuerdo "
                            f"{estab['acuerdo_misma_K_o_mas_menos_1']}/3, "
                            f"ρ mínimo {estab['rho_minimo']}.")
            else:
                est_ok, est_nota = None, "Estabilidad no calculada."
        dims = dimensiones(ev_base, ev, arb, arrep_base, est_ok, est_nota)
        condiciones = {c: bool(dims[c]["cumple"]) for c in CONDICIONES}
        estrategias[nombre] = {
            "etiqueta": etiqueta_de(nombre),
            "recomendable": not es_diagnostica(nombre) and nombre != "ACTUAL",
            "es_diagnostica": es_diagnostica(nombre),
            "K": int(ev["K"]),
            "estrategia": ev["estrategia"],
            "lam_norm": lam_norm, "beta": beta,
            "resumen": _resumen_eval(ev),
            "dimensiones": dims,
            "condiciones": condiciones,
            "cumple_todas": bool(all(condiciones.values())),
            "arrepentimiento": arb,
        }

    regla = regla_estrategias(estrategias)

    tabla = []
    for nombre, e in estrategias.items():
        d = e["dimensiones"]
        tabla.append({
            "estrategia": nombre, "etiqueta": e["etiqueta"], "K": e["K"],
            "es_diagnostica": e["es_diagnostica"],
            "costo_M": d["costo"]["costo_medio_M"],
            "delta_costo_pct": d["costo"]["delta_vs_actual_pct"],
            "delta_costo_M": d["costo"].get("delta_vs_actual_M"),
            "ic95_delta_costo_M": d["costo"].get("ic95_delta_M"),
            "ic95_delta_km": d["eficiencia"].get("ic95_delta_km"),
            "km": d["eficiencia"]["km_medio"],
            "ratio_km": d["eficiencia"]["ratio_vs_actual"],
            "dias_sobre_cap": d["factibilidad"]["dias_sobre_cap"],
            "cobertura": d["cumplimiento"]["cobertura"],
            "delta_cobertura_pp": d["cumplimiento"]["delta_pp_vs_actual"],
            "cv_carga": d["balance"]["cv_carga"],
            "km_extra_test_pct": d["generalizacion"]["km_extra_test_pct"],
            "arrepentimiento_pct": d["generalizacion"]["arrepentimiento_pct"],
            "compacidad": d["compacidad"]["visitas_por_asesor_dia"],
            "cumple_todas": e["cumple_todas"],
            "recomendable": e["recomendable"],
        })

    f2 = reporte_f2(modelos["REBAL_58"], evals["REBAL_58"], ev_base)

    return {
        "meta": {
            "semilla_eleccion": cfg.SEMILLA_MODELO, "semilla_reporte": semilla,
            "lam_norm": lam_norm, "beta": beta, "n_sim": cfg.N_SIM if con_mc else 0,
            "politica_sobrecarga": cfg.POLITICA_SOBRECARGA,
            "cap_dinamico": cfg.cap_dinamico(), "cap_tope": cfg.CAP_TOPE,
            "n_train": int(len(train)), "n_test": int(len(test)),
            "n_universo": int(len(universo)),
            "umbrales": dict(cfg.UMBRALES_DECISION),
            "nota_umbrales": "UMBRALES_DECISION: propuesta, no calibrada.",
        },
        "estrategias": estrategias,
        "tabla": tabla,
        "regla": regla,
        "f2_rebalanceo": f2,
        "estabilidad": estab,
        "criterio": CRITERIO_ESTRATEGIAS,
    }


# ---------------------------------------------------------------------------
# Orquestador (precompute / informe)
# ---------------------------------------------------------------------------
def evaluacion_completa(train: pd.DataFrame, test: pd.DataFrame,
                        universo: pd.DataFrame,
                        lam_norm: float | None = None, beta: float | None = None,
                        semilla: int | None = None) -> dict[str, Any]:
    """Elección (semilla del modelo) + números finales (semilla de reporte) + estabilidad.

    La elección de la estrategia recomendable se hace con `SEMILLA_MODELO`; los
    números que van al informe salen con `SEMILLA_REPORTE` (distinta), y la
    estabilidad del ranking de K con `SEMILLAS_ESTABILIDAD`.
    """
    eleccion = evaluacion(train, test, universo, lam_norm=lam_norm, beta=beta,
                          semilla=cfg.SEMILLA_MODELO)
    estab = estabilidad_k(train, lam_norm=lam_norm)
    informe = evaluacion(train, test, universo, lam_norm=lam_norm, beta=beta,
                         semilla=cfg.SEMILLA_REPORTE, estab=estab)
    return {
        "meta": informe["meta"],
        "eleccion": eleccion,
        "informe": informe,
        "nota": ("La estrategia se elige con SEMILLA_MODELO y los números del informe "
                 "salen con SEMILLA_REPORTE (distintas a propósito)."),
    }
