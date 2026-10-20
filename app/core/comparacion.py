"""Comparación de alternativas y estabilidad multi-semilla (E1, E5).

E1 — La tabla de estabilidad original invertía el orden de "% de semillas en que
gana": la fila NN recibía `100*(ACO<NN)` (las victorias de ACO), de modo que
mostraba "NN 100 % / ACO 0 %" cuando ACO era más corto en las 10 semillas. Aquí
cada fila recibe su propia condición de victoria.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import config as cfg


def resumen_estabilidad(exp: pd.DataFrame) -> pd.DataFrame:
    """E1: media/desv/min/max de NN y ACO con % de victorias correcto por fila.

    `exp` tiene columnas `NN` y `ACO` (distancia por semilla). Una estrategia
    "gana" cuando su distancia es MENOR. Cada fila muestra su propio % de victorias.
    """
    r = exp[["NN", "ACO"]].agg(["mean", "std", "min", "max"]).T
    # E1 fix: la fila NN usa NN<ACO; la fila ACO usa ACO<NN (antes iban invertidas)
    r["% semillas en que gana"] = [
        100.0 * (exp["NN"] < exp["ACO"]).mean(),
        100.0 * (exp["ACO"] < exp["NN"]).mean(),
    ]
    r["empates"] = [int((exp["NN"] == exp["ACO"]).sum())] * 2
    return r


def victorias(exp: pd.DataFrame, estrategia: str) -> float:
    """% de semillas en que `estrategia` (NN|ACO) tiene la ruta más corta (E1)."""
    otro = "ACO" if estrategia == "NN" else "NN"
    return float(100.0 * (exp[estrategia] < exp[otro]).mean())


def test_wilcoxon(exp: pd.DataFrame) -> dict:
    """E5: prueba de Wilcoxon por pareado sobre las semillas (NN vs ACO)."""
    from scipy.stats import wilcoxon
    try:
        stat, p = wilcoxon(exp["NN"], exp["ACO"])
        return {"estadistico": float(stat), "p_value": float(p),
                "significativo_005": bool(p < 0.05)}
    except ValueError:
        return {"estadistico": np.nan, "p_value": np.nan, "significativo_005": None}


# ---------------------------------------------------------------------------
# Experimento NN vs ACO (sección "Modelo de ruteo" del dashboard)
# ---------------------------------------------------------------------------
def _recta_r2(xs: list[float], ys: list[float]) -> dict:
    """Recta de tendencia y = a + b·x por mínimos cuadrados + coeficiente R²."""
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return {"pendiente": None, "intercepto": None, "r2": None}
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - pred) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"pendiente": round(float(b), 4), "intercepto": round(float(a), 4),
            "r2": round(float(r2), 4) if r2 is not None else None}


def _tendencia(filas_sr: list[dict]) -> dict:
    """Rectas + R² de distancia vs nº de clientes (NN, ACO) y de ahorro vs tamaño."""
    if not filas_sr:
        return {}
    n = [f["n_clientes"] for f in filas_sr]
    return {
        "NN": _recta_r2(n, [f["dist_NN"] for f in filas_sr]),
        "ACO": _recta_r2(n, [f["dist_ACO"] for f in filas_sr]),
        "ahorro_pct": _recta_r2(n, [f["ahorro_pct"] for f in filas_sr]),
    }


def _ruta_ejemplo(coord_sr: dict[str, dict]) -> dict | None:
    """Secuencia NN y ACO (coordenadas) de una sub-ruta representativa (la mayor)."""
    if not coord_sr:
        return None
    nombre = max(coord_sr, key=lambda k: coord_sr[k]["n_clientes"])
    c = coord_sr[nombre]

    def _secuencia(orden: list[int]) -> list[list[float]]:
        return [[round(c["lats"][i], 5), round(c["lons"][i], 5)] for i in orden]

    return {
        "sub_ruta": nombre,
        "n_clientes": c["n_clientes"],
        "dist_NN_km": c["dist_NN"],
        "dist_ACO_km": c["dist_ACO"],
        # índice 0 = depósito; el resto en el orden del recorrido
        "nn": _secuencia(c["orden_NN"]),
        "aco": _secuencia(c["orden_ACO"]),
    }


def experimento_nn_vs_aco(df: pd.DataFrame, n_seeds: int | None = None,
                          por_rango: int = 4, m: int | None = None,
                          iteraciones: int | None = None,
                          beta: float | None = None) -> dict:
    """Compara Alternativa A (K-Means+NN) vs B (K-Means+ACO) sobre una muestra
    estratificada de sub-rutas, con `n_seeds` semillas (correcciones E2/E3/E4).

    Devuelve por-semilla NN/ACO/tiempos + resumen + Wilcoxon + muestra usada.
    Rápido por diseño (muestra chica) para servirlo por API sin precompute largo.
    """
    from .estructura import _depot_de
    from .geo import proyectar
    from .ruteo import (aco, dist_ruta, factible_orden, formar_subrutas,
                        matriz_subruta, nearest_neighbor, muestra_estratificada)

    n_seeds = cfg.N_SEEDS if n_seeds is None else n_seeds
    m = cfg.ACO_SEED_M if m is None else m
    iteraciones = cfg.ACO_SEED_ITER if iteraciones is None else iteraciones
    beta = cfg.ACO_BETA if beta is None else beta
    d = df.copy().reset_index(drop=True)
    if "_uid" not in d.columns:
        d["_uid"] = np.arange(len(d))
    proyectar(d, col_grupo=None)

    # grupos = ZON_REP (D1); depot = centroide del grupo
    grupos: list[pd.DataFrame] = []
    DEP: dict[str, dict] = {}
    for gid, g in d.groupby(cfg.COL_ASESOR):
        g = g.copy()
        g["grupo_id"] = gid
        g["sub_ruta"] = gid
        lat0, lon0 = _depot_de(g, "territorio")
        DEP[gid] = {"dep_lat": lat0, "dep_lon": lon0}
        grupos.append(g)

    # sub-rutas por capacidad (E4) y muestra estratificada (E3)
    subrutas: list[pd.DataFrame] = []
    for g in grupos:
        rutas, _, _ = formar_subrutas(g, g, cfg.CAP_TOPE, cfg.SEMILLA)
        for j, r in enumerate(rutas):
            r = r.copy()
            r["sub_ruta"] = f"{g['grupo_id'].iloc[0]}#R{j+1:02d}"
            subrutas.append(r)
    muestra = muestra_estratificada(subrutas, por_rango=por_rango)
    # (c) evita sub-rutas degeneradas (1-2 clientes) que no son un recorrido real
    muestra = [r for r in muestra if len(r) >= 3] or muestra
    if not muestra:
        muestra = subrutas[: max(1, 4 * por_rango)]

    # ---------------------------------------------------------------- por sub-ruta
    # Distancia NN y ACO por cada sub-ruta de la muestra (semilla base) para la
    # curva "distancia vs nº de clientes" y su recta de tendencia con R².
    filas_sr: list[dict] = []
    coord_sr: dict[str, dict] = {}     # sub_ruta → {lats, lons, orden_NN, orden_ACO}
    for ruta in muestra:
        gid = ruta["grupo_id"].iloc[0]
        dep = DEP[gid]
        D, slots, lats, lons = matriz_subruta(ruta, (dep["dep_lat"], dep["dep_lon"]))
        oA = nearest_neighbor(D, slots)
        oB, _, _ = aco(D, slots, m=m, alfa=cfg.ACO_ALFA, beta=beta, rho=cfg.ACO_RHO,
                       Q=cfg.ACO_Q, iteraciones=iteraciones, semilla=cfg.SEMILLA)
        dA, dB = dist_ruta(D, oA), dist_ruta(D, oB)
        nombre = ruta["sub_ruta"].iloc[0]
        filas_sr.append(dict(sub_ruta=nombre, n_clientes=int(len(ruta)),
                             dist_NN=round(dA, 3), dist_ACO=round(dB, 3),
                             ahorro_pct=round(100.0 * (dA - dB) / dA, 2) if dA else 0.0))
        coord_sr[nombre] = dict(lats=[float(x) for x in lats],
                                lons=[float(x) for x in lons],
                                orden_NN=[int(i) for i in oA],
                                orden_ACO=[int(i) for i in oB],
                                n_clientes=int(len(ruta)), dist_NN=round(dA, 3),
                                dist_ACO=round(dB, 3))
    tendencia = _tendencia(filas_sr)
    ruta_ejemplo = _ruta_ejemplo(coord_sr)

    filas = []
    for s in range(n_seeds):
        dA = dB = tA = tB = 0.0
        okA = 0
        for ruta in muestra:
            gid = ruta["grupo_id"].iloc[0]
            dep = DEP[gid]
            D, slots, _, _ = matriz_subruta(ruta, (dep["dep_lat"], dep["dep_lon"]))
            t0 = time.perf_counter()
            oA = nearest_neighbor(D, slots)
            tA += time.perf_counter() - t0
            t0 = time.perf_counter()
            oB, _, _ = aco(D, slots, m=m, alfa=cfg.ACO_ALFA, beta=beta, rho=cfg.ACO_RHO,
                           Q=cfg.ACO_Q, iteraciones=iteraciones, semilla=cfg.SEMILLA + s)
            tB += time.perf_counter() - t0
            dA += dist_ruta(D, oA)
            dB += dist_ruta(D, oB)
            okA += int(factible_orden(oA, slots))   # E2: incluye el último cliente
        filas.append(dict(semilla=s, NN=dA, ACO=dB, t_NN=tA, t_ACO=tB,
                          factible=okA // max(1, len(muestra))))

    exp = pd.DataFrame(filas)
    resumen = resumen_estabilidad(exp).reset_index().rename(columns={"index": "estrategia"})
    wilcox = test_wilcoxon(exp)
    return {
        "n_seeds": n_seeds,
        "sub_rutas_muestra": int(len(muestra)),
        "rango_muestra": [int(min(len(r) for r in muestra)), int(max(len(r) for r in muestra))],
        # unidad de `exp`/`resumen`: DISTANCIA TOTAL de la muestra en KILOMETROS
        "unidad": "km",
        "nota_nn": ("NN es determinista (voraz sin aleatoriedad): sus valores no varían "
                    "entre semillas. ACO se aleatoriza por semilla. La diferencia no es "
                    "sobreajuste sino la naturaleza de cada método."),
        "exp": exp.round(3).to_dict(orient="records"),
        "resumen": resumen.round(3).to_dict(orient="records"),
        "wilcoxon": wilcox,
        "victorias_NN": round(victorias(exp, "NN"), 1),
        "victorias_ACO": round(victorias(exp, "ACO"), 1),
        # curva distancia vs nº de clientes + R² (¿dónde conviene ACO?)
        "por_subruta": filas_sr,
        "tendencia": tendencia,
        "rutas_ejemplo": ruta_ejemplo,
    }
