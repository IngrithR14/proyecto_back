"""Validación cruzada (K-Fold) y selección de hiperparámetros (sección 1).

Todo ocurre SOLO sobre el 80 % de train: el test queda sellado para la evaluación
final. En cada pliegue se ajusta en k−1 y se evalúa en el retenido, que entra
únicamente por inserción más barata (respetando el tope, la ventana horaria y las
fechas del cliente), lo que da las métricas de generalización:

  - km extra MARGINAL por cliente de test (p50 y p90),
  - % de clientes de test que exigen una ruta nueva o exceden el tope,
  - arrepentimiento = km(plan con inserción) − km(re-plan de las MISMAS rutas con
    todos los clientes del pliegue), en % sobre el re-plan. Mide lo que costó no
    haber visto a los clientes retenidos al construir las rutas.

Esto es validación de un problema de OPTIMIZACIÓN, no de clasificación: no hay
Accuracy/Precision/Recall porque no hay etiquetas que predecir. `costo_det_M` es
el costo fijo + combustible (determinista); las horas extra requieren el
simulador F3 y se reportan aparte en la sección de estrategias.

Selección (rejilla declarada de antemano en `cfg`): λ ∈ `LAM_GRID`, K ∈
`K_CANDIDATOS`, β ∈ `BETA_GRID`. El criterio de elección está escrito ANTES de
ver los resultados (ver `seleccionar_config`).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import config as cfg

METRICAS_KFOLD: tuple[str, ...] = ("km_ciclo", "costo_det_M", "pico_carga",
                                   "dias_sobre_cap", "cv_carga")
METRICAS_TEST: tuple[str, ...] = ("km_extra_p50", "km_extra_p90", "pct_ruta_nueva",
                                  "pct_excede_cap", "pct_no_insertados",
                                  "arrepentimiento_pct")


# ---------------------------------------------------------------------------
# Pliegues
# ---------------------------------------------------------------------------
def kfold_indices(n: int, k_pliegues: int | None = None,
                  semilla: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pliegues estratificados por la nada (sobre train): (índices ajuste, retenido)."""
    k_pliegues = cfg.K_PLIEGUES if k_pliegues is None else int(k_pliegues)
    semilla = cfg.SEMILLA_MODELO if semilla is None else int(semilla)
    rng = np.random.default_rng(semilla)
    folds = np.array_split(rng.permutation(n), k_pliegues)
    return [(np.concatenate([folds[j] for j in range(k_pliegues) if j != i]), folds[i])
            for i in range(k_pliegues)]


def pliegues_ajuste(train: pd.DataFrame, estrategia: str = "k",
                    K: int | None = None, lam_norm: float | None = None,
                    beta: float | None = None, k_pliegues: int | None = None,
                    semilla: int | None = None,
                    universo_rebalanceo: pd.DataFrame | None = None,
                    cap: int | None = None,
                    con_arrepentimiento: bool = False) -> dict[str, Any]:
    """K-Fold: ajusta en k−1 pliegues y EVALÚA en el retenido (por inserción)."""
    from .ajuste import ajustar, evaluar

    semilla = cfg.SEMILLA_MODELO if semilla is None else int(semilla)
    d = train.copy().reset_index(drop=True)
    if "_uid" not in d.columns:
        d["_uid"] = np.arange(len(d))
    if "x_km" not in d.columns:
        from .geo import proyectar
        proyectar(d, col_grupo=None)

    filas: list[dict[str, Any]] = []
    for i, (tr_i, te_i) in enumerate(kfold_indices(len(d), k_pliegues, semilla)):
        tr = d.iloc[tr_i].reset_index(drop=True)
        te = d.iloc[te_i].reset_index(drop=True)
        todo = pd.concat([tr, te], ignore_index=True)
        uni = todo if (universo_rebalanceo is not None
                       or estrategia == "rebalanceo") else None
        mod = ajustar(tr, estrategia=estrategia, K=K, lam_norm=lam_norm, beta=beta,
                      semilla=semilla, cap=cap, universo_rebalanceo=uni)
        ev = evaluar(mod, todo)
        ins = ev["insercion"]
        fact = ev["factibilidad"]
        km_plan = ev["km_total"]
        costo_det_M = (cfg.costo_fijo_ciclo() * mod.K
                       + km_plan * cfg.COSTO_KM) / 1e6
        fila: dict[str, Any] = {
            "pliegue": i,
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_asesores": int(mod.K),
            "K": int(mod.K),
            "lam_norm": float(mod.lam_norm), "beta": float(mod.beta),
            "km_ciclo": round(float(km_plan), 1),
            "costo_det_M": round(float(costo_det_M), 2),
            "pico_carga": int(fact.get("pico_global", 0)),
            "dias_sobre_cap": int(fact.get("dias_sobre_cap", 0)),
            "factible": bool(fact.get("factible", False)),
            "cv_carga": round(float(fact.get("cv_carga_asesor", float("nan"))), 4),
            "cobertura_train_exacta": bool(ev["cobertura_train_exacta"]),
            # generalización: qué cuesta meter a los clientes de test
            "km_extra_p50": ins.get("km_extra_cliente_p50"),
            "km_extra_p90": ins.get("km_extra_cliente_p90"),
            "pct_ruta_nueva": ins.get("pct_ruta_nueva"),
            "pct_excede_cap": ins.get("pct_excede_cap"),
            "pct_no_insertados": round(100.0 - ins.get("pct_insertados_completos", 100.0), 2)
            if ins.get("n_test") else 0.0,
            "arrepentimiento_pct": None,
        }
        if con_arrepentimiento:
            # arrepentimiento: ¿cuánto km cuesta haber planificado las rutas sin ver
            # a los clientes retenidos? Se reconstruyen las rutas con los MISMOS
            # territorios y depósitos del modelo (sin re-agrupar) y TODOS los
            # clientes del pliegue.
            from .ajuste import km_de_rutas, rutas_diarias
            labels_t = np.array([
                mod.territorio_de[int(u)] if int(u) in mod.vistos
                else mod.territorio_para_cliente(z, float(x), float(y))
                for u, z, x, y in zip(todo["_uid"], todo[cfg.COL_ASESOR],
                                      todo["x_km"], todo["y_km"])], dtype=int)
            km_refit = km_de_rutas(rutas_diarias(todo, labels_t))
            fila["arrepentimiento_pct"] = (round(100.0 * (km_plan - km_refit) / km_refit, 3)
                                           if km_refit else None)
        filas.append(fila)

    resumen: dict[str, dict] = {}
    for k in METRICAS_KFOLD + METRICAS_TEST:
        vals = np.asarray([f[k] for f in filas if f.get(k) is not None], dtype=float)
        if not len(vals):
            resumen[k] = {"media": None, "desv": None, "min": None, "max": None}
            continue
        resumen[k] = {
            "media": round(float(np.nanmean(vals)), 4),
            "desv": round(float(np.nanstd(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
            "min": round(float(np.nanmin(vals)), 4),
            "max": round(float(np.nanmax(vals)), 4),
        }
    resumen["pct_factible"] = round(
        100.0 * float(np.mean([f["factible"] for f in filas])), 1)

    # K real del modelo ajustado (para "rebalanceo"/"actual" K no se pasa como
    # parámetro: sale del primer pliegue, que tiene el mismo K que el modelo final)
    k_real = (int(filas[0]["n_asesores"]) if filas
              else (int(K) if K is not None else cfg.K_ACTUAL))
    return {
        "estrategia": estrategia,
        "K": k_real,
        "lam_norm": lam_norm, "beta": beta,
        "k_territorios": k_real,
        "k_pliegues": cfg.K_PLIEGUES if k_pliegues is None else int(k_pliegues),
        "metricas": list(METRICAS_KFOLD + METRICAS_TEST),
        "nota": ("K-Fold sobre el 80 % de train (optimización, no clasificación): en "
                 "cada pliegue se ajusta en k−1 y se evalúa en el retenido, que entra "
                 "SOLO por inserción más barata (tope, ventana y fechas). "
                 "`km_extra_p50/p90` = km extra marginal por cliente de test; "
                 "`arrepentimiento_pct` = km(plan) − km(re-plan de las mismas rutas "
                 "con todos los clientes del pliegue); "
                 "`costo_det_M` = fijo + combustible (sin horas extra: requieren F3)."),
        "pliegues": filas,
        "resumen": resumen,
    }


# ---------------------------------------------------------------------------
# β por pliegues (ACO sobre rutas diarias muestreadas)
# ---------------------------------------------------------------------------
def beta_por_pliegues(train: pd.DataFrame, beta_grid: tuple[float, ...] | None = None,
                      semilla: int | None = None,
                      n_rutas: int = 6) -> dict[str, Any]:
    """Elige β comparando ACO en rutas diarias de cada pliegue (`BETA_SEEDS` semillas).

    Criterio declarado: menor km medio; si dos β difieren < 1 % se prefiere el β
    menor (no se paga complejidad por una mejoria imperceptible).
    """
    from .ajuste import ajustar, rutas_diarias
    from .ruteo import aco

    beta_grid = cfg.BETA_GRID if beta_grid is None else tuple(beta_grid)
    semilla = cfg.SEMILLA_MODELO if semilla is None else int(semilla)
    d = train.copy().reset_index(drop=True)
    if "_uid" not in d.columns:
        d["_uid"] = np.arange(len(d))
    if "x_km" not in d.columns:
        from .geo import proyectar
        proyectar(d, col_grupo=None)

    muestra: list[dict] = []
    for tr_i, _te_i in kfold_indices(len(d), None, semilla):
        tr = d.iloc[tr_i].reset_index(drop=True)
        mod = ajustar(tr, estrategia="actual")
        rutas = rutas_diarias(tr, np.array([mod.territorio_de[int(u)]
                                            for u in tr["_uid"]]))
        # rutas medianas/grandes (las pequeñas no distinguen β)
        rutas.sort(key=lambda r: r["n_clientes"])
        muestra.extend(rutas[len(rutas) // 3: len(rutas) // 3 + n_rutas])

    filas: list[dict[str, Any]] = []
    for b in tuple(beta_grid):
        kms = []
        for s in range(cfg.BETA_SEEDS):
            tot = 0.0
            for r in muestra:
                _orden, d_aco, _t = aco(r["D"], r["slots"], m=cfg.ACO_M,
                                        alfa=cfg.ACO_ALFA, beta=float(b),
                                        rho=cfg.ACO_RHO, Q=cfg.ACO_Q,
                                        iteraciones=cfg.ACO_ITER,
                                        semilla=semilla + s)
                tot += d_aco
            kms.append(tot / max(1, len(muestra)))
        filas.append({"beta": float(b), "km_medio": round(float(np.mean(kms)), 3),
                      "desv": round(float(np.std(kms, ddof=1)), 3) if len(kms) > 1 else 0.0,
                      "semillas": cfg.BETA_SEEDS, "n_rutas": len(muestra)})

    filas.sort(key=lambda f: (f["km_medio"], f["beta"]))
    mejor = filas[0]
    segunda = filas[1] if len(filas) > 1 else None
    mejora_vs_siguiente = (round(100.0 * (segunda["km_medio"] - mejor["km_medio"])
                                 / segunda["km_medio"], 3) if segunda else None)
    return {
        "grid": [float(b) for b in beta_grid],
        "beta_elegido": float(mejor["beta"]),
        "criterio": ("menor km medio con ACO; si dos β difieren < 1 % se prefiere el "
                     "β menor (no se paga complejidad por una mejoria imperceptible)"),
        "mejora_vs_siguiente_pct": mejora_vs_siguiente,
        "filas": filas,
        "nota": ("β se calibra con las mismas rutas que se comparan (NN vs ACO), "
                 "usando las semillas de `ACO_SEEDS`. El plan NO usa ACO: β sólo "
                 "afecta a la ruta ACO del experimento comparativo."),
    }


# ---------------------------------------------------------------------------
# Selección de λ, K y β con los pliegues del 80 % de train
# ---------------------------------------------------------------------------
CRITERIO_SELECCION: str = (
    "1) se descarta toda configuración con días sobre el tope en ALGÚN pliegue; "
    "2) entre las factibles se elige la de menor costo determinista medio "
    "(fijo + combustible, M COP del ciclo); "
    "3) empate técnico (< 0,5 %) → menor km medio; si persiste → la λ menor y, "
    "para K, el K más pequeño.")


def _mejor_config(res: dict[str, dict]) -> tuple:
    """Aplica CRITERIO_SELECCION a un dict {clave: resultado de pliegues_ajuste}."""
    factibles = {k: v for k, v in res.items()
                 if v["resumen"]["pct_factible"] == 100.0}
    pool = factibles or res
    costo = {k: v["resumen"]["costo_det_M"]["media"] for k, v in pool.items()}
    mejor_costo = min(costo.values())
    # empate técnico: 0,5 % del mejor costo
    cerca = [k for k, c in costo.items() if c <= mejor_costo * 1.005]
    if len(cerca) > 1:
        kms = {k: pool[k]["resumen"]["km_ciclo"]["media"] for k in cerca}
        mejor_km = min(kms.values())
        cerca = [k for k, v in kms.items() if v <= mejor_km * 1.005]
    elegido = sorted(cerca)[0] if len(cerca) > 1 else cerca[0]
    return elegido, bool(factibles)


def seleccionar_config(train: pd.DataFrame,
                       k_grid: tuple[int, ...] | None = None,
                       lam_grid: tuple[float, ...] | None = None,
                       beta_grid: tuple[float, ...] | None = None,
                       semilla: int | None = None) -> dict[str, Any]:
    """Elige λ, K y β con los pliegues del 80 % de train (ver `CRITERIO_SELECCION`).

    λ y K se eligen con la estrategia "k" (reagrupación con restricción de
    capacidad); β con `beta_por_pliegues`. La rejilla y el criterio están
    declarados ANTES de ver los resultados.
    """
    k_grid = tuple(k_grid or cfg.K_CANDIDATOS)
    lam_grid = tuple(lam_grid or cfg.LAM_GRID)
    semilla = cfg.SEMILLA_MODELO if semilla is None else int(semilla)

    # 1) λ con K = K_ACTUAL
    por_lam = {float(l): pliegues_ajuste(train, estrategia="k", K=cfg.K_ACTUAL,
                                         lam_norm=float(l), semilla=semilla)
               for l in lam_grid}
    lam_elegido, hay_factibles_lam = _mejor_config(por_lam)

    # 2) K con la λ elegida
    por_k = {int(k): pliegues_ajuste(train, estrategia="k", K=int(k),
                                     lam_norm=float(lam_elegido), semilla=semilla)
             for k in k_grid}
    k_elegido, hay_factibles_k = _mejor_config(por_k)

    # 3) β con pliegues (ACO)
    beta = beta_por_pliegues(train, beta_grid=beta_grid, semilla=semilla)

    def _tabla(res: dict, clave: str) -> list[dict]:
        filas = []
        for k, v in res.items():
            filas.append({
                clave: k,
                "costo_det_M": v["resumen"]["costo_det_M"]["media"],
                "km_ciclo": v["resumen"]["km_ciclo"]["media"],
                "pico_carga": v["resumen"]["pico_carga"]["media"],
                "dias_sobre_cap": v["resumen"]["dias_sobre_cap"]["max"],
                "pct_factible": v["resumen"]["pct_factible"],
                "km_extra_p90": v["resumen"]["km_extra_p90"]["media"],
            })
        return filas

    return {
        "k_pliegues": cfg.K_PLIEGUES,
        "semilla": semilla,
        "rejilla": {"K": list(k_grid), "lam_norm": [float(x) for x in lam_grid],
                    "beta": list(beta_grid or cfg.BETA_GRID)},
        "criterio": CRITERIO_SELECCION,
        "lambda_elegida": lam_elegido,
        "K_elegido": int(k_elegido),
        "beta_elegido": beta["beta_elegido"],
        "beta": beta,
        "tabla_lambda": _tabla(por_lam, "lam_norm"),
        "tabla_K": _tabla(por_k, "K"),
        "todas_las_lambda_factibles": hay_factibles_lam,
        "todos_los_K_factibles": hay_factibles_k,
        "pliegues_lambda": por_lam[lam_elegido],
        "pliegues_K": por_k[int(k_elegido)],
        "nota": ("Selección hecha SOLO con el 80 % de train (K-Fold con los clientes "
                 "retenidos entrando por inserción). El test no participa: se usa una "
                 "sola vez en la evaluación final."),
    }
