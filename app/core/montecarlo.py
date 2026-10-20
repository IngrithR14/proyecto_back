"""Simulación de Monte Carlo (E6, E7, D3).

- E6  balance por asesor con llave única (garantizada por `territorio` global).
- E7  factibilidad por asesor-DÍA contra el CAP dinámico (restricción dura), por
      quincena; horas simuladas y prob. de no completar la jornada.
- D3  costo: fijo (mes×dur/30) + combustible (km×COSTO_KM) + SOLO horas extra.
      Visitas no completadas → métrica de cobertura, no pesos (COSTO_VISITA_PERDIDA=None).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import config as cfg
from .calendario import fechas_permitidas, quincena_de

T_SERV_H = cfg.T_SERV_MEDIO_MIN / 60.0


# ---------------------------------------------------------------------------
# Incertidumbre (números aleatorios comunes + columnas reservadas)
# ---------------------------------------------------------------------------
def generar_entradas(n_sim: int, n_cli: int, seed: int,
                     n_reserva: int | None = None,
                     p_asist: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Asistencia Bernoulli (n_sim × [n_cli+n_reserva]) y tráfico Triangular.

    `p_asist` permite evaluar escenarios distintos (0,70 / 0,85 / 1,00) sin tocar
    `cfg.P_ASIST`; la secuencia aleatoria es la misma (semilla fija).
    """
    n_reserva = cfg.N_RESERVA if n_reserva is None else n_reserva
    p = cfg.P_ASIST if p_asist is None else float(p_asist)
    rng = np.random.default_rng(seed)
    asistencia = rng.random((n_sim, n_cli + n_reserva)) < p
    trafico = rng.triangular(cfg.TRI_MIN, cfg.TRI_MOD, cfg.TRI_MAX, n_sim)
    return asistencia, trafico


# ---------------------------------------------------------------------------
# Carga por territorio y fecha (base de la factibilidad E7)
# ---------------------------------------------------------------------------
def carga_por_territorio_fecha(clientes: pd.DataFrame,
                               labels: np.ndarray | None = None,
                               col_asesor: str | None = None) -> pd.DataFrame:
    """Visitas por (territorio, fecha). `labels` remapea clientes a otro territorio."""
    col_asesor = col_asesor or cfg.COL_ASESOR
    df = clientes.copy().reset_index(drop=True)
    df["_terr"] = labels if labels is not None else df[col_asesor].values
    filas = []
    for _, r in df.iterrows():
        permitidas = fechas_permitidas(r["DIAS_REP"], r["FRE_REP"])
        if not permitidas:
            continue
        objetivo = int(r["visitas_mes_rep"])
        base, extra = divmod(objetivo, len(permitidas))
        for i, f in enumerate(permitidas):
            v = base + (1 if i < extra else 0)
            if v > 0:
                filas.append({"fecha": f, "territorio": r["_terr"], "visitas": v})
    out = pd.DataFrame(filas)
    if len(out):
        out["q"] = out["fecha"].map(quincena_de)
    return out


def factibilidad(ctf: pd.DataFrame, cap: int | None = None) -> dict[str, Any]:
    """E7: restricción dura por asesor-día (ningún día supera CAP) + utilización."""
    cap = cap or cfg.cap_dinamico()
    if not len(ctf):
        return {"cap_dinamico": cap, "factible": False}
    pico = ctf.groupby(["territorio", "fecha"])["visitas"].sum()
    carga_media = ctf.groupby("territorio")["visitas"].sum() / cfg.DIAS_HABILES_CICLO
    utilizacion = carga_media / cap
    util_q: dict[str, float] = {}
    for q, g in ctf.groupby("q"):
        x = g.groupby("territorio")["visitas"].sum() / cfg.DIAS_HABILES_QUINCENA
        util_q[f"Q{int(q)}"] = float((x / cap).max())
    dias_sobre = int((pico > cap).sum())
    terr_pico = pico.groupby(level=0).max()
    return {
        "cap_dinamico": cap,
        "cap_tope": cfg.CAP_TOPE,
        "factible": bool(dias_sobre == 0),
        "dias_sobre_cap": dias_sobre,
        "pico_global": int(pico.max()),
        "utilizacion_max": float(utilizacion.max()),
        "utilizacion_media": float(utilizacion.mean()),
        "utilizacion_por_quincena": util_q,
        "pct_asesores_pico_sobre_cap": float((terr_pico > cap).mean() * 100),
        "cv_carga_asesor": float(carga_media.std(ddof=0) / carga_media.mean())
        if carga_media.mean() else float("nan"),
        "cv_asesor_dia": float(ctf.groupby(["territorio", "fecha"])["visitas"]
                               .sum().std(ddof=0) / pico.mean()) if pico.mean() else float("nan"),
    }


def horas_por_escenario(ctf: pd.DataFrame, km_por_terr: dict,
                        trafico: np.ndarray) -> dict[str, Any]:
    """Horas por asesor-día y prob. de no completar la jornada (D3).

    Viaje = km/día × tráfico / VEL_BASE; servicio = visitas/día × T_SERV (10 min).

    ⚠ Referencia agregada (promedio de km por territorio ÷ días). El simulador F3
    real (`jornadas_por_escenario`) trabaja ruta por ruta con los km del escenario
    y el servicio triangular por visita; las cifras de la API salen de F3.
    """
    dias = cfg.DIAS_HABILES_CICLO
    km_dia = pd.Series({k: v / dias for k, v in km_por_terr.items()}, dtype=float)
    visitas_dia = ctf.groupby("territorio")["visitas"].sum() / dias
    km_dia = km_dia.reindex(visitas_dia.index).fillna(0.0)
    h_viaje = (km_dia * float(np.mean(trafico)) / cfg.VEL_BASE).to_numpy()
    h_serv = (visitas_dia * T_SERV_H).to_numpy()
    h_dia = h_viaje + h_serv
    extra_dia = np.maximum(0.0, h_dia - cfg.JORNADA_HORAS)
    return {
        "horas_media_dia": float(h_dia.mean()),
        "horas_max_dia": float(h_dia.max()),
        "prob_no_completar": float((h_dia > cfg.JORNADA_HORAS).mean()),
        "horas_extra_ciclo": float(extra_dia.sum() * dias),
        "pct_asesores_sobre_jornada": float((h_dia > cfg.JORNADA_HORAS).mean() * 100),
    }


# ---------------------------------------------------------------------------
# F3 — Simulador que sí ve la sobrecarga (ruta por ruta, no promedio del ciclo)
# ---------------------------------------------------------------------------
def jornadas_por_escenario(rutas: list[dict], asistencia: np.ndarray,
                           trafico: np.ndarray, t_serv_min: float | None = None,
                           semilla: int | None = None) -> dict[str, Any]:
    """F3: horas POR RUTA DIARIA en cada escenario y visitas completadas.

    - Viaje: km reales de la ruta en ese escenario (previo estricto: si un cliente
      no asiste, se lo salta) × tráfico / VEL_BASE.
    - Servicio: por visita atendida, Triangular(T_SERV_MIN, MOD, MAX) muestreada
      (o `t_serv_min` fijo para la sensibilidad de 8 min).
    - Visita **no completada** si el acumulado de horas supera JORNADA_HORAS.
    - Política de sobrecarga (cfg.POLITICA_SOBRECARGA):
        "recortar"    → la jornada termina en JORNADA_HORAS (0 horas extra, se
                        pierden las visitas que no caben);
        "horas_extra" → se atienden todas las visitas atendidas y el exceso se
                        paga como hora extra.
    """
    S = asistencia.shape[0]
    if not rutas:
        return {"visitas_prog": 0, "km": np.zeros(S), "visitas_atendidas": np.zeros(S),
                "visitas_completadas": np.zeros(S), "horas": np.zeros(S),
                "horas_extra": np.zeros(S), "dias_sobre_jornada": np.zeros(S),
                "n_rutas": 0, "cobertura_si_horas_extra": 1.0}
    fijo = t_serv_min is not None
    rng = np.random.default_rng(cfg.SEMILLA if semilla is None else semilla)

    km_tot = np.zeros(S)
    h_pot = np.zeros(S)                 # horas del día si se atiende todo lo atendido
    h_trab = np.zeros(S)                # horas realmente trabajadas (política activa)
    h_ext = np.zeros(S)                 # horas extra (solo "horas_extra")
    atend = np.zeros(S)
    comp = np.zeros(S)                  # visitas completadas (F3, política activa)
    comp_extra = np.zeros(S)            # visitas completadas si hubiera horas extra
    dias_sobre = np.zeros(S)
    n_rutas = 0
    recortar = cfg.POLITICA_SOBRECARGA == "recortar"

    for r in rutas:
        uid = np.asarray(r["uid"], int)
        n = len(uid)
        if n == 0:
            continue
        n_rutas += 1
        pos = np.arange(n + 1)
        A = np.concatenate([np.ones((S, 1), bool), asistencia[:, uid]], axis=1)
        idx = np.where(A, pos, -1)
        prev = np.maximum.accumulate(idx, axis=1)
        prev = np.concatenate([np.full((S, 1), -1), prev[:, :-1]], axis=1)   # previo estricto
        Do = r["D"][np.ix_(r["orden"], r["orden"])]
        contrib = Do[np.clip(prev, 0, None), pos] * A            # (S, n+1)
        km_acum = np.cumsum(contrib, axis=1)[:, 1:]              # km hasta cada visita
        km_r = contrib.sum(axis=1)

        if fijo:
            t = np.full((S, n), float(t_serv_min) / 60.0)
        else:
            t = rng.triangular(cfg.T_SERV_MIN, cfg.T_SERV_MOD, cfg.T_SERV_MAX,
                               (S, n)) / 60.0
        t = np.where(A[:, 1:], t, 0.0)
        tcum = np.cumsum(t, axis=1)
        h_cum = km_acum * trafico[:, None] / cfg.VEL_BASE + tcum      # horas por visita

        km_tot += km_r
        at = A[:, 1:].sum(axis=1)
        atend += at
        h_dia_r = km_r * trafico / cfg.VEL_BASE + t.sum(axis=1)    # horas potenciales
        h_pot += h_dia_r
        h_trab += np.minimum(h_dia_r, cfg.JORNADA_HORAS) if recortar else h_dia_r
        h_ext += np.maximum(0.0, h_dia_r - cfg.JORNADA_HORAS) if not recortar \
            else np.zeros(S)
        ok = (h_cum <= cfg.JORNADA_HORAS).sum(axis=1)                 # en orden de ruta
        comp_extra += ok                                              # con horas extra
        dias_sobre += (h_cum[:, -1] > cfg.JORNADA_HORAS).astype(float)
        comp += ok if recortar else at                                # política activa

    horas = h_trab
    horas_extra = h_ext

    prog = int(sum(len(r["uid"]) for r in rutas))
    return {
        "visitas_prog": prog,
        "km": km_tot,
        "visitas_atendidas": atend,
        "visitas_completadas": comp,
        "horas": horas,                     # horas trabajadas del ciclo por escenario
        "horas_potencial": h_pot,
        "horas_extra": horas_extra,
        "dias_sobre_jornada": dias_sobre,   # nº de rutas que exceden la jornada
        "n_rutas": n_rutas,
        "cobertura_si_horas_extra": float(comp_extra.mean() / prog) if prog else np.nan,
    }


def resumen_jornadas(jor: dict, trafico: np.ndarray) -> dict[str, Any]:
    """Resumen F3 para `resumen_estrategia`: horas/día, prob. y horas extra."""
    n = max(1, jor["n_rutas"])
    S = len(jor["horas"])
    h_dia = jor["horas_potencial"] / n
    return {
        "horas_media_dia": float(h_dia.mean()),
        "horas_max_dia": float(h_dia.max()),
        "prob_no_completar": float(jor["dias_sobre_jornada"].mean() / n),
        "horas_extra_ciclo": float(jor["horas_extra"].mean()),
        "pct_asesores_sobre_jornada": float(jor["dias_sobre_jornada"].mean() / n * 100),
        "visitas_completadas_media": float(jor["visitas_completadas"].mean()),
        "cobertura_si_horas_extra": jor["cobertura_si_horas_extra"],
        "politica_sobrecarga": cfg.POLITICA_SOBRECARGA,
    }


# ---------------------------------------------------------------------------
# Motor de costo por escenario (previo estricto)
# ---------------------------------------------------------------------------
def evaluar_estrategia(rutas: list[dict], asistencia: np.ndarray,
                       trafico: np.ndarray) -> dict[str, Any]:
    """km, visitas y horas por escenario; acumula km y visitas por territorio."""
    S = asistencia.shape[0]
    km = np.zeros(S)
    visitas = np.zeros(S)
    uniq = list(dict.fromkeys(r["territorio"] for r in rutas))
    idx_t = {t: i for i, t in enumerate(uniq)}
    km_T = np.zeros((S, len(uniq)))
    vis_T = np.zeros((S, len(uniq)))

    for r in rutas:
        uid = r["uid"]
        n = len(uid)
        pos = np.arange(n + 1)
        A = np.concatenate([np.ones((S, 1), bool), asistencia[:, uid]], axis=1)
        idx = np.where(A, pos, -1)
        prev = np.maximum.accumulate(idx, axis=1)
        prev = np.concatenate([np.full((S, 1), -1), prev[:, :-1]], axis=1)  # previo estricto
        Do = r["D"][np.ix_(r["orden"], r["orden"])]
        contrib = Do[np.clip(prev, 0, None), pos] * A
        k = contrib.sum(axis=1)
        v = A[:, 1:].sum(axis=1)
        km += k
        visitas += v
        ti = idx_t[r["territorio"]]
        km_T[:, ti] += k
        vis_T[:, ti] += v

    h_viaje = km * trafico / cfg.VEL_BASE
    h_serv = visitas * T_SERV_H
    return dict(km=km, visitas=visitas, horas=h_viaje + h_serv,
                km_territorio=km_T, visitas_territorio=vis_T, territorios=uniq)


def horas_extra_escenario(ev: dict, trafico: np.ndarray) -> np.ndarray:
    """Horas extra del ciclo POR ESCENARIO (D3): suma de excesos sobre JORNADA/día."""
    dias = cfg.DIAS_HABILES_CICLO
    km_dia = ev["km_territorio"] / dias          # (S, T)
    vis_dia = ev["visitas_territorio"] / dias    # (S, T)
    h = km_dia * trafico[:, None] / cfg.VEL_BASE + vis_dia * T_SERV_H   # (S, T)
    extra = np.maximum(0.0, h - cfg.JORNADA_HORAS)
    return extra.sum(axis=1) * dias               # horas extra del ciclo (S,)


# ---------------------------------------------------------------------------
# E6 · balance por asesor (llave única)
# ---------------------------------------------------------------------------
def balance_carga(carga_por_asesor: np.ndarray) -> float:
    x = np.asarray(carga_por_asesor, float)
    return float(x.std(ddof=0) / x.mean()) if x.mean() > 0 else float("nan")


# ---------------------------------------------------------------------------
# Costos (D3)
# ---------------------------------------------------------------------------
def costo_escenario(ev: dict, trafico: np.ndarray, n_asesores: int,
                    horas_extra_h: np.ndarray | None = None
                    ) -> dict[str, np.ndarray | float]:
    """Fijo + combustible + horas extra por escenario (D3).

    `horas_extra_h` (F3) trae las horas extra POR ESCENARIO que ya calculó el
    simulador con la política de sobrecarga activa; si no llega, se usa la
    referencia agregada `horas_extra_escenario` (legado).
    """
    km = ev["km"]
    extra_h = horas_extra_escenario(ev, trafico) if horas_extra_h is None \
        else np.asarray(horas_extra_h, dtype=float)
    fijo = cfg.costo_fijo_ciclo() * n_asesores
    combustible = km * cfg.COSTO_KM
    horas_extra = extra_h * cfg.COSTO_HORA * cfg.HORA_EXTRA_FACTOR
    total = fijo + combustible + horas_extra
    return dict(fijo=float(fijo), combustible=combustible, horas_extra=horas_extra,
                horas_extra_h=extra_h, total=total)


def resumen_estrategia(nombre: str, ev: dict, costo: dict, n_asesores: int,
                       fact: dict, horas: dict, visitas_prog: int,
                       visitas_completadas: float | None = None) -> dict[str, Any]:
    """Métricas de una estrategia: costo con IC95, km, factibilidad, balance, cobertura.

    `cobertura` = visitas COMPLETADAS dentro de la jornada ÷ programadas (F3).
    Con `visitas_completadas=None` (cálculo legado) se usa la asistencia, pero la
    cobertura real sólo existe con el simulador F3.
    """
    total = np.asarray(costo["total"]) / 1e6        # M COP
    km = ev["km"]
    ee = total.std(ddof=1) / np.sqrt(len(total))
    p95 = np.percentile(total, 95)
    comb = float(np.mean(costo["combustible"]))
    hext = float(np.mean(costo["horas_extra"]))
    fijo = costo["fijo"]
    suma = fijo + comb + hext
    visitas_sim = float(ev["visitas"].mean())
    asistencia = float(visitas_sim / visitas_prog) if visitas_prog else np.nan
    completadas = visitas_sim if visitas_completadas is None else float(visitas_completadas)
    cobertura = float(completadas / visitas_prog) if visitas_prog else np.nan
    return {
        "estrategia": nombre,
        "K": n_asesores,
        "costo_medio_M": float(total.mean()),
        "ic95_inf": float(total.mean() - 1.96 * ee),
        "ic95_sup": float(total.mean() + 1.96 * ee),
        "mediana_M": float(np.median(total)),
        "p95_M": float(p95),
        "cvar5_M": float(total[total >= p95].mean()),
        "km_medio": float(km.mean()),
        "visitas_media": visitas_sim,
        "cobertura": cobertura,                       # F3: completadas/programadas
        "cobertura_asistencia": asistencia,
        "cobertura_si_horas_extra": horas.get("cobertura_si_horas_extra"),
        "politica_sobrecarga": horas.get("politica_sobrecarga", cfg.POLITICA_SOBRECARGA),
        "visitas_perdidas": float(max(0.0, visitas_prog - completadas)),
        "factible": fact.get("factible"),
        "dias_sobre_cap": fact.get("dias_sobre_cap"),
        "utilizacion_max": fact.get("utilizacion_max"),
        "cv_carga": fact.get("cv_carga_asesor"),
        "cv_asesor_dia": fact.get("cv_asesor_dia"),
        "pico_carga": fact.get("pico_global"),
        "prob_no_completar": horas.get("prob_no_completar"),
        "horas_media_dia": horas.get("horas_media_dia"),
        "pct_fijo": float(fijo / suma * 100) if suma else np.nan,
        "pct_combustible": float(comb / suma * 100) if suma else np.nan,
        "pct_extra": float(hext / suma * 100) if suma else np.nan,
    }


# ---------------------------------------------------------------------------
# Comparación pareada contra ACTUAL (P2, D3.7)
# ---------------------------------------------------------------------------
def comparar_pareada(costo_K: np.ndarray, costo_actual: np.ndarray) -> dict[str, Any]:
    """Δ de costo por escenario contra ACTUAL (K=58) con IC95 y P(mejor)."""
    d = (np.asarray(costo_K) - np.asarray(costo_actual)) / 1e6
    ee = d.std(ddof=1) / np.sqrt(len(d))
    return {
        "dif_media_M": float(d.mean()),
        "ic95_inf": float(d.mean() - 1.96 * ee),
        "ic95_sup": float(d.mean() + 1.96 * ee),
        "p_mejor": float((d < 0).mean()),
    }
