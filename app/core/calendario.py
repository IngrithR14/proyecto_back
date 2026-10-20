"""Calendario de referencia (D2): reparte FRE_REP + DIAS_REP en fechas reales.

Genera la carga **asesor-día** (restricción dura de factibilidad) a partir del
mes ANIO_REF/MES_REF. El total de visitas cuadra con visitas_mes_rep.sum().
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .. import config as cfg

_DIAS_PY = {
    "LUNES": calendar.MONDAY, "MARTES": calendar.TUESDAY,
    "MIERCOLES": calendar.WEDNESDAY, "JUEVES": calendar.THURSDAY,
    "VIERNES": calendar.FRIDAY, "SABADO": calendar.SATURDAY,
    "DOMINGO": calendar.SUNDAY,
}


def meses_referencia() -> list[date]:
    """Todas las fechas del mes de referencia."""
    n = calendar.monthrange(cfg.ANIO_REF, cfg.MES_REF)[1]
    return [date(cfg.ANIO_REF, cfg.MES_REF, d) for d in range(1, n + 1)]


def quincena_de(f: date) -> int:
    return 1 if f.day <= 15 else 2


def fechas_permitidas(dias_rep: str, fre_rep: str) -> list[date]:
    """Fechas en que aplica un cliente: su día de la semana, en sus quincenas."""
    wd = _DIAS_PY.get(str(dias_rep).strip().upper())
    if wd is None:
        return []
    todas = [f for f in meses_referencia() if f.weekday() == wd]
    if cfg.DURACION_CICLO_DIAS == 15 or str(fre_rep).strip() == "Q1":
        return [f for f in todas if quincena_de(f) == 1]
    if str(fre_rep).strip() == "Q2":
        return [f for f in todas if quincena_de(f) == 2]
    return todas  # Q1-Q2 → todo el mes


def construir_calendario(clientes: pd.DataFrame, col_asesor: str | None = None
                         ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Devuelve (visitas_asignadas, meta).

    visitas_asignadas: una fila por (fecha, asesor) con nº de visitas.
    Reparte visitas_mes_rep sobre las fechas permitidas; si difiere del nº de
    ocurrencias del día, distribuye visitas_mes_rep para que el total cuadre.
    """
    col_asesor = col_asesor or cfg.COL_ASESOR
    filas: list[dict[str, Any]] = []
    descuadre = 0
    for _, r in clientes.iterrows():
        permitidas = fechas_permitidas(r["DIAS_REP"], r["FRE_REP"])
        if not permitidas:
            continue
        objetivo = int(r["visitas_mes_rep"])
        if len(permitidas) != objetivo:
            descuadre += 1
        # reparto: se llenan las fechas permitidas; si hay más fechas que visitas,
        # se distribuyen equitativamente; si menos, se apila (más de 1 visita/día)
        base, extra = divmod(objetivo, len(permitidas))
        for i, f in enumerate(permitidas):
            v = base + (1 if i < extra else 0)
            if v > 0:
                filas.append({"fecha": f, "dia_semana": r["DIAS_REP"],
                              col_asesor: r[col_asesor], "visitas": v,
                              "equipo": r.get(cfg.COL_EQUIPO)})
    va = pd.DataFrame(filas)
    meta = {
        "anio_ref": cfg.ANIO_REF, "mes_ref": cfg.MES_REF,
        "n_fechas": len(meses_referencia()),
        "descuadre_ocurrencias": descuadre,   # clientes cuyas ocurrencias ≠ visitas_mes_rep
        "total_visitas": int(va["visitas"].sum()) if len(va) else 0,
        "escenario_dias": cfg.ESCENARIO_DIAS,
    }
    return va, meta


def carga_asesor_dia(va: pd.DataFrame, col_asesor: str | None = None
                     ) -> pd.DataFrame:
    """Carga por (asesor, fecha) — la unidad de la restricción dura."""
    col_asesor = col_asesor or cfg.COL_ASESOR
    return (va.groupby([col_asesor, "fecha"], as_index=False)["visitas"].sum()
              .rename(columns={col_asesor: "asesor"}))


def resumen_carga(va: pd.DataFrame, col_asesor: str | None = None
                  ) -> dict[str, Any]:
    """Métricas de carga: promedio, pico, CV (ambas lecturas), por día de semana."""
    col_asesor = col_asesor or cfg.COL_ASESOR
    if not len(va):
        return {}
    cad = carga_asesor_dia(va, col_asesor)

    # lectura (a): media mensual por asesor / (días que opera) → visitas/día
    mensual = va.groupby(col_asesor)["visitas"].sum()
    dias_op = va.groupby(col_asesor)["fecha"].nunique()
    media_por_asesor = (mensual / dias_op).replace([np.inf, -np.inf], np.nan).dropna()

    # lectura (b): carga asesor-día
    cargas_dia = cad["visitas"]
    pico_por_asesor = cad.groupby("asesor")["visitas"].max()

    def _cv(x: pd.Series) -> float:
        return float(x.std(ddof=0) / x.mean()) if len(x) and x.mean() else float("nan")

    tot_dia = va.groupby("dia_semana")["visitas"].sum()
    n_ocur = va.groupby("dia_semana")["fecha"].nunique().replace(0, 1)
    por_dia = tot_dia / n_ocur

    return {
        "cv_mensual": _cv(media_por_asesor),          # hoy ≈ 0,106
        "cv_asesor_dia": _cv(cargas_dia),             # incluye efecto DIAS_REP
        "media_visitas_dia": float(cargas_dia.mean()),
        "pico_global": int(cargas_dia.max()),
        "pico_medio_por_asesor": float(pico_por_asesor.mean()),
        "pico_max_por_asesor": int(pico_por_asesor.max()),
        "pct_sobre_cap": float((pico_por_asesor > cfg.CAP_TOPE).mean() * 100),
        "pct_mayor_45": float((cargas_dia > 45).mean() * 100),
        "pct_menor_40": float((cargas_dia < 40).mean() * 100),
        "pct_asesores_pico_gt45": float(
            (pico_por_asesor > 45).mean() * 100),
        "pct_asesores_pico_gt40": float(
            (pico_por_asesor > 40).mean() * 100),
        "visitas_por_dia_semana": {k: round(float(v), 1) for k, v in por_dia.items()},
        "total_visitas": int(va["visitas"].sum()),
        "n_asesores": int(va[col_asesor].nunique()),
    }


def fechas_por_cliente(clientes: pd.DataFrame,
                       col_asesor: str | None = None) -> tuple[list[np.ndarray], list[date], np.ndarray]:
    """Índices de fecha de visita de CADA cliente (para balanceo por fecha).

    Devuelve (indices_por_cliente, fechas_unicas, total_por_fecha). Cada visita
    cae en una fecha; `total_por_fecha[f]` es el total global de esa fecha.
    """
    col_asesor = col_asesor or cfg.COL_ASESOR
    # fecha -> índice
    todas = meses_referencia()
    f2i = {f: i for i, f in enumerate(todas)}
    n = len(todas)
    total = np.zeros(n)
    idx_por_cliente: list[np.ndarray] = []
    for _, r in clientes.iterrows():
        permitidas = fechas_permitidas(r["DIAS_REP"], r["FRE_REP"])
        objetivo = int(r["visitas_mes_rep"])
        if not permitidas:
            idx_por_cliente.append(np.array([], dtype=int))
            continue
        base, extra = divmod(objetivo, len(permitidas))
        sel: list[int] = []
        for i, f in enumerate(permitidas):
            v = base + (1 if i < extra else 0)
            for _ in range(v):
                sel.append(f2i[f])
                total[f2i[f]] += 1
        idx_por_cliente.append(np.array(sel, dtype=int))
    return idx_por_cliente, todas, total
