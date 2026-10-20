"""Carga única del universo de trabajo (E11).

Una sola función `cargar_universo` devuelve embudo y clientes para que el EDA,
el modelo de ruteo y el Monte Carlo usen el MISMO conjunto.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .. import config as cfg

# Rangos de coordenadas (caja Boyacá/Cundinamarca)
LAT_MIN, LAT_MAX = 3.0, 6.2
LON_MIN, LON_MAX = -77.0, -72.5


@dataclass
class Universo:
    """Resultado de cargar_universo: clientes + embudo + métricas de calidad."""
    clientes: pd.DataFrame
    embudo: list[dict[str, int]] = field(default_factory=list)
    columnas_danadas: list[str] = field(default_factory=list)
    n_zonas_base: int = 0
    n_zonas_rutable: int = 0
    zonas_perdidas: list[str] = field(default_factory=list)
    visitas_totales_mes: int = 0

    @property
    def n(self) -> int:
        return len(self.clientes)


def _coords_ok(df: pd.DataFrame) -> pd.Series:
    lat = pd.to_numeric(df["Latitud"], errors="coerce")
    lon = pd.to_numeric(df["Longitud"], errors="coerce")
    return lat.between(LAT_MIN, LAT_MAX) & lon.between(LON_MIN, LON_MAX)


def columnas_con_reemplazo(df: pd.DataFrame) -> list[str]:
    """Columnas de texto que contienen caracteres U+FFFD dañados (E10)."""
    malas: list[str] = []
    for c in df.columns:
        try:
            if not pd.api.types.is_string_dtype(df[c]) and df[c].dtype != object:
                continue
        except (TypeError, ValueError):
            continue
        s = df[c].astype(str)
        if s.str.contains("\ufffd", regex=False).any():
            malas.append(c)
    return malas


def cargar_universo(ruta: str | None = None, umbral_grupo: int = cfg.UMBRAL_GRUPO,
                    umbral_cd: int = cfg.UMBRAL_CD) -> Universo:
    """Carga la base y aplica los filtros del universo de ruteo (E11).

    Filtros en orden: con asesor → activos → con visita → coordenadas válidas.
    """
    ruta = ruta or os.path.join(cfg.DATA_DIR, cfg.BASE_CSV)
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    n_base = len(df)

    zonas_base = set(df[cfg.COL_ASESOR].astype(str).str.strip()) - {""}

    embudo: list[dict[str, int]] = [{"paso": "Base limpia completa", "filas": n_base}]
    # El asesor es ZON_REP; no se exige Supervisor para definir el universo (D1).
    df = df[df[cfg.COL_ASESOR].astype(str).str.strip().ne("")].copy()
    embudo.append({"paso": f"Con {cfg.COL_ASESOR} (zona)", "filas": len(df)})

    df = df[df["activo"] == 1].copy()
    embudo.append({"paso": "Solo activos", "filas": len(df)})

    df = df[df["visitas_mes_rep"] > 0].copy()
    embudo.append({"paso": "Con visita asignada (visitas_mes_rep>0)", "filas": len(df)})

    df = df[_coords_ok(df)].copy()
    embudo.append({"paso": "Con coordenadas válidas", "filas": len(df)})

    # Feature engineering básico: ventana horaria por categoría (motor de ruteo)
    df["slots"] = df["CategoriaActividad"].map(lambda c: cfg.VENTANAS.get(c, {0, 1, 2}))
    df["visitas_mes_rep"] = pd.to_numeric(df["visitas_mes_rep"], errors="coerce").fillna(0).astype(int)

    # columnas con caracteres dañados (E10) — ANTES de normalizar
    malas = columnas_con_reemplazo(df)

    # Normaliza nombres de equipo dañados (E10): Sirley Rodríguez Usaquén, etc.
    from .asesores import normalizar_equipo
    df = normalizar_equipo(df)

    zonas_rut = set(df[cfg.COL_ASESOR].astype(str).str.strip())
    for i, paso in enumerate(embudo):
        prev = embudo[i - 1]["filas"] if i else n_base
        paso["eliminadas"] = prev - paso["filas"]

    u = Universo(
        clientes=df.reset_index(drop=True),
        embudo=embudo,
        columnas_danadas=malas,
        n_zonas_base=len(zonas_base),
        n_zonas_rutable=len(zonas_rut),
        zonas_perdidas=sorted(zonas_base - zonas_rut),
        visitas_totales_mes=int(pd.to_numeric(df["visitas_mes_rep"], errors="coerce").sum()),
    )
    return u


def resumen_universos(u: Universo) -> dict[str, Any]:
    """Vista para /api/meta/universos: embudo, zonas y diferencias por umbral."""
    df = u.clientes
    grupos = (df.groupby([cfg.COL_EQUIPO, "Cod CD"]).size().reset_index(name="n"))
    return {
        "unidad_asesor": cfg.UNIDAD_ASESOR,
        "n_clientes": u.n,
        "visitas_totales_mes": u.visitas_totales_mes,
        "visitas_por_dia": round(u.visitas_totales_mes / cfg.DIAS_HABILES_CICLO, 1),
        "embudo": u.embudo,
        "zonas_base": u.n_zonas_base,
        "zonas_rutable": u.n_zonas_rutable,
        "zonas_perdidas": u.zonas_perdidas,
        "grupos_asesor_cd": int(len(grupos)),
        "grupos_bajo_umbral": int((grupos["n"] < cfg.UMBRAL_GRUPO).sum()),
        "columnas_danadas": u.columnas_danadas,
    }
