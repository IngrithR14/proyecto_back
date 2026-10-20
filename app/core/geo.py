"""Utilidades geográficas: haversina y proyección a km."""
from __future__ import annotations

import numpy as np

from .. import config as cfg


def haversine_mat(lat, lon) -> np.ndarray:
    """Matriz de distancias haversina (km) para arreglos de lat/long."""
    la = np.radians(np.asarray(lat, float))[:, None]
    lo = np.radians(np.asarray(lon, float))[:, None]
    dla = la - la.T
    dlo = lo - lo.T
    a = np.sin(dla / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlo / 2) ** 2
    return 2 * cfg.R_TIERRA * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def proyectar(df, col_lat: str = "Latitud", col_lon: str = "Longitud",
              col_grupo: str | None = None) -> None:
    """Añade columnas x_km/y_km (equirrectangular, φ₀ = lat media del grupo)."""
    if col_grupo and col_grupo in df.columns:
        phi0 = np.radians(df.groupby(col_grupo)[col_lat].transform("mean"))
    else:
        phi0 = np.radians(df[col_lat].mean())
    df["x_km"] = cfg.R_TIERRA * np.radians(df[col_lon]) * np.cos(phi0)
    df["y_km"] = cfg.R_TIERRA * np.radians(df[col_lat])
