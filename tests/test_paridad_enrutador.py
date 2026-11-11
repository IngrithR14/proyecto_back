"""Paridad de enrutador: ACTUAL y las estrategias deben dar los mismos km.

`estructura.construir_rutas_diarias` (usada por ACTUAL / `precompute`) y
`ajuste.rutas_diarias` (usada por las estrategias del protocolo) son dos
implementaciones de "una ruta por (territorio, fecha)" con el mismo estándar:
NN + 2-opt sobre `dist_ruta` y depósito = centroide del territorio.

Este test las ejecuta con el MISMO input sintético y exige los mismos km, para
que una divergencia futura salte aquí.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from app.core import ajuste, estructura


def _clientes_sinteticos() -> tuple[pd.DataFrame, np.ndarray]:
    """12 clientes ficticios en 3 territorios (sin CSV, sin cache)."""
    filas = [{
        "DIAS_REP": ["LUNES", "MARTES", "MIERCOLES"][i % 3],
        "FRE_REP": "Q1-Q2",
        "visitas_mes_rep": 6,
        "Latitud": 4.60 + 0.01 * i,
        "Longitud": -74.10 - 0.01 * (i % 4),
        "slots": {0, 1, 2},
    } for i in range(12)]
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    return pd.DataFrame(filas), labels


def test_mismo_input_mismas_rutas_y_mismos_km():
    df, labels = _clientes_sinteticos()
    r_estr = estructura.construir_rutas_diarias(df, labels=labels)
    r_ajus = ajuste.rutas_diarias(df, labels)

    assert r_estr and r_ajus, "alguna rutina no produjo rutas"
    # mismo desglose (territorio, fecha) y mismos clientes por ruta
    clave = lambda rs: [(r["territorio"], str(r["fecha"]), r["n_clientes"])
                        for r in rs]
    assert clave(r_estr) == clave(r_ajus)

    # mismos km por ruta y el mismo total del ciclo (D3)
    km_estr = [r["km_base"] for r in r_estr]
    km_ajus = [r["km_base"] for r in r_ajus]
    assert km_estr == pytest.approx(km_ajus, rel=1e-9, abs=1e-9)
    assert estructura.km_total_ciclo(r_estr) == pytest.approx(
        ajuste.km_de_rutas(r_ajus), rel=1e-9, abs=1e-9)


def test_mismo_estandar_de_secuenciacion():
    """Ambas usan nearest_neighbor + dos_opt sobre dist_ruta (mismo estándar)."""
    for fn in (estructura.construir_rutas_diarias, ajuste.rutas_diarias):
        src = inspect.getsource(fn)
        for pieza in ("nearest_neighbor", "dos_opt", "dist_ruta", "haversine_mat"):
            assert pieza in src, f"{fn.__name__}: falta {pieza}"


def test_mismo_deposito():
    """Depósito = centroide del territorio en las dos (DEPOSITO_MODO='territorio')."""
    assert estructura.DEPOSITO_MODO == "territorio"
    for fn in (estructura.construir_rutas_diarias, ajuste.rutas_diarias):
        assert "_depot_de" in inspect.getsource(fn), fn.__name__
