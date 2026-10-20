"""Invariantes del núcleo y del MonteCarlo (FASE 6).

Verifica propiedades que deben cumplir SIEMPRE los resultados del núcleo: embudo
monotónico, cobertura, desglose de costos, IC95 que contiene la media y la
frontera de factibilidad. Lee la cache (`cache/montecarlo.json`) cuando existe y
la omite (`pytest.skip`) si aún no está; así el test es rápido y estable.
"""
from __future__ import annotations

import json
import os

import pytest

from app import config as cfg


def _mc() -> dict:
    ruta = os.path.join(cfg.CACHE_DIR, "montecarlo.json")
    if not os.path.exists(ruta):
        pytest.skip("cache/montecarlo.json no existe (ejecuta `python -m app.precompute`)")
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Universo (E11) — embudo monotónico
# ---------------------------------------------------------------------------
def test_universo_embudo_monotonico(universo):
    filas = [p["filas"] for p in universo.embudo]
    assert filas == sorted(filas, reverse=True), "el embudo debe ser decreciente"
    assert universo.n == filas[-1]
    assert universo.n == 14757
    assert universo.n_zonas_base == 67 and universo.n_zonas_rutable == 58
    assert len(universo.zonas_perdidas) == 9


def test_calendario_cubre_universo(universo):
    from app.core.calendario import construir_calendario
    va, meta = construir_calendario(universo.clientes)
    # `va` tiene una fila por ocurrencia de visita; meta resume el total
    assert meta["total_visitas"] == 45212
    assert meta["descuadre_ocurrencias"] == 0
    assert len(va) == 45212


# ---------------------------------------------------------------------------
# ACTUAL (E8) — construir_actual desde ZON_REP
# ---------------------------------------------------------------------------
def test_actual_cobertura(universo):
    from app.core.estructura import construir_actual
    rutas = construir_actual(universo.clientes)          # devuelve sólo la lista
    zonas = {r["territorio"] for r in rutas}
    total = sum(len(r["uid"]) for r in rutas)
    assert total == len(universo.clientes), "ACTUAL debe cubrir todo el universo"
    assert len(zonas) == 58


def test_actual_no_factible(universo):
    """El ACTUAL tiene días sobre CAP (discrepancia #6 documentada)."""
    from app.core.montecarlo import carga_por_territorio_fecha, factibilidad
    ctf = carga_por_territorio_fecha(universo.clientes, labels=None)   # labels=None → ZON_REP
    f = factibilidad(ctf)
    assert f["pico_global"] > cfg.cap_dinamico()
    assert f["dias_sobre_cap"] >= 1
    assert f["factible"] is False


# ---------------------------------------------------------------------------
# MonteCarlo desde cache — invariantes D3 / IC95
# ---------------------------------------------------------------------------
def test_cache_invariantes_basicos():
    mc = _mc()
    a = mc.get("actual")
    assert a, "falta ACTUAL en cache"
    assert 0 <= a["cobertura"] <= 1
    assert a["ic95_inf"] <= a["costo_medio_M"] <= a["ic95_sup"]
    # desglose de costos ≈ 100 %
    for pct in ("pct_fijo", "pct_combustible", "pct_extra"):
        assert a[pct] >= 0
    assert a["pct_fijo"] + a["pct_combustible"] + a["pct_extra"] <= 101
    assert a["km_medio"] > 0 and a["costo_medio_M"] > 0


def test_cache_estrategias_ic95():
    mc = _mc()
    est = mc.get("estrategias", {})
    if len(est) < 5:
        pytest.skip("pocas estrategias cacheadas aún")
    for k, r in est.items():
        assert r["ic95_inf"] <= r["costo_medio_M"] <= r["ic95_sup"], f"K={k} IC95"
        assert 0 <= r["cobertura"] <= 1, f"K={k}"
        assert 0 <= r["pct_fijo"] + r["pct_combustible"] + r["pct_extra"] <= 101


def test_cache_frontera_factibilidad():
    """Frontera de factibilidad: K=35 infeasible; una vez factible no regresa.

    El K_mín real se DERIVA de la cache (no se hardcodea): es el primer K con
    `factible=True` y resulta ser 43 (pico=44 = CAP, límite exacto del criterio
    estricto `pico > CAP`). El teórico ⌈pico/CAP⌉≈41 queda como discrepancia.
    """
    mc = _mc()
    est = {int(k): r for k, r in mc.get("estrategias", {}).items()}
    if len(est) < 10:
        pytest.skip("pocas estrategias cacheadas aún")
    ks = sorted(est)
    if 35 in est:
        assert est[35]["factible"] is False, "K=35 debe ser infeasible"
    # monótona: una vez factible, ningún K superior debe ser infeasible
    visto_si = False
    for k in ks:
        if est[k]["factible"]:
            visto_si = True
        elif visto_si:
            pytest.fail(f"K={k} infeasible tras un K factible (no monótono)")
    factibles = [k for k in ks if est[k]["factible"]]
    if factibles:
        kmin = min(factibles)
        assert 41 <= kmin <= 46, f"frontera K_mín={kmin} fuera del rango esperado"
        assert all(est[k]["factible"] for k in ks if k >= kmin), \
            f"todo K>={kmin} debe ser factible"


def test_cache_comparaciones_pareadas():
    mc = _mc()
    comp = mc.get("comparaciones", {})
    if not comp:
        pytest.skip("sin comparaciones aún")
    for k, c in comp.items():
        assert "dif_media_M" in c and "ic95_inf" in c and "p_mejor" in c
        assert 0 <= c["p_mejor"] <= 1, f"K={k} p_mejor fuera de [0,1]"


# ---------------------------------------------------------------------------
# Paridad / coherencia entre secciones
# ---------------------------------------------------------------------------
def test_visitas_mes_coherentes(universo, base_eda):
    """45.212 = suma de visitas_mes_rep del universo; 48.494 de la base EDA."""
    assert int(universo.clientes["visitas_mes_rep"].sum()) == 45212
    assert int(base_eda["visitas_mes_rep"].sum()) == 48494


def test_unidad_asesor_zon_rep():
    assert cfg.UNIDAD_ASESOR == "zon_rep" and cfg.COL_ASESOR == "ZON_REP"
    # la advertencia obsoleta de E8 fue eliminada (D1)
    import app.core.estructura as es
    assert not hasattr(es, "ASESOR_DEFINICION_PENDIENTE")


def test_costo_asesor_1800000():
    """D3: el costo mensual del asesor es 1.800.000 COP (discrepancia #2)."""
    assert cfg.COSTO_ASESOR_MES == 1_800_000
    assert cfg.costo_fijo_ciclo() == pytest.approx(1_800_000 * cfg.DURACION_CICLO_DIAS / 30)


def test_ruta_abierta():
    """D4: ruta abierta (el último cliente se sirve)."""
    assert cfg.RUTA_CERRADA is False
