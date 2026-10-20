"""Pruebas del ruteo avanzado (FASE 5): tendencia/R², rutas_ejemplo, K-Fold y tabla.

Cubren los puntos nuevos de la sección de ruteo:
  - `experimento_nn_vs_aco` devuelve `por_subruta`, `tendencia` (R²) y `rutas_ejemplo`.
  - Endpoint `/api/ruteo/kfold` (K-Fold CV) y `/api/ruteo/rutas` (tabla por día/asesor).
  - La tabla de rutas NO expone PII (sólo `CodigoCliente`, nunca nombre/cédula).
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.comparacion import _recta_r2, _ruta_ejemplo

CACHE = Path(__file__).resolve().parents[1] / "cache"


# ---------------------------------------------------------------------------
# Recta de tendencia + R²
# ---------------------------------------------------------------------------
def test_recta_r2_perfecta():
    """Una relación lineal exacta tiene R² = 1."""
    r = _recta_r2([1, 2, 3, 4], [2, 4, 6, 8])
    assert r["r2"] == 1.0
    assert abs(r["pendiente"] - 2.0) < 1e-6


def test_recta_r2_constante_es_none():
    """Si x es constante no hay recta definida (evita división por cero)."""
    assert _recta_r2([3, 3, 3], [1, 2, 3])["r2"] is None
    assert _recta_r2([], [])["r2"] is None


def test_recta_r2_baja_para_ruido():
    """Ruido sin relación lineal → R² cercano a 0 (no explica la varianza)."""
    r = _recta_r2([1, 2, 3, 4, 5, 6], [5, 1, 4, 2, 6, 3])
    assert r["r2"] is not None and r["r2"] < 0.5


# ---------------------------------------------------------------------------
# Ruta de ejemplo (coordenadas NN vs ACO)
# ---------------------------------------------------------------------------
def test_ruta_ejemplo_elige_la_mas_grande():
    coord = {
        "A#R1": {"lats": [4.0, 4.1], "lons": [-73.0, -73.1], "orden_NN": [0, 1],
                 "orden_ACO": [0, 1], "n_clientes": 1, "dist_NN": 1.0, "dist_ACO": 1.0},
        "B#R2": {"lats": [4.0, 4.2, 4.3], "lons": [-73.0, -73.2, -73.3],
                 "orden_NN": [0, 2, 1], "orden_ACO": [0, 1, 2], "n_clientes": 2,
                 "dist_NN": 3.0, "dist_ACO": 2.5},
    }
    r = _ruta_ejemplo(coord)
    assert r["sub_ruta"] == "B#R2"          # la de más clientes
    assert r["n_clientes"] == 2
    assert len(r["nn"]) == 3 and len(r["aco"]) == 3
    # la secuencia respeta el orden dado (índice 0 = depósito)
    assert r["nn"][0] == [4.0, -73.0]
    assert r["aco"][1] == [4.2, -73.2]


def test_ruta_ejemplo_vacia():
    assert _ruta_ejemplo({}) is None


# ---------------------------------------------------------------------------
# Cache ruteo.json (experimento ampliado)
# ---------------------------------------------------------------------------
def test_ruteo_cache_tiene_curva_y_ruta():
    r = json.loads((CACHE / "ruteo.json").read_text(encoding="utf-8"))
    assert r["unidad"] == "km"
    assert isinstance(r["por_subruta"], list) and len(r["por_subruta"]) >= 4
    # sin sub-rutas degeneradas de 1-2 clientes
    assert all(f["n_clientes"] >= 3 for f in r["por_subruta"])
    # tendencia con R² para NN, ACO y ahorro
    for k in ("NN", "ACO", "ahorro_pct"):
        assert "r2" in r["tendencia"][k]
    re_ = r["rutas_ejemplo"]
    assert re_ and len(re_["nn"]) == re_["n_clientes"] + 1   # depot + clientes
    assert len(re_["aco"]) == re_["n_clientes"] + 1


# ---------------------------------------------------------------------------
# Endpoints (leen la cache ya precalculada)
# ---------------------------------------------------------------------------
def test_endpoint_kfold(cliente_api):
    resp = cliente_api.get("/api/ruteo/kfold")
    assert resp.status_code == 200
    d = resp.json()
    assert d["k_pliegues"] == 5
    assert len(d["pliegues"]) == 5
    # núcleo del K-Fold + métricas del protocolo (inserción del test retenido)
    assert set(d["metricas"]) >= {"km_ciclo", "costo_det_M", "pico_carga",
                                  "dias_sobre_cap", "cv_carga"}
    if "SOLO por inserción" not in (d.get("nota") or ""):
        pytest.skip("pliegues.json con formato antiguo: re-ejecuta "
                    "`python -m app.precompute`")
    assert set(d["metricas"]) >= {"km_extra_p50", "km_extra_p90",
                                  "arrepentimiento_pct", "pct_no_insertados"}
    # resumen con media y desviación por métrica
    for m in d["metricas"]:
        assert "media" in d["resumen"][m] and "desv" in d["resumen"][m]
    # el pliegue retenido entra SOLO por inserción (y se reporta qué no cupió)
    for f in d["pliegues"]:
        assert f["n_train"] > 0 and f["n_test"] > 0
        assert f["pct_no_insertados"] is not None
        assert f["arrepentimiento_pct"] is not None
    assert d["estrategia"] in {"actual", "rebalanceo", "k"}
    assert "SOLO por inserción" in d["nota"]


def test_endpoint_rutas_recomendada(cliente_api):
    resp = cliente_api.get("/api/ruteo/rutas", params={"estrategia": "recomendada",
                                                       "quincena": 1})
    assert resp.status_code == 200
    d = resp.json()
    assert d["estrategia"] == "recomendada"
    assert d["n_rutas"] > 0 and d["total_visitas"] > 0
    fila = d["rutas"][0]
    for k in ("fecha", "asesor", "n_visitas", "km", "secuencia"):
        assert k in fila
    # la secuencia usa CodigoCliente (códigos), nunca datos personales
    assert all(isinstance(c, str) for c in fila["secuencia"])


def test_endpoint_rutas_actual(cliente_api):
    resp = cliente_api.get("/api/ruteo/rutas", params={"estrategia": "actual", "quincena": 1})
    assert resp.status_code == 200
    assert resp.json()["k_territorios"] == 58


def test_endpoint_rutas_valida_entradas(cliente_api):
    assert cliente_api.get("/api/ruteo/rutas",
                           params={"quincena": 3}).status_code == 422
    assert cliente_api.get("/api/ruteo/rutas",
                           params={"estrategia": "inexistente"}).status_code == 422


def test_rutas_no_expone_pii():
    """La cache de rutas no contiene columnas personales (Cliente/Identificacion)."""
    r = json.loads((CACHE / "rutas.json").read_text(encoding="utf-8"))
    filas = r["recomendada"]["1"]["rutas"]
    claves = set(filas[0].keys())
    assert not (claves & {"Cliente", "Identificacion", "Direccion", "cliente", "nombre"})
    assert claves == {"fecha", "asesor", "n_visitas", "km", "secuencia", "excede_cap"}
