"""Pruebas de la API FastAPI (FASE 3) — contrato de endpoints y privacidad.

Usa `TestClient`. Los endpoints del MonteCarlo devuelven 503 si la cache aún no
existe (precompute en curso): esos tests toleran 503 con `pytest.skip`.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Meta / salud
# ---------------------------------------------------------------------------
def test_salud(cliente_api):
    r = cliente_api.get("/api/salud")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_meta_params(cliente_api):
    r = cliente_api.get("/api/meta/params")
    assert r.status_code == 200
    d = r.json()
    assert d["k_min"] == 35 and d["k_max"] == 70 and d["k_actual"] == 58
    assert d["cap_dinamico"] == 44
    assert "params" in d and "COSTO_ASESOR_MES" in d["params"]
    # toda entrada lleva etiqueta de supuesto/calibrado
    for nombre, v in d["params"].items():
        assert "etiqueta" in v and "valor" in v, nombre


def test_meta_universos_sin_pii(cliente_api):
    r = cliente_api.get("/api/meta/universos")
    assert r.status_code == 200
    d = r.json()
    assert d["n_clientes"] == 14757
    assert "columnas_danadas" not in d   # interno E10, no se expone
    # privacidad: nada de columnas personales
    assert "Identificacion" not in d and "Cliente" not in d


def test_meta_calendario(cliente_api):
    r = cliente_api.get("/api/meta/calendario")
    assert r.status_code == 200
    assert "meta" in r.json() or "resumen" in r.json()


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def test_eda_resumen(cliente_api):
    r = cliente_api.get("/api/eda/resumen")
    assert r.status_code == 200
    d = r.json()
    for seccion in ("kpis", "distribuciones", "histogramas", "carga_zona",
                    "scatter", "crosstabs", "correlaciones"):
        assert seccion in d, seccion
    assert d["kpis"]["n_clientes"] == 16560
    # scatter sólo coordenadas + categorías (sin PII)
    if d["scatter"]:
        assert set(d["scatter"][0]) == {"lat", "lon", "cat", "dep"}


def test_eda_carga_zona(cliente_api):
    r = cliente_api.get("/api/eda/carga-zona", params={"top": 5})
    assert r.status_code == 200
    d = r.json()
    assert len(d["zonas"]) == 5 and d["n_zonas"] >= 58


def test_eda_correlaciones(cliente_api):
    r = cliente_api.get("/api/eda/correlaciones")
    assert r.status_code == 200
    d = r.json()
    assert "pearson" in d and "cramers_v" in d and "eta" in d


# ---------------------------------------------------------------------------
# Ruteo
# ---------------------------------------------------------------------------
def test_ruteo_comparacion(cliente_api):
    r = cliente_api.get("/api/ruteo/comparacion")
    assert r.status_code == 200
    d = r.json()
    assert set(d) >= {"n_seeds", "exp", "resumen", "wilcoxon", "victorias_NN", "victorias_ACO"}
    # E1 correcto: las filas tienen su propio % de victorias
    filas = {f["estrategia"]: f for f in d["resumen"]}
    assert set(filas) == {"NN", "ACO"}
    assert 0 <= filas["NN"]["% semillas en que gana"] <= 100
    assert 0 <= filas["ACO"]["% semillas en que gana"] <= 100


# ---------------------------------------------------------------------------
# MonteCarlo (toleran 503 si el precompute no terminó)
# ---------------------------------------------------------------------------
def _o_o_skip(cliente_api, *args, **kw):
    r = cliente_api.get(*args, **kw)
    if r.status_code == 503:
        pytest.skip("precompute MonteCarlo aún en curso")
    return r


def test_mc_tabla(cliente_api):
    r = _o_o_skip(cliente_api, "/api/montecarlo/tabla")
    assert r.status_code == 200
    d = r.json()
    assert d["actual"] is not None
    assert d["actual"]["K"] is None or d["actual"]["K"] == 58
    assert len(d["estrategias"]) >= 1


def test_mc_actual_factibilidad(cliente_api):
    r = _o_o_skip(cliente_api, "/api/montecarlo/actual")
    if r.status_code == 503:
        pytest.skip("ACTUAL no precalculado")
    a = r.json()
    # el ACTUAL NO es factible estricto (discrepancia documentada)
    assert a["factible"] is False
    assert a["pico_carga"] > 44


def test_mc_estrategia_rango(cliente_api):
    r = cliente_api.get("/api/montecarlo/estrategia/999")
    assert r.status_code == 422   # fuera de rango


def test_mc_estrategia_k(cliente_api):
    r = _o_o_skip(cliente_api, "/api/montecarlo/estrategia/45")
    if r.status_code == 503:
        pytest.skip("K=45 no precalculada")
    d = r.json()
    assert d["K"] == 45
    assert d["cobertura"] is not None and 0 <= d["cobertura"] <= 1
    # IC95 contiene la media
    assert d["ic95_inf"] <= d["costo_medio_M"] <= d["ic95_sup"]


def test_mc_comparacion(cliente_api):
    r = _o_o_skip(cliente_api, "/api/montecarlo/comparacion/45")
    if r.status_code == 503:
        pytest.skip("comparación K=45 no precalculada")
    d = r.json()
    assert d["k_actual"] == 58
    assert "dif_media_M" in d and "p_mejor" in d


def test_mc_mejor(cliente_api):
    r = _o_o_skip(cliente_api, "/api/montecarlo/mejor")
    d = r.json()
    if d["n_estrategias"] == 0:
        pytest.skip("sin estrategias aún")
    assert d["minima_factible"] is None or 35 <= d["minima_factible"] <= 70


# ---------------------------------------------------------------------------
# Geografía
# ---------------------------------------------------------------------------
def test_geo_actual(cliente_api):
    r = cliente_api.get("/api/geo/actual")
    if r.status_code == 503:
        pytest.skip("geo no precalculado")
    d = r.json()
    assert "territorios" in d and "puntos" in d
    if d["territorios"]:
        t = d["territorios"][0]
        assert set(t) == {"id", "n", "visitas", "lat", "lon"}


# ---------------------------------------------------------------------------
# Chatbot vía API
# ---------------------------------------------------------------------------
def test_chat_salud(cliente_api):
    r = cliente_api.get("/api/chat/salud")
    assert r.status_code == 200
    d = r.json()
    assert "configurado" in d and "modelo" in d
    # nunca filtra el valor real de la key
    from app.chatbot.orquestador import GroqProveedor
    key = GroqProveedor().api_key
    if key:
        assert key not in r.text


def test_chat_sin_key_503(cliente_api, llm_disponible):
    if llm_disponible:
        pytest.skip("LLM configurado; el chat sí responde")
    r = cliente_api.post("/api/chat", json={"pregunta": "hola", "sesion": "t"})
    assert r.status_code == 503
    assert "error" in r.json()


def test_chat_pregunta_vacia(cliente_api):
    r = cliente_api.post("/api/chat", json={"pregunta": "   "})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Privacidad global: ninguna respuesta expone columnas personales
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ep", [
    "/api/meta/universos", "/api/eda/resumen", "/api/eda/carga-zona?top=10",
    "/api/meta/params", "/api/ruteo/comparacion",
])
def test_no_pii_en_respuestas(cliente_api, ep):
    r = cliente_api.get(ep)
    assert r.status_code == 200
    texto = r.text
    for prohibida in ('"Identificacion"', '"Direccion"', '"id_cliente"'):
        assert prohibida not in texto, f"{ep} expone {prohibida}"
