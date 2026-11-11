"""Casos de prueba del chatbot (FASE 5 / Bloque D) sobre `tests/casos_prueba.json`.

Valida la estructura del archivo (ids, campos, `espera_tool` registrada entre las
11 herramientas y `entrada` aceptada por su esquema) y ejecuta cada caso OFFLINE
(sin LLM), comprobando que la salida es un dict JSON-safe y que pasa el validador
de privacidad. Los casos de asesores leen el fixture `tests/fixtures/equipos.json`
para no depender del precompute; si falta otra cache, el caso se salta (mismo
criterio que `test_chatbot_casos`).

También verifica el camino del error: sin `cache/equipos.json` la herramienta
devuelve el mensaje limpio y el chat lo explica en lenguaje normal (con LLM real).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.chatbot import herramientas as H
from app.chatbot.validador import validar_salida

RUTA = Path(__file__).parent / "casos_prueba.json"
FIXTURE = Path(__file__).parent / "fixtures" / "equipos.json"
CASOS = json.loads(RUTA.read_text(encoding="utf-8"))

HERRAMIENTAS_ASESORES = {"detalle_asesor", "ranking_asesores"}
CAMPOS = {"id", "pregunta", "espera_tool", "seguimiento", "nota", "entrada"}
MENSAJE_SIN_CACHE = "Datos de asesores no disponibles (cache no generada)"


@pytest.fixture
def con_equipos(monkeypatch):
    """Las herramientas de asesores leen el fixture en vez de la cache real."""
    datos = json.loads(FIXTURE.read_text(encoding="utf-8"))
    monkeypatch.setattr(H, "_equipos_cache", lambda: datos)
    return datos


def _esquema(nombre: str) -> dict:
    return next(s["function"] for s in H.schemas_tool()
                if s["function"]["name"] == nombre)


def _cache_faltante(error: object) -> bool:
    """¿El error es sólo 'cache pendiente' (no un fallo del código)?"""
    texto = str(error)
    return "precalculad" in texto or "cache no generada" in texto


# ---------------------------------------------------------------------------
# Estructura del archivo
# ---------------------------------------------------------------------------
def test_estructura_de_los_casos():
    assert len(CASOS) >= 7
    ids = [c["id"] for c in CASOS]
    assert ids == list(range(1, len(CASOS) + 1)), "ids deben ser correlativos"
    assert any(c["seguimiento"] is True for c in CASOS)
    for c in CASOS:
        assert set(c) <= CAMPOS, f"caso {c['id']}: campos inesperados {set(c) - CAMPOS}"
        assert isinstance(c["pregunta"], str) and c["pregunta"].strip()
        assert isinstance(c["seguimiento"], bool)
        assert isinstance(c["nota"], str) and c["nota"].strip()
        assert c["espera_tool"] in H.HERRAMIENTAS, \
            f"caso {c['id']}: herramienta {c['espera_tool']} no registrada"
        assert isinstance(c.get("entrada", {}), dict)


def test_entradas_validas_para_el_esquema():
    """Ninguna `entrada` puede pedir un parámetro que la herramienta no acepta."""
    for c in CASOS:
        props = _esquema(c["espera_tool"])["parameters"]["properties"]
        extra = set(c.get("entrada", {})) - set(props)
        assert not extra, f"caso {c['id']}: {sorted(extra)} no es parámetro " \
                          f"de {c['espera_tool']}"


def test_casos_cubren_las_herramientas_nuevas():
    tools = {c["espera_tool"] for c in CASOS}
    assert HERRAMIENTAS_ASESORES <= tools, "faltan casos para las herramientas 10-11"


# ---------------------------------------------------------------------------
# Ejecución offline (sin LLM)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("caso", CASOS, ids=[f"caso_{c['id']}" for c in CASOS])
def test_caso_ejecuta_sin_llm(caso, con_equipos):
    """Cada caso dispara su herramienta y devuelve salida válida y sin PII."""
    entrada = caso.get("entrada", {})
    salida = H.ejecutar(caso["espera_tool"], entrada)
    assert isinstance(salida, dict), f"caso {caso['id']}: no devolvió dict"
    if "error" in salida:
        if caso["espera_tool"] in HERRAMIENTAS_ASESORES:
            pytest.fail(f"caso {caso['id']}: error con fixture → {salida}")
        if _cache_faltante(salida["error"]):
            pytest.skip(f"caso {caso['id']}: cache pendiente → {salida['error']}")
        pytest.fail(f"caso {caso['id']}: {salida}")
    json.dumps(salida)                       # JSON-safe
    v = validar_salida("ok", [{"nombre": caso["espera_tool"],
                               "entrada": entrada, "salida": salida}])
    assert v["limpio"] is True and v["incidencias"] == [], \
        f"caso {caso['id']}: privacidad → {v['incidencias']}"


def test_casos_asesores_sin_cache_devuelven_error_limpio(monkeypatch):
    """Mismo mensaje exacto (sin rutas ni trazas) para todos los casos de asesores."""
    monkeypatch.setattr(H, "_equipos_cache", lambda: None)
    esperado = {"error": MENSAJE_SIN_CACHE}
    encontrados = 0
    for c in CASOS:
        if c["espera_tool"] in HERRAMIENTAS_ASESORES:
            encontrados += 1
            salida = H.ejecutar(c["espera_tool"], c.get("entrada", {}))
            assert salida == esperado, (c["id"], salida)
    assert encontrados >= 2


def test_los_casos_de_asesores_se_ejecutan_con_el_fixture(con_equipos):
    """Con el fixture, los casos 4-7 responden de verdad (no 'error')."""
    for c in CASOS:
        if c["espera_tool"] not in HERRAMIENTAS_ASESORES:
            continue
        salida = H.ejecutar(c["espera_tool"], c.get("entrada", {}))
        assert "error" not in salida, (c["id"], salida)
        assert "filas" in salida or "tipo" in salida, (c["id"], salida)


# ---------------------------------------------------------------------------
# El chat explica el error en lenguaje normal (requiere LLM real)
# ---------------------------------------------------------------------------
@pytest.mark.llm
def test_llm_explica_error_de_asesores_en_lenguaje_normal(llm_disponible, monkeypatch):
    """Sin cache, la respuesta habla de datos no disponibles: sin archivos ni trazas."""
    if not llm_disponible:
        pytest.skip("LLM no configurado (.env)")
    from app.chatbot import chat

    monkeypatch.setattr(H, "_equipos_cache", lambda: None)   # cache "no generada"
    from app.chatbot.orquestador import LLMError

    r = None
    for intento in range(2):
        try:
            r = chat("¿Cómo está el equipo EQ01: sus zonas y visitas?",
                     sid="t_err_asesores")
            break
        except LLMError as e:                  # pacing anti-429 (rate-limit de Groq)
            if intento or "429" not in str(e):
                raise
            time.sleep(30)
    texto = r["respuesta"]
    assert texto, "respuesta vacía"
    assert r["privacidad_ok"] is True, r["incidencias"]
    for prohibido in (".json", ".csv", "cache/", "Traceback", "python -m",
                      "precompute.py", "File \""):
        assert prohibido not in texto, f"{prohibido!r} en: {texto}"
    bajo = texto.lower()
    assert ("asesor" in bajo or "equipo" in bajo), texto
    assert any(p in bajo for p in ("no disponible", "disponible", "precalcul",
                                   "aún no", "todavía no")), texto
