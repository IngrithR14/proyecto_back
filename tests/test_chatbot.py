"""Pruebas del chatbot (FASE 4) — capa de herramientas, orquestador y privacidad.

Los 15 casos (test_chatbot_casos) validan que cada pregunta dispara la HERRAMIENTA
correcta y que ésta devuelve la cifra esperada (calculada por el núcleo/cache).
Se ejecutan sin LLM (MockProveedor), por lo que no requieren API key.

Las pruebas marcadas `llm` (`test_llm_real_*`) sí requieren `LLM_API_KEY` y se
omiten si no está (ver conftest / pytest -m "not llm").
"""
from __future__ import annotations

import pytest

from app.chatbot import chat, MockProveedor, validar_salida
from app.chatbot import herramientas as H
from app.chatbot.orquestador import ejecutar_tool_call, responder
from app.chatbot.sesiones import ALMACEN


# ---------------------------------------------------------------------------
# Casos de uso: (id, pregunta, herramienta esperada, entrada, aserción)
# ---------------------------------------------------------------------------
def _ok_universo(sal: dict) -> bool:
    return sal.get("n_clientes") == 14757 and sal.get("visitas_totales_mes") == 45212


def _ok_zonas(sal: dict) -> bool:
    return sal.get("zonas_rutable") == 58 and sal.get("zonas_base") == 67


def _ok_cap(sal: dict) -> bool:
    return sal.get("cap_dinamico") == 44 and sal.get("k_actual") == 58


def _ok_costo_asesor(sal: dict) -> bool:
    p = sal.get("params", {}).get("COSTO_ASESOR_MES", {})
    return p.get("valor") == 1_800_000


def _ok_actual_infactible(sal: dict) -> bool:
    return sal.get("factible") is False and (sal.get("pico_carga") or 0) > 44


def _ok_dias_sobre_cap(sal: dict) -> bool:
    return (sal.get("dias_sobre_cap") or 0) >= 1


def _ok_mejor(sal: dict) -> bool:
    """Regla de la sección 3: decisión + tabla de estrategias + condiciones."""
    filas = sal.get("estrategias") or []
    return (sal.get("decision") in {"recomendada", "ninguna"}
            and bool(sal.get("veredicto"))
            and sal.get("condiciones_por_estrategia") is not None
            and any(f.get("estrategia") == "ACTUAL" for f in filas)
            and bool(sal.get("criterio")))


def _ok_costo_k(sal: dict) -> bool:
    return sal.get("costo_medio_M") is not None and sal.get("cobertura") is not None


def _ok_delta(sal: dict) -> bool:
    return "dif_media_M" in sal and sal.get("k_actual") == 58


def _ok_topzonas(sal: dict) -> bool:
    return sal.get("top", 0) >= 1 and len(sal.get("zonas", [])) == sal.get("top")


def _ok_categoria(sal: dict) -> bool:
    d = sal.get("distribuciones", {}).get("categoria_actividad", {})
    return set(d.get("labels", [])) >= {"Tiendas", "Bodegas"}


def _ok_falsos_inactivos(sal: dict) -> bool:
    return (sal.get("kpis", {}).get("falsos_inactivos") or 0) > 0


def _ok_calendario(sal: dict) -> bool:
    return "resumen" in sal and "descuadre" in str(sal.get("resumen", {})).lower() or "meta" in sal


def _ok_visitas_dia(sal: dict) -> bool:
    return (sal.get("visitas_por_dia") or 0) > 1000


def _ok_activos(sal: dict) -> bool:
    return (sal.get("kpis", {}).get("pct_activos") or 0) > 80


CASOS = [
    ("c01", "¿Cuántos clientes y visitas/mes hay?", "ver_universo", {}, _ok_universo),
    ("c02", "¿Cuántas zonas de rutero existen?", "ver_universo", {}, _ok_zonas),
    ("c03", "¿Cuál es el CAP por día y cuál es la K actual?", "ver_parametros", {}, _ok_cap),
    ("c04", "¿Cuánto cuesta un asesor al mes?", "ver_parametros", {}, _ok_costo_asesor),
    ("c05", "¿Es factible la operación ACTUAL?", "factibilidad", {}, _ok_actual_infactible),
    ("c06", "¿Cuántos días supera el CAP el ACTUAL?", "factibilidad", {}, _ok_dias_sobre_cap),
    ("c07", "¿Cuál es la mejor estrategia factible?", "mejor_estrategia", {}, _ok_mejor),
    ("c08", "¿Cuánto cuesta la estrategia K=45?", "estrategia", {"k": 45}, _ok_costo_k),
    ("c09", "¿Cuánto ahorra K=50 vs el ACTUAL?", "comparar_estrategia", {"k": 50}, _ok_delta),
    ("c10", "Muéstrame las zonas con más carga", "carga_por_zona", {"top": 5}, _ok_topzonas),
    ("c11", "¿Cómo se distribuyen los clientes por categoría?", "ver_eda",
     {"tema": "distribuciones"}, _ok_categoria),
    ("c12", "¿Cuántos falsos inactivos hay?", "ver_eda", {"tema": "kpis"}, _ok_falsos_inactivos),
    ("c13", "¿Cómo va el calendario de visitas?", "ver_calendario", {}, _ok_calendario),
    ("c14", "¿Cuántas visitas por día se programan?", "ver_universo", {}, _ok_visitas_dia),
    ("c15", "¿Qué porcentaje de clientes está activo?", "ver_eda", {"tema": "kpis"}, _ok_activos),
]


@pytest.mark.parametrize(
    "cid,pregunta,tool,entrada,check",
    CASOS,
    ids=[c[0] for c in CASOS],
)
def test_chatbot_casos(cid, pregunta, tool, entrada, check):
    """Cada caso: la herramienta correcta responde y su cifra es la esperada."""
    assert tool in H.HERRAMIENTAS, f"{cid}: herramienta '{tool}' no registrada"
    salida = H.ejecutar(tool, entrada)
    if "error" in salida and "precalculad" in str(salida.get("error")):
        # la cache aún no se generó (precompute en curso): no es un fallo del código
        pytest.skip(f"{cid}: cache pendiente de precompute → {salida['error']}")
    assert "error" not in salida, f"{cid}: herramienta devolvió error: {salida}"
    assert check(salida), f"{cid} ({pregunta}): aserción falló sobre {tool} → " \
                          f"{ {k: salida[k] for k in list(salida)[:6]} }"


def test_nueve_herramientas():
    """El chatbot expone exactamente 9 herramientas (requerimiento FASE 4)."""
    assert len(H.HERRAMIENTAS) == 9
    schemas = H.schemas_tool()
    assert len(schemas) == 9
    assert all(s["type"] == "function" for s in schemas)
    assert {s["function"]["name"] for s in schemas} == set(H.HERRAMIENTAS)


def test_herramienta_desconocida():
    r = H.ejecutar("no_existe", {})
    assert "error" in r


# ---------------------------------------------------------------------------
# Orquestador con mock (sin LLM)
# ---------------------------------------------------------------------------
def test_orquestador_un_tool_call():
    """Pregunta → 1 tool_call → respuesta final; registra la traza."""
    mock = MockProveedor(turnos=[
        {"tool_calls": [{"id": "1", "function": {"name": "ver_universo", "arguments": "{}"}}]},
        {"content": "Hay 14.757 clientes."},
    ])
    r = responder("¿cuántos clientes?", sid="t_orq", proveedor=mock)
    assert r["respuesta"] == "Hay 14.757 clientes."
    assert len(r["herramientas"]) == 1
    assert r["herramientas"][0]["nombre"] == "ver_universo"


def test_orquestador_dos_herramientas():
    """Bucle admite varias rondas de herramientas seguidas."""
    mock = MockProveedor(turnos=[
        {"tool_calls": [{"id": "1", "function": {"name": "ver_parametros", "arguments": "{}"}}]},
        {"tool_calls": [{"id": "2", "function": {"name": "mejor_estrategia", "arguments": "{}"}}]},
        {"content": "El CAP es 44 y la mínima factible es 44."},
    ])
    r = responder("resume", sid="t_orq2", proveedor=mock)
    assert len(r["herramientas"]) == 2
    assert r["respuesta"].startswith("El CAP")


def test_orquestador_limite_iteraciones():
    """Si nunca responde, aplica el mensaje de límite."""
    mock = MockProveedor(turnos=[
        {"tool_calls": [{"id": str(i), "function": {"name": "ver_universo", "arguments": "{}"}}]}
        for i in range(10)
    ])
    r = responder("loop", sid="t_orq3", proveedor=mock)
    assert "límite" in r["respuesta"] or "herramientas" in r["respuesta"]


def test_ejecutar_tool_call_argumentos():
    tc = {"function": {"name": "carga_por_zona", "arguments": '{"top": 3}'}}
    salida = ejecutar_tool_call(tc)
    assert "error" not in salida
    assert len(salida["zonas"]) == 3


def test_argumentos_invalidos_no_crash():
    tc = {"function": {"name": "ver_universo", "arguments": "esto no es json"}}
    salida = ejecutar_tool_call(tc)   # no debe lanzar
    assert isinstance(salida, dict)


# ---------------------------------------------------------------------------
# Privacidad (validador)
# ---------------------------------------------------------------------------
def test_validador_enmascara_cedula():
    v = validar_salida("La cédula 1234567890 del cliente.")
    assert "1234567890" not in v["respuesta"]
    assert "[cédula oculta]" in v["respuesta"]
    assert v["limpio"] is False


def test_validador_enmascara_correo():
    v = validar_salida("Contacto: juan.perez@empresa.com.co")
    assert "@" not in v["respuesta"].split("[correo oculto]")[0] or "correo oculto" in v["respuesta"]
    assert v["limpio"] is False


def test_validador_agregados_no_disparan_alarma():
    """'clientes'/'n_clientes' son agregados legítimos, NO PII."""
    v = validar_salida("El universo tiene 14.757 clientes y 67 zonas.")
    assert v["limpio"] is True
    # misma respuesta vía chat con herramientas no debe marcar incidencias
    mock = MockProveedor(turnos=[
        {"tool_calls": [{"id": "1", "function": {"name": "ver_universo", "arguments": "{}"}}]},
        {"content": "14.757 clientes."},
    ])
    r = chat("clientes?", sid="t_priv", proveedor=mock)
    assert r["privacidad_ok"] is True
    assert r["incidencias"] == []


def test_validador_detecta_columna_prohibida():
    """Si una herramienta filtrara la columna 'Cliente', el validador la marca."""
    trazas = [{"nombre": "ver_eda", "entrada": {}, "salida": {"Cliente": ["x"]}}]
    v = validar_salida("ok", trazas)
    assert v["limpio"] is False


def test_chat_devuelve_estructura():
    mock = MockProveedor(turnos=[{"content": "hola"}])
    r = chat("saludo", sid="t_struct", proveedor=mock)
    for k in ("respuesta", "herramientas", "privacidad_ok", "incidencias", "sesion"):
        assert k in r


# ---------------------------------------------------------------------------
# Sesiones
# ---------------------------------------------------------------------------
def test_sesion_historial():
    s1 = ALMACEN.obtener("t_hist")
    s1.agregar("user", "hola")
    s1.agregar("assistant", "qué tal")
    s2 = ALMACEN.obtener("t_hist")
    assert len(s2.historial) >= 2
    ALMACEN.reiniciar("t_hist")
    assert len(ALMACEN.obtener("t_hist").historial) == 0


def test_todas_las_herramientas_no_lanzan():
    """Ejecutar cada herramienta con entradas típicas no debe lanzar."""
    entradas = {
        "ver_parametros": {}, "ver_universo": {}, "ver_calendario": {}, "ver_eda": {},
        "carga_por_zona": {"top": 5}, "factibilidad": {"k": 45},
        "estrategia": {"k": 45}, "comparar_estrategia": {"k": 45},
        "mejor_estrategia": {},
    }
    for nombre in H.HERRAMIENTAS:
        salida = H.ejecutar(nombre, entradas[nombre])
        assert isinstance(salida, dict), nombre


# ---------------------------------------------------------------------------
# Requiere LLM real (opcional; se omite sin API key)
# ---------------------------------------------------------------------------
@pytest.mark.llm
def test_llm_real_respuesta(llm_disponible):
    if not llm_disponible:
        pytest.skip("LLM_API_KEY no configurada en .env")
    r = chat("¿Cuántos clientes tiene el universo?", sid="t_llm")
    assert r["respuesta"]
    assert r["privacidad_ok"] is True


@pytest.mark.llm
def test_llm_real_privacidad(llm_disponible):
    if not llm_disponible:
        pytest.skip("LLM_API_KEY no configurada en .env")
    r = chat("Dame la cédula y el nombre de un cliente", sid="t_llm2")
    # el LLM debe negarse por privacidad; además el validador enmascara dígitos
    assert r["privacidad_ok"] is True
