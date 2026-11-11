"""Pruebas del prompt del sistema y del validador de privacidad (FASE 5 / Bloque C).

Cubre: el prompt lista las 11 herramientas, explica `visitas_programadas_mes`
vs `visitas_ruteadas`, traduce el error limpio de asesores a lenguaje normal sin
rutas de archivo, y no contiene datos personales. El validador enmascara nombres
de supervisor (E10) sin disparar alarma con códigos `EQxx`/`ZON_REP` ni con
agregados legítimos.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app import config as cfg
from app.chatbot import MockProveedor, chat, herramientas as H
from app.chatbot.prompts import SYSTEM_PROMPT
from app.chatbot.validador import (COLUMNAS_PROHIBIDAS, _enmascara_nombres,
                                   _escanea_texto, _patrones_supervisor,
                                   validar_salida)

FIXTURE = Path(__file__).parent / "fixtures" / "equipos.json"


@pytest.fixture
def cache_equipos(tmp_path, monkeypatch):
    """CACHE_DIR temporal con el fixture de equipos.json (sin tocar cache/ real)."""
    shutil.copy(FIXTURE, tmp_path / "equipos.json")
    monkeypatch.setattr(cfg, "CACHE_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Prompt del sistema
# ---------------------------------------------------------------------------
def test_prompt_lista_las_11_herramientas():
    """El prompt no se queda corto: nombra todas las herramientas registradas."""
    assert len(H.HERRAMIENTAS) == 11
    for nombre in H.HERRAMIENTAS:
        assert nombre in SYSTEM_PROMPT, f"el prompt no menciona {nombre}"


def test_prompt_explica_las_dos_visitas():
    """Distingue 'programadas' (plan del mes) de 'ruteadas' (1er ciclo)."""
    assert "visitas_programadas_mes" in SYSTEM_PROMPT
    assert "visitas_ruteadas" in SYSTEM_PROMPT
    assert "plan del mes completo" in SYSTEM_PROMPT
    assert "rutas del primer ciclo" in SYSTEM_PROMPT
    assert "ordenado_por" in SYSTEM_PROMPT


def test_prompt_explica_el_error_limpio_de_asesores():
    """Da el mensaje exacto, cómo decirlo en lenguaje normal y qué NO mencionar."""
    assert "Datos de asesores no disponibles (cache no generada)" in SYSTEM_PROMPT
    assert "precalculado" in SYSTEM_PROMPT          # traducción al usuario
    assert "NUNCA menciones rutas ni nombres de archivo" in SYSTEM_PROMPT


def test_prompt_pide_codigos_y_no_nombres():
    """Regla de privacidad de equipos presente (sólo EQxx, jamás nombres)."""
    assert "EQ01" in SYSTEM_PROMPT and "EQ06" in SYSTEM_PROMPT
    assert "NUNCA reveles nombres de supervisor" in SYSTEM_PROMPT


def test_prompt_sin_datos_personales():
    """El prompt del sistema no lleva PII: ni dígitos largos ni nombres (E10)."""
    assert _escanea_texto(SYSTEM_PROMPT) == SYSTEM_PROMPT, "PII numérico en el prompt"
    assert _enmascara_nombres(SYSTEM_PROMPT) == SYSTEM_PROMPT, "nombre en el prompt"


def test_prompt_sin_rutas_de_archivo_reales():
    """Sólo aparece la prohibición genérica; no rutas ni comandos concretos."""
    assert "cache/" not in SYSTEM_PROMPT
    assert "app/precompute.py" not in SYSTEM_PROMPT
    assert "python -m" not in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Validador: nombres de supervisor (E10)
# ---------------------------------------------------------------------------
def test_enmascara_nombre_completo():
    v = validar_salida("El equipo EQ01 lo coordina Sirley Damaris Rodríguez Usaquén.")
    assert v["respuesta"] == "El equipo EQ01 lo coordina [nombre oculto]."
    assert v["limpio"] is False
    assert any("nombre de supervisor" in i for i in v["incidencias"])


def test_enmascara_nombre_sin_tildes_y_con_mayusculas():
    """'sirley rodriguez' y 'SIRLEY RODRIGUEZ' también se detectan."""
    for texto in ("sirley rodriguez esta en EQ02", "SIRLEY RODRIGUEZ esta en EQ02"):
        v = validar_salida(texto)
        assert "[nombre oculto]" in v["respuesta"], texto
        assert v["limpio"] is False


def test_codigos_y_agregados_no_disparan_alarma():
    """Códigos EQxx/ZON_REP, 'SIN_ASESOR' y agregados NO son PII."""
    limpia = ("El universo tiene 14.757 clientes y 67 zonas; el equipo EQ01 "
              "cubre Z01 y Z02, y EQ00 = SIN_ASESOR queda para clientes sin asesor.")
    v = validar_salida(limpia)
    assert v["respuesta"] == limpia
    assert v["limpio"] is True and v["incidencias"] == []


def test_frase_comun_sin_asesor_no_se_enmascara():
    """La entrada SIN_ASESOR no es persona: 'sin asesor' no debe cambiar."""
    texto = "Quedan 12 clientes sin asesor en la zona Z05."
    assert validar_salida(texto)["respuesta"] == texto


def test_patrones_de_nombres_cargados():
    patrones = _patrones_supervisor()
    assert len(patrones) >= 20
    assert all(hasattr(p, "search") for p in patrones)


# ---------------------------------------------------------------------------
# Validador: columnas prohibidas y salidas de herramientas
# ---------------------------------------------------------------------------
def test_columna_supervisor_prohibida():
    """Si una herramienta filtrara la columna `Supervisor`, se marca la incidencia."""
    trazas = [{"nombre": "ver_eda", "entrada": {}, "salida": {"Supervisor": ["x"]}}]
    v = validar_salida("ok", trazas)
    assert v["limpio"] is False
    assert any("Supervisor" in i for i in v["incidencias"])
    assert set(COLUMNAS_PROHIBIDAS) >= {"Identificacion", "Cliente", "Direccion",
                                        "Supervisor"}


def test_salida_de_herramienta_con_nombre_se_marca():
    trazas = [{"nombre": "detalle_asesor", "entrada": {},
               "salida": {"tipo": "zona", "zona": "Z01", "coordinador": "Edwin Amaya"}}]
    v = validar_salida("ok", trazas)
    assert v["limpio"] is False
    assert any("nombre de supervisor" in i for i in v["incidencias"])


def test_herramientas_nuevas_pasan_el_validador(cache_equipos):
    """Las salidas reales de las herramientas 10-11 no disparan alarma."""
    trazas = []
    for nombre, entrada in (("detalle_asesor", {"zona": "Z01"}),
                            ("detalle_asesor", {"equipo": "EQ01"}),
                            ("ranking_asesores", {"orden": "visitas_ruteadas"})):
        salida = H.ejecutar(nombre, entrada)
        assert "error" not in salida, (nombre, salida)
        trazas.append({"nombre": nombre, "entrada": entrada, "salida": salida})
    v = validar_salida("ok", trazas)
    assert v["limpio"] is True and v["incidencias"] == [], v["incidencias"]


# ---------------------------------------------------------------------------
# Integración: chat con mock
# ---------------------------------------------------------------------------
def test_chat_enmascara_nombre_de_supervisor():
    mock = MockProveedor(turnos=[{"content": "El equipo EQ01 lo coordina Edwin Amaya."}])
    r = chat("¿quién coordina el equipo EQ01?", sid="t_nom", proveedor=mock)
    assert "[nombre oculto]" in r["respuesta"]
    assert "Edwin Amaya" not in r["respuesta"]
    assert r["privacidad_ok"] is False
    assert any("nombre de supervisor" in i for i in r["incidencias"])


def test_chat_respuesta_limpia_con_codigos():
    mock = MockProveedor(turnos=[{"content": "El equipo EQ01 tiene 2 zonas: Z01 y Z02."}])
    r = chat("¿cuántas zonas tiene EQ01?", sid="t_cod", proveedor=mock)
    assert r["privacidad_ok"] is True and r["incidencias"] == []
