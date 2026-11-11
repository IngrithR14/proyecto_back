"""Pruebas de las herramientas 10-11 (FASE 5): `detalle_asesor` y `ranking_asesores`.

Ambas leen SOLO `cache/equipos.json`; aquí se apunta `cfg.CACHE_DIR` a un
fixture pequeño (`tests/fixtures/equipos.json`) para no tocar la cache real,
el CSV ni `cargar_universo()`. La salida sólo lleva códigos (`zona`,
`equipo_codigo` EQ00..EQ06) y agregados: nunca nombres de supervisor ni PII.
"""
from __future__ import annotations

import builtins
import json
import re
import shutil
from pathlib import Path

import pytest

from app import config as cfg
from app.chatbot import herramientas as H
from app.chatbot.validador import COLUMNAS_PROHIBIDAS

FIXTURE = Path(__file__).parent / "fixtures" / "equipos.json"

CAMPOS_FILA = {"zona", "equipo_codigo", "n_clientes", "visitas_ruteadas",
               "visitas_programadas_mes", "dias_operados", "media_visitas_dia",
               "pico_visitas_dia", "sobre_cap", "en_objetivo_40_50",
               "n_rutas", "km"}


@pytest.fixture
def cache_equipos(tmp_path, monkeypatch):
    """CACHE_DIR temporal con el fixture de equipos.json (sin tocar cache/ real)."""
    shutil.copy(FIXTURE, tmp_path / "equipos.json")
    monkeypatch.setattr(cfg, "CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def cache_vacia(tmp_path, monkeypatch):
    """CACHE_DIR temporal SIN equipos.json (simula precompute pendiente)."""
    monkeypatch.setattr(cfg, "CACHE_DIR", str(tmp_path))
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Registro de las herramientas
# ---------------------------------------------------------------------------
def test_herramientas_registradas():
    assert {"detalle_asesor", "ranking_asesores"} <= set(H.HERRAMIENTAS)
    assert len(H.HERRAMIENTAS) == 11
    schemas = {s["function"]["name"]: s for s in H.schemas_tool()}
    assert {"detalle_asesor", "ranking_asesores"} <= set(schemas)
    ordenes = schemas["ranking_asesores"]["function"]["parameters"]["properties"]
    assert ordenes["orden"]["enum"] == ["visitas", "visitas_ruteadas", "rutas",
                                        "km", "pico"]


def test_fixture_respeta_contrato():
    """El contrato de cache/equipos.json no admite columnas con nombres/PII."""
    datos = json.loads(FIXTURE.read_text(encoding="utf-8"))
    filas = datos["asesores"]
    assert len(filas) == 5
    for f in filas:
        assert set(f) == CAMPOS_FILA, f"columnas inesperadas en el fixture: {set(f)}"


# ---------------------------------------------------------------------------
# detalle_asesor
# ---------------------------------------------------------------------------
def test_detalle_por_zona(cache_equipos):
    r = H.ejecutar("detalle_asesor", {"zona": "z01"})   # sin distinguir mayúsculas
    assert "error" not in r, r
    assert r["ok"] is True and r["tipo"] == "zona"
    assert r["zona"] == "Z01" and r["equipo_codigo"] == "EQ01"
    assert r["n_clientes"] == 300
    assert r["visitas_ruteadas"] == 812
    assert r["visitas_programadas_mes"] == 870
    assert r["dias_operados"] == 20
    assert r["media_visitas_dia"] == 43.5
    assert r["pico_visitas_dia"] == 52
    assert r["sobre_cap"] is True and r["en_objetivo_40_50"] is True
    # los 10 campos pedidos (más ok/tipo) están en la salida
    assert CAMPOS_FILA - {"n_rutas", "km"} <= set(r)


def test_detalle_por_equipo(cache_equipos):
    r = H.ejecutar("detalle_asesor", {"equipo": "eq02"})
    assert "error" not in r, r
    assert r["tipo"] == "equipo" and r["equipo_codigo"] == "EQ02"
    assert r["n_asesores"] == 2
    assert {a["zona"] for a in r["asesores"]} == {"Z03", "Z04"}
    assert all(a["equipo_codigo"] == "EQ02" for a in r["asesores"])


def test_detalle_equipo_sin_zonas(cache_equipos):
    r = H.ejecutar("detalle_asesor", {"equipo": "EQ09"})
    assert "error" not in r, r
    assert r["ok"] is True and r["n_asesores"] == 0 and r["asesores"] == []
    assert "nota" in r


def test_detalle_no_encontrado(cache_equipos):
    r = H.ejecutar("detalle_asesor", {"zona": "NOEXISTE"})
    assert r.get("ok") is False
    assert r["error_codigo"] == "ASESOR_NO_ENCONTRADO"
    assert r["equipo_codigo_validos"] == ["EQ01", "EQ02", "EQ03"]


def test_detalle_sin_parametros(cache_equipos):
    r = H.ejecutar("detalle_asesor", {})
    assert "error" in r and "zona" in r["error"] and "equipo" in r["error"]


# ---------------------------------------------------------------------------
# ranking_asesores
# ---------------------------------------------------------------------------
def test_ranking_visitas_mayor(cache_equipos):
    r = H.ejecutar("ranking_asesores",
                   {"orden": "visitas", "sentido": "mayor", "top": 3})
    assert "error" not in r, r
    assert r["n_asesores"] == 5 and r["top"] == 3 and r["orden"] == "visitas"
    # `visitas` = plan del mes completo: la columna exacta queda en `ordenado_por`
    assert r["ordenado_por"] == "visitas_programadas_mes"
    assert [f["zona"] for f in r["filas"]] == ["Z03", "Z01", "Z02"]
    assert r["filas"][0]["visitas_programadas_mes"] == 930
    assert set(r["filas"][0]) == {"zona", "equipo_codigo", "visitas_programadas_mes",
                                  "visitas_ruteadas", "n_rutas", "km", "pico",
                                  "sobre_cap"}


def test_ranking_visitas_ruteadas(cache_equipos):
    """`orden=visitas_ruteadas` ordena por lo que quedó en las rutas del 1er ciclo."""
    r = H.ejecutar("ranking_asesores", {"orden": "visitas_ruteadas"})
    assert "error" not in r, r
    assert r["ordenado_por"] == "visitas_ruteadas"
    assert [f["zona"] for f in r["filas"]] == ["Z03", "Z01", "Z02", "Z05", "Z04"]
    assert r["filas"][0]["visitas_ruteadas"] == 871
    # las dos columnas de visitas viajan separadas y con valores distintos
    assert r["filas"][0]["visitas_programadas_mes"] == 930
    assert r["filas"][0]["visitas_programadas_mes"] != r["filas"][0]["visitas_ruteadas"]


def test_ranking_pico_menor(cache_equipos):
    r = H.ejecutar("ranking_asesores", {"orden": "pico", "sentido": "menor"})
    assert "error" not in r, r
    assert [f["zona"] for f in r["filas"]][:2] == ["Z04", "Z02"]
    assert r["filas"][0]["pico"] == 43


def test_ranking_rutas_y_km(cache_equipos):
    por_rutas = H.ejecutar("ranking_asesores", {"orden": "rutas"})
    assert por_rutas["ordenado_por"] == "n_rutas"
    assert por_rutas["filas"][0]["zona"] == "Z03"
    assert por_rutas["filas"][0]["n_rutas"] == 81
    por_km = H.ejecutar("ranking_asesores", {"orden": "km"})
    assert por_km["ordenado_por"] == "km"
    assert por_km["filas"][0]["zona"] == "Z03"
    assert por_km["filas"][0]["km"] == 733.8


def test_ranking_top_limitado_a_20(cache_equipos):
    r = H.ejecutar("ranking_asesores", {"top": 999})
    assert r["top"] == 20
    assert len(r["filas"]) == 5 == r["n_asesores"]


def test_ranking_orden_invalido(cache_equipos):
    r = H.ejecutar("ranking_asesores", {"orden": "costo"})
    assert "error" in r
    assert r["ordenes_validos"] == ["km", "pico", "rutas", "visitas",
                                    "visitas_ruteadas"]


def test_ranking_sentido_invalido(cache_equipos):
    r = H.ejecutar("ranking_asesores", {"sentido": "al azar"})
    assert "error" in r and r["sentidos_validos"] == ["mayor", "menor"]


def test_ranking_top_no_numerico(cache_equipos):
    r = H.ejecutar("ranking_asesores", {"top": "diez"})
    assert "error" not in r, r
    assert r["top"] == 10


# ---------------------------------------------------------------------------
# Privacidad y límites de arquitectura
# ---------------------------------------------------------------------------
def test_salidas_solo_codigos_y_agregados(cache_equipos):
    """Ni columnas PII ni valores que no sean códigos (zona/EQ) o agregados."""
    salidas = [H.ejecutar("detalle_asesor", {"zona": "Z01"}),
               H.ejecutar("detalle_asesor", {"equipo": "EQ01"}),
               H.ejecutar("ranking_asesores", {})]
    codigos: list[str] = []

    def _recorre(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("zona", "equipo_codigo") and isinstance(v, str) and v:
                    codigos.append(v)
                _recorre(v)
        elif isinstance(o, list):
            for v in o:
                _recorre(v)

    for s in salidas:
        assert "error" not in s, s
        for prohibida in COLUMNAS_PROHIBIDAS:
            assert prohibida not in json.dumps(s, ensure_ascii=False)
        _recorre(s)
    assert codigos
    assert all(re.fullmatch(r"(EQ\d\d|Z\d\d)", c) for c in codigos), codigos


def test_no_abre_csv_ni_universo(cache_equipos, monkeypatch):
    """En runtime no se abre el CSV ni se llama `cargar_universo()`."""
    import app.core.universo as universo

    abiertas: list[str] = []
    real_open = builtins.open

    def _espia(ruta, *args, **kwargs):
        abiertas.append(str(ruta))
        return real_open(ruta, *args, **kwargs)

    def _prohibido(*args, **kwargs):
        raise AssertionError("cargar_universo() no debe llamarse en runtime")

    monkeypatch.setattr(builtins, "open", _espia)
    monkeypatch.setattr(universo, "cargar_universo", _prohibido)
    assert "error" not in H.ejecutar("detalle_asesor", {"zona": "Z01"})
    assert "error" not in H.ejecutar("ranking_asesores", {})
    assert not any(p.endswith(".csv") or "/data/" in p for p in abiertas), abiertas


def test_sin_cache_devuelve_error_limpio(cache_vacia):
    """Sin cache/equipos.json: mensaje limpio, sin rutas de archivo ni trazas."""
    assert not (Path(cache_vacia) / "equipos.json").exists()
    esperado = {"error": "Datos de asesores no disponibles (cache no generada)"}
    for nombre, entrada in (("detalle_asesor", {"zona": "Z01"}),
                            ("detalle_asesor", {"equipo": "EQ01"}),
                            ("ranking_asesores", {}),
                            ("ranking_asesores", {"orden": "visitas_ruteadas"})):
        r = H.ejecutar(nombre, entrada)
        assert r == esperado, (nombre, entrada, r)
        crudo = json.dumps(r, ensure_ascii=False)
        assert ".json" not in crudo and "/" not in crudo and "\\" not in crudo
        assert "Traceback" not in crudo and "Exception" not in crudo
        assert "precompute" not in crudo and "cache/" not in crudo
