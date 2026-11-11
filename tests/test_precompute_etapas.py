"""Pruebas del precompute modular (FASE 5 / Bloque E).

Cubre `--todo` y `--solo <etapa>` (con sus dependencias de cache), el orden de
dependencias documentado en el docstring, y la etapa NUEVA `equipos` que genera
`cache/equipos.json` con el contrato exacto que leen las herramientas 10-11.

Ninguna prueba toca la cache real ni el CSV: todas las etapas pesadas se
sustituyen por dobles y `cfg.CACHE_DIR` se apunta a `tmp_path`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from app import config as cfg
from app import precompute as P
from app.chatbot import herramientas as H
from app.chatbot.validador import _enmascara_nombres

ETAPAS = ["general", "mc", "geo", "seleccion", "pliegues", "evaluacion",
          "comparacion", "rutas", "equipos"]

# nombre de etapa -> atributo de la función en app.precompute
FUNCS = {"general": "precompute_general", "mc": "precompute_mc",
         "geo": "precompute_geo", "seleccion": "precompute_seleccion",
         "pliegues": "precompute_pliegues", "evaluacion": "precompute_evaluacion",
         "comparacion": "precompute_comparacion_rutas",
         "rutas": "precompute_rutas", "equipos": "precompute_equipos"}


@pytest.fixture
def cache_tmp(tmp_path, monkeypatch):
    """CACHE_DIR temporal: nada escribe en cache/ real."""
    monkeypatch.setattr(cfg, "CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def plan_falso(cache_tmp, monkeypatch):
    """Dobles para TODAS las etapas: devuelve {etapa: (args, kwargs)} en orden."""
    llamadas: dict[str, tuple] = {}

    def _doble(nombre: str):
        def fn(*a, **k):
            llamadas[nombre] = (a, k)
        return fn

    for nombre, attr in FUNCS.items():
        monkeypatch.setattr(P, attr, _doble(nombre))
    return llamadas


def _nombres(llamadas: dict) -> list[str]:
    return list(llamadas)


# ---------------------------------------------------------------------------
# --todo / orden de dependencias
# ---------------------------------------------------------------------------
def test_todo_ejecuta_las_nueve_etapas_en_orden(plan_falso):
    P.main(["--todo"])
    assert _nombres(plan_falso) == ETAPAS


def test_por_defecto_es_lo_mismo_que_todo(plan_falso):
    """Sin flags también sale la etapa `equipos` (todo lo que leen las tools)."""
    P.main([])
    assert _nombres(plan_falso) == ETAPAS


def test_todo_ignora_los_flags_de_recorte(plan_falso):
    P.main(["--todo", "--sin-mc", "--sin-evaluacion", "--rapido"])
    assert _nombres(plan_falso) == ETAPAS


def test_sin_mc_solo_general(plan_falso):
    """Comportamiento previo conservado: --sin-mc se para tras general."""
    P.main(["--sin-mc"])
    assert _nombres(plan_falso) == ["general"]


def test_sin_seleccion_y_sin_comparacion(plan_falso):
    P.main(["--sin-seleccion", "--sin-comparacion"])
    assert _nombres(plan_falso) == ["general", "mc", "geo", "pliegues",
                                    "evaluacion", "rutas", "equipos"]


def test_k_llegan_al_mc(plan_falso):
    P.main(["--k", "40", "50"])
    args, kwargs = plan_falso["mc"]
    assert args[0] == [40, 50]
    assert "rehacer" in kwargs and "n_sim" in kwargs


# ---------------------------------------------------------------------------
# --solo: una etapa, sin dependencias
# ---------------------------------------------------------------------------
def test_solo_equipos_ejecuta_solo_esa_etapa(plan_falso, cache_tmp):
    (cache_tmp / "rutas.json").write_text("{}", encoding="utf-8")   # dependencia
    P.main(["--solo", "equipos"])
    assert _nombres(plan_falso) == ["equipos"]


def test_solo_general_no_pide_nada(plan_falso, cache_tmp):
    P.main(["--solo", "general"])
    assert _nombres(plan_falso) == ["general"]


def test_solo_sin_dependencia_da_error_claro(plan_falso, cache_tmp, capsys):
    """Falta cache/rutas.json → código 2, NO se ejecuta y dice qué correr."""
    with pytest.raises(SystemExit) as exc:
        P.main(["--solo", "equipos"])
    assert exc.value.code == 2
    assert _nombres(plan_falso) == []
    salida = capsys.readouterr().out
    assert "cache/rutas.json" in salida
    assert "--solo rutas" in salida                  # la etapa que la genera


def test_solo_rutas_pide_evaluacion(plan_falso, cache_tmp, capsys):
    with pytest.raises(SystemExit) as exc:
        P.main(["--solo", "rutas"])
    assert exc.value.code == 2
    assert _nombres(plan_falso) == []
    salida = capsys.readouterr().out
    assert "cache/evaluacion.json" in salida
    assert "--solo evaluacion" in salida


def test_solo_etapa_desconocida(plan_falso, cache_tmp, capsys):
    with pytest.raises(SystemExit) as exc:
        P.main(["--solo", "noexiste"])
    assert exc.value.code == 2
    assert _nombres(plan_falso) == []
    err = capsys.readouterr().err
    assert "etapa desconocida" in err
    assert "equipos" in err                          # lista las opciones válidas


# ---------------------------------------------------------------------------
# Todo lo que leen las herramientas (y la API) queda generado
# ---------------------------------------------------------------------------
def test_todo_genera_todo_lo_que_leen_las_herramientas():
    """Requisito del bloque: ninguna cache leída por las tools queda fuera."""
    fuentes = [Path(H.__file__).resolve(),
               Path(P.__file__).resolve().parent / "main.py"]
    generados = {c for gens in P.GENERADOS.values() for c in gens}
    for fuente in fuentes:
        leidos = set(re.findall(r'_cache\("([^"]+)"\)',
                                fuente.read_text(encoding="utf-8")))
        assert leidos, f"{fuente.name}: no se detectó ninguna cache"
        fuera = leidos - generados
        assert not fuera, f"{fuente.name} lee caches que no se generan: {sorted(fuera)}"
    # la nueva cache está en ambos lados
    assert "equipos.json" in generados


def test_las_nueve_etapas_tienen_cache_propia():
    assert set(P.ORDEN_ETAPAS) == set(P.GENERADOS) == set(P.DEPENDENCIAS)
    assert P.ORDEN_ETAPAS[-1] == "equipos"
    # cada etapa sólo depende de caches que genera una etapa ANTERIOR
    for etapa, deps in P.DEPENDENCIAS.items():
        for cache in deps:
            productor = P._etapa_que_genera(cache)
            assert productor in P.ORDEN_ETAPAS
            assert P.ORDEN_ETAPAS.index(productor) < P.ORDEN_ETAPAS.index(etapa), \
                (etapa, cache, productor)


# ---------------------------------------------------------------------------
# Etapa `equipos`: contrato de cache/equipos.json (con datos sintéticos)
# ---------------------------------------------------------------------------
class _UniversoFalso:
    def __init__(self, clientes: pd.DataFrame) -> None:
        self.clientes = clientes


def _datos_sinteticos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """3 zonas, sin leer el CSV real (los tests nunca abren data/*.csv)."""
    clientes = pd.DataFrame({
        "ZON_REP": ["Z01", "Z01", "Z02", "Z03"],
        "equipo_codigo": ["EQ01", "EQ01", "EQ02", ""],
        "Cod CD": ["CD1", "CD1", "CD2", "CD3"],
    })
    va = pd.DataFrame({
        "ZON_REP": ["Z01", "Z01", "Z02", "Z03"],
        "fecha": ["2026-03-02", "2026-03-03", "2026-03-02", "2026-03-02"],
        "visitas": [40, 60, 55, 30],
    })
    return clientes, va


def _rutas_sinteticas() -> dict:
    """Misma forma que `cache/rutas.json` real: cada quincena es un DICT con
    `rutas` (no una lista), que es lo que lee `precompute_equipos()`.

    Datos ficticios: zonas Z01..Z03 y códigos de cliente cortos (sin dígitos
    largos, para no parecer cédulas).
    """
    def _quincena(n: int, rutas: list[dict]) -> dict:
        return {
            "quincena": n,
            "n_rutas": len(rutas),
            "total_visitas": sum(r["n_visitas"] for r in rutas),
            "total_km": round(sum(r["km"] for r in rutas), 2),
            "n_asesores": len({r["asesor"] for r in rutas}),
            "rutas": rutas,
        }

    q1 = [
        {"fecha": "2026-03-02", "asesor": "Z01", "n_visitas": 38, "km": 12.5,
         "secuencia": ["C01", "C02"], "excede_cap": False},
        {"fecha": "2026-03-02", "asesor": "Z02", "n_visitas": 50, "km": 9.0,
         "secuencia": ["C03"], "excede_cap": True},
    ]
    q2 = [
        {"fecha": "2026-03-16", "asesor": "Z01", "n_visitas": 41, "km": 13.5,
         "secuencia": ["C04", "C05"], "excede_cap": False},
    ]
    return {
        "actual": {"k_territorios": 3, "es_recomendada": True,
                   "etiqueta": "ACTUAL (zonas ficticias)",
                   "1": _quincena(1, q1), "2": _quincena(2, q2)},
        "recomendada": {"k_territorios": 3, "es_recomendada": False,
                        "etiqueta": "Recomendada (ficticia)",
                        "1": _quincena(1, []), "2": _quincena(2, [])},
    }


@pytest.fixture
def equipo_generado(cache_tmp, monkeypatch):
    """Corre la etapa `equipos` REAL con datos sintéticos y devuelve su JSON."""
    clientes, va = _datos_sinteticos()
    monkeypatch.setattr(P, "cargar_universo", lambda: _UniversoFalso(clientes))
    monkeypatch.setattr(P, "construir_calendario", lambda df: (va, {"n_fechas": 3}))
    (cache_tmp / "rutas.json").write_text(json.dumps(_rutas_sinteticas()),
                                          encoding="utf-8")
    P.precompute_equipos()
    return json.loads((cache_tmp / "equipos.json").read_text(encoding="utf-8"))


def test_equipos_genera_el_contrato_completo(equipo_generado):
    assert set(equipo_generado) == {"meta", "asesores"}
    meta = equipo_generado["meta"]
    assert meta["n_asesores"] == 3
    assert meta["cap_dinamico"] == cfg.cap_dinamico()
    assert (meta["obj_min"], meta["obj_max"]) == (cfg.OBJ_MIN, cfg.OBJ_MAX)
    for fila in equipo_generado["asesores"]:
        assert set(fila) == set(P.CAMPOS_EQUIPOS), \
            sorted(set(fila) ^ set(P.CAMPOS_EQUIPOS))


def test_equipos_agrega_calendario_y_rutas(equipo_generado):
    por_zona = {f["zona"]: f for f in equipo_generado["asesores"]}
    z01 = por_zona["Z01"]
    assert z01["equipo_codigo"] == "EQ01"
    assert z01["n_clientes"] == 2
    assert z01["visitas_programadas_mes"] == 100      # 40 + 60 del calendario
    assert z01["dias_operados"] == 2
    assert z01["media_visitas_dia"] == 50.0
    assert z01["pico_visitas_dia"] == 60
    assert z01["sobre_cap"] is True                   # 60 > CAP 44
    assert z01["en_objetivo_40_50"] is True           # media 50 ∈ [40, 50]
    assert z01["visitas_ruteadas"] == 79              # 38 + 41 del 1er ciclo
    assert z01["n_rutas"] == 2
    assert z01["km"] == 26.0
    # zona sin equipo en el CSV -> EQ00 (SIN_ASESOR), jamás un nombre
    assert por_zona["Z03"]["equipo_codigo"] == "EQ00"
    assert por_zona["Z03"]["visitas_ruteadas"] == 0   # no tiene rutas


def test_equipos_no_escribe_nombres(equipo_generado):
    """Ni claves ni valores con nombres de persona (sólo códigos EQxx/Zxx)."""
    for fila in equipo_generado["asesores"]:
        for clave, valor in fila.items():
            assert "nombre" not in clave.lower(), clave
            assert "supervisor" not in clave.lower(), clave
            if isinstance(valor, str):
                assert re.fullmatch(r"EQ\d\d|Z\d\d", valor), valor
    crudo = json.dumps(equipo_generado, ensure_ascii=False)
    assert _enmascara_nombres(crudo) == crudo          # ningún nombre real (E10)


def test_las_herramientas_leen_lo_generado_por_la_etapa(equipo_generado,
                                                        monkeypatch):
    """Fin a fin: la cache escrita por precompute la consumen las herramientas."""
    monkeypatch.setattr(H, "_equipos_cache", lambda: equipo_generado)

    det = H.ejecutar("detalle_asesor", {"zona": "Z01"})
    assert "error" not in det, det
    assert det["equipo_codigo"] == "EQ01"
    assert det["visitas_programadas_mes"] == 100

    eq = H.ejecutar("detalle_asesor", {"equipo": "EQ01"})
    assert eq["n_asesores"] == 1                    # sólo Z01 está en EQ01
    assert [a["zona"] for a in eq["asesores"]] == ["Z01"]
    eq2 = H.ejecutar("detalle_asesor", {"equipo": "EQ00"})
    assert eq2["n_asesores"] == 1                   # Z03 quedó SIN_ASESOR

    rank = H.ejecutar("ranking_asesores", {"orden": "visitas_ruteadas"})
    assert "error" not in rank, rank
    assert rank["ordenado_por"] == "visitas_ruteadas"
    assert [f["zona"] for f in rank["filas"]] == ["Z01", "Z02", "Z03"]
    assert rank["filas"][0]["visitas_ruteadas"] == 79


def test_equipos_sin_rutas_avisa_y_rellena_ceros(cache_tmp, monkeypatch, capsys):
    """Sin rutas.json la etapa avisa (no rompe) y deja los agregados en 0."""
    clientes, va = _datos_sinteticos()
    monkeypatch.setattr(P, "cargar_universo", lambda: _UniversoFalso(clientes))
    monkeypatch.setattr(P, "construir_calendario", lambda df: (va, {}))
    P.precompute_equipos()
    assert "falta cache/rutas.json" in capsys.readouterr().out
    datos = json.loads((cache_tmp / "equipos.json").read_text(encoding="utf-8"))
    assert all(f["n_rutas"] == 0 and f["visitas_ruteadas"] == 0
               for f in datos["asesores"])
