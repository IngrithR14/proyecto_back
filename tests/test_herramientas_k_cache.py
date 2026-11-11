"""`_asegurar_k` sin cálculo en runtime (FASE 5).

Si la K pedida no está en `cache/montecarlo.json`, la herramienta devuelve
`{"error": "K no precalculado", "k_disponibles": [...]}` **sin** abrir el CSV ni
llamar a `cargar_universo()` ni `evaluar()`. Sólo se lee la cache.
"""
from __future__ import annotations

import inspect

import pytest

from app.chatbot import herramientas as H


@pytest.fixture
def sin_calculo(monkeypatch):
    """Cualquier intento de calcular en runtime revienta el test."""
    from app import precompute
    from app.core import universo

    def _prohibido(*a, **k):
        raise AssertionError("no debe calcularse en runtime")

    monkeypatch.setattr(universo, "cargar_universo", _prohibido)
    monkeypatch.setattr(precompute, "evaluar", _prohibido)
    return True


def test_k_no_precalculada_devuelve_error(sin_calculo):
    res = H._asegurar_k(9999)
    assert res["error"] == "K no precalculado"
    assert isinstance(res["k_disponibles"], list) and res["k_disponibles"]
    assert res["k_disponibles"] == sorted(res["k_disponibles"])
    assert all(isinstance(k, int) for k in res["k_disponibles"])


def test_las_herramientas_propagan_el_error(sin_calculo):
    """`estrategia` y `factibilidad` no se tragan el error (ni lo disfrazan)."""
    for nombre in ("estrategia", "factibilidad"):
        res = H.ejecutar(nombre, {"k": 9999})
        assert res.get("error") == "K no precalculado", (nombre, res)
        assert res.get("k_disponibles"), f"{nombre}: sin K disponibles"


def test_k_precalculada_sigue_funcionando():
    """La K cacheada (35..70) no cambia de comportamiento."""
    res = H.ejecutar("estrategia", {"k": 45})
    assert "error" not in res, res
    assert res.get("costo_medio_M") is not None and res.get("cobertura") is not None


def test_asegurar_k_no_importa_csv_ni_evaluar():
    """El cuerpo de `_asegurar_k` ya no llama al CSV ni al simulador (AST)."""
    import ast

    arbol = ast.parse(inspect.getsource(H._asegurar_k))
    nombres: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Name):
            nombres.add(nodo.id)
        elif isinstance(nodo, ast.Attribute):
            nombres.add(nodo.attr)
        elif isinstance(nodo, (ast.Import, ast.ImportFrom)):
            nombres.update(a.name for a in nodo.names)
    for prohibido in ("cargar_universo", "evaluar", "read_csv", "asignacion_capacidad"):
        assert prohibido not in nombres, prohibido
    assert "K no precalculado" in inspect.getsource(H._asegurar_k)
