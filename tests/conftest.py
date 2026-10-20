"""Configuración de pytest: marcadores y fixtures compartidos (FASE 6)."""
from __future__ import annotations

import os
import sys

import pytest

# asegurar que `app` sea importable desde la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm: pruebas que requieren LLM/API key real (se omiten sin ella)")


@pytest.fixture(scope="session")
def universo():
    """Universo de trabajo (14.757 clientes) — caro, se cachea por sesión."""
    from app.core.universo import cargar_universo
    return cargar_universo()


@pytest.fixture(scope="session")
def base_eda():
    """Base limpia del EDA (16.560 filas)."""
    from app.core.eda import cargar_base
    return cargar_base()


@pytest.fixture(scope="session")
def cliente_api():
    """Cliente de prueba de FastAPI."""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture
def llm_disponible() -> bool:
    from app.chatbot.orquestador import GroqProveedor
    return GroqProveedor().disponible()
