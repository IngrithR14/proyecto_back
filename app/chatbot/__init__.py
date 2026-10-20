"""Chatbot (FASE 4) — API pública.

`chat(pregunta, sid)` ejecuta el bucle orquestador y aplica el validador de
privacidad. El proveedor LLM es intercambiable (Groq real o Mock en pruebas).
"""
from __future__ import annotations

from typing import Any

from . import herramientas
from .orquestador import GroqProveedor, MockProveedor, responder
from .sesiones import ALMACEN
from .validador import COLUMNAS_PROHIBIDAS, validar_salida

__all__ = ["chat", "herramientas", "responder", "GroqProveedor", "MockProveedor",
           "validar_salida", "COLUMNAS_PROHIBIDAS", "ALMACEN"]


def chat(pregunta: str, sid: str = "default",
         proveedor: Any | None = None) -> dict[str, Any]:
    """Pregunta → orquestador → validador de privacidad → respuesta segura."""
    res = responder(pregunta, sid=sid, proveedor=proveedor)
    val = validar_salida(res["respuesta"], res["herramientas"])
    return {
        "respuesta": val["respuesta"],
        "herramientas": [t["nombre"] for t in res["herramientas"]],
        "detalle_herramientas": res["herramientas"],
        "privacidad_ok": val["limpio"],
        "incidencias": val["incidencias"],
        "sesion": res["sesion"],
        "modelo": res.get("modelo"),   # modelo real usado (principal o fallback)
    }
