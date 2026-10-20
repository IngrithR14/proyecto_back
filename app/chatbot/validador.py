"""Validador de privacidad (FASE 4) — red de seguridad tras las herramientas.

Aunque las herramientas sólo devuelven agregados, este validador escanea la
respuesta final y las salidas de herramientas en busca de posibles datos
personales (PII) y los enmascara. Es una barrera adicional, no la única.
"""
from __future__ import annotations

import re
from typing import Any

# Columnas que NUNCA deben aparecer en la salida del chatbot.
# Se comparan con límites de palabra y sensibles a mayúsculas: la columna PII es
# `Cliente`/`Identificacion`/`Direccion`; los agregados legítimos son `clientes`,
# `n_clientes`, etc. (no deben disparar la alarma).
COLUMNAS_PROHIBIDAS = ("Identificacion", "Cliente", "Direccion")

_COL_RE = {c: re.compile(rf"\b{re.escape(c)}\b") for c in COLUMNAS_PROHIBIDAS}

# Patrones de PII en texto libre
_PATRONES = [
    (re.compile(r"\b\d{6,12}\b"), "[cédula oculta]"),           # cédula/NIT
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[correo oculto]"),
    (re.compile(r"\b(?:\+?57\s?)?(?:3\d{9}|[1-9]\d{6,7})\b"), "[teléfono oculto]"),
]


def _escanea_texto(texto: str) -> str:
    for p, rep in _PATRONES:
        texto = p.sub(rep, texto)
    return texto


def _recorre(obj: Any, out: list[str]) -> None:
    """Recolecta todo texto de un objeto anidado."""
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            _recorre(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _recorre(v, out)


def validar_salida(respuesta: str, herramientas: list[dict] | None = None
                   ) -> dict[str, Any]:
    """Escanea la respuesta y las salidas de herramientas; enmascara PII.

    Devuelve {respuesta, limpio: bool, incidencias: [...]}. Sustituye claves
    prohibidas por su valor agregado si aparecieran.
    """
    incidencias: list[str] = []
    texto_resp = respuesta

    # 1. respuesta final
    limpio_resp = _escanea_texto(texto_resp)
    if limpio_resp != texto_resp:
        incidencias.append("respuesta: se enmascararon dígitos/correos/teléfonos")

    # 2. salidas de herramientas: claves prohibidas
    for tr in herramientas or []:
        salida = tr.get("salida", {})
        textos: list[str] = []
        _recorre(salida, textos)
        for t in textos:
            for col in COLUMNAS_PROHIBIDAS:
                if _COL_RE[col].search(t):
                    incidencias.append(f"herramienta {tr.get('nombre')}: mención a '{col}'")

    return {
        "respuesta": limpio_resp,
        "limpio": len(incidencias) == 0,
        "incidencias": incidencias,
    }
