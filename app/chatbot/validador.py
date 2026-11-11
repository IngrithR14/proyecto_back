"""Validador de privacidad (FASE 4 + FASE 5) — red de seguridad tras las herramientas.

Aunque las herramientas sólo devuelven agregados y códigos (`zona`,
`equipo_codigo`), este validador escanea la respuesta final y las salidas de
herramientas en busca de posibles datos personales (PII) y los enmascara. Es una
barrera adicional, no la única.

Dos capas:
1. Columnas prohibidas (`Identificacion`, `Cliente`, `Direccion`, `Supervisor`):
   si una herramienta filtrara la columna con nombres, se marca la incidencia.
2. Nombres de supervisor (E10): se contrasta con `app/knowledge/asesores.json`
   (interno; NUNCA se envía al LLM) y se enmascaran en la respuesta. Los códigos
   `EQ01..EQ06` y `ZON_REP` no son PII: pasan sin alarma.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

# Columnas que NUNCA deben aparecer en la salida del chatbot.
# Se comparan con límites de palabra y sensibles a mayúsculas: la columna PII es
# `Cliente`/`Identificacion`/`Direccion`; los agregados legítimos son `clientes`,
# `n_clientes`, etc. (no deben disparar la alarma). `Supervisor` es el nombre de
# la persona del equipo: sólo puede salir `equipo_codigo` (EQxx).
COLUMNAS_PROHIBIDAS = ("Identificacion", "Cliente", "Direccion", "Supervisor")

_COL_RE = {c: re.compile(rf"\b{re.escape(c)}\b") for c in COLUMNAS_PROHIBIDAS}

# Patrones de PII en texto libre
_PATRONES = [
    (re.compile(r"\b\d{6,12}\b"), "[cédula oculta]"),           # cédula/NIT
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[correo oculto]"),
    (re.compile(r"\b(?:\+?57\s?)?(?:3\d{9}|[1-9]\d{6,7})\b"), "[teléfono oculto]"),
]

_RUTA_ASESORES = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "knowledge", "asesores.json")

# Variantes ortográficas toleradas al detectar nombres (tildes y eñe)
_VARIANTES = {"a": "aáàä", "e": "eéèë", "i": "iíìï",
              "o": "oóòö", "u": "uúùü", "n": "nñ"}


def _patron_nombre(frase: str) -> re.Pattern[str]:
    """Regex de frase con límites de palabra, tolerante a mayúsculas y tildes.

    Los espacios se aceptan flexibles (`Orlando  Sandoval` ≈ `Orlando Sandoval`).
    """
    partes: list[str] = []
    for ch in frase.lower():
        if ch in _VARIANTES:
            partes.append(f"[{_VARIANTES[ch]}]")
        elif ch.isalnum():
            partes.append(re.escape(ch))
        else:
            partes.append(r"\s+")
    return re.compile(r"(?<!\w)" + "".join(partes) + r"(?!\w)", re.IGNORECASE)


@lru_cache(maxsize=1)
def _patrones_supervisor() -> tuple[re.Pattern[str], ...]:
    """Patrones de nombres de supervisor (E10) para detectar fugas.

    Se leen de `app/knowledge/asesores.json` (interno, NUNCA se envía al LLM).
    La entrada `SIN_ASESOR` no es una persona y se excluye para no marcar la
    frase común "sin asesor". Si falta el archivo o no se puede leer, la lista
    queda vacía y el resto del validador sigue funcionando.
    """
    if not os.path.exists(_RUTA_ASESORES):
        return ()
    try:
        with open(_RUTA_ASESORES, encoding="utf-8") as f:
            datos = json.load(f)
    except (OSError, ValueError):
        return ()

    patrones: dict[str, re.Pattern[str]] = {}
    for a in datos.get("asesores", []):
        if str(a.get("equipo", "")).upper() == "SIN_ASESOR":
            continue                                  # "sin asesor" no es persona
        for alias in [a.get("nombre_canonico", ""), *a.get("alias", [])]:
            alias = str(alias or "").strip()
            if len(alias) < 3:
                continue
            patrones.setdefault(alias.lower(), _patron_nombre(alias))
    return tuple(patrones.values())


def _escanea_texto(texto: str) -> str:
    for p, rep in _PATRONES:
        texto = p.sub(rep, texto)
    return texto


def _enmascara_nombres(texto: str) -> str:
    """Sustituye nombres de supervisor por `[nombre oculto]` (tildes ignoradas)."""
    for p in _patrones_supervisor():
        texto = p.sub("[nombre oculto]", texto)
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

    # 1. respuesta final: dígitos/correos/teléfonos
    limpio_resp = _escanea_texto(texto_resp)
    if limpio_resp != texto_resp:
        incidencias.append("respuesta: se enmascararon dígitos/correos/teléfonos")

    # 2. respuesta final: nombres de supervisor (E10) -> [nombre oculto]
    sin_nombres = _enmascara_nombres(limpio_resp)
    if sin_nombres != limpio_resp:
        incidencias.append("respuesta: se enmascaró un nombre de supervisor")
        limpio_resp = sin_nombres

    # 3. salidas de herramientas: claves prohibidas y nombres de supervisor
    for tr in herramientas or []:
        salida = tr.get("salida", {})
        textos: list[str] = []
        _recorre(salida, textos)
        marcadas: set[str] = set()
        for t in textos:
            for col in COLUMNAS_PROHIBIDAS:
                if col not in marcadas and _COL_RE[col].search(t):
                    marcadas.add(col)
                    incidencias.append(
                        f"herramienta {tr.get('nombre')}: mención a '{col}'")
            if "nombre" not in marcadas:
                if any(p.search(t) for p in _patrones_supervisor()):
                    marcadas.add("nombre")
                    incidencias.append(
                        f"herramienta {tr.get('nombre')}: mención a nombre de supervisor")

    return {
        "respuesta": limpio_resp,
        "limpio": len(incidencias) == 0,
        "incidencias": incidencias,
    }
