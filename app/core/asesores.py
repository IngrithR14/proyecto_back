"""Tabla puente de asesores (ZON_REP) y equipos — E10.

Con el CSV seudonimizado la columna de equipo es `cfg.COL_EQUIPO`
(`equipo_codigo`: EQ01..EQ06, EQ00 = SIN_ASESOR); nunca se propagan nombres
de supervisor. `buscar_asesor` sigue resolviendo por nombre con coincidencia
aproximada (difflib, umbral 0,8) sólo para búsquedas internas/tests; la salida
al usuario es siempre el código.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import unicodedata
from typing import Any

import pandas as pd

from .. import config as cfg

_RUTA_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "knowledge", "asesores.json")
UMBRAL_FUZZY = 0.8


def _norm(s: str) -> str:
    """Minúsculas sin tildes, signos ni caracteres dañados (U+FFFD).

    El U+FFFD se ELIMINA (no se sustituye por espacio) para que
    `Rodr\ufffdguez` se acerque a `rodriguez` y la búsqueda difusa lo resuelva.
    """
    s = str(s).replace("\ufffd", "")          # E10: carácter de reemplazo dañado
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in s.lower()).split())


def cargar_asesores() -> list[dict[str, Any]]:
    with open(_RUTA_JSON, encoding="utf-8") as f:
        return json.load(f)["asesores"]


def _indice_alias() -> dict[str, str]:
    """alias normalizado -> asesor_id."""
    idx: dict[str, str] = {}
    for a in cargar_asesores():
        for al in a["alias"] + [a["nombre_canonico"]]:
            idx[_norm(al)] = a["asesor_id"]
    return idx


def buscar_asesor(texto: str) -> dict[str, Any]:
    """Resuelve un nombre de equipo (o código de zona) al asesor/equipo correcto (E10)."""
    q = _norm(texto)
    idx = _indice_alias()
    if q in idx:
        a = next(x for x in cargar_asesores() if x["asesor_id"] == idx[q])
        return {"ok": True, "tipo": "equipo", "asesor": a}

    # coincidencia aproximada sobre alias y nombres canónicos
    candidatos = list(idx.keys())
    cerca = difflib.get_close_matches(q, candidatos, n=3, cutoff=UMBRAL_FUZZY)
    if cerca:
        aid = idx[cerca[0]]
        a = next(x for x in cargar_asesores() if x["asesor_id"] == aid)
        return {"ok": True, "tipo": "equipo_aprox", "asesor": a,
                "coincidencia": round(difflib.SequenceMatcher(None, q, cerca[0]).ratio(), 3),
                "sugerencias": [next(x["nombre_canonico"] for x in cargar_asesores()
                                     if x["asesor_id"] == idx[c]) for c in cerca]}
    return {"ok": False, "error_codigo": "ASESOR_NO_ENCONTRADO",
            "mensaje": f"No existe un asesor o equipo llamado {texto!r}.",
            "sugerencias": [a["nombre_canonico"] for a in cargar_asesores()[:4]]}


_RE_CODIGO_EQ = re.compile(r"^EQ\d{2}$")


def es_codigo_equipo(v: Any) -> bool:
    """¿El valor ya es un código de equipo seudonimizado (EQ00..EQ99)?"""
    return bool(_RE_CODIGO_EQ.match(str(v).strip().upper()))


def normalizar_equipo(df: pd.DataFrame, col: str | None = None) -> pd.DataFrame:
    """Sustituye los nombres de Supervisor dañados por su forma canónica (E10).

    Combina coincidencia exacta (tras `_norm`) con difusa (difflib ≥ UMBRAL_FUZZY)
    para resolver nombres con caracteres U+FFFD que no empatan exactamente.

    Si la columna ya contiene CÓDIGOS (`equipo_codigo`: EQ01..EQ06) la función
    NO hace nada: no hay nombres que reparar ni con qué cruzarlos. Sólo se
    normalizan valores que parecen nombres (antes de la seudonimización).
    """
    import difflib
    out = df.copy()
    col = col or cfg.COL_EQUIPO
    if col not in out.columns:
        return out
    if len(out[col].dropna()) and out[col].dropna().map(es_codigo_equipo).all():
        return out                              # ya seudonimizado: nada que reparar

    mapa: dict[str, str] = {}
    for a in cargar_asesores():
        for al in a["alias"] + [a["nombre_canonico"]]:
            mapa[_norm(al)] = a["equipo"]
        mapa[_norm(a["equipo"])] = a["equipo"]
    claves = list(mapa.keys())

    cache: dict[str, str] = {}

    def _resolver(v):
        if es_codigo_equipo(v):                 # código: se deja tal cual
            return v
        k = _norm(v)
        if k in cache:
            return cache[k]
        if k in mapa:
            res = mapa[k]
        else:
            cerca = difflib.get_close_matches(k, claves, n=1, cutoff=UMBRAL_FUZZY)
            res = mapa[cerca[0]] if cerca else v
        cache[k] = res
        return res

    out[col] = out[col].map(_resolver)
    return out


def tabla_asesores(va: pd.DataFrame, clientes: pd.DataFrame | None = None,
                   col_asesor: str | None = None) -> list[dict[str, Any]]:
    """Carga, equipo, CDs, ocupación y utilización por asesor (ZON_REP)."""
    col_asesor = col_asesor or cfg.COL_ASESOR
    if not len(va):
        return []
    cad = va.groupby([col_asesor, "fecha"])["visitas"].sum().reset_index()
    filas = []
    for az, g in cad.groupby(col_asesor):
        cargas = g["visitas"]
        equipo, cds = "", []
        if clientes is not None and col_asesor in clientes.columns:
            sub = clientes[clientes[col_asesor] == az]
            if len(sub) and cfg.COL_EQUIPO in sub.columns:
                eq = sub[cfg.COL_EQUIPO].dropna()
                if len(eq):
                    # con el CSV seudonimizado aquí llega el CÓDIGO (EQ01..EQ06),
                    # nunca el nombre del supervisor
                    equipo = str(eq.value_counts().index[0])
                cds = sorted(sub["Cod CD"].dropna().astype(str).unique().tolist())
        filas.append({
            "asesor_id": az,
            "nombre": az,
            "equipo_codigo": equipo,
            "cds": cds,
            "n_clientes": int((clientes[col_asesor] == az).sum())
            if clientes is not None and col_asesor in clientes.columns else None,
            "visitas_mes": int(cargas.sum()),
            "dias_operados": int(g["fecha"].nunique()),
            "media_visitas_dia": round(float(cargas.mean()), 2),
            "pico_visitas_dia": int(cargas.max()),
            "cap_dinamico_ref": cfg.cap_dinamico(),
            "sobre_cap": bool(cargas.max() > cfg.cap_dinamico()),
            "en_objetivo_40_50": bool(cfg.OBJ_MIN <= cargas.mean() <= cfg.OBJ_MAX),
        })
    filas.sort(key=lambda x: -x["visitas_mes"])
    return filas


def _mapa_nombre_a_codigo() -> dict[str, str]:
    """Mapa privado `_norm(nombre) -> EQxx`. Vacío si no existe el archivo."""
    ruta = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "knowledge", "asesores_privado.json")
    if not os.path.exists(ruta):
        return {}
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def detalle_asesor(asesor_id: str, va: pd.DataFrame, clientes: pd.DataFrame | None = None
                   ) -> dict[str, Any]:
    """Detalle por código de zona (ZON_REP), por código de equipo (EQxx) o por
    nombre de equipo (E10). TODA la salida usa `equipo_codigo`; los nombres de
    supervisor no se devuelven nunca.
    """
    tabla = tabla_asesores(va, clientes)
    q = str(asesor_id).strip()

    # 1) código de equipo (EQ00..EQ06)
    if es_codigo_equipo(q):
        cod = q.upper()
        rel = [f for f in tabla if f["equipo_codigo"] == cod]
        if rel:
            return {"ok": True, "tipo": "equipo", "equipo_codigo": cod, "asesores": rel}
        return {"ok": True, "tipo": "equipo", "equipo_codigo": cod, "asesores": [],
                "nota": "El equipo no tiene asesores (zonas) en el universo."}

    # 2) código de zona (ZON_REP)
    for f in tabla:
        if f["asesor_id"] == asesor_id:
            return {"ok": True, "tipo": "zona", "asesor": f}

    # 3) nombre de equipo -> código, con el mapa privado (si existe)
    mapa = _mapa_nombre_a_codigo()
    codigo = mapa.get(_norm(q))
    if codigo:
        rel = [f for f in tabla if f["equipo_codigo"] == codigo]
        return {"ok": True, "tipo": "equipo", "equipo_codigo": codigo, "asesores": rel}

    codigos = sorted({f["equipo_codigo"] for f in tabla if f["equipo_codigo"]})
    return {"ok": False, "error_codigo": "ASESOR_NO_ENCONTRADO",
            "mensaje": f"No existe una zona ni un equipo identificado por {q!r}.",
            "equipo_codigo_validos": codigos}
