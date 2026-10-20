"""Capa de herramientas del chatbot (FASE 4) — 9 funciones que el LLM puede llamar.

Cada herramienta es una función pura (testeable sin LLM) que devuelve un dict
JSON-safe. **Privacidad**: nunca exponen `Identificacion`, `Cliente` ni
`Direccion`; sólo agregados y coordenadas. Leen de la cache (`cache/*.json`) que
genera `app/precompute.py`; si falta una K, la calculan al vuelo.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable

import numpy as np
import pandas as pd

from .. import config as cfg

_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Acceso a cache (con cálculo bajo demanda para K no cacheadas)
# ---------------------------------------------------------------------------
def _cache(nombre: str) -> Any | None:
    ruta = os.path.join(cfg.CACHE_DIR, nombre)
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    return None


def _montecarlo() -> dict:
    return _cache("montecarlo.json") or {"actual": None, "estrategias": {}, "comparaciones": {}}


def _asegurar_k(K: int) -> dict:
    """Devuelve la estrategia K; si no está en cache, la calcula (con lock)."""
    mc = _montecarlo()
    if str(K) in mc.get("estrategias", {}):
        return mc["estrategias"][str(K)]
    with _LOCK:
        mc = _montecarlo()
        if str(K) in mc.get("estrategias", {}):
            return mc["estrategias"][str(K)]
        from ..core.estructura import asignacion_capacidad
        from ..core.geo import proyectar
        from ..core.montecarlo import generar_entradas
        from ..core.universo import cargar_universo
        from ..precompute import evaluar
        u = cargar_universo()
        df = u.clientes.copy().reset_index(drop=True)
        df["_uid"] = np.arange(len(df))
        proyectar(df, col_grupo=None)
        asistencia, trafico = generar_entradas(cfg.N_SIM, len(df), cfg.SEMILLA)
        labels = asignacion_capacidad(df, K)
        res, _ = evaluar(df, labels, asistencia, trafico, f"K={K}")
        return res


# ---------------------------------------------------------------------------
# 9 herramientas
# ---------------------------------------------------------------------------
def ver_parametros(_entrada: dict | None = None) -> dict:
    """1. Parámetros del modelo y supuestos (calibrado/supuesto)."""
    p = _cache("params.json")
    if p:
        return {"cap_dinamico": p["cap_dinamico"], "costo_fijo_ciclo": p["costo_fijo_ciclo"],
                "k_min": p["k_min"], "k_max": p["k_max"], "k_actual": p["k_actual"],
                "params": p["params"]}
    return {"params": cfg.params(), "cap_dinamico": cfg.cap_dinamico()}


def ver_universo(_entrada: dict | None = None) -> dict:
    """2. Universo de clientes/visitas y embudo (E11). Sin `columnas_danadas` (interno E10)."""
    u = _cache("universos.json") or {"error": "universos no precalculado"}
    u.pop("columnas_danadas", None)
    return u


def ver_calendario(_entrada: dict | None = None) -> dict:
    """3. Calendario de visitas por fecha y descuadre (D2)."""
    return _cache("calendario.json") or {"error": "calendario no precalculado"}


def ver_eda(entrada: dict | None = None) -> dict:
    """4. EDA: KPIs, distribuciones, carga por zona y correlaciones."""
    e = _cache("eda.json")
    if not e:
        return {"error": "EDA no precalculado"}
    tema = (entrada or {}).get("tema")
    if tema and tema in e:
        return {tema: e[tema]}
    return {"kpis": e["kpis"], "distribuciones": e["distribuciones"]}


def carga_por_zona(entrada: dict | None = None) -> dict:
    """5. Carga por zona de rutero (top N)."""
    e = _cache("eda.json")
    if not e:
        return {"error": "EDA no precalculado"}
    cz = e["carga_zona"]
    top = int((entrada or {}).get("top", 10))
    top = max(1, min(top, len(cz["zonas"])))
    return {"top": top, "zonas": cz["zonas"][:top], "clientes": cz["clientes"][:top],
            "visitas_mes_rep": cz["visitas_mes_rep"][:top], "visitas_dia": cz["visitas_dia"][:top],
            "n_zonas": len(cz["zonas"])}


def factibilidad(entrada: dict | None = None) -> dict:
    """6. Factibilidad (restricción dura por asesor-día vs CAP) de una K o ACTUAL."""
    K = (entrada or {}).get("k")
    mc = _montecarlo()
    if K is None:
        res = mc.get("actual")
        etiqueta = "ACTUAL"
    else:
        res = _asegurar_k(int(K))
        etiqueta = f"K={int(K)}"
    if not res:
        return {"error": f"{etiqueta} no disponible aún"}
    return {"estrategia": etiqueta, "factible": res.get("factible"),
            "cap_dinamico": cfg.cap_dinamico(), "pico_carga": res.get("pico_carga"),
            "dias_sobre_cap": res.get("dias_sobre_cap"),
            "utilizacion_max": res.get("utilizacion_max"),
            "cv_carga": res.get("cv_carga"), "cobertura": res.get("cobertura")}


def estrategia(entrada: dict | None = None) -> dict:
    """7. Métricas MonteCarlo de una K o ACTUAL (costo, km, cobertura, factibilidad)."""
    K = (entrada or {}).get("k")
    mc = _montecarlo()
    if K is None:
        res = mc.get("actual")
        etiqueta = "ACTUAL"
    else:
        res = _asegurar_k(int(K))
        etiqueta = f"K={int(K)}"
    if not res:
        return {"error": f"{etiqueta} no disponible aún"}
    claves = ["estrategia", "K", "costo_medio_M", "ic95_inf", "ic95_sup", "p95_M",
              "km_medio", "cobertura", "visitas_perdidas", "factible", "pico_carga",
              "dias_sobre_cap", "prob_no_completar", "pct_fijo", "pct_combustible", "pct_extra"]
    return {k: res.get(k) for k in claves}


def comparar_estrategia(entrada: dict | None = None) -> dict:
    """8. Δ de costo de una K vs ACTUAL (K=58) con IC95 y P(mejor)."""
    K = (entrada or {}).get("k")
    if K is None:
        return {"error": "indica k"}
    mc = _montecarlo()
    comp = mc.get("comparaciones", {}).get(str(int(K)))
    base = mc.get("actual")
    if comp is None:
        return {"error": f"comparación K={int(K)} no disponible aún"}
    return {"k": int(K), "k_actual": cfg.K_ACTUAL, **comp,
            "costo_actual_M": (base or {}).get("costo_medio_M"),
            "costo_k_M": (mc.get("estrategias", {}).get(str(int(K))) or {}).get("costo_medio_M")}


def mejor_estrategia(_entrada: dict | None = None) -> dict:
    """9. Estrategia recomendada según la regla de decisión (6 condiciones)."""
    ev = _cache("evaluacion.json")
    if not ev:
        return {"error": "evaluación de estrategias aún no precalculada "
                         "(ejecuta `python -m app.precompute`)"}
    inf = ev.get("informe") or {}
    regla = inf.get("regla") or {}
    filas = [{k: f.get(k) for k in ("estrategia", "K", "costo_M", "delta_costo_pct",
                                   "delta_costo_M", "ic95_delta_costo_M",
                                   "km", "ratio_km", "dias_sobre_cap", "cobertura",
                                   "cv_carga", "cumple_todas", "es_diagnostica")}
             for f in (inf.get("tabla") or [])]
    umb = (inf.get("meta") or {}).get("umbrales") or {}
    return {
        "decision": regla.get("decision"),
        "recomendada": regla.get("recomendada"),
        "veredicto": regla.get("veredicto"),
        "condiciones_por_estrategia": regla.get("condiciones_por_estrategia"),
        "estrategias": filas,
        "criterio": regla.get("criterio"),
        "umbrales": umb,
        "etiqueta_umbrales": "UMBRALES_DECISION: propuesta, NO calibrada.",
        "f2_rebalanceo": {k: (ev.get("informe") or {}).get("f2_rebalanceo", {}).get(k)
                          for k in ("clientes_movidos", "clientes_movidos_pct",
                                    "dias_sobre_tope_antes", "dias_sobre_tope_despues",
                                    "delta_km_pct", "logro_factibilidad")},
        "nota": (regla.get("nota") or "") or None,
    }


# ---------------------------------------------------------------------------
# Registro (esquema OpenAI function-calling + función)
# ---------------------------------------------------------------------------
HERRAMIENTAS: dict[str, dict[str, Any]] = {
    "ver_parametros": {
        "fn": ver_parametros, "descripcion": "Parámetros del modelo y supuestos (calibrado/supuesto).",
        "parametros": {"type": "object", "properties": {}, "required": []},
    },
    "ver_universo": {
        "fn": ver_universo, "descripcion": "Universo de clientes/visitas y embudo de filtros.",
        "parametros": {"type": "object", "properties": {}, "required": []},
    },
    "ver_calendario": {
        "fn": ver_calendario, "descripcion": "Visitas por fecha y descuadre del calendario (D2).",
        "parametros": {"type": "object", "properties": {}, "required": []},
    },
    "ver_eda": {
        "fn": ver_eda, "descripcion": "EDA: KPIs, distribuciones, carga por zona, correlaciones.",
        "parametros": {"type": "object", "properties": {
            "tema": {"type": "string", "description": "kpis|distribuciones|carga_zona|correlaciones|histogramas"}},
            "required": []},
    },
    "carga_por_zona": {
        "fn": carga_por_zona, "descripcion": "Carga por zona de rutero (top N por visitas).",
        "parametros": {"type": "object", "properties": {
            "top": {"type": "integer", "description": "cuántas zonas (1-67)"}}, "required": []},
    },
    "factibilidad": {
        "fn": factibilidad, "descripcion": "¿Es factible una K (ningún asesor-día sobre CAP)? ACTUAL si no k.",
        "parametros": {"type": "object", "properties": {
            "k": {"type": "integer", "description": "número de asesores 35-70; omitir para ACTUAL"}},
            "required": []},
    },
    "estrategia": {
        "fn": estrategia, "descripcion": "Métricas MonteCarlo (costo, km, cobertura, factibilidad) de una K o ACTUAL.",
        "parametros": {"type": "object", "properties": {
            "k": {"type": "integer", "description": "35-70; omitir para ACTUAL"}}, "required": []},
    },
    "comparar_estrategia": {
        "fn": comparar_estrategia, "descripcion": "Δ de costo de una K vs ACTUAL (K=58) con IC95 y P(mejor).",
        "parametros": {"type": "object", "properties": {
            "k": {"type": "integer", "description": "35-70"}}, "required": ["k"]},
    },
    "mejor_estrategia": {
        "fn": mejor_estrategia,
        "descripcion": "Estrategia recomendada según la regla de decisión (6 condiciones "
                       "contra ACTUAL): resultado, tabla de estrategias y umbrales.",
        "parametros": {"type": "object", "properties": {}, "required": []},
    },
}


def schemas_tool() -> list[dict]:
    """Esquemas en formato OpenAI tools para enviar al LLM."""
    return [{
        "type": "function",
        "function": {
            "name": nombre,
            "description": h["descripcion"],
            "parameters": h["parametros"],
        },
    } for nombre, h in HERRAMIENTAS.items()]


def ejecutar(nombre: str, entrada: dict | None = None) -> dict:
    """Ejecuta una herramienta por nombre de forma segura (privacidad + errores)."""
    if nombre not in HERRAMIENTAS:
        return {"error": f"herramienta desconocida: {nombre}"}
    try:
        return HERRAMIENTAS[nombre]["fn"](entrada or {})
    except Exception as e:  # pragma: no cover — defensivo
        return {"error": f"{type(e).__name__}: {e}"}
