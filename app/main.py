"""API FastAPI (FASE 3) — sirve el núcleo y la cache al dashboard Angular.

Ejecutar:  uvicorn app.main:app --reload --port 8000
El frontend corre en :4200 con proxy → :8000 (ver proyecto_front/proxy.conf.json).

Privacidad: la API NO expone `Identificacion`, `Cliente` ni `Direccion`; sólo
agregados y coordenadas. El chat (`/api/chat`) aplica validador de privacidad.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import config as cfg

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(
    title="API Ruteo Hijos de León",
    version="1.0.0",
    description="EDA + Modelo de ruteo (NN vs ACO) + MonteCarlo de asesores. "
                "Privacidad: sin datos personales (sólo agregados/coordenadas).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(cfg.FRONTEND_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Utilidades de cache
# ---------------------------------------------------------------------------
def _cache(nombre: str) -> Any | None:
    ruta = os.path.join(cfg.CACHE_DIR, nombre)
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    return None


def _o_cache(nombre: str) -> Any:
    d = _cache(nombre)
    if d is None:
        raise HTTPException(503, f"'{nombre}' aún no precalculado. Ejecuta "
                                 f"`python -m app.precompute`.")
    return d


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
@app.get("/api/salud")
def salud() -> dict:
    return {"ok": True, "version": app.version, "cache_dir": cfg.CACHE_DIR,
            "montecarlo_cacheado": _cache("montecarlo.json") is not None}


@app.get("/api/meta/params")
def meta_params() -> dict:
    return _o_cache("params.json")


@app.get("/api/meta/universos")
def meta_universos() -> dict:
    d = _o_cache("universos.json")
    d.pop("columnas_danadas", None)   # interno E10, no necesario al cliente
    return d


@app.get("/api/meta/calendario")
def meta_calendario() -> dict:
    return _o_cache("calendario.json")


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
@app.get("/api/eda/resumen")
def eda_resumen() -> dict:
    return _o_cache("eda.json")


@app.get("/api/eda/carga-zona")
def eda_carga_zona(top: int = 15) -> dict:
    e = _o_cache("eda.json")["carga_zona"]
    top = max(1, min(top, len(e["zonas"])))
    return {"top": top, "zonas": e["zonas"][:top], "clientes": e["clientes"][:top],
            "activos": e["activos"][:top], "visitas_mes_rep": e["visitas_mes_rep"][:top],
            "visitas_dia": e["visitas_dia"][:top], "n_zonas": len(e["zonas"])}


@app.get("/api/eda/correlaciones")
def eda_correlaciones() -> dict:
    return _o_cache("eda.json")["correlaciones"]


# ---------------------------------------------------------------------------
# Modelo de ruteo (NN vs ACO)
# ---------------------------------------------------------------------------
@app.get("/api/ruteo/comparacion")
def ruteo_comparacion() -> dict:
    return _o_cache("ruteo.json")


@app.get("/api/ruteo/kfold")
def ruteo_kfold() -> dict:
    """K-Fold CV del ruteo: estabilidad de las métricas entre pliegues (media±desv)."""
    return _o_cache("pliegues.json")


@app.get("/api/ruteo/rutas")
def ruteo_rutas(estrategia: str = "recomendada", quincena: int = 1) -> dict:
    """Rutas del primer ciclo: por día y asesor, la secuencia de visitas y km."""
    if quincena not in (1, 2):
        raise HTTPException(422, "quincena debe ser 1 o 2")
    r = _cache("rutas.json")
    if not r:
        raise HTTPException(503, "rutas aún no precalculadas. Ejecuta `python -m app.precompute`.")
    est = r.get(estrategia)
    if not est:
        raise HTTPException(422, f"estrategia '{estrategia}' no disponible "
                                 f"(usa 'actual' o 'recomendada')")
    q = est.get(str(quincena))
    if q is None:
        raise HTTPException(422, f"quincena {quincena} no disponible para '{estrategia}'")
    return {"estrategia": estrategia, "k_territorios": est.get("k_territorios"),
            "es_recomendada": est.get("es_recomendada", True),
            "etiqueta": est.get("etiqueta"), **q}


# ---------------------------------------------------------------------------
# Protocolo de evaluación (secciones 2 y 3)
# ---------------------------------------------------------------------------
@app.get("/api/evaluacion/nn-vs-aco")
def ev_nn_vs_aco() -> dict:
    """NN vs ACO por pares de ruta: ahorros con IC95, por tamaño, robustez y regla."""
    d = _o_cache("comparacion_rutas.json")
    return {k: d.get(k) for k in ("meta", "global", "por_tamano", "por_ruta",
                                  "wilcoxon", "robustez", "regla", "criterio_regla")}


@app.get("/api/evaluacion/estrategias")
def ev_estrategias() -> dict:
    """Estrategias contra ACTUAL: 8 dimensiones por estrategia, tabla, F2 y estabilidad."""
    d = _o_cache("evaluacion.json")
    inf = d.get("informe") or {}
    return {"meta": inf.get("meta"), "tabla": inf.get("tabla"),
            "estrategias": inf.get("estrategias"), "f2_rebalanceo": inf.get("f2_rebalanceo"),
            "estabilidad": inf.get("estabilidad"),
            "eleccion": {"semilla": (d.get("eleccion") or {}).get("meta", {}).get("semilla_eleccion"),
                         "tabla": (d.get("eleccion") or {}).get("tabla"),
                         "regla": (d.get("eleccion") or {}).get("regla")},
            "nota": d.get("nota")}


@app.get("/api/evaluacion/regla")
def ev_regla() -> dict:
    """Las dos reglas de decisión con sus condiciones (semáforo) y sus criterios."""
    ev = _cache("evaluacion.json") or {}
    nn = _cache("comparacion_rutas.json") or {}
    inf = ev.get("informe") or {}
    return {
        "estrategias": inf.get("regla"),
        "nn_vs_aco": nn.get("regla"),
        "criterio_estrategias": inf.get("criterio"),
        "criterio_nn_vs_aco": nn.get("criterio_regla"),
        "umbrales": inf.get("meta", {}).get("umbrales"),
        "etiqueta_umbrales": "UMBRALES_DECISION: propuesta, no calibrada.",
    }


# ---------------------------------------------------------------------------
# MonteCarlo
# ---------------------------------------------------------------------------
@app.get("/api/montecarlo/tabla")
def mc_tabla() -> dict:
    """Tabla de todas las estrategias K + ACTUAL + comparaciones."""
    mc = _o_cache("montecarlo.json")
    return {"n_sim": mc.get("n_sim"), "actual": mc.get("actual"),
            "estrategias": mc.get("estrategias", {}),
            "comparaciones": mc.get("comparaciones", {})}


@app.get("/api/montecarlo/actual")
def mc_actual() -> dict:
    mc = _o_cache("montecarlo.json")
    a = mc.get("actual")
    if not a:
        raise HTTPException(503, "ACTUAL no precalculado")
    return a


@app.get("/api/montecarlo/estrategia/{k}")
def mc_estrategia(k: int) -> dict:
    if not (cfg.K_MIN <= k <= cfg.K_MAX):
        raise HTTPException(422, f"k debe estar entre {cfg.K_MIN} y {cfg.K_MAX}")
    mc = _o_cache("montecarlo.json")
    r = mc.get("estrategias", {}).get(str(k))
    if r is None:
        raise HTTPException(503, f"K={k} aún no precalculado")
    return r


@app.get("/api/montecarlo/comparacion/{k}")
def mc_comparacion(k: int) -> dict:
    mc = _o_cache("montecarlo.json")
    c = mc.get("comparaciones", {}).get(str(k))
    if c is None:
        raise HTTPException(503, f"comparación K={k} aún no precalculada")
    base = mc.get("actual") or {}
    est = mc.get("estrategias", {}).get(str(k)) or {}
    return {"k": k, "k_actual": cfg.K_ACTUAL, **c,
            "costo_actual_M": base.get("costo_medio_M"),
            "costo_k_M": est.get("costo_medio_M")}


@app.get("/api/montecarlo/mejor")
def mc_mejor() -> dict:
    mc = _o_cache("montecarlo.json")
    est = mc.get("estrategias", {})
    filas = [{"K": int(kk), "costo_medio_M": r.get("costo_medio_M"),
              "factible": r.get("factible"), "cobertura": r.get("cobertura"),
              "pico_carga": r.get("pico_carga"), "km_medio": r.get("km_medio")}
             for kk, r in est.items()]
    factibles = sorted([f for f in filas if f["factible"]], key=lambda x: x["costo_medio_M"])
    todos = sorted(filas, key=lambda x: x["costo_medio_M"])
    return {"minima_factible": min((f["K"] for f in factibles), default=None),
            "mejor_factible": factibles[0] if factibles else None,
            "mas_barata_global": todos[0] if todos else None,
            "n_estrategias": len(filas)}


# ---------------------------------------------------------------------------
# Geografía (mapa Leaflet)
# ---------------------------------------------------------------------------
@app.get("/api/geo/{clave}")
def geo(clave: str) -> dict:
    """'actual' u otro K (como string) → territorios + puntos muestreados."""
    g = _cache("geo.json")
    if not g or clave not in g:
        raise HTTPException(503, f"geo '{clave}' aún no precalculado")
    return g[clave]


# ---------------------------------------------------------------------------
# Chatbot (FASE 4)
# ---------------------------------------------------------------------------
class ChatIn(BaseModel):
    pregunta: str
    sesion: str = "default"


_RL: dict[str, deque] = defaultdict(deque)


def _rate_limit(sid: str, max_por_min: int = 20) -> bool:
    ahora = time.time()
    q = _RL[sid]
    while q and ahora - q[0] > 60:
        q.popleft()
    if len(q) >= max_por_min:
        return False
    q.append(ahora)
    return True


@app.post("/api/chat")
def chat(entrada: ChatIn, request: Request) -> dict:
    from . import chatbot
    sid = entrada.sesion or (request.client.host if request.client else "anon")
    if not _rate_limit(sid):
        raise HTTPException(429, "Demasiadas preguntas; espera un minuto.")
    if not entrada.pregunta.strip():
        raise HTTPException(422, "pregunta vacía")
    try:
        return chatbot.chat(entrada.pregunta, sid=sid)
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.get("/api/chat/salud")
def chat_salud() -> dict:
    """¿Está configurado el LLM? (no expone la key; modelos desde .env)"""
    from .chatbot.orquestador import GroqProveedor
    g = GroqProveedor()
    return {"proveedor": cfg.LLM_PROVIDER, "modelo": cfg.LLM_MODEL,
            "modelo_fallback": cfg.LLM_MODEL_FALLBACK or None,
            "configurado": g.disponible(),
            "nota": None if g.disponible()
                    else "Falta LLM_API_KEY o LLM_MODEL en .env; el chat no funcionará."}


# ---------------------------------------------------------------------------
# Manejo de errores global (JSON limpio)
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def _http_exc(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
