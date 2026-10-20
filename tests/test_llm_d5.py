"""Pruebas del ajuste D5 del LLM (FASE 4): modelos desde .env, fallback, privacidad.

Cubren los 7 puntos de D5:
  1. Nombres de modelo SOLO en .env (test que escanea el código fuente).
  2. Fallback ante 429/5xx/timeout/modelo no disponible; 401 sin reintento.
  3. Sólo `message.content`; el campo `reasoning` se descarta.
  4. `content` vacío o `finish_reason="length"` → reintento con tokens dobles.
  5. Tools en formato OpenAI.
Las pruebas de red se SIMULAN (monkeypatch), por lo que no requieren API key.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
import pytest

from app import config as cfg
from app.chatbot import MockProveedor, responder
from app.chatbot.orquestador import GroqProveedor

RAIZ = Path(__file__).resolve().parents[2]          # .../proyecto_especializacion


# ---------------------------------------------------------------------------
# 1. Ningún nombre de modelo escrito en el código fuente (sólo .env / .env.example)
# ---------------------------------------------------------------------------
# Patrón genérico: familias de modelos (llama-, gpt-, mixtral-, ...). Captura
# tanto el modelo antiguo (`llama-...`) como el nuevo (`openai/gpt-oss-120b`, vía
# `gpt-oss`). Se construye sin contener nombres literales completos, para que el
# propio test no se auto-gatille. El formato `proveedor/modelo` se omite a propósito
# para no confundirlo con códigos internos (`E2/E3`, `Q1/Q2`, `D2/D3`).
_PATRON_MODELO = re.compile(
    r"\b(?:llama|gpt|mixtral|mistral|qwen|gemma|phi|deepseek|claude|gemini)"
    r"[-_.][\w.-]*",
    re.IGNORECASE,
)
_URL = re.compile(r"(?:https?|ftp)://\S+|www\.\S+", re.IGNORECASE)


def _archivos_fuente() -> list[Path]:
    fuentes: list[Path] = []
    fuentes += sorted((RAIZ / "proyecto_back" / "app").rglob("*.py"))
    front = RAIZ / "proyecto_front" / "src"
    if front.exists():
        fuentes += sorted(front.rglob("*.ts")) + sorted(front.rglob("*.html"))
    readme = RAIZ / "README.md"
    if readme.exists():
        fuentes.append(readme)
    # .env y .env.example quedan EXCLUIDOS por diseño (no están en estos globs)
    return [f for f in fuentes if f.is_file()]


def test_ningun_nombre_de_modelo_en_el_codigo():
    """D5.1: falla si aparece un nombre de modelo fuera de .env / .env.example."""
    infracciones: list[str] = []
    for f in _archivos_fuente():
        texto = _URL.sub("", f.read_text(encoding="utf-8", errors="ignore"))
        for m in _PATRON_MODELO.finditer(texto):
            infracciones.append(f"{f.relative_to(RAIZ)}: {m.group(0)!r}")
    assert not infracciones, (
        "Se encontraron nombres de modelo escritos en el código fuente "
        "(deben vivir sólo en .env / .env.example):\n  " + "\n  ".join(infracciones))


def test_config_sin_modelo_fijo():
    """D5.1: `LLM_MODEL` y `LLM_MODEL_FALLBACK` no traen valor por defecto en config."""
    src = (RAIZ / "proyecto_back" / "app" / "config.py").read_text(encoding="utf-8")
    # los defaults deben venir de getenv con cadena vacía, no de un literal de modelo
    assert 'os.getenv("LLM_MODEL", "")' in src
    assert 'os.getenv("LLM_MODEL_FALLBACK", "")' in src


# ---------------------------------------------------------------------------
# 2. Fallback (429/5xx/timeout/404) — 401 sin reintento
# ---------------------------------------------------------------------------
def _exc_estado(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://test")
    resp = httpx.Response(code, request=req)
    return httpx.HTTPStatusError(f"{code}", request=req, response=resp)


def _proveedor() -> GroqProveedor:
    return GroqProveedor(api_key="k-de-prueba", model="principal",
                         model_fallback="respaldo", timeout=5)


@pytest.mark.parametrize("code", [429, 500, 503, 404])
def test_fallback_reintenta_una_vez(code, monkeypatch):
    """D5.2: ante 429/5xx/404 reintenta con el fallback y registra el modelo."""
    g = _proveedor()
    llamados: list[str] = []

    def fake(model, *a, **k):
        llamados.append(model)
        if model == "principal":
            raise _exc_estado(code)
        return {"content": "ok", "tool_calls": [], "finish_reason": "stop",
                "modelo": model}

    monkeypatch.setattr(g, "_llamar", fake)
    r = g.completar([{"role": "user", "content": "hola"}], [])
    assert r["content"] == "ok"
    assert llamados == ["principal", "respaldo"]
    assert g.modelo_usado == "respaldo"


def test_fallback_por_timeout(monkeypatch):
    """D5.2: un timeout también dispara el fallback."""
    g = _proveedor()
    llamados: list[str] = []

    def fake(model, *a, **k):
        llamados.append(model)
        if model == "principal":
            raise httpx.TimeoutException("timeout")
        return {"content": "ok", "tool_calls": [], "finish_reason": "stop",
                "modelo": model}

    monkeypatch.setattr(g, "_llamar", fake)
    r = g.completar([{"role": "user", "content": "hola"}], [])
    assert r["content"] == "ok"
    assert llamados == ["principal", "respaldo"]


def test_401_key_invalida_no_reintenta(monkeypatch):
    """D5.2: un 401 (key inválida) NUNCA se reintenta con el fallback."""
    g = _proveedor()
    llamados: list[str] = []

    def fake(model, *a, **k):
        llamados.append(model)
        raise _exc_estado(401)

    monkeypatch.setattr(g, "_llamar", fake)
    with pytest.raises(httpx.HTTPStatusError):
        g.completar([{"role": "user", "content": "hola"}], [])
    assert llamados == ["principal"]      # sólo el principal, sin reintento


def test_sin_fallback_configurado_propaga_error(monkeypatch):
    """Sin `LLM_MODEL_FALLBACK` el error original se propaga (no hay a dónde ir)."""
    g = GroqProveedor(api_key="k", model="principal", model_fallback="")
    monkeypatch.setattr(g, "_llamar", lambda *a, **k: (_ for _ in ()).throw(_exc_estado(500)))
    with pytest.raises(httpx.HTTPStatusError):
        g.completar([{"role": "user", "content": "hola"}], [])


# ---------------------------------------------------------------------------
# 3. Sólo content; reasoning descartado
# ---------------------------------------------------------------------------
def test_llamar_descarta_reasoning(monkeypatch):
    """D5.3: la respuesta expone sólo content/tool_calls/finish_reason (sin reasoning)."""
    g = _proveedor()

    class _Respuesta:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "hola", "reasoning": "pensamiento oculto",
                                             "tool_calls": []},
                                 "finish_reason": "stop"}]}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **k):
            return _Respuesta()

    monkeypatch.setattr(httpx, "Client", _Client)
    r = g._llamar("principal", [{"role": "user", "content": "hola"}], [], 900)
    assert r["content"] == "hola"
    assert "reasoning" not in r
    assert set(r) == {"content", "tool_calls", "finish_reason", "modelo"}


def test_tools_en_formato_openai(monkeypatch):
    """D5.5: los tools viajan en `payload['tools']` con `tool_choice='auto'`."""
    g = _proveedor()
    capturado: dict = {}

    class _Respuesta:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None, **k):
            capturado.update(json)
            return _Respuesta()

    monkeypatch.setattr(httpx, "Client", _Client)
    tools = [{"type": "function", "function": {"name": "ver_universo"}}]
    g._llamar("principal", [{"role": "user", "content": "hola"}], tools, 900)
    assert capturado["tools"] == tools
    assert capturado["tool_choice"] == "auto"
    assert capturado["model"] == "principal"


# ---------------------------------------------------------------------------
# 4. Content vacío / truncado → reintento con tokens dobles
# ---------------------------------------------------------------------------
def test_contenido_vacio_reintenta_con_mas_tokens():
    """D5.4: si la primera respuesta viene vacía, se reintenta con max_tokens dobles."""
    mock = MockProveedor(turnos=[
        {"content": "", "finish_reason": "length"},
        {"content": "respuesta completa", "finish_reason": "stop"},
    ])
    r = responder("hola", sid="t_d5_vacio", proveedor=mock)
    assert r["respuesta"] == "respuesta completa"
    # la segunda llamada debió pedir el doble de tokens
    assert mock.llamadas[1]["max_tokens"] == 2 * cfg.LLM_MAX_TOKENS


def test_contenido_truncado_persistente_da_error_controlado():
    """D5.4: si tras el reintento sigue truncado, devuelve mensaje controlado."""
    mock = MockProveedor(turnos=[
        {"content": "cortado", "finish_reason": "length"},
        {"content": "también cortado", "finish_reason": "length"},
    ])
    r = responder("hola", sid="t_d5_trunc", proveedor=mock)
    assert "No pude generar una respuesta completa" in r["respuesta"]


def test_respuesta_normal_usa_content():
    """Un turno normal (finish_reason='stop') no dispara reintento."""
    mock = MockProveedor(turnos=[{"content": "todo bien", "finish_reason": "stop"}])
    r = responder("hola", sid="t_d5_ok", proveedor=mock)
    assert r["respuesta"] == "todo bien"
    assert len(mock.llamadas) == 1


# ---------------------------------------------------------------------------
# casos_prueba.json (seguimiento multi-turno) — sólo su estructura (sin LLM)
# ---------------------------------------------------------------------------
def test_casos_prueba_existe_y_tiene_seguimiento():
    ruta = Path(__file__).parent / "casos_prueba.json"
    assert ruta.exists(), "falta tests/casos_prueba.json"
    casos = json.loads(ruta.read_text(encoding="utf-8"))
    assert len(casos) >= 3
    # los casos 2 y 3 (índice 1 y 2) son preguntas de seguimiento
    assert casos[1].get("seguimiento") is True
    assert casos[2].get("seguimiento") is True


# ---------------------------------------------------------------------------
# Re-ejecución de los casos de seguimiento con el LLM REAL (D5.7)
# ---------------------------------------------------------------------------
@pytest.mark.llm
def test_llm_real_casos_seguimiento(llm_disponible):
    """D5.7: casos 1-3 (2 y 3 = seguimiento) con el modelo configurado en .env.

    Valida que cada turno responda de forma no vacía/no truncada y respete la
    privacidad, usando la MISMA sesión para que los seguimientos tengan contexto.
    """
    if not llm_disponible:
        pytest.skip("LLM no configurado (.env)")
    from app.chatbot import chat

    ruta = Path(__file__).parent / "casos_prueba.json"
    casos = json.loads(ruta.read_text(encoding="utf-8"))
    sid = "t_llm_seguimiento"
    modelos: set[str] = set()
    for c in casos:
        r = chat(c["pregunta"], sid=sid)
        if r.get("modelo"):
            modelos.add(r["modelo"])
        assert r["respuesta"], f"caso {c['id']}: respuesta vacía"
        assert "No pude generar una respuesta completa" not in r["respuesta"], \
            f"caso {c['id']}: respuesta truncada/controlada"
        assert r["privacidad_ok"] is True, f"caso {c['id']}: privacidad violada"
    # se usó al menos un modelo configurado (principal o fallback)
    assert modelos, "no se registró el modelo usado"
