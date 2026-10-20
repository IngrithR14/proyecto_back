"""Orquestador del chatbot (FASE 4): bucle LLM ↔ herramientas con proveedor intercambiable.

`GroqProveedor` llama al API real (OpenAI-compatible). `MockProveedor` simula
respuestas para pruebas sin red ni API key (FASE 6). El bucle ejecuta hasta
`CHAT_MAX_TOOL_ITER` rondas de herramientas antes de responder.

Ajuste D5: los modelos (`LLM_MODEL`, `LLM_MODEL_FALLBACK`) se leen SOLO desde
.env; si el principal falla (429/5xx/timeout/modelo no disponible) se reintenta
una vez con el fallback y se registra en el log. Un 401 (key inválida) NO se
reintenta. Sólo se usa `message.content`; cualquier campo `reasoning` se ignora.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol

import httpx

from .. import config as cfg
from . import herramientas as H
from .prompts import SYSTEM_PROMPT
from .sesiones import ALMACEN

log = logging.getLogger("app.chatbot")


class Proveedor(Protocol):
    def completar(self, mensajes: list[dict], tools: list[dict],
                  max_tokens: int | None = None) -> dict:
        """Devuelve {tool_calls:[...]} o {content:str} (formato OpenAI simplificado)."""
        ...


class GroqProveedor:
    """Cliente Groq (OpenAI-compatible). La key NUNCA se versiona; va en .env."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, model_fallback: str | None = None,
                 timeout: float | None = None):
        self.base_url = (base_url or cfg.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or cfg.LLM_API_KEY
        self.model = model or cfg.LLM_MODEL
        self.model_fallback = model_fallback or cfg.LLM_MODEL_FALLBACK
        self.timeout = timeout or cfg.LLM_TIMEOUT_S
        self.modelo_usado: str = self.model   # se actualiza si se usa el fallback

    def disponible(self) -> bool:
        return bool(self.api_key and self.model)

    def _llamar(self, model: str, mensajes: list[dict], tools: list[dict],
                max_tokens: int) -> dict:
        """Una llamada al API; devuelve sólo content/tool_calls/finish_reason.

        El campo `reasoning` (que algunos modelos devuelven) se DESCARTA: no se
        guarda en el historial ni se muestra al usuario (D5.3).
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": mensajes,
            "temperature": cfg.LLM_TEMPERATURE,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools          # formato OpenAI (D5.5)
            payload["tool_choice"] = "auto"
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(f"{self.base_url}/chat/completions",
                       headers={"Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"},
                       json=payload)
            r.raise_for_status()
            data = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        return {
            "content": msg.get("content"),
            "tool_calls": msg.get("tool_calls") or [],
            "finish_reason": choice.get("finish_reason"),
            "modelo": model,
        }

    @staticmethod
    def _reintentable(exc: Exception) -> bool:
        """¿Vale la pena reintentar con el fallback? El 401 (key) NUNCA (D5.2)."""
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code == 401:                       # key inválida: no reintenta
                return False
            return code == 429 or code >= 500 or code == 404   # rate/ servidor/ modelo
        # timeout o error de red
        return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))

    def completar(self, mensajes: list[dict], tools: list[dict],
                  max_tokens: int | None = None) -> dict:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY no configurada (.env). Usa MockProveedor en pruebas.")
        if not self.model:
            raise RuntimeError("LLM_MODEL no definido en .env.")
        max_tokens = max_tokens or cfg.LLM_MAX_TOKENS
        try:
            r = self._llamar(self.model, mensajes, tools, max_tokens)
            self.modelo_usado = self.model
            return r
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as e:
            if not self._reintentable(e) or not self.model_fallback:
                raise
            log.warning("Modelo principal (%s) falló (%s); reintento con fallback",
                        self.model, e)
            r = self._llamar(self.model_fallback, mensajes, tools, max_tokens)
            self.modelo_usado = self.model_fallback
            log.info("Respuesta atendida por el modelo fallback: %s", self.model_fallback)
            return r


class MockProveedor:
    """Simula el LLM con una secuencia guionizada de turnos (para pruebas).

    Cada turno es {"tool_calls":[{function:{name, arguments}}]} o {"content": "..."}.
    Si se acaban los turnos, responde un contenido fijo.
    """

    def __init__(self, turnos: list[dict] | None = None, contenido_final: str = "(mock)"):
        self.turnos = list(turnos or [])
        self.contenido_final = contenido_final
        self.llamadas: list[dict] = []
        self.modelo_usado = "mock"

    def completar(self, mensajes: list[dict], tools: list[dict],
                  max_tokens: int | None = None) -> dict:
        self.llamadas.append({"n_mensajes": len(mensajes), "n_tools": len(tools),
                              "max_tokens": max_tokens})
        if self.turnos:
            t = self.turnos.pop(0)
            return {**t, "finish_reason": t.get("finish_reason")}
        return {"content": self.contenido_final, "tool_calls": [],
                "finish_reason": "stop"}


# ---------------------------------------------------------------------------
# Bucle orquestador
# ---------------------------------------------------------------------------
def _parse_args(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return {}


def ejecutar_tool_call(tc: dict) -> dict:
    """Ejecuta un tool_call (formato OpenAI) y devuelve la salida como dict."""
    fn = tc.get("function", {})
    nombre = fn.get("name", "")
    args = _parse_args(fn.get("arguments"))
    return H.ejecutar(nombre, args)


def responder(pregunta: str, sid: str = "default",
              proveedor: Proveedor | None = None) -> dict:
    """Bucle completo: pregunta → (herramientas)* → respuesta final.

    Devuelve {respuesta, herramientas:[{nombre,entrada,salida}], sesion}.
    """
    proveedor = proveedor or GroqProveedor()
    sesion = ALMACEN.obtener(sid)
    sesion.agregar("user", pregunta)

    mensajes: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    mensajes += sesion.historial
    tools = H.schemas_tool()

    trazas: list[dict] = []
    respuesta_final = None
    for _ in range(cfg.CHAT_MAX_TOOL_ITER):
        r = proveedor.completar(mensajes, tools)
        tool_calls = r.get("tool_calls") or []
        if not tool_calls:
            contenido = r.get("content") or ""
            truncado = r.get("finish_reason") == "length"
            # D5.4: content vacío o truncado (finish_reason="length") no es válido
            if (not contenido.strip() or truncado):
                r2 = proveedor.completar(mensajes, tools,
                                         max_tokens=2 * cfg.LLM_MAX_TOKENS)
                c2 = r2.get("content") or ""
                if c2.strip() and r2.get("finish_reason") != "length":
                    contenido = c2
                    truncado = False
            if not contenido.strip() or truncado:
                respuesta_final = ("No pude generar una respuesta completa. "
                                   "Reformula la pregunta.")
            else:
                respuesta_final = contenido
            break
        # registrar mensaje del asistente con tool_calls (sin reasoning)
        mensajes.append({"role": "assistant", "content": r.get("content"),
                         "tool_calls": tool_calls})
        for tc in tool_calls:
            nombre = tc.get("function", {}).get("name", "?")
            args = _parse_args(tc.get("function", {}).get("arguments"))
            salida = H.ejecutar(nombre, args)
            trazas.append({"nombre": nombre, "entrada": args, "salida": salida})
            mensajes.append({
                "role": "tool",
                "tool_call_id": tc.get("id", nombre),
                "content": json.dumps(salida, ensure_ascii=False, default=str),
            })
    if respuesta_final is None:
        respuesta_final = ("No pude completar la respuesta (límite de herramientas "
                           "alcanzado). Reformula la pregunta.")

    sesion.agregar("assistant", respuesta_final)
    sesion.trazas.extend(trazas)
    return {"respuesta": respuesta_final, "herramientas": trazas, "sesion": sid,
            "modelo": getattr(proveedor, "modelo_usado", None)}
