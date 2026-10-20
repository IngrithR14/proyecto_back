"""Sesiones del chatbot en memoria (FASE 4). TTL + límite de mensajes."""
from __future__ import annotations

import threading
import time
from typing import Any

TTL_S = 3600.0          # 1 h
MAX_MENSAJES = 40       # por sesión


class Sesion:
    def __init__(self, sid: str):
        self.sid = sid
        self.creada = time.time()
        self.activa = time.time()
        self.historial: list[dict[str, str]] = []   # {role, content}
        self.trazas: list[dict[str, Any]] = []      # llamadas a herramientas

    def tocar(self) -> None:
        self.activa = time.time()

    def agregar(self, role: str, content: str) -> None:
        self.historial.append({"role": role, "content": content})
        if len(self.historial) > MAX_MENSAJES:
            self.historial = self.historial[-MAX_MENSAJES:]


class _Almacen:
    def __init__(self):
        self._d: dict[str, Sesion] = {}
        self._lock = threading.Lock()

    def obtener(self, sid: str) -> Sesion:
        with self._lock:
            self._podar()
            if sid not in self._d:
                self._d[sid] = Sesion(sid)
            s = self._d[sid]
            s.tocar()
            return s

    def _podar(self) -> None:
        ahora = time.time()
        caducos = [k for k, s in self._d.items() if ahora - s.activa > TTL_S]
        for k in caducos:
            del self._d[k]

    def reiniciar(self, sid: str) -> Sesion:
        with self._lock:
            self._d[sid] = Sesion(sid)
            return self._d[sid]

    def contar(self) -> int:
        with self._lock:
            self._podar()
            return len(self._d)


ALMACEN = _Almacen()
