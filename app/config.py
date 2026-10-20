"""Fuente única de parámetros de negocio y modelado.

Todo parámetro que afecta resultados vive aquí. Las etiquetas
`calibrado`/`supuesto` alimentan /api/meta/supuestos.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

# Carga `.env` (LLM_API_KEY, LLM_MODEL, LLM_MODEL_FALLBACK, ciclo, etc.) desde la
# raíz de proyecto_back. No pisa variables ya exportadas en el shell (override=False).
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ---------------------------------------------------------------------------
# Ciclo (D2)
# ---------------------------------------------------------------------------
DURACION_CICLO_DIAS: int = int(os.getenv("DURACION_CICLO_DIAS", "30"))  # 15 o 30
if DURACION_CICLO_DIAS not in (15, 30):
    raise ValueError("DURACION_CICLO_DIAS debe ser 15 o 30")
N_QUINCENAS: int = DURACION_CICLO_DIAS // 15          # 2 o 1
DIAS_HABILES_QUINCENA: int = 13
DIAS_HABILES_CICLO: int = DIAS_HABILES_QUINCENA * N_QUINCENAS  # 26 o 13

# Calendario de referencia (heredado del EDA) para repartir FRE_REP/DIAS_REP en fechas
ANIO_REF: int = int(os.getenv("ANIO_REF", "2026"))
MES_REF: int = int(os.getenv("MES_REF", "3"))         # marzo

# ---------------------------------------------------------------------------
# Unidad de asesor (D1) — ZON_REP
# ---------------------------------------------------------------------------
UNIDAD_ASESOR: str = "zon_rep"
COL_ASESOR: str = "ZON_REP"
COL_EQUIPO: str = "Supervisor"          # nivel equipo, atributo del asesor

# ---------------------------------------------------------------------------
# Jornada, servicio y capacidad (D3)
# ---------------------------------------------------------------------------
JORNADA_HORAS: float = 8.0              # incluye viaje
T_SERV_MIN: float = 5.0                 # min
T_SERV_MOD: float = 10.0                # min (moda / media de la triangular)
T_SERV_MAX: float = 15.0                # min
T_SERV_MEDIO_MIN: float = (T_SERV_MIN + T_SERV_MOD + T_SERV_MAX) / 3.0   # 10.0
T_SERV_MEDIO_HORAS: float = T_SERV_MEDIO_MIN / 60.0
T_SERV_SENSIBILIDAD_MIN: float = 8.0    # corrida de sensibilidad con valor fijo
HORAS_VIAJE_RUTA_REF: float = 0.6       # h de viaje por ruta (referencia p/ CAP ~44)
CAP_TOPE: int = 45                      # tope duro de diseño
# Objetivo blando de carga (visitas/día)
OBJ_MIN: int = 40
OBJ_MAX: int = 50

# ---------------------------------------------------------------------------
# Costos (D3 · versión corregida) — "supuesto no calibrado"
# ---------------------------------------------------------------------------
COSTO_ASESOR_MES: float = 1_800_000.0   # COP/mes — supuesto, NO calibrado
COSTO_KM: float = 1_200.0               # COP/km combustible+mantenimiento
COSTO_HORA: float = 12_000.0            # COP/hora SOLO horas extra
HORA_EXTRA_FACTOR: float = 1.25         # factor sobre la hora extra
COSTO_VISITA_PERDIDA: float | None = None   # None → no se convierte a pesos
# Sensibilidad del costo fijo y del km (-30%, base, +30%)
SENSIBILIDAD_PCT: tuple[float, ...] = (-0.30, 0.0, 0.30)

# ---------------------------------------------------------------------------
# Incertidumbre (Monte Carlo)
# ---------------------------------------------------------------------------
SEMILLA: int = 2026
N_SIM: int = 1000
P_ASIST: float = 0.85
TRI_MIN, TRI_MOD, TRI_MAX = 0.7, 1.0, 1.4     # factor de tráfico
VEL_BASE: float = 30.0                        # km/h sin tráfico
N_RESERVA: int = 3000                         # columnas p/ clientes hipotéticos

# ---------------------------------------------------------------------------
# Ruteo
# ---------------------------------------------------------------------------
R_TIERRA: float = 6371.0
RUTA_CERRADA: bool = False             # D4: abierta por defecto
UMBRAL_GRUPO: int = 20                 # mínimo de clientes por grupo Asesor×CD (modelo)
UMBRAL_CD: int = 20                    # mínimo de clientes por CD (MonteCarlo original)
TEST_SIZE: float = 0.20
SEMILLA_MODELO: int = 42
N_SEEDS: int = 10
K_EXTRA_MAX: float = 1.0               # E4: ancla k = k_cap (antes 1.6)
MIN_CLIENTES_RUTA: int = 15            # E4: umbral de fusión de sub-rutas
BETA_SEEDS: int = 3                    # E3: semillas por beta
BETA_GRID: tuple[float, ...] = (1.0, 3.0, 5.0)

# ---------------------------------------------------------------------------
# Protocolo de evaluación (secciones 1-7): ajuste vs evaluación
# ---------------------------------------------------------------------------
# Partición 80/20 estratificada por ZON_REP (TEST_SIZE y SEMILLA_MODELO de arriba).
# El test queda SELLADO: se usa una sola vez, en la evaluación final.
K_PLIEGUES: int = 5                    # K-Fold sobre el 80 % de train
# λ del re-balanceo NORMALIZADO por la distancia típica (λ_norm·d_typ²), en vez
# de 1000/400 absolutos. Se elige por K-Fold; es una rejilla declarada de antemano.
LAM_GRID: tuple[float, ...] = (1.0, 3.0, 10.0)
LAM_FECHA_REL: float = 0.4             # λ_fecha = 0.4·λ (mismo peso relativo que E8)

# Rejilla de K declarada de antemano para la comparación de estrategias
# (K_ACTUAL −15, −10, −5 y K_ACTUAL). ACTUAL (58 zonas reales) es la línea base.
K_CANDIDATOS: tuple[int, ...] = (43, 48, 53, 58)

# Semillas: elección (SEMILLA_MODELO), reporte final y estabilidad son DISTINTAS.
SEMILLA_REPORTE: int = 2027            # resultados finales (≠ SEMILLA_MODELO=42, ≠ SEMILLA=2026)
SEMILLAS_ESTABILIDAD: tuple[int, ...] = (101, 202, 303)   # ranking de K (Spearman)

# NN vs ACO por ruta
ACO_SEEDS_RUTA: int = 5                # semillas ACO por ruta (mediana = valor ACO)
TOL_EMPATE_ACO: float = 0.1            # % de km: empate técnico NN vs ACO
RANGOS_TAMANO_AHORRO: tuple[tuple[int, int], ...] = ((1, 10), (11, 20),
                                                     (21, 30), (31, 45))
BOOTSTRAP_N: int = 10_000              # remuestreos del IC95 por ruta
BOOTSTRAP_SEMILLA: int = 777           # semilla fija del bootstrap (reproducible)
P_ASIST_EVAL: tuple[float, ...] = (0.70, 0.85, 1.00)   # escenarios de robustez
# Procesos para el experimento por ruta (0 = automático: todos los núcleos)
N_JOBS: int = int(os.getenv("N_JOBS", "0")) or (os.cpu_count() or 2)

# Política de sobrecarga (F3): "recortar" pierde las visitas que no caben en la
# jornada; "horas_extra" las atiende pagando hora extra.
POLITICA_SOBRECARGA: str = os.getenv("POLITICA_SOBRECARGA", "recortar")
if POLITICA_SOBRECARGA not in ("recortar", "horas_extra"):
    raise ValueError("POLITICA_SOBRECARGA debe ser 'recortar' u 'horas_extra'")

# F2 — población sobre la que se calcula el re-balanceo desde ACTUAL.
# "todos" (decisión registrada explícitamente): el re-balanceo es una REPARACIÓN
# ESTRUCTURAL que usa sólo atributos de entrada (ZON_REP, DIAS_REP, FRE_REP,
# visitas y coordenadas) de todos los clientes; no usa resultados ni métricas de
# test. Sin esto, en el 80 % de train no hay ningún día sobre el tope (pico 38)
# y el re-balanceo no movería a nadie mientras ACTUAL sí tiene días sobre el
# tope con la población completa (pico 47). El test sigue sellado para β/K/λ,
# para las métricas de inserción y para la generalización.
# "train" aplica el protocolo estricto (resultado: 0 movidos).
REBAL_POBLACION: str = os.getenv("REBAL_POBLACION", "todos")
if REBAL_POBLACION not in ("todos", "train"):
    raise ValueError("REBAL_POBLACION debe ser 'todos' o 'train'")

# Umbrales de la regla de decisión. ETIQUETA: "propuesta, no calibrada".
UMBRALES_DECISION: dict[str, float] = {
    "ahorro_aco_pct": 2.0,        # ahorro mínimo de ACO sobre NN
    "tiempo_ciclo_min": 30.0,     # cómputo de todo el ciclo (min)
    "cobertura_pp": 0.5,          # pérdida máxima de cobertura vs ACTUAL (puntos)
    "km_extra_test_pct": 5.0,     # km extra al insertar clientes de test (%)
    "arrepentimiento_pct": 5.0,   # arrepentimiento medio (%)
    "ratio_km_vs_actual": 1.25,   # km de la estrategia <= 1.25 × ACTUAL
    "estabilidad_semillas": 2,    # misma K (o ±1) en 2 de 3 semillas
}

# Parámetros ACO (heredados del notebook Modelo, sección 2)
ACO_M: int = 12                        # hormigas por iteración
ACO_ALFA: float = 1.0                  # peso de la feromona
ACO_BETA: float = 3.0                  # peso de la heurística (1/distancia)
ACO_RHO: float = 0.30                  # evaporación de feromona
ACO_Q: float = 1.0                     # depósito de feromona
ACO_ITER: int = 40                     # iteraciones (corrida completa)
ACO_SEED_M: int = 8                    # hormigas (experimento multi-semilla)
ACO_SEED_ITER: int = 25                # iteraciones (experimento multi-semilla)

# Ventanas horarias por categoría
VENTANAS: dict[str, set[int]] = {
    "Tiendas": {0}, "Bodegas": {0}, "Entretenimiento": {2}, "Otros": {0, 1, 2},
}

# ---------------------------------------------------------------------------
# Rango K del Monte Carlo (P2)
# ---------------------------------------------------------------------------
K_MIN: int = 35
K_MAX: int = 70
K_PASO: int = 1
K_ACTUAL: int = 58                    # nº de zonas ACTUAL (referencia fija)

# Escenarios de balanceo (P1)
ESCENARIO_DIAS: str = "fijo"          # "fijo" (A) | "reprogramado" (B)

# ---------------------------------------------------------------------------
# Chatbot (D5 · Groq) — nunca versionar la key
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
# Los modelos se leen SOLO desde .env; el código no escribe ningún nombre fijo (D5).
LLM_MODEL: str = os.getenv("LLM_MODEL", "")
LLM_MODEL_FALLBACK: str = os.getenv("LLM_MODEL_FALLBACK", "")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
LLM_TIMEOUT_S: float = float(os.getenv("LLM_TIMEOUT_S", "60"))
CHAT_MAX_TOOL_ITER: int = int(os.getenv("CHAT_MAX_TOOL_ITER", "5"))
CHAT_RATE_LIMIT: str = "20/minute"

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
BASE_CSV: str = "base_clientes_limpia.csv"
FRONTEND_ORIGINS: tuple[str, ...] = (
    "http://localhost:4200", "http://127.0.0.1:4200",
)
ESPERA_SEGMENTOS_S: float = 60.0      # timeout de escenarios


def costo_fijo_ciclo() -> float:
    """Costo fijo de asesores por ciclo completo (D3.1)."""
    return COSTO_ASESOR_MES * (DURACION_CICLO_DIAS / 30.0)


def cap_dinamico(horas_viaje_ruta: float = HORAS_VIAJE_RUTA_REF,
                 t_serv_min: float | None = None) -> int:
    """CAP dinámico de una ruta (D3). Con defaults ≈ 44."""
    t = (T_SERV_MEDIO_MIN if t_serv_min is None else t_serv_min) / 60.0
    return int((JORNADA_HORAS - horas_viaje_ruta) // t)


def params() -> dict[str, Any]:
    """Todos los parámetros con etiqueta supuesto/calibrado (p/ /api/meta/supuestos)."""
    base: dict[str, tuple[Any, str, str]] = {
        "UNIDAD_ASESOR": (UNIDAD_ASESOR, "calibrado", "Unidad de asesor = ZON_REP (D1)"),
        "DURACION_CICLO_DIAS": (DURACION_CICLO_DIAS, "calibrado", "15 o 30 (D2)"),
        "N_QUINCENAS": (N_QUINCENAS, "calibrado", "Guías Q1/Q2 por ciclo"),
        "DIAS_HABILES_QUINCENA": (DIAS_HABILES_QUINCENA, "supuesto", "13 días hábiles/quincena"),
        "DIAS_HABILES_CICLO": (DIAS_HABILES_CICLO, "calibrado", "=13×N_QUINCENAS"),
        "ANIO_REF": (ANIO_REF, "calibrado", "Calendario de referencia"),
        "MES_REF": (MES_REF, "calibrado", "Calendario de referencia"),
        "JORNADA_HORAS": (JORNADA_HORAS, "supuesto", "Jornada diaria, incluye viaje"),
        "T_SERV_MEDIO_MIN": (T_SERV_MEDIO_MIN, "supuesto", "Triangular(5,10,15)"),
        "T_SERV_SENSIBILIDAD_MIN": (T_SERV_SENSIBILIDAD_MIN, "supuesto", "Sensibilidad fijo 8 min"),
        "CAP_TOPE": (CAP_TOPE, "supuesto", "Tope duro de diseño"),
        "CAP_DINAMICO_REF": (cap_dinamico(), "supuesto", "=floor((8−0,6)/10min)"),
        "OBJ_MIN": (OBJ_MIN, "supuesto", "Objetivo blando de carga (visitas/día)"),
        "OBJ_MAX": (OBJ_MAX, "supuesto", "Objetivo blando de carga (visitas/día)"),
        "COSTO_ASESOR_MES": (COSTO_ASESOR_MES, "supuesto", "COP/mes — NO calibrado (qué incluye)"),
        "costo_fijo_ciclo": (costo_fijo_ciclo(), "supuesto", "×duración/30"),
        "COSTO_KM": (COSTO_KM, "supuesto", "COP/km — NO calibrado"),
        "COSTO_HORA": (COSTO_HORA, "supuesto", "COP/hora SOLO extra"),
        "HORA_EXTRA_FACTOR": (HORA_EXTRA_FACTOR, "supuesto", "Factor hora extra"),
        "COSTO_VISITA_PERDIDA": (COSTO_VISITA_PERDIDA, "supuesto", "None → sin conversión a pesos"),
        "P_ASIST": (P_ASIST, "supuesto", "Asistencia 85 %"),
        "VEL_BASE": (VEL_BASE, "supuesto", "km/h sin tráfico"),
        "N_SIM": (N_SIM, "calibrado", "Escenarios por estrategia"),
        "SEMILLA": (SEMILLA, "calibrado", "Semilla maestra"),
        "RUTA_CERRADA": (RUTA_CERRADA, "calibrado", "D4: abierta"),
        "K_MIN": (K_MIN, "calibrado", "Rango K (P2)"),
        "K_MAX": (K_MAX, "calibrado", "Rango K (P2)"),
        "K_ACTUAL": (K_ACTUAL, "calibrado", "Zonas ACTUAL (referencia)"),
        "ESCENARIO_DIAS": (ESCENARIO_DIAS, "calibrado", "A=fijo, B=reprogramado"),
        "ACO_ALFA": (ACO_ALFA, "supuesto", "Peso de feromona"),
        "ACO_BETA": (ACO_BETA, "supuesto", "Peso de heurística (E3 calibra)"),
        "ACO_RHO": (ACO_RHO, "supuesto", "Evaporación de feromona"),
        "K_EXTRA_MAX": (K_EXTRA_MAX, "calibrado", "E4: ancla k=k_cap"),
        "MIN_CLIENTES_RUTA": (MIN_CLIENTES_RUTA, "calibrado", "E4: fusión sub-rutas"),
        "TEST_SIZE": (TEST_SIZE, "calibrado", "Fracción de clientes de test (20 %)"),
        "SEMILLA_MODELO": (SEMILLA_MODELO, "calibrado", "Semilla de la partición train/test"),
        "K_PLIEGUES": (K_PLIEGUES, "calibrado", "K-Fold sobre el 80 % de train"),
        "K_CANDIDATOS": (list(K_CANDIDATOS), "calibrado", "Rejilla de K declarada de antemano"),
        "LAM_GRID": (list(LAM_GRID), "supuesto", "λ normalizado (× d_typ²); elegido por K-Fold"),
        "POLITICA_SOBRECARGA": (POLITICA_SOBRECARGA, "supuesto",
                                "recortar pierde visitas; horas_extra las paga"),
        "REBAL_POBLACION": (REBAL_POBLACION, "supuesto",
                            "Re-balanceo F2 calculado sobre todos los clientes "
                            "(sólo atributos de entrada, sin resultados de test)"),
        "ACO_SEEDS_RUTA": (ACO_SEEDS_RUTA, "calibrado", "Semillas ACO por ruta (mediana)"),
        "N_JOBS": (N_JOBS, "operativo", "Procesos del experimento por ruta"),
        "BOOTSTRAP_N": (BOOTSTRAP_N, "calibrado", "Remuestreos del IC95 por ruta"),
        "SEMILLA_REPORTE": (SEMILLA_REPORTE, "calibrado", "Semilla de los resultados finales"),
        "UMBRALES_DECISION": (UMBRALES_DECISION, "propuesta, no calibrada",
                              "Umbrales de la regla de decisión: se declararon ANTES "
                              "de ver los resultados y NO se calibraron"),
    }
    return {k: {"valor": v, "etiqueta": e, "nota": n} for k, (v, e, n) in base.items()}
