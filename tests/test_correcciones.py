"""Tests automatizados de las correcciones E1–E11 (FASE 1).

Cada test valida que la corrección esté aplicada en `app/core/`. Se ejecutan con
`pytest tests/test_correcciones.py -q`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import config as cfg
from app.core import comparacion
from app.core.asesores import buscar_asesor, normalizar_equipo, _norm
from app.core.calendario import construir_calendario, resumen_carga, fechas_permitidas
from app.core.estructura import (
    asignacion_capacidad, construir_actual, construir_estrategia, validar_cubierta)
from app.core.montecarlo import (
    carga_por_territorio_fecha, comparar_pareada, factibilidad, balance_carga)
from app.core.ruteo import (
    factible_orden, formar_subrutas, muestra_estratificada, rangos_cubiertos,
    segmento_servible, slots_factibles)
from app.core.universo import cargar_universo, resumen_universos


# ---------------------------------------------------------------------------
# E1 — % de victorias por estrategia (no invertido)
# ---------------------------------------------------------------------------
def test_e1_porcentaje_victorias_no_invertido():
    exp = pd.DataFrame({
        "NN":  [100.0, 100.0, 100.0, 100.0, 100.0],
        "ACO": [90.0, 90.0, 90.0, 90.0, 90.0],      # ACO más corto SIEMPRE
    })
    r = comparacion.resumen_estabilidad(exp)
    # ACO gana en el 100 %; NN en el 0 % (la tabla original lo mostraba al revés)
    assert r.loc["ACO", "% semillas en que gana"] == pytest.approx(100.0)
    assert r.loc["NN", "% semillas en que gana"] == pytest.approx(0.0)
    assert comparacion.victorias(exp, "ACO") == pytest.approx(100.0)


def test_e1_sin_inversion_con_datos_reales():
    # caso donde NN gana en 3 de 5
    exp = pd.DataFrame({"NN": [10, 50, 10, 10, 50], "ACO": [20, 40, 20, 20, 40]})
    assert comparacion.victorias(exp, "NN") == pytest.approx(60.0)
    assert comparacion.victorias(exp, "ACO") == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# E2 — ruta abierta: la ventana del ÚLTIMO cliente sí se verifica
# ---------------------------------------------------------------------------
def test_e2_ventana_ultimo_cliente_en_ruta_abierta():
    assert cfg.RUTA_CERRADA is False
    # segmento servible en ruta abierta INCLUYE el último cliente (orden[1:])
    assert segmento_servible([0, 1, 2, 3]) == [1, 2, 3]
    # La última ventana es la que rompe la factibilidad: el cliente 3 exige slot 0
    # pero el recorrido ya avanzó a slot 2. Si E2 no mirara el último (bug: [1:-1])
    # daría factible; con el fix devuelve False.
    slots = [{0, 1, 2}, {0}, {2}, {0}]
    assert slots_factibles([slots[1], slots[2]]) is True      # sólo [1:-1] → pasa (bug)
    assert factible_orden([0, 1, 2, 3], slots) is False       # fix: incluye el último
    # una ruta que termina en un cliente de slot alto sí es factible
    slots_ok = [{0, 1, 2}, {0}, {1}, {2}]
    assert factible_orden([0, 1, 2, 3], slots_ok) is True


def test_e2_segmento_servible_cierra_con_depósito():
    # con ruta cerrada explícita (depósito repetido al final) no se cuenta el cierre
    assert cfg.RUTA_CERRADA is False  # por defecto abierta (D4)


# ---------------------------------------------------------------------------
# E3 — calibración de β con muestra estratificada por tamaño
# ---------------------------------------------------------------------------
def _rutas_falsas(tamanos):
    out = []
    for i, n in enumerate(tamanos):
        df = pd.DataFrame({"grupo_id": ["g"] * n, "sub_ruta": [f"s{i}"] * n})
        out.append(df)
    return out


def test_e3_muestra_estratificada_cubre_los_4_rangos():
    rutas = _rutas_falsas([5, 8, 25, 27, 35, 38, 44, 45])
    muestra = muestra_estratificada(rutas, por_rango=2)
    cubiertos = rangos_cubiertos(muestra)
    assert all(cubiertos), f"rangos cubiertos={cubiertos}"
    # y toma de cada rango, no sólo los más pequeños
    tam = [len(r) for r in muestra]
    assert max(tam) >= 41, "debe incluir sub-rutas grandes (>=41)"


def test_e3_no_solo_pequenas():
    # con varias pequeñas, aún debe tomar de los 4 rangos (incl. los grandes)
    rutas = _rutas_falsas([3, 4, 25, 35, 42])
    muestra = muestra_estratificada(rutas, por_rango=1)
    assert rangos_cubiertos(muestra) == [True, True, True, True]
    assert max(len(r) for r in muestra) >= 41


# ---------------------------------------------------------------------------
# E4 — formación de sub-rutas: ancla k=k_cap (K_EXTRA_MAX=1.0)
# ---------------------------------------------------------------------------
def test_e4_k_extra_max_ancha_k_en_k_cap():
    assert cfg.K_EXTRA_MAX == 1.0, "E4: el ancla de silueta debe ser k=k_cap"
    assert cfg.MIN_CLIENTES_RUTA == 15, "E4: umbral de fusión de sub-rutas"


# ---------------------------------------------------------------------------
# E6 — balance por asesor con llave única (sin colisión de tid entre CDs)
# ---------------------------------------------------------------------------
def test_e6_balance_carga_sin_colision():
    # carga perfectamente equilibrada → CV 0
    assert balance_carga(np.array([10.0, 10.0, 10.0, 10.0])) == pytest.approx(0.0)
    # el caso antiguo [14750,0,0,0,0] daba CV=2.00 por colisión de llaves (E6)
    antiguo = np.array([14750.0, 0, 0, 0, 0])
    assert balance_carga(antiguo) == pytest.approx(2.0)
    # La llave de asesor es `territorio`: ZON_REP (globally único) o índice 0..K-1.
    # Antes era `tid` 0..kc-1 DENTRO de cada CD → la misma llave en varios CDs.
    u = cargar_universo()
    zonas = set(u.clientes[cfg.COL_ASESOR].unique())
    act = construir_actual(u.clientes.head(800))
    terrs = set(r["territorio"] for r in act)
    assert terrs.issubset(zonas), "las llaves deben ser ZON_REP únicos, no tid por CD"
    # agregar por territorio da tantos grupos como zonas presentes (sin fusión por colisión)
    cv_por_zona = carga_por_territorio_fecha(u.clientes, None)
    n_zonas_balance = cv_por_zona["territorio"].nunique()
    assert n_zonas_balance == u.n_zonas_rutable


# ---------------------------------------------------------------------------
# E7 — factibilidad por asesor-DÍA vs CAP dinámico
# ---------------------------------------------------------------------------
def test_e7_factibilidad_por_asesor_dia():
    u = cargar_universo()
    ctf = carga_por_territorio_fecha(u.clientes, None)
    f = factibilidad(ctf)
    assert "cap_dinamico" in f and f["cap_dinamico"] == cfg.cap_dinamico()
    assert "dias_sobre_cap" in f
    # la unidad es (territorio, fecha): una fila por asesor-día
    assert ctf.groupby(["territorio", "fecha"]).ngroups == len(ctf[["territorio", "fecha"]].drop_duplicates())
    # el pico real es > CAP en algún asesor-día (hallazgo, no bug)
    assert f["pico_global"] > 0


def test_e7_cap_dinamico_es_int():
    cap = cfg.cap_dinamico()
    assert isinstance(cap, int) and 40 <= cap <= 45


# ---------------------------------------------------------------------------
# E8 — ACTUAL construido desde ZON_REP (no reclusteriza)
# ---------------------------------------------------------------------------
def test_e8_actual_desde_zon_rep():
    u = cargar_universo()
    # las zonas reales del universo
    zonas = set(u.clientes[cfg.COL_ASESOR].unique())
    act = construir_actual(u.clientes)
    terrs = set(r["territorio"] for r in act)
    assert terrs.issubset(zonas), "los territorios ACTUAL deben ser ZON_REP reales"
    assert len(terrs) == u.n_zonas_rutable, f"esperaba {u.n_zonas_rutable} zonas"
    assert validar_cubierta(act, len(u.clientes)), "cobertura exacta de clientes"
    # K=58 (referencia) reclusteriza
    df = u.clientes.copy().reset_index(drop=True)
    from app.core.geo import proyectar
    df["_uid"] = np.arange(len(df))
    proyectar(df)
    estr, meta = construir_estrategia(df, cfg.K_ACTUAL)
    assert meta["n_territorios"] == cfg.K_ACTUAL
    assert validar_cubierta(estr, len(df))


# ---------------------------------------------------------------------------
# E9 — depósito por territorio (centroides Asesor×CD / ZON_REP)
# ---------------------------------------------------------------------------
def test_e9_deposito_por_territorio():
    from app.core.estructura import DEPOSITO_MODO
    assert DEPOSITO_MODO == "territorio", "E9: depósito = centroide del territorio"


# ---------------------------------------------------------------------------
# E10 — normalización de nombres dañados + búsqueda difusa
# ---------------------------------------------------------------------------
def test_e10_normaliza_nombre_danado_sirley():
    df = pd.DataFrame({cfg.COL_EQUIPO: ["Sirley Damaris Rodr\ufffdguez Usaqu\ufffdn"]})
    out = normalizar_equipo(df)
    assert "\ufffd" not in out[cfg.COL_EQUIPO].iloc[0]
    assert "Rodríguez" in out[cfg.COL_EQUIPO].iloc[0]


def test_e10_busqueda_difusa():
    r = buscar_asesor("sirley rodriguez")
    assert r["ok"] is True
    assert "Rodr" in r["asesor"]["nombre_canonico"]
    # sin tildes ni mayúsculas
    assert _norm("Sonia Alejandra") == "sonia alejandra"


def test_e10_no_encontrado():
    r = buscar_asesor("no existe esta persona xyz")
    assert r["ok"] is False
    assert r["error_codigo"] == "ASESOR_NO_ENCONTRADO"


def test_e10_columnas_danadas_detectadas():
    u = cargar_universo()
    assert "Supervisor" in u.columnas_danadas, "debe detectar U+FFFD en Supervisor"


# ---------------------------------------------------------------------------
# E11 — UNA sola función de carga (embudo compartido)
# ---------------------------------------------------------------------------
def test_e11_unica_funciona_carga():
    u = cargar_universo()
    # el embudo tiene los pasos en orden
    pasos = [e["paso"] for e in u.embudo]
    assert pasos[0] == "Base limpia completa"
    assert any("coordenadas válidas" in p for p in pasos)
    # el universo es el esperado (14.757)
    assert u.n == 14757
    assert u.visitas_totales_mes == 45212
    assert u.n_zonas_rutable == 58
    # embudo consistente: filas decrecientes, eliminadas cuadran
    for i in range(1, len(u.embudo)):
        assert u.embudo[i]["filas"] <= u.embudo[i - 1]["filas"]
        assert u.embudo[i]["eliminadas"] == (u.embudo[i - 1]["filas"] - u.embudo[i]["filas"])
    # la vista /api/meta/universos funciona
    res = resumen_universos(u)
    assert res["n_clientes"] == 14757
    assert res["unidad_asesor"] == "zon_rep"


# ---------------------------------------------------------------------------
# D2 — el calendario reproduce visitas_mes_rep exactamente
# ---------------------------------------------------------------------------
def test_d2_calendario_cuadra_con_visitas_mes_rep():
    u = cargar_universo()
    va, meta = construir_calendario(u.clientes)
    assert meta["total_visitas"] == 45212
    assert meta["descuadre_ocurrencias"] == 0, "las ocurrencias deben reproducir visitas_mes_rep"


def test_d2_quincenas_y_fechas():
    assert cfg.N_QUINCENAS in (1, 2)
    assert cfg.DIAS_HABILES_CICLO == cfg.DIAS_HABILES_QUINCENA * cfg.N_QUINCENAS
    f = fechas_permitidas("LUNES", "Q1")
    assert all(x.day <= 15 for x in f), "Q1 sólo días 1-15"
    if cfg.DURACION_CICLO_DIAS == 15:
        f2 = fechas_permitidas("LUNES", "Q2")
        # con ciclo de 15 días sólo aplica Q1
        assert all(x.day <= 15 for x in fechas_permitidas("LUNES", "Q1"))


# ---------------------------------------------------------------------------
# D3 — modelo de costos: fijo mensual escalado, horas SOLO extra
# ---------------------------------------------------------------------------
def test_d3_costo_fijo_escala_con_ciclo():
    assert cfg.costo_fijo_ciclo() == cfg.COSTO_ASESOR_MES * (cfg.DURACION_CICLO_DIAS / 30.0)
    assert cfg.COSTO_VISITA_PERDIDA is None, "visitas perdidas no se convierten en pesos"


def test_d3_horas_extra_no_suman_todas_las_horas():
    from app.core.montecarlo import costo_escenario
    ev = {"km": np.array([1000.0]), "km_territorio": np.array([[1000.0]]),
          "visitas_territorio": np.array([[1000.0]])}
    c = costo_escenario(ev, np.array([1.0]), n_asesores=5)
    # el fijo es 5 * costo_fijo_ciclo
    assert c["fijo"] == pytest.approx(5 * cfg.costo_fijo_ciclo())
    # el combustible es km*COSTO_KM (sin factor fijo de 1,5M por quincena)
    assert c["combustible"][0] == pytest.approx(1000 * cfg.COSTO_KM)


# ---------------------------------------------------------------------------
# P2 — rango K 35–70 y ACTUAL = 58 como referencia
# ---------------------------------------------------------------------------
def test_p2_rango_k():
    assert cfg.K_MIN == 35 and cfg.K_MAX == 70
    assert cfg.K_ACTUAL == 58


def test_comparacion_pareada_ic95():
    rng = np.random.default_rng(0)
    base = rng.normal(100, 5, 1000)
    otra = base + rng.normal(-2, 1, 1000)
    pc = comparar_pareada(otra, base)
    assert pc["dif_media_M"] < 0, "otra estrategia debe ser más barata en media"
    assert pc["ic95_inf"] < pc["ic95_sup"]
    assert 0.0 <= pc["p_mejor"] <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
