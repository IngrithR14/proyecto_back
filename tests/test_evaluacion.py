"""Pruebas del protocolo de evaluación y de las métricas de decisión (secciones 1-7).

Cubren lo exigido en la sección de pruebas:
  - FUGA: el test sellado (20 % estratificado por ZON_REP) no entra a `ajustar()`,
    los depósitos salen sólo del train, y la ÚNICA excepción declarada (re-balanceo
    F2 con `REBAL_POBLACION="todos"`) queda visible en el reporte.
  - RUTAS SINTÉTICAS: con rutas donde ACO gana de verdad la regla adopta "aco";
    si ACO empata con NN la regla lo rechaza ("nn"); y si sólo gana en un rango de
    tamaño de ruta la decisión es "mixta".
  - BOOTSTRAP reproducible con su semilla fija (y distinta semilla → distinto IC95).
  - P_ASIST = 1,00 ⇒ cobertura de ACTUAL < 1 (se pierden visitas por SOBRECARGA,
    no por ausencias).
  - Regla de estrategias: si ninguna candidata cumple las 6 condiciones →
    "ninguna estrategia cumple las condiciones" (sin tocar umbrales).
  - Los tres endpoints nuevos responden y con la forma esperada.

Todos los números salen del núcleo; nada está pintado a mano.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app import config as cfg
from app.core import comparacion_rutas as cr
from app.core import evaluacion as ev_mod


# ---------------------------------------------------------------------------
# Fixtures (baratas y por sesión/módulo)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def particion():
    """Partición real 80/20 estratificada (la MISMA que usa el protocolo)."""
    from app.precompute import _particion
    return _particion()


@pytest.fixture(scope="module")
def muestra():
    """Submuestra barata (1.500 train / 300 test) para pruebas que ajustan modelos."""
    tr, te, _ = _particion()
    return (tr.sample(1500, random_state=7).reset_index(drop=True),
            te.sample(300, random_state=7).reset_index(drop=True))


def _particion():
    from app.precompute import _particion as p
    return p()


# ---------------------------------------------------------------------------
# Rutas sintéticas (anello: un orden bueno y un orden en zigzag)
# ---------------------------------------------------------------------------
def _anillo(nc: int = 10):
    """Depósito + nc clientes en un anillo: devuelve geometría y dos órdenes."""
    ang = 2 * math.pi * np.arange(1, nc + 1) / nc
    R = 0.05                                     # ≈ 5,5 km de radio
    lats = [4.6] + [4.6 + R * math.sin(a) for a in ang]
    lons = [-74.1] + [-74.1 + R * math.cos(a) for a in ang]
    bueno = list(range(nc + 1))                  # vuelta angular suave
    izq = list(range(1, 1 + nc // 2))
    der = list(range(1 + nc // 2, nc + 1))
    malo = [0]                                   # zigzag: cruza el depósito
    while izq or der:
        if izq:
            malo.append(izq.pop(0))
        if der:
            malo.append(der.pop(0))
    return lats, lons, bueno, malo


def _cache_sintetico(ganancia: bool, n: int = 30, nc: int = 10):
    """Cache + serial sintéticos: `ganancia=True` (ACO con el orden bueno) o
    `ganancia=False` (ACO con el MISMO orden que NN ⇒ empate exacto)."""
    from app.core.geo import haversine_mat
    from app.core.ruteo import dist_ruta

    lats, lons, bueno, malo = _anillo(nc)
    D = haversine_mat(np.asarray(lats), np.asarray(lons))
    km_bueno, km_malo = dist_ruta(D, bueno), dist_ruta(D, malo)
    orden_aco = bueno if ganancia else malo
    km_aco = km_bueno if ganancia else km_malo

    serial, corridas, meta = {}, {}, {}
    for i in range(n):
        rid = f"t{i}_2026-01-05"
        serial[rid] = {"territorio": str(i), "fecha": "2026-01-05", "n_clientes": nc,
                       "uids": [i * 1000 + k for k in range(nc)],
                       "lats": lats, "lons": lons,
                       "slots": [[0, 1, 2]] * (nc + 1)}
        corridas[rid] = {"km_nn": km_malo, "orden_nn": malo,
                         "km_aco_seeds": [km_aco] * cfg.ACO_SEEDS_RUTA,
                         "km_aco_mediana": km_aco, "orden_aco": orden_aco,
                         "t_nn_s": 0.01, "t_aco_s": 1.0,
                         "violaciones_ventana": 0, "n_clientes": nc}
        meta[rid] = {"territorio": str(i), "fecha": "2026-01-05", "n_clientes": nc}
    cache = {"rutas": corridas, "meta_rutas": meta,
             "meta": {"beta": 3.0, "semilla": 1, "seeds_por_ruta": cfg.ACO_SEEDS_RUTA,
                      "n_rutas": n, "completadas": n, "n_jobs": 2, "t_ciclo_min": 1.5}}
    return serial, cache


def _regla_desde_cache(serial, cache):
    df = cr.filas_por_ruta(cache)
    glob = cr.resumen_global(df, cache)
    por_tam = cr.resumen_por_tamano(df)
    rob = cr.robustez_montecarlo(serial, cache, n_sim=50)
    return df, glob, por_tam, rob, cr.regla_nn_vs_aco(glob, por_tam, rob)


# ---------------------------------------------------------------------------
# 1) Fuga de información
# ---------------------------------------------------------------------------
def test_particion_sellada_y_estratificada(particion):
    """80/20, disjuntos por `_uid` y estratificados por ZON_REP."""
    tr, te, uni = particion
    assert set(tr["_uid"]).isdisjoint(set(te["_uid"]))
    assert set(tr["_uid"]) | set(te["_uid"]) == set(uni["_uid"])
    assert abs(len(te) / len(uni) - 0.20) < 0.01
    # la mezcla de zonas del test es la del universo (no hay arrastre de una sola zona)
    d = (tr["ZON_REP"].value_counts(normalize=True)
         - te["ZON_REP"].value_counts(normalize=True)).abs().max()
    assert d < 0.05, f"desbalance estratificado: {d:.4f}"
    # semillas declaradas y DISTINTAS (elección ≠ reporte ≠ simulación)
    assert len({cfg.SEMILLA, cfg.SEMILLA_MODELO, cfg.SEMILLA_REPORTE}) == 3


def test_sin_fuga_en_ajuste(muestra):
    """`ajustar(train)` no ve ningún `_uid` de test; depósitos sólo del train."""
    from app.core.ajuste import ajustar, verificar_fuga
    tr, te = muestra
    mod = ajustar(tr, estrategia="actual")
    rep = verificar_fuga(mod, te, tr)
    assert rep["n_uids_test_en_ajuste"] == 0
    assert rep["depositos_solo_de_train"] is True
    assert rep["ok"] is True
    assert set(te["_uid"]).isdisjoint(set(mod.vistos))
    # los clientes de test se asignan POR REGLA (zona real), nunca se firman
    z = te.iloc[0]
    t = mod.territorio_para_cliente(z[cfg.COL_ASESOR], float(z["x_km"]), float(z["y_km"]))
    assert int(t) in mod.depositos


def test_rebalanceo_exige_universo_declarado(muestra):
    """La excepción F2 es explícita: sin `universo_rebalanceo` no hay re-balanceo."""
    from app.core.ajuste import ajustar
    tr, _te = muestra
    with pytest.raises(ValueError):
        ajustar(tr, estrategia="rebalanceo")


def test_excepcion_rebalanceo_declarada(muestra):
    """El reporte deja visible la ÚNICA excepción declarada (atributos de entrada)."""
    from app.core.ajuste import ajustar, verificar_fuga
    tr, te = muestra
    uni = pd.concat([tr, te], ignore_index=True)
    mod = ajustar(tr, estrategia="rebalanceo", universo_rebalanceo=uni)
    rep = verificar_fuga(mod, te, tr)
    assert rep["ok"] is True
    assert rep["excepcion_rebalanceo_declarada"] is True
    assert "atributos de entrada" in rep["nota_excepcion_rebalanceo"]
    # el reporte F2 trae antes/después y movidos (los km sólo con `con_km`)
    for k in ("clientes_movidos", "clientes_movidos_pct", "dias_sobre_tope_antes",
              "dias_sobre_tope_despues", "factible_antes", "factible_despues"):
        assert k in mod.reporte, k
    assert cfg.REBAL_POBLACION == "todos"        # excepción declarada en config


def test_sin_fuga_en_pliegues():
    """Los índices K-Fold se construyen sólo sobre train (el test no se parte)."""
    from app.core.pliegues import kfold_indices
    n, k = 100, cfg.K_PLIEGUES
    folds = kfold_indices(n, k)
    assert len(folds) == 5
    vistos = []
    for i_tr, i_va in folds:
        assert set(i_tr).isdisjoint(set(i_va))
        assert len(i_tr) + len(i_va) == n
        vistos.extend(i_va.tolist())
    # cada cliente es retenido exactamente una vez
    assert sorted(vistos) == list(range(n))


# ---------------------------------------------------------------------------
# 2) Regla NN vs ACO con rutas sintéticas
# ---------------------------------------------------------------------------
def test_aco_mejor_se_adopta():
    """ACO gana de verdad (orden mejor, IC95 > 0, robustez y 0 violaciones) → aco."""
    serial, cache = _cache_sintetico(ganancia=True)
    df, glob, por_tam, rob, regla = _regla_desde_cache(serial, cache)
    assert glob["ahorro_medio_pct"] > cfg.UMBRALES_DECISION["ahorro_aco_pct"]
    assert glob["ic95_inf_pct"] > 0
    assert glob["violaciones_ventana"] == 0
    assert glob["pct_mejor"] == 100.0 and glob["pct_peor"] == 0.0
    assert glob["dispersion_aco"]["semillas_por_ruta"] == cfg.ACO_SEEDS_RUTA
    r85 = [r for r in rob if abs(r["p_asist"] - 0.85) < 1e-9][0]
    assert r85["ic95_inf_pct"] > 0
    assert regla["decision"] == "aco", regla["veredicto"]
    assert all(regla["cumple"].values())
    assert len(regla["criterio"]) > 0


def test_aco_empatado_se_rechaza():
    """Con el MISMO orden que NN (empate técnico) la regla no adopta ACO."""
    serial, cache = _cache_sintetico(ganancia=False)
    df, glob, por_tam, rob, regla = _regla_desde_cache(serial, cache)
    assert abs(glob["ahorro_medio_pct"]) < 1e-6
    assert glob["pct_empate"] == 100.0
    assert regla["decision"] == "nn"
    assert regla["cumple"]["ahorro_minimo"] is False
    assert regla["cumple"]["ic95_positivo"] is False
    assert regla["rangos_con_aco"] == []


def test_aco_mixto_solo_por_rango():
    """Si sólo gana en un rango de tamaño de ruta → decisión 'mixta'."""
    from app.core.geo import haversine_mat
    from app.core.ruteo import dist_ruta

    # --- rutas PEQUEÑAS (5 clientes): ACO y NN dan el mismo orden → empate ---
    lats5, lons5, bueno5, _ = _anillo(nc=5)
    D5 = haversine_mat(np.asarray(lats5), np.asarray(lons5))
    km5 = dist_ruta(D5, bueno5)

    # --- rutas GRANDES (40 clientes): ACO gana ~10 % sobre un orden NN torpe ---
    lats40, lons40, base40, _ = _anillo(nc=40)
    D40 = haversine_mat(np.asarray(lats40), np.asarray(lons40))
    km_base = dist_ruta(D40, base40)
    mejor_torpe, ratio = None, None
    for i in range(1, len(base40) - 1):            # buscar un orden NN ~10 % peor
        for j in range(i + 2, len(base40) - 1):
            if i == 1 and j == len(base40) - 1:
                continue                            # no tocar el cierre
            cand = list(base40)
            cand[i], cand[j] = cand[j], cand[i]
            km = dist_ruta(D40, cand)
            if km > km_base and (ratio is None or abs(km / km_base - 1.10) < abs(ratio - 1.10)):
                mejor_torpe, ratio = cand, km / km_base
    km_nn40 = dist_ruta(D40, mejor_torpe)
    km_aco40 = km_base
    assert 1.05 < km_nn40 / km_aco40 < 1.20, "el orden NN torpe debe costar ~10 %"

    serial, corridas, meta = {}, {}, {}
    i = 0
    for nc, lats, lons, km_nn, km_aco, ord_nn, ord_aco, n in (
            (5, lats5, lons5, km5, km5, bueno5, bueno5, 40),
            (40, lats40, lons40, km_nn40, km_aco40, mejor_torpe, base40, 6)):
        for _ in range(n):
            rid = f"t{i}_2026-01-05"
            serial[rid] = {"territorio": str(i), "fecha": "2026-01-05",
                           "n_clientes": nc, "uids": [i * 1000 + k for k in range(nc)],
                           "lats": lats, "lons": lons,
                           "slots": [[0, 1, 2]] * (nc + 1)}
            corridas[rid] = {"km_nn": km_nn, "orden_nn": ord_nn,
                             "km_aco_seeds": [km_aco] * cfg.ACO_SEEDS_RUTA,
                             "km_aco_mediana": km_aco, "orden_aco": ord_aco,
                             "t_nn_s": 0.01, "t_aco_s": 1.0,
                             "violaciones_ventana": 0, "n_clientes": nc}
            meta[rid] = {"territorio": str(i), "fecha": "2026-01-05", "n_clientes": nc}
            i += 1
    cache = {"rutas": corridas, "meta_rutas": meta,
             "meta": {"n_rutas": i, "completadas": i, "n_jobs": 2, "t_ciclo_min": 1.0}}
    _df, glob, por_tam, _rob, regla = _regla_desde_cache(serial, cache)

    umb = cfg.UMBRALES_DECISION
    assert glob["ahorro_medio_pct"] < umb["ahorro_aco_pct"], "global por debajo del umbral"
    rango = [r for r in por_tam if r["lo"] == 31 and r["n_rutas"] > 0]
    assert rango and rango[0]["ahorro_medio_pct"] >= umb["ahorro_aco_pct"]
    assert rango[0]["ic95_inf_pct"] > 0
    assert regla["decision"] == "mixta"
    assert rango[0]["rango"] in regla["rangos_con_aco"]


def test_criterio_nn_vs_aco_esta_escrito():
    """El criterio es texto fijo y declarado ANTES de ver resultados."""
    txt = cr.CRITERIO_NN_VS_ACO
    assert "cinco" in txt.lower()
    assert "ahorro_aco_pct" in txt and "(2 %)" in txt      # umbral 1
    assert "(30 min)" in txt                                # umbral 5
    assert "0,85" in txt                                    # robustez (umbral 3)
    assert "violaciones de ventana" in txt                  # umbral 4
    assert "En ningún caso se ajusta un umbral" in txt


# ---------------------------------------------------------------------------
# 3) Bootstrap reproducible
# ---------------------------------------------------------------------------
def test_bootstrap_reproducible():
    vals = np.random.default_rng(0).normal(1.0, 2.0, 60)
    a = cr._bootstrap_media(vals, 10_000, cfg.BOOTSTRAP_SEMILLA)
    b = cr._bootstrap_media(vals, 10_000, cfg.BOOTSTRAP_SEMILLA)
    assert a == b, "la misma semilla debe dar el mismo IC95"
    c = cr._bootstrap_media(vals, 10_000, cfg.BOOTSTRAP_SEMILLA + 1)
    assert c != a, "semillas distintas deben dar remuestreos distintos"
    media, inf, sup = a
    assert inf <= media <= sup
    assert abs(media - vals.mean()) < 1e-9        # la media no cambia
    assert cfg.BOOTSTRAP_N == 10_000 and cfg.BOOTSTRAP_SEMILLA == 777


def test_bootstrap_vacio_no_crashea():
    media, inf, sup = cr._bootstrap_media(np.array([]), 100, 1)
    assert math.isnan(media) and math.isnan(inf) and math.isnan(sup)


# ---------------------------------------------------------------------------
# 4) P_ASIST = 1,00 ⇒ cobertura de ACTUAL < 1
# ---------------------------------------------------------------------------
def test_p_asist_1_cobertura_menor_que_uno(particion):
    """Con todos los clientes presentes, ACTUAL pierde visitas por SOBRECARGA."""
    from app.core.ajuste import ajustar, evaluar
    from app.core.montecarlo import generar_entradas

    tr, _te, uni = particion
    asistencia, trafico = generar_entradas(40, len(uni), cfg.SEMILLA, p_asist=1.0)
    assert bool(asistencia.all()), "con P_ASIST=1 todos asisten"
    mod = ajustar(tr, estrategia="actual")
    e = evaluar(mod, uni, asistencia, trafico)
    cob = e["montecarlo"]["cobertura"]
    assert cob is not None and cob < 1.0, f"cobertura={cob}"
    # la causa NO son ausencias: es la sobrecarga (días sobre el tope) de ACTUAL
    assert e["factibilidad"]["dias_sobre_cap"] >= 1
    assert e["montecarlo"]["visitas_perdidas"] > 0
    assert e["factibilidad"]["cap_dinamico"] <= cfg.CAP_TOPE


# ---------------------------------------------------------------------------
# 5) Regla de estrategias: "ninguna cumple"
# ---------------------------------------------------------------------------
def _estrategia(condiciones: dict, diagnostica: bool = False, costo: float = 100.0):
    return {
        "etiqueta": "x", "recomendable": not diagnostica, "es_diagnostica": diagnostica,
        "K": 58, "estrategia": "actual", "lam_norm": 1.0, "beta": 3.0,
        "resumen": {},
        "dimensiones": {"costo": {"costo_medio_M": costo,
                                  "delta_vs_actual_pct": None, "cumple": False}},
        "condiciones": {c: condiciones.get(c, False) for c in ev_mod.CONDICIONES},
        "cumple_todas": all(condiciones.get(c, False) for c in ev_mod.CONDICIONES),
        "arrepentimiento": {},
    }


def test_regla_ninguna_cumple():
    """Ninguna candidata pasa las 6 condiciones → literal 'ninguna ... condiciones'."""
    estrategias = {
        "ACTUAL": _estrategia({}, costo=100.0),
        "REBAL_58": _estrategia({"factibilidad": True, "cumplimiento": True,
                                 "eficiencia": True, "generalizacion": True,
                                 "estabilidad": True}, costo=101.0),   # costo NO menor
        "K43": _estrategia({c: True for c in ev_mod.CONDICIONES},
                           diagnostica=True, costo=50.0),               # diagnostica
    }
    r = ev_mod.regla_estrategias(estrategias)
    assert r["decision"] == "ninguna"
    assert r["recomendada"] is None
    assert "ninguna estrategia cumple las condiciones" in r["veredicto"]
    # la reagrupación global nunca entra al pool aunque cumpla todo
    assert r["pool_candidatas"] == ["REBAL_58"]
    assert r["diagnosticas"] == ["K43"]
    assert r["condiciones_por_estrategia"]["REBAL_58"]["costo"] is False


def test_regla_recomienda_si_cumple_todo():
    """Control positivo: REBAL con las seis condiciones en verde → recomendada."""
    estrategias = {
        "ACTUAL": _estrategia({}, costo=100.0),
        "REBAL_58": _estrategia({c: True for c in ev_mod.CONDICIONES}, costo=99.0),
    }
    r = ev_mod.regla_estrategias(estrategias)
    assert r["decision"] == "recomendada"
    assert r["recomendada"] == "REBAL_58"


def test_diagnosticas_nunca_recomendables():
    """Aunque la reagrupación cumpla todo, no puede ser recomendada."""
    estrategias = {
        "ACTUAL": _estrategia({}, costo=100.0),
        "K48": _estrategia({c: True for c in ev_mod.CONDICIONES}, diagnostica=True),
    }
    r = ev_mod.regla_estrategias(estrategias)
    assert r["decision"] == "ninguna"
    assert r["recomendada"] is None
    assert "jamás" in (r["nota"] or "").lower()        # nunca se recomienda una K
    assert "diagnóstica" in r["criterio"].lower()


def test_ocho_dimensiones_y_seis_condiciones():
    assert len(ev_mod.DIMENSIONES) == 8
    assert set(ev_mod.CONDICIONES) <= set(ev_mod.DIMENSIONES)
    assert len(ev_mod.CONDICIONES) == 6
    assert set(ev_mod.DIMENSIONES) == {"costo", "eficiencia", "factibilidad",
                                       "cumplimiento", "balance", "generalizacion",
                                       "estabilidad", "compacidad"}


def test_umbrales_etiquetados_no_calibrados():
    p = cfg.params()["UMBRALES_DECISION"]
    assert "no calibrada" in p["etiqueta"].lower()
    assert cfg.UMBRALES_DECISION["ahorro_aco_pct"] == 2.0
    assert cfg.UMBRALES_DECISION["tiempo_ciclo_min"] == 30.0
    assert cfg.K_CANDIDATOS == (43, 48, 53, 58)
    assert cfg.K_PLIEGUES == 5 and len(cfg.LAM_GRID) == 3


# ---------------------------------------------------------------------------
# 6) Endpoints
# ---------------------------------------------------------------------------
def test_endpoints_evaluacion(cliente_api):
    r1 = cliente_api.get("/api/evaluacion/nn-vs-aco")
    r2 = cliente_api.get("/api/evaluacion/estrategias")
    r3 = cliente_api.get("/api/evaluacion/regla")
    if 503 in (r1.status_code, r2.status_code, r3.status_code):
        pytest.skip("precompute del protocolo aún en curso")
    assert r1.status_code == r2.status_code == r3.status_code == 200

    nn = r1.json()
    assert nn["global"]["n_rutas"] > 0
    assert {"ahorro_medio_pct", "ic95_inf_pct", "ic95_sup_pct", "violaciones_ventana",
            "pct_mejor", "pct_empate", "pct_peor", "t_ciclo_min"} <= set(nn["global"])
    assert len(nn["por_tamano"]) == len(cfg.RANGOS_TAMANO_AHORRO)
    assert len(nn["robustez"]) == len(cfg.P_ASIST_EVAL)
    assert nn["regla"]["decision"] in {"aco", "nn", "mixta"}
    assert len(nn["criterio_regla"]) > 50

    es = r2.json()
    assert {"ACTUAL", "REBAL_58"} <= set(es["estrategias"])
    assert es["tabla"] and len(es["estrategias"]["ACTUAL"]["dimensiones"]) == 8
    assert es["f2_rebalanceo"]["poblacion_del_rebalanceo"] == cfg.REBAL_POBLACION
    assert es["estabilidad"]["semillas"] == list(cfg.SEMILLAS_ESTABILIDAD)

    reg = r3.json()
    assert reg["estrategias"]["decision"] in {"recomendada", "ninguna"}
    assert reg["nn_vs_aco"]["decision"] in {"aco", "nn", "mixta"}
    assert "propuesta" in reg["etiqueta_umbrales"]
    # ninguna respuesta expone datos personales
    import json as _json
    for resp in (r1, r2, r3):
        txt = _json.dumps(resp.json(), ensure_ascii=False).lower()
        for prohibido in ("identificacion", "direccion", '"cliente"'):
            assert prohibido not in txt, f"{prohibido} filtrado en {resp.request.url}"


def test_regla_estrategias_coherente_con_cache(cliente_api):
    """La regla del endpoint es la MISMA que la cache (no se recalcula al vuelo)."""
    import json
    from pathlib import Path

    cache = Path(cfg.CACHE_DIR) / "evaluacion.json"
    if not cache.exists():
        pytest.skip("evaluacion.json aún no precalculado")
    d = json.loads(cache.read_text(encoding="utf-8"))
    reg = cliente_api.get("/api/evaluacion/regla").json()["estrategias"]
    assert reg == d["informe"]["regla"]
    # las condiciones están por cada estrategia evaluada (ni más ni menos)
    assert set(reg["condiciones_por_estrategia"]) == {
        f["estrategia"] for f in d["informe"]["tabla"]}
