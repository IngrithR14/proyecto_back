"""Separación ajuste/evaluación (sección 1 del protocolo de evaluación).

El ruteo no entrena pesos, pero **sí ajusta** cosas a los datos: territorios
(K-Means/re-balanceo), depósitos (centroides), β del ACO, el K elegido y los
parámetros del re-balanceo. Todo eso se ajusta SOLO con datos de entrenamiento.

- `particionar(df)`          → (train, test) 80/20 estratificado por ZON_REP.
- `ajustar(train)`           → `ModeloRuteo` (territorios, depósitos, β, λ).
- `evaluar(modelo, datos)`   → solución y métricas; los clientes que el modelo
                               no vio (test) entran SOLO por inserción más barata
                               en las rutas existentes, respetando CAP, ventana
                               horaria y las fechas del cliente.
- `verificar_fuga(...)`      → test automático: ningún `_uid` de test en `ajustar`
                               y depósitos derivados exclusivamente de train.

LÍMITES (se declaran siempre en el informe):
  * hay UNA sola base (un mes) → no existe validación temporal;
  * la partición es estratificada por `ZON_REP` con semilla fija.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .. import config as cfg


# ---------------------------------------------------------------------------
# Partición estratificada 80/20 (test SELLADO)
# ---------------------------------------------------------------------------
def particionar(df: pd.DataFrame, test_size: float | None = None,
                semilla: int | None = None, col: str | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide en train (80 %) y test (20 %) estratificado por `col` (ZON_REP).

    Semilla fija (`cfg.SEMILLA_MODELO`). Los estratos con un solo cliente quedan
    enteros en train (no se puede estratificar con n=1); se reporta en `attrs`.
    Estratificación propia y determinista: mismo resultado con la misma semilla.
    """
    test_size = cfg.TEST_SIZE if test_size is None else float(test_size)
    semilla = cfg.SEMILLA_MODELO if semilla is None else int(semilla)
    col = col or cfg.COL_ASESOR

    d = df.copy().reset_index(drop=True)
    if "_uid" not in d.columns:
        d["_uid"] = np.arange(len(d))
    d = d.sort_values("_uid", kind="mergesort").reset_index(drop=True)

    rng = np.random.default_rng(semilla)
    test_idx: list[int] = []
    estratos_n1 = 0
    for _valor, g in d.groupby(col, sort=True):
        pos = rng.permutation(g.index.to_numpy())
        n_t = int(round(len(pos) * test_size))
        n_t = max(0, min(n_t, len(pos) - 1))     # siempre queda ≥1 en train
        if len(pos) == 1:
            estratos_n1 += 1
        test_idx.extend(pos[:n_t].tolist())
    test_idx = np.asarray(sorted(test_idx), dtype=int)
    mascara = np.zeros(len(d), bool)
    mascara[test_idx] = True
    train = d[~mascara].reset_index(drop=True)
    test = d[mascara].reset_index(drop=True)
    train.attrs["estratos_n1_train"] = estratos_n1
    train.attrs["n_zonas_train"] = int(train[col].nunique())
    train.attrs["n_zonas_test"] = int(test[col].nunique())
    return train, test


def distancia_tipica(df: pd.DataFrame, labels: np.ndarray | None = None) -> float:
    """Distancia típica cliente–centroide (mediana, km) sobre ESTAS filas.

    Es la escala con la que se normaliza λ (λ = λ_norm · d_typ²) para que los
    pesos del re-balanceo no dependan de la unidad de los datos.
    """
    d = df
    if "x_km" not in d.columns:
        d = df.copy()
        from .geo import proyectar
        proyectar(d, col_grupo=None)
    xy = d[["x_km", "y_km"]].values
    if labels is None:
        _, labels = np.unique(d[cfg.COL_ASESOR].astype(str).values, return_inverse=True)
    labels = np.asarray(labels)
    dists = np.empty(len(d), dtype=float)
    for t in np.unique(labels):
        m = labels == t
        c = xy[m].mean(axis=0)
        dists[m] = np.hypot(xy[m, 0] - c[0], xy[m, 1] - c[1])
    return float(np.median(dists))


# ---------------------------------------------------------------------------
# Modelo ajustado (solo con train)
# ---------------------------------------------------------------------------
@dataclass
class ModeloRuteo:
    """Resultado de `ajustar(train)`: territorios, depósitos y parámetros.

    `asignar_por` dice cómo se asigna el territorio de un cliente NUEVO (de test):
      "zona"     → por su `ZON_REP` (atributo de entrada de cada registro, igual
                   que `DIAS_REP`/`FRE_REP`/coordenadas); vale para las estrategias
                   cuyos territorios SON las zonas (ACTUAL y re-balanceo).
      "centroide"→ por el centroide más cercano; vale para la re-agrupación global,
                   donde los territorios son conglomerados aprendidos y un cliente
                   nuevo sólo puede ubicarse por proximidad.
    """
    estrategia: str
    K: int
    beta: float
    lam_norm: float
    lam: float                       # λ absoluto = λ_norm · d_typ²
    lam_fecha: float
    d_typ: float
    semilla: int
    vistos: frozenset[int]                       # _uid vistos por AJUSTAR
    territorio_de: dict[int, int]                # uid → territorio (train)
    centroides: dict[int, tuple[float, float]]   # territorio → (x_km, y_km)
    depositos: dict[int, tuple[float, float]]    # territorio → (lat, lon)
    asignar_por: str = "centroide"
    zona_a_territorio: dict[str, int] = field(default_factory=dict)
    reporte: dict[str, Any] = field(default_factory=dict)

    @property
    def n_vistos(self) -> int:
        return len(self.vistos)

    def territorio_para(self, uid: int) -> int:
        """Territorio de un cliente: el suyo si lo vio, si no el centroide más cercano."""
        t = self.territorio_de.get(int(uid))
        if t is not None:
            return t
        raise KeyError(f"uid {uid} no visto por ajustar()")

    def territorio_para_xy(self, x: float, y: float) -> int:
        mejor, dmin = None, np.inf
        for t, (cx, cy) in self.centroides.items():
            dd = (cx - x) ** 2 + (cy - y) ** 2
            if dd < dmin:
                mejor, dmin = t, dd
        return int(mejor)

    def territorio_para_cliente(self, zona: str, x: float, y: float) -> int:
        """Territorio de un cliente que el modelo NO vio (test / nuevo)."""
        if self.asignar_por == "zona":
            t = self.zona_a_territorio.get(str(zona))
            if t is not None:
                return int(t)
        return self.territorio_para_xy(x, y)


def ajustar(train: pd.DataFrame, estrategia: str = "rebalanceo",
            K: int | None = None, beta: float | None = None,
            lam_norm: float | None = None, semilla: int | None = None,
            cap: int | None = None, con_km: bool = False,
            universo_rebalanceo: pd.DataFrame | None = None) -> ModeloRuteo:
    """Construye el modelo con los territorios/depósitos/parámetros de la estrategia.

    Estrategias:
      "actual"      → territorios = ZON_REP reales (sin reclusterizar).
      "rebalanceo"  → F2: re-balanceo mínimo de ACTUAL (solo K = K_ACTUAL).
      "k"           → reagrupación global con restricción de capacidad (DIAGNÓSTICA).

    `lam_norm` es la λ normalizada de la rejilla `cfg.LAM_GRID` (la absoluta es
    λ_norm·d_typ² con d_typ = distancia típica cliente–centroide de ESTE train).

    Excepción declarada (`cfg.REBAL_POBLACION`): para la estrategia "rebalanceo"
    la reparación estructural se calcula con los atributos de ENTRADA (zona,
    fechas, visitas, coordenadas) de `universo_rebalanceo`; si la política es
    "todos" hay que pasar el universo completo (si no, `ajustar` falla a propósito
    para que no se ejecute un re-balanceo sin efecto). No se usa ningún resultado
    ni métrica de test. `ModeloRuteo.vistos` deja constancia de las filas que
    alimentan territorios/depósitos y las verifica `verificar_fuga()`.

    Nota: el territorio de un cliente que el modelo no vio sale de su `ZON_REP`
    (atributo de entrada) cuando la estrategia tiene zonas como territorios, y del
    centroide más cercano cuando la estrategia re-agrupa (`asignar_por`).
    """
    from .estructura import asignacion_capacidad, rebalancear_desde_actual
    from .geo import proyectar

    semilla = cfg.SEMILLA_MODELO if semilla is None else int(semilla)
    lam_norm = cfg.LAM_GRID[1] if lam_norm is None else float(lam_norm)
    beta = cfg.ACO_BETA if beta is None else float(beta)

    d = train.copy().reset_index(drop=True)
    if "_uid" not in d.columns:
        d["_uid"] = np.arange(len(d))
    proyectar(d, col_grupo=None)

    # territorios base = asignación real (ACTUAL) sobre train
    zonas = d[cfg.COL_ASESOR].astype(str).values
    _codes, labels_base = np.unique(zonas, return_inverse=True)
    d_typ = distancia_tipica(d, labels_base)
    lam = lam_norm * d_typ * d_typ
    lam_fecha = cfg.LAM_FECHA_REL * lam

    reporte: dict[str, Any] = {}
    codes_territorio = _codes
    if estrategia == "actual":
        labels = labels_base
        K = int(len(np.unique(labels_base)))
    elif estrategia == "rebalanceo":
        K = cfg.K_ACTUAL
        if universo_rebalanceo is None and cfg.REBAL_POBLACION == "todos":
            raise ValueError(
                "REBAL_POBLACION='todos': hay que pasar `universo_rebalanceo` "
                "(todos los clientes) para calcular el re-balanceo F2.")
        # Decisión registrada en `cfg.REBAL_POBLACION`: la reparación estructural
        # del re-balanceo se calcula con los atributos de entrada de TODOS los
        # clientes (no con resultados). Ver `cfg.REBAL_POBLACION`.
        base = d if universo_rebalanceo is None else universo_rebalanceo
        base = base.copy().reset_index(drop=True)
        if "_uid" not in base.columns:
            base["_uid"] = np.arange(len(base))
        if "x_km" not in base.columns:
            proyectar(base, col_grupo=None)
        _codes_u, labels_u = np.unique(base[cfg.COL_ASESOR].astype(str).values,
                                       return_inverse=True)
        labels_u, reporte = rebalancear_desde_actual(base, labels=labels_u, cap=cap,
                                                     con_km=con_km)
        mapa = {int(u): int(t) for u, t in zip(base["_uid"].astype(int), labels_u)}
        labels = np.array([mapa[int(u)] for u in d["_uid"].astype(int)], dtype=int)
        codes_territorio = _codes_u
    elif estrategia == "k":
        if K is None:
            raise ValueError("estrategia 'k' requiere K explícito")
        labels = asignacion_capacidad(d, int(K), semilla=semilla,
                                      lam=lam, lam_fecha=lam_fecha)
    else:
        raise ValueError(f"estrategia desconocida: {estrategia}")

    labels = np.asarray(labels)
    K = int(len(np.unique(labels)))
    uids = d["_uid"].astype(int).values
    territorio_de = {int(u): int(t) for u, t in zip(uids, labels)}
    asignar_por = "zona" if estrategia in ("actual", "rebalanceo") else "centroide"
    # códigos de territorio = zonas únicas (orden alfabético), su posición
    zona_a_territorio = {str(z): int(i) for i, z in enumerate(codes_territorio)}

    centroides: dict[int, tuple[float, float]] = {}
    depositos: dict[int, tuple[float, float]] = {}
    d["_t"] = labels
    for t, g in d.groupby("_t"):
        centroides[int(t)] = (float(g["x_km"].mean()), float(g["y_km"].mean()))
        # E9: depósito por territorio = centroide (lat/lon) del grupo
        depositos[int(t)] = (float(g["Latitud"].mean()), float(g["Longitud"].mean()))

    return ModeloRuteo(
        estrategia=estrategia, K=K, beta=beta, lam_norm=lam_norm, lam=lam,
        lam_fecha=lam_fecha, d_typ=d_typ, semilla=semilla,
        vistos=frozenset(int(u) for u in uids),
        territorio_de=territorio_de, centroides=centroides, depositos=depositos,
        asignar_por=asignar_por, zona_a_territorio=zona_a_territorio,
        reporte=reporte,
    )


# ---------------------------------------------------------------------------
# Rutas diarias con territorios fijados por el modelo
# ---------------------------------------------------------------------------
def rutas_diarias(df: pd.DataFrame, labels: np.ndarray, cap: int | None = None,
                  semilla: int | None = None) -> list[dict[str, Any]]:
    """Una ruta por (territorio, fecha) — idéntica a `construir_rutas_diarias`
    pero con coordenadas, ventanas y depósito guardados para poder insertar."""
    from .calendario import fechas_permitidas
    from .estructura import _depot_de
    from .geo import haversine_mat
    from .ruteo import dist_ruta, dos_opt, nearest_neighbor

    cap = cfg.CAP_TOPE if cap is None else int(cap)
    semilla = cfg.SEMILLA if semilla is None else semilla
    d = df.copy().reset_index(drop=True)
    if "_uid" not in d.columns:
        d["_uid"] = np.arange(len(d))
    d["_terr"] = np.asarray(labels)

    dias: dict[tuple, list[int]] = {}
    for _, r in d.iterrows():
        permitidas = fechas_permitidas(r["DIAS_REP"], r["FRE_REP"])
        if not permitidas:
            continue
        objetivo = int(r["visitas_mes_rep"])
        base, extra = divmod(objetivo, len(permitidas))
        for i, f in enumerate(permitidas):
            v = base + (1 if i < extra else 0)
            if v > 0:
                dias.setdefault((r["_terr"], f), []).append(int(r["_uid"]))

    depot = {t: _depot_de(terr, "territorio") for t, terr in d.groupby("_terr")}
    idx = d.set_index("_uid")

    rutas: list[dict[str, Any]] = []
    for (t, f), uids in dias.items():
        sub = idx.loc[uids]
        dep = depot[t]
        lats = np.concatenate([[dep[0]], sub["Latitud"].values])
        lons = np.concatenate([[dep[1]], sub["Longitud"].values])
        D = haversine_mat(lats, lons)
        slots = [{0, 1, 2}] + list(sub["slots"].values)
        orden = nearest_neighbor(D, slots)
        orden = dos_opt(orden, D, slots)
        uid_ord = sub.index.values[np.asarray(orden[1:]) - 1]
        rutas.append(dict(territorio=t, fecha=f, D=D, lats=lats, lons=lons,
                          slots=slots, orden=orden, uid=uid_ord,
                          km_base=float(dist_ruta(D, orden)),
                          n_clientes=int(len(sub)),
                          excede_cap=int(len(sub) > cap)))
    rutas.sort(key=lambda r: (str(r["territorio"]), str(r["fecha"])))
    return rutas


def km_de_rutas(rutas: list[dict]) -> float:
    return float(sum(r["km_base"] for r in rutas))


def pulir_rutas(rutas: list[dict]) -> None:
    """Re-secuencia cada ruta con NN + 2-opt (el mismo estándar de planificación).

    NO cambia quién está en cada ruta (ni territorios ni fechas): sólo el orden de
    las paradas. Se aplica después de insertar clientes de test para que la ruta
    final quede planificada igual que cualquier otra del proyecto; el coste
    MARGINAL de la inserción se mide antes de esto, en `insertar_test`.
    """
    from .ruteo import dist_ruta, dos_opt, nearest_neighbor

    for r in rutas:
        if len(r["orden"]) < 3:
            continue
        node_uid = {int(r["orden"][k]): int(r["uid"][k - 1])
                    for k in range(1, len(r["orden"]))}
        nuevo = dos_opt(nearest_neighbor(r["D"], r["slots"]), r["D"], r["slots"])
        if len(nuevo) != len(r["orden"]):
            continue
        r["orden"] = nuevo
        r["uid"] = np.asarray([node_uid[int(n)] for n in nuevo[1:]], dtype=int)
        r["km_base"] = float(dist_ruta(r["D"], nuevo))


# ---------------------------------------------------------------------------
# Inserción más barata (test → rutas existentes)
# ---------------------------------------------------------------------------
def _hav(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * cfg.R_TIERRA * np.arcsin(np.sqrt(min(1.0, a))))


def insertar_en_ruta(ruta: dict, uid: int, lat: float, lon: float, slot: set[int],
                     cap: int) -> tuple[float | None, str | None]:
    """Inserta `uid` en la posición más barata de `ruta` si respeta CAP y ventana.

    Devuelve (Δkm, None) si la aceptó o (None, motivo) si no: motivo "cap" cuando
    la ruta ya está en el tope y "ventana" cuando ninguna posición respeta las
    ventanas horarias. Modifica `ruta` in place.
    """
    from .ruteo import factible_orden

    if ruta["n_clientes"] + 1 > cap:
        return None, "cap"
    n = len(ruta["orden"])            # incluye el depósito (nodo 0)
    D = ruta["D"]
    orden = list(ruta["orden"])
    cerrada = bool(cfg.RUTA_CERRADA)
    nuevo_id = n                      # id del nodo nuevo (se anade al final)
    d_new = np.empty(n + 1)
    for i in range(n):
        d_new[i] = _hav(lat, lon, float(ruta["lats"][i]), float(ruta["lons"][i]))
    d_new[nuevo_id] = 0.0

    mejor: tuple[float, int] | None = None    # (Δkm, posición)
    for p in range(1, n + 1):                 # posición en `orden`
        a = orden[p - 1]
        b = orden[p] if p < n else None
        delta = d_new[a]
        if b is not None:
            delta += d_new[b] - D[a, b]
        elif cerrada:
            delta += d_new[orden[0]] - D[a, orden[0]]
        if mejor is not None and delta >= mejor[0]:
            continue
        nuevo_orden = orden[:p] + [nuevo_id] + orden[p:]
        if not factible_orden(nuevo_orden, ruta["slots"] + [slot]):
            continue
        if mejor is None or delta < mejor[0]:
            mejor = (float(delta), p)

    if mejor is None:
        return None, "ventana"
    delta, p = mejor
    D_n = np.zeros((n + 1, n + 1))
    D_n[:n, :n] = D
    D_n[n, :n] = d_new[:n]
    D_n[:n, n] = d_new[:n]
    ruta["D"] = D_n
    ruta["lats"] = np.append(ruta["lats"], lat)
    ruta["lons"] = np.append(ruta["lons"], lon)
    ruta["slots"] = ruta["slots"] + [slot]
    ruta["orden"] = orden[:p] + [nuevo_id] + orden[p:]
    nueva_uid = list(ruta["uid"])
    nueva_uid.insert(p - 1, int(uid))
    ruta["uid"] = np.asarray(nueva_uid)
    ruta["km_base"] = float(ruta["km_base"] + delta)
    ruta["n_clientes"] = int(ruta["n_clientes"] + 1)
    ruta["excede_cap"] = int(ruta["n_clientes"] > cfg.CAP_TOPE)
    return float(delta), None


def insertar_test(modelo: ModeloRuteo, rutas: list[dict], test: pd.DataFrame,
                  cap: int | None = None) -> tuple[list[dict], dict[str, Any]]:
    """Clientes de test → inserción más barata en las rutas existentes.

    Reglas: territorio = el natural de la estrategia (`ModeloRuteo.asignar_por`:
    `ZON_REP` del cliente para ACTUAL/re-balanceo, centroide más cercano para la
    re-agrupación); fecha = las que permite el cliente (`DIAS_REP`/`FRE_REP`); el
    tope usado es el mismo que la regla de factibilidad (`cap_dinamico`) y la
    ventana horaria se respeta. Si no hay ruta ese día → "ruta nueva"; si no
    cabe → "excede_cap"; si no hay posición factible por ventana → "ventana".
    """
    from .calendario import fechas_permitidas

    cap = cfg.cap_dinamico() if cap is None else int(cap)
    d = test.copy().reset_index(drop=True)
    if "_uid" not in d.columns:
        d["_uid"] = np.arange(len(d))
    if "x_km" not in d.columns:
        from .geo import proyectar
        proyectar(d, col_grupo=None)

    indice: dict[tuple, dict] = {}
    for r in rutas:
        indice[(r["territorio"], r["fecha"])] = r

    motivos: dict[str, int] = {"sin_ruta_en_fecha": 0, "excede_cap": 0, "ventana": 0}
    completos = 0
    visitas_prog = 0
    visitas_ok = 0
    km_antes = km_de_rutas(rutas)
    movidos: list[dict[str, Any]] = []

    for _, c in d.iterrows():
        uid = int(c["_uid"])
        terr = modelo.territorio_para_cliente(c[cfg.COL_ASESOR], float(c["x_km"]),
                                              float(c["y_km"]))
        permitidas = fechas_permitidas(c["DIAS_REP"], c["FRE_REP"])
        objetivo = int(c["visitas_mes_rep"])
        visitas_prog += objetivo
        if not permitidas or objetivo <= 0:
            completos += 1                                # no pide visita
            continue
        base, extra = divmod(objetivo, len(permitidas))
        fallas: list[str] = []
        for i, f in enumerate(permitidas):
            v = base + (1 if i < extra else 0)
            for _ in range(v):
                ruta = indice.get((terr, f))
                if ruta is None:
                    fallas.append("sin_ruta_en_fecha")
                    continue
                dm, motivo = insertar_en_ruta(ruta, uid, float(c["Latitud"]),
                                              float(c["Longitud"]), set(c["slots"]), cap)
                if dm is None:
                    fallas.append("excede_cap" if motivo == "cap" else (motivo or "excede_cap"))
                else:
                    visitas_ok += 1
                    movidos.append({"uid": uid, "fecha": str(f), "delta_km": round(dm, 3)})
        for m in set(fallas):
            motivos[m] = motivos.get(m, 0) + 1
        if not fallas:
            completos += 1

    n = len(d)
    km_despues = km_de_rutas(rutas)                # acumulado de los Δ de inserción
    km_marginal = km_despues - km_antes
    # km extra MARGINAL por cliente de test (p50/p90 son las métricas de generalización)
    por_cliente: dict[int, float] = {}
    for m in movidos:
        por_cliente[m["uid"]] = por_cliente.get(m["uid"], 0.0) + m["delta_km"]
    arr_c = np.asarray(list(por_cliente.values()), dtype=float)
    p50 = float(np.percentile(arr_c, 50)) if len(arr_c) else 0.0
    p90 = float(np.percentile(arr_c, 90)) if len(arr_c) else 0.0
    pulir_rutas(rutas)                             # mismo estándar NN+2-opt que train
    km_plan = km_de_rutas(rutas)
    rep = {
        "n_test": int(n),
        "n_insertados_completos": int(completos),
        "pct_insertados_completos": round(100.0 * completos / n, 2) if n else 0.0,
        "visitas_test": int(visitas_prog),
        "visitas_insertadas": int(visitas_ok),
        "pct_visitas_insertadas": round(100.0 * visitas_ok / visitas_prog, 2)
        if visitas_prog else 0.0,
        "pct_ruta_nueva": round(100.0 * motivos["sin_ruta_en_fecha"] / n, 2) if n else 0.0,
        "pct_excede_cap": round(100.0 * motivos["excede_cap"] / n, 2) if n else 0.0,
        "pct_ventana": round(100.0 * motivos["ventana"] / n, 2) if n else 0.0,
        "n_rutas_nuevas": int(motivos["sin_ruta_en_fecha"]),
        "n_clientes_con_insercion": int(len(por_cliente)),
        "km_extra_cliente_p50": round(p50, 3),
        "km_extra_cliente_p90": round(p90, 3),
        # coste MARGINAL de meter cada cliente de test (método: inserción más barata)
        "km_antes_insercion": round(km_antes, 1),
        "km_tras_insercion": round(km_despues, 1),
        "km_marginal_insercion": round(km_marginal, 1),
        "km_marginal_pct": round(100.0 * km_marginal / km_antes, 3) if km_antes else None,
        # plan final (misma ruta, re-secuenciada con NN+2-opt igual que train)
        "km_plan_final": round(km_plan, 1),
        "km_plan_vs_antes_pct": round(100.0 * (km_plan - km_antes) / km_antes, 3)
        if km_antes else None,
        "inserciones": movidos[:100],
    }
    return rutas, rep


# ---------------------------------------------------------------------------
# Evaluación (nunca re-ajusta)
# ---------------------------------------------------------------------------
def evaluar(modelo: ModeloRuteo, datos: pd.DataFrame,
            asistencia: np.ndarray | None = None,
            trafico: np.ndarray | None = None) -> dict[str, Any]:
    """Solución de `modelo` sobre `datos`.

    Los clientes vistos por `ajustar` ocupan sus rutas; los que no (test) entran
    SOLO por inserción. Nunca se re-ajustan territorios ni depósitos aquí.
    """
    from .montecarlo import (evaluar_estrategia, factibilidad,
                             jornadas_por_escenario, resumen_jornadas)

    d = datos.copy().reset_index(drop=True)
    if "_uid" not in d.columns:
        d["_uid"] = np.arange(len(d))
    if "x_km" not in d.columns:
        from .geo import proyectar
        proyectar(d, col_grupo=None)

    conocidos = d[d["_uid"].astype(int).isin(modelo.vistos)]
    nuevos = d[~d["_uid"].astype(int).isin(modelo.vistos)]

    labels = np.array([modelo.territorio_de[int(u)] for u in conocidos["_uid"]],
                      dtype=int)
    rutas = rutas_diarias(conocidos, labels) if len(conocidos) else []
    km_train = km_de_rutas(rutas)

    ins: dict[str, Any] = {"n_test": 0}
    if len(nuevos):
        rutas, ins = insertar_test(modelo, rutas, nuevos)

    # Factibilidad medida sobre TODAS las visitas de la población evaluada: una
    # visita que no cupo dentro del tope no desaparece, cuenta como día sobre el
    # tope (igual que en la línea base ACTUAL). Se añade la factibilidad del plan
    # ruteado (sólo lo que cabió) para comparar.
    from .calendario import quincena_de
    from .montecarlo import carga_por_territorio_fecha
    labels_todo = np.array([
        modelo.territorio_de[int(u)] if int(u) in modelo.vistos
        else modelo.territorio_para_cliente(z, float(x), float(y))
        for u, z, x, y in zip(d["_uid"], d[cfg.COL_ASESOR], d["x_km"], d["y_km"])],
        dtype=int)
    ctf = carga_por_territorio_fecha(d, labels=labels_todo)
    fact = factibilidad(ctf)
    ctf_rut = pd.DataFrame([{"fecha": r["fecha"], "territorio": r["territorio"],
                             "visitas": r["n_clientes"]} for r in rutas])
    if len(ctf_rut):
        ctf_rut["q"] = ctf_rut["fecha"].map(quincena_de)
    fact_plan = factibilidad(ctf_rut)

    uids_plan = [int(u) for r in rutas for u in r["uid"]]
    uids_train = [u for u in uids_plan if u in modelo.vistos]
    uids_uniq_train = set(uids_train)
    out: dict[str, Any] = {
        "estrategia": modelo.estrategia,
        "K": modelo.K,
        "n_train": int(len(conocidos)),
        "n_test": int(len(nuevos)),
        "n_rutas": len(rutas),
        "n_territorios": len({int(r["territorio"]) for r in rutas}),
        "km_train": round(km_train, 1),
        "km_total": round(km_de_rutas(rutas), 1),
        # cobertura: TODOS los clientes de ajuste terminaron en alguna ruta
        # (un cliente con varias visitas aparece en varias rutas: se compara por conjunto)
        "cobertura_train_exacta": bool(uids_uniq_train
                                       == set(int(u) for u in conocidos["_uid"])),
        "pct_clientes_plan": round(100.0 * len(set(uids_plan)) / len(d), 2)
        if len(d) else 0.0,
        "factibilidad": fact,                                   # todas las visitas
        "factibilidad_plan_ruteado": fact_plan,                 # sólo lo que cabió
        "visitas_programadas": int(ctf["visitas"].sum()) if len(ctf) else 0,
        "insercion": ins,
        "rutas": rutas,
    }
    if asistencia is not None and trafico is not None:
        ev = evaluar_estrategia(rutas, asistencia, trafico)
        jor = jornadas_por_escenario(rutas, asistencia, trafico)
        extra_h = jor["horas_extra"]
        costo = _costo(ev, trafico, modelo.K, extra_h)
        res = _resumen(ev, costo, modelo.K, fact,
                       resumen_jornadas(jor, trafico), out["visitas_programadas"],
                       jor["visitas_completadas"].mean())
        out["montecarlo"] = res
        out["costo_total"] = np.asarray(costo["total"])
        out["km_escenario"] = ev["km"]
    return out


def _costo(ev, trafico, n_asesores, extra_h):
    from .montecarlo import costo_escenario
    return costo_escenario(ev, trafico, n_asesores, horas_extra_h=extra_h)


def _resumen(ev, costo, n_asesores, fact, horas, visitas_prog, completadas):
    from .montecarlo import resumen_estrategia
    return resumen_estrategia("eval", ev, costo, n_asesores, fact, horas,
                              visitas_prog, visitas_completadas=completadas)


# ---------------------------------------------------------------------------
# Test automático de fuga
# ---------------------------------------------------------------------------
def verificar_fuga(modelo: ModeloRuteo, test: pd.DataFrame,
                   train: pd.DataFrame | None = None) -> dict[str, Any]:
    """¿Ningún `_uid` de test entró a `ajustar()` y los depósitos son de train?

    Además deja visible la ÚNICA excepción declarada del protocolo: el re-balanceo
    F2 (`cfg.REBAL_POBLACION="todos"`) se calcula con los atributos de ENTRADA de
    todos los clientes (zona, fechas, visitas, coordenadas), nunca con resultados
    o métricas. Si `cfg.REBAL_POBLACION="train"` no hay excepción.
    """
    test_uids = set(int(u) for u in test.get("_uid", pd.Series([], dtype=int)))
    if test_uids and "_uid" not in test.columns:
        test_uids = set()
    filtradas = sorted(set(modelo.vistos) & test_uids)

    excepcion = bool(modelo.estrategia == "rebalanceo"
                     and cfg.REBAL_POBLACION == "todos")

    depositos_ok = True
    detalle = None
    if train is not None and len(train):
        d = train.copy().reset_index(drop=True)
        if "_uid" not in d.columns:
            d["_uid"] = np.arange(len(d))
        d["_t"] = [modelo.territorio_de[int(u)] for u in d["_uid"]]
        max_err = 0.0
        for t, g in d.groupby("_t"):
            lat, lon = float(g["Latitud"].mean()), float(g["Longitud"].mean())
            dl, dn = modelo.depositos[int(t)]
            max_err = max(max_err, abs(lat - dl), abs(lon - dn))
        depositos_ok = bool(max_err < 1e-9)
        detalle = round(float(max_err), 12)

    return {
        "ok": bool(not filtradas and depositos_ok),
        "n_test": len(test_uids),
        "n_vistos": modelo.n_vistos,
        "uids_test_en_ajuste": filtradas[:50],
        "n_uids_test_en_ajuste": len(filtradas),
        "depositos_solo_de_train": depositos_ok,
        "max_error_deposito": detalle,
        "excepcion_rebalanceo_declarada": excepcion,
        "nota_excepcion_rebalanceo": (
            "Re-balanceo F2 (cfg.REBAL_POBLACION='todos'): la reparación estructural "
            "usa sólo atributos de entrada (ZON_REP, DIAS_REP, FRE_REP, visitas y "
            "coordenadas) de todos los clientes; no usa resultados ni métricas de "
            "test. Territorios, depósitos, β, K y λ siguen saliendo del train."
            if excepcion else None),
        "nota": (f"Las {modelo.n_vistos} filas de ajuste (territorios, depósitos, β, K "
                 f"y λ) no incluyen ningún `_uid` de test ({len(test_uids)} en el "
                 f"conjunto de prueba)."),
    }
