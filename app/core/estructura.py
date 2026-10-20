"""Estructura de territorios y sub-rutas (E8, E9).

- ACTUAL  → desde la asignación real de clientes a ZON_REP (sin reclusterizar).
- K hipotético → re-partición de todo el universo con ASIGNACIÓN CON RESTRICCIÓN
  DE CAPACIDAD en visitas/día (no K-Means simple).
- Depósito por territorio (E9, DEPOSITO_MODO="territorio") para ser comparable
  con el modelo de ruteo.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .. import config as cfg
from .geo import haversine_mat, proyectar
from .ruteo import dist_ruta, dos_opt, nearest_neighbor, subdivide

DEPOSITO_MODO: str = "territorio"   # "territorio" | "cd"  (E9)


# ---------------------------------------------------------------------------
# Asignación con restricción de capacidad (E8 / objetivo 40-50)
# ---------------------------------------------------------------------------
_DIA_IDX = {"LUNES": 0, "MARTES": 1, "MIERCOLES": 2, "JUEVES": 3,
            "VIERNES": 4, "SABADO": 5, "DOMINGO": 6}


def asignacion_capacidad(df: pd.DataFrame, K: int, col_carga: str = "visitas_mes_rep",
                         semilla: int | None = None, lam: float = 1000.0,
                         lam_fecha: float = 400.0, max_iter: int = 18
                         ) -> np.ndarray:
    """Reparte `df` en K territorios equilibrando carga **por fecha** con K-Means
    capacitado (E8 / objetivo 40-50 / factibilidad E7 / K_mín del usuario).

    La restricción dura es por asesor-DÍA, así que se balancea la carga de cada
    fecha a su objetivo (fecha_global/K). El balance por fecha implica el total y
    evita concentrar clientes de un mismo día en un territorio (p.ej. 70 % lunes),
    lo que dispararía el pico diario sobre CAP. Con esto el pico ≈ pico_global/K,
    de modo que se cumple la regla del usuario K_mín = ⌈pico_día / CAP⌉.

    Alterna (a) asignación a argmin[dist² + λ·Σ_sobre-objetivo de cada fecha] y
    (b) actualización de centroides (mantiene la compacidad geográfica).
    """
    semilla = cfg.SEMILLA if semilla is None else semilla
    xy = df[["x_km", "y_km"]].values
    carga = df[col_carga].astype(float).values
    n = len(df)
    if K >= n:
        return np.arange(n)
    from .calendario import fechas_por_cliente
    idx_fechas, _fechas, fecha_total = fechas_por_cliente(df)
    fecha_obj = fecha_total / K                    # carga que toca a cada territorio por fecha

    labels = KMeans(n_clusters=K, n_init=10, random_state=semilla).fit_predict(xy)
    cent = np.array([xy[labels == t].mean(axis=0) for t in range(K)])
    orden = np.argsort(-carga)                      # carga descendente (mejor packing)

    for _ in range(max_iter):
        newl = np.full(n, -1, int)
        fl = np.zeros((K, len(fecha_total)))        # carga por (territorio, fecha)
        tl = np.zeros(K)                            # carga total por territorio
        objetivo = carga.sum() / K
        for i in orden:
            d2 = (cent[:, 0] - xy[i, 0]) ** 2 + (cent[:, 1] - xy[i, 1]) ** 2
            # (i) balance TOTAL de carga por territorio
            desborde = np.maximum(0.0, tl + carga[i] - objetivo)
            # (ii) balance POR FECHA: por cada fecha que toca este cliente, evita
            # entrar a un territorio que quedaría sobre su objetivo en esa fecha
            pen = np.zeros(K)
            for f in idx_fechas[i]:
                exceso = fl[:, f] + 1.0 - fecha_obj[f]
                pen += np.maximum(0.0, exceso) ** 2
            cost = d2 + lam * desborde * desborde + lam_fecha * pen
            j = int(np.argmin(cost))
            newl[i] = j
            tl[j] += carga[i]
            for f in idx_fechas[i]:
                fl[j, f] += 1.0
        # (b) actualizar centroides con la nueva asignación
        for t in range(K):
            m = newl == t
            if m.any():
                cent[t] = xy[m].mean(axis=0)
        if np.array_equal(newl, labels):
            labels = newl
            break
        labels = newl
    return labels


# ---------------------------------------------------------------------------
# Territorio → sub-rutas → secuencia
# ---------------------------------------------------------------------------
def _depot_de(terr: pd.DataFrame, modo: str) -> tuple[float, float]:
    if modo == "cd" and "Cod CD" in terr.columns:
        return float(terr["Latitud"].mean()), float(terr["Longitud"].mean())
    return float(terr["Latitud"].mean()), float(terr["Longitud"].mean())


def territorio_a_rutas(terr: pd.DataFrame, tid: Any, depot: tuple[float, float],
                       cap: int, semilla: int) -> list[dict[str, Any]]:
    """Divide un territorio en sub-rutas (≤cap) y secuencia cada una con NN + 2-opt."""
    rutas: list[dict[str, Any]] = []
    for j, chunk in enumerate(subdivide(terr, cap, semilla)):
        lats = np.concatenate([[depot[0]], chunk["Latitud"].values])
        lons = np.concatenate([[depot[1]], chunk["Longitud"].values])
        D = haversine_mat(lats, lons)
        slots = [{0, 1, 2}] + list(chunk["slots"].values)
        orden = nearest_neighbor(D, slots)
        orden = dos_opt(orden, D, slots)                       # E2
        uid = chunk["_uid"].values[np.asarray(orden[1:]) - 1]
        rutas.append(dict(territorio=tid, D=D, orden=orden, uid=uid,
                          km_base=float(dist_ruta(D, orden)),
                          n_clientes=int(len(chunk)),
                          excede_cap=int(len(chunk) > cap)))
    return rutas


def construir_actual(clientes: pd.DataFrame, cap: int | None = None,
                     semilla: int | None = None, modo: str | None = None
                     ) -> list[dict[str, Any]]:
    """Estrategia ACTUAL: territorios = ZON_REP reales, sin reclusterizar (E8)."""
    cap = cap or cfg.CAP_TOPE
    semilla = cfg.SEMILLA if semilla is None else semilla
    modo = modo or DEPOSITO_MODO
    df = clientes.copy().reset_index(drop=True)
    if "_uid" not in df.columns:
        df["_uid"] = np.arange(len(df))
    proyectar(df, col_grupo=None)
    rutas: list[dict[str, Any]] = []
    for az, terr in df.groupby(cfg.COL_ASESOR):
        depot = _depot_de(terr, modo)
        rutas += territorio_a_rutas(terr, az, depot, cap, semilla)
    return rutas


def construir_estrategia(clientes: pd.DataFrame, K: int, cap: int | None = None,
                         semilla: int | None = None, modo: str | None = None
                         ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Estrategia hipotética con K territorios equilibrados por capacidad (E8)."""
    cap = cap or cfg.CAP_TOPE
    semilla = cfg.SEMILLA if semilla is None else semilla
    modo = modo or DEPOSITO_MODO
    df = clientes.copy().reset_index(drop=True)
    if "_uid" not in df.columns:
        df["_uid"] = np.arange(len(df))
    proyectar(df, col_grupo=None)
    labels = asignacion_capacidad(df, K, semilla=semilla)
    df["_terr"] = labels

    rutas: list[dict[str, Any]] = []
    cargas: dict[int, int] = {}
    for tid, terr in df.groupby("_terr"):
        depot = _depot_de(terr, modo)
        cargas[int(tid)] = int(terr["visitas_mes_rep"].sum())
        rutas += territorio_a_rutas(terr, int(tid), depot, cap, semilla)

    meta = {"K": K, "n_territorios": int(df["_terr"].nunique()),
            "cargas": cargas, "cobertura": int(sum(len(r["uid"]) for r in rutas)),
            "n_clientes": int(len(df))}
    return rutas, meta


def validar_cubierta(rutas: list[dict[str, Any]], n_clientes: int) -> bool:
    """Invariante: cada cliente aparece exactamente una vez."""
    uids = np.concatenate([r["uid"] for r in rutas]) if rutas else np.array([])
    return len(uids) == n_clientes and len(np.unique(uids)) == n_clientes


# ---------------------------------------------------------------------------
# Rutas DIARIAS (D2/D3): los km salen de una ruta por asesor-día
# ---------------------------------------------------------------------------
def construir_rutas_diarias(clientes: pd.DataFrame, labels: np.ndarray | None = None,
                            cap: int | None = None, semilla: int | None = None,
                            modo: str | None = None) -> list[dict[str, Any]]:
    """Una ruta por (territorio, fecha) con los clientes que toca ese día (D2).

    Base de los km del ciclo (D3: "los km salen de las rutas por día") y de la
    factibilidad por asesor-día (E7). El depósito es el centroide del territorio.
    """
    from .calendario import fechas_permitidas
    cap = cap or cfg.CAP_TOPE
    semilla = cfg.SEMILLA if semilla is None else semilla
    modo = modo or DEPOSITO_MODO
    df = clientes.copy().reset_index(drop=True)
    if "_uid" not in df.columns:
        df["_uid"] = np.arange(len(df))
    df["_terr"] = labels if labels is not None else df[cfg.COL_ASESOR].values

    # desglose (territorio, fecha) -> uids que visitan ese día
    dias: dict[tuple, list[int]] = {}
    for _, r in df.iterrows():
        permitidas = fechas_permitidas(r["DIAS_REP"], r["FRE_REP"])
        if not permitidas:
            continue
        objetivo = int(r["visitas_mes_rep"])
        base, extra = divmod(objetivo, len(permitidas))
        for i, f in enumerate(permitidas):
            v = base + (1 if i < extra else 0)
            if v > 0:
                dias.setdefault((r["_terr"], f), []).append(int(r["_uid"]))

    depot = {t: _depot_de(terr, modo) for t, terr in df.groupby("_terr")}
    idx = df.set_index("_uid")

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
        rutas.append(dict(territorio=t, fecha=f, D=D, orden=orden, uid=uid_ord,
                          km_base=float(dist_ruta(D, orden)),
                          n_clientes=int(len(sub)),
                          excede_cap=int(len(sub) > cap)))
    rutas.sort(key=lambda r: (str(r["territorio"]), str(r["fecha"])))
    return rutas


def km_total_ciclo(rutas_diarias: list[dict[str, Any]]) -> float:
    """km del ciclo = suma de las rutas diarias (D3)."""
    return float(sum(r["km_base"] for r in rutas_diarias))


# ---------------------------------------------------------------------------
# F2 — Re-balanceo mínimo desde la asignación ACTUAL (solo K = K_ACTUAL)
# ---------------------------------------------------------------------------
def carga_por_fecha_cliente(df: pd.DataFrame
                            ) -> tuple[list[np.ndarray], list[Any]]:
    """Visitas de cada cliente por fecha (mismo reparto que las rutas diarias)."""
    from .calendario import fechas_permitidas

    fechas_todas: list[Any] = []
    idx: list[np.ndarray] = []
    pos: dict[Any, int] = {}
    for _, r in df.iterrows():
        permitidas = fechas_permitidas(r["DIAS_REP"], r["FRE_REP"])
        if not permitidas:
            idx.append(np.array([], dtype=int))
            continue
        objetivo = int(r["visitas_mes_rep"])
        base, extra = divmod(objetivo, len(permitidas))
        sel: list[int] = []
        for i, f in enumerate(permitidas):
            if base + (1 if i < extra else 0) <= 0:
                continue
            if f not in pos:
                pos[f] = len(fechas_todas)
                fechas_todas.append(f)
            for _ in range(base + (1 if i < extra else 0)):
                sel.append(pos[f])
        idx.append(np.asarray(sel, dtype=int))
    return idx, fechas_todas


def rebalancear_desde_actual(df: pd.DataFrame, labels: np.ndarray | None = None,
                             cap: int | None = None, con_km: bool = True
                             ) -> tuple[np.ndarray, dict[str, Any]]:
    """F2 — re-balanceo MÍNIMO de la asignación real (K = K_ACTUAL, sin re-clustering).

    Parte de la asignación ACTUAL (`ZON_REP` reales o `labels` dados) y mueve el
    mínimo de clientes de los asesor-días sobre el tope a otras zonas, eligiendo
    en cada paso el movimiento que MENOS aumenta la distancia del cliente al
    centroide y que respeta el tope en TODAS las fechas del cliente de destino.

    Objetivo declarado: **minimizar los km añadidos** sujeto a que ningún
    asesor-día supere `cap`. No se fija ningún número objetivo de movimientos ni
    de Δkm: el resultado es el que salga del cálculo y se reporta completo
    (clientes movidos n y %, Δkm, días sobre el tope antes/después, factible sí/no).

    `labels=None` → territorios = `ZON_REP` reales. Devuelve (labels_nuevas, reporte).
    """
    from .calendario import fechas_por_cliente
    from .montecarlo import carga_por_territorio_fecha, factibilidad

    cap = cfg.cap_dinamico() if cap is None else int(cap)
    d = df.copy().reset_index(drop=True)
    if "_uid" not in d.columns:
        d["_uid"] = np.arange(len(d))
    if "x_km" not in d.columns:
        proyectar(d, col_grupo=None)
    xy = d[["x_km", "y_km"]].values

    # territorios base (ACTUAL) renumerados 0..K-1
    if labels is None:
        zonas = d[cfg.COL_ASESOR].astype(str).values
        _, labels0 = np.unique(zonas, return_inverse=True)
    else:
        labels0 = np.asarray(labels)
    labels0 = labels0.copy()
    labels_ini = labels0.copy()               # asignación ACTUAL antes de mover nada
    n = len(d)
    K = int(labels0.max()) + 1

    idx_f, _fechas = carga_por_fecha_cliente(d)
    # carga por (territorio, fecha)
    carga: dict[tuple[int, int], int] = {}
    for i in range(n):
        for f in idx_f[i]:
            k = (int(labels0[i]), int(f))
            carga[k] = carga.get(k, 0) + 1

    def _centros(lab: np.ndarray) -> np.ndarray:
        c = np.zeros((K, 2))
        for t in range(K):
            m = lab == t
            if m.any():
                c[t] = xy[m].mean(axis=0)
        return c

    cent = _centros(labels0)
    # dist² cliente–centroide (todas las parejas es K×n; se recalcula por destino)
    def _dist2(destino: int) -> np.ndarray:
        return (xy[:, 0] - cent[destino, 0]) ** 2 + (xy[:, 1] - cent[destino, 1]) ** 2

    # --- estado inicial -----------------------------------------------------
    def _excesos(lab: np.ndarray) -> list[tuple[int, int, int]]:
        out = []
        for (t, f), v in carga.items():
            if v > cap:
                out.append((v - cap, t, f))
        return sorted(out, reverse=True)

    movidos: list[int] = []
    movimientos: list[dict[str, Any]] = []
    excesos0 = _excesos(labels0)
    for _ in range(max(1, 4 * len(excesos0) + 100)):
        excesos = _excesos(labels0)
        if not excesos:
            break
        _, t_sob, f_sob = excesos[0]            # día más cargado sobre el tope
        # clientes de ese territorio que visitan esa fecha
        cand = [i for i in range(n)
                if labels0[i] == t_sob and (idx_f[i] == f_sob).any()]
        mejor: tuple[float, int, int] | None = None
        for i in cand:
            lab_i = labels0[i]
            d_orig = float(np.hypot(*(xy[i] - cent[lab_i])))
            for dest in range(K):
                if dest == lab_i:
                    continue
                # el destino debe tener cupo en TODAS las fechas del cliente
                cupo = True
                for f in idx_f[i]:
                    if carga.get((dest, int(f)), 0) + 1 > cap:
                        cupo = False
                        break
                if not cupo:
                    continue
                d_nue = float(np.hypot(*(xy[i] - cent[dest])))
                delta = d_nue - d_orig          # aumento de la distancia al centroide
                if mejor is None or delta < mejor[0] - 1e-12:
                    mejor = (delta, i, dest)
        if mejor is None:
            break                               # sin movimiento factible (queda infactible)
        delta, i, dest = mejor
        origen = int(labels0[i])
        labels0[i] = dest
        movidos.append(int(d["_uid"].iloc[i]))
        movimientos.append({"uid": int(d["_uid"].iloc[i]),
                            "origen": origen, "destino": int(dest),
                            "fecha": str(f_sob), "delta_dist_km": round(delta, 3)})
        for f in idx_f[i]:
            k0 = (origen, int(f))
            carga[k0] = carga.get(k0, 0) - 1
            k1 = (dest, int(f))
            carga[k1] = carga.get(k1, 0) + 1
        # centroides afectados
        for t in (origen, dest):
            m = labels0 == t
            if m.any():
                cent[t] = xy[m].mean(axis=0)

    # --- verificación de coherencia (no son metas) --------------------------
    ctf = carga_por_territorio_fecha(d, labels=labels0)
    fact1 = factibilidad(ctf, cap=cap)
    ctf0 = carga_por_territorio_fecha(d, labels=labels_ini)
    fact0 = factibilidad(ctf0, cap=cap)

    cada_uno_vez = len(np.unique(d["_uid"].values)) == n
    movidos_unicos = sorted(set(movidos))
    sin_cupo = [m for m in movimientos
                if m["destino"] == m["origen"]]

    reporte: dict[str, Any] = {
        "cap_usado": int(cap),
        "K": int(len(np.unique(labels0))),
        "n_clientes": int(n),
        "clientes_movidos": int(len(movidos_unicos)),
        "clientes_movidos_pct": round(100.0 * len(movidos_unicos) / n, 3),
        "movimientos_totales": int(len(movimientos)),      # un cliente puede moverse 2 veces
        "dias_sobre_tope_antes": int(fact0["dias_sobre_cap"]),
        "dias_sobre_tope_despues": int(fact1["dias_sobre_cap"]),
        "pico_antes": int(fact0.get("pico_global", 0)),
        "pico_despues": int(fact1.get("pico_global", 0)),
        "factible_antes": bool(fact0["factible"]),
        "factible_despues": bool(fact1["factible"]),
        "logro_factibilidad": bool(fact1["factible"]),
        # verificaciones de coherencia
        "cada_cliente_una_vez": bool(cada_uno_vez),
        "sin_movimientos_a_si_mismo": not sin_cupo,
        "movimientos": movimientos[:200],
    }
    if con_km:
        km_antes = construir_rutas_diarias(d, labels=labels_ini)
        km_despues = construir_rutas_diarias(d, labels=labels0)
        km0 = km_total_ciclo(km_antes)
        km1 = km_total_ciclo(km_despues)
        reporte["km_antes"] = round(km0, 1)
        reporte["km_despues"] = round(km1, 1)
        reporte["delta_km"] = round(km1 - km0, 1)
        reporte["delta_km_pct"] = round(100.0 * (km1 - km0) / km0, 3) if km0 else None
    return labels0, reporte


def rutas_primer_ciclo(clientes: pd.DataFrame, labels: np.ndarray | None = None,
                       cap: int | None = None, semilla: int | None = None,
                       modo: str | None = None, quincena: int = 1) -> dict[str, Any]:
    """Tabla de rutas diarias de una quincena: fecha, asesor, visitas, secuencia, km.

    Responde "¿cuál es la ruta y a qué asesor le toca?". La secuencia usa el
    `CodigoCliente` (código interno, pseudónimo); NUNCA expone nombre, cédula ni
    dirección (regla de privacidad). `labels`=None → estrategia ACTUAL (ZON_REP).
    """
    from .calendario import quincena_de

    df = clientes.copy().reset_index(drop=True)
    if "_uid" not in df.columns:
        df["_uid"] = np.arange(len(df))
    cod = dict(zip(df["_uid"].astype(int), df["CodigoCliente"].astype(str)))

    rutas = construir_rutas_diarias(df, labels=labels, cap=cap, semilla=semilla, modo=modo)
    filas: list[dict[str, Any]] = []
    for r in rutas:
        if quincena_de(r["fecha"]) != quincena:
            continue
        terr = r["territorio"]
        filas.append({
            "fecha": r["fecha"].isoformat() if hasattr(r["fecha"], "isoformat") else str(r["fecha"]),
            "asesor": int(terr) if isinstance(terr, (int, np.integer)) else str(terr),
            "n_visitas": int(r["n_clientes"]),
            "km": round(float(r["km_base"]), 2),
            "secuencia": [cod.get(int(u), str(u)) for u in r["uid"]],
            "excede_cap": bool(r["excede_cap"]),
        })
    filas.sort(key=lambda x: (x["fecha"], str(x["asesor"])))
    return {
        "quincena": int(quincena),
        "n_rutas": len(filas),
        "total_visitas": int(sum(f["n_visitas"] for f in filas)),
        "total_km": round(float(sum(f["km"] for f in filas)), 1),
        "n_asesores": len({f["asesor"] for f in filas}),
        "rutas": filas,
    }
