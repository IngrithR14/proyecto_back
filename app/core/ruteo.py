"""Motor de ruteo: ventanas, NN, ACO, 2-opt y formación de sub-rutas.

Correcciones aplicadas:
  E2 — ruta abierta: la ventana del ÚLTIMO cliente se verifica y el 2-opt puede
       invertir tramos que incluyen el último nodo.
  E3 — calibración de β con muestra estratificada por tamaño (no las más pequeñas).
  E4 — K_EXTRA_MAX ancla k=k_cap; métrica de ocupación; fusión de sub-rutas mínimas.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .. import config as cfg
from .geo import haversine_mat


# ---------------------------------------------------------------------------
# Ventanas horarias (E2)
# ---------------------------------------------------------------------------
def slots_factibles(slots_orden) -> bool:
    """¿La secuencia de ventanas es servible en orden creciente de slot? (0→1→2)."""
    actual = 0
    for s in slots_orden:
        while actual not in s and actual <= 2:
            actual += 1
        if actual > 2:
            return False
    return True


def segmento_servible(orden) -> list:
    """Nodos cuya ventana debe verificarse.

    Ruta abierta (E2): el último cliente es real → `orden[1:]`.
    Ruta cerrada: el depósito cierra la ruta → `orden[1:-1]`.
    """
    if cfg.RUTA_CERRADA and len(orden) > 1 and orden[0] == orden[-1]:
        return list(orden[1:-1])
    if cfg.RUTA_CERRADA:
        # el depósito no está en `orden`; se asume cerrado solo si se agregó
        return list(orden[1:-1]) if len(orden) > 1 else []
    return list(orden[1:])


def factible_orden(orden, slots) -> bool:
    """True si la secuencia respeta las ventanas de todos los clientes servidos (E2)."""
    return slots_factibles([slots[i] for i in segmento_servible(orden)])


# ---------------------------------------------------------------------------
# Recorrido NN
# ---------------------------------------------------------------------------
def nearest_neighbor(D, slots) -> list[int]:
    """Recorrido voraz desde el depósito (índice 0) respetando ventanas."""
    n = D.shape[0]
    visitados = np.zeros(n, bool)
    visitados[0] = True
    orden = [0]
    actual = 0
    while len(orden) < n:
        u = orden[-1]
        cand = [v for v in range(n) if not visitados[v] and actual in slots[v]]
        if not cand:
            actual += 1
            if actual > 2:
                cand = [v for v in range(n) if not visitados[v]]
                if not cand:
                    break
            continue
        v = min(cand, key=lambda i: D[u, i])
        visitados[v] = True
        orden.append(v)
    return orden


def dist_ruta(D, orden) -> float:
    """Distancia de la secuencia; si RUTA_CERRADA, regresa al depósito."""
    if len(orden) < 2:
        return 0.0
    d = float(D[orden[:-1], orden[1:]].sum())
    if cfg.RUTA_CERRADA and orden[0] != orden[-1]:
        d += float(D[orden[-1], orden[0]])
    return d


def matriz_subruta(ruta: pd.DataFrame, depot: tuple[float, float]):
    """Construye D (haversine) y slots con el depósito en el índice 0."""
    lats = np.concatenate([[depot[0]], ruta["Latitud"].values])
    lons = np.concatenate([[depot[1]], ruta["Longitud"].values])
    D = haversine_mat(lats, lons)
    slots = [{0, 1, 2}] + list(ruta["slots"].values)
    return D, slots, lats, lons


# ---------------------------------------------------------------------------
# 2-opt (E2: puede invertir hasta el último nodo en ruta abierta)
# ---------------------------------------------------------------------------
def dos_opt(orden, D, slots) -> list[int]:
    """Mejora 2-opt; solo acepta inversiones factibles por ventana.

    En ruta abierta se permiten inversiones hasta el último nodo (E2). Evalúa los
    candidatos con la fórmula de delta O(1) y sólo confirma con distancia completa
    al aceptar (el 2-opt ingenuo era O(n³) y recalculaba todo en cada candidato).
    """
    orden = list(orden)
    if len(orden) < 4:
        return orden
    cerrada = bool(cfg.RUTA_CERRADA)
    while True:
        n = len(orden)
        j_max = n - 2 if cerrada else n - 1     # abierta: incluye el último nodo (E2)
        if j_max < 2:
            break
        movido = False
        for i in range(1, n - 1):
            if movido:
                break
            for j in range(i + 1, j_max + 1):
                o = orden
                a, fs, ls = o[i - 1], o[i], o[j]
                # delta de invertir [i..j]: sólo cambian los bordes
                delta = D[a, ls] - D[a, fs]
                if j + 1 < n:
                    delta += D[fs, o[j + 1]] - D[ls, o[j + 1]]
                if cerrada and j == n - 1:
                    delta += D[fs, o[0]] - D[ls, o[0]]      # arista de regreso
                if delta >= -1e-9:
                    continue
                nuevo = o[:i] + o[i:j + 1][::-1] + o[j + 1:]
                if not factible_orden(nuevo, slots):        # ventana (E2)
                    continue
                if dist_ruta(D, nuevo) < dist_ruta(D, orden) - 1e-9:
                    orden = nuevo
                    movido = True
                    break
        if not movido:
            break
    return orden


# ---------------------------------------------------------------------------
# ACO
# ---------------------------------------------------------------------------
def _construir(tau, eta, D, slots, kvec, alfa, beta, rng):
    n = D.shape[0]
    visitados = np.zeros(n, bool)
    visitados[0] = True
    orden = [0]
    actual = 0
    while len(orden) < n:
        u = orden[-1]
        cand = [v for v in kvec[u] if not visitados[v] and actual in slots[v]]
        if not cand:
            cand = [v for v in range(n) if not visitados[v] and actual in slots[v]]
        if not cand:
            actual += 1
            if actual > 2:
                cand = [v for v in range(n) if not visitados[v]]
                if not cand:
                    break
            continue
        cand = np.array(cand)
        peso = (tau[u, cand] ** alfa) * (eta[u, cand] ** beta)
        suma = peso.sum()
        if suma <= 0 or not np.isfinite(suma):
            v = int(rng.choice(cand))
        else:
            v = int(rng.choice(cand, p=peso / suma))
        visitados[v] = True
        orden.append(v)
    return orden


def aco(D, slots, m, alfa, beta, rho, Q, iteraciones, semilla, kvecinos=15):
    """Colonia de hormigas + 2-opt (E2) sobre una sub-ruta."""
    n = D.shape[0]
    K = min(kvecinos, n - 1)
    order = np.argsort(D, axis=1)
    kvec = [[v for v in order[u] if v != u][:K] for u in range(n)]
    eta = np.zeros_like(D)
    nz = D > 0
    eta[nz] = 1.0 / D[nz]
    tau = np.ones((n, n))
    rng = np.random.default_rng(semilla)
    mejor, md = None, np.inf
    historial = []
    for _ in range(iteraciones):
        for _ in range(m):
            o = _construir(tau, eta, D, slots, kvec, alfa, beta, rng)
            d = dist_ruta(D, o)
            if d < md:
                md, mejor = d, o[:]
        tau *= (1 - rho)
        for a, b in zip(mejor[:-1], mejor[1:]):
            tau[a, b] += Q / md
            tau[b, a] += Q / md
        historial.append(md)
    mejor = dos_opt(mejor, D, slots)
    return mejor, dist_ruta(D, mejor), historial


# ---------------------------------------------------------------------------
# Formación de sub-rutas (E4)
# ---------------------------------------------------------------------------
def subdivide(chunk: pd.DataFrame, cap: int, semilla: int) -> list[pd.DataFrame]:
    """Divide recursivamente hasta que ninguna sub-ruta supere `cap`."""
    if len(chunk) <= cap:
        return [chunk]
    k2 = int(np.ceil(len(chunk) / cap))
    km2 = KMeans(n_clusters=k2, n_init=10, random_state=semilla).fit(chunk[["x_km", "y_km"]])
    tmp = chunk.copy()
    tmp["_c2"] = km2.labels_
    res: list[pd.DataFrame] = []
    for _, s2 in tmp.groupby("_c2"):
        res += subdivide(s2.drop(columns=["_c2"]), cap, semilla)
    return res


def formar_subrutas(grupo_train: pd.DataFrame, grupo_all: pd.DataFrame, cap: int,
                    semilla: int, k_extra_max: float | None = None) -> tuple[list, int, float]:
    """Ajusta K-Means sobre train y forma sub-rutas sobre train+test (E4).

    `k_extra_max` (default cfg.K_EXTRA_MAX=1.0) ancla k en [k_cap, k_extra_max*k_cap].
    """
    k_extra_max = cfg.K_EXTRA_MAX if k_extra_max is None else k_extra_max
    xy_train = grupo_train[["x_km", "y_km"]].values
    n = len(grupo_train)
    k_cap = max(1, int(np.ceil(n / cap)))

    k_sil = k_cap
    mejor = np.nan
    if n > 2 * k_cap and k_extra_max > 1.0:
        k_max = min(n - 1, max(k_cap, int(np.ceil(k_extra_max * k_cap))))
        k_max = max(k_max, k_cap + 1)
        idx = np.linspace(0, n - 1, min(n, 1500)).astype(int)
        best_s = -1.0
        for k in range(k_cap, k_max + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(xy_train)
            if len(set(km.labels_)) < 2:
                continue
            s = silhouette_score(xy_train[idx], km.labels_[idx])
            if s > best_s:
                best_s, k_sil = s, k
        mejor = best_s
    k = max(k_sil, k_cap)

    km = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(xy_train)
    g = grupo_all.copy()
    g["_cl"] = km.predict(g[["x_km", "y_km"]].values)

    rutas: list[pd.DataFrame] = []
    for _, cl in g.groupby("_cl"):
        rutas += subdivide(cl.drop(columns=["_cl"]), cap, semilla)
    return rutas, k, float(mejor)


def ocupacion(rutas: list[pd.DataFrame], cap: int) -> pd.Series:
    """E4: ocupación = clientes/CAP por sub-ruta."""
    return pd.Series([len(r) / cap for r in rutas])


def fusionar_subrutas(rutas: list[pd.DataFrame], cap: int,
                      min_clientes: int | None = None) -> list[pd.DataFrame]:
    """E4: fusiona sub-rutas con < min_clientes con la más cercana del mismo grupo
    si la suma no supera `cap`."""
    min_clientes = cfg.MIN_CLIENTES_RUTA if min_clientes is None else min_clientes
    if not rutas:
        return rutas
    grupos: dict[str, list[pd.DataFrame]] = {}
    for r in rutas:
        gid = r["grupo_id"].iloc[0] if "grupo_id" in r.columns else "_"
        grupos.setdefault(gid, []).append(r)

    out: list[pd.DataFrame] = []
    for gid, rs in grupos.items():
        rs = sorted(rs, key=len, reverse=True)
        pequenas = [r for r in rs if len(r) < min_clientes]
        grandes = [r for r in rs if len(r) >= min_clientes]
        for p in pequenas:
            if not grandes:
                grandes = [p]
                continue
            # más cercana por centroide
            cx, cy = p["x_km"].mean(), p["y_km"].mean()
            dists = [(abs(g["x_km"].mean() - cx) + abs(g["y_km"].mean() - cy), i)
                     for i, g in enumerate(grandes)]
            dists.sort()
            fusionada = False
            for _, i in dists:
                if len(grandes[i]) + len(p) <= cap:
                    grandes[i] = pd.concat([grandes[i], p], ignore_index=True)
                    fusionada = True
                    break
            if not fusionada:
                grandes.append(p)  # no cabe: se conserva separada
        out.extend(grandes)
    return out


# ---------------------------------------------------------------------------
# E3 — calibración de β estratificada
# ---------------------------------------------------------------------------
RANGOS_TAMANO: tuple[tuple[int, int], ...] = ((1, 20), (21, 30), (31, 40), (41, 10_000))


def muestra_estratificada(rutas: list[pd.DataFrame], por_rango: int = 2) -> list[pd.DataFrame]:
    """E3: sub-rutas por rango de tamaño (2 por cada rango ≤20, 21-30, 31-40, 41-45)."""
    seleccion: list[pd.DataFrame] = []
    for lo, hi in RANGOS_TAMANO:
        cand = [r for r in rutas if lo <= len(r) <= hi]
        cand.sort(key=len)
        seleccion.extend(cand[:por_rango])
    return seleccion


def rangos_cubiertos(muestra: list[pd.DataFrame]) -> list[bool]:
    """E3 test: ¿la muestra cubre los 4 rangos?"""
    cubiertos = []
    for lo, hi in RANGOS_TAMANO:
        cubiertos.append(any(lo <= len(r) <= hi for r in muestra))
    return cubiertos


def calibrar_beta(rutas, DEP, aco_params: dict, semilla: int,
                  grid=None, seeds: int | None = None) -> tuple[float, pd.DataFrame]:
    """E3: elige β con muestra estratificada y `seeds` semillas por β."""
    grid = cfg.BETA_GRID if grid is None else grid
    seeds = cfg.BETA_SEEDS if seeds is None else seeds
    muestra = muestra_estratificada(rutas)
    filas = []
    mejor_beta, mejor_prom = None, np.inf
    for beta in grid:
        tots = []
        for s in range(seeds):
            tot = 0.0
            for ruta in muestra:
                gid = ruta["grupo_id"].iloc[0]
                dep = DEP[gid]
                D, slots, _, _ = matriz_subruta(ruta, (dep["dep_lat"], dep["dep_lon"]))
                _, d, _ = aco(D, slots, m=6, alfa=aco_params["alfa"], beta=beta,
                              rho=aco_params["rho"], Q=aco_params["Q"],
                              iteraciones=15, semilla=semilla + s)
                tot += d
            tots.append(tot / len(muestra))
        prom = float(np.mean(tots))
        filas.append(dict(beta=beta, dist_media_km=prom, semillas=seeds,
                          rangos_cubiertos=sum(rangos_cubiertos(muestra))))
        if prom < mejor_prom:
            mejor_prom, mejor_beta = prom, beta
    return float(mejor_beta), pd.DataFrame(filas)


def correr_nn(subrutas, DEP, cap: int | None = None) -> tuple[pd.DataFrame, float]:
    """Ejecuta NN sobre todas las sub-rutas; usa la ventana completa (E2)."""
    from geopy.distance import geodesic
    cap = cap or cfg.CAP_TOPE
    resultados = []
    t0 = time.perf_counter()
    for ruta in subrutas:
        gid = ruta["grupo_id"].iloc[0]
        dep = DEP[gid]
        D, slots, lats, lons = matriz_subruta(ruta, (dep["dep_lat"], dep["dep_lon"]))
        orden = nearest_neighbor(D, slots)
        d = dist_ruta(D, orden)
        ok = factible_orden(orden, slots)          # E2: incluye último cliente
        puntos = [(lats[i], lons[i]) for i in orden]
        d_geo = sum(geodesic(puntos[i], puntos[i + 1]).km for i in range(len(puntos) - 1))
        if cfg.RUTA_CERRADA:
            d_geo += geodesic(puntos[-1], puntos[0]).km
        resultados.append(dict(sub_ruta=ruta["sub_ruta"].iloc[0], grupo=gid,
                               clientes=len(ruta), dist_km=d_geo, dist_hav=d,
                               factible=int(ok), excede_cap=int(len(ruta) > cap)))
    return pd.DataFrame(resultados), time.perf_counter() - t0
