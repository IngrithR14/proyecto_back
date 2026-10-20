"""EDA portado del notebook `EDA_Distribuidora_Hijos_de_Leon_09.10.ipynb` (FASE 5).

Calcula, sobre la base limpia (`base_clientes_limpia.csv`, 16.560 filas), todas las
agregaciones que alimentan el dashboard (distribuciones, carga por zona, dispersión
geográfica y correlaciones). Ningún número está hardcodeado: todo sale de cálculos
sobre la base. El notebook no se ejecuta; esta lógica es la autoritativa.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from .. import config as cfg

# Rango de fechas/constantes del EDA (etiquetadas como supuesto/calibrado)
FECHA_BASE_EDA = pd.Timestamp(cfg.ANIO_REF, cfg.MES_REF, 22)  # antigüedad (del notebook)
DIAS_HABILES_EDA = 26  # ~26 días hábiles lun-sáb por mes (del notebook)

_CATEGORIAS = ["Tiendas", "Bodegas", "Entretenimiento", "Otros"]


def cargar_base(ruta: str | None = None) -> pd.DataFrame:
    """Carga la base limpia (16.560 filas × 48 cols) ya feature-engineered."""
    ruta = ruta or os.path.join(cfg.DATA_DIR, cfg.BASE_CSV)
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    df["Latitud"] = pd.to_numeric(df["Latitud"], errors="coerce")
    df["Longitud"] = pd.to_numeric(df["Longitud"], errors="coerce")
    df["visitas_mes_rep"] = pd.to_numeric(df["visitas_mes_rep"], errors="coerce").fillna(0).astype(int)
    return df


# ---------------------------------------------------------------- correlaciones
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V con corrección de sesgo (0=independiente, 1=asociación plena)."""
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(x, y)
    if ct.size == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    if n == 0:
        return float("nan")
    r, k = ct.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    den = min((kcorr - 1), (rcorr - 1))
    return float(np.sqrt(phi2corr / den)) if den > 0 else float("nan")


def eta_cuadrado(num: pd.Series, cat: pd.Series) -> float:
    """Razón de correlación η (0=sin relación, 1=determinista) num↔categ."""
    d = pd.DataFrame({"n": pd.to_numeric(num, errors="coerce"), "g": cat}).dropna()
    if d.empty or d["g"].nunique() < 2:
        return float("nan")
    grand = d["n"].mean()
    ss_tot = ((d["n"] - grand) ** 2).sum()
    if ss_tot == 0:
        return float("nan")
    ss_b = d.groupby("g").apply(lambda x: len(x) * (x["n"].mean() - grand) ** 2,
                                include_groups=False).sum()
    return float(np.sqrt(ss_b / ss_tot))


def _barras(df: pd.DataFrame, col: str, top: int | None = None,
            orden: list[str] | None = None) -> dict[str, Any]:
    """Serie de conteos → {labels, values} para Chart.js, con % de cada categoría.

    Las columnas numéricas (p.ej. `Estrato`, float 1.0..6.0) se normalizan a
    entero antes de pasar a texto, para que `"1.0"` coincida con el orden `"1"`.
    """
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        # 1.0 → "1" (evita que el `reindex` de `orden` descarte todo)
        s = s.astype("Int64")
    s = s.astype(str).replace({"<NA>": "SIN_DATO", "nan": "SIN_DATO", "None": "SIN_DATO"})
    vc = s.value_counts(dropna=False)
    if orden:
        vc = vc.reindex([o for o in orden if o in vc.index]).dropna()
    if top:
        vc = vc.head(top)
    total = int(vc.sum())
    return {
        "labels": vc.index.tolist(),
        "values": [int(v) for v in vc.values],
        "pct": [round(100 * v / total, 1) for v in vc.values] if total else [],
        "total": total,
    }


# ---------------------------------------------------------------- resumen EDA
def resumen_eda(df: pd.DataFrame | None = None, n_scatter: int = 2500) -> dict[str, Any]:
    """Todas las agregaciones del EDA como dict JSON-serializable para el dashboard."""
    if df is None:
        df = cargar_base()
    n = len(df)

    # --- 1. KPIs generales ---
    kpis = {
        "n_clientes": n,
        "n_zonas": int(df[cfg.COL_ASESOR].nunique()),
        "visitas_mes_rep": int(df["visitas_mes_rep"].sum()),
        "visitas_por_dia": round(df["visitas_mes_rep"].sum() / DIAS_HABILES_EDA, 1),
        "activos": int((df["activo"] == 1).sum()),
        "pct_activos": round(100 * (df["activo"] == 1).mean(), 1),
        "falsos_inactivos": int(df["posible_falso_inactivo"].sum()) if "posible_falso_inactivo" in df else None,
        "es_rural": int((df["es_rural"] == 1).sum()) if "es_rural" in df else None,
        "sucursales": int(df["es_sucursal"].sum()) if "es_sucursal" in df else None,
    }

    # --- 2. Distribuciones (barras) ---
    orden_dias = ["LUNES", "MARTES", "MIERCOLES", "JUEVES", "VIERNES", "SABADO", "SIN_ASIGNAR"]
    distribuciones = {
        "departamento": _barras(df, "Departamento"),
        "categoria_actividad": _barras(df, "CategoriaActividad", orden=_CATEGORIAS),
        "activo_comercial": _barras(df, "ActivoComercial"),
        "periodo_rep": _barras(df, "periodo_rep"),
        "dias_rep": _barras(df, "DIAS_REP", orden=orden_dias),
        "dias_bees": _barras(df, "DIAS_BEES", top=8),
        "tipo_poblacion": _barras(df, "TipoPoblacion"),
        "estrato": _barras(df, "Estrato", orden=[str(i) for i in range(1, 7)]),
        "unidad_negocio": _barras(df, "UnidadNegocio", top=8),
    }

    # --- 3. Histogramas (numéricos) ---
    def _hist(col: str, bins: int = 30) -> dict[str, Any]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return {"bins": [], "counts": []}
        counts, edges = np.histogram(s, bins=bins)
        return {
            "bins": [round(float(x), 2) for x in edges[:-1]],
            "counts": [int(c) for c in counts],
            "media": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
        }

    histogramas = {
        "visitas_mes_rep": _hist("visitas_mes_rep"),
        "antiguedad_anios": _hist("antiguedad_anios"),
    }

    # --- 4. Carga por zona de rutero (ZON_REP) ---
    carga = (df.groupby(cfg.COL_ASESOR)
             .agg(clientes=("CodigoCliente", "size"),
                  activos=("activo", "sum"),
                  visitas_mes_rep=("visitas_mes_rep", "sum"))
             .reset_index())
    carga["visitas_dia"] = (carga["visitas_mes_rep"] / DIAS_HABILES_EDA).round(1)
    carga = carga.sort_values("visitas_mes_rep", ascending=False)
    carga_zona = {
        "zonas": carga[cfg.COL_ASESOR].tolist(),
        "clientes": [int(x) for x in carga["clientes"]],
        "activos": [int(x) for x in carga["activos"]],
        "visitas_mes_rep": [int(x) for x in carga["visitas_mes_rep"]],
        "visitas_dia": [float(x) for x in carga["visitas_dia"]],
    }

    # --- 5. Dispersión geográfica (para Leaflet), muestreada ---
    geo = df.dropna(subset=["Latitud", "Longitud"])[["Latitud", "Longitud", "CategoriaActividad", "Departamento"]]
    if len(geo) > n_scatter:
        geo = geo.sample(n_scatter, random_state=cfg.SEMILLA)
    scatter = [{
        "lat": round(float(r["Latitud"]), 5), "lon": round(float(r["Longitud"]), 5),
        "cat": r["CategoriaActividad"], "dep": r["Departamento"],
    } for _, r in geo.iterrows()]

    # --- 6. Crosstabs apiladas (categoría × variable) ---
    def _crosstab_pct(fila: str, col: str) -> dict[str, Any]:
        ct = pd.crosstab(df[fila], df[col], normalize="index") * 100
        return {
            "filas": ct.index.tolist(),
            "columnas": ct.columns.tolist(),
            "matriz": [[round(float(v), 1) for v in row] for row in ct.to_numpy()],
        }

    crosstabs = {
        "cat_vs_activo": _crosstab_pct("CategoriaActividad", "ActivoComercial"),
        "cat_vs_poblacion": _crosstab_pct("CategoriaActividad", "TipoPoblacion"),
        "cat_vs_departamento": _crosstab_pct("CategoriaActividad", "Departamento"),
    }

    # --- 7. Correlaciones ---
    num_cols = ["Estrato", "EquiposFrio", "Latitud", "Longitud", "antiguedad_anios",
                "visitas_mes_rep", "visitas_mes_bees", "n_dias_bees", "n_sucursales",
                "activo", "es_rural", "es_sucursal", "sin_dias_rep", "sin_dias_bees",
                "tiene_codigo_anterior", "tiene_atributo4", "posible_falso_inactivo"]
    num_cols = [c for c in num_cols if c in df.columns]
    pearson = df[num_cols].corr(method="pearson")
    # pares más fuertes (|r|>=0.5)
    pares = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            r = pearson.iloc[i, j]
            if pd.notna(r) and abs(r) >= 0.5:
                pares.append({"a": num_cols[i], "b": num_cols[j], "r": round(float(r), 3)})
    pares = sorted(pares, key=lambda x: -abs(x["r"]))

    cat_cols = ["Departamento", "CategoriaActividad", "ActivoComercial", "TipoPoblacion",
                "FRE_REP", "DIAS_REP", "UnidadNegocio", "CondicionPago", "ABC",
                "SegmentacionID", cfg.COL_EQUIPO]
    cat_cols = [c for c in cat_cols if c in df.columns]
    cramers = pd.DataFrame(np.nan, index=cat_cols, columns=cat_cols, dtype=float)
    for a in cat_cols:
        for b in cat_cols:
            cramers.loc[a, b] = cramers_v(df[a].astype(str), df[b].astype(str))

    eta_fils = ["CategoriaActividad", "Departamento", "ABC"]
    eta_cols = ["Estrato", "EquiposFrio", "antiguedad_anios", "visitas_mes_rep",
                "visitas_mes_bees", "n_dias_bees", "activo"]
    eta_fils = [c for c in eta_fils if c in df.columns]
    eta_cols = [c for c in eta_cols if c in df.columns]
    eta = pd.DataFrame(np.nan, index=eta_fils, columns=eta_cols, dtype=float)
    for f in eta_fils:
        for c in eta_cols:
            eta.loc[f, c] = eta_cuadrado(df[c], df[f].astype(str))

    def _matriz(m: pd.DataFrame) -> dict[str, Any]:
        return {
            "index": m.index.tolist(),
            "columns": m.columns.tolist(),
            "matriz": [[None if pd.isna(v) else round(float(v), 2) for v in row]
                       for row in m.to_numpy()],
        }

    correlaciones = {
        "pearson": _matriz(pearson),
        "pearson_pares": pares,
        "cramers_v": _matriz(cramers),
        "eta": _matriz(eta),
    }

    return {
        "kpis": kpis,
        "distribuciones": distribuciones,
        "histogramas": histogramas,
        "carga_zona": carga_zona,
        "scatter": scatter,
        "crosstabs": crosstabs,
        "correlaciones": correlaciones,
    }
