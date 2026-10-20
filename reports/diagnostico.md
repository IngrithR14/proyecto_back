# Diagnóstico de solo lectura

**Alcance:** verificación de sobreajuste/re-balanceo/datos/chatbot. **No se modificó código, caché
ni notebooks.** Único archivo creado: este informe (`reports/diagnostico.md`). Los cálculos se
hicieron en memoria con los módulos del proyecto (scripts en `/tmp/opencode/diag/`), leyendo la
caché existente sin escribirla. **No se incluyen cédulas, nombres de clientes, direcciones ni el
contenido de `.env`.** Todas las cifras salen de los datos/caché o de ejecuciones nuevas.

**Convenciones:** veredicto `OK` / `PROBLEMA` / `NO CONCLUYENTE`. Cada punto trae pregunta,
evidencia (números + fragmento ≤30 líneas con `archivo:línea`) y veredicto. No se ajustaron
parámetros para que algo salga bien.

**Datos base (cache `universos.json`):** base limpia 16.560 → activos 14.796 → con visita 14.757 →
coords válidas 14.757. Zonas base **67**, zonas rutable **58**. Visitas/mes (rutable) 45.212.

---

## A. SOBREAJUSTE Y FUGA DE DATOS

### A1. ¿Con qué filas se calculan depósitos (DEP), territorios y K-Means en cada experimento?

**Pregunta.** ¿Algún experimento usa clientes de test?

**Evidencia.** No existe ninguna partición train/test activa en el proyecto; todos los cálculos
usan **todos** los clientes del universo rutable (14.757):

1. **Territorios (K-Means capacitado):** todas las filas de `df`:

```python
# app/core/estructura.py:47-57
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
```

2. **Depósitos (DEP):** media de **todos** los clientes del territorio (no hay muestra):

```python
# app/core/estructura.py:97-100
def _depot_de(terr: pd.DataFrame, modo: str) -> tuple[float, float]:
    if modo == "cd" and "Cod CD" in terr.columns:
        return float(terr["Latitud"].mean()), float(terr["Longitud"].mean())
    return float(terr["Latitud"].mean()), float(terr["Longitud"].mean())
# usado en app/core/estructura.py:134-136 (construir_actual: df.groupby(ZON_REP) sobre TODO el df)
# y app/core/comparacion.py:129-134 (grupos = ZON_REP sobre TODO el df)
```

3. **Experimento NN vs ACO:** el "train" que recibe `formar_subrutas` es el mismo `g` completo
   (argumento repetido, `train == all`):

```python
# app/core/comparacion.py:139-145
    for g in grupos:
        rutas, _, _ = formar_subrutas(g, g, cfg.CAP_TOPE, cfg.SEMILLA)
        for j, r in enumerate(rutas):
            r = r.copy()
            r["sub_ruta"] = f"{g['grupo_id'].iloc[0]}#R{j+1:02d}"
            subrutas.append(r)
    muestra = muestra_estratificada(subrutas, por_rango=por_rango)
```

4. **Único split real (K-Fold):** el pliegue retenido (test) **queda fuera** de la construcción y
   de la evaluación:

```python
# app/core/pliegues.py:82-89
        test_idx = folds[i]                                   # retenido (excluido)
        train_idx = np.concatenate([folds[j] for j in range(k_pliegues) if j != i])
        train = d.iloc[train_idx].reset_index(drop=True)
        # territorios + rutas diarias SOLO sobre el entrenamiento (k−1 pliegues)
        labels_train = asignacion_capacidad(train, k_territorios, semilla=semilla)
        m = _metricas_pliegue(train, labels_train, cap=cap)
        m.update(pliegue=i, n_train=int(len(train_idx)), n_test=int(len(test_idx)))
        filas.append(m)
```

5. `TEST_SIZE = 0.20` y `SEMILLA_MODELO = 42` (`app/config.py:82-83`) **no se usan en ninguna parte**
   (`grep TEST_SIZE|SEMILLA_MODELO app/ --include=*.py` → solo `config.py:82-83`): es un split
   configurado que nunca se aplica.

**Pero el NOTEBOOK de origen sí tiene train/test — y ahí hay fuga de test en los depósitos.**
Los números del modelo de ruteo nacieron en `Modelo_Ruteo_KMeans_NN_ACO_11.10.ipynb` (y su versión
v2 `modelo_ruteo_v2.ipynb`); en ambos el orden de las celdas es:

| celda (v1 / v2) | contenido | filas usadas |
|---|---|---|
| **9 / 10** | `dep = df.groupby("grupo_id").agg(dep_lat=("Latitud","mean"), dep_lon=("Longitud","mean"))` — **depósito por grupo = media de TODOS los clientes** | **train + test** (corre ANTES del split) |
| 13 | `train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEMILLA, stratify=df["Supervisor"])` | 80/20 estratificado |
| 15 | `rutas, … = formar_subrutas(grupo_train, grupo_all, CAP, SEMILLA)` (docstring: "Ajusta K-Means en los clientes TRAIN … y forma sub-rutas sobre train+test") | K-Means: train; sub-rutas: **train + test** |
| 19 | `DEP = dep.set_index("grupo_id")[["dep_lat","dep_lon"]].to_dict("index")` — el DEP que usan NN y ACO | **train + test** |
| 25 / 34 | `pipeline_test(…)`: evalúa NN vs ACO sobre `sub_test` (sub-rutas que contienen ≥1 cliente de test), usando `DEP[gid]` | test mixto con train |

```python
# Modelo_Ruteo_KMeans_NN_ACO_11.10.ipynb, celda 9 (antes del split de la celda 13)
dep = (df.groupby("grupo_id")
         .agg(dep_lat=("Latitud", "mean"), dep_lon=("Longitud", "mean"),
              n=("CodigoCliente", "size"), nombre_cd=("Nombre CD", "first"))
         .reset_index())
# Modelo_Ruteo_KMeans_NN_ACO_11.10.ipynb, celda 13
train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEMILLA,
                               stratify=df["Supervisor"])
```

Es decir: **los depósitos (DEP) se calculan con train+test** (posiciones de las rutas contaminadas
por el test), las sub-rutas evaluadas **mezclan** train y test, y `config.TEST_SIZE=0.20` existe en
la API precisamente por ese split pero **nunca se aplica** (todo el universo es "train"). Verificado
también sobre los notebooks del proyecto: `train_test_split` solo aparece en
`Modelo_Ruteo_KMeans_NN_ACO_11.10.ipynb` y `modelo_ruteo_v2.ipynb`; en `ejemplos/` solo hay
notebooks de curso (churn/spam) ajenos al ruteo.

**Discrepancias (regla 8, no resueltas):** (a) el docstring de `app/core/pliegues.py:5` dice
"asigna cada cliente retenido al territorio más cercano", pero el código de ese módulo **no lo
hace**; (b) `TEST_SIZE`/`SEMILLA_MODELO` (`app/config.py:82-83`) son parámetros muertos en la API;
(c) el notebook comenta el depósito como si fuera neutral, pero `dep` se calcula antes del split.

**Veredicto: PROBLEMA** (fuga de test: los depósitos DEP del notebook se calculan con train+test
celda 9/10 antes del split celda 13, y las sub-rutas evaluadas mezclan train con test). En la API
no hay partición (todo usa train=all) y el K-Fold excluye el test sin evaluarlo. Archivos a
corregir: `Modelo_Ruteo_KMeans_NN_ACO_11.10.ipynb` (celdas 9 y 19) / `modelo_ruteo_v2.ipynb`
(celda 10), `app/config.py:82-83` y `app/core/pliegues.py:5`.

### A2. Calibración de β del ACO: cuántas sub-rutas, tamaños y si son de train

**Pregunta.** ¿Sobre qué muestra se calibra β y de dónde sale?

**Evidencia.**

```python
# app/core/ruteo.py:331-335 (calibrar_beta)
    grid = cfg.BETA_GRID if grid is None else grid
    seeds = cfg.BETA_SEEDS if seeds is None else seeds
    muestra = muestra_estratificada(rutas)          # por_rango=2 (ruteo.py:311-318)
    filas = []
    mejor_beta, mejor_prom = None, np.inf
```

- `muestra_estratificada` toma las **2 más pequeñas** sub-rutas de cada rango
  `(1-20), (21-30), (31-40), (41-10000)` (`app/core/ruteo.py:308-318`), sin filtrar tamaños
  degenerados (a diferencia del experimento, que sí filtra `>=3` en `app/core/comparacion.py:147`).
- **Ejecución diagnóstica (nueva, en memoria)** sobre las sub-rutas reales:
  - sub-rutas formadas con **todos** los clientes: **657** (58 grupos ZON_REP; tamaño 1/23/45);
  - muestra de `calibrar_beta`: **8 sub-rutas, tamaños `[1, 1, 21, 21, 31, 31, 41, 41]`**
    → **2 sub-rutas de 1 cliente** entran en la calibración (ruta trivial depósito→cliente);
  - resultados por β (media de 3 semillas, km): β=1.0 → 6.9497, β=3.0 → 6.9408, β=5.0 → 6.9351;
    **elegiría β=5.0**;
  - producción usa `ACO_BETA = 3.0` (`app/config.py:93`), heredado del notebook.
- **`calibrar_beta` no tiene NINGUNA llamada en el proyecto** (`grep calibrar_beta --include=*.py`
  → solo su definición en `app/core/ruteo.py:329` y una mención en un informe). La β efectiva del
  modelo es la constante de config, no el resultado de la calibración.
- **No hay train/test:** las sub-rutas salen de `formar_subrutas(g, g, …)` sobre el conjunto
  completo (ver A1.3), igual que la muestra.

**En el notebook de origen** la calibración original usaba las **6 sub-rutas más pequeñas**, con un
comentario que dice "train" pero que toma `subrutas` construidas sobre `grupo_all` (train+test):

```python
# Modelo_Ruteo_KMeans_NN_ACO_11.10.ipynb, celda 22 (versión original)
# Unas pocas sub-rutas pequeñas/train para calibrar beta rápidamente
muestra_cal = sorted(subrutas, key=len)[:6]
mejor_beta, mejor_prom = None, np.inf
pruebas = []
for beta in [1.0, 3.0, 5.0]:
```

La versión v2 (`modelo_ruteo_v2.ipynb`, celda 27) corrigió eso a muestra estratificada por tamaño
(`_muestra_estrat(subrutas, por_rango=2)`), que es exactamente lo que replicó el código
(`app/core/ruteo.py:311-318`) — pero conserva el comentario "pequeñas/train" que no corresponde.

**Veredicto: PROBLEMA** (calibración desconectada del código + muestra con 2 sub-rutas
degeneradas de 1 cliente; el impacto numérico es pequeño: 0,2 % entre β). Archivos a corregir:
`app/core/ruteo.py:334` (filtrar tamaño mínimo, como `comparacion.py:147`),
`app/core/ruteo.py:329`/`app/config.py:93` (β usada 3,0 ≠ β calibrada 5,0) y el comentario de
`modelo_ruteo_v2.ipynb` celda 27.

### A3. Selección del mejor K: ¿se elige y se reporta con las MISMAS simulaciones?

**Pregunta.** ¿La selección del K usa los mismos MonteCarlo con los que se reporta? Re-evaluación
con semilla distinta.

**Evidencia.** Sí: **una sola corrida** de números aleatorios alimenta a ACTUAL y a los 36 K, y
con esos mismos resultados se elige el K recomendado y se publica en la API/tabla:

```python
# app/precompute.py:119-120  (una sola vez para TODAS las estrategias)
    # números aleatorios comunes (paired comparison) — una sola vez
    asistencia, trafico = generar_entradas(n_sim, n_cli, cfg.SEMILLA)

# app/precompute.py:139-144
    for K in pendientes:
        t1 = time.time()
        labels = _labels_de(df, K)
        res_k, costo_k = evaluar(df, labels, asistencia, trafico, f"K={K}")
        estrategias[str(K)] = res_k
        comparaciones[str(K)] = comparar_pareada(costo_k, costo_actual_ref)

# app/precompute.py:222-229  (selección con ESOS MISMOS resultados)
def _k_recomendado(mc: dict | None) -> int:
    """K de la estrategia recomendada = factible más barata (si no, K_ACTUAL)."""
    est = (mc or {}).get("estrategias", {})
    factibles = sorted(
        ({"K": int(k), "costo": r.get("costo_medio_M")}
         for k, r in est.items() if r.get("factible")),
        key=lambda x: (x["costo"] is None, x["costo"]))
    return factibles[0]["K"] if factibles else cfg.K_ACTUAL
```

La API `/api/montecarlo/mejor` (`app/main.py:188-201`) y la tabla del dashboard
(`montecarlo.component.ts:187-210`) muestran exactamente esos mismos números.

**Re-evaluación con semilla distinta (cálculo nuevo, en memoria, sin tocar la caché).** Se repitió
el pipeline completo para los 36 K con la semilla del proyecto (**2026**, alimenta K-Means en
`app/core/estructura.py:57` y el MonteCarlo en `app/precompute.py:120`) y con una semilla distinta
(**2027**), usando las MISMAS fórmulas que `precompute` (reparto de visitas, rutas diarias,
`evaluar_estrategia`, `costo_escenario`, `horas_por_escenario`). **Validación:** la corrida
semilla-2026 reproduce la caché **exactamente** (máx |Δcosto| = **0,000000 M COP** en los 36 K;
ACTUAL 133,320124… idéntico), así que la comparación es válida.

Resultados por K (M COP por ciclo; fact. = factible con pico ≤ 44 y 0 días sobre CAP):

| K | costo s2026 | costo s2027 | fact s2026 | fact s2027 | K | costo s2026 | costo s2027 | fact s2026 | fact s2027 |
|---|---|---|---|---|---|---|---|---|---|
| 35 | 165,6 | 175,4 | no | no | 53 | 183,6 | 184,2 | sí | sí |
| 36 | 168,8 | 171,4 | no | no | 54 | 179,7 | 182,9 | sí | sí |
| 37 | 171,6 | 173,0 | no | no | 55 | 187,3 | 185,0 | sí | sí |
| 38 | 165,8 | 168,3 | no | no | 56 | 183,4 | 194,8 | sí | sí |
| 39 | 168,7 | 168,4 | no | no | 57 | 192,2 | 196,0 | sí | sí |
| 40 | 167,5 | 165,8 | no | no | 58 | 189,9 | 193,6 | sí | sí |
| 41 | 167,2 | 169,0 | no | no | 59 | 190,9 | 188,4 | sí | sí |
| 42 | 165,0 | 164,2 | no | no | 60 | 198,8 | 195,2 | sí | sí |
| **43** | **169,9** | **165,5** | sí | sí | 61 | 196,3 | 193,4 | sí | sí |
| 44 | 169,0 | 171,1 | sí | sí | 62 | 200,7 | 199,1 | sí | sí |
| 45 | 174,5 | 170,3 | sí | sí | 63 | 199,7 | 200,3 | sí | sí |
| 46 | 170,4 | 171,9 | sí | sí | 64 | 202,4 | 202,5 | sí | sí |
| 47 | 174,3 | 178,6 | sí | sí | 65 | 206,1 | 204,8 | sí | sí |
| **48** | **168,3** | **177,3** | sí | sí | 66 | 200,8 | 208,9 | sí | sí |
| 49 | 181,1 | 178,0 | sí | sí | 67 | 205,9 | 206,5 | sí | sí |
| 50 | 179,8 | 176,9 | sí | sí | 68 | 209,7 | 204,7 | sí | sí |
| 51 | 179,5 | 178,7 | sí | sí | 69 | 212,1 | 213,4 | sí | sí |
| 52 | 183,6 | 180,4 | sí | sí | 70 | 215,0 | 215,0 | sí | sí |

- **¿Se elige y se reporta con las MISMAS simulaciones? SÍ** (evidencia de código arriba: una sola
  corrida `generar_entradas` para ACTUAL + los 36 K, y con esos resultados `_k_recomendado` elige
  y la API los publica).
- **Correlación de Spearman entre los dos rankings de costo: 0,9439** (alta, pero < 1).
- **K mínimo factible: 43 en las dos semillas** (28 de 36 K factibles en ambas; ningún K cambia
  de factibilidad; picos/días fluctúan ±1-2 visitas).
- **K más barato factible: 48 (168,3 M) con semilla 2026 vs 43 (165,5 M) con semilla 2027 →
  LA RECOMENDACIÓN CAMBIA.** K=48 pasa de 168,3 a 177,3 M (+5,4 %) y K=43 de 169,9 a 165,5 M
  (−2,6 %): la elección del "más barato" cae dentro del ruido de la simulación (diferencia relativa
  entre semillas: media 1,66 %, máx. 6,00 %). El más barato sin filtrar factibilidad es K=42
  (165,0/164,2 M) en ambas, pero no es factible (10 y 8 días sobre CAP).

**Veredicto: PROBLEMA** (la selección del K se hace y se publica con la MISMA corrida de MonteCarlo;
re-evaluado con otra semilla el K recomendado cambia de 48 a 43 pese a un Spearman de 0,944; el
K mínimo factible 43 sí es estable). Archivos a corregir: `app/precompute.py:120` (semillas/
réplicas independientes para selección vs reporte) y `app/precompute.py:222-229` (criterio de
selección sin robustez).

### A4. K-Fold: km de train y de test por cliente, brecha y ranking NN vs ACO

**Pregunta.** Brecha porcentual por pliegue (PROBLEMA si >10 %) y estabilidad del ranking NN vs ACO.

**Evidencia (implementación actual).** `kfold_ruteo` construye y evalúa **solo sobre train**
(`app/core/pliegues.py:82-89`, ver A1.4); **no calcula ninguna métrica de test** (el test queda
excluido, pese al docstring `app/core/pliegues.py:5`). Caché actual (`cache/pliegues.json`),
K=48, 5 pliegues:

| pliegue | n_train | n_test | km ciclo (train) | costo det. M COP | pico | días>cap | factible |
|---|---|---|---|---|---|---|---|
| 0 | 11.805 | 2.952 | 68.645,8 | 168,77 | 32 | 0 | sí |
| 1 | 11.805 | 2.952 | 66.372,1 | 166,05 | 32 | 0 | sí |
| 2 | 11.806 | 2.951 | 67.037,0 | 166,84 | 32 | 0 | sí |
| 3 | 11.806 | 2.951 | 72.851,4 | 173,82 | 32 | 0 | sí |
| 4 | 11.806 | 2.951 | 66.894,1 | 166,67 | 32 | 0 | sí |

**Cálculo nuevo (en memoria).** Como `kfold_ruteo` no evalúa el test, se replicó la misma
partición (`np.random.default_rng(cfg.SEMILLA)` → 5 pliegues, K=48) y se midieron ambos conjuntos:
los clientes retenidos se asignan al **territorio más cercano** (los centroides del train, que es
lo que el docstring de `app/core/pliegues.py:5` promete) y se construyen las rutas diarias con los
14.757 clientes; los km abiertos se atribuyen al **cliente destino de cada tramo**
(depósito→c1, c1→c2 …), de modo que train+test = km total (chequeo cerrado a ≤4 km de redondeo).

| pliegue | n_train | n_test | km rutas train (valida contra caché) | km/train-cliente | km/test-cliente | **brecha %** | km marginal por cliente test | km_all (con test) |
|---|---|---|---|---|---|---|---|---|
| 0 | 11.805 | 2.952 | 68.645,8 (= cache) | 5,391 | 2,577 | **−52,20 %** | 0,882 | 71.249,4 |
| 1 | 11.805 | 2.952 | 66.372,1 (= cache) | 5,297 | 2,297 | **−56,63 %** | 0,996 | 69.311,1 |
| 2 | 11.806 | 2.951 | 67.037,0 (= cache) | 5,298 | 2,606 | **−50,81 %** | 1,085 | 70.238,8 |
| 3 | 11.806 | 2.951 | 72.851,4 (= cache) | 5,805 | 2,471 | **−57,44 %** | 1,009 | 75.829,9 |
| 4 | 11.806 | 2.951 | 66.894,1 (= cache) | 5,294 | 2,425 | **−54,19 %** | 0,936 | 69.655,3 |

- **train: 5,417 ± 0,221 km/cliente · test: 2,475 ± 0,124 km/cliente ·
  brecha media 54,25 % ± 2,82 (máx. 57,44 %)** — todas las variantes (con o sin la atribución
  por tramo: 50,8 %-57,4 %; o usando el km marginal del test: 0,88-1,09 km/cliente) superan con
  creces el umbral del 10 %.
- **Ranking NN vs ACO por pliegue** (muestra estratificada de 12 sub-rutas del train de cada
  pliegue, tamaños 21×4 / 31×4 / 41×4, 10 semillas para ACO):

| pliegue | NN (km) | ACO (km, media 10 sem.) | ahorro ACO | gana |
|---|---|---|---|---|
| 0 | 192,3 | 172,7 | 10,2 % | ACO |
| 1 | 107,4 | 101,1 | 5,9 % | ACO |
| 2 | 241,3 | 216,3 | 10,4 % | ACO |
| 3 | 161,2 | 152,3 | 5,5 % | ACO |
| 4 | 154,8 | 139,1 | 10,1 % | ACO |

  → **el ranking NN vs ACO NO cambia entre pliegues** (ACO gana en los 5; ahorro 5,5-10,4 %).

**Lectura honesta de la brecha.** No es un error de predicción clásico: el test se asigna al
centroide más cercano (sin la penalización de balanceo que sí sufren los clientes de train en
`asignacion_capacidad`, `app/core/estructura.py:61-78`), de modo que los retenidos quedan cerca
del depósito y sus tramos de llegada son sistemáticamente más cortos. Es un artefacto de
asignación+atribución, **no** evidencia de que "el modelo se sobreajuste al train"; pero, por la
regla acordada (brecha > 10 %), el K-Fold tal como está **no mide** el desempeño sobre los
retenidos.

**Veredicto: PROBLEMA** (brecha train/test = 54 % >> 10 %, y hoy `kfold_ruteo` no calcula
métrica alguna de test). El ranking NN vs ACO es estable (OK). Archivo a corregir:
`app/core/pliegues.py:52-89` (falta evaluación de test; el docstring promete una asignación que
el código no hace).

---

## B. RE-BALANCEO Y COSTOS

### B1. Algoritmo de re-balanceo: descripción, criterio y origen

**Pregunta.** ¿Parte de ACTUAL o reagrupa desde cero? Métricas por K.

**Evidencia.** La estrategia hipotética **reagrupa DESDE CERO**: `asignacion_capacidad` nunca lee
`ZON_REP` (no hay ninguna referencia a `COL_ASESOR` en esa función); parte de un K-Means sobre
todas las coordenadas y luego reasigna con penalización de desborde total y por fecha:

```python
# app/core/estructura.py:57-61 (K-Means inicial sobre TODO el universo)
    labels = KMeans(n_clusters=K, n_init=10, random_state=semilla).fit_predict(xy)
    cent = np.array([xy[labels == t].mean(axis=0) for t in range(K)])
    orden = np.argsort(-carga)                      # carga descendente (mejor packing)

    for _ in range(max_iter):
        newl = np.full(n, -1, int)
```

```python
# app/core/estructura.py:66-78 (criterio de asignación por cliente)
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
```

- **Criterio:** argmin `dist² + λ·(desborde_total)² + λ_fecha·Σ(exceso_fecha)²` con
  `lam=1000`, `lam_fecha=400` (`app/core/estructura.py:31-34`): los pesos de balanceo son 2-3
  órdenes de magnitud mayores que `dist²` (típico d² ≤ ~10.000 km² para clientes a <100 km), es
  decir, **el equilibrio pesa más que la compactación geográfica**.
- **ACTUAL** sí parte de las zonas reales: `construir_actual` agrupa por `ZON_REP` sin reclustering
  (`app/core/estructura.py:122-137`).

Cálculo nuevo (en memoria; partición del proyecto, semilla 2026) para los 36 K. "Movidos" = %
de clientes cuyo ZON_REP no es la zona mayoritaria de su nuevo territorio; distancia = media
cliente→centroide de su zona, en km proyectados (como el objetivo de K-Means); km = ciclo
determinista completo (todas las visitas):

| K | km/ciclo | movidos vs ACTUAL | dist. cliente-centroide (antes → después) | pico | días>cap | K | km/ciclo | movidos | dist. antes → después | pico | días>cap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 35 | 71.339 | 62,43 % | 5,240 → 7,743 | 54 | 874 | 53 | 77.797 | 51,77 % | 5,240 → 6,669 | 36 | 0 |
| 36 | 74.219 | 61,44 % | 5,240 → 7,653 | 53 | 877 | 54 | 72.267 | 50,49 % | 5,240 → 6,609 | 36 | 0 |
| 37 | 76.114 | 61,38 % | 5,240 → 8,024 | 51 | 871 | 55 | 77.638 | 52,92 % | 5,240 → 6,701 | 34 | 0 |
| 38 | 72.794 | 59,40 % | 5,240 → 7,410 | 51 | 865 | 56 | 72.691 | 51,53 % | 5,240 → 6,579 | 34 | 0 |
| 39 | 75.419 | 60,12 % | 5,240 → 7,486 | 49 | 652 | 57 | 79.017 | 52,25 % | 5,240 → 6,786 | 34 | 0 |
| 40 | 74.659 | 60,84 % | 5,240 → 7,459 | 48 | 306 | **58** | 74.914 | **49,42 %** | 5,240 → 6,576 | 33 | 0 |
| 41 | 74.180 | 58,75 % | 5,240 → 7,498 | 46 | 68 | 59 | 74.998 | 49,18 % | 5,240 → 6,562 | 33 | 0 |
| 42 | 72.224 | 57,27 % | 5,240 → 7,412 | 45 | 10 | 60 | 80.092 | 50,29 % | 5,240 → 6,729 | 32 | 0 |
| 43 | 74.579 | 59,65 % | 5,240 → 7,609 | 44 | 0 | 61 | 76.066 | 50,25 % | 5,240 → 6,690 | 31 | 0 |
| 44 | 74.921 | 58,35 % | 5,240 → 7,174 | 44 | 0 | 62 | 79.726 | 49,13 % | 5,240 → 6,496 | 32 | 0 |
| 45 | 77.942 | 57,32 % | 5,240 → 7,333 | 42 | 0 | 63 | 75.757 | 50,86 % | 5,240 → 6,632 | 31 | 0 |
| 46 | 74.598 | 55,15 % | 5,240 → 7,023 | 42 | 0 | 64 | 76.726 | 50,64 % | 5,240 → 6,482 | 30 | 0 |
| 47 | 75.972 | 55,65 % | 5,240 → 7,459 | 40 | 0 | 65 | 79.252 | 48,78 % | 5,240 → 6,640 | 30 | 0 |
| **48** | **70.646** | **55,06 %** | 5,240 → **6,979** | 40 | 0 | 66 | 73.065 | 48,26 % | 5,240 → 6,342 | 30 | 0 |
| 49 | 78.731 | 53,40 % | 5,240 → 7,042 | 39 | 0 | 67 | 75.669 | 49,58 % | 5,240 → 6,465 | 30 | 0 |
| 50 | 76.650 | 54,93 % | 5,240 → 6,951 | 38 | 0 | 68 | 77.545 | 50,89 % | 5,240 → 6,438 | 28 | 0 |
| 51 | 76.883 | 52,12 % | 5,240 → 6,683 | 38 | 0 | 69 | 78.426 | 49,14 % | 5,240 → 6,410 | 28 | 0 |
| 52 | 77.480 | 54,61 % | 5,240 → 7,019 | 37 | 0 | 70 | 79.676 | 49,79 % | 5,240 → 6,512 | 28 | 0 |

**Referencia ACTUAL** (58 zonas reales, sin re-balancear): km/ciclo **25.439**, movidos **0 %**,
dist. cliente-centroide **5,240 km**, pico 47, 4 días sobre CAP=44.

**Lectura.**
- En **todos** los K el re-balanceo mueve a **48-62 %** de los clientes (media 54 %) y sube la
  dispersión cliente-centroide de **5,240 → 6,34-8,02 km** (media 6,95 km, +33 %).
- Los km pasan de 25.439 a **70.646-80.092** (×2,78-3,15). Nota clave para B2: en **K=58 — el
  mismo número de zonas que ACTUAL — ya son 74.914 km (×2,94)**, así que el salto **no** viene de
  reducir asesores, sino del re-clustering.
- Los km crecen mucho más (×2,8-3,1) de lo que sugiere la dispersión puntual (+26-53 %): el
  factor dominante es la **forma** de los territorios (rutas diarias serpenteantes con depósito por
  territorio en zonas largas), no la distancia media puntual al centroide.
- La factibilidad mejora con K (K ≥ 43 tiene 0 días sobre CAP), pero a costa de esa estructura.

**Veredicto: PROBLEMA** (el "re-balanceo" no re-balancea las zonas existentes: re-agrupa desde cero
sobre todo el universo con penalizaciones de balance muy fuertes y mueve a ~la mitad de los
clientes en todos los K; el coste son ×2,8-3,1 km). Archivo a corregir:
`app/core/estructura.py:31-91` (`asignacion_capacidad`).

### B2. ¿Por qué el re-balanceo sube los km 2,7 veces (24.100 vs 66.513)?

**Pregunta.** Explicación + variante que mueve solo lo necesario (diagnóstico, sin reemplazar).

**Evidencia numérica.**

| escenario | km/ciclo (MonteCarlo 85 % asistencia, cache) | km/ciclo (determinista, todas las visitas, `cache/rutas.json`) |
|---|---|---|
| ACTUAL (58 zonas reales) | **24.099,9** (`montecarlo.json → actual.km_medio`) | **25.438,7** |
| K=48 re-balanceado | **66.512,8** (`estrategias["48"].km_medio`) | **70.645,7** |
| razón | **×2,76** | **×2,78** |

**Por qué sube (mecanismo, no opinión):** el re-balanceo (B1) optimiza
`dist² + 1000·desborde² + 400·Σ_exceso_fecha²`: para igualar la carga **por fecha** reparte
clientes de la misma zona/día entre territorios geográficamente lejanos, alarga los territorios
(y con ellos la ruta diaria y el viaje depot→primera visita) y reduce K de 58 a 48 (territorios
más grandes). La distancia media cliente-centroide sube respecto de las zonas reales (ver B1) y
los km crecen en la misma proporción (~×2,8).

**Variante diagnóstica (solo lo necesario):** parte de ACTUAL y mueve un cliente a otra zona
solo si ese (asesor, fecha) excede CAP=44, eligiendo la zona que menos aumenta la distancia
cliente-centroide y respetando la capacidad en **todas** las fechas del cliente (script
`/tmp/opencode/diag/b2.py`, no modifica nada):

```python
# lógica central de la variante (script diagnóstico, en memoria)
        for i in cands:                       # candidatos de la zona sobrecargada ese día
            for z2 in range(len(zonas)):
                if z2 == lab[i]:
                    continue
                # capacidad en TODAS las fechas del cliente (y en la del problema)
                if any(carga.get((zonas[z2], ff), 0) + vv > CAP for ff, vv in fechas_cli[i]):
                    continue
                delta = float(np.linalg.norm(xy[i] - cent[z2]) -
                              np.linalg.norm(xy[i] - cent[lab[i]]))
                if mejor is None or delta < mejor[0]:
                    mejor = (delta, i, z2)
```

Resultado:

| | días sobre cap | exceso (visitas) | clientes movidos | km/ciclo (det.) | dist. media cliente-centroide |
|---|---|---|---|---|---|
| ACTUAL | 4 (todos en COM2Z6) | 10 | – | 25.438,7 | 5,240 km |
| **variante "solo lo necesario"** | **0 (factible, pico 44)** | 0 | **5** | **25.834,3 (×1,016)** | 5,247 km |
| K=58 re-clustering (mismo nº de zonas que ACTUAL) | 0 | 0 | 49,42 % de los clientes | 74.914,4 (×2,94) | 6,576 km |
| K=48 re-balanceo completo | 0 | 0 | 55,06 % de los clientes | 70.645,7 (×2,78) | 6,979 km |

La fila K=58 descarta la hipótesis "sube porque hay menos asesores": con **el mismo número de
zonas** que ACTUAL el re-clustering ya triplica los km (ver B1).

**Veredicto: PROBLEMA** (el salto ×2,8 no viene de la restricción de capacidad — con **5 clientes
movidos** ACTUAL ya es factible —, sino del re-clustering desde cero con penalizaciones de balance
muy fuertes). Archivo a corregir: `app/core/estructura.py:31-91` (`asignacion_capacidad`), dejando
esta variante como diagnóstico (no sustituye la estrategia acordada).

### B3. Valor de COSTO_ASESOR_MES y desglose del costo de ACTUAL

**Pregunta.** Valor, archivo y cuenta desglosada fijo + combustible + horas extra.

**Evidencia.**

```python
# app/config.py:57-60
COSTO_ASESOR_MES: float = 1_800_000.0   # COP/mes por asesor (confirmado)
COSTO_KM: float = 1_200.0               # COP/km combustible+mantenimiento
COSTO_HORA: float = 12_000.0            # COP/hora SOLO horas extra
HORA_EXTRA_FACTOR: float = 1.25         # factor sobre la hora extra

# app/config.py:143-145
def costo_fijo_ciclo() -> float:
    """Costo fijo de asesores por ciclo completo (D3.1)."""
    return COSTO_ASESOR_MES * (DURACION_CICLO_DIAS / 30.0)
```

Cuenta de ACTUAL (K=58, ciclo 30 días, semilla 2026, de `cache/montecarlo.json`):

| componente | cuenta | M COP | % |
|---|---|---|---|
| fijo | 1.800.000 × 30/30 × **58** asesores | **104,400** | 78,308 % |
| combustible | **24.099,9 km** × 1.200 COP/km | **28,920** | 21,692 % |
| horas extra | horas_extra × 12.000 × 1,25 (≈0 h; ver B4) | **0,000197** (≈197 COP) | 0,000148 % |
| **total** | 104,400 + 28,920 + 0,000 | **133,320** | 100 % |

El total recalculado **cuadra exactamente** con `costo_medio_M = 133,320124…` de la caché.

**Discrepancias (regla 8):** (a) `app/config.py:57` lo etiqueta "(confirmado)" mientras que
`app/config.py:172` lo publica como `"supuesto" … "NO calibrado (qué incluye)"` — etiquetas
inconsistentes dentro del mismo archivo; (b) sigue abierta la duda 1,8 M vs 1,5 M COP/mes sin
confirmación del usuario.

**Veredicto: OK** (la cuenta es exacta y reproducible) con las discrepancias (a)/(b) abiertas en
`app/config.py:57` vs `app/config.py:172`.

### B4. Cobertura: fórmula exacta, por qué ACTUAL da 0,85 = P_ASIST y corrida con P_ASIST = 1,0

**Pregunta.** ¿La cobertura mide sobrecarga o solo asistencia?

**Evidencia.** La cobertura es `visitas atendidas / visitas programadas`, donde las atendidas son
Bernoulli(`P_ASIST`) **sin ningún filtro horario**:

```python
# app/core/montecarlo.py:199-212
    visitas_sim = float(ev["visitas"].mean())
    return {
        "estrategia": nombre,
        ...
        "visitas_media": visitas_sim,
        "cobertura": float(visitas_sim / visitas_prog) if visitas_prog else np.nan,
        "visitas_perdidas": float(max(0.0, visitas_prog - visitas_sim)),
```

```python
# app/core/montecarlo.py:134-141 (el motor sólo aplica asistencia; la jornada NO descarta visitas)
        A = np.concatenate([np.ones((S, 1), bool), asistencia[:, uid]], axis=1)
        idx = np.where(A, pos, -1)
        prev = np.maximum.accumulate(idx, axis=1)
        ...
        v = A[:, 1:].sum(axis=1)
```

```python
# app/core/montecarlo.py:99-106 (las horas usan PROMEDIOS de ciclo, no el día concreto)
    dias = cfg.DIAS_HABILES_CICLO
    km_dia = pd.Series({k: v / dias for k, v in km_por_terr.items()}, dtype=float)
    visitas_dia = ctf.groupby("territorio")["visitas"].sum() / dias
    ...
    h_dia = h_viaje + h_serv
    extra_dia = np.maximum(0.0, h_dia - cfg.JORNADA_HORAS)
```

**Corridas nuevas (en memoria, misma ruta ACTUAL):**

| corrida | visitas_prog | visitas_sim | cobertura |
|---|---|---|---|
| P_ASIST = 0,85 (cache) | 45.212 | 38.433,691 | **0,8500772140139785** (idéntica a la caché) |
| **P_ASIST = 1,0** | 45.212 | 45.212,0 | **1,000 exacto** |

- Con P_ASIST=1,0 la cobertura es **1,0 aunque ACTUAL tenga 4 días sobre CAP** → la sobrecarga
  **no** reduce la cobertura.
- `horas_por_escenario` con la ruta ACTUAL: `horas_media_dia = 5,58 h`, `horas_max_dia = 7,969 h`
  (< 8 porque son promedios de ciclo) → `prob_no_completar = 0`, `horas_extra_ciclo = 0`.
  **Visitas perdidas por exceder la jornada de 8 h = 0** (el mecanismo no existe: nada descarta
  visitas por horas).
- Lo que sí ocurre en la realidad de los datos (estimación con el mismo modelo horario, sólo para
  dimensionar): los 4 días sobre CAP (todos en COM2Z6: 2026-03-02 **46**, 03-04 **47**,
  03-09 **46**, 03-11 **47**) implican 8,48-8,65 h de viaje+servicio → **≈2,9-3,9 visitas/día que
  no caben** (≈13,6 visitas en total; el exceso sobre CAP=44 suma **10 visitas**). Esas visitas
  el motor las cuenta como cubiertas.

**Veredicto: PROBLEMA** (`cobertura ≡ P_ASIST` en esperanza: no mide sobrecarga; con
P_ASIST=1 da 1,0 pese a 4 días sobre CAP; visitas perdidas por jornada = 0). Archivos a corregir:
`app/core/montecarlo.py:199-212` (fórmula de cobertura) y `app/core/montecarlo.py:99-113`
(horas por promedios en vez del día concreto).

---

## C. DATOS Y CARGA

### C1. Los conteos de FRE_REP y DIAS_REP del diagnóstico anterior suman 5.000

**Pregunta.** ¿Sobre qué filas se calcularon? Repetición sobre las 14.757.

**Evidencia.** Reproducción exacta de los números previos (Mes completo 1.989 / 1ra 1.441 /
2da 1.399 / Sin asignar 171 = **5.000**) al calcularlos sobre **`df.head(5000)`** (las primeras
5.000 filas del CSV), no sobre el universo:

| fila | FRE_REP (Q1-Q2 / Q1 / Q2 / SIN_ASIGNAR) | suma |
|---|---|---|
| **`base.head(5000)`** | **1989 / 1441 / 1399 / 171** | **5.000** ← coincide con el diagnóstico anterior |
| base completa 16.560 | 6258 / 4936 / 4936 / 430 | 16.560 |
| **universo rutable 14.757** | **6114 / 4321 / 4322 / 0** | **14.757** |

Conteos correctos sobre las **14.757** (rutable):

- `FRE_REP`: Q1-Q2 = 6.114, Q2 = 4.322, Q1 = 4.321 (sin SIN_ASIGNAR: ya quedó fuera en el embudo).
- `periodo_rep`: Mes completo = 6.114, 2da quincena = 4.322, 1ra quincena = 4.321.
- `DIAS_REP`: JUEVES 2.501 · LUNES 2.492 · MIERCOLES 2.490 · SABADO 2.461 · MARTES 2.430 ·
  VIERNES 2.383 (= 14.757).

**Veredicto: PROBLEMA** (el diagnóstico anterior midió sobre las primeras 5.000 filas, no sobre
14.757; sesga los conteos). Archivo a corregir: el diagnóstico anterior en `reports/` (los datos
y el embudo de `app/core/universo.py:71-99` están bien).

### C2. Pico de ACTUAL: ¿por qué 47 (4 días sobre el tope) y antes 52?

**Pregunta.** ¿Se mide por fecha o por día de la semana? Top-5 asesor-día.

**Evidencia.** Se mide **por FECHA** (no por día de la semana):

```python
# app/core/montecarlo.py:67 (factibilidad)
    pico = ctf.groupby(["territorio", "fecha"])["visitas"].sum()
# app/core/calendario.py:90 (carga_asesor_dia)
    return (va.groupby([col_asesor, "fecha"], as_index=False)["visitas"].sum()
```

Top-5 asesor-día de ACTUAL (1.508 rutas asesor-día), CAP dinámico = **44** (`config.py:49`
`CAP_TOPE=45`; `config.py:148-152` `cap_dinamico()=44`):

| # | zona (ZON_REP) | fecha | visitas |
|---|---|---|---|
| 1 | COM2Z6 | 2026-03-11 | 47 |
| 2 | COM2Z6 | 2026-03-04 | 47 |
| 3 | COM2Z6 | 2026-03-09 | 46 |
| 4 | COM2Z6 | 2026-03-02 | 46 |
| 5 | COM2D4 | 2026-03-12 | 44 |

Los 4 días sobre el tope son todos de COM2Z6 (exceso total 10 visitas); `factible=False`.

**¿De dónde salía el 52?** Variante calculada sobre la **base completa (16.560 filas)** — es decir,
incluyendo las 1.803 filas que el embudo excluye, de las cuales **1.373 tienen visita>0 y
`DIAS_REP` válido**:

| variante (mismo reparto por fecha) | pico |
|---|---|
| rutable 14.757, `visitas_mes_rep`, por fecha (el del proyecto) | **47** |
| activos + visita (sin filtro de coords) 14.757 | 47 |
| **base completa 16.560 (los inactivos/sin visita con fechas)** | **52** |
| rutable, agrupando por día de semana en vez de fecha | 78 |
| rutable, sin filtro `FRE_REP` (todo el mes) | 78 |
| rutable, contando clientes en vez de visitas | 47 |

**Veredicto: OK** (47 es correcto para el universo rutable y está medido por fecha; el 52 anterior
corresponde a la base completa sin el embudo de `app/core/universo.py:73-83`). Archivo a corregir:
el diagnóstico anterior (`reports/`), no el código.

### C3. Distribución de visitas por asesor-día (percentiles y umbrales)

**Pregunta.** Percentiles, conteos sobre 45/52/70, repetición con `visitas_mes_bees`, zonas con
más de un asesor.

**Evidencia.** Sobre los 1.508 (territorio, fecha) de ACTUAL (semilla/embudo del proyecto):

| escenario | n | p50 | p90 | p99 | máx | >45 | >52 | >70 |
|---|---|---|---|---|---|---|---|---|
| solo `visitas_mes_rep` (lo que rutea el modelo) | 1.508 | 30 | 35 | 43 | 47 | 4 | 0 | 0 |
| **incluyendo `visitas_mes_bees`** (misma zona, fechas de `DIAS_BEES`/`FRE_BEES`) | 1.508 | 82 | 234 | 369 | **409** | 938 | 919 | 829 |

- `visitas_mes_rep` total = **45.212**/mes; `visitas_mes_bees` total = **122.182**/mes (14.929
  clientes con bees). **El motor de ruteo solo usa `visitas_mes_rep`** (`app/core/estructura.py:199`,
  `app/core/montecarlo.py:50`): las visitas bees no están modeladas en ningún K ni en ACTUAL.
- **Zonas con más de un asesor:** ninguna. La unidad de asesor **es** `ZON_REP`
  (`app/config.py:34-35`), hay 58 zonas = 58 asesores y **0 clientes** aparecen bajo 2 asesores
  distintos en `cache/rutas.json` (ACTUAL y recomendada; verificación sobre las 14.757 secuencias),
  y **0 `CodigoCliente` duplicados** en el universo. No existe en el CSV ninguna columna de
  persona asesora (solo `Supervisor`, que es el equipo).

**Veredicto: OK** (distribución reportada; zonas 1:1 con asesor). Discrepancia abierta (regla 8,
no resuelta): las **122.182 visitas bees/mes no están en el modelo** — si el mismo asesor las
atendiera, la carga real por asesor-día se multiplica (p50 30 → 82). A confirmar con el usuario
si bees corresponde a la misma fuerza de campo. Archivo relacionado: `app/core/estructura.py:199`.

### C4. Rango de K en la caché (35 a 67) contra el acordado (35 a 70)

**Pregunta.** Motivo del rango.

**Evidencia.** El rango hoy en caché **es 35–70 (36 valores), igual que el acordado**:

```python
# verificación sobre cache/montecarlo.json
n_K: 36 min: 35 max: 70
faltan: []                      # 35..70 sin huecos
claves mc: ['actual', 'n_sim', 'estrategias', 'comparaciones']   # comparaciones también 35..70
# cache/precompute_mc.log (una sola corrida, 1358 s)
• K pendientes: [35, …, 67, 68, 69, 70]
  K=67: pico=30 factible=True costo=205.9M … (35s)
  K=70: pico=28 factible=True costo=215.0M … (32s)
MonteCarlo listo en 1358s
```

- `app/config.py:108-110`: `K_MIN=35`, `K_MAX=70`, `K_PASO=1`; `cache/params.json` → `k_max=70`;
  endpoint `/api/montecarlo/estrategia/{k}` valida 35≤k≤70 (`app/main.py:166-167`);
  el dashboard usa `kMax = signal(70)` (`montecarlo.component.ts:54`) y muestra **todas** las
  filas del mapa sin filtros (`montecarlo.component.ts:192-210`).
- El "67" solo aparece como **número de zonas**: `eda.json → carga_zona.n_zonas = 67`,
  `universos.json → zonas_base = 67` y el texto del tool del chatbot
  `app/chatbot/herramientas.py:210` ("cuántas zonas (1-67)").
- La caché se escribe de forma **incremental y reanudable** (`app/precompute.py:137` pendientes y
  `app/precompute.py:148-151` guardado tras cada K): una lectura hecha **durante** una corrida
  parcial habría visto hasta el último K guardado (p. ej. 67).

**Veredicto: OK** (hoy el rango de caché = rango acordado 35–70; no se reproduce "35–67").
**NO CONCLUYENTE** sobre el origen exacto del "67" en el diagnóstico anterior: las dos causas
posibles son una lectura de caché a mitad de corrida (`app/precompute.py:148-151`) o confusión con
las **67 zonas** de `app/chatbot/herramientas.py:210`/`eda.json`. No hay código que corte en 67.

---

## D. CHATBOT

### D1. ¿El validador verifica cifras contra las salidas de las herramientas, o solo privacidad?

**Pregunta.** Alcance real del validador.

**Evidencia.** **Solo privacidad.** No hay ninguna comparación numérica entre la respuesta y las
salidas de las herramientas:

```python
# app/chatbot/validador.py:47-70
def validar_salida(respuesta: str, herramientas: list[dict] | None = None
                   ) -> dict[str, Any]:
    """Escanea la respuesta y las salidas de herramientas; enmascara PII.

    Devuelve {respuesta, limpio: bool, incidencias: [...]}. Sustituye claves
    prohibidas por su valor agregado si aparecieran.
    """
    incidencias: list[str] = []
    texto_resp = respuesta

    # 1. respuesta final
    limpio_resp = _escanea_texto(texto_resp)
    if limpio_resp != texto_resp:
        incidencias.append("respuesta: se enmascararon dígitos/correos/teléfonos")

    # 2. salidas de herramientas: claves prohibidas
    for tr in herramientas or []:
        salida = tr.get("salida", {})
        ...
                incidencias.append(f"herramienta {tr.get('nombre')}: mención a '{col}'")
```

- `_escanea_texto` solo sustituye cédulas/correos/teléfonos (`validador.py:21-31`); el paso 2 solo
  busca las columnas `Identificacion/Cliente/Direccion` (`validador.py:16,62-70`).
- Único punto de validación: `app/chatbot/__init__.py:22-23` → `validar_salida(respuesta, herramientas)`.
- El bucle (`app/chatbot/orquestador.py:176-217`) tampoco valida cifras: solo controla
  `content` vacío/truncado (`orquestador.py:183-196`).

**Veredicto: PROBLEMA** (no hay verificación de cifras; solo red de privacidad). Archivo a
corregir: `app/chatbot/validador.py:47-76`.

### D2. Casos 2 y 3 (seguimiento) con el LLM real

**Pregunta.** Herramienta usada, respuesta y si el validador la aprobó.

**Ejecución real** (misma sesión que el caso 1; modelo principal `openai/gpt-oss-120b` desde
`.env`; sin fallback):

| caso | pregunta | herramienta usada | ¿esperada? | respuesta (resumen) | validador |
|---|---|---|---|---|---|
| 1 | "¿Cuántos clientes y visitas al mes tiene el universo?" | `ver_universo` | sí ✔ | "Clientes: **14.757**; visitas/mes: **45.212** (≈1.738,9/día)" — correcto | `privacidad_ok=True`, 0 incidencias |
| **2** | "¿Y cuántas de esas zonas de rutero son rutable?" (seguimiento) | **`carga_por_zona`** | **no** (`espera_tool: ver_universo`) ✘ | "hay **67 zonas de rutero que son efectivamente ruteables**" — **incorrecto**: lo rutable son **58** (`universos.json`); 67 son las zonas de la base/EDA, que es lo que devuelve `carga_por_zona` (`herramientas.py:98-108` → `eda.json n_zonas=67`) | `privacidad_ok=True`, 0 incidencias → **aprobada pese al error** |
| **3** | "¿Y cuál es el máximo de visitas que puede hacer un asesor al día?" (seguimiento) | **NINGUNA** (`herramientas: []`) | **no** (`espera_tool: ver_parametros`) ✘ | "máximo **≈44 visitas/día** (CAP)" — coincide con `cap_dinamico()=44`, pero el número **no vino de una herramienta**: el prompt de sistema ya lo escribe (`app/chatbot/prompts.py:15`: "ningún asesor supere el CAP por día (≈44)") | `privacidad_ok=True`, 0 incidencias → **aprobada sin herramienta** |

- El test D5.7 `test_llm_real_casos_seguimiento` solo exige respuesta no vacía, no truncada y
  privacidad (`tests/test_llm_d5.py:294-297`); **no** verifica `espera_tool` ni la corrección de
  las cifras — por eso los 3 casos "pasan" aunque el caso 2 responda 67 en vez de 58 y el 3 no
  llame ninguna herramienta.
- La afirmación errónea "67 zonas ruteables" es además indistinguible para el validador: el número
  67 **sí** existe en la salida de la herramienta usada (`n_zonas`), pero mal etiquetado.

**Veredicto: PROBLEMA** (caso 2 con herramienta errónea y cifra incorrecta 67/58; caso 3 sin
herramienta; ambos aprobados porque el validador solo mira PII). Archivos a corregir:
`app/chatbot/validador.py:47-76` (sin verificación de cifras, ver D1) y
`tests/test_llm_d5.py:276-299` (no valida `espera_tool`).

---

## Tabla resumen

| Punto | Veredicto | Archivo a corregir |
|---|---|---|
| A1 fuga de datos / filas de DEP-territorios-KMeans | **PROBLEMA** (el notebook calcula DEP con train+test antes del split; en la API no hay split y el K-Fold excluye el test sin evaluarlo) | `Modelo_Ruteo_KMeans_NN_ACO_11.10.ipynb` celdas 9 y 19 / `modelo_ruteo_v2.ipynb` celda 10; `app/config.py:82-83`; `app/core/pliegues.py:5` |
| A2 calibración de β | **PROBLEMA** (calibrador nunca llamado; muestra con 2 sub-rutas de 1 cliente; β usada 3,0 ≠ calibrada 5,0) | `app/core/ruteo.py:334` (sin filtro ≥3) y `app/config.py:93` |
| A3 selección del K con las mismas simulaciones | **PROBLEMA** (misma corrida MC para elegir y reportar) + resultados de la re-evaluación con semilla distinta | `app/precompute.py:120` y `app/precompute.py:222-229` |
| A4 K-Fold train/test por cliente y ranking NN vs ACO | **PROBLEMA** (brecha 54,25 % ± 2,82 >> 10 % en los 5 pliegues y hoy no se calcula métrica de test) / ranking **OK** (ACO gana en los 5 pliegues) | `app/core/pliegues.py:52-89` |
| A5 experimento multi-semilla, NN con desv. 0 | **OK** (NN determinista; la variación solo entra en `aco(..., semilla=SEMILLA+s)`; no hay split train/test) | `app/core/comparacion.py:178-196` (documenta/limite) |
| B1 algoritmo de re-balanceo | pendiente de la tabla por K | `app/core/estructura.py:31-91` |
| B2 km ×2,7 del re-balanceo | **PROBLEMA** (con 5 clientes movidos ACTUAL ya es factible; el salto viene del re-clustering con penalizaciones de balance) | `app/core/estructura.py:31-91` |
| B3 COSTO_ASESOR_MES y cuenta ACTUAL | **OK** (133,320 M cuadra: 104,400 + 28,920 + 0,000) con etiquetas contradictorias | `app/config.py:57` vs `app/config.py:172` |
| B4 cobertura = P_ASIST | **PROBLEMA** (con P_ASIST=1 la cobertura da 1,0 pese a 4 días sobre CAP; visitas perdidas por jornada = 0) | `app/core/montecarlo.py:199-212` y `app/core/montecarlo.py:99-113` |
| C1 conteos que suman 5.000 | **PROBLEMA** (eran `df.head(5000)`; conteos reales de 14.757 arriba) | diagnóstico anterior en `reports/` |
| C2 pico 47 vs 52 | **OK** (47 = universo rutable, por fecha; 52 = base completa 16.560) | diagnóstico anterior en `reports/` |
| C3 distribución y visitas bees | **OK** + discrepancia abierta: 122.182 visitas bees/mes sin modelar; ninguna zona con >1 asesor | `app/core/estructura.py:199` (bees fuera del ruteo) |
| C4 rango de K 35–67 vs 35–70 | **OK** (caché = 35–70 con 36 K) / **NO CONCLUYENTE** en el origen del "67" (¿lectura a mitad de corrida o confusión con 67 zonas?) | `app/precompute.py:148-151`, `app/chatbot/herramientas.py:210` |
| D1 alcance del validador | **PROBLEMA** (solo PII; no verifica cifras) | `app/chatbot/validador.py:47-76` |
| D2 casos 2 y 3 con LLM real | **PROBLEMA** (caso 2: herramienta errónea y cifra 67≠58; caso 3: sin herramienta; ambos aprobados) | `app/chatbot/validador.py:47-76` y `tests/test_llm_d5.py:276-299` |
