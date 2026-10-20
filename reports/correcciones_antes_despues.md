# Correcciones E1–E11 · antes y después

> **FASE 1** · Cada corrección tiene (1) test automatizado en
> `tests/test_correcciones.py`, (2) tabla antes/después y (3) su ubicación en el
> núcleo `app/core/`. Los notebooks originales **no se ejecutaron**; se portó la
> lógica corregida al núcleo y se dejaron copias `*_v2.ipynb` con las celdas
> corregidas y la nota `⚠ Corregido: re-ejecutar`.
>
> **Cifras:** todas salen de cálculos sobre `data/base_clientes_limpia.csv`
> (fuente de verdad). Nada inventado. Los supuestos llevan su etiqueta.

## Resumen ejecutivo

| ID | Corrección | Test | Estado |
|----|-----------|------|--------|
| E1 | `% de semillas en que gana` invertido en la tabla de estabilidad | `test_e1_*` | ✅ |
| E2 | Ruta abierta: la ventana del **último** cliente no se verificaba | `test_e2_*` | ✅ |
| E3 | Calibración de β sobre las sub-rutas **más pequeñas** | `test_e3_*` | ✅ |
| E4 | Ancla de silueta `k=1.6·k_cap` y sin fusión de sub-rutas mínimas | `test_e4_*` | ✅ |
| E5 | `pipeline_test` sin comparación por pareado/Wilcoxon | `test_e1_*` (victorias) | ✅ |
| E6 | `balance_carga` con llave `tid` colisionaba entre CDs (CV=2,00) | `test_e6_*` | ✅ |
| E7 | Factibilidad por nº de sub-rutas, no por **asesor-día** vs CAP | `test_e7_*` | ✅ |
| E8 | ACTUAL desde pares `(Supervisor, Cod CD)`, no desde **ZON_REP** | `test_e8_*` | ✅ |
| E9 | Depósito compartido por CD (5) en vez de por **territorio** | `test_e9_*` | ✅ |
| E10 | Nombres de equipo con caracteres dañados U+FFFD sin normalizar | `test_e10_*` | ✅ |
| E11 | Tres funciones de carga distintas (universos distintos) | `test_e11_*` | ✅ |

---

## E1 · % de victorias invertido (Modelo, sección 8.2)

**Antes** — `resumen(exp)` asignaba `[100*(ACO<NN), 100*(NN<ACO)]` a un índice
cuyo orden era `["NN","ACO"]`; la fila **NN** recibía las victorias de **ACO**.

| Fila | Antes | Después |
|------|-------|---------|
| NN  | **100 %** (incorrecto) | 0 % (ACO más corto en 5/5 semillas) |
| ACO | **0 %** (incorrecto) | 100 % |

Con datos de prueba `NN=[100]×5`, `ACO=[90]×5`: antes mostraba "NN 100 % / ACO 0 %"
**y ACO era más corto**; ahora `ACO 100 % / NN 0 %`.

- **Fix:** `app/core/comparacion.py::resumen_estabilidad` — cada fila recibe su
  propia condición de victoria (`NN<ACO` para NN, `ACO<NN` para ACO).
- **Test:** `test_e1_porcentaje_victorias_no_invertido`, `test_e1_sin_inversion_con_datos_reales`.

## E2 · Ruta abierta: se ignoraba la ventana del último cliente (Modelo, ACO/2-opt)

**Antes** — `slots_factibles([slots[i] for i in orden[1:-1]])` descartaba
`orden[-1]`. Con `RUTA_CERRADA=False` (D4) el último cliente **sí** se sirve y su
ventana debe verificarse. El 2-opt tampoco podía invertir tramos con el último nodo.

**Después** — `segmento_servible` devuelve `orden[1:]` en ruta abierta; el 2-opt
recorre `j` hasta `n-1`.

| Caso | Antes (`[1:-1]`) | Después (`[1:]`) |
|------|------------------|------------------|
| slots `[{0},{2},{0}]` (último exige slot 0 y ya avanzó a 2) | factible ❌ | **infactible ✅** |

- **Fix:** `app/core/ruteo.py::segmento_servible`, `factible_orden`, `dos_opt`.
- **Test:** `test_e2_ventana_ultimo_cliente_en_ruta_abierta`.

## E3 · Calibración de β sin estratificación (Modelo, sección 6)

**Antes** — se calibraba β sobre "unas pocas sub-rutas pequeñas/train", sesgando
hacia rutas cortas.

**Después** — `muestra_estratificada` toma ≥1 sub-ruta de **cada** rango
(1–20, 21–30, 31–40, 41–45).

| Muestra | Antes | Después |
|---------|-------|---------|
| rangos cubiertos | sólo los pequeños | **[1–20, 21–30, 31–40, 41+]** ✅ |

- **Fix:** `app/core/ruteo.py::muestra_estratificada`, `rangos_cubiertos`, `calibrar_beta`.
- **Test:** `test_e3_muestra_estratificada_cubre_los_4_rangos`, `test_e3_no_solo_pequenas`.

## E4 · Ancla de silueta y sub-rutas mínimas (Modelo, sección 5)

**Antes** — ventana de silueta `[k_cap, 1.6·k_cap]` (sobre-clusteriza) y sub-rutas
muy pequeñas se conservaban sueltas.

**Después** — `K_EXTRA_MAX = 1.0` (ancla exacta `k = k_cap`) y fusión de
sub-rutas con `< MIN_CLIENTES_RUTA = 15` clientes con la más cercana si cabe en CAP.

- **Fix:** `app/config.py::K_EXTRA_MAX`, `MIN_CLIENTES_RUTA`; `app/core/ruteo.py::formar_subrutas`, `fusionar_subrutas`.
- **Test:** `test_e4_k_extra_max_ancha_k_en_k_cap`.

## E5 · Comparación A vs B sin pareado (Modelo, sección 8.2)

**Antes** — sólo media/desv por estrategia, sin prueba por pareado sobre las
mismas semillas.

**Después** — `comparacion.victorias` + `test_wilcoxon` (Wilcoxon pareado).

- **Fix:** `app/core/comparacion.py::test_wilcoxon`.
- **Test:** cubierto junto con E1.

## E6 · `balance_carga` con llave colisionante (MonteCarlo)

**Antes** — `r["asesor"]` era el `tid` (0..kc−1) **dentro de cada CD**; con K=5
todos los `tid=0` caían en la misma llave → `[14750,0,0,0,0]` → **CV = 2,00**.

**Después** — la llave es `territorio` = `ZON_REP` (globally único) o índice `0..K−1`.

| Métrica | Antes | Después |
|---------|-------|---------|
| CV carga (K=5) | **2,00** (colisión) | — |
| CV carga ACTUAL (58 zonas) | no calculable | **0,106** |
| CV carga K=58 balanceado | no calculable | **0,006** |

- **Fix:** `app/core/montecarlo.py::balance_carga`; llave `territorio` en `estructura`.
- **Test:** `test_e6_balance_carga_sin_colision`.

## E7 · Factibilidad sin restricción por asesor-día (MonteCarlo)

**Antes** — la factibilidad se infería del nº de sub-rutas y `CAP` fijo; no se
evaluaba **cada día** contra la capacidad dinámica.

**Después** — `factibilidad` cuenta los asesor-días `(territorio, fecha)` que
superan `CAP dinámico = 44`, con utilización por quincena.

| Métrica (ACTUAL) | Antes | Después |
|------------------|-------|---------|
| unidad de restricción | sub-ruta | **asesor-día (fecha real)** |
| CAP | fijo 45 | **dinámico 44** (D3) |
| pico ACTUAL | sin evaluar | **47** visitas/día |
| asesor-días sobre CAP | sin evaluar | **4** (de 1.508) |
| K_mín (fórmula teórica `⌈pico/CAP⌉`) | K=5 (histórico) | **41 teórico / 43 real** |

- **Fix:** `app/core/montecarlo.py::factibilidad`, `horas_por_escenario`.
- **Test:** `test_e7_factibilidad_por_asesor_dia`, `test_e7_cap_dinamico_es_int`.

## E8 · ACTUAL desde `(Supervisor, Cod CD)` en vez de ZON_REP (MonteCarlo)

**Antes** — `asignar_asesores(6, dep)` repartía proporcionalmente sobre pares
`(Supervisor, Cod CD)` (15 grupos, 2 bajo umbral) → no reflejaba la operación real.

**Después** — ACTUAL se construye desde la asignación real de clientes a
**ZON_REP** (D1). El Supervisor queda como atributo de equipo.

| Concepto | Antes | Después |
|----------|-------|---------|
| unidad de asesor | `(Supervisor, Cod CD)` | **`ZON_REP`** |
| nº territorios ACTUAL | 15 grupos (2 infrautilizados) | **58 zonas** |
| carga/zona | no equiparable | **21–40 (media 30) visitas/día** |
| zonas base → rutable | no rastreado | **67 → 58** (9 perdidas: 100 % inactivas) |

- **Fix:** `app/core/estructura.py::construir_actual`.
- **Test:** `test_e8_actual_desde_zon_rep`.

## E9 · Depósito por CD en vez de por territorio (MonteCarlo)

**Antes** — depósito compartido por CD (5 puntos), no comparable con el modelo de
ruteo (centroides Asesor×CD).

**Después** — `DEPOSITO_MODO = "territorio"`: centroide de cada territorio.

- **Fix:** `app/core/estructura.py::DEPOSITO_MODO`, `_depot_de`.
- **Test:** `test_e9_deposito_por_territorio`.

## E10 · Nombres de equipo dañados U+FFFD (EDA / chatbot)

**Antes** — `Sirley Damaris Rodr\ufffdguez Usaqu\ufffdn` se usaba tal cual
(búsqueda por nombre fallaba, agrupaciones se partían).

**Después** — tabla puente `app/knowledge/asesores.json` + `normalizar_equipo`
(exacta + difusa ≥0,8) + `_norm` que elimina U+FFFD.

| Antes | Después |
|-------|---------|
| `Sirley Damaris Rodr\ufffdguez Usaqu\ufffdn` | **`Sirley Damaris Rodríguez Usaquén`** |
| 6 equipos con 1 dañado | **6 equipos canónicos limpios** |
| `buscar_asesor("sirley rodriguez")` | **`ok=True`, coincidencia≈0,98** |

Columnas detectadas con U+FFFD (privacidad/limpieza): `Cliente`, `Direccion`, `Supervisor`.

- **Fix:** `app/core/asesores.py::normalizar_equipo`, `buscar_asesor`, `_norm`.
- **Test:** `test_e10_normaliza_nombre_danado_sirley`, `test_e10_busqueda_difusa`, `test_e10_columnas_danadas_detectadas`.

## E11 · Tres funciones de carga (universos distintos)

**Antes** — EDA, Modelo y MonteCarlo cargaban/filtraban por separado → universos
diferentes y no reconciliables.

**Después** — **una** función `cargar_universo` con embudo compartido.

| Paso (embudo) | Filas | Eliminadas |
|---------------|------:|-----------:|
| Base limpia completa | 16.560 | 0 |
| Con ZON_REP (zona) | 16.560 | 0 |
| Solo activos | 14.796 | 1.764 |
| Con visita asignada (`visitas_mes_rep>0`) | 14.757 | 39 |
| Con coordenadas válidas | 14.757 | 0 |

Universo único: **14.757 clientes**, **45.212 visitas/mes**, **58 zonas rutable**
(67 base − 9 perdidas), **15 grupos Asesor×CD** (2 bajo umbral de 20).

- **Fix:** `app/core/universo.py::cargar_universo`, `resumen_universos`.
- **Test:** `test_e11_unica_funciona_carga`.

---

## Supuestos adjuntos (etiquetados)

| Parámetro | Valor | Etiqueta |
|-----------|------:|----------|
| `UNIDAD_ASESOR` | `zon_rep` | calibrado (D1) |
| `DURACION_CICLO_DIAS` | 30 | calibrado (D2) |
| `CAP dinámico` | 44 | supuesto (D3) |
| `COSTO_ASESOR_MES` | 1.800.000 COP | **supuesto — ⚠ sin confirmar (¿1,5 M?)** |
| `COSTO_KM` | 1.200 COP/km | supuesto |
| `COSTO_HORA` (solo extra) | 12.000 COP/h | supuesto |
| `COSTO_VISITA_PERDIDA` | `None` | supuesto (sin conversión a pesos) |
| `P_ASIST` | 0,85 | supuesto |
| `RUTA_CERRADA` | `False` | calibrado (D4) |
| `K` rango | 35–70, ACTUAL=58 | calibrado (P2) |

---

## ⚠ Histórico NO comparable (regla 8 — no mezclar con las cifras nuevas)

Las cifras de los notebooks originales y del planteamiento **no son comparables**
con las de este núcleo, porque cambió el modelo (ciclo, visitas, depósitos, costos):

| Cifra publicada / histórica | Por qué no es comparable con el núcleo nuevo |
|-----------------------------|-----------------------------------------------|
| **66,9 M (K=5)** y demás costos por K (notebook MonteCarlo original) | K=5..24 y costo fijo **1.500.000/ciclo**; ahora K=35..70 y costo fijo **mes×duración/30**. Distinto régimen de K y de costo. |
| **costo fijo 1,5 M por quincena** | El usuario lo reclasificó a **mensual** (D3). Ver discrepancia #2. |
| **12.698 km / 692 sub-rutas** (notebook Modelo) | Depósito por `(Supervisor×CD)` y **una pasada** de clientes; ahora rutas **por día** con depósito por territorio y 45.212 visitas/ciclo. |
| **CAP=45, K_MIN=5** (parámetros originales) | Sustituidos por **CAP dinámico=44** y **K=35..70** (P2). |
| **70–80 visitas/día** (planteamiento) | Los datos dan **máx 52** visitas/día por zona (pico asesor-día ACTUAL 47). Ver discrepancia #7 al final de la entrega. |

> Cualquier comparación de costo/km del núcleo nuevo contra estas cifras es
> **inválida**. Los deltas del dashboard son **Δ vs ACTUAL (K=58)** dentro del
> mismo modelo nuevo, con IC95.

## Discrepancias abiertas (regla 8 — reportadas, no resueltas en silencio)

1. **Nombre de notebooks v2** — se adoptó `modelo_ruteo_v2.ipynb` / `montecarlo_asesores_v2.ipynb` (creados; salidas limpiadas + notas `⚠ Corregido: re-ejecutar`). *Resuelto salvo objeción.*
2. **Costo fijo** — punto 5 del usuario dijo "1,5 M", decisión D3 lo reclasificó a mensual **1,8 M**; **sin confirmar**. Afecta el nivel absoluto de costo, no el orden por K.
3. **67 vs 58 zonas** — 9 zonas de la base se pierden (100 % inactivas/sin visitas).
4. **K_mín** — fórmula teórica `⌈1785/44⌉ = 41`; con territorios geográficos el real es **43** (primer K factible; en K=43 el pico es 44 = CAP, justo en el límite del criterio estricto `pico > CAP`). De 44 en adelante el pico queda por debajo del CAP. La frontera teórica (41) subestima en ~2 asesores la necesidad real por la concentración geográfica de `DIAS_REP`.
5. **CV asesor-día (0,161) > CV mensual (0,106)** — el efecto `DIAS_REP` empeora la desigualdad diaria (P1: leer en las dos).
6. **ACTUAL no es factible estricto** — 4 asesor-días sobre CAP (pico 47 > 44).
7. **70–80 visitas/día vs máx 52** — el planteamiento habla de 70-80 pero los datos dan máx 52 (pendiente de investigar: días con 5 ocurrencias, visitas BEES, varias zonas por persona).
