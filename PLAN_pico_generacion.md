# Plan de diagnóstico: pico de generación post-percolación

**Objetivo:** determinar la causa del llenado en un solo paso de la banda de generación
tras percolar el primer filamento, distinguir entre error de programación, decisión de
modelado provisional y física intrínseca mal resuelta — y decidir la corrección.

---

## Hechos ya establecidos (autopsia de `Results/simulation_1`)

| # | Hecho | Evidencia |
|---|-------|-----------|
| H1 | El pico ocurre en el paso 8567 (V = 0.9424 V), exactamente 1001 pasos tras percolar el filamento 1 (paso 7566): salto de 184 → 902 vacantes. Es la creación instantánea del filamento 2. | `Data_pp_set_1.npz` (vacantes) + `logs/log_simulacion_1.log` |
| H2 | En el pre-pico (paso 8550) la temperatura es 572 K **uniforme en toda la matriz**: rama escalar `Temperature_Joule` con `r_termica_no_percola·5` + `np.full_like` (activa mientras `all_CFs_created=False`). | `Estado_pp_set_sim_1_paso_8550.npz` |
| H3 | La FVM resuelta offline para ese mismo estado (misma I = 5.77 mA, misma R_cell, mismo factor_generar_calor) da 383 K en el núcleo del filamento 1 y 301–365 K en la banda del filamento 2. La rama escalar sobreestima la tasa de generación en la zona del filamento 2 en un factor ~5·10⁵. | Script offline con `Temperature.solve_thermal_state` |
| H4 | Umbral de saturación: con ν·Δt = 10¹⁰, P→1 cuando T ≳ 530–580 K (menos con campo). El ×5 arbitrario (572 K) cruza el umbral; sin él (354 K) queda 200 K por debajo (τ ≈ 45 min: no generaría nada). | Cálculo Arrhenius con las constantes del config |
| H5 | Bug real de signo contrario: en `PP_set`/`SP_set` el bucle de campo es `for i in range(0, params.x_size)` (40 iteraciones) sobre un estado de 120 filas → `E_field_vector[40:120] = 0` siempre. En `PP_reset` está bien (`shape[0]`). Corregirlo **agrava** el pico, no lo causa. | `phases_set.py:174` vs `phases_reset.py:130` |
| H6 | El modelo de gap del campo es correcto (fila = 40 celdas = 10 nm = device_size_x); no hay cruce de ejes ni L negativo. | Verificado con la forma real (120, 40) |
| H7 | `P = min(r·Δt, 1)` es la aproximación lineal de `P = 1 − exp(−r·Δt)`; sobreestima en la zona intermedia (38% a 570 K) y el clip enmascara desbordes de órdenes de magnitud. | `Generation.py:66-69` |
| H8 | Δt = 1 ms vs τ(650 K) ≈ 80 µs: incluso con la probabilidad exacta, a T de filamento realistas (600–700 K) el paso no resuelve la transición ni su realimentación interna (cada vacante nueva cambia E, I, T). | Tabla de tasas Arrhenius |

**Hipótesis de trabajo:** el pico lo fabrica la rama provisional de temperatura
(uniforme + ×5 arbitrario); el muestreo de paso fijo con clip colapsa la transición en
1 paso; el bug del bucle E es independiente y de signo contrario; y existe un hueco de
diseño latente: sin el crosstalk térmico artificial, no está claro qué física nuclea el
filamento 2.

---

## Fase 0 — Preparación (una sola vez, antes de cualquier corrida)

- [ ] **P0.1 Semilla fija.** `np.random.seed(N)` al inicio de `PP_set` para que las corridas sean comparables entre sí.
- [ ] **P0.2 Instrumentación por paso** (parche temporal, ~20 líneas): loguear `k, V, I, R_total, T_max, T_mediana_banda, max(r·Δt) SIN clip, nº celdas con P ≥ 0.99, vacantes nuevas en el paso`. Esto convierte cada corrida en evidencia cuantitativa.
- [ ] **P0.3 Quitar la pausa artificial** de 1000 pasos (`PASOS_PAUSA_GENERACION_POST_PERCOLACION`): ya cumplió su función diagnóstica y contamina las comparaciones.

**Criterios de medida comunes a todas las corridas:**
- *Pico* = existe un paso con Δvacantes > 5% de la banda de la máscara.
- *Gradual* = el llenado equivalente se reparte en ≥ 100 pasos.
- Registrar siempre: paso y V de percolación, paso y V de creación del filamento 2, `max(r·Δt)` alcanzado.

---

## Fase 1 — Ablación: aislar el mecanismo (1 cambio por corrida)

| Corrida | Cambio único vs. baseline | Predicción | Si la predicción falla, significa… |
|---------|---------------------------|------------|-------------------------------------|
| **R1 Baseline** | Ninguno (solo P0) | Reproduce el pico; `max(r·Δt)` salta a ≳10³ en el paso de percolación y se mantiene (régimen estacionario, no transitorio) | Los datos guardados no eran representativos; repetir autopsia |
| **R2 Sin ×5** | `r_termica_no_percola·5 → ·1` en `phases_set.py:356` | El pico desaparece; probablemente el filamento 2 **no se forma nunca** (T ≈ 354 K, τ ≈ 45 min) | Si el pico persiste: hay otro aporte térmico o de campo no identificado |
| **R3 T localizada** | `np.full_like` → aplicar la T escalar solo a la banda del filamento ya creado (centro ± grosor); resto a `init_temp`. Alternativa más fiel: arrancar la FVM desde el primer filamento percolado | El filamento 1 engorda de forma acotada; la banda del filamento 2 queda fría y no se dispara | Si el pico persiste en la banda del f2: el driver no era el crosstalk térmico |
| **R4 Fix bucle E** | `range(0, params.x_size)` → `range(0, actual_state.shape[0])` en `PP_set` y `SP_set` | El pico se adelanta o intensifica levemente (las filas 40–119 recuperan su campo); cuantificar cuánto aporta el campo a la nucleación del f2 | Si cambia drásticamente el comportamiento pre-percolación: revisar la calibración de γ |
| **R5 T congelada en generación** | Pasar `T = 300 K` **solo** a `update_state_generation` (el resto del bucle intacto) | El pico desaparece por completo → confirma driver 100% térmico | Si queda pico residual: el campo de gap contribuye de forma no despreciable |
| **R6 Campo simple** *(opcional)* | Sustituir `GapElectricField` por `SimpleElectricField` solo en generación | Cambio menor (el término de campo aporta ~0.25 eV a V≈1); el pico se mantiene | Si el pico desaparece: el gap contribuía más de lo estimado |

**Salida de Fase 1:** atribución cuantitativa del pico entre (a) nivel térmico (×5),
(b) extensión espacial (full_like), (c) campo. Con R1, R2, R3 y R5 basta para el
veredicto; R4 y R6 acotan el papel del campo.

---

## Fase 2 — Corrección del muestreo: decidir el integrador

Estas corridas prueban las correcciones acumulativas (aquí sí se combinan, porque son
capas de la misma solución):

| Corrida | Cambio (acumulativo) | Qué valida |
|---------|----------------------|------------|
| **R7 Probabilidad exacta** | `P = min(r·Δt, 1)` → `P = 1 − exp(−r·Δt)` en `calcular_probabilidad_generacion` (elimina también el `np.minimum`) | Corrección matemática sin coste. NO evita el llenado rápido a T altas (a 650 K, P exacta ≈ 0.99999), pero elimina el sesgo del 38% en zona intermedia y el clip |
| **R8 Sub-stepping adaptativo** | R7 + si `max(r·Δt) > 0.1`: dividir el paso de generación en `n = ceil(max(r)·Δt / 0.1)` sub-pasos con `Δt/n`, re-evaluando entre sub-pasos el campo de gap por fila (barato) y la corriente/temperatura escalar; FVM cada m sub-pasos si cuesta. V constante dentro del paso (la rampa es ~10⁶ veces más lenta) | La transición se resuelve en el tiempo: el llenado pasa de 1 paso a una secuencia ordenada donde la realimentación actúa. Medir: nº de sub-pasos consumidos, evolución de I y T dentro de la transición |
| **R9 Compliance de corriente** | R8 + límite `I_cc` (tomar el del setup experimental): si `I ≥ I_cc`, el voltaje efectivo sobre el dispositivo pasa a `V_dev = I_cc · R_total` y con él se recalculan E y T | El freno físico: a 600–700 K el crecimiento debe autolimitarse (menos V_dev → menos potencia → menos T → menos tasa), como en el dispositivo real. Sin compliance, un modelo a V impuesto SIEMPRE completará el SET de forma "instantánea" a escala de rampa — eso es física del modelo, no bug |

**Nota sobre Δt:** no hace falta reducir `num_pasos`/`total_simulation_time`
globalmente (cambiaría todo el bucle, como señalaste). El sub-stepping refina solo el
muestreo de generación dentro del paso, manteniendo la rampa experimental de 10 s.

---

## Fase 3 — Matriz de decisión

| Resultado observado | Conclusión | Acción |
|---------------------|------------|--------|
| R2 o R3 eliminan el pico por separado | **Confirmado: error de modelado provisional** (T uniforme + ×5 arbitrario), no física intrínseca | Retirar la rama provisional: usar FVM (o T por filamento, rama `feature-Temperatura_Individual`) desde el primer filamento percolado; eliminar el ×5 o calibrarlo contra la FVM |
| R5 elimina el pico y R6 no cambia nada | Driver 100% térmico; el campo es secundario | Prioridad total al modelo térmico |
| Tras R2+R3+R4, el filamento 2 se forma gradualmente vía campo a mayor V | El modelo físico es autosuficiente para nuclear el f2 | Documentar el mecanismo; validar V de creación del f2 contra datos experimentales |
| Tras R2+R3+R4, el filamento 2 NO se forma | **Hueco de diseño**: el crosstalk artificial tapaba la falta de mecanismo de nucleación del f2 | Decisión de modelado (no bug): ¿FVM con acoplo térmico real entre filamentos? ¿recalibrar γ/Ea? ¿semilla de nucleación explícita? |
| Con R8, `max(r·Δt)` sigue >0.1 en regímenes legítimos (600–700 K) y el llenado es rápido pero ordenado | El SET abrupto es física real del modelo | Mantener sub-stepping como resolución y R9 (compliance) como límite físico; el pico "en 1 paso" queda explicado como artefacto ya corregido |
| El pico persiste con TODO lo anterior | Hay un mecanismo no identificado | Volver a la autopsia con la instrumentación de P0.2 sobre la nueva corrida |

---

## Fase 4 — Validación final

- [ ] Corrida con el paquete de correcciones decidido (semillas distintas, ≥3 repeticiones).
- [ ] Comparar contra experimento: V_set, abruptez de la transición I–V, resistencia post-SET, V de creación de cada filamento.
- [ ] Verificar que `max(r·Δt) ≤ 0.1` en todos los sub-pasos de toda la corrida (el "guardarraíl" ya no aborta: refina).
- [ ] Retirar la instrumentación temporal de P0.2 o dejarla tras un flag de debug.

## Correcciones de código independientes del diagnóstico (hacer de todos modos)

- [x] Desajuste cabecera/datos `I_i, R_i` en los ficheros de datos (ya corregido).
- [ ] Bug del bucle E-field en `PP_set`/`SP_set` (`range(x_size)` → `range(shape[0])`) — se prueba en R4 y se incorpora al final.
- [ ] `P = 1 − exp(−r·Δt)` (R7) — correcto siempre, sin contrapartidas.
- [ ] Eliminar la pausa artificial post-percolación y el bloque "TEST TEMPORAL" comentado en `state_updates.py`.

---
---

# PLAN DE ACTUACIÓN DETALLADO POR CORRIDA

## Infraestructura común (se hace UNA vez, sirve para todas las R)

Para no editar líneas a mano antes de cada corrida (fuente clásica de errores en
ablaciones), todas las variantes se controlan con **flags de entorno** leídos por un
módulo central nuevo:

**Nuevo fichero `RRAM/diag.py`:**

```python
"""Flags de diagnóstico para el plan del pico de generación. Leídos del entorno.
Todos apagados por defecto => comportamiento baseline. ELIMINAR al cerrar el plan."""
import os

def _b(name): return os.environ.get(name, "0") == "1"

SEED           = int(os.environ.get("DIAG_SEED", "0"))      # 0 = sin semilla
NO_X5          = _b("DIAG_NO_X5")           # R2: r_termica*5 -> *1
T_LOCAL        = _b("DIAG_T_LOCAL")         # R3: T escalar solo en banda del CF creado
FIX_EFIELD     = _b("DIAG_FIX_EFIELD")      # R4: bucle E sobre las 120 filas
T_FROZEN_GEN   = _b("DIAG_T_FROZEN_GEN")    # R5: T=300K solo en la generación
SIMPLE_FIELD   = _b("DIAG_SIMPLE_FIELD")    # R6: campo simple V/L en la generación
PROB_EXP       = _b("DIAG_PROB_EXP")        # R7: P = 1-exp(-r*dt)
SUBSTEP        = _b("DIAG_SUBSTEP")         # R8: sub-stepping adaptativo
ICC            = float(os.environ.get("DIAG_ICC", "0"))     # R9: compliance [A], 0 = off
```

**Instrumentación (P0.2), en `RRAM/state_updates.py` dentro de
`update_state_generation`**, justo después de calcular `prob_final` (línea ~60):
el dato clave es `r·Δt` **sin** clip. `get_generation_probabilities_matrix` ya
clipa a 1, así que se recalcula el valor crudo o (mejor) se añade a
`Generation.get_generation_probabilities_matrix` un retorno extra `prob_raw`
(la matriz antes del `np.minimum`). Volcar por paso a un CSV:

```python
# --- DIAG: métricas por paso (append a Results/diag_run.csv) ---
diag_row = dict(
    k=num_iteracion,
    max_rdt=float(prob_raw.max()),            # r*dt sin clip
    n_sat=int((prob_final >= 0.99).sum()),    # celdas saturadas
    n_cand=int((prob_final > 0).sum()),
    T_max=float(np.max(temp_matrix)), T_med=float(np.median(temp_matrix)),
)
```

y en `PP_set`, tras la llamada, añadir `vacantes_nuevas = np.sum(actual_state) - total_vacantes`,
`V`, `I`, `R_total` a la misma fila. Un CSV por corrida.

**Semilla (P0.1), al inicio de `PP_set` (`phases_set.py`, tras la línea 69):**

```python
from . import diag
if diag.SEED: np.random.seed(diag.SEED)
```

**Pausa artificial (P0.3):** eliminar el bloque de `phases_set.py:92-95` y `:392-396`
(las líneas marcadas `TEMPORAL`).

**Preservar resultados:** cada corrida sobreescribe `Results/simulation_<N>/`. Tras
cada R, archivar:

```bash
mv Results/simulation_1 Results_diag/R<X>_simulation_1 && cp Results/diag_run.csv Results_diag/R<X>.csv
```

**Lanzamiento común** (solo la fase donde vive el pico):

```bash
DIAG_SEED=42 [flags de la R] python -m RRAM exec 1 --stop-at pp_set
```

**Análisis común tras cada corrida** (script único, p. ej. `Utilidades/analisis_diag.py`):
lee el CSV y reporta: paso/V de percolación, paso/V de creación del CF2, máximo salto
de vacantes en un paso, serie temporal de `max_rdt` y `T_max`, nº de pasos que dura el
llenado de la banda. Ese resumen de ~8 números es lo que se compara entre corridas.

---

## R1 — Baseline instrumentado (control)

- **Objetivo:** reproducir el pico con métricas; línea de referencia de todas las comparaciones.
- **Cambios de código:** solo infraestructura común (P0.1–P0.3). Ningún flag activo.
- **Lanzar:** `DIAG_SEED=42 python -m RRAM exec 1 --stop-at pp_set`
- **Registrar:** paso de percolación k_p; verificar que en k_p+1 (ya sin pausa) `max_rdt` salta ≥3 órdenes de magnitud y `vacantes_nuevas` ≈ tamaño de banda.
- **Criterio de éxito del control:** pico presente (Δvac > 5% de banda en 1 paso) y `max_rdt ≫ 1` estacionario desde k_p. Si `max_rdt` fuese < 1 en el pico, la hipótesis térmica está mal y hay que volver a la autopsia.
- **Nota:** sin pausa el pico ocurrirá en k_p+1 (≈7567), no en 8567 — esperado.

## R2 — Sin el ×5 arbitrario

- **Objetivo:** medir cuánto del pico lo causa la parametrización arbitraria del nivel térmico.
- **Cambio:** `phases_set.py:356`:
  ```python
  r_term = sim_ctes.r_termica_no_percola * (1 if diag.NO_X5 else 5)
  temperatura = Temperature.Temperature_Joule(voltage, current, T_0=params.init_temp, r_termica=r_term)
  ```
- **Lanzar:** `DIAG_SEED=42 DIAG_NO_X5=1 python -m RRAM exec 1 --stop-at pp_set`
- **Predicción cuantitativa:** T post-percolación ≈ 354 K → `max_rdt` ≈ 4×10⁻⁷ → cero generación térmica. El CF2 probablemente no se forma ⇒ la corrida terminará con `FilamentosNoFormadosException` o `NoPercolationException` del segundo filamento. **Eso no es un fallo de la corrida: es el resultado** (anotarlo).
- **Decisión:** pico desaparece → el ×5 es condición necesaria del pico (parametrización, no física). Pico persiste → hay otra fuente térmica o de campo: pasar a R5/R6 con prioridad.

## R3 — Temperatura localizada (sin `np.full_like` global)

- **Objetivo:** aislar el efecto de la extensión espacial: el calor del CF1 no debe llegar íntegro a la banda del CF2.
- **Cambio:** `phases_set.py:361`. Sustituir
  ```python
  temperatura = np.full_like(actual_state, temperatura)
  ```
  por:
  ```python
  if diag.T_LOCAL:
      T_matrix = np.full(actual_state.shape, params.init_temp, dtype=float)
      for centro, grosor, creado in zip(CF_centros, sim_ctes.grosor_filamento, CF_creado):
          if creado:  # solo la banda de los filamentos YA creados recibe el calor Joule
              T_matrix[max(0, centro - grosor): centro + grosor + 1, :] = temperatura
      temperatura = T_matrix
  else:
      temperatura = np.full_like(actual_state, temperatura, dtype=float)
  ```
  (Mantiene el ×5: se testea SOLO la localización. `CF_creado` y `grosor_filamento` ya están disponibles en ese scope.)
- **Lanzar:** `DIAG_SEED=42 DIAG_T_LOCAL=1 python -m RRAM exec 1 --stop-at pp_set`
- **Predicción:** banda del CF2 a 300 K → no se dispara; la banda del CF1 sigue a ~570 K → el CF1 puede engordar rápido dentro de su banda (llenado local, no global). `n_sat` ≈ celdas libres de la banda del CF1, no de toda la máscara.
- **Decisión:** si el pico global desaparece pero el CF1 se llena en pocos pasos → confirma dos capas: full_like causaba la extensión, y el nivel térmico (×5) causa el llenado local. Si aún así el CF2 se dispara → algo más calienta o el campo domina (ir a R6).

## R4 — Corregir el bucle del campo eléctrico

- **Objetivo:** cuantificar el aporte real del campo (hoy las filas 40–119 tienen E=0 por el bug) y comprobar si el campo puede nuclear el CF2 legítimamente.
- **Cambio:** `phases_set.py:174` (y el equivalente en `SP_set`, línea ~674):
  ```python
  n_filas = actual_state.shape[0] if diag.FIX_EFIELD else params.x_size
  for i in range(0, n_filas):
      E_field_vector[i] = ElectricField.GapElectricField(...)
  ```
- **Lanzar dos variantes:**
  - R4a: `DIAG_SEED=42 DIAG_FIX_EFIELD=1` (fix solo) → ¿cuánto se adelanta/agrava el pico?
  - R4b: `DIAG_SEED=42 DIAG_FIX_EFIELD=1 DIAG_NO_X5=1 DIAG_T_LOCAL=1` (fix + térmica corregida) → **la corrida más importante del plan**: ¿se forma el CF2 por campo a mayor V, y de forma gradual?
- **Predicción R4a:** pre-percolación cambia poco (las filas activas de la banda baja ya tenían campo); post-percolación el pico se intensifica.
- **Decisión con R4b:** CF2 se forma gradual a V ∈ (0.94, 1.1) → el modelo es autosuficiente: adoptar fix + quitar ×5 + T localizada como paquete. CF2 no se forma antes de V=1.1 → hueco de diseño de nucleación (ver matriz de decisión, fila 4): habrá que decidir física adicional, no parches.

## R5 — Temperatura congelada solo en la generación

- **Objetivo:** prueba de driver puro: si con T=300 K en la generación no hay pico, el driver es 100% térmico.
- **Cambio:** en `phases_set.py`, en la llamada a `update_state_generation` (línea ~398):
  ```python
  temperatura_gen = params.init_temp if diag.T_FROZEN_GEN else temperatura
  ```
  y pasar `temperatura_gen`. **Nada más cambia** (corriente, FVM, etc. siguen igual).
- **Lanzar:** `DIAG_SEED=42 DIAG_T_FROZEN_GEN=1 python -m RRAM exec 1 --stop-at pp_set`
- **Predicción:** a 300 K y con el campo actual (bug incluido), `max_rdt < 10⁻³` siempre → sin pico. Probablemente ni siquiera percola el CF1 en el tiempo simulado (la generación pre-percolación también depende de T Joule): si ocurre, anotarlo — también es información (mide cuánto de la percolación inicial es térmica).
- **Decisión:** sin pico → driver térmico confirmado al 100%. Pico residual → el campo contribuye: R6 pasa de opcional a obligatoria.

## R6 — Campo simple en la generación (opcional; obligatoria si R5 deja pico residual)

- **Objetivo:** acotar el aporte del modelo de gap (E = V/(L−gap)) frente al campo plano V/L.
- **Cambio:** en `phases_set.py`, en el bucle del campo:
  ```python
  if diag.SIMPLE_FIELD:
      E_field_vector[:] = ElectricField.SimpleElectricField(voltage, params.device_size_x)
  else:
      # bucle GapElectricField actual
  ```
- **Lanzar:** `DIAG_SEED=42 DIAG_SIMPLE_FIELD=1 python -m RRAM exec 1 --stop-at pp_set`
- **Predicción:** a V≈0.94, γaE pasa de hasta ~2 eV (filas casi llenas) a 0.24 eV fijo. Si el pico se mantiene igual → el campo era secundario. La percolación inicial se retrasará (el gap acelera el crecimiento de puntas): comparar k_p con R1.
- **Decisión:** solo interpretable junto a R5. R5 sin pico + R6 sin cambios → térmico puro. R5 con residuo + R6 lo elimina → el gap participa; revisar si γ=10 está justificado.

## R7 — Probabilidad exponencial exacta

- **Objetivo:** corrección matemática del muestreo; medir su efecto real (esperado: modesto por sí solo).
- **Cambio:** `Generation.py:65-69`:
  ```python
  rdt = time_stp * vibration_frequency * np.exp(-exponente)   # r*dt, sin acotar
  if diag.PROB_EXP:
      prob_matrix = 1.0 - np.exp(-rdt)     # exacta: satura a 1 por sí sola
  else:
      prob_matrix = np.minimum(rdt, 1.0)   # comportamiento actual
  return prob_matrix, rdt                   # devolver también el crudo para diag
  ```
  (Ojo: los factores `factor_vecinos`/`factor_sin_vecinos` y la máscara multiplican DESPUÉS
  en `get_generation_probabilities_matrix`; con la forma exponencial lo correcto es que esos
  factores multipliquen a `rdt` ANTES de exponenciar — mover la multiplicación:
  `P = 1 - exp(-factor * rdt)`. Con factores ≈1 la diferencia es pequeña, pero hacerlo bien.)
- **Lanzar:** `DIAG_SEED=42 DIAG_PROB_EXP=1 python -m RRAM exec 1 --stop-at pp_set`
- **Predicción:** el pico NO desaparece (a 570 K: P pasa de 0.68 a 0.49; a 650 K ambas ≈1). Ligero retraso del llenado. Esto **demuestra** que la fórmula no era la causa raíz — evita que alguien lo cierre con este parche y dé el problema por resuelto.
- **Decisión:** incorporar siempre (es lo correcto); no cierra el diagnóstico.

## R8 — Sub-stepping adaptativo

- **Objetivo:** resolver la transición EN el tiempo: llenado ordenado con realimentación, en vez de simultáneo.
- **Cambio:** en `phases_set.py`, sustituir la llamada única a `update_state_generation` por:
  ```python
  if diag.SUBSTEP:
      RDT_MAX = 0.1
      dt_restante = params.paso_temporal
      while dt_restante > 0:
          # 1) tasa con el estado ACTUAL (E depende de las filas; recalcular es barato)
          for i in range(actual_state.shape[0]):
              E_field_vector[i] = ElectricField.GapElectricField(voltage, i, actual_state, ...)
          rdt_max = max_rdt_estimado(actual_state, E_field_vector, temperatura, ...)  # r*dt_restante
          n_sub = max(1, int(np.ceil(rdt_max / RDT_MAX)))
          dt_sub = dt_restante / n_sub
          # 2) un sub-paso con dt_sub
          actual_state, prob = update_state_generation(..., paso_temporal=dt_sub, ...)
          dt_restante -= dt_sub
          # 3) T: congelada dentro del macro-paso (limitación documentada);
          #    opcional v2: recalcular I (R_total del grafo) y T escalar cada m sub-pasos
      diag_log(n_sub_total=...)  # cuántos sub-pasos consumió el paso
  ```
  Requiere que `update_state_generation` acepte `paso_temporal` como argumento (hoy lo
  lee de `params`); cambio de firma trivial.
- **Lanzar:** `DIAG_SEED=42 DIAG_PROB_EXP=1 DIAG_SUBSTEP=1 python -m RRAM exec 1 --stop-at pp_set`
  (con la térmica del baseline, para ver el pico "resuelto"; luego repetir sobre el paquete corregido de R4b).
- **Registrar:** nº de sub-pasos por paso (serie), orden de llenado (guardar estados intermedios densos alrededor de k_p), evolución de E dentro del macro-paso.
- **Predicción:** el llenado deja de ser simultáneo; con T congelada intra-paso la banda caliente seguirá llenándose dentro de 1–2 macro-pasos (la T no baja sola). **El sub-stepping resuelve el orden, no frena la física** — el freno es R9.
- **Decisión:** si con tasas realistas (tras R2/R3) los sub-pasos rara vez superan n=1, el mecanismo queda como guardarraíl barato permanente. Si se activa a menudo, plantear la v2 (I,T re-evaluadas intra-paso).

## R9 — Compliance de corriente

- **Objetivo:** freno físico: reproducir la autolimitación del SET real a T de filamento legítimas (600–700 K).
- **Cambio:** en `phases_set.py`, justo tras calcular la corriente óhmica (línea ~264):
  ```python
  if diag.ICC > 0 and current > diag.ICC:
      current = diag.ICC
      voltage_dev = current * R_total          # V efectivo sobre el dispositivo
  else:
      voltage_dev = voltage
  ```
  y usar `voltage_dev` (no `voltage`) en: `Temperature_Joule`, `calculate_heat_source`
  (vía I ya limitada), y el bucle de `E_field_vector` de ese paso. La rampa externa
  sigue con `voltage`; solo el dispositivo ve `voltage_dev`.
- **Valor de I_cc:** tomar el del setup experimental de las medidas de referencia
  (típico 0.1–1 mA en las curvas I–V de Datos_Experimentales; confirmar con el dato real).
- **Lanzar:** `DIAG_SEED=42 DIAG_PROB_EXP=1 DIAG_SUBSTEP=1 DIAG_ICC=1e-3 python -m RRAM exec 1 --stop-at pp_set` sobre el paquete corregido (NO_X5 + T_LOCAL + FIX_EFIELD).
- **Predicción:** al percolar, I toca I_cc → V_dev cae → T baja → `max_rdt` cae por debajo de 0.1 → el engorde del filamento se autolimita. La I–V simulada debe mostrar el codo de compliance como la experimental.
- **Decisión:** si estabiliza el post-SET en el rango de R experimental → adoptar. Comparar R_final con las medidas.

---

## Orden de ejecución recomendado y esfuerzo

| Orden | Corrida | Flags | Duración | Qué desbloquea |
|-------|---------|-------|----------|----------------|
| 1 | R1 | (ninguno) | ~min | Referencia; valida instrumentación |
| 2 | R2 | NO_X5 | ~min | Veredicto sobre el ×5 |
| 3 | R5 | T_FROZEN_GEN | ~min | Veredicto térmico puro |
| 4 | R3 | T_LOCAL | ~min | Veredicto sobre full_like |
| 5 | R4a/R4b | FIX_EFIELD (+NO_X5+T_LOCAL) | ~min ×2 | ¿El campo nuclea el CF2? — la pregunta de diseño |
| 6 | R6 | SIMPLE_FIELD | ~min | Solo si R5 dejó residuo |
| 7 | R7 | PROB_EXP | ~min | Corrección matemática (se queda) |
| 8 | R8 | +SUBSTEP | ~min | Integrador resuelto (se queda como guardarraíl) |
| 9 | R9 | +ICC | ~min | Freno físico; validación I–V |

Total: ~10 corridas de minutos + 1 tarde de análisis. Las corridas 1–4 dan el
**veredicto** (bug/modelado/física); las 5–9 dan la **solución** y su validación.

**Al cerrar:** eliminar `RRAM/diag.py` y los `if diag.*`, dejando incorporado de forma
permanente lo decidido (previsiblemente: fix del bucle E, probabilidad exponencial,
sub-stepping como guardarraíl, T por filamento/FVM en vez de full_like+×5, y compliance).
