# ROL
Actúa como un Desarrollador Senior Full-Stack en Python y Especialista en Integridad de Activos (Asset Integrity Management).

# OBJETIVO
Desarrollar una aplicación para calcular la **Evolución de la Probabilidad de Falla (POF)** en tuberías de transporte. La app debe integrar datos estáticos, dinámicos y ambientales para calcular la POF a lo largo del tiempo en cada junta de soldadura.

# 1. INPUTS (ARCHIVOS CSV)
La aplicación debe permitir la carga de 10 archivos CSV.

**A. Datos de Tubería (Base):**
1.  `juntas_soldadura.csv`: `distancia_m`, `lat`, `lon`, `diametro`, `espesor`, `SMYS`, `tasa_corrosion_mm_ano`.
2.  `anomalias.csv`: `distancia_m`, `profundidad_mm`, `ancho`, `largo`, `tipo_defecto`.
3.  `presion.csv`: `distancia_m`, `presion`.

**B. Datos Ambientales (Variables Explicativas para ML):**
4.  `resistividad.csv`: `distancia_m`, `resistividad_suelo_ohm_cm`.
5.  `tipo_suelo.csv`: `distancia_m`, `tipo_suelo`.
6.  `potencial.csv`: `distancia_m`, `cp_potencial_on_mv`.
7.  `interferencia.csv`: `distancia_m`, `interferencia_dc`.
8.  `tipo_recubrimiento.csv`: `distancia_m`, `tipo_recubrimiento`.
9.  `edad_recubrimiento.csv`: `distancia_m`, `edad_recubrimiento_anos`.

**C. Inspecciones directas ((Datos de Campo / Verdad Terreno)):**
10. `inspecciones_directas.csv`:  `distancia_m`, `profundidad_campo_mm`, `ancho_campo_mm`, `largo_campo_mm`, `tipo_defecto_campo`  

# 2. INTERFAZ DE USUARIO (FRONTEND)
Añade una sección de **"Configuración de Evaluación Temporal"**:
1.  **Input Date 1 (Fecha Base):** "Fecha de Inspección ILI" (DD/MM/YYYY).
2.  **Input Date 2 (Fecha Objetivo):** "Fecha Final de Proyección" (DD/MM/YYYY).
3.  **Tolerancias de dimensionamiento de anomalias** Tabla para ingresar tolerancias (desviacion estandar directa) de las medidas de profundidad segun el `tipo_defecto`, asi: 
    -   **Tipo de defecto** | **Tolerancia**
    -   **General** | **10%** (default)
    -   **Pitting** | **10%** (default)
    -   **Axial grooving** | **15%** (default)
    -   **Circumferential grooving** | **15%** (default)
    -   **Pinhole** | **10%** (default)
    -   **Axial slotting** | **15%** (default)
    -   **Circumferential slotting** | **10%** (default)
4. **Input umbral_deteccion_ILI:** umbral deteccion de la corrida ILI (default 10%)
3.  **Botón de Acción:** "Calcular Evolución de POF".

# 3. LÓGICA DE PROCESAMIENTO (BACKEND)

## Fase A: Consolidación y ML (Estado Inicial T=0)
1.  Generar el `DataFrame Maestro` uniendo todos los CSVs a nivel de Defecto Individual (permitiendo múltiples filas por junta con la misma data ambiental y la data de campo).
2.  Entrena un modelo (Random Forest Regressor) para estimar la profundidad de defectos donde el ILI no reportó nada.
    -   **Set de Entrenamiento:** juntas donde `anomalias.profundidad > 0`.
    -   **Features (X):** -   `resistividad_suelo_ohm_cm`
        -   `cp_potencial_on_mv`
        -   `interferencia_dc`
        -   `edad_recubrimiento_anos`
        -   `tipo_suelo` (Encoded)
        -   `tipo_recubrimiento` (Encoded)
    -   **Target (Y):** `profundidad_mm` (del ILI).
    -   **Predicción:** Aplica el modelo a las juntas donde `anomalias.profundidad = 0` o no existen defectos para obtener una  `profundidad_estimada_ml`.
    -   **Aplicar Restricción Física (Censura):**
    -   El hecho de que el ILI no reportara nada es información valiosa. Significa que, si existe un defecto, es muy probable que sea menor al `umbral_deteccion_ILI` (ej. 10%).
    -   *Conflicto:* Si `profundidad_estimada_ml` > `umbral_deteccion_ILI`, asumimos que el ILI "falló" (falso negativo) O que el modelo ML sobreestima.
    -   *Regla de Asignación:*
        Para la simulación, define la variable aleatoria de profundidad como una **Truncada Normal**:
        -   **Límite Inferior:** 0%.
        -   **Límite Superior:** `umbral_deteccion_ILI` (ej. 10%).
        -   **Sesgo de la curva:**
            -   Si `profundidad_estimada_ml` es ALTA -> La curva se inclina hacia el límite superior (ej. media = 8%).
            -   Si `profundidad_estimada_ml` es BAJA -> La curva se inclina hacia 0% (ej. media = 2%).

## Fase B: Ajuste de Modelo con Datos de Campo

Una vez entrenado el modelo Random Forest:
1.  **Cruce de Validación:** Identifica las juntas donde existen datos en `inspecciones_directas.csv` y no cuentan con datos de ILI.
2.  **Comparativa:** Para esas juntas, extrae la `prediccion_ml` y compárala contra `profundidad_real_campo`.
3.  **Cálculo de Métricas:** Calcula la desviacion estandar del modelo como la desviacion estandar de la diferencia entre la prediccion y el dato real. en caso de que no exista datos de campo, se usa la desviacion estandar del 10%

## Fase C: Bucle Temporal (Time-Dependent POF)
Debes iterar año por año desde `Fecha_Base` (ILI) hasta `Fecha_Objetivo` (Objetivo).
Para cada AÑO ($t$) y para cada JUNTA ($i$):

1. **Jerarquía de Datos**
    -   **Nivel 1 (Máxima Certeza):** Si hay dato en `inspecciones_directas.csv` para esa junta -> USAR ESTE VALOR (ignorando ILI y ML). Usar desviacion estandar de 1%, ya que es una medición manual precisa.
    -   **Nivel 2 (Certeza Media):** Si no hay inspección directa, pero hay dato ILI -> USAR DATO ILI. Usar desviacion estandar especificada en la tabla de tolerancias.
    -   **Nivel 3 (Inferencia):** Si no hay ni ILI ni Directa -> USAR PREDICCIÓN ML. Usar desviacion calculada en la fase B (desviacion estandar de la diferencia entre la prediccion y el dato real).

2.  **Modelo de Crecimiento (Growth Model):**
    $$Profundidad_{i,t} = Profundidad_{i,t-1} + TasaCorrosion_{i}$$
    -   *Conversión:* Asegúrate de sumar mm con mm.
    -   *Límite:* La profundidad máxima es el espesor de la tubería (Fuga).
    $$longitud_{i,t} = longitud_{i,t-1} + TasaCorrosion_{i}$$

3.  **Cálculo de POF (Probabilidad de Excedencia):**
    -   El POF se define como la Probabilidad de Excedencia de la profundidad del defecto sobre un límite crítico.
    -   **Distribución de Profundidad:** Normal, con Valor Medio y desviacion estandar segun el nivel de certeza.
    -   **Valor Límite:** El mínimo entre:
        -   100% del espesor de pared ($t_{wall}$).
        -   Profundidad teórica donde $P_{fail}$ (ASME B31G Modificado) = $P_{operacion}$.la profundidad teorica se estima como ((Presion-FlowStress)*espesor)/(((Presion/M)-FlowStress)*0.85)
    -   **Cálculo:** Probabilidad de que la profundidad modelada ($d >$ Valor Límite).
    -   **Acumulación de POF:** la POF de la junta se calcula como $1 - \Pi(1 - POF_i)$

4.  **Almacenamiento:** Guardar la POF resultante para ese año en una estructura de datos `Resultados[Año][Junta]`.

# 4. VISUALIZACIÓN DE RESULTADOS
1.  **Gráfico de Evolución (Line Chart):**
    -   Permite seleccionar una Junta crítica específica.
    -   Eje X: Años.
    -   Eje Y: POF.
    -   Línea de corte: Dibujar una línea horizontal en POF = $10^{-3}$ (Umbral de riesgo).
2.  **Matriz de Calor (Heatmap) Espacio-Temporal:**
    -   Eje X: distancia_m (Km de la tubería).
    -   Eje Y: Años (Desde ILI hasta Futuro).
    -   Color: Valor de POF (Verde a Rojo).
    -   *Esto permite ver cómo "crece" la mancha de riesgo a lo largo de la tubería con el tiempo.*
3.  **Gráfico de Paridad (Unity Plot):**
    -   Scatter plot interactivo.
    -   Eje X: Profundidad Medida en Campo.
    -   Eje Y: Profundidad Predicha por ML.
    -   Línea diagonal (1:1) de referencia.
4. **Tabla de resultados**
    -   Mostrar una tabla con los datos de entrada y resultados de POF para cada año.
5. **Importancia de Variables (Feature Importance) por Junta:**
    -   Mostrar un grafico de barras con la importancia de las variables para la prediccion de profundidad de defecto seleccionado en la tabla de resultados.

# 5. EXPORTACIÓN DE RESULTADOS
1.  **Exportación a CSV:**
    -   Boton para exportar a CSV de la tabla de resultados.

# 6. ARQUITECTURA
Arquitectura modular con las siguientes capas:
-   `data_loader.py`: Carga y validación de datos desde CSV.
-   `ml_model.py`: Entrenamiento y predicción del modelo de Machine Learning.
-   `growth_model.py`: Modelo de crecimiento de defectos.
-   `pof_calculator.py`: Cálculo de POF.
-   `visualization.py`: Visualización de resultados utilizando streamlit
-   `main.py`: Orquestación de la aplicación.

# RESTRICCIONES TÉCNICAS
-   Optimización: Usa `numpy` arrays para vectorizar el cálculo de todos los segmentos simultáneamente por año.
-   Manejo de Fechas: Usa la librería `datetime` para calcular la diferencia de años (float) entre la fecha ILI y la fecha de evaluación.