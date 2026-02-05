import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import shap
import matplotlib.pyplot as plt

# Import local modules
import data_loader
import main as orchestration_module  # Orchestration module

# Set Page Config
st.set_page_config(
    page_title="Pipeline POF Assessment",
    page_icon="running_process:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium" look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🛡️ Calculo de POF Estructural por Corrosión")
    
    # --- NAVIGATION ---
    with st.sidebar:
        st.header("Navegación")
        selected_tab = st.radio(
            "Ir a:",
            ["Análisis POF", "Funcionalidad 2", "Funcionalidad 3"]
        )
        st.divider()

    if selected_tab == "Análisis POF":
        st.markdown("### Preparación de Datos")
        
        # --- INPUTS & CONFIG (Moved to Main Area) ---
        col_input1, col_input2 = st.columns([1, 1])
        
        with col_input1:
            st.subheader("1. Cargue de datos")
            st.info("cargue los archivos CSV requeridos. Los nombres de los archivos deben coincidir con el esquema.")
            
            with st.expander("Esquema de archivos requeridos"):
                st.json(data_loader.REQUIRED_COLUMNS)  
            
            with st.expander("Metodología de Cálculo de POF"):
                st.markdown("""
                El cálculo de la Probabilidad de Falla (POF) estructural por corrosión sigue un enfoque probabilístico basado en la **Confiabilidad Estructural**:
                
                **Preparacion y cargue de datos:**
                Los archivos solicitados permiten integrar la información física, operativa y del entorno para evaluar la integridad de la tubería, asi:
                
                *  **Definición del Activo y Estado Actual:**
                    Se requieren datos del activo y del estado actual de la tubería (juntas, anomalias, presiones, etc.), 
                *  **Modelo de Machine Learning (Contexto Ambiental):**    
                    Datos del entorno (resistividad, potencial, tipo de suelo, etc.) para para predecir mediante Machine Learning condiciones corrosivas en zonas donde el ILI no reportó hallazgos o para refinar las tasas de crecimiento de corrosión.
                *  **Validación de Campo:**
                    Datos de inspeccion directa en campo para comparar y calibrar las predicciones del modelo con datos reales medidos en excavaciones.

                **Modelo de Machine Learning (ML) para Profundidades:**
                Para mejorar la estimación en juntas donde la herramienta ILI no reportó anomalías (posibles defectos bajo el umbral), se incorpora un modelo de **Random Forest Regressor**.
                *   **Objetivo:** Predecir la profundidad más probable de defectos latentes basándose en las condiciones del entorno y del activo.
                *   **Variables (Features):** Resistividad del suelo, Potencial ON (CP), Interferencia DC, Tipo de suelo, Tipo de recubrimiento y Edad.
                *   **Salida:** La predicción del ML se utiliza para parametrizar una distribución **Normal Truncada** que simula la profundidad inicial del defecto en la simulación de Monte Carlo.

                **Cálculo de Probabilidad de Falla (POF):**
                El cálculo de POF se realiza de la siguiente manera:
                *  **Determinacion de Profundidades iniciales y proyectadas:**
                    Se determinan las profundidades iniciales y proyectadas de los defectos con el siguiente nivel de prioridad: 1) Datos de inspeccion directa en campo, 2) Datos de ILI, 3) Predicciones del modelo de Machine Learning.
                *  **Modelo estructural:**   
                    Se definen las incertidumbres mediante un análisis de **Probabilidad de Excedencia:**, donde se determina la probabilidad que que la profundidad del defecto simulado a lo largo de las vigencias supere la profundidad crítica definida por el modelo ASME B31G Modificado.
                """)

            uploaded_files = st.file_uploader(
                "cargue los archivos CSV", 
                accept_multiple_files=True,
                type=['csv']
            )
            
            # Convert list of uploaded files to dict {filename: file_obj}
            file_dict = {f.name: f for f in uploaded_files} if uploaded_files else {}
            
            # Check for missing files
            required_files = set(data_loader.REQUIRED_COLUMNS.keys())
            uploaded_filenames = set(file_dict.keys())
            missing = required_files - uploaded_filenames
            
            if missing:
                st.warning(f"Missing {len(missing)} files: \n" + "\n".join([f"- {f}" for f in missing]))
            else:
                st.success("All required files uploaded! ✅")

        with col_input2:
            st.subheader("2. Configuración de la simulación")
            
            # Dates
            c1, c2 = st.columns(2)
            with c1:
                ili_date = st.date_input("Fecha de Corrida ILI", value=datetime(2023, 1, 1))
            with c2:
                target_date = st.date_input("Fecha de Proyección", value=datetime(2028, 1, 1))
                
            # Tolerances
            st.markdown("**Tolerancias & umbrales**")
            detection_threshold = st.slider("Umbral de detección de ILI (%)", 0, 20, 10, format="%d%%") / 100.0
            
            # Default Tolerances Table
            default_tolerances = pd.DataFrame({
                "Defect Type": ["General", "Pitting", "Axial Grooving", "Circumferential Grooving", "Pinhole", "Axial Slotting", "Circumferential Slotting"],
                "Tolerance": [0.10, 0.10, 0.15, 0.15, 0.10, 0.15, 0.10]
            })
            
            tolerances_df = st.data_editor(
                default_tolerances,
                column_config={
                    "Tolerance": st.column_config.NumberColumn(
                        "Std Dev (%)",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        format="%.2f"
                    )
                },
                hide_index=True,
                use_container_width=True,
                height=250
            )

        st.divider()
        run_btn = st.button("EJECUTAR ANALISIS 🚀", type="primary", disabled=(len(missing) > 0))

        # --- PROCESS LOGIC ---
        # Check if results exist in session state
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = None

        if run_btn and not missing:
            with st.spinner("Procesando datos & ejecutando simulación..."):
                try:
                    # 1. Load Data
                    dfs = data_loader.load_data_from_dict(file_dict)
                    st.toast("Datos cargados exitosamente", icon="✅")
                    
                    # 2. Run Simulation
                    st.info("Ejecutando simulación... Esto puede tomar un momento.")
                    results = orchestration_module.run_simulation(dfs, ili_date, target_date, tolerances_df, detection_threshold)
                    st.session_state.simulation_results = results # Persistence
                    st.success("Simulación completada!")
                except Exception as e:
                     st.error(f"Error durante la ejecución: {str(e)}")
                     st.exception(e)

        # --- RESULTS DISPLAY (Sequential) ---
        if st.session_state.simulation_results:
            results = st.session_state.simulation_results
            master_df = results['master_df']
            pof_results = results['pof_results']

            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Grafica Resultados POF", "🗺️ Mapa Resultados POF", "📊 Tabla de Resultados", "🤖 Diagnóstico ML"])
            
            # Section 1: Dashboard
            with tab1:
                st.subheader("Perfil de POF a lo largo de la tubería")
                
                # 1. Year Selection
                years = sorted(pof_results['Year'].unique())
                if years:
                    selected_year = st.selectbox("Seleccione Año", options=years, index=len(years)-1)
                    
                    # 2. Filter & Prepare Data
                    # Filter for selected year
                    plot_df = pof_results[pof_results['Year'] == selected_year].copy()
                    
                    # Merge with master_df to get metadata for hover
                    # pof_results['Junta_ID'] corresponds to master_df index
                    # We merge left to keep plot structure
                    merged_df = plot_df.merge(
                        master_df[['profundidad_campo_mm', 'profundidad_mm', 'pred_depth_ml', 'tasa_corrosion_mm_ano']], 
                        left_on='Junta_ID', 
                        right_index=True, 
                        how='left'
                    )
                    
                    # Rename columns for cleaner hover labels if needed, or use customdata
                    merged_df = merged_df.rename(columns={
                        'profundidad_campo_mm': 'Prof. Directa (mm)',
                        'profundidad_mm': 'Prof. ILI (mm)',
                        'pred_depth_ml': 'Prof. ML (mm)',
                        'tasa_corrosion_mm_ano': 'Tasa Corr. (mm/año)'
                    })
                    
                    # Sort by Distance to ensure proper line connection
                    merged_df = merged_df.sort_values('Distance')

                    # 3. Create Scatter Plot
                    fig = px.scatter(
                        merged_df, 
                        x='Distance', 
                        y='POF',
                        log_y=True,
                        title=f"Perfil de POF (Año {selected_year})",
                        labels={'Distance': 'Distancia (m)', 'POF': 'Probabilidad de Falla'},
                        hover_data={
                            'Distance': True,
                            'POF': ':.2e',
                            'Junta_ID': True,
                            'Prof. Directa (mm)': ':.2f',
                            'Prof. ILI (mm)': ':.2f',
                            'Prof. ML (mm)': ':.2f',
                            'Tasa Corr. (mm/año)': ':.4f'
                        }
                    )
                    
                    # Connect points with a thin line
                    fig.update_traces(mode='lines+markers', line=dict(width=1))
                    
                    # Update Layout
                    fig.update_layout(
                        yaxis=dict(
                            range=[-6, 0], # log scale 1e-6 to 1 (10^-6 to 10^0) -> log10 ranges -6 to 0
                            tickformat=".0e"
                        ), 
                        hovermode="closest"
                    )
                    
                    # Add limit line
                    fig.add_hline(y=1e-3, line_dash="dash", line_color="red", annotation_text="Límite (1e-3)")
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No results available.")

            # Section 2: Heatmap
            # Section 2: Geographic Map
            with tab2:
                st.subheader("Visualización Geográfica de Riesgo (POF)")
                
                # 1. Year Selection for Map
                years_map = sorted(pof_results['Year'].unique())
                if years_map:
                    selected_year_map = st.selectbox("Seleccione Año para el Mapa", options=years_map, index=len(years_map)-1, key='year_select_map')
                    
                    # 2. Filter & Prepare Data
                    map_df_pof = pof_results[pof_results['Year'] == selected_year_map].copy()
                    
                    # Merge with master_df to get Lat/Lon and details
                    # We need 'latitud' and 'longitud' from master_df
                    cols_to_merge = ['latitud', 'longitud', 'profundidad_campo_mm', 'profundidad_mm', 'pred_depth_ml', 'tasa_corrosion_mm_ano', 'distancia_inicio_m']
                    # Ensure columns exist before merging to avoid errors
                    actual_cols = [c for c in cols_to_merge if c in master_df.columns]
                    
                    if 'latitud' not in actual_cols or 'longitud' not in actual_cols:
                        st.error("No coordinates (latitud/longitud) found in the data. Cannot display map.")
                    else:
                        # Use master_df as the base to include all joints
                        master_base = master_df[actual_cols].copy()
                        
                        # Use distancia_inicio_m as Distance if available
                        if 'distancia_inicio_m' in master_base.columns:
                            master_base = master_base.rename(columns={'distancia_inicio_m': 'Distance'})
                        
                        # Merge with POF results (left join to keep all master points)
                        # map_df_pof has 'Junta_ID' and 'POF'
                        map_merged = master_base.merge(
                            map_df_pof[['Junta_ID', 'POF']],
                            left_index=True, # master_df index is the key
                            right_on='Junta_ID',
                            how='left'
                        )
                        
                        # Fill POF with 0 for joints without anomalies
                        map_merged['POF'] = map_merged['POF'].fillna(0)

                        # Filter out invalid coordinates
                        map_merged = map_merged.dropna(subset=['latitud', 'longitud'])
                        
                        if map_merged.empty:
                            st.warning(f"No valid coordinate data found for year {selected_year_map}.")

                        else:
                            # Sort by POF to ensure high POF points are plotted on top
                            map_merged = map_merged.sort_values(by='POF', ascending=True)

                            # 3. Create Map
                            # Normalize POF for sizing or just use color
                            # We want a trace colored by POF.
                            
                            fig_map = go.Figure(go.Scattermapbox(
                                lat=map_merged['latitud'],
                                lon=map_merged['longitud'],
                                mode='markers',
                                marker=go.scattermapbox.Marker(
                                    size=8,
                                    color=map_merged['POF'],
                                    colorscale='RdYlBu_r', # Red for high POF, Blue for low (better contrast with satellite)
                                    cmin=0,
                                    cmax=1e-3, # Adjust max based on typical risk threshold or data max
                                    colorbar=dict(title="POF", tickformat=".2e"),
                                    opacity=0.8
                                ),
                                text=map_merged.apply(lambda row: f"Dist: {row['Distance']:.1f}m<br>POF: {row['POF']:.2e}", axis=1),
                                hovertemplate=(
                                    "<b>Distancia:</b> %{customdata[0]:.2f} m<br>" +
                                    "<b>POF:</b> %{marker.color:.2e}<br>" +
                                    "<b>Prof. Campo:</b> %{customdata[1]:.2f} mm<br>" +
                                    "<b>Prof. ILI:</b> %{customdata[2]:.2f} mm<br>" +
                                    "<b>Prof. ML:</b> %{customdata[3]:.2f} mm<br>" +
                                    "<b>Tasa Corr.:</b> %{customdata[4]:.4f} mm/año<br>" +
                                    "<extra></extra>"
                                ),
                                customdata=map_merged[[
                                    'Distance', 
                                    'profundidad_campo_mm', 
                                    'profundidad_mm', 
                                    'pred_depth_ml', 
                                    'tasa_corrosion_mm_ano'
                                ]].fillna(0).values # Fill NaNs for display stability
                            ))

                            # Default center
                            center_lat = map_merged['latitud'].mean()
                            center_lon = map_merged['longitud'].mean()

                            fig_map.update_layout(
                                mapbox=dict(
                                    style="white-bg", # Required for custom layers
                                    center=dict(lat=center_lat, lon=center_lon),
                                    zoom=12,
                                    layers=[
                                        {
                                            "below": 'traces',
                                            "sourcetype": "raster",
                                            "sourceattribution": "Google Maps",
                                            "source": [
                                                "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
                                            ]
                                        }
                                    ]
                                ),
                                margin={"r":0,"t":0,"l":0,"b":0},
                                height=600
                            )
                            
                            st.plotly_chart(fig_map, use_container_width=True, config={'scrollZoom': True})
                else:
                    st.warning("No results available.")
            
            # Section 3: Detailed Data
            with tab3:
                st.subheader("Resultados Detallados")
                st.write("**Master Data with POF Results**")
                
                drop_cols = [c for c in ['distancia_inicio_m', 'distancia_fin_m', 'distancia_inicio_m_resistividad', 'distancia_fin_m_resistividad', 'distancia_inicio_m_tipo_suelo', 'distancia_fin_m_tipo_suelo', 'distancia_inicio_m_potencial', 'distancia_fin_m_potencial', 'distancia_inicio_m_interferencia', 'distancia_fin_m_interferencia', 'distancia_inicio_m_tipo_recubrimiento', 'distancia_fin_m_tipo_recubrimiento', 'distancia_inicio_m_presion', 'distancia_fin_m_presion', 'latitud', 'longitud'] if c in master_df.columns]
                
                pof_cols = [c for c in master_df.columns if 'POF_' in c]
                column_config = {col: st.column_config.NumberColumn(format="%.2e") for col in pof_cols}
                
                st.dataframe(master_df.drop(columns=drop_cols), use_container_width=True, column_config=column_config)
                
                csv = master_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Unified Results CSV", csv, "master_results_pof.csv", "text/csv", key='download-master')

            # Section 4: ML Diagnostics
            with tab4:
                st.subheader("Diagnóstico del Modelo ML")
                shap_values = results.get('shap_values')
                
                col_ml1, col_ml2 = st.columns(2)
                with col_ml1:
                    st.write(f"**Uncertainty (Std Dev):** {results.get('ml_uncertainty_status', 'N/A')}")
                    if shap_values is not None:
                        st.markdown("#### Global Interpretability (Beeswarm)")
                        try:
                            fig_beeswarm, ax = plt.subplots()
                            shap.plots.beeswarm(shap_values, show=False)
                            st.pyplot(fig_beeswarm)
                            plt.close(fig_beeswarm)
                        except Exception as e:
                            st.error(f"Error plotting Beeswarm: {e}")
                    else:
                        st.info("SHAP values not available.")
                
                with col_ml2:
                    if 'feature_importance' in results:
                        st.markdown("#### Importancia de las características")
                        fi_df = results['feature_importance']
                        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h')
                        st.plotly_chart(fig_fi, use_container_width=True)

        elif not run_btn and not st.session_state.simulation_results:
             # Instructions when waiting for input
             st.info("Por favor, sube los datos y haz clic en 'Ejecutar análisis' para ver los resultados.")
             

    elif selected_tab == "Funcionalidad 2":
        st.header("Funcionalidad 2")
        st.info("Funcionalidad pendiente de definición.")

    elif selected_tab == "Funcionalidad 3":
        st.header("Funcionalidad 3")
        st.info("Funcionalidad pendiente de definición.")

if __name__ == "__main__":
    main()
