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
            tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Heatmap", "Resultados Detallados", "Diagnóstico ML"])
            
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
            with tab2:
                st.subheader("Mapa de calor de riesgo en el espacio y el tiempo")
                heatmap_data = pof_results.pivot_table(index='Year', columns='Distance', values='POF', aggfunc='max')
                
                fig_heat = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index,
                    colorscale='RdYlGn_r', zmin=0, zmax=1e-3
                ))
                fig_heat.update_layout(title='POF Heatmap (Distance vs Time)', xaxis_title='Distance (m)', yaxis_title='Year')
                st.plotly_chart(fig_heat, use_container_width=True)
            
            # Section 3: Detailed Data
            with tab3:
                st.subheader("Resultados Detallados")
                st.write("**Master Data with POF Results**")
                
                drop_cols = [c for c in ['distancia_inicio_m', 'distancia_fin_m', 'distancia_inicio_m_resistividad', 'distancia_fin_m_resistividad', 'distancia_inicio_m_tipo_suelo', 'distancia_fin_m_tipo_suelo', 'distancia_inicio_m_potencial', 'distancia_fin_m_potencial', 'distancia_inicio_m_interferencia', 'distancia_fin_m_interferencia', 'distancia_inicio_m_tipo_recubrimiento', 'distancia_fin_m_tipo_recubrimiento', 'distancia_inicio_m_presion', 'distancia_fin_m_presion'] if c in master_df.columns]
                
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
