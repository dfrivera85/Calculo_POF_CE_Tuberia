import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Import local modules
import data_loader
import pof_calculator
import ml_model
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
    st.title("🛡️ Pipeline Integrity: POF Evolution")
    st.markdown("### Asset Integrity Management System")
    
    # --- SIDEBAR: CONFIGURATION ---
    with st.sidebar:
        st.header("1. Data Input")
        st.info("Upload the required CSV files. Filenames must match the schema.")
        
        uploaded_files = st.file_uploader(
            "Upload 10 CSV Files", 
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
            
        st.divider()
        
        st.header("2. Simulation Config")
        
        # Dates
        col1, col2 = st.columns(2)
        with col1:
            ili_date = st.date_input("ILI Date (T=0)", value=datetime(2023, 1, 1))
        with col2:
            target_date = st.date_input("Target Date", value=datetime(2028, 1, 1))
            
        # Tolerances
        st.subheader("Tolerances & Thresholds")
        detection_threshold = st.slider("ILI Detection Threshold (%)", 0, 20, 10, format="%d%%") / 100.0
        
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
            use_container_width=True
        )
        
        run_btn = st.button("RUN ANALYSIS 🚀", type="primary", disabled=(len(missing) > 0))

    # --- MAIN AREA ---
    # --- MAIN AREA ---
    # Check if results exist in session state
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None

    if run_btn and not missing:
        with st.spinner("Processing Data & Running Simulation..."):
            try:
                # 1. Load Data
                dfs = data_loader.load_data_from_dict(file_dict)
                st.toast("Data Loaded Successfully", icon="✅")
                
                # 2. Run Simulation
                st.info("Running Simulation... This might take a moment.")
                results = orchestration_module.run_simulation(dfs, ili_date, target_date, tolerances_df, detection_threshold)
                st.session_state.simulation_results = results # Persistence
                st.success("Simulation Complete!")
            except Exception as e:
                 st.error(f"An error occurred during execution: {str(e)}")
                 st.exception(e)

    # Display results if they exist (either from this run or previous)
    if st.session_state.simulation_results:
        results = st.session_state.simulation_results
        
        # Unpack Results
        master_df = results['master_df']
        pof_results = results['pof_results']
        
        # TABS
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🗺️ Heatmap", "📋 Data Inspection", "⚙️ ML Diagnostics"])
        
        with tab1:
            st.subheader("POF Evolution (Critical Joints)")
            
            # Filter for top risk joints
            # Get max POF per joint per time
            max_pof = pof_results.groupby('Junta_ID')['POF'].max()
            critical_joints = max_pof[max_pof > 1e-4].index # Show only joints with some risk
            
            if len(critical_joints) > 0:
                plot_df = pof_results[pof_results['Junta_ID'].isin(critical_joints)]
                # Limit to top 20 to avoid clutter if too many
                if len(critical_joints) > 20:
                    top_20 = max_pof.nlargest(20).index
                    plot_df = pof_results[pof_results['Junta_ID'].isin(top_20)]
                    st.warning(f"Showing top 20 critical joints out of {len(critical_joints)} found.")
                
                fig = px.line(
                    plot_df, 
                    x='Year', 
                    y='POF', 
                    color='Junta_ID',
                    markers=True,
                    title="POF Evolution over Time"
                )
                # Add Threshold Line
                fig.add_hline(y=1e-3, line_dash="dash", line_color="red", annotation_text="Limit (1e-3)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No joints exceeded POF 1e-4. Excellent Integrity!")

        with tab2:
            st.subheader("Space-Time Risk Heatmap")
            
            # Prepare Matrix for Heatmap: Rows=Years, Cols=Distance (Segments)
            # We need to pivot
            heatmap_data = pof_results.pivot_table(index='Year', columns='Distance', values='POF', aggfunc='max')
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlGn_r', # Red High Risk, Green Low
                zmin=0,
                zmax=1e-3 # Cap visualization at limit
            ))
            fig_heat.update_layout(
                title='POF Heatmap (Distance vs Time)',
                xaxis_title='Distance (m)',
                yaxis_title='Year'
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
        with tab3:
            st.subheader("Detailed Results")
            
            # Show consolidated Master DataFrame (Input + Output)
            st.write("**Master Data with POF Results**")
            
            # Remove start/end distance columns for visualization as requested
            # We handle likely column names including those with '_m' suffix
            drop_cols = [c for c in ['distancia_inicio_m', 'distancia_fin_m', 'distancia_inicio_m_resistividad', 'distancia_fin_m_resistividad', 'distancia_inicio_m_tipo_suelo', 'distancia_fin_m_tipo_suelo', 'distancia_inicio_m_potencial', 'distancia_fin_m_potencial', 'distancia_inicio_m_interferencia', 'distancia_fin_m_interferencia', 'distancia_inicio_m_tipo_recubrimiento', 'distancia_fin_m_tipo_recubrimiento', 'distancia_inicio_m_presion', 'distancia_fin_m_presion'] if c in master_df.columns]
            
            # Configure scientific notation for POF columns
            pof_cols = [c for c in master_df.columns if 'POF_' in c]
            column_config = {col: st.column_config.NumberColumn(format="%.2e") for col in pof_cols}
            
            st.dataframe(master_df.drop(columns=drop_cols), use_container_width=True, column_config=column_config)
            
            # Export Master DataFrame
            csv = master_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Unified Results CSV",
                csv,
                "master_results_pof.csv",
                "text/csv",
                key='download-master'
            )
            
        with tab4:
            st.subheader("ML Model Diagnostics")
            
            col_ml1, col_ml2 = st.columns(2)
            
            with col_ml1:
                st.write(f"**ML Uncertainty (Std Dev):** {results.get('ml_uncertainty_status', 'N/A')}")
                
                # Parity Plot
                # Filter rows where we have field data (validation set)
                val_mask = (master_df['profundidad_campo_mm'].notna()) & (master_df['profundidad_campo_mm'] > 0)
                val_df = master_df[val_mask]
                
                if not val_df.empty:
                   fig_parity = px.scatter(
                       val_df, 
                       x='profundidad_campo_mm', 
                       y='pred_depth_ml',
                       title="Parity Plot: Field vs ML",
                       labels={'profundidad_campo_mm': 'Field Measured Depth (mm)', 'pred_depth_ml': 'ML Predicted Depth (mm)'}
                   )
                   fig_parity.add_shape(type="line", line=dict(dash='dash'), x0=0, y0=0, x1=max(val_df['profundidad_campo_mm']), y1=max(val_df['profundidad_campo_mm']))
                   st.plotly_chart(fig_parity, use_container_width=True)
                else:
                    st.info("No overlapping Field Data for Parity Plot.")
                    
            with col_ml2:
                # Feature Importance
                if 'feature_importance' in results:
                    st.write("**Feature Importance**")
                    fi_df = results['feature_importance']
                    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h')
                    st.plotly_chart(fig_fi, use_container_width=True)

    elif not run_btn and not st.session_state.simulation_results:
        # Welcome State
        st.write("### Instructions")
        st.write("""
        1. **Upload Data**: Use the sidebar to upload all 10 required CSV files.
        2. **Configure**: Set the ILI inspection date and the target projection date.
        3. **Calibrate**: Adjust measurement tolerances if necessary.
        4. **Run**: Click the 'Run Analysis' button to generate predictions.
        """)
        
        # Example Data Structure (Static help)
        with st.expander("Required File Schema"):
            st.json(data_loader.REQUIRED_COLUMNS)

if __name__ == "__main__":
    main()
