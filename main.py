import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Local imports
import data_loader
from ml_model import DefectDepthEstimator
import growth_model
import pof_calculator

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_simulation(dfs, ili_date, target_date, tolerances_df, detection_threshold=0.10):
    """
    Orchestrates the full POF simulation pipeline.
    
    Args:
        dfs (dict): Dictionary of loaded DataFrames.
        ili_date (datetime): Base date of ILI inspection (T=0).
        target_date (datetime): Target date for projection.
        tolerances_df (pd.DataFrame): DataFrame mapping Defect Type to Std Dev (Tolerance).
        detection_threshold (float): Detection threshold (fraction, e.g. 0.10).
        
    Returns:
        dict: Simulation results containing:
              - 'master_df': The enriched master dataframe at T=0.
              - 'pof_results': DataFrame with POF evolution (Year, Joint, POF).
              - 'model_metrics': Dictionary of ML metrics.
              - 'feature_importance': Feature importance from ML model.
    """
    results = {}
    
    # --- PHASE A: CONSOLIDATION & ML ---
    logging.info("Starting Phase A: Data Consolidation")
    master_df = data_loader.create_master_dataframe(dfs)
    
    # Initialize ML Model
    estimator = DefectDepthEstimator()
    
    # Train Model (on defects found by ILI)
    try:
        training_df = master_df[(master_df['profundidad_mm'].notna()) & (master_df['profundidad_mm'] > 0)]
        estimator.train(training_df)
        
        # Calculate Model Uncertainty (Validation on Field Data)
        ml_uncertainty_mm = estimator.calculate_uncertainty(master_df) * master_df['espesor']
                    
        logging.info(f"ML Uncertainty (Std Dev): {ml_uncertainty_mm}")
        
        # Predict for ALL rows (to fill gaps where ILI = 0)
        # We only really need it where depth is 0/NaN, but let's predict all for Parity Plot
        raw_predictions = estimator.predict(master_df)
        
        # Apply Physical Restrictions (Censoring)
        # upper = detection_threshold * thickness
        # We iterate to apply the specific threshold per joint
        thicknesses = master_df['espesor'].fillna(0).values
        adjusted_preds = []
        
        for pred, thick in zip(raw_predictions, thicknesses):
            limit_mm = detection_threshold * thick
            # Get the adjusted mean ('mu') from the restrictions logic
            # returns dict {'mu': ..., ...}
            res = estimator.apply_physical_restrictions(pred, limit_mm)
            adjusted_preds.append(res['mu'])
            
        master_df['pred_depth_ml'] = adjusted_preds
        
        # Save feature importance if available
        if hasattr(estimator.model, 'feature_importances_'):
            feat_imp = pd.DataFrame({
                'Feature': estimator.feature_cols,
                'Importance': estimator.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            results['feature_importance'] = feat_imp

        # Generate SHAP values
        try:
            logging.info("Generating SHAP explanation...")
            results['shap_values'] = estimator.explain_model(master_df)
        except Exception as e:
            logging.warning(f"SHAP explanation failed: {e}")

    except Exception as e:
        logging.error(f"ML Step Failed: {e}")
        # Continue with critical error or fallback?
        # For now, propagate exception
        raise e
        
    results['master_df'] = master_df.copy()

    # --- PHASE B: INITIALIZE T=0 STATE ---
    # Determine Initial Depth and Uncertainty per Joint based on Hierarchy
    
    # Convert tolerances to dictionary for fast lookup {Type: Tolerance}
    # Tolerance is fractional (e.g. 0.10 for 10%)
    tol_dict = dict(zip(tolerances_df['Defect Type'], tolerances_df['Tolerance']))
    default_tol = 0.10
    
    # Vectorized initialization
    n_rows = len(master_df)
    
    # Arrays for evolution
    current_depths = np.zeros(n_rows)
    current_std_devs = np.zeros(n_rows)
    current_lengths = master_df['largo'].fillna(10.0).values # Default length if missing
    
    # Wall Thickness
    thickness = master_df['espesor'].values
    
    # Logic:
    # Level 1: Field Data
    # Level 2: ILI Data
    # Level 3: ML Data
    
    # We can create masks
    has_field = (master_df['profundidad_campo_mm'].notna()) & (master_df['profundidad_campo_mm'] > 0)
    has_ili = (master_df['profundidad_mm'].notna()) & (master_df['profundidad_mm'] > 0)
    
    # Iterate to set initial values (Iterating arrays is fast enough for <100k rows, or use np.where)
    # Using np.where is cleaner but logic is tri-state.
    
    # 1. Field Data
    # prioritizing field
    
    # 2. ILI Data (where no field)
    # detecting tolerance based on defect type
    # map defect types to tolerance values
    defect_types = master_df['tipo_defecto'].fillna('General')
    ili_tols_fraction = defect_types.map(tol_dict).fillna(default_tol).values
    # ILI Std Dev = Tolerance * Depth? Or Tolerance * Thickness? OR just Tolerance (which is 10%... of what?)
    # "Tolerancias de dimensionamiento... 10%". Usually means +/- 10% of Depth with 80% confidence -> implies std dev logic.
    # Common industry standard: Tolerance = 3 * StdDev? 
    # Or Tolerance is the Std Dev itself?
    # Requirement: "Tabla para ingresar tolerancias (desviacion estandar directa)"
    # Ah, explicit! "Tolerance IS the direct standard deviation" (fractional or absolute?)
    # "General | 10%". This implies 10% of Depth.
    # Let's assume StdDev = Value * Depth.
    
    ili_std_devs = ili_tols_fraction * master_df['espesor'].values
    
    # 3. ML Data (where no field and no ILI)
    # Censored Logic (Truncated Normal) occurs here?
    # Logic says: "Si no hay ni ILI ni Directa -> USAR PREDICCION ML"
    # Wait, Phase A logic says: "Para la simulación, define la variable aleatoria... como Truncada Normal"
    # This applies to the ML case.
    # The pure ML prediction is just a number. The distribution is truncated.
    # For numeric simulation here (deterministic growth with probabilistic POF), we typically track Mean and StdDev.
    # But POF calculator usually assumes Normal. 
    # If distribution is Truncated Normal, the POF calc (SF) needs to handle it or we approximate.
    # Requirement 3.3 says: "Distribución de Profundidad: Normal, con Valor Medio y desviacion estandar..."
    # So we force it to Normal? Or we pass parameters?
    # Let's approximate the Truncated Normal as a Normal for the POF step to keep it efficient, 
    # calculate effective Mean and effective StdDev of the truncated distribution.
    
    ml_preds = master_df['pred_depth_ml'].values
    # Censoring Logic Application (Vectorized approximations)
    # If pred is high (> threshold), mean -> shifted up.
    # If pred is low, mean -> shifted down.
    # Let's use the helper logic logic roughly.
    
    # Initial Assignment Loop (using vectors)
    # Depths
    current_depths = np.where(has_field, master_df['profundidad_campo_mm'].values,
                        np.where(has_ili, master_df['profundidad_mm'].values,
                            # ML Case: Logic is complex, let's just use prediction for Mean now
                            # Refinement: Add logic to shift mean based on threshold?
                            # For simplicity in V1: Use raw ML prediction. 
                            ml_preds
                        )
                     )
                     
    # Std Devs
    # Field: 1% (of depth? or absolute? Requirement says "desviacion estandar de 1%") -> Assume 1% of Depth
    field_std = 0.01 * current_depths
    
    # ML: "Desviacion calculada en fase B" (ml_uncertainty_mm - absolute value)
    
    current_std_devs = np.where(has_field, field_std,
                            np.where(has_ili, ili_std_devs,
                                ml_uncertainty_mm
                            )
                       )

    # --- PHASE C: BUCLE TEMPORAL ---
    years_duration = target_date.year - ili_date.year
    # If same year, at least 1 point (T=0)
    # If target 2025, base 2023 -> duration 2. Points 0,1,2.
    
    # We want to include the target year.
    # range 0 to duration implies duration+1 points.
    
    years = np.arange(0, years_duration + 1) # 0, 1, ... duration
    
    # Storage
    pof_history = []
    
    # Corrosion Rates
    corrosion_rates = master_df['tasa_corrosion_mm_ano'].fillna(0.1).values # Default 0.1 mm/yr
    
    # Pipe Props
    pressures = master_df['presion'].fillna(0).values # Ensure pressure exists
    diameters = master_df['diametro'].values
    smys_vals = master_df['SMYS'].values
    
    logging.info(f"Starting Simulation Loop for {len(years)} years.")
    
    for yr_idx in years:
        year_num = ili_date.year + yr_idx
        
        # 1. Calculate POF for current state
        pof_values = pof_calculator.calculate_pof_analytics(
            current_depths,
            current_lengths,
            pressures,
            diameters,
            thickness,
            smys_vals,
            current_std_devs
        )
        
        # Store results
        # We need to map back to Joint ID (index or distance)
        # Efficient storage: List of dicts or append to a big list
        # Creating a DF per year is heavy.
        # Let's store a summary or just the POF array to reconstruction later.
        # Actually for the UI heatmap, we need structured data.
        
        # Snapshot DataFrame
        snapshot = pd.DataFrame({
            'Year': year_num,
            'Junta_ID': master_df.index, # Proxy ID
            'Distance': master_df['distancia_m'],
            'Depth': current_depths.copy(),
            'POF': pof_values,
            'LimitState': 'B31G'
        })
        pof_history.append(snapshot)
        
        # 2. Grow Defects (for next step)
        if yr_idx < years[-1]:
            current_depths, current_lengths = growth_model.grow_defects(
                current_depths,
                current_lengths,
                corrosion_rates,
                thickness, # Max depth limit
                time_step=1.0
            )
            # Std Dev growth? Uncertainty grows with time usually.
            # Simple model: Std Dev usually increases sqrt(t) * rate_uncertainty.
            # Requirement doesn't specify uncertainty growth.
            # We will keep Std Dev constant (Assumption) or grow it slightly?
            # Let's keep distinct logic: Future uncertainty = Sqrt(Initial^2 + (RateUncertainty*t)^2).
            # For now, keep constant as per explicit instruction absence.
            
    # Combine History
    full_results = pd.concat(pof_history, ignore_index=True)
    results['pof_results'] = full_results
    
    # --- ADD POF COLUMNS TO MASTER_DF ---
    # We want columns: POF_2023, POF_2024, etc.
    # pof_history is a list of DataFrames per year.
    # Each snapshot matches master_df by index (since we used master_df.index)
    
    for snapshot in pof_history:
        year = snapshot['Year'].iloc[0]
        # Ensure alignment by index
        master_df[f'POF_{year}'] = snapshot['POF'].values

    results['master_df'] = master_df
    
    logging.info("Simulation Complete.")
    return results

if __name__ == "__main__":
    # Test stub
    pass
