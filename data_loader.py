import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REQUIRED_COLUMNS = {
    'juntas_soldadura.csv': ['distancia_inicio_m', 'distancia_fin_m', 'lat', 'lon', 'diametro', 'espesor', 'SMYS', 'tasa_corrosion_mm_ano'],
    'anomalias.csv': ['distancia_m', 'profundidad_mm', 'ancho', 'largo', 'tipo_defecto', 'latitud', 'longitud'],
    'presion.csv': ['distancia_inicio_m', 'distancia_fin_m', 'presion'],
    'resistividad.csv': ['distancia_inicio_m', 'distancia_fin_m', 'resistividad_suelo_ohm_cm'],
    'tipo_suelo.csv': ['distancia_inicio_m', 'distancia_fin_m', 'tipo_suelo'],
    'potencial.csv': ['distancia_inicio_m', 'distancia_fin_m', 'cp_potencial_on_mv'],
    'interferencia.csv': ['distancia_inicio_m', 'distancia_fin_m', 'interferencia_dc'],
    'tipo_recubrimiento.csv': ['distancia_inicio_m', 'distancia_fin_m', 'tipo_recubrimiento'],
    'edad_recubrimiento.csv': ['distancia_inicio_m', 'distancia_fin_m', 'edad_recubrimiento_anos'],
    'inspecciones_directas.csv': ['distancia_m', 'profundidad_campo_mm', 'ancho_campo_mm', 'largo_campo_mm', 'tipo_defecto_campo']
}

def validate_columns(df, filename):
    """
    Validates that the DataFrame contains the required columns for the given filename.
    """
    if filename not in REQUIRED_COLUMNS:
        logging.warning(f"No validation schema defined for {filename}. Skipping validation.")
        return

    required = set(REQUIRED_COLUMNS[filename])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"File {filename} is missing required columns: {missing}")

def load_single_file(file_source, filename):
    """
    Helper function to load and validate a single CSV file.
    file_source: File path (str) or file-like object.
    filename: Name of the file (e.g., 'juntas_soldadura.csv') used for schema validation.
    """
    if filename not in REQUIRED_COLUMNS:
        logging.warning(f"No validation schema defined for {filename}. Skipping validation.")
        return None

    try:
        df = pd.read_csv(file_source)
        validate_columns(df, filename)
        logging.info(f"Successfully loaded {filename} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading {filename}: {e}")
        # Return empty dataframe with required columns to create a robust fail-soft behavior
        return pd.DataFrame(columns=REQUIRED_COLUMNS[filename])

def load_data_from_folder(base_path):
    """
    Loads and validates the 10 CSV files from the specified directory.
    Returns a dictionary of DataFrames.
    """
    data_frames = {}
    
    for filename in REQUIRED_COLUMNS.keys():
        file_path = os.path.join(base_path, filename)
        if not os.path.exists(file_path):
            logging.warning(f"File {filename} not found at {base_path}. Creating empty dataframe.")
            data_frames[filename.replace('.csv', '')] = pd.DataFrame(columns=REQUIRED_COLUMNS[filename])
            continue

        df = load_single_file(file_path, filename)
        if df is not None: # df could be None if no schema, or an empty DF if error
             data_frames[filename.replace('.csv', '')] = df
        else:
             data_frames[filename.replace('.csv', '')] = pd.DataFrame(columns=REQUIRED_COLUMNS[filename])

    return data_frames

def load_data_from_dict(uploaded_files):
    """
    Loads data from a dictionary of file-like objects (e.g., from Streamlit uploader).
    uploaded_files: dict {filename: file_object}
    """
    data_frames = {}
    
    # Iterate over REQUIRED keys to ensure we look for everything we need
    for filename in REQUIRED_COLUMNS.keys():
        file_obj = uploaded_files.get(filename)
        
        if file_obj is None:
            logging.warning(f"File {filename} not provided. Creating empty dataframe.")
            data_frames[filename.replace('.csv', '')] = pd.DataFrame(columns=REQUIRED_COLUMNS[filename])
            continue
            
        # Reset pointer if it's a file object just in case
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
            
        df = load_single_file(file_obj, filename)
        if df is not None:
            data_frames[filename.replace('.csv', '')] = df
        else:
            data_frames[filename.replace('.csv', '')] = pd.DataFrame(columns=REQUIRED_COLUMNS[filename])
            
    return data_frames

# Backward compatibility alias
load_data = load_data_from_folder

def create_master_dataframe(dfs):
    """
    Merges all DataFrames into a single Master DataFrame at the Defect/Segment level.
    """
    # 1. Prepare Juntas (Base Backbone)
    juntas = dfs.get('juntas_soldadura').sort_values('distancia_inicio_m').reset_index(drop=True)
    anomalias = dfs.get('anomalias').sort_values('distancia_m').reset_index(drop=True)
    
    if juntas.empty:
         raise ValueError("juntas_soldadura.csv is empty or missing. Cannot build master dataframe.")

    # 2. Map Anomalies to Juntas
    
    # Use merge_asof with direction='backward' to find the nearest joint start <= anomaly dist
    anomalias_merged = pd.merge_asof(
        anomalias,
        juntas,
        left_on='distancia_m',
        right_on='distancia_inicio_m',
        direction='backward',
        suffixes=('_anomalia', '_junta')
    )
    
    # Filter to ensure the anomaly actually falls within the joint's range
    # tolerance could be used in merge_asof if we had a fixed length, but length varies.
    anomalias_merged = anomalias_merged[
        anomalias_merged['distancia_m'] <= anomalias_merged['distancia_fin_m']
    ].copy()

    # Combine coordinates: use 'lat'/'lon' from junta if 'latitud'/'longitud' from anomalia is missing
    if 'latitud' not in anomalias_merged.columns:
        anomalias_merged['latitud'] = pd.NA
    if 'longitud' not in anomalias_merged.columns:
        anomalias_merged['longitud'] = pd.NA

    if 'lat' in anomalias_merged.columns:
        anomalias_merged['latitud'] = anomalias_merged['latitud'].fillna(anomalias_merged['lat'])
    if 'lon' in anomalias_merged.columns:
        anomalias_merged['longitud'] = anomalias_merged['longitud'].fillna(anomalias_merged['lon'])
    
    # 3. Create "Clean Joint" segments (Dummy Anomalies for ML predictions later)
    # Identify joints that HAVE anomalies
    joints_with_anomalies = set(anomalias_merged['distancia_inicio_m'].unique())
    
    # Identify clean joints
    # specific logic: filter original juntas where start_dist is NOT in joints_with_anomalies
    clean_juntas = juntas[~juntas['distancia_inicio_m'].isin(joints_with_anomalies)].copy()
    
    # Create rows for clean joints
    if not clean_juntas.empty:
        # Set anomaly properties to null/zero or indicators
        clean_juntas['profundidad_mm'] = 0
        clean_juntas['ancho'] = 0
        clean_juntas['largo'] = 0
        clean_juntas['tipo_defecto'] = 'Sin Defecto'
        
        # For the master DF, we need a 'distancia_m' column to sort everything linearly.
        # For a clean joint (segment), we can use its start point as the reference distance.
        clean_juntas['distancia_m'] = clean_juntas['distancia_inicio_m']

        # Map 'lat'/'lon' to 'latitud'/'longitud' for consistency with anomalias
        if 'lat' in clean_juntas.columns:
             clean_juntas['latitud'] = clean_juntas['lat']
        if 'lon' in clean_juntas.columns:
             clean_juntas['longitud'] = clean_juntas['lon']
        
        # Combine
        master_df = pd.concat([anomalias_merged, clean_juntas], ignore_index=True)
    else:
        master_df = anomalias_merged.copy()
        
    master_df = master_df.sort_values('distancia_m').reset_index(drop=True)

    # 4. Merge Environmental Data (Spatial Lookups - Segments)
    # Env files now have [distancia_inicio_m, distancia_fin_m]
    env_files = ['resistividad', 'tipo_suelo', 'potencial', 'interferencia', 'tipo_recubrimiento', 'edad_recubrimiento', 'presion']
    
    for key in env_files:
        if key in dfs and not dfs[key].empty:
            env_df = dfs[key].sort_values('distancia_inicio_m')
            
            # Ensure merge keys are float to avoid MergeError (int vs float)
            master_df['distancia_m'] = master_df['distancia_m'].astype(float)
            env_df['distancia_inicio_m'] = env_df['distancia_inicio_m'].astype(float)
            
            # Similar logic: merge master(distancia_m) to env(distancia_inicio_m)
            master_df = pd.merge_asof(
                master_df,
                env_df,
                left_on='distancia_m',
                right_on='distancia_inicio_m',
                direction='backward',
                suffixes=('', f'_{key}')
            )
                       
            # If the merged value is out of bounds, set the env columns to NaN.
            # Columns to clean: the value columns from env_df.
            value_cols = [c for c in env_df.columns if c not in ['distancia_inicio_m', 'distancia_fin_m']]
            
            # Check conditions
            # We need to know which columns correspond to the start/end from the RIGHT table.
            
            right_start = f'distancia_inicio_m_{key}'
            right_end = f'distancia_fin_m_{key}'
            
            # Filter condition: where (distancia_m > right_end) -> invalidate
            # Note: We must handle NaNs if no match found (though merge_asof usually finds something unless empty)
            
            if right_end in master_df.columns:
                 mask_invalid = master_df['distancia_m'] > master_df[right_end]
                 if mask_invalid.any():
                     for col in value_cols:
                         if col in master_df.columns:
                             master_df.loc[mask_invalid, col] = None
                             
                     # Optionally drop the generic start/end cols from env to keep df clean
                     master_df.drop(columns=[right_start, right_end], axis=1, inplace=True, errors='ignore')

    # 5. Merge Field Data (Direct Inspections)
    # This remains point-based: ['distancia_m', ...]
    if 'inspecciones_directas' in dfs and not dfs['inspecciones_directas'].empty:
        insp_df = dfs['inspecciones_directas'].sort_values('distancia_m')
        insp_df['distancia_m'] = insp_df['distancia_m'].astype(float)
        
        master_df = pd.merge_asof(
            master_df,
            insp_df,
            on='distancia_m',
            direction='nearest',
            tolerance=2, 
            suffixes=('', '_field')
        )

    ##debugginf - borrar
    master_df.to_csv('master_df.csv', index=False)

    return master_df
