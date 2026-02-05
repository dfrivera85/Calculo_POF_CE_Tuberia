import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import main

class TestPOFSimulation(unittest.TestCase):
    def setUp(self):
        # Create dummy dataframes mimicking the 10 inputs
        self.dfs = {}
        
        # 1. Juntas
        self.dfs['juntas_soldadura'] = pd.DataFrame({
            'distancia_m': [0, 100, 200, 300, 400],
            'lat': [0, 0, 0, 0, 0],
            'lon': [0, 0, 0, 0, 0],
            'diametro': [300]*5, # mm
            'espesor': [10]*5, # mm
            'SMYS': [300]*5, # MPa
            'tasa_corrosion_mm_ano': [0.1]*5
        })
        
        # 2. Anomalies (ILI)
        # Anomaly at 150m (Joint 100m) with depth 2mm
        self.dfs['anomalias'] = pd.DataFrame({
            'distancia_m': [150],
            'profundidad_mm': [2.0],
            'ancho': [10],
            'largo': [20],
            'tipo_defecto': ['General']
        })
        
        # 3. Environment (Needs to cover distance)
        env_files = ['resistividad', 'tipo_suelo', 'potencial', 'interferencia', 'tipo_recubrimiento', 'edad_recubrimiento', 'presion']
        for key in env_files:
            if key == 'presion':
                self.dfs[key] = pd.DataFrame({'distancia_m': [0, 500], 'presion': [5.0, 5.0]}) # MPa
            elif 'tipo' in key:
                 self.dfs[key] = pd.DataFrame({'distancia_m': [0, 500], key: ['Clay', 'Clay'] if 'suelo' in key else ['FBE', 'FBE']})
            elif key == 'edad_recubrimiento':
                 self.dfs[key] = pd.DataFrame({'distancia_m': [0, 500], 'edad_recubrimiento_anos': [10, 10]})
            else:
                 self.dfs[key] = pd.DataFrame({'distancia_m': [0, 500], key.replace('.csv','_suelo_ohm_cm' if 'res' in key else '_on_mv' if 'pot' in key else '_dc'): [1000, 1000]})

        # Add Field Data (Validation)
        # Validation point matches Anomaly at 150m? OR new point?
        # Let's add field data at 150m
        self.dfs['inspecciones_directas'] = pd.DataFrame({
            'distancia_m': [150],
            'profundidad_campo_mm': [2.2], # Slightly deeper than ILI (2.0)
            'ancho_campo_mm': [10],
            'largo_campo_mm': [20],
            'tipo_defecto_campo': ['General']
        })
        
        # Tolerances
        self.tolerances_df = pd.DataFrame({
            'Defect Type': ['General'],
            'Tolerance': [0.10]
        })
        
    def test_run_simulation(self):
        ili_date = datetime(2023, 1, 1)
        target_date = datetime(2025, 1, 1) # 3 years (2023, 2024, 2025)
        
        try:
            results = main.run_simulation(
                self.dfs, 
                ili_date, 
                target_date, 
                self.tolerances_df, 
                detection_threshold=0.10
            )
            
            # Checks
            self.assertIn('master_df', results)
            self.assertIn('pof_results', results)
            self.assertIn('ml_uncertainty_status', results)
            
            master = results['master_df']
            self.assertFalse(master.empty)
            print(f"Master DF rows: {len(master)}")
            
            pof_res = results['pof_results']
            self.assertFalse(pof_res.empty)
            print(f"POF Results rows: {len(pof_res)}")
            
            # Check Years
            unique_years = sorted(pof_res['Year'].unique())
            self.assertEqual(unique_years, [2023, 2024, 2025])
            
            # Check Logic: Field vs ILI vs ML
            # Row at 150m has Field Data (2.2mm). 
            # Row at 0m has no data -> ML Prediction.
            
            # Let's inspect T=0 (2023)
            res_2023 = pof_res[pof_res['Year'] == 2023]
            
            # Find row near 150m
            # Distance might be slightly different if merged or whatever, but merge_asof should align
            row_150 = res_2023[res_2023['Distance'] == 150]
            if not row_150.empty:
                # Field depth should be initial depth = 2.2
                # But growth might be applied if loop applies growth BEFORE storage or AFTER?
                # Code says: 1. Calc POF (Current Depth) -> Store. 2. Grow.
                # So stored depth for 2023 should be Initial.
                self.assertAlmostEqual(row_150['Depth'].values[0], 2.2, places=2)
                print("Field Data Priority Validated")
            
        except Exception as e:
            self.fail(f"Simulation failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
