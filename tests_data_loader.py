import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
from data_loader import load_data, create_master_dataframe

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create dummy CSVs
        self.juntas_data = {
            'distancia_m': [0, 10, 20, 30],
            'lat': [1, 1, 1, 1],
            'lon': [2, 2, 2, 2],
            'diametro': [12, 12, 12, 12],
            'espesor': [0.25, 0.25, 0.25, 0.25],
            'SMYS': [52000, 52000, 52000, 52000],
            'tasa_corrosion_mm_ano': [0.1, 0.1, 0.1, 0.1]
        }
        pd.DataFrame(self.juntas_data).to_csv(os.path.join(self.test_dir, 'juntas_soldadura.csv'), index=False)
        
        self.anomalias_data = {
            'distancia_m': [5, 15, 16], # Two on "Joint 10", one on "Joint 0"
            'profundidad_mm': [1.5, 2.0, 1.1],
            'ancho': [10, 20, 10],
            'largo': [10, 20, 15],
            'tipo_defecto': ['General', 'Pitting', 'General']
        }
        pd.DataFrame(self.anomalias_data).to_csv(os.path.join(self.test_dir, 'anomalias.csv'), index=False)
        
        # Environmental
        self.resistividad_data = {
            'distancia_m': [0, 20],
            'resistividad_suelo_ohm_cm': [1000, 2000]
        }
        pd.DataFrame(self.resistividad_data).to_csv(os.path.join(self.test_dir, 'resistividad.csv'), index=False)

        # Create empty dummy files for others to pass validation
        dummy_cols = {
            'presion.csv': ['distancia_m', 'presion'],
            'tipo_suelo.csv': ['distancia_m', 'tipo_suelo'],
            'potencial.csv': ['distancia_m', 'cp_potencial_on_mv'],
            'interferencia.csv': ['distancia_m', 'interferencia_dc'],
            'tipo_recubrimiento.csv': ['distancia_m', 'tipo_recubrimiento'],
            'edad_recubrimiento.csv': ['distancia_m', 'edad_recubrimiento_anos'],
            'inspecciones_directas.csv': ['distancia_m', 'profundidad_campo_mm', 'ancho_campo_mm', 'largo_campo_mm', 'tipo_defecto_campo']
        }
        for fname, cols in dummy_cols.items():
            pd.DataFrame(columns=cols).to_csv(os.path.join(self.test_dir, fname), index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_data(self):
        dfs = load_data(self.test_dir)
        self.assertIn('juntas_soldadura', dfs)
        self.assertIn('anomalias', dfs)
        self.assertEqual(len(dfs['juntas_soldadura']), 4)
        self.assertEqual(len(dfs['anomalias']), 3)

    def test_create_master_dataframe(self):
        dfs = load_data(self.test_dir)
        master = create_master_dataframe(dfs)
        
        # Test 1: Total Rows
        # We have 3 anomalies.
        # We have 4 joints total.
        # Anomalies are at 5 (matches Joint 0), 15 (Joint 10), 16 (Joint 10).
        # Joint 20 and Joint 30 are "clean".
        # So we expect: 3 (anomalies) + 2 (clean joints) = 5 rows total.
        self.assertEqual(len(master), 5)
        
        # Test 2: Clean Joints have 0 depth
        clean_rows = master[master['tipo_defecto'] == 'Sin Defecto']
        self.assertEqual(len(clean_rows), 2)
        self.assertTrue((clean_rows['profundidad_mm'] == 0).all())
        
        # Test 3: Joint Attributes are correct
        # Anomaly at 5 should have Joint properties of joint at 0
        point_5 = master[master['distancia_m'] == 5].iloc[0]
        self.assertEqual(point_5['joint_start_m'], 0)
        
        # Test 4: Environmental Merge
        # Resistivity at 0 is 1000, at 20 is 2000.
        # Point at 5 is closer to 0 -> should be 1000
        # Point at 15 is closer to 20 -> should be 2000
        point_5_res = master[master['distancia_m'] == 5].iloc[0]['resistividad_suelo_ohm_cm']
        point_15_res = master[master['distancia_m'] == 15].iloc[0]['resistividad_suelo_ohm_cm']
        
        self.assertEqual(point_5_res, 1000)
        self.assertEqual(point_15_res, 2000)

if __name__ == '__main__':
    unittest.main()
