
import unittest
import pandas as pd
import numpy as np
from ml_model import DefectDepthEstimator

class TestDefectDepthEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = DefectDepthEstimator()
        
        # Create dummy training data
        self.train_data = pd.DataFrame({
            'resistividad_suelo_ohm_cm': [1000, 2000, 1500, 1200],
            'cp_potencial_on_mv': [-850, -900, -870, -880],
            'interferencia_dc': [0, 1, 0, 1],
            'edad_recubrimiento_anos': [10, 12, 11, 10],
            'tipo_suelo': ['Sand', 'Clay', 'Sand', 'Clay'],
            'tipo_recubrimiento': ['FBE', 'FBE', 'FBE', 'Epoxy'],
            'profundidad_mm': [5.0, 3.0, 4.0, 2.0], # > 0
            'profundidad_campo_mm': [np.nan, np.nan, np.nan, np.nan]
        })
        
        # Data for prediction (depth=0)
        self.predict_data = pd.DataFrame({
            'resistividad_suelo_ohm_cm': [1100, 1900],
            'cp_potencial_on_mv': [-860, -890],
            'interferencia_dc': [0, 1],
            'edad_recubrimiento_anos': [10, 12],
            'tipo_suelo': ['Sand', 'Clay'],
            'tipo_recubrimiento': ['FBE', 'FBE'],
            'profundidad_mm': [0, 0]
        })

        # Data for validation
        self.val_data = pd.DataFrame({
            'resistividad_suelo_ohm_cm': [1100, 1900],
            'cp_potencial_on_mv': [-860, -890],
            'interferencia_dc': [0, 1],
            'edad_recubrimiento_anos': [10, 12],
            'tipo_suelo': ['Sand', 'Clay'], # Note: Unseen label 'Clay' in predict but present in train
            'tipo_recubrimiento': ['FBE', 'PE'], # 'PE' is new/unseen
            'profundidad_mm': [0, 0],
            'profundidad_campo_mm': [4.5, 3.2]
        })

    def test_train_and_predict(self):
        # Train
        self.estimator.train(self.train_data)
        
        # Predict
        preds = self.estimator.predict(self.predict_data)
        self.assertEqual(len(preds), 2)
        print(f"Predictions: {preds}")
        
    def test_uncertainty(self):
        self.estimator.train(self.train_data)
        sigma = self.estimator.calculate_uncertainty(self.val_data)
        # Should return a float or None if failed, but here we have data
        # Note: 'PE' is unseen in training, so it might get encoded to -1
        print(f"Uncertainty Sigma: {sigma}")
        self.assertTrue(isinstance(sigma, float))

    def test_physical_restrictions(self):
        # Case 1: High prediction
        params = self.estimator.apply_physical_restrictions(8.0, 10.0)
        print(f"Params High: {params}")
        self.assertAlmostEqual(params['upper'], 10.0)
        self.assertTrue(params['mu'] > 5.0) # Should be biased up

        # Case 2: Low prediction
        params_low = self.estimator.apply_physical_restrictions(1.0, 10.0)
        print(f"Params Low: {params_low}")
        self.assertEqual(params_low['mu'], 1.0) # Should stay as is if > lower

if __name__ == '__main__':
    unittest.main()
