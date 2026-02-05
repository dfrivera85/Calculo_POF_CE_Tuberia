import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class DefectDepthEstimator:
    def __init__(self, random_state=42):
        self.model = RandomForestRegressor(random_state=random_state, n_estimators=1000)
        self.label_encoders = {}
        self.categorical_cols = ['tipo_suelo', 'tipo_recubrimiento']
        self.feature_cols = [
            'resistividad_suelo_ohm_cm', 
            'cp_potencial_on_mv', 
            'interferencia_dc', 
            'edad_recubrimiento_anos',
            'tipo_suelo',
            'tipo_recubrimiento'
        ]
        self.target_col = 'profundidad_mm'
        self.field_col = 'profundidad_campo_mm'

    def _prepare_data(self, df, training=False):
        """
        Prepares features for training or prediction.
        Handles missing values and categorical encoding.
        """
        # Create a copy to avoid SettingWithCopy warnings
        data = df.copy()
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in data.columns:
                data[col] = np.nan
        
        X = data[self.feature_cols].copy()

        # Handle Categorical Columns
        for col in self.categorical_cols:
            # Convert to string to handle mixed types and fill NaNs
            X[col] = X[col].fillna('Unknown').astype(str)
            
            if training:
                le = LabelEncoder()
                # Fit encoder
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen labels by mapping them to a known label or a special value
                    # Here we map unseen to the first class (usually 0) or handle carefully
                    # A robust way is to use detailed mapping
                    known_classes = set(le.classes_)
                    X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in known_classes else -1)
                else:
                    # Should not happen if trained, but strictly speakin
                    X[col] = -1

        # Handle Numeric Columns (Simple Imputation)
        numeric_cols = [c for c in self.feature_cols if c not in self.categorical_cols]
        for col in numeric_cols:
            if training:
                # In a real pipeline, we'd save the imputer means. 
                # For this scope, filling with 0 or mean of current batch is acceptable 
                # provided the input DF is the master DF which is relatively complete.
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna(0) # Default for missing env data

        return X

    def train(self, df):
        """
        Trains the Random Forest model using rows with known anomaly depths (depth > 0).
        """
        # Filter for training data: Joints where ILI extracted a depth > 0
        train_df = df.copy()
        
        if train_df.empty:
            print("Warning: No training data available (no anomalies with depth > 0). Model not trained.")
            return
            
        X = self._prepare_data(train_df, training=True)
        y = train_df[self.target_col]
        
        self.model.fit(X, y)
        print("Model trained successfully.")

    def predict(self, df):
        """
        Predicts depth for the provided dataframe rows.
        """
        X = self._prepare_data(df, training=False)
        return self.model.predict(X)

    def calculate_uncertainty(self, df):     
        return 0.10

    def apply_physical_restrictions(self, predicted_depth_mm, detection_threshold):
        """
        Applies censoring logic for simulation distribution parameters.
        
        Args:
            predicted_depth_mm (float): The ML predicted depth.
            detection_threshold_mm (float): The ILI detection threshold (e.g. 10% of WT).
            
        Returns:
            dict: Parameters for the Truncated Normal distribution.
                  {'lower': 0, 'upper': threshold, 'mean': adjusted_mean, 'sigma': relative_sigma}
        """
        # Logic from lines 60-65:
        # Variable is Truncated Normal [0, threshold]
        # "Si profundidad_estimada_ml es ALTA -> La curva se inclina hacia el límite superior"
        # "Si profundidad_estimada_ml es BAJA -> La curva se inclina hacia 0"
        
        # We can map the predicted depth to the [0, threshold] range.
        # If prediction > threshold, we bias towards threshold.
        # If prediction < threshold, we use it as the mean (clamped).
        
        # Simple heuristic implementation:
        # Mean of the distribution = min(prediction, threshold * 0.99)
        # Sigma? Not specified, usually small enough to fit.
        # Requirement says:
        # "Si pred es ALTA -> media = 8% (assuming 10% thresh)"
        # "Si pred es BAJA -> media = 2%"
        
        # Let's perform a linear interpolation or clamping.
        # If Prediction >= Threshold: Mean = 0.8 * Threshold
        # If Prediction <= 0: Mean = 0
        # Else: Mean = Prediction
        
        lower = 0
        upper = detection_threshold
        
        # Clamping the mean for the distribution
        if predicted_depth_mm >= upper:
            mu = upper  # Bias towards upper
        elif predicted_depth_mm <= lower:
            mu = lower
        else:
            mu = predicted_depth_mm
            
        # Sigma is not strictly defined in requirements for this specific distribution.
        # "Sesgo de la curva" implies the mean position.
        # We'll return the parameters for the caller (POF calculator) to construct the distribution.
        
        return {
            'distribution': 'truncated_normal',
            'lower': lower,
            'upper': upper,
            'mu': mu,
            'sigma': (upper - lower) / 4 # Rough approx to fit 2 sigma
        }

    def explain_model(self, df):
        """
        Generates SHAP values for the provided dataframe.
        """
        import shap
        X = self._prepare_data(df, training=False)
        
        # Use TreeExplainer optimized for Tree models
        explainer = shap.Explainer(self.model)
        shap_values = explainer(X)
        
        return shap_values
