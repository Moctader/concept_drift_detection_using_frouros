from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from frouros.detectors.concept_drift import DDM, DDMConfig  # Assuming FREBOUS has similar APIs for concept drift detection

class MultiModelPipeline:
    def __init__(self, models):
        """
        Initialize the MultiModelPipeline with several models.
        
        Args:
            models (dict): A dictionary of model names and their corresponding instances.
        """
        self.models = models
        self.fitted_models = {}

    def fit(self, X_train, y_train):
        """
        Fit each model in the pipeline.
        
        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels.
        """
        for name, model in self.models.items():
            pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
            pipeline.fit(X_train, y_train)
            self.fitted_models[name] = pipeline

    def predict(self, X):
        """
        Predict using each model in the pipeline and return their predictions.
        
        Args:
            X (np.array): Input features for prediction.
        
        Returns:
            dict: A dictionary of predictions from each model.
        """
        predictions = {}
        for name, model in self.fitted_models.items():
            predictions[name] = model.predict(X)
        return predictions

class DriftDetector:
    def __init__(self, detector, config):
        """
        Initialize the DriftDetector with a specific drift detection algorithm.
        
        Args:
            detector (BaseEstimator): The drift detector (e.g., FREBOUS DDM).
            config (dict): Configuration for the drift detector.
        """
        self.detector = detector(config=config)

    def detect_drift(self, errors):
        """
        Run the drift detection process and check for drift.
        
        Args:
            errors (list): List of error values for each prediction.
        
        Returns:
            bool: True if drift is detected, False otherwise.
        """
        drift_flag = False
        for error in errors:
            self.detector.update(error)
            if self.detector.status["drift"]:
                drift_flag = True
        return drift_flag

    def reset(self):
        """Reset the drift detector."""
        self.detector.reset()

# Example usage:

# Define models to use in the pipeline
models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
}

# Create an instance of MultiModelPipeline with multiple models
multi_model_pipeline = MultiModelPipeline(models=models)

# Example stock market training data (replace with actual stock data)
X_train = np.random.randn(100, 5)  # 100 samples, 5 features
y_train = np.random.randint(0, 2, 100)  # Binary classification (0 or 1)

# Fit the models in the pipeline
multi_model_pipeline.fit(X_train, y_train)

# Define DDM drift detector configuration (using FREBOUS)
ddm_config = DDMConfig(warning_level=2.0, drift_level=3.0, min_num_instances=25)
drift_detector = DriftDetector(detector=DDM, config=ddm_config)

# Simulate stock data stream (real-time predictions)
X_test = np.random.randn(100, 5)  # New test data for streaming
y_test = np.random.randint(0, 2, 100)  # Corresponding labels

# Predict using each model and check for drift
for i in range(len(X_test)):
    # Get predictions from all models
    predictions = multi_model_pipeline.predict(X_test[i].reshape(1, -1))

    # Calculate error (for simplicity, assume error is the difference from true label)
    errors = [1 - (pred.item() == y_test[i].item()) for pred in predictions.values()]

    # Detect concept drift
    drift_detected = drift_detector.detect_drift(errors)

    if drift_detected:
        print(f"Concept drift detected at step {i}. Predictions: {predictions}")
        drift_detector.reset()  # Reset detector after drift detection
        break
else:
    print("No concept drift detected in the stream.")

