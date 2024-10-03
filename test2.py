import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from typing import Callable

from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.metrics import PrequentialError

np.random.seed(seed=31)

class DriftDetection:
    def __init__(self, pipeline: Pipeline, detector: BaseEstimator, metric: Callable):
        """
        Initialize the DriftDetection class.
        
        Args:
            pipeline (Pipeline): The model pipeline (preprocessing + model).
            detector (BaseEstimator): The drift detection algorithm (e.g., DDM).
            metric (Callable): The metric for evaluation (e.g., accuracy).
        """
        self.pipeline = pipeline
        self.detector = detector
        self.metric = metric
        self.metric_error = None

    def fit(self, X_train, y_train):
        """Fit the model pipeline."""
        self.pipeline.fit(X_train, y_train)
    
    def simulate_data_stream(self, X_test, y_test):
        """
        Simulate the data stream and detect concept drift.
        
        Args:
            X_test (np.array): Test feature set.
            y_test (np.array): Test labels.
        """
        drift_flag = False
        
        for i, (X, y) in enumerate(zip(X_test, y_test)):
            # Make a prediction
            y_pred = self.pipeline.predict(X.reshape(1, -1))
            error = 1 - (y_pred.item() == y.item())
            
            # Update the metric
            self.metric_error = self.metric(error_value=error)
            
            # Update the drift detector
            self.detector.update(value=error)
            status = self.detector.status
            
            # Check for concept drift
            if status["drift"] and not drift_flag:
                drift_flag = True
                print(f"Concept drift detected at step {i}. Accuracy: {1 - self.metric_error:.4f}")
        
        if not drift_flag:
            print("No concept drift detected")
        print(f"Final accuracy: {1 - self.metric_error:.4f}\n")
    
    def induce_drift(self, y_test, drift_percentage=0.2):
        """
        Introduce concept drift into the test labels.
        
        Args:
            y_test (np.array): Test labels.
            drift_percentage (float): Proportion of labels to modify to simulate drift.
        """
        drift_size = int(len(y_test) * drift_percentage)
        y_test_drift = y_test[-drift_size:].copy()  # Copy last 20% of the labels
        modify_idx = np.random.rand(len(y_test_drift)) <= 0.5
        y_test_drift[modify_idx] = (y_test_drift[modify_idx] + 1) % len(np.unique(y_test))
        y_test[-drift_size:] = y_test_drift
        return y_test

    def reset(self):
        """Reset the detector and metric for a fresh run."""
        self.detector.reset()
        self.metric.reset()

# Generate synthetic time series data
n_samples = 1000
n_features = 1

# Before drift
time = np.arange(n_samples // 2)
X1 = np.sin(time / 10).reshape(-1, 1) + np.random.randn(n_samples // 2, n_features) * 0.1
y1 = (X1[:, 0] > 0).astype(int)

# After drift
time = np.arange(n_samples // 2, n_samples)
X2 = np.cos(time / 10).reshape(-1, 1) + np.random.randn(n_samples // 2, n_features) * 0.1
y2 = (X2[:, 0] > 0).astype(int)

# Combine data
X = np.vstack((X1, X2))
y = np.hstack((y1, y2))

# Split train (70%) and test (30%) while maintaining time order
split_index = int(len(X) * 0.7)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define and fit model pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])
pipeline.fit(X=X_train, y=y_train)

# Define drift detector configuration and instance
config = DDMConfig(warning_level=2.0, drift_level=3.0, min_num_instances=25)
detector = DDM(config=config)

# Define metric (accuracy in this case)
metric = PrequentialError(alpha=1.0)

# Create an instance of DriftDetection class
drift_detection = DriftDetection(pipeline=pipeline, detector=detector, metric=metric)

# Run drift detection on test data (no drift expected)
drift_detection.simulate_data_stream(X_test, y_test)

# Introduce drift by modifying the last 20% of labels
y_test_with_drift = drift_detection.induce_drift(y_test)

# Reset detector and metric
drift_detection.reset()

# Run drift detection again (drift expected)
drift_detection.simulate_data_stream(X_test, y_test_with_drift)


