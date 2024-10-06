'''
Concept drift detection is a critical aspect of machine learning model monitoring and maintenance.
1. observe drift metrics over time
2. Understand model accuracy decay
3. identify a clear point when the model should be retrained based on performance degradation due to concept or data drift

'''

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from omegaconf import DictConfig
import hydra
from datetime import date
import matplotlib.pyplot as plt
from frouros.detectors.data_drift import KSTest, ChiSquareTest
from frouros.metrics import PrequentialError
from frouros.detectors.concept_drift import DDM, ADWIN, PageHinkley
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.callbacks import ModelCheckpoint
from frouros.metrics import PrequentialError
from sklearn.metrics import precision_score, recall_score, f1_score
from Data_class_abstraction import BaseMetrics, Concept_Drift_Metrics, Target_drift_Metircs, Profit_loss_Metrics
from plotting import plot_results
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from plotting import plot_results

metric = PrequentialError(alpha=1.0)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataPipeline:
    def __init__(self, stock_symbol, split_ratio=0.7, sequence_length=50, lstm_config=None):
        self.stock_symbol = stock_symbol
        self.split_ratio = split_ratio
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.lstm_config = lstm_config
        
    def fetch_data(self):
        """Fetch stock data using yfinance and preprocess it."""
        try:
            stock_data = yf.download(self.stock_symbol, START, TODAY)
            stock_data.dropna(inplace=True)
            open_data = stock_data['Open'].values.reshape(-1, 1)
            data = stock_data['Close'].values.reshape(-1, 1)
            data_scaled = self.scaler.fit_transform(data)
            open_data_scaled = self.scaler.fit_transform(open_data)  # Use transform instead of fit_transform
            
            # Split into training (70%) and current (30%) data
            split_idx = int(len(data_scaled) * self.split_ratio)
            self.X_ref, self.y_ref = self._create_sequences(data_scaled[:split_idx])
            self.X_curr, self.y_curr = self._create_sequences(data_scaled[split_idx:])
            self.open_reference = open_data_scaled[:split_idx]
            self.open_current = open_data_scaled[split_idx:]
            logger.info("Data fetched and preprocessed successfully.")
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def _create_sequences(self, data):
        X = []
        y = []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, 0])
            y.append(data[i, 0])  # Take the last column as y_true
        return np.array(X).reshape(-1, self.sequence_length, 1), np.array(y)
    
    def build_and_train_lstm(self):
        try:
            X = self.X_ref.reshape(self.X_ref.shape[0], self.X_ref.shape[1], 1)
            
            model = Sequential()
            for layer in self.lstm_config['layers']:
                layer_type = list(layer.keys())[0]
                layer_params = layer[layer_type]
                if layer_type == 'LSTM':
                    model.add(LSTM(**layer_params))
                elif layer_type == 'Dropout':
                    model.add(Dropout(**layer_params))
                elif layer_type == 'BatchNormalization':
                    model.add(BatchNormalization(**layer_params))
                elif layer_type == 'Dense':
                    model.add(Dense(**layer_params))
            
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MAE'])
            
            self.model = model
            logger.info("LSTM model built successfully.")
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            raise

    def train(self, epochs, batch_size, validation_split, callbacks):
        try:
            self.model.fit(self.X_ref, self.y_ref, validation_split=validation_split, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
            logger.info("Model trained successfully.")
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def predict(self, X):
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

class ConceptDriftDetector:
    def __init__(self, warning_level=1.0, drift_level=2.0, min_num_instances=25, feature_drift_threshold=0.05, target_drift_threshold=0.05):
        self.detector = PageHinkley()  
        self.feature_drift_detector = KSTest()  
        self.target_drift_detector = ChiSquareTest() 
        self.drift_detected = False
        self.drift_point = None  
        self.target_drift_threshold = target_drift_threshold  
        self.feature_drift_threshold = feature_drift_threshold
        self.y_true_list = []  
        self.y_pred_list = []  
        self.metrics = Concept_Drift_Metrics()
        self.Target_metrics = Target_drift_Metircs()

        # Initialize config with necessary parameters
        self.config = {
            'alpha': 0.99,  # Example value, adjust as needed
            'mean_error_rate': np.mean(self.y_true_list) if self.y_true_list else 0,
            'delta': 0.005  # Example value, adjust as needed
        }

    def detect_target_drift(self, y_train, y_test):
        y_train = np.array(y_train).flatten()  
        y_test = np.array(y_test).flatten()  
        self.target_drift_detector.fit(X=y_train)
        drift_detected = self.target_drift_detector.compare(X=y_test)[0]
        if drift_detected.p_value < self.target_drift_threshold:
            print(f"Target drift detected: {drift_detected}")
            self.Target_metrics.update_metrics(
                step=len(self.Target_metrics.steps),
                drift_point=len(self.Target_metrics.steps),
                p_value=drift_detected.p_value  
            )
        else:
            print(f"No target drift detected")
            self.Target_metrics.update_metrics(
                step=len(self.Target_metrics.steps),
                drift_point=None,
                p_value=drift_detected.p_value  
            )

    def stream_test(self, X_curr, y_curr, pipeline):
        """Simulate data stream over current_data."""
        drift_flag = False
        self.y_true_list = []  
        self.y_pred_list = [] 
        self.sum_ = 0.0

        for i, (X, y) in enumerate(zip(X_curr, y_curr)):
            y_true = y  
            y_pred = pipeline.predict(X.reshape(1, -1, 1)).item()
            self.y_true_list.append(y_true)
            self.y_pred_list.append(y_pred)

            error_rate = y_true - y_pred
            self.sum_ = self.config['alpha'] * self.sum_ + (
                error_rate - self.config['mean_error_rate'] - self.config['delta']
            )
            self.detector.update(value=error_rate * 90)
            status = self.detector.status
            if status["drift"] and not drift_flag:
                drift_flag = True
                self.drift_point = i  # Store drift point
                print(f"Concept drift detected at step {i}.")
                self.metrics.update_metrics(step=i, drift_point=i)
    
        if not drift_flag:
            print("No concept drift detected")

        mse = mean_squared_error(self.y_true_list, self.y_pred_list)
        mae = mean_absolute_error(self.y_true_list, self.y_pred_list)

        self.metrics.update_metrics(
            step=len(self.metrics.steps),
            accuracy=None,
            mae_error=mae,
            precision=None,
            recall=None,
            f1_score=None
        )
        
        return self.metrics.get_latest_metrics()
    


class StreamProcessor:
    def __init__(self, pipeline, detector):
        self.pipeline = pipeline
        self.detector = detector
        
    def run(self, X_curr, y_curr):
        return self.detector.stream_test(X_curr, y_curr, self.pipeline)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Create and train the pipeline
    pipeline = StockDataPipeline(
        stock_symbol=cfg.pipeline.stock_symbol,
        split_ratio=cfg.pipeline.split_ratio,
        sequence_length=cfg.pipeline.sequence_length,
        lstm_config=cfg.pipeline.lstm_model
    )

    pipeline.fetch_data()
    pipeline.build_and_train_lstm()

    # Define the ModelCheckpoint callback using config.yaml
    model_checkpoint_cfg = cfg.train.model_checkpoint
    best_model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_cfg.filepath,
        monitor=model_checkpoint_cfg.monitor,
        save_best_only=model_checkpoint_cfg.save_best_only,
        mode=model_checkpoint_cfg.mode,
        verbose=model_checkpoint_cfg.verbose
    )

    pipeline.train(
        epochs=cfg.train.epochs, 
        batch_size=cfg.train.batch_size, 
        validation_split=cfg.train.validation_split, 
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor=cfg.train.early_stopping.monitor, 
                patience=cfg.train.early_stopping.patience, 
                restore_best_weights=cfg.train.early_stopping.restore_best_weights
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=cfg.train.reduce_lr.monitor, 
                factor=cfg.train.reduce_lr.factor, 
                patience=cfg.train.reduce_lr.patience, 
                min_lr=cfg.train.reduce_lr.min_lr
            ),
            best_model_checkpoint_callback  # Add the ModelCheckpoint callback here
        ]
    )
    
    # Create a drift detector
    detector = ConceptDriftDetector(
        warning_level=cfg.drift_detection.warning_level,
        drift_level=cfg.drift_detection.drift_level,
        min_num_instances=cfg.drift_detection.min_num_instances,
        feature_drift_threshold=cfg.drift_detection.feature_drift.threshold,
        target_drift_threshold=cfg.drift_detection.target_drift.threshold
    )
        
    # Run the stream processor for drift detection
    stream_processor = StreamProcessor(pipeline, detector)
    stream_processor.run(pipeline.X_curr, pipeline.y_curr)    

    detector.detect_target_drift(pipeline.y_ref, pipeline.y_curr)
    plot_results(detector.y_true_list, detector.y_pred_list, detector.drift_point)

if __name__ == "__main__":
    main()