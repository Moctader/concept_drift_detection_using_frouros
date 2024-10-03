import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from frouros.detectors.data_drift import KSTest, ChiSquareTest  
from frouros.detectors.concept_drift import DDM, DDMConfig

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import importlib

class StockDataPipeline:
    def __init__(self, stock_symbol, split_ratio=0.7, sequence_length=50):
        self.stock_symbol = stock_symbol
        self.split_ratio = split_ratio
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.model = None
        
    def fetch_data(self):
        """Fetch stock data using yfinance and preprocess it."""
        stock_data = yf.download(self.stock_symbol)
        stock_data['Returns'] = stock_data['Close'].pct_change()  # Use return percentage as feature
        stock_data.dropna(inplace=True)
        open_data = stock_data['Open'].values.reshape(-1, 1)
        
        # Use returns for modeling
        data = stock_data['Returns'].values.reshape(-1, 1)
    
        
        # Scale the data
        data_scaled = self.scaler.fit_transform(data)
        
        # Split into training (70%) and current (30%) data
        split_idx = int(len(data_scaled) * self.split_ratio)
        self.X_train, self.X_test = self._create_sequences(data_scaled[:split_idx]), self._create_sequences(data_scaled[split_idx:])
        self.open_reference = open_data[:split_idx]
        self.open_current = open_data[split_idx:]   
        
    def _create_sequences(self, data):
        """Create sequences of stock data for LSTM input."""
        X = []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, 0])
        return np.array(X).reshape(-1, self.sequence_length, 1)
    
    def build_lstm_model(self, layers):
        """Build the LSTM model."""
        self.model = Sequential()
        for layer in layers:
            layer_name, layer_params = list(layer.items())[0]
            layer_class = getattr(importlib.import_module('keras.layers'), layer_name)
            self.model.add(layer_class(**layer_params))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def train(self, epochs=10, batch_size=32):
        """Train the LSTM model on the historical data."""
        self.model.fit(self.X_train, self.X_train[:, -1], epochs=epochs, batch_size=batch_size)
        
    def predict(self, X):
        """Make predictions with the LSTM model."""
        return self.model.predict(X)

class ConceptDriftDetector:
    def __init__(self, warning_level=1.0, drift_level=2.0, min_num_instances=25, feature_drift_threshold=0.05, target_drift_threshold=0.05):
        self.detector = DDM()  # Initialize DDM without config
        self.feature_drift_detector = KSTest()  # Initialize without threshold
        self.target_drift_detector = ChiSquareTest()  # Initialize without threshold
        self.drift_detected = False
        self.drift_point = None
        self.target_drift_threshold = target_drift_threshold  
        self.feature_drift_threshold = feature_drift_threshold
        
    def detect_drift(self, y_true, y_pred):
        """Update the drift detector with new prediction errors."""
        error = mean_squared_error([y_true], [y_pred])
        self.detector.update(value=error)
        if self.detector.drift_state:
            self.drift_detected = True
            print("Concept drift detected!")
        return error

    def detect_feature_drift(self, open_reference, open_current):
        """Detect feature drift using KSTest."""
        self.feature_drift_detector.fit(X=open_reference)
        drift_detected = self.feature_drift_detector.compare(X=open_current)[0]
        if drift_detected.p_value < self.feature_drift_threshold:
            print(f"Feature drift detected: {drift_detected}")
        else:
            print(f"No feature drift detected")


    def detect_target_drift(self, y_train, y_test):
        """Detect target drift using ChiSquareTest."""
        drift_flag = False
        y_train = np.array(y_train).flatten()  # Ensure 1-dimensional array
        y_test = np.array(y_test).flatten()  # Ensure 1-dimensional array
        self.target_drift_detector.fit(X=y_train)
        drift_detected = self.target_drift_detector.compare(X=y_test)[0]
        if drift_detected.p_value < self.target_drift_threshold and not drift_flag:
            drift_flag = True
            print(f"Target drift detected: {drift_detected}")
        else:
            print(f"No target drift detected")


    def stream_test(self, X_current, pipeline):
        """Simulate data stream over X_current."""
        drift_flag = False
        y_true_list = []
        y_pred_list = []

        for i, X in enumerate(X_current):
            y_true = X[-1, 0]  # The true value is the last value in the sequence
            y_pred = pipeline.predict(X.reshape(1, -1, 1)).item()
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            error = mean_squared_error([y_true], [y_pred])  # Use mean_squared_error for error calculation
            self.detector.update(value=error*40)
            status = self.detector.status
            if status["drift"] and not drift_flag:
                drift_flag = True
                self.drift_point = i
                print(f"Concept drift detected at step {i}.")
        
        if not drift_flag:
            print("No concept drift detected")

        self.detect_target_drift(y_true_list, y_pred_list)

        
        # Calculate regression metrics
        mse = mean_squared_error(y_true_list, y_pred_list)
        mae = mean_absolute_error(y_true_list, y_pred_list)
        r2 = r2_score(y_true_list, y_pred_list)
        
        print(f"Final Mean Squared Error (MSE): {mse:.4f}")
        print(f"Final Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Final R^2 Score: {r2:.4f}")
        
        # Plot the results
        fig, axs = plt.subplots(3, 1, figsize=(14, 15))

        # Plot true values
        axs[0].plot(y_true_list, color='g', label='True Values', linewidth=2, marker='o', markersize=4, alpha=0.7)
        axs[0].set_title('True Values', fontsize=16)
        axs[0].set_xlabel('Time Step', fontsize=14)
        axs[0].set_ylabel('Value', fontsize=14)
        axs[0].grid(True)
        axs[0].legend(fontsize=12)

        # Plot predicted values
        axs[1].plot(y_pred_list, color='y', label='Predicted Values', linewidth=2, marker='x', markersize=4, alpha=0.7)
        axs[1].set_title('Predicted Values', fontsize=16)
        axs[1].set_xlabel('Time Step', fontsize=14)
        axs[1].set_ylabel('Value', fontsize=14)
        axs[1].grid(True)
        axs[1].legend(fontsize=12)

        # Combined plot with drift point
        axs[2].plot(y_true_list, color='g', label='True Values', linewidth=2, marker='o', markersize=4, alpha=0.7)
        axs[2].plot(y_pred_list, color='y', label='Predicted Values', linewidth=2, marker='x', markersize=4, alpha=0.7)
        if self.drift_point is not None:
            axs[2].axvline(x=self.drift_point, color='r', linestyle='--', label='Concept Drift', linewidth=2, alpha=0.7)
        axs[2].set_title('True vs Predicted Values with Concept Drift', fontsize=16)
        axs[2].set_xlabel('Time Step', fontsize=14)
        axs[2].set_ylabel('Value', fontsize=14)
        axs[2].grid(True)
        axs[2].legend(fontsize=12)

        plt.tight_layout()
        plt.show()

class StreamProcessor:
    def __init__(self, pipeline, detector):
        self.pipeline = pipeline
        self.detector = detector
        
    def run(self, X_current):
        """Run the stream test for concept drift detection."""
        self.detector.stream_test(X_current, self.pipeline)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    
    # Create and train the pipeline
    pipeline = StockDataPipeline(
        stock_symbol=cfg.pipeline.stock_symbol,
        split_ratio=cfg.pipeline.split_ratio,
        sequence_length=cfg.pipeline.sequence_length
    )
    pipeline.fetch_data()
    pipeline.build_lstm_model(cfg.pipeline.lstm_model.layers)
    pipeline.train(epochs=cfg.pipeline.train.epochs, batch_size=cfg.pipeline.train.batch_size)
    
    # Create a drift detector
    detector = ConceptDriftDetector(
        warning_level=cfg.drift_detection.warning_level,
        drift_level=cfg.drift_detection.drift_level,
        min_num_instances=cfg.drift_detection.min_num_instances,
        feature_drift_threshold=cfg.drift_detection.feature_drift.threshold,
        target_drift_threshold=cfg.drift_detection.target_drift.threshold
    )
    
    # Get the current data
    X_current = pipeline.X_test
    
    # Run the stream processor for drift detection
    stream_processor = StreamProcessor(pipeline, detector)
    stream_processor.run(X_current)
    
    # Perform feature drift detection
    detector.detect_feature_drift(pipeline.open_reference, pipeline.open_current)
    
    # Perform target drift detection
    detector.detect_target_drift(pipeline.X_train[:, -1], pipeline.X_test[:, -1])

if __name__ == "__main__":
    main()