import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from frouros.detectors.concept_drift import DDM, DDMConfig

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
        
        # Use returns for modeling
        data = stock_data['Returns'].values.reshape(-1, 1)
        
        # Scale the data
        data_scaled = self.scaler.fit_transform(data)
        
        # Split into training (70%) and current (30%) data
        split_idx = int(len(data_scaled) * self.split_ratio)
        self.X_train, self.X_test = self._create_sequences(data_scaled[:split_idx]), self._create_sequences(data_scaled[split_idx:])
        
    def _create_sequences(self, data):
        """Create sequences of stock data for LSTM input."""
        X = []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, 0])
        return np.array(X).reshape(-1, self.sequence_length, 1)
    
    def build_lstm_model(self):
        """Build the LSTM model."""
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            LSTM(units=50),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def train(self, epochs=10, batch_size=32):
        """Train the LSTM model on the historical data."""
        self.model.fit(self.X_train, self.X_train[:, -1], epochs=epochs, batch_size=batch_size)
        
    def predict(self, X):
        """Make predictions with the LSTM model."""
        return self.model.predict(X)

class ConceptDriftDetector:
    def __init__(self, warning_level=2.0, drift_level=3.0, min_num_instances=25):
        config = DDMConfig(warning_level=warning_level, drift_level=drift_level, min_num_instances=min_num_instances)
        self.detector = DDM(config=config)
        self.drift_detected = False
        
    def detect_drift(self, y_true, y_pred):
        """Update the drift detector with new prediction errors."""
        error = mean_squared_error([y_true], [y_pred])
        self.detector.update(value=error)
        if self.detector.status["drift"]:
            self.drift_detected = True
            print("Concept drift detected!")
        return error

    def stream_test(self, X_current, y_current, pipeline):
        """Simulate data stream over X_current and y_current."""
        drift_flag = False
        y_true_list = []
        y_pred_list = []
        
        for i, (X, y) in enumerate(zip(X_current, y_current)):
            y_pred = pipeline.predict(X.reshape(1, -1, 1)).item()
            y_true_list.append(y)
            y_pred_list.append(y_pred)
            error = float(1 - (y_pred == y))  # Ensure error is a float
            self.detector.update(value=error)
            status = self.detector.status
            if status["drift"] and not drift_flag:
                drift_flag = True
                print(f"Concept drift detected at step {i}.")
        
        if not drift_flag:
            print("No concept drift detected")
        
        # Calculate regression metrics
        mse = mean_squared_error(y_true_list, y_pred_list)
        mae = mean_absolute_error(y_true_list, y_pred_list)
        r2 = r2_score(y_true_list, y_pred_list)
        
        print(f"Final Mean Squared Error (MSE): {mse:.4f}")
        print(f"Final Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Final R^2 Score: {r2:.4f}")

class StreamProcessor:
    def __init__(self, pipeline, detector):
        self.pipeline = pipeline
        self.detector = detector
        
    def run(self, X_current, y_current):
        """Run the stream test for concept drift detection."""
        self.detector.stream_test(X_current, y_current, self.pipeline)

# Example usage
if __name__ == "__main__":
    # Create and train the pipeline
    pipeline = StockDataPipeline(stock_symbol="AAPL")
    pipeline.fetch_data()
    pipeline.build_lstm_model()
    pipeline.train(epochs=5)
    
    # Create a drift detector
    detector = ConceptDriftDetector()
    
    # Get the current data
    X_current = pipeline.X_test
    y_current = pipeline.X_test[:, -1]
    
    # Run the stream processor for drift detection
    stream_processor = StreamProcessor(pipeline, detector)
    stream_processor.run(X_current, y_current)


# Output: