import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from frouros.detectors.concept_drift import DDM, DDMConfig

# Fetch and preprocess data
data = yf.download('AAPL', start='2018-01-01', end='2023-01-01')
data['Return'] = data['Close'].pct_change().dropna()

# Introduce manual drift
drift_point = len(data) // 2
data['Close'][drift_point:] += 100

# Prepare LSTM dataset
window_size = 30
X, y = [], []
for i in range(window_size, len(data)):
    X.append(data['Close'].values[i-window_size:i])
    y.append(data['Close'].values[i])
X, y = np.array(X), np.array(y)

# Normalize data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
y_scaled = scaler.transform(y.reshape(-1, 1))

# Split data into training and test sets
split_idx = int(0.7 * len(X_scaled))
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

# Reshape input for LSTM
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_reshaped, y_train, epochs=20, batch_size=64, validation_data=(X_test_reshaped, y_test))

# Drift detection setup
config = DDMConfig(warning_level=0.005, drift_level=0.01, min_num_instances=len(X_test_reshaped))
detector = DDM(config=config)

# Track model accuracy and drift
errors = []
ddm_warnings = []
ddm_drifts = []
for i in range(len(X_test_reshaped)):
    X_sample = X_test_reshaped[i].reshape(1, window_size, 1)
    y_true = y_test[i]
    y_pred = model.predict(X_sample)
    
    # Calculate error
    error = mean_squared_error(y_true, y_pred)
    errors.append(error)
    
    # Update drift detector
    detector.update(value=error)
    status = detector.status
    
    if status['warning']:
        ddm_warnings.append(i)
    if status['drift']:
        ddm_drifts.append(i)
        detector.reset()

# Inverse transform predicted and actual values for plotting
y_pred_scaled = model.predict(X_test_reshaped)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_original = scaler.inverse_transform(y_test)

# Plot the results
plt.figure(figsize=(12, 8))

# Plot normalized stock prices
plt.subplot(3, 1, 1)
plt.plot(range(len(y_test)), y_test_original, label='True Prices')
plt.plot(range(len(y_test)), y_pred, label='Predicted Prices', alpha=0.6)
plt.scatter(ddm_drifts, y_test_original[ddm_drifts], color='red', label='Drift Points', marker='x')
plt.axvline(x=drift_point - split_idx, color='blue', linestyle='--', label='Manual Drift Point')
plt.title('Stock Prices and Drift Detection')
plt.legend()

# Plot errors and drift/warning zones
plt.subplot(3, 1, 2)
plt.plot(errors, label='Prediction Error')
plt.scatter(ddm_warnings, [errors[i] for i in ddm_warnings], color='yellow', label='Warning Zone')
plt.scatter(ddm_drifts, [errors[i] for i in ddm_drifts], color='red', label='Drift Detected', marker='x')
plt.axhline(y=np.mean(errors), color='green', linestyle='--', label='Mean Error')
plt.title('Prediction Errors and Drift Detection')
plt.legend()

# Plot drift points over time
plt.subplot(3, 1, 3)
plt.plot(range(len(errors)), errors, label='Prediction Error')
plt.scatter(ddm_drifts, [errors[i] for i in ddm_drifts], color='red', label='Drift Points', marker='x')
plt.axvline(x=drift_point - split_idx, color='blue', linestyle='--', label='Manual Drift Point')
plt.title('Drift Points Over Time')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate drift detection performance
true_drifts = [drift_point - split_idx]
detected_drifts = ddm_drifts

# Calculate metrics
tp = len(set(true_drifts) & set(detected_drifts))
fp = len(set(detected_drifts) - set(true_drifts))
fn = len(set(true_drifts) - set(detected_drifts))
tn = len(errors) - tp - fp - fn

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")