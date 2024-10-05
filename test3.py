from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yfinance as yf
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.tree import DecisionTreeClassifier
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.metrics.prequential_error import PrequentialError

# 1. Download Stock Data
data = yf.download('AAPL', start='2019-01-01', end='2023-01-01')
data['Return'] = data['Close'].pct_change()

# 2. Preprocessing: Create features (e.g., lagged returns) and labels
data.dropna(inplace=True)  # Drop missing values
X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
y = (data['Return'] > 0).astype(int).values  # Binary classification: Up or Down

# 3. Apply Scaling to the features (Standard Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scaling the features

# Shuffle and split data
concept_samples = len(X_scaled)
idx = np.arange(concept_samples)
np.random.shuffle(idx)
X_scaled, y = X_scaled[idx], y[idx].reshape(-1, 1)

split_idx = concept_samples // 2  # Half training, half warmup
X_train, y_train = X_scaled[:split_idx], y[:split_idx]
X_warmup, y_warmup = X_scaled[split_idx:], y[split_idx:]

# 4. Setup Drift Detection
config = DDMConfig(
    warning_level=2.0,
    drift_level=3.0,
    min_num_instances=len(X_warmup),
)

detector = DDM(config=config)
model = DecisionTreeClassifier(random_state=31)
model.fit(X_train, y_train)

metrics = [
    PrequentialError(alpha=alpha, name=f"α={alpha}")
    for alpha in [1.0, 0.9999, 0.999]
]
metrics_historic_detector = {f"{metric.name}": [] for metric in metrics}

def error_scorer(y_true, y_pred):  # Error function
    return 1 - (y_true.item() == y_pred.item())

# 5. Warm-up Detector
for X, y in zip(X_warmup, y_warmup):
    y_pred = model.predict(X.reshape(1, -1))
    error = error_scorer(y_true=y, y_pred=y_pred)
    _ = detector.update(value=error)
    
    for metric_historic, metric in zip(metrics_historic_detector.keys(), metrics):
        metrics_historic_detector[metric_historic].append(metric(error))

# 6. Test and Detect Drift
X_test, y_test = X_warmup, y_warmup  # Assuming the warmup data for simplicity
idx_drift, idx_warning = [], []
i = len(X_warmup)

for X, y in zip(X_test, y_test):
    y_pred = model.predict(X.reshape(1, -1))
    error = error_scorer(y_true=y, y_pred=y_pred)
    _ = detector.update(value=error)
    
    for metric_historic, metric in zip(metrics_historic_detector.keys(), metrics):
        metrics_historic_detector[metric_historic].append(metric(error))

    status = detector.status
    if status["drift"]:
        print(f"Drift detected at index: {i}")
        idx_drift.append(i)
        detector.reset()
        for metric in metrics:
            metric.reset()
        break
    elif status["warning"]:
        idx_warning.append(i)
    i += 1

# 7. Normalize data for better visualization (bring everything to the same range)
price_scaler = MinMaxScaler()
normalized_prices = price_scaler.fit_transform(data[['Close']])  # Normalizing stock prices

error_scaler = MinMaxScaler()
normalized_errors = {}
for metric_name, metric_values in metrics_historic_detector.items():
    normalized_errors[metric_name] = error_scaler.fit_transform(np.array(metric_values).reshape(-1, 1))

# 8. Enhanced Plotting with normalized range
plt.rcParams.update({"font.size": 14})
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])

# First plot: Normalized stock prices with drift and warning zones
ax1 = plt.subplot(gs[0])
ax1.plot(data.index, normalized_prices, label="Normalized Closing Price", color="blue")
ax1.set_ylabel("Normalized Stock Price")
ax1.set_title("AAPL Stock Prices (Normalized) with Drift/Warning Detection")

# Highlight warning and drift regions on the price plot
for idx in idx_warning:
    ax1.axvspan(data.index[idx], data.index[min(idx+10, len(data.index)-1)], color='yellow', alpha=0.3, label="Warning Zone")
for idx in idx_drift:
    ax1.axvspan(data.index[idx], data.index[min(idx+10, len(data.index)-1)], color='red', alpha=0.3, label="Drift Detected")

ax1.legend()

# Second plot: Normalized Prequential Errors
ax2 = plt.subplot(gs[1], sharex=ax1)
for metric_name, metric_values in normalized_errors.items():
    ax2.plot(metric_values, label=metric_name)
ax2.set_ylabel("Normalized Prequential Error")
ax2.legend(loc="upper left")
ax2.set_title("Normalized Prequential Error Over Time")

# Third plot: Detector Status (Warm-up, Warning, Drift)
ax3 = plt.subplot(gs[2], sharex=ax1)
ax3.set_ylabel("Detector Status")
ax3.set_yticks([])

# Mark the warm-up phase
warmup_color = "grey"
for idx in range(0, len(X_warmup)):
    ax3.axvline(x=idx, color=warmup_color, linewidth=1.0)

# Warning and drift lines in the detector status plot
for idx in idx_warning:
    ax3.axvline(x=idx, color='yellow', linewidth=2)
for idx in idx_drift:
    ax3.axvline(x=idx, color='red', linestyle='--', linewidth=2)

# Legends for drift and warning
drift_patch = mpatches.Patch(color='red', label='Drift Detected', alpha=0.5)
warning_patch = mpatches.Patch(color='yellow', label='Warning Zone', alpha=0.5)
ax3.legend(handles=[drift_patch, warning_patch], loc="upper left")

plt.tight_layout()
plt.show()
