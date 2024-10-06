import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler

# Fetch stock data using yfinance
def fetch_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data['Close']

# Define stock symbol and date ranges
ticker = 'AAPL'
reference_start = '2020-01-01'
reference_end = '2020-12-31'
current_start = '2023-01-01'
current_end = '2023-12-31'

# Fetch stock data
reference_data = fetch_stock_data(ticker, reference_start, reference_end)
current_data = fetch_stock_data(ticker, current_start, current_end)

# Drop missing values
reference_data = reference_data.dropna()
current_data = current_data.dropna()

# Perform KS test to check for drift
def ks_test(reference_data, current_data):
    statistic, p_value = ks_2samp(reference_data, current_data)
    return {
        "statistic": statistic,
        "p_value": p_value
    }

# Run the KS test
alpha = 0.05  # Significance level
ks_test_result = ks_test(reference_data, current_data)

# Visualization
def plot_histograms(reference_data, current_data, ks_test_result, alpha=0.05):
    # Scale the data to the same range
    scaler = MinMaxScaler()
    reference_data_scaled = scaler.fit_transform(reference_data.values.reshape(-1, 1)).flatten()
    current_data_scaled = scaler.transform(current_data.values.reshape(-1, 1)).flatten()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)  # Adjusted figure size and DPI

    # Plot histograms of reference and current data
    ax.hist(reference_data_scaled, bins=30, alpha=0.5, label="Reference Data", color="blue", edgecolor='black')
    ax.hist(current_data_scaled, bins=30, alpha=0.5, label="Current Data", color="green", edgecolor='black')

    # KS statistic and p-value
    statistic = ks_test_result["statistic"]
    p_value = ks_test_result["p_value"]

    # Drift decision
    drift = p_value <= alpha
    drift_str = (
        f"Drift detected\np-value = {p_value:.4f}"
        if drift
        else f"No drift detected\np-value = {p_value:.4f}"
    )
    ax.text(0.7, 0.875, drift_str, transform=ax.transAxes, fontsize=10, bbox={
        "boxstyle": "round", "facecolor": "red" if drift else "green", "alpha": 0.5,
    })

    ax.legend()
    ax.set_title("Histogram of Scaled Stock Data with KS Test Results")
    ax.set_xlabel("Scaled Stock Price")
    ax.set_ylabel("Frequency")
    plt.show()

# Plot the histograms with KS test results
plot_histograms(reference_data, current_data, ks_test_result, alpha=alpha)