import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch AAPL stock data for the last year
data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')

# Calculate required indicators
data['PX_LOW'] = data['Low']  # Minimum daily price
data['AVG_60'] = data['PX_LOW'].rolling(window=60).mean()  # 60-day moving average

# Clean the dataset by dropping rows with NaN values
data = data.dropna()

# Sliding Window Parameters
block_size = 100
drift_buffer_size = 15

# Prepare to store drift error results
drift_errors = []

# Simulate the processing of the stream with a sliding window
for i in range(block_size, len(data)):
    # Get the current window
    window = data.iloc[i - block_size:i]
    
    # Example error calculation: difference between PX_LOW and its 60-day average
    error = window['PX_LOW'].iloc[-1] - window['AVG_60'].iloc[-1]
    drift_errors.append(error)

# Create a DataFrame for drift error results
drift_results = pd.DataFrame({
    'Date': data.index[block_size:],
    'Drift Error': drift_errors
})

# Apply a moving average to the drift error for smoothing
drift_results['Smoothed Drift'] = drift_results['Drift Error'].rolling(window=drift_buffer_size).mean()

# Calculate the drift trend
drift_results['Drift Trend'] = drift_results['Drift Error'] - drift_results['Smoothed Drift']

# Plotting the results in subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 18))

# a) Cleaned drift curve and its moving average
axes[0].plot(drift_results['Date'], drift_results['Drift Error'], label='Drift Error', alpha=0.5, color='blue')
axes[0].plot(drift_results['Date'], drift_results['Smoothed Drift'], label='Smoothed Drift (15-day MA)', color='orange', linewidth=2)
axes[0].set_title('a) Cleaned Drift Curve and its Moving Average', fontsize=16)
axes[0].set_ylabel('Drift Error', fontsize=14)
axes[0].legend()
axes[0].grid()

# b) The trend drift curve
axes[1].plot(drift_results['Date'], drift_results['Drift Trend'], label='Trend Drift', color='purple', linewidth=2)
axes[1].axhline(0, color='red', linestyle='--', label='Zero Line')
axes[1].set_title('b) The Trend Drift Curve', fontsize=16)
axes[1].set_ylabel('Drift Trend', fontsize=14)
axes[1].legend()
axes[1].grid()

# c) The PX LOW time series
axes[2].plot(data.index, data['PX_LOW'], label='PX LOW (Minimum Daily Price)', color='green')
axes[2].set_title('c) The AAPL PX LOW Time Series', fontsize=16)
axes[2].set_ylabel('PX LOW Price', fontsize=14)
axes[2].legend()
axes[2].grid()

# Show the plots
plt.tight_layout()
plt.show()
