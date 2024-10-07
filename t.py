
'''
When detecting data drift in stock market data, trends pose a challenge because they represent natural, systematic changes over time. Without accounting for these trends, drift detection methods may falsely identify drift even when the changes are part of an ongoing trend. To handle trends while checking for data drift, you need to detrend the data or use methods that are robust to trends.

Here’s how you can account for trends in stock market data when checking for data drift:

Approaches to Handling Trends in Data Drift Detection
1. Detrending the Data
Remove the trend from the data before applying drift detection methods. This ensures that you're only checking for changes in the residuals (i.e., deviations from the trend) rather than the trend itself.
Methods for Detrending:
Differencing: A common technique in time series analysis where you subtract the previous time step from the current time step to remove trends.
Moving Average: Apply a moving average to smooth out the trend and detect drift in deviations from the moving average.
Linear Regression Detrending: Fit a linear (or polynomial) regression line to the data and subtract the trend line from the original data to obtain the detrended data.


'''

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose

# Define stock ticker and date range
stock_ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2023-01-01"
stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

# Method 1: Calculate the rolling mean
window_size = 30
rolling_mean = stock_data['Close'].rolling(window=window_size).mean()
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], label='Original Close Price', color='blue', alpha=0.5)
plt.plot(rolling_mean.index, rolling_mean, label=f'{window_size}-Day Rolling Mean', color='orange')
plt.axvline(pd.to_datetime("2022-01-01"), color='red', linestyle='--', label='Split Date')
plt.title(f'{stock_ticker} - Close Price and {window_size}-Day Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()





# Method 2 : Detrend the data using differencing
detrended_data = stock_data['Close'].diff().dropna()
split_date = "2022-01-01"
reference_data = detrended_data[:split_date]
current_data = detrended_data[split_date:]

#  Use the KS test on the detrended data to detect drift
ks_stat, ks_p_value = ks_2samp(reference_data, current_data)

print(f"KS Statistic (detrended): {ks_stat}")
print(f"KS p-value (detrended): {ks_p_value}")

plt.figure(figsize=(10, 6))
plt.plot(detrended_data.index, detrended_data, label='Detrended Close Price')
plt.axvline(pd.to_datetime(split_date), color='r', linestyle='--', label='Split Date')
plt.title(f'{stock_ticker} Detrended Stock Price with Data Split')
plt.xlabel('Date')
plt.ylabel('Detrended Close Price')
plt.legend()
plt.show()

if ks_p_value < 0.05:
    print("Data drift detected in detrended data.")
else:
    print("No significant data drift detected in detrended data.")





# Method 3: Decompose the stock data into trend, seasonal, and residual components
decomposition = seasonal_decompose(stock_data['Close'], model='additive', period=30)
trend = decomposition.trend.dropna()
residual = decomposition.resid.dropna()

# Use the residual component for drift detection
reference_residual = residual[:split_date]
current_residual = residual[split_date:]

# KS test on residuals
ks_stat, ks_p_value = ks_2samp(reference_residual, current_residual)

print(f"KS Statistic (residual): {ks_stat}")
print(f"KS p-value (residual): {ks_p_value}")

# Visualize residuals
plt.figure(figsize=(10, 6))
plt.plot(residual.index, residual, label='Residuals')
plt.axvline(pd.to_datetime(split_date), color='r', linestyle='--', label='Split Date')
plt.title(f'{stock_ticker} Residuals with Data Split')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.legend()
plt.show()



if ks_p_value < 0.05:
    print("Data drift detected in detrended data.")
else:
    print("No significant data drift detected in detrended data.")