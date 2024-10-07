import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')

# Extract the last 1000 data points
last_1000_data = data['close'].tail(1000)

# Calculate the moving average
window_size = 10
moving_average = last_1000_data.rolling(window=window_size).mean().dropna()

# Calculate the deviations from the moving average
deviations = last_1000_data[window_size-1:] - moving_average

# Print the deviations to check their range
print(deviations.describe())

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot the original data and the moving average
ax1.plot(last_1000_data.index, last_1000_data, label='Original Close Price', color='blue', alpha=0.5)
ax1.plot(moving_average.index, moving_average, label=f'{window_size}-Period Moving Average', color='orange')
ax1.set_title(f'EUR/USD - Close Price and {window_size}-Period Moving Average')
ax1.set_xlabel('Index')
ax1.set_ylabel('Price')
ax1.legend()

# Plot the deviations on a separate subplot
ax2.plot(deviations.index, deviations, label='Deviation from Moving Average', color='red')
ax2.set_title('Deviation from Moving Average')
ax2.set_xlabel('Index')
ax2.set_ylabel('Deviation')
ax2.legend()

# Adjust layout
plt.tight_layout()
plt.show()