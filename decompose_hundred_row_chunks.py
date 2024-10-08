import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from frouros.detectors.data_drift import KSTest
from statsmodels.tsa.seasonal import seasonal_decompose

# Read the CSV file into a DataFrame
data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')

# Extract the last 1000 data points
last_1000_data = data['close'].tail(1000)

# Decompose the stock data into trend, seasonal, and residual components
decomposition = seasonal_decompose(last_1000_data, model='additive', period=100)
residual = decomposition.resid.dropna()

# Split the residual component into 5 segments
segments = np.array_split(residual, 5)

# Initialize lists to store KS statistics and p-values
ks_stats = []
ks_p_values = []

# Create a figure for subplots
fig, axes = plt.subplots(len(segments) - 1, 1, figsize=(12, 6 * (len(segments) - 1)))

# Create an instance of KSTest
ks_test = KSTest()

# Perform the KS test between consecutive segments and plot
for i in range(len(segments) - 1):
    # Fit the KS test on the first segment
    ks_test.fit(segments[i].values.reshape(-1, 1))
    
    # Compare the next segment with the fitted KS test
    ks_result, _ = ks_test.compare(segments[i + 1].values.reshape(-1, 1))
    ks_stat = ks_result.statistic[0]
    ks_p_value = ks_result.p_value[0]
    
    ks_stats.append(ks_stat)
    ks_p_values.append(ks_p_value)
    
    # Plot the segments with KS p-value in the label of the first segment
    axes[i].plot(segments[i].index, segments[i], label=f'Segment {i+1}, p-value: {ks_p_value:.4f}', color='blue')
    axes[i].plot(segments[i + 1].index, segments[i + 1], label=f'Segment {i+2}', color='orange')
    axes[i].set_xlabel('Index')
    axes[i].set_ylabel('Residual')
    axes[i].legend()
    
    # Print if data drift is detected
    if ks_p_value < 0.05:
        print(f"Data drift detected between Segment {i+1} and Segment {i+2} (p-value: {ks_p_value:.4f})")

# Adjust layout
plt.tight_layout()
plt.show()