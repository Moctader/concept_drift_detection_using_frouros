import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Read the CSV file into a DataFrame
data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')

# Extract the last 1000 data points
last_1000_data = data['close'].tail(1000)

# Detrend the data using differencing
detrended_data = last_1000_data.diff().dropna()

# Split the detrended data into 10 segments of 100 data points each
segments = np.array_split(detrended_data, 5)

# Print the length and first content of each segment to verify
# for i, segment in enumerate(segments):
#     print(f"Segment {i+1} length: {len(segment)}, first content index: {segment.index[0]}")

# Initialize lists to store KS statistics and p-values
ks_stats = []
ks_p_values = []

# Create a figure for subplots
fig, axes = plt.subplots(len(segments) - 1, 1, figsize=(12, 6 * (len(segments) - 1)))

# Perform the KS test between consecutive segments and plot
for i in range(len(segments) - 1):
    ks_stat, ks_p_value = ks_2samp(segments[i], segments[i + 1])
    ks_stats.append(ks_stat)
    ks_p_values.append(ks_p_value)
    
    # Plot the segments
    axes[i].plot(segments[i].index, segments[i], label=f'Segment {i+1}')
    axes[i].plot(segments[i + 1].index, segments[i + 1], label=f'Segment {i+2}')
    axes[i].set_title(f'Segment {i+1} vs Segment {i+2} - KS p-value: {ks_p_value:.4f}')
    axes[i].set_xlabel('Index')
    axes[i].set_ylabel('D_Close')
    axes[i].legend()
    
    # Print if data drift is detected

# Adjust layout
plt.tight_layout()
plt.show()

if ks_p_value < 0.05:
    print(f"Data drift detected between Segment {i+1} and Segment {i+2} (p-value: {ks_p_value:.4f})")