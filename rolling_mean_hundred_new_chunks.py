import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Read the CSV file into a DataFrame
data = pd.read_csv('EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')

# Extract the last 1000 data points
last_1000_data = data['close'].tail(1000)

# Calculate the moving average
window_size = 30
moving_average = last_1000_data.rolling(window=window_size).mean().dropna()

# Calculate the deviations from the moving average
deviations = last_1000_data[window_size-1:] - moving_average

# Split the deviations into 5 chunks
chunks = np.array_split(deviations, 5)

# Initialize lists to store KS statistics and p-values
ks_stats = []
ks_p_values = []

# Create a figure for subplots
fig, axes = plt.subplots(len(chunks) - 1, 1, figsize=(12, 6 * (len(chunks) - 1)))

# Perform the KS test between consecutive chunks and plot
for i in range(len(chunks) - 1):
    ks_stat, ks_p_value = ks_2samp(chunks[i], chunks[i + 1])
    ks_stats.append(ks_stat)
    ks_p_values.append(ks_p_value)
    
    # Plot the chunks with KS p-value in the label of the first chunk
    axes[i].plot(chunks[i].index, chunks[i], label=f'Chunk {i+1}, p-value: {ks_p_value:.4f}', color='blue')
    axes[i].plot(chunks[i + 1].index, chunks[i + 1], label=f'Chunk {i+2}', color='orange')
    axes[i].set_title(f'Comparison between Chunk {i+1} and Chunk {i+2}')
    axes[i].set_xlabel('Index')
    axes[i].set_ylabel('Deviation')
    axes[i].legend()
    
    # Print if data drift is detected
    if ks_p_value < 0.05:
        print(f"Data drift detected between Chunk {i+1} and Chunk {i+2} (p-value: {ks_p_value:.4f})")

# Adjust layout
plt.tight_layout()
plt.show()