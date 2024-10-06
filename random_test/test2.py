import numpy as np
import matplotlib.pyplot as plt
from frouros.detectors.concept_drift import ADWIN

# Set a seed for reproducibility
np.random.seed(42)

# Generate synthetic stock-like data with multiple known drifts
data_segment_1 = np.cumsum(np.random.normal(loc=0.5, scale=0.1, size=300))  # First segment
data_segment_2 = np.cumsum(np.random.normal(loc=0.8, scale=0.1, size=300)) + data_segment_1[-1]  # Second segment
data_segment_3 = np.cumsum(np.random.normal(loc=0.3, scale=0.1, size=300)) + data_segment_2[-1]  # Third segment
data_segment_4 = np.cumsum(np.random.normal(loc=0.6, scale=0.1, size=300)) + data_segment_3[-1]  # Fourth segment

# Combine all segments to form the synthetic dataset
synthetic_data = np.concatenate((data_segment_1, data_segment_2, data_segment_3, data_segment_4))

# Known drift points (start of new data segments)
known_drift_points = [300, 600, 900]

# Plot the synthetic data with known drift points
plt.figure(figsize=(12, 6))
plt.plot(synthetic_data, color='blue', label='Synthetic Data')
for i, point in enumerate(known_drift_points):
    if i == 0:
        plt.axvline(x=point, color='green', linestyle='--', label='Known Drift Point')
    else:
        plt.axvline(x=point, color='green', linestyle='--')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Synthetic Stock-like Data with Known Drift Points')
plt.legend()
#plt.show()

# Instantiate ADWIN drift detector with tuned delta parameter
detector = ADWIN()  # Adjust delta to reduce sensitivity

# List to store detected drift indices
drift_indices = []

# Iterate over the synthetic data and update the detector with new values
for i, value in enumerate(synthetic_data):
    detector.update(value=value)  # Update the detector with the current value
    
    if detector.drift:  # If drift is detected
        drift_indices.append(i)  # Store the index where drift is detected
        detector.reset()  # Reset the detector after detecting a drift

# Plot the synthetic data with both known and detected drifts
plt.figure(figsize=(12, 6))
plt.plot(synthetic_data, color='blue', label='Synthetic Data')
# Plot known drift points
for i, point in enumerate(known_drift_points):
    if i == 0:
        plt.axvline(x=point, color='green', linestyle='--', label='Known Drift Point')
    else:
        plt.axvline(x=point, color='green', linestyle='--')
# Plot detected drift points
plt.scatter(drift_indices, [synthetic_data[i] for i in drift_indices], color='red', label='Detected Drift', zorder=5)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('ADWIN Drift Detection in Synthetic Stock-like Data')
plt.legend()
plt.show()