import numpy as np
import matplotlib.pyplot as plt

# Page-Hinkley test implementation
class PageHinkley:
    def __init__(self, threshold=50, min_num_instances=30, delta=0.01, lambda_=50):
        self.threshold = threshold  # Detection threshold
        self.min_num_instances = min_num_instances  # Minimum number of data points before detecting drift
        self.delta = delta  # Small positive value to detect changes
        self.lambda_ = lambda_  # A threshold parameter for drift detection
        
        # Variables to track mean and cumulative sum of deviations
        self.x_mean = 0
        self.sum = 0
        self.num_instances = 0
        self.drift_detected = False

    def update(self, x):
        self.num_instances += 1
        
        # Update mean and cumulative sum of deviations
        self.x_mean += (x - self.x_mean) / self.num_instances
        self.sum += x - self.x_mean - self.delta
        
        # Check if drift is detected
        if self.num_instances > self.min_num_instances and self.sum > self.lambda_:
            self.drift_detected = True
            return True
        return False

# Simulate some data with concept drift
np.random.seed(42)
n_samples = 200

# Data stream with two different concepts (before and after drift)
data_before_drift = np.random.normal(loc=0.5, scale=0.1, size=100)  # Before drift (mean=0.5)
data_after_drift = np.random.normal(loc=1.5, scale=0.1, size=100)   # After drift (mean=1.5)

# Combine into a single data stream
data_stream = np.concatenate([data_before_drift, data_after_drift])

# Apply the Page-Hinkley test to detect drift
ph = PageHinkley(threshold=50, lambda_=8)
drift_points = []

for i, data_point in enumerate(data_stream):
    if ph.update(data_point):
        drift_points.append(i)

# Plotting the data stream and the drift detection points
plt.figure(figsize=(10, 6))
plt.plot(data_stream, label="Data Stream")
plt.axvline(x=100, color='r', linestyle='--', label='True Drift Point')
for drift_point in drift_points:
    plt.axvline(x=drift_point, color='g', linestyle='--', label=f'Detected Drift (at {drift_point})')
plt.title('Concept Drift Detection using Page-Hinkley Test')
plt.xlabel('Time (Samples)')
plt.ylabel('Value')
plt.legend(loc="best")
plt.show()
