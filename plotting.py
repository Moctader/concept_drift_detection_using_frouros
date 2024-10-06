from matplotlib import pyplot as plt

# Plot the results
def plot_results(y_true_list, y_pred_list, drift_point):
    """Plot the true values, predicted values, and concept drift point."""
    fig, axs = plt.subplots(3, 1, figsize=(14, 15))

    # Plot true values
    axs[0].plot(y_true_list, color='g', label='True Values', linewidth=2, marker='o', markersize=4, alpha=0.7)
    axs[0].set_title('True Values', fontsize=16)
    axs[0].set_xlabel('Time Step', fontsize=14)
    axs[0].set_ylabel('Value', fontsize=14)
    axs[0].grid(True)
    axs[0].legend(fontsize=12)

    # Plot predicted values
    axs[1].plot(y_pred_list, color='y', label='Predicted Values', linewidth=2, marker='x', markersize=4, alpha=0.7)
    axs[1].set_title('Predicted Values', fontsize=16)
    axs[1].set_xlabel('Time Step', fontsize=14)
    axs[1].set_ylabel('Value', fontsize=14)
    axs[1].grid(True)
    axs[1].legend(fontsize=12)

    # Combined plot with drift point
    axs[2].plot(y_true_list, color='g', label='True Values', linewidth=2, marker='o', markersize=4, alpha=0.7)
    axs[2].plot(y_pred_list, color='y', label='Predicted Values', linewidth=2, marker='x', markersize=4, alpha=0.7)
    if drift_point is not None:
        axs[2].axvline(x=drift_point, color='r', linestyle='--', label='Concept Drift', linewidth=2, alpha=0.7)
    axs[2].set_title('True vs Predicted Values with Concept Drift', fontsize=16)
    axs[2].set_xlabel('Time Step', fontsize=14)
    axs[2].set_ylabel('Value', fontsize=14)
    axs[2].grid(True)
    axs[2].legend(fontsize=12)

    plt.tight_layout()
    plt.show()





def plot_target_drift(p_values):
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, label='p-value')
    # plt.axhline(target_drift_threshold, color='r', linestyle='--', label='Threshold')
    # plt.xlabel('Steps')
    plt.ylabel('p-value')
    plt.title('Target Drift Detection')
    plt.legend()
    plt.show()

    