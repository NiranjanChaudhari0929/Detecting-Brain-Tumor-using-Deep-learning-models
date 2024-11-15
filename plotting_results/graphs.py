import matplotlib.pyplot as plt

# Create a figure and a set of subplots
fig, ax1 = plt.subplots(figsize=(6, 4))

# Plot accuracy on the left y-axis with markers
ax1.plot(history.history['accuracy'], label='Training Accuracy', color='tab:blue', marker='o')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='tab:cyan', marker='s')
ax1.set_xlabel('Epochs', color='black', fontsize=10)
ax1.set_ylabel('Accuracy', color='black', fontsize=10)
ax1.tick_params(axis='y', labelcolor='black')
ax1.tick_params(axis='x', labelcolor='black')

# Create a second y-axis to plot the loss with markers
ax2 = ax1.twinx()
ax2.plot(history.history['loss'], label='Training Loss', color='tab:red', marker='^')
ax2.plot(history.history['val_loss'], label='Validation Loss', color='tab:orange', marker='d')
ax2.set_ylabel('Loss', color='black', fontsize=10)
ax2.tick_params(axis='y', labelcolor='black')

# Add title
fig.suptitle('Accuracy and Loss vs. Epochs for CNN', fontsize=12)

# Create a single legend for both plots
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
# Place the legend at the bottom of the plot area with appropriate font size
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Show the plot
plt.show()

