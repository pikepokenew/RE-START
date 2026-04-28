import matplotlib.pyplot as plt
import numpy as np

# Data from the table
# N represents the number of samples (Best-of-N)
n_samples = np.array([1, 2, 4, 8, 16])

# Unsafe response rates in percentages
unsafe_rates_percent = np.array([9.6, 4.35, 1.9, 0.85, 0.3])

# Convert percentages to a decimal format for plotting
unsafe_rates = unsafe_rates_percent

# Create the plot
plt.figure(figsize=(8, 6))

# Plot the data as a line with markers
plt.plot(n_samples, unsafe_rates, marker='o', linestyle='-', color='b', label='Unsafe Response Rate')

# Set the y-axis to a logarithmic scale for better visualization of exponential decay
# This is a common practice for data that decreases rapidly.
plt.yscale('log')

# Add a title and labels
plt.title('Impact of Best-of-N Sampling on Unsafe Response Rate')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Unsafe Response Rate')

# Customize the plot
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.xticks(n_samples)  # Ensure only the given N values are shown on the x-axis

# Add text labels for each data point
for i, txt in enumerate(unsafe_rates_percent):
    # Adjust position slightly for better readability
    plt.annotate(f'{txt}%', (n_samples[i], unsafe_rates[i]), textcoords="offset points", xytext=(0,10), ha='center')

# Add a legend
plt.legend()

# Save the plot to a high-quality PDF file, suitable for a paper
plt.savefig('unsafe_response_rate_plot.pdf', bbox_inches='tight')

# Display the plot
# plt.show()
plt.save("Best-of-N.py")

print("Plot saved as 'unsafe_response_rate_plot.pdf'")
