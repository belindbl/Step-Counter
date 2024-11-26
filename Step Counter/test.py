import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the file path
path = 'accelerometer.csv'

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(path, header=None, names=['timestamp', 'id', 'x', 'y', 'z'])

# Convert the timestamp to relative time (elapsed seconds)
df['time_elapsed'] = (df['timestamp'] - df['timestamp'].min()) / 1e6  # Adjust scale if needed

# Calculate total acceleration magnitude
df['acceleration_magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

# Select the first 1000 rows
df_subset = df.iloc[:500]

# Plot the total acceleration magnitude
plt.figure(figsize=(15, 8))
plt.plot(df_subset['time_elapsed'], df_subset['acceleration_magnitude'], label='Total Acceleration Magnitude', color='purple', alpha=0.8)

# Add labels, title, and legend
plt.xlabel('Time Elapsed (seconds)')
plt.ylabel('Acceleration Magnitude (m/s²)')
plt.title('Total Acceleration Magnitude (First 1000 Rows)')
plt.legend()
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()
