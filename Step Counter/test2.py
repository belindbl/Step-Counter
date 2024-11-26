import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Define the file path
path2 = 'sensorlog.csv'
path1 = 'accelerometer.csv'

path = path2

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(path, header=None, names=['timestamp', 'id', 'x', 'y', 'z'])

# Convert the timestamp to relative time (elapsed seconds)
#df['time_elapsed'] = (df['timestamp'] - df['timestamp'].min()) / 1e6  # Adjust scale if needed

# Calculate total acceleration magnitude
df['acceleration_magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

# Normalise the total acceleration (subtract 9.81 m/s²)
df['acceleration_normalised'] = df['acceleration_magnitude'] - 9.81

# Select the first 1000 rows for analysis
df_subset = df.iloc[:100]

# Detect peaks with a minimum height
peaks, _ = find_peaks(df_subset['acceleration_normalised'], height=0.5, distance=20)

# Ensure valleys between peaks
valid_peaks = []
for i in range(1, len(peaks)):
    # Check if there's a valley (negative acceleration) between consecutive peaks
    start, end = peaks[i - 1], peaks[i]
    if df_subset['acceleration_normalised'].iloc[start:end].min() < 0:
        valid_peaks.append(peaks[i])

# Plot the normalised acceleration with detected steps
plt.figure(figsize=(15, 8))
plt.plot(df_subset['time_elapsed'], df_subset['acceleration_normalised'], label='Normalised Acceleration', color='blue', alpha=0.7)
plt.scatter(df_subset['time_elapsed'].iloc[valid_peaks], df_subset['acceleration_normalised'].iloc[valid_peaks], color='red', label='Detected Steps', zorder=5)

# Add labels, title, and legend
plt.xlabel('Time Elapsed (seconds)')
plt.ylabel('Normalised Acceleration (m/s²)')
plt.title('Normalised Acceleration with Valid Steps')
plt.legend()
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()

# Print the number of detected steps
print(f"Total Steps Detected: {len(valid_peaks)}")
