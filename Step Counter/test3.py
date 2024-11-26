import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset 1
def load_dataset_1(path):
    df = pd.read_csv(path, header=None, names=['timestamp', 'id', 'x', 'y', 'z'])
    df = df.iloc[:5000] #limit the data to the first 5000 points
    df['time_elapsed'] = (df['timestamp'] - df['timestamp'].min()) / 1e9  # Convert nanoseconds to seconds
    df['acceleration_magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    df['acceleration_normalised'] = df['acceleration_magnitude'] - 9.81
    return df

# Load dataset 2
def load_dataset_2(path):
    df = pd.read_csv(path, delimiter=';', header=None, names=['timestamp', 'x', 'y', 'z'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].str.strip("'"), format='%d-%b-%Y %H:%M:%S.%f')
    df['time_elapsed'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()  # Convert to seconds
    df['acceleration_magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    df['acceleration_normalised'] = df['acceleration_magnitude'] - 9.81
    return df

# Compare datasets
def compare_datasets(df1, df2):
    plt.figure(figsize=(15, 8))
    
    # Plot dataset 1
    plt.plot(df1['time_elapsed'], df1['acceleration_normalised'], label='Dataset 1', alpha=0.7)
    
    # Plot dataset 2
    plt.plot(df2['time_elapsed'], df2['acceleration_normalised'], label='Dataset 2', alpha=0.7)
    
    # Add labels, title, and legend
    plt.xlabel('Time Elapsed (seconds)')
    plt.ylabel('Normalised Acceleration (m/s²)')
    plt.title('Comparison of Normalised Accelerometer Data')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# File paths
path_dataset_1 = 'accelerometer.csv'
path_dataset_2 = 'sensorlog.csv'

# Load datasets
df1 = load_dataset_1(path_dataset_1)
df2 = load_dataset_2(path_dataset_2)

# Compare datasets
compare_datasets(df1, df2)
