import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import find_peaks

def load_accelerometer_data(filepath):
    """
    Load accelerometer data from CSV file
    
    Parameters:
    filepath (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Loaded accelerometer data
    """
    # Read the CSV file using the correct delimiter
    df = pd.read_csv(filepath, header=None, delimiter=';', names=['timestamp', 'x', 'y', 'z'])
    
    # Debugging: Print the first few rows
    print("\n=== Data Preview ===")
    print(df.head())
    
    try:
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'].str.strip("'"), format='%d-%b-%Y %H:%M:%S.%f')
    except ValueError as e:
        print("\nError converting timestamps:", e)
        print("\nSample problematic data:")
        print(df['timestamp'].head())
        raise
    
    # Calculate time elapsed in seconds
    df['time_elapsed'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    
    return df


def analyze_accelerometer_data(df):
    """
    Perform comprehensive analysis of accelerometer data
    
    Parameters:
    df (pandas.DataFrame): Accelerometer data
    
    Returns:
    dict: Analysis results
    """
    # Calculate total acceleration
    df['total_acceleration'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    
    # Basic statistics
    analysis = {
        'duration': df['time_elapsed'].max(),
        'axes_stats': {
            'x': {
                'mean': df['x'].mean(),
                'std': df['x'].std(),
                'min': df['x'].min(),
                'max': df['x'].max()
            },
            'y': {
                'mean': df['y'].mean(),
                'std': df['y'].std(),
                'min': df['y'].min(),
                'max': df['y'].max()
            },
            'z': {
                'mean': df['z'].mean(),
                'std': df['z'].std(),
                'min': df['z'].min(),
                'max': df['z'].max()
            }
        },
        'total_acceleration': {
            'mean': df['total_acceleration'].mean(),
            'std': df['total_acceleration'].std(),
            'min': df['total_acceleration'].min(),
            'max': df['total_acceleration'].max()
        }
    }
    
    return analysis

def plot_accelerometer_data(df):
    """
    Create comprehensive plots of accelerometer data
    
    Parameters:
    df (pandas.DataFrame): Accelerometer data
    """
    # Create a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Accelerometer Data Analysis', fontsize=16)
    
    # Plot individual axis data
    axs[0, 0].plot(df['time_elapsed'], df['x'], label='X-axis', color='red')
    axs[0, 0].set_title('X-axis Acceleration')
    axs[0, 0].set_xlabel('Time (seconds)')
    axs[0, 0].set_ylabel('Acceleration (m/s²)')
    axs[0, 0].legend()
    
    axs[0, 1].plot(df['time_elapsed'], df['y'], label='Y-axis', color='green')
    axs[0, 1].set_title('Y-axis Acceleration')
    axs[0, 1].set_xlabel('Time (seconds)')
    axs[0, 1].set_ylabel('Acceleration (m/s²)')
    axs[0, 1].legend()
    
    axs[1, 0].plot(df['time_elapsed'], df['z'], label='Z-axis', color='blue')
    axs[1, 0].set_title('Z-axis Acceleration')
    axs[1, 0].set_xlabel('Time (seconds)')
    axs[1, 0].set_ylabel('Acceleration (m/s²)')
    axs[1, 0].legend()
    
    # Plot total acceleration
    total_accel = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    axs[1, 1].plot(df['time_elapsed'], total_accel, label='Total Acceleration', color='purple')
    axs[1, 1].set_title('Total Acceleration')
    axs[1, 1].set_xlabel('Time (seconds)')
    axs[1, 1].set_ylabel('Acceleration Magnitude (m/s²)')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def detect_steps(df, height_threshold=1.0, distance=20):
    """
    Detect steps from accelerometer data based on peak analysis.

    Parameters:
    df (pandas.DataFrame): Accelerometer data with `total_acceleration`.
    height_threshold (float): Minimum height for peaks to be considered as steps.
    distance (int): Minimum number of samples between consecutive peaks.

    Returns:
    int: Number of steps detected.
    """
    # Smooth the total acceleration signal using a rolling mean
    df['smoothed_acceleration'] = df['total_acceleration'].rolling(window=5, center=True).mean()
    
    # Detect peaks in the smoothed signal
    peaks, _ = find_peaks(df['smoothed_acceleration'], height=height_threshold, distance=distance)
    
    # Count peaks as steps
    step_count = len(peaks)
    
    return step_count, peaks

def main(filepath):
    """
    Main function to process accelerometer data and count steps.
    
    Parameters:
    filepath (str): Path to the CSV file.
    """
    # Load data
    df = load_accelerometer_data(filepath)
    
    # Analyze data
    analysis = analyze_accelerometer_data(df)
    
    # Print analysis results
    print("\n=== Accelerometer Data Analysis ===")
    print(f"Data Duration: {analysis['duration']:.2f} seconds")
    
    print("\nAxis Statistics:")
    for axis in ['x', 'y', 'z']:
        print(f"\n{axis.upper()}-axis:")
        for stat, value in analysis['axes_stats'][axis].items():
            print(f"  {stat.capitalize()}: {value:.4f}")
    
    print("\nTotal Acceleration:")
    for stat, value in analysis['total_acceleration'].items():
        print(f"  {stat.capitalize()}: {value:.4f}")
    
    # Detect and count steps
    step_count, peaks = detect_steps(df)
    print(f"\nTotal Steps Detected: {step_count}")
    
    # Plot data with detected steps
    plot_accelerometer_data_with_steps(df, peaks)

def plot_accelerometer_data_with_steps(df, peaks):
    """
    Plot accelerometer data with detected steps.
    
    Parameters:
    df (pandas.DataFrame): Accelerometer data.
    peaks (list): Indices of detected steps.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['time_elapsed'], df['smoothed_acceleration'], label='Smoothed Total Acceleration')
    plt.scatter(df['time_elapsed'].iloc[peaks], df['smoothed_acceleration'].iloc[peaks], 
                color='red', label='Detected Steps', zorder=5)
    plt.title('Step Detection via Peak Analysis')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration Magnitude (m/s²)')
    plt.legend()
    plt.grid()
    plt.show()

# Run the script
if __name__ == "__main__":
    filepath = 'sensorlog.csv'  # Update with your CSV file path
    main(filepath)