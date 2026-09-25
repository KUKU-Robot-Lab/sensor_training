import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

def get_all_csv_files(root_dir):
    """Finds all CSV files in the root directory and its subdirectories."""
    return glob.glob(os.path.join(root_dir, '**', '*.csv'), recursive=True)

def load_data(filepath):
    """Loads a CSV file into a pandas DataFrame and performs basic preprocessing."""
    try:
        df = pd.read_csv(filepath)
        if 'Motor_Position_mm' in df.columns:
            df.rename(columns={'Motor_Position_mm': 'Depth_mm'}, inplace=True)
        
        non_zero_data = df[df['Sensor_Raw_Value'] != 0]
        if not non_zero_data.empty:
            first_press_index = non_zero_data.index[0]
            temp_df = df.loc[first_press_index:]
            first_zero_indices_after_press = temp_df.index[temp_df['Sensor_Raw_Value'] == 0].tolist()
            
            if first_zero_indices_after_press:
                end_index = first_zero_indices_after_press[0]
                df = df.iloc[:end_index + 1]

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Elapsed_Time_s'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def plot_comparison(dataframes, names, plot_type):
    """Plots a comparison of all dataframes for a given plot type."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for df, name in zip(dataframes, names):
        if df is None:
            continue

        if plot_type == 'time_vs_polymer':
            ax.plot(df['Elapsed_Time_s'], df['Sensor_Raw_Value'], label=name)
        elif plot_type == 'time_vs_force':
            ax.plot(df['Elapsed_Time_s'], df['AFD_Fz'], label=name)
        elif plot_type == 'depth_vs_polymer':
            ax.scatter(df['Depth_mm'], df['Sensor_Raw_Value'], label=name, s=10)

    if plot_type == 'time_vs_polymer':
        ax.set_title('Comparison: Time vs. Polymer Sensor Value', fontsize=16)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Sensor Raw Value', fontsize=12)
    elif plot_type == 'time_vs_force':
        ax.set_title('Comparison: Time vs. AFD50 Sensor Value', fontsize=16)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Force (N)', fontsize=12)
    elif plot_type == 'depth_vs_polymer':
        ax.set_title('Comparison: Depth vs. Polymer Sensor Value', fontsize=16)
        ax.set_xlabel('Depth (mm)', fontsize=12)
        ax.set_ylabel('Sensor Raw Value', fontsize=12)

    ax.legend()
    ax.grid(True)
    output_filename = f"comparison_{plot_type}.png"
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"Comparison plot saved: {output_filename}")

def plot_3d_comparison(dataframes, names):
    """Plots a 3D comparison of all dataframes."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for df, name in zip(dataframes, names):
        if df is not None:
            ax.scatter(df['Elapsed_Time_s'], df['Depth_mm'], df['AFD_Fz'], label=name)

    ax.set_title('3D Plot Comparison: Time, Depth, and Force', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Depth (mm)', fontsize=12)
    ax.set_zlabel('Force (N)', fontsize=12)
    ax.legend()

    output_filename = "comparison_3d_plot.png"
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"Comparison plot saved: {output_filename}")

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = get_all_csv_files(root_dir)

    if not csv_files:
        print("No CSV files found.")
        return

    dataframes = []
    names = []
    for filepath in csv_files:
        df = load_data(filepath)
        if df is not None:
            dataframes.append(df)
            names.append(os.path.basename(os.path.dirname(filepath)))

    plot_comparison(dataframes, names, 'time_vs_polymer')
    plot_comparison(dataframes, names, 'time_vs_force')
    plot_comparison(dataframes, names, 'depth_vs_polymer')
    plot_3d_comparison(dataframes, names)

if __name__ == "__main__":
    main()