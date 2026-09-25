import os
import glob
from plotting_utils import plot_csv_data

def main():
    """
    Finds all CSV files in the current directory and its subdirectories and plots them.
    """
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Search for all CSV files recursively
    csv_files = glob.glob(os.path.join(script_dir, '**', '*.csv'), recursive=True)

    if not csv_files:
        print("No CSV files found in the current directory or its subdirectories.")
        return

    print(f"Found {len(csv_files)} CSV files to analyze.")

    for filepath in csv_files:
        try:
            print(f"\n--- Processing file: {os.path.basename(filepath)} ---")
            plot_csv_data(filepath)
        except Exception as e:
            print(f"An error occurred while processing {filepath}: {e}")

if __name__ == "__main__":
    main()
