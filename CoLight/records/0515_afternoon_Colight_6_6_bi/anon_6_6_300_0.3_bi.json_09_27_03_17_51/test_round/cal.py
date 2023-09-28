import pandas as pd

# Load the CSV file into a DataFrame
# df = pd.read_csv('/mnt/data/vehicle_inter_30.csv')

def compute_time_difference_for_file(file_path):
    """Compute the total time difference for a given CSV file."""
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Filter out rows with NaN values in either 'enter_time' or 'leave_time' columns
    filtered_df = df.dropna(subset=['enter_time', 'leave_time'])
    
    # Calculate the difference between 'leave_time' and 'enter_time' for each row
    # time_difference = (filtered_df['leave_time'] - filtered_df['enter_time']).sum()
    
    first_200_rows = filtered_df.iloc[:200]
    time_difference = (first_200_rows['leave_time'] - first_200_rows['enter_time']).sum()
    
    return time_difference

for j in range(100):

    # List of file names from 0 to 35
    file_names = [f'./round_{j}/vehicle_inter_{i}.csv' for i in range(36)]

    # Sum time differences across all files
    total_time_difference_all_files = sum(compute_time_difference_for_file(file_name) for file_name in file_names)

    print(total_time_difference_all_files)

