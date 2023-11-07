import pandas as pd
import os

class IntersectionTimeCalculator:
    def __init__(self, directory):
        self.directory = directory

    def calculate_time_differences(self):
        total_sum = 0

        # Loop over the files in the directory
        for i in range(16):  # Assuming you have files for intersections 0 to 15
            file_name = f'vehicle_inter_{i}.csv'
            file_path = os.path.join(self.directory, file_name)

            # Check if the file exists
            if os.path.isfile(file_path):
                # Load the data
                vehicle_data = pd.read_csv(file_path)

                # Calculate the sum of the differences for this file
                sum_difference = (vehicle_data['leave_time'] - vehicle_data['enter_time']).sum()
                total_sum += sum_difference
                print(f'Sum for {file_name}: {sum_difference}')
            else:
                print(f'File {file_name} does not exist.')

        return total_sum

