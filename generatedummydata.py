import pandas as pd
from datetime import timedelta

# Load the original data
file_path = 'your_file_path.csv'  # Replace with your actual file path
cars_data = pd.read_csv(file_path)

# Find the highest cars_id in the original data
max_cars_id = cars_data['cars_id'].max()

# Total number of duplications needed
total_duplications = 100

# Pre-compute the increments for cars_id for all duplications
cars_id_increments = [max_cars_id * (i + 1) for i in range(total_duplications)]

# Pre-compute the time increments for all duplications
time_increments = [timedelta(days=i+1, hours=i+1) for i in range(total_duplications)]

# Initialize an empty list to store all duplicated DataFrames
optimized_duplicated_data = []

# Perform duplication
for i in range(total_duplications):
    # Duplicate the data
    duplicate = cars_data.copy()

    # Increment cars_id and cars_dt_taken using vectorized operations
    duplicate['cars_id'] += cars_id_increments[i]
    duplicate['cars_dt_taken'] = pd.to_datetime(duplicate['cars_dt_taken']) + time_increments[i]

    # Append the modified duplicate to the list
    optimized_duplicated_data.append(duplicate)

# Combine all duplicated data
all_duplicated_combined_optimized = pd.concat(optimized_duplicated_data, ignore_index=True)

# Combine the original data with all duplicated data
final_combined_data_optimized = pd.concat([cars_data, all_duplicated_combined_optimized], ignore_index=True)

# Save the final combined data to a new CSV file
final_combined_file_path_optimized = 'final_combined_cars_data_optimized.csv'
final_combined_data_optimized.to_csv(final_combined_file_path_optimized, index=False)
