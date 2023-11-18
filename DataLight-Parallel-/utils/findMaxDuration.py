import pandas as pd

# Load the data into a DataFrame
df = pd.read_csv(file_path, header=None, names=['Time', 'Action'])

# Pair each row with the next one
df['Next_Time'] = df['Time'].shift(-1)
df['Next_Action'] = df['Action'].shift(-1)

# Filter out the pairs where the action is not the same
same_action_pairs = df[df['Action'] == df['Next_Action']]

# Calculate the duration for these pairs
same_action_pairs['Duration'] = same_action_pairs['Next_Time'] - same_action_pairs['Time']

# Filter out pairs where the duration is less than or equal to 50.0 seconds
filtered_pairs = same_action_pairs[same_action_pairs['Duration'] > 30.0]

# We don't need the 'Next_Time' and 'Next_Action' columns anymore
filtered_pairs = filtered_pairs[['Time', 'Action', 'Duration']]

filtered_pairs.reset_index(drop=True, inplace=True)
filtered_pairs
