from utils import BagReader, get_mean_values_from_bag
import pandas as pd
import os

bags_folder = "/home/docker/ros_ws/data/bags/raw/filtered_bags/"
file_name = "KATE_BO_U010_bayesian_optimisation_trial_9_force_1_10_2025-10-29-12-14-46_cutted.bag"
force_index = 8

df_mean = pd.DataFrame()

file_path = os.path.join(bags_folder, file_name)
if os.path.isfile(file_path):
    print(f"Processing file: {file_path}")
    bag_reader = BagReader()
    mean_values = get_mean_values_from_bag(file_path, bag_reader)
    print(f"Mean values: {mean_values}")
    if "initial" in file_name:
        force = 50
    else:    
        force = int(file_name.split('_')[force_index])
    print(f"Extracted force: {force}")
    row_data = {'File': file_name, 'Force': force}
    row_data.update(mean_values)  # Add all key-value pairs from mean_values to row_data
    print(row_data)
    df_mean = pd.concat([df_mean, pd.DataFrame([row_data])], ignore_index=True)

print(df_mean.loc[:, ['File', 'Force', 'wrench_force_y']])