import os
import re
from pathlib import Path

import pandas as pd

from robotrainer_bayesian_optimization.utils import (
    BagReader,
    get_mean_values_from_bag,
    normalize_raw_values,
)


def get_data_root():
    """Get data root path - auto-detects Docker vs local environment."""
    docker_path = Path("/home/docker/ros_ws/data")
    local_path = Path("../../data").resolve()
    
    
    # Use Docker path if it exists, otherwise use local path
    if docker_path.exists():
        return docker_path
    return local_path


def extract_force(filename):
    match = re.search(r"force_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None

def extract_experiment_name(filename):
    match = re.search(r"experiment_(\d+)", filename)
    if match:
        return match.group(0)
    return None


def process_bags_in_folder(bags_folder):
    print(f"Processing bags in folder: {bags_folder}")
    df_mean = pd.DataFrame()
    for file_name in os.listdir(bags_folder):
        file_path = os.path.join(bags_folder, file_name)
        if os.path.isfile(file_path):
            print(f"Processing file: {file_path}")
            bag_reader = BagReader()
            mean_values = get_mean_values_from_bag(file_path, bag_reader)
            # force = extract_force(file_name)
            # print(f"Extracted force: {force}")
            # row_data = {"File": file_name, "force": force}
            experiment_name = extract_experiment_name(file_name)
            print(f"Extracted experiment name: {experiment_name}")
            row_data = {"File": file_name, "experiment": experiment_name}
            row_data.update(
                mean_values
            )  # Add all key-value pairs from mean_values to row_data
            print(row_data)
            df_mean = pd.concat([df_mean, pd.DataFrame([row_data])], ignore_index=True)
    return df_mean


def main():
    data_root = get_data_root()

    user = "U020"  # Change this to the appropriate user ID

    bags_folder = str(data_root / "consistency_test" / user / "bags" / "raw")

    df_mean = process_bags_in_folder(bags_folder)
    df_mean = normalize_raw_values(df_mean)

    # Save the DataFrame to a CSV file
    data_path = data_root / "consistency_test" / user
    data_path.mkdir(parents=True, exist_ok=True)
    output_csv_path = data_path / f"mean_values_KATE_BO_{user}.csv"
    df_mean.to_csv(output_csv_path)
    print(f"mean values saved to {output_csv_path}")

if __name__ == "__main__":
    main()
