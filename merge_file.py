import os
import pickle
import numpy as np
import re

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def merge_pkl_files(folder):
    # Get all .pkl files in the folder
    pkl_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pkl')]

    if not pkl_files:
        print("No .pkl files found in the folder!")
        return

    # Load the first dataset to initialize the structure
    merged_data = load_data(pkl_files[0])
    keys = merged_data.keys()

    # Iterate through the remaining files and merge them
    for file in pkl_files[1:]:
        data = load_data(file)

        # Ensure the structure matches
        assert data.keys() == keys, f"Dataset structure mismatch in {file}!"

        # Concatenate along axis 0 if it's a numpy array, else keep the first value
        for key in keys:
            if isinstance(merged_data[key], np.ndarray):
                merged_data[key] = np.concatenate((merged_data[key], data[key]), axis=0)
            elif isinstance(merged_data[key], list):
                # Convert both lists to numpy arrays before concatenation
                merged_data[key] = np.concatenate((np.array(merged_data[key]), np.array(data[key])), axis=0)

    # Save the merged dataset
    output_filename = os.path.join(folder, 'merged_dataset.pkl')
    with open(output_filename, 'wb') as f:
        pickle.dump(merged_data, f)

    print(f'Merged dataset saved as {output_filename}')


def merge_pkl_files_by_prefix(folder, prefix):
    """
    Merge all .pkl files with the same prefix in the given folder.

    Args:
        folder (str): Path to the folder containing .pkl files.
        prefix (str): The prefix to match (e.g., "10AP_2UE_2SR_").

    Returns:
        None (saves merged .pkl file).
    """
    # Get all files that start with the exact prefix
    pkl_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.pkl')]

    if not pkl_files:
        print(f"No matching .pkl files found with prefix '{prefix}'!")
        return

    # Sort numerically by extracting the number part (assuming format: prefix_xxx.pkl)
    pkl_files.sort(key=lambda x: int(x.split("_")[-1].replace(".pkl", "")))

    # Load the first dataset to initialize the structure
    merged_data = load_data(pkl_files[0])
    keys = merged_data.keys()

    # Iterate through the remaining files and merge them
    for file in pkl_files[1:]:
        data = load_data(file)

        # Ensure the structure matches
        assert data.keys() == keys, f"Dataset structure mismatch in {file}!"

        # Concatenate along axis 0 if it's a numpy array or list
        for key in keys:
            if isinstance(merged_data[key], np.ndarray):
                merged_data[key] = np.concatenate((merged_data[key], data[key]), axis=0)
            elif isinstance(merged_data[key], list):
                merged_data[key] = np.concatenate((np.array(merged_data[key]), np.array(data[key])), axis=0)

    # Save the merged dataset
    output_filename = os.path.join(folder, f'eval_{prefix[:-1]}.pkl')  # Remove trailing "_"
    with open(output_filename, 'wb') as f:
        pickle.dump(merged_data, f)

    print(f'Merged dataset saved as {output_filename}')

# Example usage
if __name__ == "__main__":
    # folder = "Data/12Mar"
    # merge_pkl_files(folder)
    folder = "Data/17Mar"
    prefix_pattern = "10AP_2UE_2SR_"  # Prefix to match
    merge_pkl_files_by_prefix(folder, prefix_pattern)


