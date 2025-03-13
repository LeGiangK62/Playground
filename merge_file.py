import os
import pickle
import numpy as np


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

# Example usage
if __name__ == "__main__":
    folder = "Data/12Mar"
    merge_pkl_files(folder)


