import os
import json

# File Paths
base_path = "/nlp/data/smenta/twitter_matched_results/"
dir_path = "/nlp/data/smenta/twitter_combined_results/bank_files/"

# Function to append contents of each JSON file
def append_json_files(subdirectory):
    combined_data = []

    # List all files in the subdirectory
    for file in os.listdir(subdirectory):
        if file.endswith(".json"):
            with open(os.path.join(subdirectory, file), 'r') as f:
                data = json.load(f)
                combined_data.extend(data)

    return combined_data

def combine_files():
    # Ensure the output directory exists
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Go through each subdirectory and combine JSON files
    for subdir in os.listdir(base_path):
        full_subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(full_subdir_path):
            combined_json = append_json_files(full_subdir_path)

            # Write combined data to a new JSON file in dir_path
            output_file = os.path.join(dir_path, f"combined_{subdir}.json")
            with open(output_file, 'w') as f:
                json.dump(combined_json, f, indent=4)
    

if __name__ == "__main__":
    combine_files()
