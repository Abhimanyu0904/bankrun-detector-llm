#!/usr/bin/env python
import os

# Purpose: Combine reddit files in a subfolder into one file.

def combine_files_in_subdirectory(subdirectory, output_file):
    # List all files in the given subdirectory
    files = sorted(os.listdir(subdirectory))
    # Open the output file in write mode
    with open(output_file, "w") as outfile:
        # Iterate over each file
        for filename in files:
            filepath = os.path.join(subdirectory, filename)
            # Check if it's a file and not a directory
            if os.path.isfile(filepath):
                # Open each file in read mode
                with open(filepath, "r") as infile:
                    # Append its content to the output file
                    outfile.write(infile.read())
                    outfile.write("\n")  # Optional: Add a newline between files


def combine_files_in_directory(out_path, directory):
    os.makedirs(os.path.dirname(out_path + "/"), exist_ok=True)
    # Iterate over each subdirectory in the given directory
    for subdir_name in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir_name)
        # Check if it's a directory
        if os.path.isdir(subdir_path) and subdir_path != out_path:
            # Define output file name for the combined content
            output_file = os.path.join(out_path, subdir_name + "_combined.json")
            # Combine files in the subdirectory
            combine_files_in_subdirectory(subdir_path, output_file)


# Example usage
directory_path = "/nlp/data/vdelopez/test/"
out_path = "/nlp/data/vdelopez/test/combined_json_files"
combine_files_in_directory(out_path, directory_path)
