#!/usr/bin/env python
import json
import os
import re
base_path = "/nlp/data/smenta/"
results_path = base_path + "matched_results/"
cleaned_data_path = base_path + "cleaned_data/"
attributes_filepath = base_path + "data_fields.txt"
# File I/O Helper Functions
def write_json_to_file(post, file_path):
  with open(file_path, "a+") as output_file:
    json.dump(post, output_file)
    output_file.write('\n')
def create_file_structure(bank_name):
  os.makedirs(os.path.dirname(cleaned_data_path + bank_name + "/"), exist_ok=True)
def read_in_keep_attributes():
  keep_attributes = []
  with open(attributes_filepath, 'r') as file:
    for line in file.readlines():
      keep_attributes.append(line.strip())
  keep_attributes = set(keep_attributes)
  return keep_attributes
def get_bank_name(bank_idx):
  entries = os.listdir(results_path)
  bank_list = [entry for entry in entries if os.path.isdir(os.path.join(results_path, entry))]
  return sorted(bank_list)[bank_idx-1]
def get_filter_out_terms(bank_name):
  data = None
  with open(base_path + "filtered_terms.json", "r") as file:
    data = json.load(file)
  for bank in data:
    if bank == bank_name:
      return data[bank]
  return []
# Other Helper Functions
def get_regex(bank_name):
  filter_terms = get_filter_out_terms(bank_name)
  return "|".join(filter_terms)
def is_valid_post(body, regex):
  return regex == "" or re.search(regex, body, re.IGNORECASE) is None
def clean_json(data, keep_attributes):
  new_data = {}
  for key in data:
    if key in keep_attributes:
      new_data[key] = data[key]
  return new_data
def parse_and_filter_json_file(bank_name, regex, file_name, keep_attributes):
  file_in = results_path + bank_name + "/" + file_name
  file_out = cleaned_data_path + bank_name + "/" + file_name
  with open(file_in, "r") as input_file:
    for line in input_file.readlines():
      try:
        data = json.loads(line.strip())
        if is_valid_post(data["body"], regex):
          cleaned_json = clean_json(data, keep_attributes)
          new_file_path = cleaned_data_path + bank_name + "/" + file_name
          write_json_to_file(cleaned_json, new_file_path)
      except Exception as e:
        print("Error while reading in JSON: " + str(e))
# Main Function
def clean_bank_files():
  # Read in Post Attributes to Keep
  keep_attributes = read_in_keep_attributes()
  # Get Bank Folder Name
  bank_name = get_bank_name(int(os.environ.get("SGE_TASK_ID")))
  # Create Directories (as necessary)
  create_file_structure(bank_name)
  # Get Filter Out Regex
  regex = get_regex(bank_name)
  # Iterate Through Files
  bank_file_path = results_path + bank_name
  for _, _, files in os.walk(bank_file_path):
    for file in files:
      if file.startswith("f_"):
        parse_and_filter_json_file(bank_name, regex, file, keep_attributes)
if __name__ == "__main__":
  clean_bank_files()