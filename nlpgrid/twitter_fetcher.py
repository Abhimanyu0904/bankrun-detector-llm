#!/usr/bin/env python
import json
import os
import re
import subprocess
from tqdm import tqdm

dir_path = "/nlp/data/corpora/twitter/analysis/"
json_path = "/nlp/data/vdelopez/bank_list.json"
out_path = "/nlp/data/vdelopez/twitter_matched_results/"


# File Helpers
def write_json_to_file(tweet, bank_name, file_name):
    os.makedirs(os.path.dirname(out_path + bank_name + "/"), exist_ok=True)
    with open(
        out_path + bank_name + "/" + file_name[:-4] + ".json", "a+"
    ) as output_file:
        output_file.write(json.dumps(tweet) + "\n")


def create_file_structure(bank_name):
    os.makedirs(os.path.dirname(out_path + bank_name + "/"), exist_ok=True)


def get_bank_info(bank_name):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        if bank_name in data:
            return data[bank_name]
        else:
            return [""]


def get_all_bank_keywords():
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    all_keywords = []
    for bank in data:
        if bank != "General":
            all_keywords.extend(data[bank])
    return all_keywords


def get_file_name(idx):
    file_list = []
    for _, _, files in os.walk(dir_path):
        file_list.extend(files)
    return sorted(file_list)[idx - 1]


# Regex Helpers
def get_regex(keywords):
    return "|".join(keywords)


def contains_keywords(line, expr):
    keyword_pattern = re.compile(expr, re.IGNORECASE)
    return bool(keyword_pattern.search(line))


# Searching Scripts
def search_for_bank(file_name, bank_name):
    file_path = dir_path + file_name
    regex = get_regex(get_bank_info(bank_name))
    match_count = tweet_count = 0
    if regex == "":
        print("Invalid Bank Name")
        return
    command = ["lzop", "-dc", file_path]
    try:
        # Run the command and capture the output
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        # Split the output into lines
        lines = result.stdout.splitlines()
        for line in lines:
            tweet_count += 1
            if contains_keywords(line, regex):
                try:
                    match_count += 1
                    json_data = json.loads(line)
                    write_json_to_file(json_data, bank_name, file_name)
                except Exception as e:
                    print(e)
                    print("Error while reading JSON")
    except Exception as e:
        print(e)
        print("Error in subprocess")
    print(f"Out of {tweet_count} tweets read, there were {match_count} matches found!")


def search_for_bank_w_args():
    bank_name = os.environ.get("BANK_NAME")
    file_idx = os.environ.get("SGE_TASK_ID")
    file_name = get_file_name(int(file_idx))
    # Skipping the .index files
    if file_name[-3:] == "lzo":
        create_file_structure(bank_name)
        search_for_bank(file_name, bank_name)


# This line would initiate the entire process when the script is run
if __name__ == "__main__":
    search_for_bank_w_args()
