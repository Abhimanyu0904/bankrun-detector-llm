#!/usr/bin/env python
import json
import os
import re
import datetime
from collections import defaultdict

base_path = "/nlp/data/vdelopez/"
data_path = base_path + "cleaned_data/"


def get_bank_name(bank_idx):
    entries = os.listdir(data_path)
    bank_list = [
        entry for entry in entries if os.path.isdir(os.path.join(data_path, entry))
    ]
    return sorted(bank_list)[bank_idx - 1]


def parse_file(bank_file_path, file, year_matches):
    matches = 0
    with open(bank_file_path + "/" + file, encoding="utf-8") as file:
        for line in file.readlines():
            try:
                data = json.loads(line.strip())
                matches += 1
                time = datetime.datetime.utcfromtimestamp(
                    int(data["created_utc"])
                ).strftime("%Y-%m-%d %H:%M:%S")
                year = time[0:4]
                year_matches[year] += 1
            except Exception as e:
                print("Error while reading in JSON: " + str(e))
    return matches


# Main Function
def analyze_data():
    bank_name = get_bank_name(int(os.environ.get("SGE_TASK_ID")))
    print(bank_name)
    bank_file_path = data_path + bank_name
    year_matches = defaultdict(int)
    sum = 0
    for _, _, files in os.walk(bank_file_path):
        for file in files:
            if file.startswith("f_"):
                sum += parse_file(bank_file_path, file, year_matches)
    print("Number of matches: " + str(sum))
    print(year_matches)


if __name__ == "__main__":
    analyze_data()
