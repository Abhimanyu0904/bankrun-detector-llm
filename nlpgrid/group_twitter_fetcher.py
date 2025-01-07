#!/usr/bin/env python
import json
import os
import re
import subprocess
from collections import defaultdict

TWEET_BATCH_SIZE = 1000

dir_path = "/nlp/data/corpora/twitter/analysis/"
json_path = "/nlp/data/smenta/bank_list.json"
out_path = "/nlp/data/smenta/twitter_matched_results/"

bank_match_regex = {}
bank_skip_regex = {}
bank_tweets = defaultdict(list)

# File Helpers
def get_file_name(idx):
    file_list = []
    for _, _, files in os.walk(dir_path):
        file_list.extend(files)
    return sorted(file_list)[idx - 1]

def read_bank_regex():
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        for bank in data:
            bank_match_regex[bank] = f"(?:^|\s)({get_regex(data[bank]["match"])})(?:\s|$)"
            bank_skip_regex[bank] = get_regex(data[bank]["skip"])

def flush_bank(bank_name, file_name):
    # Create File Structure
    os.makedirs(os.path.dirname(out_path + bank_name + "/"), exist_ok=True)
    
    # Write Batch of Tweets
    with open(
        out_path + bank_name + "/" + file_name[:-4] + ".json", "a+"
    ) as output_file:
        for tweet in bank_tweets[bank_name]:
            output_file.write(json.dumps(tweet) + "\n")
        bank_tweets[bank_name] = []

def flush_all(file_name):
    for bank in bank_match_regex:
        if len(bank_tweets[bank]) > 0:
            flush_bank(bank, file_name)

# JSON Helpers
def get_relevant_json(tweet_line):
    new_data = {}
    try:
        data = json.loads(tweet_line)
        text = data["text"].strip()

        if text == "":
            return new_data

        new_data["text"] = text
        new_data["retweet_count"] = data["retweet_count"]
        new_data["created_at"] = data["created_at"]
        if "user" in data and data["user"] is not None:
            new_data["user_id"] = data["user"]["id"]
            new_data["user_followers"] = data["user"]["followers_count"]
    except Exception as e:
        print(e)
        print(tweet_line)
        print("Error while reading JSON.")
        return {}

    return new_data


# Regex Helpers
def get_regex(keywords):
    res = []
    for keyword in keywords:
        if keyword[0] == "$":
            res.append("\\" + keyword)
        else:
            res.append(keyword)
    return "|".join(res)

def contains_keywords(line, expr):
    keyword_pattern = re.compile(expr, re.IGNORECASE)
    return bool(keyword_pattern.search(line))

# Main Functions
def add_tweet(bank_name, tweet, file_name):
    bank_tweets[bank_name].append(tweet)
    if len(bank_tweets[bank_name]) == TWEET_BATCH_SIZE:
        flush_bank(bank_name, file_name)

def search_file_index():
    file_idx = os.environ.get("SGE_TASK_ID")
    file_name = get_file_name(int(file_idx))
    tweet_count = match_count = 0

    # Check to make sure file is not an index file and there is no dedup version (reduces redundancies).
    is_lzo = file_name[-3:] == "lzo"
    is_dedup_if_exists = file_name[-9:] == "dedup.lzo" or not os.path.exists(file_name.split(".")[0] + "-dedup.lzo")

    if not is_lzo or not is_dedup_if_exists:
        print("Invalid File Index")
        return

    file_path = dir_path + file_name
    read_bank_regex()

    command = ["lzop", "-dc", file_path]
    try:
        # Capture text output of file.
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        # Split file output into lines.
        lines = result.stdout.splitlines()
        line_buffer = ""

        for new_line in lines:
            line = new_line
            if not (new_line.startswith("{") and new_line.endswith("}")):
                line_buffer += new_line
                if line_buffer.startswith("{") and line_buffer.endswith("}"):
                    line = line_buffer
                else:
                    continue
            line_buffer = ""

            tweet_count += 1
            data = get_relevant_json(line)
            
            if len(data) == 0:
                continue

            for bank, bank_reg in bank_match_regex.items():
                if contains_keywords(data["text"], bank_reg) and (bank_skip_regex[bank] == "" or not contains_keywords(data["text"], bank_skip_regex[bank])):
                    match_count += 1
                    add_tweet(bank, data, file_name)

        # Flush any remaining buffered messages.
        flush_all(file_name)
    except Exception as e:
        print(e)
        print("Error in subprocess")
    print(f"Out of {tweet_count} tweets read, there were {match_count} matches found!")


if __name__ == "__main__":
    search_file_index()
