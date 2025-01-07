import json
import re
import os
import io
from collections import defaultdict

# Purpose (DEPRECATED): Used to run new regex on data scraped with old regex.

dir_path = os.environ.get("LS_DATA_PATH")
out_dir = os.environ.get("BASE_DATA_PATH")
json_path = os.environ.get("PROJECT_PATH") + "nlpgrid/bank_list.json"

bank_match_regex = {}
bank_skip_regex = {}


def read_bank_regex():
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        for bank in data:
            match_regex = get_regex(data[bank]["match"])
            bank_match_regex[bank] = f"(^|\s)({match_regex})(\s|$)"
            bank_skip_regex[bank] = get_regex(data[bank]["skip"])


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


def run_spacing_regex():
    read_bank_regex()
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        out_path = os.path.join(out_dir, filename)
        if os.path.isfile(file_path):
            bank_tweets = []
            bank_name = filename[9:-5]
            with open(file_path, "r") as json_file:
                lines = json_file.readlines()
                for line in lines:
                    data = json.loads(line)
                    if contains_keywords(
                        data["text"], bank_match_regex[bank_name]
                    ) and (
                        bank_skip_regex[bank_name] == ""
                        or not contains_keywords(
                            data["text"], bank_skip_regex[bank_name]
                        )
                    ):
                        bank_tweets.append(data)

            with open(out_path, "w", encoding="utf-8") as out_file:
                for tweet in bank_tweets:
                    out_file.write(json.dumps(tweet) + "\n")


if __name__ == "__main__":
    run_spacing_regex()