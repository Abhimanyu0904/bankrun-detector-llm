import json
import random
import os
import sys

# sys.path.append("..")
sys.path.append(".")
from constants import ENTITY_KEY, RISK_KEY

# Purpose: Changing data format to be compatible with data studio labeling.

BASE_DATA_PATH = os.environ.get("BASE_DATA_PATH")
EMBEDDINGS_DATA_PATH = os.environ.get("EMBEDDINGS_DATA_PATH")
TGT_PATH = BASE_DATA_PATH  # change to BASE_DATA_PATH if you don't care about embeddings / haven't generated embeddings yet

LS_DATA_PATH = os.environ.get("LS_DATA_PATH")
TARGET_BANKS = ["fifth_third"]
TRAINING_SAMPLES = 100
random.seed(0)


# to use when generating your training sample
def format_for_entity_labeling():
    global TARGET_BANKS
    for bank in TARGET_BANKS:
        in_path = TGT_PATH + f"{bank}.json"
        out_path = LS_DATA_PATH + f"{bank}.json"
        labeled_json = []

        # Open file
        with open(in_path, "r") as in_file:
            for line in in_file.readlines():
                data = json.loads(line)
                new_data = {"data": data}
                labeled_json.append(new_data)

        # Randomize order of files based on seed.
        random.shuffle(labeled_json)
        labeled_json = labeled_json[: min(TRAINING_SAMPLES, len(labeled_json))]

        # Write to file
        with open(out_path, "w") as out_file:
            json.dump(labeled_json, out_file)


# to use when preparing the data for another round of labeling
def format_for_risk_classification():
    global TARGET_BANKS
    for bank in TARGET_BANKS:
        in_path = LS_DATA_PATH + f"{bank}.json"
        out_path = LS_DATA_PATH + f"{bank}.json"
        labeled_json = []

        # Open file
        with open(in_path, "r") as in_file:
            jsons = json.load(in_file)
            for old_data in jsons:
                new_data = {"data": {}}
                for key in [
                    "text",
                    "retweet_count",
                    "user_id",
                    "user_followers",
                    "created_date",
                    "created_time",
                    "embedding",
                    ENTITY_KEY,
                ]:
                    if key in old_data:
                        new_data["data"][key] = old_data[key]
                labeled_json.append(new_data)

        # Write to file
        with open(out_path, "w") as out_file:
            json.dump(labeled_json, out_file)


# to use when all labeling is done, to get rid of any excess attributes
def format_after_labeling():
    global TARGET_BANKS
    for bank in TARGET_BANKS:
        # in_path = LS_DATA_PATH + f"{bank}.json"
        # out_path = LS_DATA_PATH + f"{bank}.json"
        in_path = LS_DATA_PATH + "combined.json"
        out_path = LS_DATA_PATH + "combined_formatted.json"
        labeled_json = []

        # Open file
        with open(in_path, "r") as in_file:
            jsons = json.load(in_file)
            for old_data in jsons:
                new_data = {}
                for key in [
                    "text",
                    # "retweet_count",
                    # "user_id",
                    # "user_followers",
                    # "created_date",
                    # "created_time",
                    # "embedding",
                    "entity",
                    ENTITY_KEY,
                    RISK_KEY,
                ]:
                    if key in old_data:
                        new_data[key] = old_data[key]
                labeled_json.append(new_data)

        # Write to file
        with open(out_path, "w") as out_file:
            json.dump(labeled_json, out_file)


if __name__ == "__main__":
    if not os.path.exists(LS_DATA_PATH):
        os.makedirs(LS_DATA_PATH)
    if not TARGET_BANKS:
        print(
            "You have not specified any target banks. This script requires you to specify target banks."
        )

    # format_for_entity_labeling()
    # format_for_risk_classification()
    format_after_labeling()
