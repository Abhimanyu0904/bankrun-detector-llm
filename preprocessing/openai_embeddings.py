from openai import OpenAI
import numpy as np
import json
import os
import time

# Purpose: generate open_ai embeddings.

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
BASE_DATA_PATH = os.environ.get("BASE_DATA_PATH")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL")
EMBEDDINGS_DATA_PATH = os.environ.get("EMBEDDINGS_DATA_PATH")
BATCH_SIZE = 600 # could theoretically go up to 700 based on token estimates
TARGET_BANKS = ["all_risk"]

def bank_name(string):
    return string[:-5]

def get_batch_embeddings(sentences):
    return client.embeddings.create(
        model=EMBEDDINGS_MODEL,
        input=sentences
    )

def get_all_embeddings(tweets):
    all_tweets = []
    for i in range(0, len(tweets), BATCH_SIZE):
        batched_tweets = tweets[i:min(i + BATCH_SIZE, len(tweets))]
        batched_text = [tweet["text"] for tweet in batched_tweets]
        batched_embedding_objects = get_batch_embeddings(batched_text)
        time.sleep(20)
        for j in range(len(batched_tweets)):
            batched_tweets[j]["embedding"] = batched_embedding_objects.data[j].embedding
        
        all_tweets.extend(batched_tweets)
    
    return all_tweets

def generate_tweet_embeddings():
    global TARGET_BANKS
    if not os.path.exists(EMBEDDINGS_DATA_PATH):
        os.makedirs(EMBEDDINGS_DATA_PATH)

    if not TARGET_BANKS:
        print("You haven't specified any banks! This script will default to running on all banks on this folder. Make sure this is the intended behavior.")
        TARGET_BANKS = [bank_name(name) for name in os.listdir(BASE_DATA_PATH) if not os.path.isdir(os.path.join(BASE_DATA_PATH, name))]

    for bank in TARGET_BANKS:
        in_path = BASE_DATA_PATH + f"{bank}.json"
        out_path = EMBEDDINGS_DATA_PATH + f"{bank}.json"

        # Read file
        with open(in_path, "r") as f:
            tweets = json.load(f)
        
        # Batch tweets and write out embeddings
        with open(out_path, "w") as f:
            f.writelines([json.dumps(tweet) + "\n" for tweet in get_all_embeddings(tweets)])

if __name__ == "__main__":
    generate_tweet_embeddings()