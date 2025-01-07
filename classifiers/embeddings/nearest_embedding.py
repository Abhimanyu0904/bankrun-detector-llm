import json
import numpy as np
import os
import random
import sys

sys.path.append("..")
from accuracy_metrics import generate_accuracy_metrics
from constants import POSITIVE_ENTITY_LABEL, POSITIVE_RISK_LABEL, NEGATIVE_ENTITY_LABEL, NEGATIVE_RISK_LABEL, ENTITY_KEY, RISK_KEY

# Purpose: nearest embedding training approach (from a set of training data, for each new label, find the most similar data point and label it based on that).

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
LS_DATA_PATH = os.environ.get("LS_DATA_PATH")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL")

TEST_PROPORTION = 0.5
random.seed(0)
relevant_training_tweets, non_relevant_training_tweets = [], []
test_tweets = []
confusion_matrix = [0, 0, 0, 0] # TP, FP, FN, TN
TARGET_BANKS = ["doral_bank"]
TASK = "risk_classification" # chose between "entity_classification" and "risk_classification"
POSITIVE_LABEL = POSITIVE_ENTITY_LABEL if TASK == "entity_classification" else POSITIVE_RISK_LABEL
NEGATIVE_LABEL = NEGATIVE_ENTITY_LABEL if TASK == "entity_classification" else NEGATIVE_RISK_LABEL
DATA_ATTR = ENTITY_KEY if TASK == "entity_classification" else RISK_KEY

def bank_name(string):
    return string[:-5]
  
def load_data(bank):
    filepath = LS_DATA_PATH + f"{bank}.json"
    with open(filepath, "r") as file:
        data = json.load(file)
    
    subsample_size = int(TEST_PROPORTION * len(data))
    if subsample_size > len(data):
        raise ValueError("Subsample size must be smaller than the number of data points.")
    subsample = set(random.sample(list(range(len(data))), subsample_size))

    for i, tweet in enumerate(data):
        if i in subsample:
            test_tweets.append(tweet)
        else:
            if tweet[DATA_ATTR] == POSITIVE_LABEL:
                relevant_training_tweets.append(tweet["embedding"])
            else:
                non_relevant_training_tweets.append(tweet["embedding"])

def find_most_similar(tweet_embedding, embeddings):
    max_similarity = -np.inf
    for emb in embeddings:
        similarity = np.dot(tweet_embedding, emb) / (np.linalg.norm(tweet_embedding) * np.linalg.norm(emb))
        if similarity > max_similarity:
            max_similarity = similarity
    return max_similarity

def predict_relevance(tweet):
    tweet_embedding = tweet["embedding"]
    max_similarity_relevant = find_most_similar(tweet_embedding, relevant_training_tweets)
    max_similarity_non_relevant = find_most_similar(tweet_embedding, non_relevant_training_tweets)
    if max_similarity_relevant > max_similarity_non_relevant:
        return POSITIVE_LABEL
    else:
        return NEGATIVE_LABEL
    
def nearest_embedding_predictions():
    global TARGET_BANKS
    if not TARGET_BANKS:
        print("You haven't specified any banks! This script will default to running on all banks on this folder. Make sure this is the intended behavior.")
        TARGET_BANKS = [bank_name(name) for name in os.listdir(LS_DATA_PATH) if os.path.isdir(os.path.join(LS_DATA_PATH, name))]
        
    for bank in TARGET_BANKS:
        load_data(bank)
        for tweet in test_tweets:
            if TASK == "risk_classification" and tweet[ENTITY_KEY] == "Incorrect Entity":
                continue
            result = predict_relevance(tweet)
            if result == tweet[DATA_ATTR] == POSITIVE_LABEL:
                confusion_matrix[0] += 1
            elif result == tweet[DATA_ATTR] == NEGATIVE_LABEL:
                confusion_matrix[3] += 1
            elif result == POSITIVE_LABEL:
                confusion_matrix[1] += 1
            else:
                confusion_matrix[2] += 1

    generate_accuracy_metrics(confusion_matrix)


if __name__ == "__main__":
    nearest_embedding_predictions()