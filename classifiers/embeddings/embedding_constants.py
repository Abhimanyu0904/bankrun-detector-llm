import json
import numpy as np
import os
import sys

sys.path.append("..")
from accuracy_metrics import generate_accuracy_metrics
from constants import POSITIVE_ENTITY_LABEL, POSITIVE_RISK_LABEL, NEGATIVE_ENTITY_LABEL, NEGATIVE_RISK_LABEL, ENTITY_KEY, RISK_KEY

# Purpose: use embedding constants (e.g. bank descriptions, risk desecription) and generate classifications based on that.

PROJECT_PATH = os.environ.get("PROJECT_PATH")
LS_DATA_PATH = os.environ.get("LS_DATA_PATH")

data = []
label_embeddings = {}
confusion_matrix = [0, 0, 0, 0] # TP, FP, FN, TN
TARGET_BANKS = ["training_data"]
TASK = "risk_classification" # chose between "entity_classification" and "risk_classification"
POSITIVE_LABEL = POSITIVE_ENTITY_LABEL if TASK == "entity_classification" else POSITIVE_RISK_LABEL
NEGATIVE_LABEL = NEGATIVE_ENTITY_LABEL if TASK == "entity_classification" else NEGATIVE_RISK_LABEL
DATA_ATTR = ENTITY_KEY if TASK == "entity_classification" else RISK_KEY

def bank_name(string):
    return string[9:-4]

def load_data(bank):
    global data
    filepath = LS_DATA_PATH + f"{bank}.json"
    with open(filepath, "r") as file:
        data = json.load(file)

def load_label_embeddings():
    global label_embeddings
    embeddings_file = "bank_embeddings.json" if TASK == "entity_classification" else "risk_embeddings.json"
 
    filepath = f"{PROJECT_PATH}classifiers/storage/" + embeddings_file
    with open(filepath, "r") as file:
        label_embeddings = json.load(file)

def cosine_similarity(e1, e2):
    embedding1 = np.array(e1)
    embedding2 = np.array(e2)

    dot_product = np.dot(embedding1, embedding2)
    magnitude1 = np.linalg.norm(embedding1)
    magnitude2 = np.linalg.norm(embedding2)
    
    cos_sim = dot_product / (magnitude1 * magnitude2)
    return max(0, cos_sim)

def generate_label(e1, e2, threshold):
    score = cosine_similarity(e1, e2)
    return POSITIVE_LABEL if score >= threshold else NEGATIVE_LABEL

def cosine_similarity_predictions(threshold):
    global TARGET_BANKS
    if not TARGET_BANKS:
        print("You haven't specified any banks! This script will default to running on all banks on this folder. Make sure this is the intended behavior.")
        TARGET_BANKS = [name for name in os.listdir(LS_DATA_PATH) if os.path.isdir(os.path.join(LS_DATA_PATH, name))]

    # read tweets in
    confusion_matrix = [0, 0, 0, 0]
    for bank in TARGET_BANKS:
        load_data(bank)
        
        # generate cosine similarity
        for tweet in data:
            to_match = label_embeddings[tweet["entity"]] if TASK == "entity_classification" else label_embeddings["risk"]
            if TASK == "risk_classification" and tweet[ENTITY_KEY] == "Incorrect Entity":
                continue
            result = generate_label(to_match, tweet["embedding"], threshold)
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
    load_label_embeddings()
    for threshold in [x * 0.02 for x in range(0, 50)]:
        print(f"Current Threshold is: {round(threshold, 2)}")
        cosine_similarity_predictions(threshold)
        print("")