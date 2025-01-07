from openai import OpenAI
import numpy as np
import json
import os

# Purpose: generate open_ai embeddings.

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
PROJECT_PATH = os.environ.get("PROJECT_PATH")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL")

def generate_embeddings(sentences):
    return client.embeddings.create(
        model=EMBEDDINGS_MODEL,
        input=sentences
    )

def generate_stored_embeddings():
    for prefix in ["bank", "risk"]:
        input_file = f"{PROJECT_PATH}classifiers/storage/{prefix}_descriptions.json"
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        embeddings = {}
        sorted_banks = sorted(data.keys())
        descriptions = [data[bank] for bank in sorted_banks]
        embedding_objects = generate_embeddings(descriptions)
        for i in range(len(sorted_banks)):
            embeddings[sorted_banks[i]] = embedding_objects.data[i].embedding

        # Save the embeddings to a new JSON file
        output_file = f"{PROJECT_PATH}classifiers/storage/{prefix}_embeddings.json"
        with open(output_file, 'w') as f:
            json.dump(embeddings, f)

if __name__ == "__main__":
    generate_stored_embeddings()