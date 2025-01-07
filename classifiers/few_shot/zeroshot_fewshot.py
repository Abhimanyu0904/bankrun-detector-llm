from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter
import asyncio
import json
import os
import sys
import time
from tqdm.asyncio import tqdm_asyncio

# sys.path.append("..")
sys.path.append(".")
from accuracy_metrics import generate_accuracy_metrics
from constants import POSITIVE_ENTITY_LABEL, POSITIVE_RISK_LABEL, NEGATIVE_ENTITY_LABEL, NEGATIVE_RISK_LABEL, ENTITY_KEY, RISK_KEY

# Purpose: use embedding constants (e.g. bank descriptions, risk description) and generate classifications based on that.

limiter = AsyncLimiter(2000)
counter = 0
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
PROJECT_PATH = os.environ.get("PROJECT_PATH")
LS_DATA_PATH = os.environ.get("LS_DATA_PATH")
CHAT_MODEL = os.environ.get("CHAT_MODEL")

data = []
confusion_matrix = [0, 0, 0, 0] # TP, FP, FN, TN
TARGET_BANKS = ["all"]
TASK = "risk_classification" # choose between "entity_classification" and "risk_classification"
IS_ZERO_SHOT = True
POSITIVE_LABEL = POSITIVE_ENTITY_LABEL if TASK == "entity_classification" else POSITIVE_RISK_LABEL
NEGATIVE_LABEL = NEGATIVE_ENTITY_LABEL if TASK == "entity_classification" else NEGATIVE_RISK_LABEL
DATA_ATTR = ENTITY_KEY if TASK == "entity_classification" else RISK_KEY
BATCH_SIZE = 8

def bank_name(string):
    return string[:-5]

def load_data(bank):
    global data
    # filepath = LS_DATA_PATH + f"{bank}.json"
    filepath = PROJECT_PATH + "relabeled_training_data.json"
    with open(filepath, "r") as file:
        data = json.load(file)
    data = [tweet for tweet in data if not skip_tweet(tweet)]

def get_prompt_info():
    with open(f"{PROJECT_PATH}classifiers/few_shot/prompts.json", "r") as f:
        data = json.load(f)
    return {
        "system": data[f"{"entity" if TASK == "entity_classification" else "risk"}_system"],
        "examples": data[f"{"entity" if TASK == "entity_classification" else "risk"}_examples"]
    }

def add_to_cm(prediction, actual, tweet):
    if prediction == actual == POSITIVE_LABEL: # TP
        confusion_matrix[0] += 1
    elif prediction == actual == NEGATIVE_LABEL: # TN
        confusion_matrix[3] += 1
    elif prediction == POSITIVE_LABEL: # FP
        confusion_matrix[1] += 1
        print('Prediction:', prediction)
        print('Actual:', actual)
        print('FP:', tweet)
        print('--' * 20)
    else: # FN
        confusion_matrix[2] += 1
        print('Prediction:', prediction)
        print('Actual:', actual)
        print('FN:', tweet)
        print('--' * 20)
    
    # if prediction != actual:
    #     print(tweet)

async def get_gpt_response(system_prompt, training_tweets, prompt_request):
    global counter
    if not IS_ZERO_SHOT:
        training_input = "Examine the following examples of tweets and their classifications closely. You should use them to help you with classifying. Note this is not an exhaustive list and just gives a hint of the tweets and expected labels.\n"
        for idx, tweet in enumerate(training_tweets):
            if TASK == "entity_classification":
                training_input += f'{idx+1}) "{tweet["tweet"]}" - [Correct Label: "{tweet["label"]}", Bank Name: {tweet["bank"]}]'
            else:
                training_input += f'{idx+1}) "{tweet["tweet"]}" - [Correct Label: "{tweet["label"]}"]'
            training_input += "\n"
        prompt_request = training_input + "\n\n" + prompt_request
    # print(system_prompt)
    # print(prompt_request)
    messages_input = [{"role": "system", "content": system_prompt}]
    messages_input.append({"role": "user", "content": prompt_request})

    # counter += 1
    # if counter == 1:
    #     print(messages_input)
    
    while True:
        try:
            response = await client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages_input,
                max_tokens=1000,
            )
            response = response.choices[0].message.content
            string = json.loads(response)

            return string
        except Exception as e:
            print(f"Error Message from GPT Call: {e}")
            print(f"Response was: {response}")

def is_valid_tweet_response(resp):
    return "prediction" in resp and (resp["prediction"] == POSITIVE_LABEL or resp["prediction"] == NEGATIVE_LABEL)

def skip_tweet(tweet):
    return TASK == "risk_classification" and tweet[ENTITY_KEY] == "Incorrect Entity"

def format_prompt_message(tweets, bank_description):
    prompt = ""
    # if TASK == "entity_classification":
    #     prompt = f"Classify these tweets based on the following bank descriptions: {bank_description}.\n"
    prompt += "Please classify this list of tweets:\n"
    for idx, tweet in enumerate(tweets):
        prompt += f'{idx + 1}) "{tweet}"\n'
    
    return prompt

async def async_gpt_get(tweets, system_prompt, training_tweets, addtl_info):
    batched_text = [tweet["text"] for tweet in tweets]
    prompt = format_prompt_message(batched_text, addtl_info)

    try:
        response = await get_gpt_response(
            system_prompt,
            training_tweets,
            prompt
        )
        if isinstance(response, list) and len(response) == len(batched_text):
            for j, prediction in enumerate(response):
                if is_valid_tweet_response(prediction):
                    add_to_cm(prediction["prediction"], tweets[j][DATA_ATTR], batched_text[j])
        elif isinstance(response, json) and len(batched_text) == 1:
            if is_valid_tweet_response(response):
                add_to_cm(response["prediction"], tweets[j][DATA_ATTR], batched_text[j])
    except Exception as e:
        print(f"Exception: {e}")
        return
    return

async def generate_gpt_predictions():
    global TARGET_BANKS
    if not TARGET_BANKS:
        print("You haven't specified any banks! This script will default to running on all banks on this folder. Make sure this is the intended behavior.")
        TARGET_BANKS = [bank_name(name) for name in os.listdir(LS_DATA_PATH) if os.path.isdir(os.path.join(LS_DATA_PATH, name))]
    
    prompt_obj = get_prompt_info()
    system_prompt = prompt_obj["system"]
    training_tweets = prompt_obj["examples"]

    if TASK == "entity_classification":
        with open(f"{PROJECT_PATH}classifiers/storage/bank_descriptions.json", "r") as f:
            bank_descriptions = json.load(f)

    # perform batching based on bank compared to all
    for bank in TARGET_BANKS:
        print(f"Currently processing: {bank}")
        load_data(bank)
        addtl_info = bank_descriptions[bank] if TASK == "entity_classification" else ""

        tasks = []
        async with limiter:
            for i in range(0, len(data), BATCH_SIZE):
                batched_tweets = data[i:min(i + BATCH_SIZE, len(data))]
                batched_tweets = [tweet for tweet in batched_tweets]
                if not batched_tweets:
                    continue
                
                task = asyncio.create_task(
                    async_gpt_get(
                        batched_tweets,
                        system_prompt,
                        training_tweets,
                        addtl_info
                    )
                )
                tasks.append(task)
            await tqdm_asyncio.gather(*tasks)
    
    generate_accuracy_metrics(confusion_matrix)

if __name__ == "__main__":
    start = time.time()
    asyncio.run(generate_gpt_predictions())
    print(f"Time Elapsed: {round(time.time() - start, 2)} seconds")