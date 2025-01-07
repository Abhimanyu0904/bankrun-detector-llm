from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter
import asyncio
import json
import os
import sys
import time
from tqdm.asyncio import tqdm_asyncio
import math
import time

sys.path.append("..")
from constants import POSITIVE_ENTITY_LABEL, POSITIVE_RISK_LABEL, NEGATIVE_ENTITY_LABEL, NEGATIVE_RISK_LABEL, ENTITY_KEY, RISK_KEY

# Purpose: use embedding constants (e.g. bank descriptions, risk description) and generate classifications based on that.

limiter = AsyncLimiter(50, 60)
counter = 0
client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
PROJECT_PATH = os.environ.get("PROJECT_PATH")
BASE_DATA_PATH = os.environ.get("BASE_DATA_PATH")
LABELED_DATA_PATH = os.environ.get("LABELED_DATA_PATH")
CHAT_MODEL = os.environ.get("CHAT_MODEL")

TARGET_BANKS = ["horizon_bank", "legacy_bank", "nrb_chicago"]
TASK = "entity_classification" # choose between "entity_classification" and "risk_classification"
IS_ZERO_SHOT = False
POSITIVE_LABEL = POSITIVE_ENTITY_LABEL if TASK == "entity_classification" else POSITIVE_RISK_LABEL
NEGATIVE_LABEL = NEGATIVE_ENTITY_LABEL if TASK == "entity_classification" else NEGATIVE_RISK_LABEL
BATCH_SIZE = 20

def bank_name(string):
    return string[:-5]

def recalculate_labels():
    global POSITIVE_LABEL, NEGATIVE_LABEL
    POSITIVE_LABEL = POSITIVE_ENTITY_LABEL if TASK == "entity_classification" else POSITIVE_RISK_LABEL
    NEGATIVE_LABEL = NEGATIVE_ENTITY_LABEL if TASK == "entity_classification" else NEGATIVE_RISK_LABEL

def data_attr():
    return ENTITY_KEY if TASK == "entity_classification" else RISK_KEY

def get_prompt_info():
    with open(f"{PROJECT_PATH}classifiers/few_shot/prompts.json", "r") as f:
        data = json.load(f)
    return {
        "system": data[f"{"entity" if TASK == "entity_classification" else "risk"}_system"],
        "examples": data[f"{"entity" if TASK == "entity_classification" else "risk"}_examples"]
    }

def calculate_rate_limit():
    tokens_per_request = 0
    tokens_per_tweet = 50
    if TASK == "entity_classification" and IS_ZERO_SHOT == True:
        tokens_per_request = 450
    elif TASK == "entity_classification" and IS_ZERO_SHOT == False:
        tokens_per_request = 925
    elif IS_ZERO_SHOT == True:
        tokens_per_request = 425
    else:
        tokens_per_request = 975
    
    result = 300000 // (tokens_per_request + tokens_per_tweet * BATCH_SIZE)
    rounded_result = math.floor(result / 10) * 10

    return min(rounded_result, 1000)

async def get_gpt_response(system_prompt, training_tweets, prompt_request):
    if not IS_ZERO_SHOT:
        training_input = "Here are examples of tweets and their classifications. Note this is not an exhaustive list and just gives a hint of the tweets and expected labels.\n"
        for idx, tweet in enumerate(training_tweets):
            if TASK == "entity_classification":
                training_input += f'{idx+1}) "{tweet["tweet"]}" – [Correct Label: "{tweet["label"]}", Bank Name: {tweet["bank"]}]'
            else:
                training_input += f'{idx+1}) "{tweet["tweet"]}" – [Correct Label: "{tweet["label"]}"]'
            training_input += "\n"
        prompt_request = training_input + "\n\n" + prompt_request # Approach 2
    
    messages_input = [{"role": "system", "content": system_prompt}]
    messages_input.append({"role": "user", "content": prompt_request})
    iter_counter = 2
    
    while iter_counter:
        try:
            response = await client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages_input,
                max_tokens=1500,
            )
            response = response.choices[0].message.content
            string = json.loads(response.strip())

            return string
        except Exception as e:
            print(f"Error Message from GPT Call: {e}")
            print(f"Response was: {response}")
            iter_counter -= 1
    return ""

def is_valid_tweet_response(resp):
    return "prediction" in resp and (resp["prediction"] == POSITIVE_LABEL or resp["prediction"] == NEGATIVE_LABEL)

def skip_tweet(tweet):
    return TASK == "risk_classification" and tweet[ENTITY_KEY] == "Incorrect Entity"

def format_prompt_message(tweets, bank_description):
    prompt = ""
    if TASK == "entity_classification":
        prompt = f"Classify these tweets based on the following bank: {bank_description}.\n"
    prompt += "Please classify this list of tweets:\n"
    for idx, tweet in enumerate(tweets):
        prompt += f'{idx + 1}) "{tweet}"\n'
    
    return prompt

async def async_gpt_get(tweets, system_prompt, training_tweets, addtl_info):
    global counter
    batched_text = [tweet["text"] for tweet in tweets]
    prompt = format_prompt_message(batched_text, addtl_info)
    attr = data_attr()

    try:
        response = await get_gpt_response(
            system_prompt,
            training_tweets,
            prompt
        )
        if response == "":
            return []
        if isinstance(response, list) and len(response) == len(batched_text):
            for j, prediction in enumerate(response):
                if is_valid_tweet_response(prediction):
                    tweets[j][attr] = prediction["prediction"]
        elif isinstance(response, json) and len(batched_text) == 1:
            if is_valid_tweet_response(response):
                tweets[j][attr] = response["prediction"]
        print(f"Batch {counter} complete.")
        counter += 1
    except Exception as e:
        print(f"Exception: {e}")
        return []
    
    return tweets

async def generate_gpt_predictions():
    global TARGET_BANKS, TASK, IS_ZERO_SHOT, counter
    print(TASK, IS_ZERO_SHOT)

    if not TARGET_BANKS:
        print("You haven't specified any banks! This script will default to running on all banks on this folder. Make sure this is the intended behavior.")
        TARGET_BANKS = [bank_name(name) for name in os.listdir(LABELED_DATA_PATH) if os.path.isdir(os.path.join(BASE_DATA_PATH, name))]
    
    prompt_obj = get_prompt_info()
    system_prompt = prompt_obj["system"]
    training_tweets = prompt_obj["examples"]

    if TASK == "entity_classification":
        with open(f"{PROJECT_PATH}classifiers/storage/bank_descriptions.json", "r") as f:
            bank_descriptions = json.load(f)

    for bank in TARGET_BANKS:
        counter = 0
        print(f"Currently processing {bank} for the task: {TASK}.")
        if TASK == "entity_classification":
            filepath = BASE_DATA_PATH + f"{bank}.json"
        else:
            filepath = LABELED_DATA_PATH + f"{bank}_temp.json"
        with open(filepath, "r") as f:
            data = json.load(f)
        data = [tweet for tweet in data if not skip_tweet(tweet)]

        addtl_info = bank_descriptions[bank] if TASK == "entity_classification" else ""

        tasks = []
        all_tweets = []
        print(f"Total Batches: {len(data) // BATCH_SIZE + 1}")
        for i in range(0, len(data), BATCH_SIZE):
            batched_tweets = data[i:min(i + BATCH_SIZE, len(data))]
            batched_tweets = [tweet for tweet in batched_tweets]
            if not batched_tweets:
                continue
            async with limiter:
                task = asyncio.create_task(
                    async_gpt_get(
                        batched_tweets,
                        system_prompt,
                        training_tweets,
                        addtl_info
                    )
                )
            tasks.append(task) 
        tweet_groups = await asyncio.gather(*tasks)
        
        for lst in tweet_groups:
            all_tweets.extend(lst)
        
        if TASK == "entity_classification":
            filepath = LABELED_DATA_PATH + f"{bank}_temp.json"
        else:
            filepath = LABELED_DATA_PATH + f"{bank}.json"
        with open(filepath, "w") as f:
            json.dump(all_tweets, f)

if __name__ == "__main__":
    # entity filtering
    TASK = "entity_classification"
    IS_ZERO_SHOT = False
    recalculate_labels()

    start = time.time()
    asyncio.run(generate_gpt_predictions())
    print(f"Time Elapsed: {round(time.time() - start, 2)} seconds")

    print("Going to sleep now.")
    time.sleep(60)
    print("Done sleeping!")

    # risk labeling
    TASK = "risk_classification"
    IS_ZERO_SHOT = False
    recalculate_labels()

    start = time.time()
    asyncio.run(generate_gpt_predictions())
    print(f"Time Elapsed: {round(time.time() - start, 2)} seconds")