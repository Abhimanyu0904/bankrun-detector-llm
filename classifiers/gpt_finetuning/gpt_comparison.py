from dotenv import load_dotenv
import os
from openai import OpenAI
import json
from constants import POSITIVE_RISK_LABEL, NEGATIVE_RISK_LABEL

# load_dotenv()  # take environment variables from .env.

in_path = "training_data.json"
out_path = "finetuned_results.json"
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_KEY,
)

system_message = {
    "role": "system",
    "content": "You are trained to analyze and detect whether a given social media post is indicative of a bank run. Analyze the following posts and determine if the sentiment is: indicative of a bank run or not indicative of a bank run. Return answer as a single phrase as either 'Indicative of a Bank Run' or 'Not Indicative of a Bank Run'.",
}

result = []

num_tweets = 0
correct = 0
confusion_matrix = [0, 0, 0, 0]  # TP, FP, FN, TN

POSITIVE_LABEL = POSITIVE_RISK_LABEL
NEGATIVE_LABEL = NEGATIVE_RISK_LABEL


def generate_accuracy_metrics(confusion_matrix):
    # Unpack confusion matrix values
    TP, FP, FN, TN = confusion_matrix

    # Calculate accuracy, precision, recall, f1-score
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )

    # Print each statistic
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")


def add_to_cm(prediction, actual):
    if prediction == actual == POSITIVE_LABEL:
        confusion_matrix[0] += 1
    elif prediction == actual == NEGATIVE_LABEL:
        confusion_matrix[3] += 1
    elif prediction == POSITIVE_LABEL:
        confusion_matrix[1] += 1
    else:
        confusion_matrix[2] += 1


with open(in_path, "r") as train_data:
    data = json.load(train_data)
    for tweet in data:
        num_tweets += 1
        user_message = {"role": "user"}
        user_message["content"] = tweet["text"]
        messages = [system_message, user_message]
        response = client.chat.completions.create(
            messages=messages,
            # model="gpt-4",
            model="ft:gpt-3.5-turbo-0125:ccb-lab-members-fine-tunes::94Zfxg1k",
        )
        answer = response.choices[0].message.content
        add_to_cm(answer, tweet["risk_sentiment"])
        # if tweet["risk_sentiment"] == answer:
        #     correct += 1
        result.append(
            {
                "tweet": tweet["text"],
                "label": tweet["risk_sentiment"],
                "gpt-answer": answer,
            }
        )

with open(out_path, "w") as out_file:
    json.dump(result, out_file)

generate_accuracy_metrics(confusion_matrix)
