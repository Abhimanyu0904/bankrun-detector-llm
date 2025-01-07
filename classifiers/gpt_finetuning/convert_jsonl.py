import json

# Converts file to jsonl format, which is necessary for GPT labeling.

f = open("training_data.json")
data = json.load(f)
new_data = []

for tweet in data:
    new_data.append(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are trained to analyze and detect whether a given social media post is indicative of a bank run. Analyze the following posts and determine if the sentiment is: indicative of a bank run or not indicative of a bank run. Return answer as a single phrase as either 'Indicative of a Bank Run' or 'Not Indicative of a Bank Run'.",
                },
                {"role": "user", "content": tweet["text"]},
                {"role": "assistant", "content": tweet["risk_sentiment"]},
            ]
        }
    )

with open("train.jsonl", "w") as outfile:
    for entry in new_data:
        json.dump(entry, outfile)
        outfile.write("\n")

# data format:
# {
#     "messages": [
#         {"role": "system", "content": "You are trained to analyze and detect whether a given social media post is indicative of a bank run. Analyze the following posts and determine if the sentiment is: indicative of a bank run (Risky) or not indicative of a bank run (Non-risky). Return answer as a single word as either Risky or Non-risky."},
#         {"role": "user", "content": tweet['text']},
#         {"role": "assistant", "content": tweet['sentiment']}
#     ]
# }
