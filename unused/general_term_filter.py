#!/usr/bin/env python
import os
import re
import json

# Purpose (DEPRECATED): Filter tweets based on relevant bank terms.

keywords = [
    "liquidity (event|crisis|problems?|issues?|shortage)",
    "bank run",
    "banking panic",
    "bank failure",
    "financial contagion",
    "cash crunch",
    "money squeeze",
    "asset (liquidation|sales?|sell-off)",
    "credit (crunch|freeze|squeeze)",
    "financial emergency",
    "loan crisis",
    "debt crisis",
    "funding gap",
    "central bank (intervention|bailout|support)",
    "overleverag(e|ing)",
    "withdrawals?",
    "withdraw",
    "withdrew",
    "depositor (confidence|withdrawals?|run)",
    "cash flow (problems?|crisis|shortfall)",
    "borrowing (difficulties|challenges)",
    "credit line",
    "credit access",
    "IPO",
    "M&A",
    "VC exit",
    "PE exit",
    "divestiture",
    "solvency crisis",
    "financial obligations",
    "short-term (obligations|funding)",
    "emergency (loan|funding)",
    "interbank (lending|borrowing|market)",
    "fire sale",
    "economic (downturn|collapse)",
    "market (crash|turmoil)",
    "financial (instability|distress)",
    "panic (withdrawal|selling)",
    "insolvency",
    "bankruptcy",
    "bailout",
    "federal reserve",
    "ECB",
    "systemic risk",
    "market volatility",
    "stock market crash",
    "investment risk",
    "bubble burst",
    "financial meltdown",
    "toxic assets",
    "bad debts",
    "non-performing (loans|assets)",
    "risk management failure",
    "regulatory failure",
    "policy response",
    "government intervention",
    "economic rescue",
    "fiscal stimulus",
    "quantitative easing",
    "interest rate (hike|cut)",
    "financial regulation",
    "economic bailout",
    "debt burden",
    "credit default",
    "loan default",
    "leverage ratio",
    "capital adequacy",
    "bank distress",
    "financial shock",
    "market sentiment",
    "investor confidence",
    "economic uncertainty",
    "financial stability",
    "banking sector (woes|trouble)",
    "lender of last resort",
]
directory = "/nlp/data/vdelopez/test/combined_json_files"
output_directory = "/nlp/data/vdelopez/test/combined_json_files/filtered_json_files"
pattern = re.compile("|".join(keywords), re.IGNORECASE)


def get_file_name(idx):
    file_list = []
    for _, _, files in os.walk(directory):
        file_list.extend(files)
    return sorted(file_list)[idx - 1]


def filter_combined_tweets(file_name):
    output_data = []
    with open(os.path.join(directory, file_name), "r") as file:
        for line in file.readlines():
            if len(line) > 0:
                try:
                    new_line = line.strip()
                    data = json.loads(new_line)
                    if pattern.search(data["text"]):
                        output_data.append(new_line)
                except Exception as e:
                    print(f"Error processing line in file {file_name}: {e}")
    # Write filtered data to a new file in the output directory
    if output_data:
        with open(os.path.join(output_directory, file_name), "w") as outfile:
            for item in output_data:
                outfile.write(item + "\n")


def filter_combined_tweets_with_args():
    file_idx = os.environ.get("SGE_TASK_ID")
    file_name = get_file_name(int(file_idx))
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory + "/")
    filter_combined_tweets(file_name)


if __name__ == "__main__":
    filter_combined_tweets_with_args()
