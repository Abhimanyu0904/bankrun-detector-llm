import json
import datetime
import os

# Purpose: preprocess data for further classifications (preprocesses all files at once).

SFTP_DATA_PATH = os.environ.get("SFTP_DATA_PATH")
BASE_DATA_PATH = os.environ.get("BASE_DATA_PATH")

def bank_name(string):
    return string[9:-5]

def preprocess_data():
    if not os.path.exists(BASE_DATA_PATH):
        os.makedirs(BASE_DATA_PATH)
    
    for file_name in os.listdir(SFTP_DATA_PATH):
        in_path = SFTP_DATA_PATH + file_name
        out_path = BASE_DATA_PATH + f"{bank_name(file_name)}.json"
        modified_jsons = []

        if os.path.isfile(in_path) and in_path[-4:] == "json":
            with open(in_path, 'r') as f:
                tweets = json.load(f)
                for data in tweets:
                    data["text"] = data["text"].replace("\n", " ")
                    data["entity"] = bank_name(file_name)
                    
                    if "created_at" in data and data["created_at"] is not None:
                        created_at_datetime = datetime.datetime.strptime(data["created_at"], "%a %b %d %H:%M:%S +0000 %Y")
                        data["created_date"] = created_at_datetime.strftime('%Y-%m-%d')
                        data["created_time"] = created_at_datetime.strftime('%H:%M:%S')
                        del data["created_at"]
                    
                    modified_jsons.append(data)
            
            with open(out_path, 'w') as f:
                json.dump(modified_jsons, f)

if __name__ == "__main__":
    preprocess_data()