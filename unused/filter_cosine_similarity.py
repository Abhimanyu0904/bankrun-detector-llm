import os
from tqdm import tqdm

# this is for Reddit data


liqudity_crisis_threshold = 0.3
liqudity_event_threshold = 0.3

directory = './files/predicted/predicted_sbert_liquidity_event_crisis_json/'
output_directory = './files/filtered/filtered_cosine_similarity'

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        output_data = []
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc=f"Processing {filename}"):
                try:

                    # Find the 'predicted_label' field in the JSON line
                    label_header = line.find('"predicted_label":')
                    if label_header != -1:
                        # Extract the content of the 'predicted_label' field
                        label_start = label_header + len('"predicted_label": "')
                        label_end = line.find('"', label_start)
                        label_text = line[label_start:label_end]                            

                    # Find the 'similarity' field in the JSON line
                    similarity_header = line.find('"similarity":')
                    if similarity_header != -1:
                        # Extract the content of the 'similarity' field
                        similarity_start = similarity_header + len('"similarity": ')
                        similarity_end = line.find('}', similarity_start)
                        similarity = float(line[similarity_start:similarity_end])

                        if label_text == 'Liqudity Event' and similarity >= liqudity_event_threshold:
                            output_data.append(line.strip())

                        elif label_text == 'Liqudity Crisis' and similarity >= liqudity_crisis_threshold:
                            output_data.append(line.strip())

                except Exception as e:
                    print(f"Error processing line in file {filename}: {e}")

        # Write data with predictions to a new file in the output directory
        if output_data:
            with open(os.path.join(output_directory, filename), 'w', encoding='utf-8') as outfile:
                for item in output_data:
                    outfile.write(item + '\n')