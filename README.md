# liquidity-llm-project
Looking into the viability of using an LLM to help identify potential liquidity risk for regional banks. Please don't store any data directly in this repository. Use a `.env` file and don't hardcode file paths unless absolutely necessary. Look at Notes section for more information.

## Folder Descriptions
`/classifiers` – Files used to run different methods of filtering on the data.

`/eda` – Files used to conduct EDA on data.

`/label_studio` – Files related to preparing data for manual labeling with Label Studio.

`/nlpgrid` – Files used in nlpgrid for data scraping / modification (these aren't guaranteed to be up to date). For up to date files, use `sftp` from nlpgrid directly.

`/preprocessing` – Files to reformat data for training and add embeddings.

`/unused` – Old files that could come in handy later.

## File Descriptions
*If a description ends with a team member's name, Shiva did not modify this script and doesn't adhere to the same format as the other files.*

`classifiers/accuracy_metrics.py` – Helper function for outputting accuracy metrics (e.g. accuracy, F1, etc.).

`classifiers/embeddings/embedding_constants.py` – Script for embeddings approach that chooses label based on closeness to bank or risk description embeddings.

`classifiers/embeddings/nearest_embedding.py` – Script for embeddings approach that chooses label based on closest training sample.

`classifiers/embeddings/sbert_embeddings.py` – Script for generating labels based on sbert embeddings (Abhi's).

`classifiers/few_shot/gpt_zeroshot_fewshot.ipynb` – IPYNB for generating zero-shot / few-shot labels (Abhi's).

`classifiers/few_shot/prompts.json` – JSON file storing prompts for entity vs. risk, zero-shot vs. few-shot.

`classifiers/few_shot/zeroshot_fewshot.py` – Script for GPT approach that chooses label based on GPT output. Flexible for zero-shot and few-shot.

`classifiers/gpt_finetuning/convert_jsonl.py` – Script for modifying data to be GPT fine-tune-compatible (Ryan's).

`classifiers/gpt_finetuning/finetuned_gpt_script.py` – Script for finetuning GPT model with training data.

`classifiers/storage/bank_descriptions.json` – JSON file storing text descriptions corresponding to each bank.

`classifiers/storage/bank_embeddings.json` – JSON file storing embeddings corresponding to each bank, generated from `bank_descriptions.json`.

`classifiers/storage/generate_bank_embeddings.py` – Script for generating stored embeddings, both for entity-recognition and risk-recognition.

`classifiers/storage/risk_descriptions.json` – JSON file storing text descriptions corresponding to each risk classification class.

`classifiers/storage/risk_embeddings.json` – JSON file storing embeddings corresponding to each risk classification class, generated from `risk_descriptions.json`.

`eda/lda.ipynb` – IPYNB for running latent dirichlet allocation to explore raw data.

`eda/summary.ipynb` – IPYNB for generating some high-level summary statistics of raw data (e.g. counts, time distribution, retweet frequency).

`label_studio/convert_twitter_data.py` – Script for converting data into a form compatible with Label Studio (label training software).

`label_studio/label_studio_views.md` – Views file to use in Label Studio when assigning labels.

`preprocessing/openai_embeddings.py` – Script for generating OpenAI text embeddings for entire data files.

`preprocessing/preprocess.py` – Script for reformatting data scraped directly from `nlpgrid` to make it compatible with other scripts.
 
## Using a .env File
Note that our new file organization relies on having a `.env` in the base file path with the following attributes. This is meant to make our file names and variable unchangeable.
- OPENAI_API_KEY
- EMBEDDINGS_MODEL (OpenAI)
- CHAT_MODEL (OpenAI)
- PROJECT_PATH (Base Path to Project)
- SFTP_DATA_PATH (Base Path to raw nlpgrid data)
- BASE_DATA_PATH (Base Path to formatted data)
- LS_DATA_PATH (Base Path to Label Studio / Labeled folder data)
- EMBEDDINGS_DATA_PATH (Base Path to formatted data with embeddings)

## Notes
Make sure you are running these scripts in a `pipenv` environment started in the base level directory; this will guarantee your environment variables get loaded. 

To run commands and adhere to the use of relative file paths in some of the files, run commands like `python3 -m package.package` etc.

The first script you should run after pulling a `combined_bank.json` file from nlpgrid is running `preprocess.py`. Afterwards, if you are planning on using an embeddings approach for either entity classification or risk classification, run `openai_embeddings.py`. All embeddings scripts depend on having embeddings generated beforehand. Currently, all classifier scripts don't actually store the data anywhere, they just output accuracy metrics. Once we choose which models to pursue, we will use `aiolimiter` in a new script to asynchronously make many requests.

A quick note on the intended Label Studio workflow: There are three functions included in `convert_twitter_data.py`.
- The first function `format_for_entity_labeling()` formats data from the base data path to give them the attributes needed for using Label Studio. The first classification task should be for recognizing bank entities. Run this before using Label Studio for the first time.
- After labeling entities on Label Studio, delete the old file in your `LS_DATA_PATH` folder in the bank, and rename the exported file (JSON-MIN) from Label Studio to the bank name.
- The second function `format_for_risk_classification()` formats data from the label studio data path to clear out any unnecessary attributes and re-add new attributes needed for using Label Studio. The second classification task should be for classifying tweets as risky vs. non-risky. Run this before proceeding with risk classification and after entity labeling is done.
- After labeling entities on Label Studio, delete the old file in your `LS_DATA_PATH` folder in the bank, and rename the exported file (JSON-MIN) from Label Studio to the bank name.
- The third function `format_after_labeling()` reformats data to be used in any of the classifier scripts, removing any unnecessary attributes leftover from Label Studio. Run this after all labeling is complete and your use of Label Studio is done."

Another quick note, our `.json` data file default is to have new line delimited jsons. Label Studio however requires that these files are comma delimited (which might be a better approach).