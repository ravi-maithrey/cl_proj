import pandas as pd
import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from pysentimiento import preprocess_tweet

# Load the checkpoints and tokenizers for different languages
checkpoint_paths = {
    "es": "./output/robertuito_checkpoint-2500",
    "other": "./output/roberta_checkpoint-2500",
}
model_names = {"es": "robertuito-uncased-base", "other": "roberta-base"}

models = {
    lang: AutoModelForSequenceClassification.from_pretrained(path)
    for lang, path in checkpoint_paths.items()
}
tokenizers = {
    lang: AutoTokenizer.from_pretrained(name) for lang, name in model_names.items()
}


# Define a custom preprocessing function for the Spanish tokenizer
def preprocess_and_tokenize_es(texts):
    preprocessed_texts = [preprocess_tweet(text) for text in texts]
    return tokenizers["es"](preprocessed_texts)


# Create classification pipelines for different languages
classifiers = {
    "es": pipeline(
        "text-classification",
        model=models["es"],
        tokenizer=preprocess_and_tokenize_es,
        device=0,
    ),
    "other": pipeline(
        "text-classification",
        model=models["other"],
        tokenizer=tokenizers["other"],
        device=0,
    ),
}

# Read the test data into a pandas DataFrame
test_data = pd.read_json("dataset/test/EXIST2023_test_clean.json")


# Define a function to classify the tweets and map the predicted labels
def classify_tweet(tweet, language):
    classifier = classifiers["es"] if language == "es" else classifiers["other"]
    prediction = classifier(tweet)
    predicted_label = prediction[0]["label"]
    predicted_index = int(predicted_label.split("_")[-1])
    return "YES" if predicted_index == 1 else "NO"


# Apply the function to the 'tweet' column based on the 'language' column and store the results in a new column
test_data["hard_label"] = test_data.apply(
    lambda row: classify_tweet(row["tweet"], row["language"]), axis=1
)

# Prepare the data for JSON output
output_data = test_data[["id_EXIST", "hard_label"]]
output_dict = output_data.set_index("id_EXIST").T.to_dict("records")[0]
nested_output_dict = {key: {"hard_label": value} for key, value in output_dict.items()}

# Write the results to a new JSON file
with open("output.json", "w") as outfile:
    json.dump(nested_output_dict, outfile)
