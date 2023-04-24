import pandas as pd
import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load the best checkpoint
checkpoint_path = "./output/xlm_checkpoint-2500"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

# Load the tokenizer
model_name = (
    "xlm-roberta-base"  # or any other pre-trained MLM model available on Hugging Face
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a classification pipeline
classifier = pipeline(
    "text-classification", model=model, tokenizer=tokenizer, device=0
)  # Set device to 0 if using GPU

# Read the test data into a pandas DataFrame
test_data = pd.read_json("dataset/test/EXIST2023_test_clean.json")


# Define a function to classify the tweets and map the predicted labels
def classify_tweet(tweet):
    prediction = classifier(tweet)
    predicted_label = prediction[0]["label"]
    predicted_index = int(predicted_label.split("_")[-1])
    return "YES" if predicted_index == 1 else "NO"


# Apply the function to the 'tweet' column and store the results in a new column
test_data["hard_label"] = test_data["tweet"].apply(classify_tweet)

# Prepare the data for JSON output
output_data = test_data[["id_EXIST", "hard_label"]]
output_dict = output_data.set_index("id_EXIST").T.to_dict("records")[0]
nested_output_dict = {key: {"hard_label": value} for key, value in output_dict.items()}

# Write the results to a new JSON file
with open("output.json", "w") as outfile:
    json.dump(nested_output_dict, outfile)
