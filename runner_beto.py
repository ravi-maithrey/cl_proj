import required_funcs as rf
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
import torch
import pandas as pd

from pysentimiento.preprocessing import preprocess_tweet

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

num_labels = 2  # For binary classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        value = {key: torch.tensor(val[0]) for key, val in encoding.items()}
        value["label"] = torch.tensor(label)
        return value


train = pd.read_json("dataset/training/EXIST2023_training.json", orient="index")
train = train.sample(frac=1).reset_index(drop=True)

# loading the validation datset
validation = pd.read_json("dataset/dev/EXIST2023_dev.json", orient="index")
validation = validation.sample(frac=1).reset_index(drop=True)

train_labels1 = train["labels_task1"]
train_lang = train["lang"]
train_text = train["tweet"]

train_labels_pre = rf.majority_vote(train_labels1)
train_labels = rf.label_convertor(train_labels_pre)

english_train, eng_train_labels, espanol_train, esp_train_labels = rf.split_lang(
    train_text, train_lang, train_labels
)

valid_labels1 = validation["labels_task1"]
valid_lang = validation["lang"]
valid_text = validation["tweet"]

valid_labels_pre = rf.majority_vote(valid_labels1)
valid_labels = rf.label_convertor(valid_labels_pre)

english_valid, eng_valid_labels, espanol_valid, esp_valid_labels = rf.split_lang(
    train_text, train_lang, train_labels
)

train_dataset = CustomDataset(espanol_train, esp_train_labels, tokenizer)
valid_dataset = CustomDataset(espanol_valid, esp_valid_labels, tokenizer)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer.train()
