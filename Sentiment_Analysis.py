## Importing required libraries

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import torch
from torch.utils.data import Dataset

## Loading the Datasets

train_df = pd.read_csv("train_2kmZucJ.csv")
test_df = pd.read_csv("test_oJQbWVk.csv")

## Test-train split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["tweet"].tolist(), train_df["label"].tolist(), test_size=0.1, stratify=train_df["label"], random_state=42
)

## Loading the Alberta model

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_texts, train_labels, tokenizer)
val_dataset = TweetDataset(val_texts, val_labels, tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=False, 
    logging_dir="./logs",
)

## Training the model

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,

)

trainer.train()


## Unfortunately my kernel died after training, so I had to reload the model from checkpoint and the tokenizer as well

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "/home/tuaalab/Sentiment Analysis/results/checkpoint-1338"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment") 


## Saving the results


import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm  

test_encodings = tokenizer(test_df["tweet"].tolist(), truncation=True, padding=True, return_tensors="pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Preparing DataLoader for batching
dataset = TensorDataset(
    test_encodings["input_ids"],
    test_encodings["attention_mask"]
)
loader = DataLoader(dataset, batch_size=32)  

# Running inference in batches
all_preds = []
with torch.no_grad():
    for batch in tqdm(loader):
        input_ids, attention_mask = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, axis=1)
        all_preds.extend(preds.cpu().numpy())

# Saving predictions
submission = pd.DataFrame({"id": test_df["id"], "label": all_preds})
submission.to_csv("Predictions_Final.csv", index=False)





