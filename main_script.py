import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from torch.utils.data import Dataset
import torch
import os

# Specify the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load the dataset
os.chdir('C:/Users/Praxis/Documents')
df = pd.read_csv('combined_data.csv')

# Preprocess the data
df['text'] = df['text'].apply(lambda x: x.lower())  # convert to lowercase

# Take only the bottom part of your dataset
bottom_percentage = 0.2  # Set the percentage of the dataset to use

# Get the bottom part of the dataset
bottom_df = df.tail(int(bottom_percentage * len(df)))

# Split the data into training and testing sets for the bottom part
train_texts, test_texts, train_labels, test_labels = train_test_split(bottom_df['text'], bottom_df['label'], test_size=0.2)

# Load a pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Move the model to the device
model.to(device)
print("Model is on device:", next(model.parameters()).device)

# Convert texts to input IDs
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True)

# Create a Dataset class
class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Convert our data into torch Dataset for the bottom part
train_dataset = EmailDataset(train_encodings, train_labels.tolist())
test_dataset = EmailDataset(test_encodings, test_labels.tolist())

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training (increased)
    per_device_eval_batch_size=128,  # batch size for evaluation (increased)
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluation is performed at the end of each epoch
    save_strategy="epoch",           # save model at the end of each epoch
    fp16=True if device.type == 'cuda' else False,  # mixed precision training if GPU is available
    load_best_model_at_end=True      # Load the best model found during training at the end of training
)

# Create the Trainer and train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')

# Evaluate the model
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(-1)

accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred)
recall = recall_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
