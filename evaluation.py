import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
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

# Calculate the index to split data
bottom_percentage = 0.2  # Bottom 20% used for training
split_index = int(len(df) * (1 - bottom_percentage))

# Load a pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Load your trained model from the specified directory
model_path = 'C:/Users/Praxis/Documents/trained_model'
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Move the model to the device
model.to(device)
print("Model is on device:", next(model.parameters()).device)

# Tokenize and encode the remaining 80% of texts
encodings = tokenizer(df['text'][split_index:].tolist(), truncation=True, padding=True)

# Create a Dataset class for evaluation
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

# Convert our data into torch Dataset
dataset = EmailDataset(encodings, df['label'][split_index:].tolist())

# Evaluate the model
model.eval()  # Set the model to evaluation mode
predictions = []
true_labels = []

with torch.no_grad():
    for batch in torch.utils.data.DataLoader(dataset, batch_size=32):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
