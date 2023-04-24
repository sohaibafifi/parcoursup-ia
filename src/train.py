import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification

from ApplicationDataset import ApplicationDataset, read_files

# Read data from multiple Excel files
years = [2022]  # range(2015, 2023)  # Update the range according to your data
texts, scores, codes  = read_files(years)


# Pre-process the data
# Choose among ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased',
#               'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']
modelname = 'flaubert/flaubert_base_uncased'

tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=True)


# Train-test split
train_texts, test_texts, train_scores, test_scores = train_test_split(texts, scores, test_size=0.2, random_state=42)

# Encode the texts
train_encoded_data = tokenizer([str(text) for text in train_texts], padding=True, truncation=True, return_tensors="pt")
test_encoded_data = tokenizer([str(text) for text in test_texts], padding=True, truncation=True, return_tensors="pt")

train_dataset = ApplicationDataset(train_encoded_data, train_scores)
test_dataset = ApplicationDataset(test_encoded_data, test_scores)

# Model and DataLoader
model = FlaubertForSequenceClassification.from_pretrained(modelname, num_labels=1)

train_loader = DataLoader(train_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# Training loop
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
device = 'cuda' if torch.cuda.is_available() else 'mps' if mps_available else 'cpu'

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
num_epochs = 3

for epoch in range(num_epochs):
    # Train the model
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader):
        inputs = {key: val.to(device) for key, val in batch.items()}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Evaluate the model
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            validation_loss += loss.item()

    validation_loss /= len(test_loader)

    print(f"Epoch: {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}")

torch.save(model.state_dict(), "data/processed/model_state.pth")
