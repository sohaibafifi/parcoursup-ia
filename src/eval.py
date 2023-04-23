import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification

# Read data from multiple Excel files
years = [2022]  # range(2015, 2023)  # Update the range according to your data
texts, scores = [], []

for year in years:
    excel_file = f'data/raw/applications_{year}.xlsx'
    df = pd.read_excel(excel_file)

    # Assume the Excel file has two columns: 'motivations' and 'score'
    texts.extend(df['motivations'].tolist())
    scores.extend(df['score'].tolist())

# Pre-process the data
# Choose among ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased',
#               'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']
modelname = 'flaubert/flaubert_base_uncased'

tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)

# Train-test split
train_texts, test_texts, train_scores, test_scores = train_test_split(texts, scores, test_size=0.2, random_state=42)

# Encode the texts
test_encoded_data = tokenizer([str(text) for text in test_texts], padding=True, truncation=True, return_tensors="pt")


class ApplicationDataset(Dataset):
    def __init__(self, encoded_data, scores):
        self.encoded_data = encoded_data
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoded_data.items()}
        item['labels'] = torch.tensor(self.scores[idx], dtype=torch.float)
        return item


test_dataset = ApplicationDataset(test_encoded_data, test_scores)

# Model and DataLoader
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
device = 'cuda' if torch.cuda.is_available() else 'mps' if mps_available else 'cpu'
# Initialize a new model with the same architecture
model = FlaubertForSequenceClassification.from_pretrained(modelname, num_labels=1)

model.to(device)
# Load the saved model state
state_dict = torch.load("data/processed/model_state.pth")
model.load_state_dict(state_dict)
model.to(device)

# Evaluation
model.eval()
predictions, true_labels = [], []
test_loader = DataLoader(test_dataset, batch_size=8)
with torch.no_grad():
    for batch in test_loader:
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs)
        predictions.extend(outputs.logits.squeeze(-1).tolist())
        true_labels.extend(labels.tolist())

mse = mean_squared_error(true_labels, predictions)
print(f"Mean Squared Error: {mse}")
