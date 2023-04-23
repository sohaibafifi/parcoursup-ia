import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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

tokenizer = FlaubertTokenizer.from_pretrained( modelname, do_lowercase=False)



# Create PyTorch Dataset
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


# Train-test split
train_texts, test_texts, train_scores, test_scores = train_test_split(texts, scores, test_size=0.2, random_state=42)

# Encode the texts
train_encoded_data = tokenizer([str(text) for text in train_texts], padding=True, truncation=True, return_tensors="pt")
test_encoded_data = tokenizer([str(text) for text in test_texts], padding=True, truncation=True, return_tensors="pt")

train_dataset = ApplicationDataset(train_encoded_data, train_scores)

# Model and DataLoader
model = FlaubertForSequenceClassification.from_pretrained(modelname, num_labels=1)

train_loader = DataLoader(train_dataset, batch_size=32)

# Training loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
num_epochs = 3

for epoch in range(num_epochs):
    print('epoch {0}'.format(epoch))
    model.train()
    for batch in tqdm(train_loader):
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = torch.nn.MSELoss()(outputs.logits.squeeze(-1), labels)
        loss.backward()
        optimizer.step()
torch.save(model.state_dict(), "data/processed/model_state.pth")
