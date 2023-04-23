import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification

# Read data from multiple Excel files
years = [2023]  # range(2015, 2023)  # Update the range according to your data
texts, scores, codes  = [], [], []

for year in years:
    excel_file = f'data/raw/applications_{year}.xlsx'
    df = pd.read_excel(excel_file)

    # Assume the Excel file has two columns: 'motivations' and 'score'
    texts.extend(df['motivations'].tolist())
    scores.extend(df['score'].tolist())
    codes.extend(df['code'].tolist())

# Pre-process the data
# Choose among ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased',
#               'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']
modelname = 'flaubert/flaubert_base_uncased'

tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)


# Encode the texts
test_encoded_data = tokenizer([str(text) for text in texts], padding=True, truncation=True, return_tensors="pt")


class ApplicationDataset(Dataset):
    def __init__(self, encoded_data, scores, codes):
        self.encoded_data = encoded_data
        self.scores = scores
        self.codes = codes

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoded_data.items()}
        item['labels'] = torch.tensor(self.scores[idx], dtype=torch.float)
        item['codes'] = torch.tensor(self.codes[idx], dtype=torch.int32)
        return item


test_dataset = ApplicationDataset(test_encoded_data, scores, codes)

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
predictions = []
test_loader = DataLoader(test_dataset, batch_size=8)
results = {}
with torch.no_grad():
    for batch in tqdm(test_loader):
        print(batch.keys())
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels' and key != 'codes'}
        outputs = model(**inputs)
        codes = batch['codes']
        results.update({code: prediction for code, prediction in zip(codes.tolist(), outputs.logits.squeeze(-1).tolist())})

    for code, score in results.items() :
        print('{} => {}'.format(code, score))
# ...

