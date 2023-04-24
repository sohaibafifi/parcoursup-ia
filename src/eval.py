import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification

from ApplicationDataset import ApplicationDataset, read_files

# Read data from multiple Excel files
years = [2023]  # range(2015, 2023)  # Update the range according to your data
texts, scores, codes  = read_files(years)

# Pre-process the data
# Choose among ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased',
#               'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']
modelname = 'flaubert/flaubert_base_uncased'

tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=True)


# Encode the texts
test_encoded_data = tokenizer([str(text) for text in texts], padding=True, truncation=True, return_tensors="pt")

test_dataset = ApplicationDataset(test_encoded_data, scores, codes)

# Model and DataLoader
device = 'cpu'
# Initialize a new model with the same architecture
model = FlaubertForSequenceClassification.from_pretrained(modelname, num_labels=1)

model.to(device)
# Load the saved model state
state_dict = torch.load("data/processed/model_state.pth")
model.load_state_dict(state_dict)

# Evaluation
model.eval()
predictions = []
test_loader = DataLoader(test_dataset, batch_size=1)
results = {}
with torch.no_grad():
    for batch in tqdm(test_loader):
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels' and key != 'codes'}
        outputs = model(**inputs)
        codes = batch['codes']
        results.update({code: prediction for code, prediction in zip(codes.tolist(), outputs.logits.squeeze(-1).tolist())})

    results = dict(sorted(results.items(), key=lambda x:x[1]))
    for code, score in results.items() :
        print('{} => {}'.format(code, score))

# ...

