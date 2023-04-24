import torch
from torch.utils.data import Dataset
import pandas as pd




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


def read_files(years = ['2022']) :
    texts, scores, codes = [], [], []

    for year in years:
        excel_file = f'data/raw/applications_{year}.xlsx'
        df = pd.read_excel(excel_file)

        # Assume the Excel file has two columns: 'motivations' and 'score'
        texts.extend(df['motivations'].tolist())
        scores.extend(df['score'].tolist())
        codes.extend(df['code'].tolist())
    return texts, scores, codes
