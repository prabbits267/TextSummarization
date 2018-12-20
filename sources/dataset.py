from torch.utils.data import Dataset, DataLoader
import pandas as pd
from config import *

class Seq2SeqDataset(Dataset):
    def __init__(self):
        self.x_data, self.y_data = self.read_data()
        self.len = len(self.x_data)
        self.word_type = self.get_wtype()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def get_wtype(self):
        full_text = self.x_data + self.y_data
        full_text = ' '.join(full_text)
        return sorted(set(full_text.split()))

    def read_data(self):
        d_frame = pd.read_csv(RESOURCES_PROCESSED, encoding='utf-8')
        x_data = list(d_frame.iloc[:, 1])
        y_data = list(d_frame.iloc[:, 0])
        return x_data, y_data