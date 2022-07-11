import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from settings import DATA_DIR, BATCH_SIZE


class StyleganDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y


def load_data():
    ws = np.load(os.path.join(DATA_DIR, 'ws.npy'))
    features = np.load(os.path.join(DATA_DIR, 'downscaled.npy'))
    features = np.reshape(features, [20000, -1])
    return ws, features


def get_dataloader(shuffle=True):
    data = load_data()
    dataset = StyleganDataset(*data)
    return DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
