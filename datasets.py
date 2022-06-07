from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from settings import DATA_DIR


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
    zs = np.load(os.path.join(DATA_DIR, 'zs.npy'))
    features = np.load(os.path.join(DATA_DIR, 'downscaled.npy'))
    features = np.reshape(features, [10000, -1])
    return zs, features


def get_dataloader():
    data = load_data()
    dataset = StyleganDataset(*data)
    return DataLoader(dataset=dataset, batch_size=64, shuffle=True)