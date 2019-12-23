from collections import defaultdict
from itertools import count

import torch
from torch.utils.data.dataset import Dataset


class OCRDataset(Dataset):

    """
        Binary OCR dataset.
        Used for testing the main skeleton. Dont remove me.
    """

    def __init__(self, path, dev_fold=8, test_fold=9):
        """
        path: location of OCR data
        """
        label_counter = count()
        labels = defaultdict(lambda: next(label_counter))
        X = []
        y = []
        fold = []
        with open(path) as f:
            for line in f:
                tokens = line.split()
                pixels = [int(t) for t in tokens[6:]]
                letter = labels[tokens[1]]
                fold.append(int(tokens[5]))
                X.append(pixels)
                y.append(letter)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        fold = torch.tensor(fold, dtype=torch.long)

        # boolean masks, not indices
        train_idx = (fold != dev_fold) & (fold != test_fold)
        dev_idx = fold == dev_fold
        test_idx = fold == test_fold

        self.X = X[train_idx]
        self.y = y[train_idx]

        self.X_dev = X[dev_idx]
        self.y_dev = y[dev_idx]

        self.X_test = X[test_idx]
        self.y_test = y[test_idx]

        self.n_classes = torch.unique(self.y).shape[0]  # 26
        self.n_features = self.X.shape[1]               # 128

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Yelp13Dataset(Dataset):

    def __init__(self, path):

        # TODO

        self.X = None
        self.y = None

        self.X_dev = None
        self.y_dev = None

        self.X_test = None
        self.y_test = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Yelp14Dataset(Dataset):

    def __init__(self, path):

        # TODO

        self.X = None
        self.y = None

        self.X_dev = None
        self.y_dev = None

        self.X_test = None
        self.y_test = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Yelp15Dataset(Dataset):

    def __init__(self, path):
        # TODO

        self.X = None
        self.y = None

        self.X_dev = None
        self.y_dev = None

        self.X_test = None
        self.y_test = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class YahooDataset(Dataset):

    def __init__(self, path):
        # TODO

        self.X = None
        self.y = None

        self.X_dev = None
        self.y_dev = None

        self.X_test = None
        self.y_test = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class IMDBDataset(Dataset):

    def __init__(self, path):
        # TODO

        self.X = None
        self.y = None

        self.X_dev = None
        self.y_dev = None

        self.X_test = None
        self.y_test = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AmazonDataset(Dataset):

    def __init__(self, path):
        # TODO

        self.X = None
        self.y = None

        self.X_dev = None
        self.y_dev = None

        self.X_test = None
        self.y_test = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
