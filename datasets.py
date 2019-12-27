import torch
from torch.utils.data.dataset import Dataset
from torchtext import datasets
from torchtext.data import Field, LabelField


class Yelp13Dataset(Dataset):

    def __init__(self):

        self.n_classes = None
        self.n_words = None
        self.training = None
        self.validation = None
        self.test = None

    def __len__(self):
        return len(self.training)


class Yelp14Dataset(Dataset):

    def __init__(self):

        self.n_classes = None
        self.n_words = None
        self.training = None
        self.validation = None
        self.test = None

    def __len__(self):
        return len(self.training)


class Yelp15Dataset(Dataset):

    def __init__(self):

        self.n_classes = None
        self.n_words = None
        self.training = None
        self.validation = None
        self.test = None

    def __len__(self):
        return len(self.training)


class YahooDataset(Dataset):

    def __init__(self):

        self.n_classes = None
        self.n_words = None
        self.training = None
        self.validation = None
        self.test = None

    def __len__(self):
        return len(self.training)


class IMDBDataset(Dataset):

    def __init__(self):

        words = Field(batch_first=True)
        labels = LabelField(dtype=torch.long)
        training, test = datasets.IMDB.splits(words, labels)
        training, validation = training.split()
        words.build_vocab(training)
        labels.build_vocab(training)

        self.n_classes = len(labels.vocab)
        self.n_words = len(words.vocab)
        self.padding_value = words.vocab.itos.index(words.pad_token)
        self.training = training
        self.validation = validation
        self.test = test

    def __len__(self):
        return len(self.training)


class AmazonDataset(Dataset):

    def __init__(self):

        self.n_classes = None
        self.n_words = None
        self.training = None
        self.validation = None
        self.test = None

    def __len__(self):
        return len(self.training)

