import io
import os
import pathlib
from functools import partial

import torch
from torch.utils.data.dataset import Dataset
from torchtext import datasets
from torchtext.data import Field, LabelField, TabularDataset, Example
from torch.utils.data.dataset import random_split
from torchtext.data.utils import RandomShuffler
from torchtext.datasets import text_classification
from torchtext.utils import unicode_csv_reader
from torchtext.vocab import GloVe


class CSVDataset(Dataset):

    """
    Abstract class for Amazon, Yelp and Yahoo datasets
    """

    def __init__(self, val_ratio, embeddings_size, directory, reduced=False, word2vec="6B"):

        words = Field(batch_first=True, eos_token=".", tokenize="spacy")
        labels = LabelField(dtype=torch.long)

        if reduced: directory += "_reduced"
        training, test = TabularDataset.splits(path=directory, train='train.csv', test='test.csv', format='csv', fields=[('label', labels), ('text', words)])

        # Vocabs
        words.build_vocab(training, vectors=GloVe(name=word2vec, dim=embeddings_size))
        labels.build_vocab(training)

        # Validation Split
        full_train_len = len(training)
        val_len = int(val_ratio * full_train_len)
        train_len = int(full_train_len - val_len)
        training, validation = random_split(training, [train_len, val_len])

        training.fields = {'text': words, 'label': labels}
        test.fields = {'text': words, 'label': labels}
        validation.fields = {'text': words, 'label': labels}

        # Attributes
        self.n_classes = len(labels.vocab)
        self.n_words = len(words.vocab)
        self.training = training
        self.validation = validation
        self.test = test
        self.padding_value = words.vocab.itos.index(words.pad_token)
        self.end_of_sentence_value = words.vocab.itos.index(words.eos_token)
        self.sort_key = lambda example: example.text
        self.word2vec = words.vocab.vectors

    def __len__(self):
        return len(self.training)


class YelpDataset(CSVDataset):

    def __init__(self, embeddings_size, ngrams, full, reduced):
        self.full = full
        self.ngrams = ngrams
        directory = ".data/yelp_review_full_csv" if full else ".data/yelp_review_polarity_csv"
        super(YelpDataset, self).__init__(val_ratio=0.10, embeddings_size=embeddings_size, directory=directory, reduced=reduced)


class YahooDataset(CSVDataset):

    def __init__(self, embeddings_size, ngrams, reduced):
        self.ngrams = ngrams
        directory = ".data/yahoo_answers_csv"
        super(YahooDataset, self).__init__(val_ratio=0.10, embeddings_size=embeddings_size, directory=directory, reduced=reduced)


class AmazonDataset(CSVDataset):

    def __init__(self, embeddings_size, ngrams, full, reduced):
        self.full = full
        self.ngrams = ngrams
        directory = ".data/amazon_review_full_csv" if self.full else ".data/amazon_review_polarity_csv"
        super(AmazonDataset, self).__init__(val_ratio=0.10, embeddings_size=embeddings_size, directory=directory, reduced=reduced)


class IMDBDataset(Dataset):

    def __init__(self, embeddings_size, word2vec="6B"):
        words = Field(batch_first=True, eos_token=".", tokenize="spacy")
        labels = LabelField(dtype=torch.long)

        training, test = datasets.IMDB.splits(words, labels)
        training, validation = training.split()

        words.build_vocab(training, vectors=GloVe(name=word2vec, dim=embeddings_size))
        labels.build_vocab(training)

        self.n_classes = len(labels.vocab)
        self.n_words = len(words.vocab)
        self.padding_value = words.vocab.itos.index(words.pad_token)
        self.end_of_sentence_value = words.vocab.itos.index(words.eos_token)
        self.training = training
        self.validation = validation
        self.test = test
        self.sort_key = lambda example: example.text
        self.word2vec = words.vocab.vectors

    def __len__(self):
        return len(self.training)


def download_datasets(ngrams):

    root = ".data"
    reduced_size = 0.05 # 5% of the original

    # Download

    text_classification.YelpReviewFull(root=root, ngrams=ngrams)
    text_classification.YelpReviewPolarity(root=root, ngrams=ngrams)
    text_classification.AmazonReviewFull(ngrams=ngrams)
    text_classification.AmazonReviewPolarity(ngrams=ngrams)
    text_classification.YahooAnswers()

    # Create smaller sets
    directories = [
        "yahoo_answers_csv",
        "yelp_review_full_csv",
        "yelp_review_polarity_csv"
        "amazon_review_full_csv"
        "amazon_review_polarity_csv"
    ]
    for dataset_directory in directories:

        reduced_directory = root + "/" + dataset_directory + "_reduced"

        # 1 - Make reduced directory
        pathlib.Path(reduced_directory).mkdir(parents=True, exist_ok=True)

        # 2 - Reduce train.csv and test.csv (keeping label proportions)
        reduce_train_and_test_csv(reduced_directory, reduced_size)


def reduce_train_and_test_csv(reduced_dataset_directory, reduced_size):

    # TODO - Complete if there's time

    train_csv = reduced_dataset_directory + "/train.csv"
    test_csv = reduced_dataset_directory + "/test.csv"

    words = Field(batch_first=True, eos_token=".", tokenize="spacy")
    labels = LabelField(dtype=torch.long)
    fields = [('label', labels), ('text', words)]

    make_example = Example.fromCSV

    for path in [train_csv, test_csv]:

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            if isinstance(fields, dict):
                header = next(reader)
                field_to_index = {f: header.index(f) for f in fields.keys()}
                make_example = partial(make_example, field_to_index=field_to_index)
            examples = [make_example(line, fields) for line in reader]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list): fields.extend(field)
                else: fields.append(field)

        N = len(examples)
        randperm = RandomShuffler(range(N))
        N2 = int(round(reduced_size * N))

        index = randperm[:N2]

        reduced_dataset = [examples[i] for i in index]

        # TODO - Check code
        # TODO - Save new smaller dataset