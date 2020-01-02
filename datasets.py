from abc import abstractmethod
import csv
import os
import shutil


import torch
from torch.utils.data.dataset import Dataset
from torchtext import datasets
from torchtext.data import Field, LabelField, TabularDataset
from torch.utils.data.dataset import random_split
from torchtext.datasets import text_classification
from torchtext.vocab import GloVe


def split_csv_files(path, dataset_size):
    print("==== Split train.csv and test.csv files =====")
    for file_name in ["/train", "/test"]:
        print(path + file_name)

        with open(path + file_name + ".csv") as infile:
            reader = csv.DictReader(infile)
            header = reader.fieldnames
            rows = [row for row in reader]

            csv_rows = rows[0: int(len(rows) * dataset_size)]

            with open(path + '{}_sample.csv'.format(file_name), 'w', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=header)
                writer.writerows(csv_rows)

        print('Delete {}.csv'.format(file_name))
        os.remove(path + file_name + '.csv')

        print('Rename {}_sample.csv to {}.csv'.format(file_name, file_name))
        os.rename(path + file_name + "_sample.csv", path + file_name + ".csv")


def remove_path_if_exist(path):
    # É preciso apagar a pasta de forma a gerar os 20% do dataset com o original

    if os.path.exists(path):
        shutil.rmtree(path)


class BaseDataset(Dataset):

    def __init__(self, val_ratio, embeddings_size, word2vec="6B"):
        words = Field(batch_first=True, eos_token=".", tokenize="spacy")
        labels = LabelField(dtype=torch.long)

        directory = self.load_dataset(".data")
        training, test = TabularDataset.splits(path=directory, train='train.csv', test='test.csv', format='csv',
                                               fields=[('label', labels), ('text', words)])

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

    @abstractmethod
    def load_dataset(self, root):
        raise NotImplementedError()

    def __len__(self):
        return len(self.training)


class YelpDataset(BaseDataset):

    def __init__(self, embeddings_size, ngrams, full=True, debug=False, sample=False, dataset_size=.01):
        self.full = full
        self.ngrams = ngrams
        self.debug = debug
        self.sample = sample
        self.dataset_size = dataset_size
        super(YelpDataset, self).__init__(val_ratio=0.10,
                                          embeddings_size=embeddings_size)  # TODO - Confirm correct val ratio

    def load_dataset(self, root):
        path = ".data/yelp_review_full_csv" if self.full else ".data/yelp_review_polarity_csv"
        if not self.debug:
            if self.sample:
                remove_path_if_exist(path)
            if self.full:
                text_classification.YelpReviewFull(ngrams=self.ngrams, root=root)
            else:
                text_classification.YelpReviewPolarity(ngrams=self.ngrams, root=root)
        else:
            path += "_debug"
        if not self.debug and self.sample:
            split_csv_files(path, self.dataset_size)
        return path


class YahooDataset(BaseDataset):

    def __init__(self, embeddings_size, ngrams, debug=False, sample=False, dataset_size=0.01):
        self.ngrams = ngrams
        self.debug = debug
        self.sample = sample
        self.dataset_size = dataset_size
        super(YahooDataset, self).__init__(val_ratio=0.10,
                                           embeddings_size=embeddings_size)  # TODO - Confirm correct val ratio

    def load_dataset(self, root):
        path = ".data/yahoo_answers_csv"
        if not self.debug:
            if self.sample:
                remove_path_if_exist(path)
            text_classification.YahooAnswers(ngrams=self.ngrams)
        else:
            path += "_debug"
        if not self.debug and self.sample:
            split_csv_files(path, self.dataset_size)
        return path


class AmazonDataset(BaseDataset):

    def __init__(self, embeddings_size, ngrams, full=True, debug=False, sample=False, dataset_size=0.01):
        self.full = full
        self.ngrams = ngrams
        self.debug = debug
        self.sample = sample
        self.dataset_size=dataset_size
        super(AmazonDataset, self).__init__(val_ratio=0.10,
                                            embeddings_size=embeddings_size)  # TODO - Confirm correct val ratio

    def load_dataset(self, root):
        path = ".data/amazon_review_full_csv" if self.full else ".data/amazon_review_polarity_csv"
        if not self.debug:
            if self.sample:
                remove_path_if_exist(path)
            if self.full:
                text_classification.AmazonReviewFull(ngrams=self.ngrams)
            else:
                text_classification.AmazonReviewPolarity(ngrams=self.ngrams)
        else:
            path += "_debug"
        if not self.debug and self.sample:
            split_csv_files(path, self.dataset_size)
        return path


class IMDBDataset(Dataset):

    def __init__(self, embeddings_size, word2vec="6B", sample = False, dataset_size = 0.01):
        words = Field(batch_first=True, eos_token=".", tokenize="spacy")
        labels = LabelField(dtype=torch.long)
        self.sample = sample
        self.dataset_size = dataset_size

        training, test = datasets.IMDB.splits(words, labels)
        if self.sample:
            # TODO -> ajuda me a validar se isto funciona
            print("Validar")
            # training = training * dataset_size
            # test = test * dataset_size
            #--------------------------------
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
