import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchnlp.nn import Attention

"""
If three classes turn out to be very similar, 
we refactor their code in the end
"""


class LSTMClassifier(nn.Module):

    """
    Very simple LSTM Classifier to test the datasets.
    """

    def __init__(self, n_classes, n_words, embedding_size, layers, hidden_sizes, bidirectional, dropout, padding_value, device):

        super().__init__()

        self.embedder = nn.Embedding(n_words, embedding_size, padding_idx=padding_value)
        self.lstm = nn.LSTM(embedding_size, hidden_sizes, layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        if bidirectional: hidden_sizes *= 2
        self.bidirectional = bidirectional
        self.hidden_to_label = nn.Linear(hidden_sizes, n_classes)

        self.device = device
        self.to(device)

    def forward(self, X):
        embeddings = self.embedder(X)
        _, (hidden, _) = self.lstm(embeddings)
        hidden_last = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.bidirectional else hidden[-1]
        scores = self.hidden_to_label(hidden_last)
        return scores


class HierarchicalAttentionNetwork(nn.Module):

    """ Original model from https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf"""

    def __init__(self, n_classes, n_words, embedding_size, layers, hidden_sizes, dropout, padding_value, eos_value, device):

        super(HierarchicalAttentionNetwork, self).__init__()

        self.padding_value = padding_value
        self.end_of_sentence_value = eos_value

        # TODO -> Load pretrained Word2Vec
        self.embedder = nn.Embedding(n_words, embedding_size, padding_idx=padding_value)

        self.word_encoder = nn.GRU(embedding_size, hidden_sizes, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.word_attention = Attention(hidden_sizes*2) # TODO - Check attention type

        self.sentence_encoder = nn.GRU(hidden_sizes * 2, hidden_sizes, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.sentence_attention = Attention(hidden_sizes*2) # TODO - Check attention type

        self.hidden_to_label = nn.Linear(hidden_sizes * 2, n_classes)

        self.device = device
        self.to(device)

    def forward(self, X):

        documents_as_sentences = []

        for x in X: # TODO - Can this be vectorized?

            # Sentence batch: L words [SxL]
            document = self.split_into_sentences(x)

            # Sentence batch: L words, E embeddings [SxLxE]
            words = self.embedder(document)
            word_encodings = self.word_encoder(words)[1]
            # TODO -> Add word level attention

            # Document: S sentences of 2H gru-units [1xSx2H]
            sentences = torch.cat((word_encodings[-2], word_encodings[-1]), dim=1)
            documents_as_sentences.append(sentences)

        # Documents batch: S sentences, 2H gru-units [BxSx2H]
        documents_as_sentences = pad_sequence(documents_as_sentences, batch_first=True)
        sentence_encodings = self.sentence_encoder(documents_as_sentences)[1]
        # TODO -> Add word level attention

        # Batch of document "features": 2H gru-units [Bx2H]
        document = torch.cat((sentence_encodings[-2], sentence_encodings[-1]), dim=1)

        # Batch of document "scores": num_classes outputs [BxK]
        scores = self.hidden_to_label(document)

        return scores

    def split_into_sentences(self, document):
        ends_of_sentence = (document == self.end_of_sentence_value).nonzero()
        sentences = [document[0:eos + 1] if i == 0 else document[ends_of_sentence[i - 1] + 1:eos + 1] for i, eos in enumerate(ends_of_sentence)]
        sentences.append(document[ends_of_sentence[-1] + 1:])
        document = pad_sequence(sentences, batch_first=True, padding_value=self.padding_value)
        return document


class PrunedHierarchicalAttentionNetwork(nn.Module):

    # TODO

    def __init__(self, device):
        super(PrunedHierarchicalAttentionNetwork, self).__init__()
        self.device = device
        self.to(device)

    def forward(self, X):
        # TODO
        scores = None
        return scores


class HierarchicalSparsemaxAttentionNetwork(nn.Module):

    # TODO

    def __init__(self, device):
        super(HierarchicalSparsemaxAttentionNetwork, self).__init__()
        self.device = device
        self.to(device)

    def forward(self, X):
        # TODO
        scores = None
        return scores
