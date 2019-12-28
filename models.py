import torch
from torch import nn
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
        self.bilstm = nn.LSTM(embedding_size, hidden_sizes, layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        if bidirectional: hidden_sizes *= 2
        self.bidirectional = bidirectional
        self.hidden_to_label = nn.Linear(hidden_sizes, n_classes)

        self.device = device
        self.to(device)

    def forward(self, X):
        embeddings = self.embedder(X)
        _, (hidden, _) = self.bilstm(embeddings)
        hidden_last = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.bidirectional else hidden[-1]
        scores = self.hidden_to_label(hidden_last)
        return scores


class HierarchicalAttentionNetwork(nn.Module):

    """ Original model from https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf"""

    def __init__(self, n_classes, n_words, embedding_size, hidden_sizes, layers, dropout, padding_value, device):

        super(HierarchicalAttentionNetwork, self).__init__()

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

        word_embeddings = self.embedder(X)
        # TODO
        scores = None
        return scores


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
