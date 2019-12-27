import torch
from torch import nn

"""
If three classes turn out to be very similar, 
we refactor their code in the end
"""


class LSTMClassifier(nn.Module):

    """
    Very simple LSTM Classifier to test the datasets.
    """

    def __init__(self, n_classes, n_inputs, embedding_size, layers, hidden_sizes, bidirectional, dropout, padding_value, device):

        super().__init__()

        self.embedding = nn.Embedding(n_inputs, embedding_size, padding_idx=padding_value)
        self.bilstm = nn.LSTM(embedding_size, hidden_sizes, layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        if bidirectional: hidden_sizes *= 2
        self.bidirectional = bidirectional
        self.hidden_to_output = nn.Linear(hidden_sizes, n_classes)

        self.device = device
        self.to(device)

    def forward(self, X):
        embeddings = self.embedding(X)
        _, (hidden, _) = self.bilstm(embeddings)
        hidden_last = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.bidirectional else hidden[-1]
        scores = self.hidden_to_output(hidden_last)
        return scores


class HierarchicalAttentionNetwork(nn.Module):

    """ Original model from https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf"""

    # TODO

    def __init__(self, device):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.device = device
        self.to(device)

    def forward(self, X):
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
