import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchnlp.nn import Attention
import torch.nn.functional as F

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

        self.word_hidden_representation = nn.Sequential(nn.Linear(hidden_sizes * 2, hidden_sizes * 2), nn.Tanh())
        self.word_context_vector = nn.Parameter(torch.Tensor(hidden_sizes * 2)) # TODO -> Check initialization

        self.sentence_encoder = nn.GRU(hidden_sizes * 2, hidden_sizes, layers, batch_first=True, bidirectional=True, dropout=dropout)

        self.sentence_hidden_representation = nn.Sequential(nn.Linear(hidden_sizes * 2, hidden_sizes * 2), nn.Tanh())
        self.sentence_context_vector = nn.Parameter(torch.Tensor(hidden_sizes * 2))  # TODO -> Check initialization

        self.hidden_to_label = nn.Linear(hidden_sizes * 2, n_classes)

        self.device = device
        self.to(device)

    def forward(self, X):

        sentences = []

        for x in X: # TODO - Check if this can be vectorized?

            # S x L
            document = self.split_into_sentences(x)
            S, L = document.shape

            # S x L x E
            words = self.embedder(document)

            # S x L x 2H
            hidden, _ = self.word_encoder(words)

            # S x L x 2H
            hidden_representations = self.word_hidden_representation(hidden)

            # S x L
            attention_weights = F.softmax(hidden_representations.matmul(self.word_context_vector), dim=1)

            # S x 2H FIXME -> Is there a way to rewrite this using matmul or bmm or ...?
            sentences.append(torch.stack([torch.stack([attention_weights[i, t] * hidden[i, t] for t in range(L)]).sum(0) for i in range(S)]))

        # Documents batch: S sentences, 2H gru-units [BxSx2H]
        sentences = pad_sequence(sentences, batch_first=True)
        #sentence_encodings = self.sentence_encoder(sentences)[1]
        # B x S x 2H
        hidden, _ = self.sentence_encoder(sentences)

        # B x S x 2H
        hidden_representations = self.sentence_hidden_representation(hidden)

        # B x S
        attention_weights = F.softmax(hidden_representations.matmul(self.sentence_context_vector), dim=1)

        # Batch of document "features": 2H gru-units [Bx2H]
        # document = torch.cat((sentence_encodings[-2], sentence_encodings[-1]), dim=1)
        B = X.shape[0]
        document = torch.stack(
            [torch.stack([attention_weights[i, t] * hidden[i, t] for t in range(S)]).sum(0) for i in range(B)])

        # Batch of document "scores": num_classes outputs [BxK]
        scores = self.hidden_to_label(document)

        return scores

    def split_into_sentences(self, document):
        """
            Given a document as sequence (shape L1: total length)
            Returns a document as sentences (shape SxL2)
        """
        ends_of_sentence = (document == self.end_of_sentence_value).nonzero()
        sentences = [document[0:eos + 1] if i == 0 else document[ends_of_sentence[i - 1] + 1:eos + 1] for i, eos in enumerate(ends_of_sentence)]
        sentences.append(document[ends_of_sentence[-1] + 1:])
        #TODO - Check last sentence for non pad values
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
