import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sparseMax import SparseMax

"""
If three classes turn out to be very similar, 
we refactor their code in the end
"""


def split_into_sentences(document, padding_value, eos_value):
    """
        Given a document as sequence (shape L1: total length)
        Returns a document as sentences (shape SxL2)
    """
    ends_of_sentence = (document == eos_value).nonzero()
    sentences = [document[0:eos + 1] if i == 0 else document[ends_of_sentence[i - 1] + 1:eos + 1] for i, eos in enumerate(ends_of_sentence)]
    sentences.append(document[ends_of_sentence[-1] + 1:])
    # TODO - Check last sentence for non pad values
    document = pad_sequence(sentences, batch_first=True, padding_value=padding_value)
    return document


class LSTMClassifier(nn.Module):

    """
    Very simple LSTM Classifier to test the datasets.
    """

    def __init__(self, n_classes, n_words, embeddings, layers, hidden_sizes, bidirectional, dropout, padding_value, device):

        super().__init__()

        self.embedder = nn.Embedding(n_words, embeddings.shape[1], padding_idx=padding_value).from_pretrained(embeddings, padding_idx=padding_value)
        self.lstm = nn.LSTM(embeddings.shape[1], hidden_sizes, layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
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


class HierarchicalNetwork(nn.Module):

    """
        Original model from https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
        but without attention
    """

    def __init__(self, n_classes, n_words, embeddings, layers, hidden_sizes, dropout, padding_value, eos_value, device):

        super(HierarchicalNetwork, self).__init__()

        self.padding_value = padding_value
        self.end_of_sentence_value = eos_value

        self.embedder = nn.Embedding(n_words, embeddings.shape[1], padding_idx=padding_value).from_pretrained(embeddings, padding_idx=padding_value)

        self.word_encoder = nn.GRU(embeddings.shape[1], hidden_sizes, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.sentence_encoder = nn.GRU(hidden_sizes * 2, hidden_sizes, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.hidden_to_label = nn.Linear(hidden_sizes * 2, n_classes)

        self.device = device
        self.to(device)

    def forward(self, X):

        documents_as_sentences = []

        for x in X: # TODO - Can this be vectorized?

            # Sentence batch: L words [SxL]
            document = split_into_sentences(x, self.padding_value, self.end_of_sentence_value)

            # Sentence batch: L words, E embeddings [SxLxE]
            words = self.embedder(document)
            word_encodings = self.word_encoder(words)[1]

            # Document: S sentences of 2H gru-units [1xSx2H]
            sentences = torch.cat((word_encodings[-2], word_encodings[-1]), dim=1)
            documents_as_sentences.append(sentences)

        # Documents batch: S sentences, 2H gru-units [BxSx2H]
        documents_as_sentences = pad_sequence(documents_as_sentences, batch_first=True)
        sentence_encodings = self.sentence_encoder(documents_as_sentences)[1]

        # Batch of document "features": 2H gru-units [Bx2H]
        document = torch.cat((sentence_encodings[-2], sentence_encodings[-1]), dim=1)

        # Batch of document "scores": num_classes outputs [BxK]
        scores = self.hidden_to_label(document)

        return scores


class HierarchicalAttentionNetwork(nn.Module):

    """ Original model from https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf"""

    def __init__(self, n_classes, n_words, embeddings, layers, hidden_sizes, dropout, padding_value, eos_value, device):

        super(HierarchicalAttentionNetwork, self).__init__()

        self.padding_value = padding_value
        self.end_of_sentence_value = eos_value

        self.embedder = nn.Embedding(n_words, embeddings.shape[1], padding_idx=padding_value).from_pretrained(embeddings, padding_idx=padding_value)

        self.word_encoder = nn.GRU(embeddings.shape[1], hidden_sizes, layers, batch_first=True, bidirectional=True, dropout=dropout)

        self.word_hidden_representation = nn.Sequential(nn.Linear(hidden_sizes * 2, hidden_sizes * 2), nn.Tanh())
        self.word_context_vector = nn.Parameter(torch.Tensor(hidden_sizes * 2))
        self.word_context_vector.data.uniform_(-1, 1)

        self.sentence_encoder = nn.GRU(hidden_sizes * 2, hidden_sizes, layers, batch_first=True, bidirectional=True, dropout=dropout)

        self.sentence_hidden_representation = nn.Sequential(nn.Linear(hidden_sizes * 2, hidden_sizes * 2), nn.Tanh())
        self.sentence_context_vector = nn.Parameter(torch.Tensor(hidden_sizes * 2))
        self.sentence_context_vector.data.uniform_(-1, 1)

        self.hidden_to_label = nn.Linear(hidden_sizes * 2, n_classes)

        self.device = device
        self.to(device)

    def forward(self, X):

        sentences = []

        for x in X: # TODO - Check if this can be vectorized?

            # S x L
            document = split_into_sentences(x, self.padding_value, self.end_of_sentence_value)

            # S x L x E
            words = self.embedder(document)

            # S x L x 2H
            hidden, _ = self.word_encoder(words)

            # S x L x 2H
            hidden_representations = self.word_hidden_representation(hidden)

            # S x L
            attention_weights = F.softmax(hidden_representations @ self.word_context_vector, dim=1)
            attention_weights = attention_weights.reshape(attention_weights.shape[0], 1, attention_weights.shape[1])

            # S x 2H
            sentences.append((attention_weights @ hidden).squeeze(dim=1))

        # B x S x 2H]
        sentences = pad_sequence(sentences, batch_first=True)

        # B x S x 2H
        hidden, _ = self.sentence_encoder(sentences)

        # B x S x 2H
        hidden_representations = self.sentence_hidden_representation(hidden)

        # B x S
        attention_weights = F.softmax(hidden_representations.matmul(self.sentence_context_vector), dim=1)
        attention_weights = attention_weights.reshape(attention_weights.shape[0], 1, attention_weights.shape[1])

        document = (attention_weights @ hidden).squeeze(dim=1)

        # Batch of document "scores": num_classes outputs [BxK]
        scores = self.hidden_to_label(document)

        return scores


class PrunedHierarchicalAttentionNetwork(nn.Module):

    def __init__(self, n_classes, n_words, attention_threshold, embeddings, layers, hidden_sizes, dropout, padding_value, eos_value, device):

        super(PrunedHierarchicalAttentionNetwork, self).__init__()

        self.padding_value = padding_value
        self.end_of_sentence_value = eos_value
        self.attention_threshold = attention_threshold

        self.embedder = nn.Embedding(n_words, embeddings.shape[1], padding_idx=padding_value).from_pretrained(embeddings, padding_idx=padding_value)

        self.word_encoder = nn.GRU(embeddings.shape[1], hidden_sizes, layers, batch_first=True, bidirectional=True, dropout=dropout)

        self.word_hidden_representation = nn.Sequential(nn.Linear(hidden_sizes * 2, hidden_sizes * 2), nn.Tanh())
        self.word_context_vector = nn.Parameter(torch.Tensor(hidden_sizes * 2))
        self.word_context_vector.data.uniform_(-1, 1)

        self.sentence_encoder = nn.GRU(hidden_sizes * 2, hidden_sizes, layers, batch_first=True, bidirectional=True, dropout=dropout)

        self.sentence_hidden_representation = nn.Sequential(nn.Linear(hidden_sizes * 2, hidden_sizes * 2), nn.Tanh())
        self.sentence_context_vector = nn.Parameter(torch.Tensor(hidden_sizes * 2))
        self.sentence_context_vector.data.uniform_(-1, 1)

        self.hidden_to_label = nn.Linear(hidden_sizes * 2, n_classes)

        self.device = device
        self.to(device)

    def forward(self, X):

        sentences = []

        for x in X: # TODO - Check if this can be vectorized?

            # S x L
            document = split_into_sentences(x, self.padding_value, self.end_of_sentence_value)

            # S x L x E
            words = self.embedder(document)

            # S x L x 2H
            hidden, _ = self.word_encoder(words)

            # S x L x 2H
            hidden_representations = self.word_hidden_representation(hidden)

            # S x L
            attention_weights = F.softmax(hidden_representations @ self.word_context_vector, dim=1)
            pruned_attention_weights = self.prune_attentions(attention_weights)

            # S x 2H
            sentences.append((pruned_attention_weights @ hidden).squeeze(dim=1))

        # B x S x 2H]
        sentences = pad_sequence(sentences, batch_first=True)

        # B x S x 2H
        hidden, _ = self.sentence_encoder(sentences)

        # B x S x 2H
        hidden_representations = self.sentence_hidden_representation(hidden)

        # B x S
        attention_weights = F.softmax(hidden_representations @ self.sentence_context_vector, dim=1)
        pruned_attention_weights = self.prune_attentions(attention_weights)
        document = (pruned_attention_weights @ hidden).squeeze(dim=1)

        # Batch of document "scores": num_classes outputs [BxK]
        scores = self.hidden_to_label(document)

        return scores

    def prune_attentions(self, attention_weights):
        pruned_attention_weights = (attention_weights < self.attention_threshold) * attention_weights
        sums = pruned_attention_weights.sum(dim=1).reshape(attention_weights.shape[0], 1)
        new_attention_weights = pruned_attention_weights / sums
        new_attention_weights[torch.isnan(new_attention_weights)] = 0.0
        new_attention_weights = new_attention_weights.reshape(new_attention_weights.shape[0], 1, new_attention_weights.shape[1])
        return new_attention_weights


class HierarchicalSparsemaxAttentionNetwork(nn.Module):

    def __init__(self, n_classes, n_words, embeddings, layers, hidden_sizes, dropout, padding_value, eos_value, device):
        super(HierarchicalSparsemaxAttentionNetwork, self).__init__()

        self.padding_value = padding_value
        self.end_of_sentence_value = eos_value

        self.embedder = nn.Embedding(n_words, embeddings.shape[1], padding_idx=padding_value).from_pretrained(
            embeddings,
            padding_idx=padding_value)

        self.word_encoder = nn.GRU(embeddings.shape[1], hidden_sizes, layers, batch_first=True, bidirectional=True,
                                   dropout=dropout)

        self.word_hidden_representation = nn.Sequential(nn.Linear(hidden_sizes * 2, hidden_sizes * 2), nn.Tanh())
        self.word_context_vector = nn.Parameter(torch.Tensor(hidden_sizes * 2))
        self.word_context_vector.data.uniform_(-1, 1)

        self.sentence_encoder = nn.GRU(hidden_sizes * 2, hidden_sizes, layers, batch_first=True, bidirectional=True,
                                       dropout=dropout)

        self.sentence_hidden_representation = nn.Sequential(nn.Linear(hidden_sizes * 2, hidden_sizes * 2), nn.Tanh())
        self.sentence_context_vector = nn.Parameter(torch.Tensor(hidden_sizes * 2))
        self.sentence_context_vector.data.uniform_(-1, 1)

        self.hidden_to_label = nn.Linear(hidden_sizes * 2, n_classes)

        self.device = device
        self.to(device)

    def forward(self, X):
        sentences = []

        for x in X:  # TODO - Check if this can be vectorized?

            # S x L
            document = split_into_sentences(x, self.padding_value, self.end_of_sentence_value)

            # S x L x E
            words = self.embedder(document)

            # S x L x 2H
            hidden, _ = self.word_encoder(words)

            # S x L x 2H
            hidden_representations = self.word_hidden_representation(hidden)

            # S x L
            # attention_weights = F.softmax(hidden_representations @ self.word_context_vector, dim=1)
            attention_weights = SparseMax.apply((hidden_representations @ self.word_context_vector), 1, None)

            attention_weights = attention_weights.reshape(attention_weights.shape[0], 1, attention_weights.shape[1])

            # S x 2H
            sentences.append((attention_weights @ hidden).squeeze(dim=1))

        # B x S x 2H]
        sentences = pad_sequence(sentences, batch_first=True)

        # B x S x 2H
        hidden, _ = self.sentence_encoder(sentences)

        # B x S x 2H
        hidden_representations = self.sentence_hidden_representation(hidden)

        # B x S
        attention_weights = SparseMax.apply(hidden_representations.matmul(self.sentence_context_vector), 1, None)
        # attention_weights = F.softmax(hidden_representations.matmul(self.sentence_context_vector), dim=1)
        attention_weights = attention_weights.reshape(attention_weights.shape[0], 1, attention_weights.shape[1])

        document = (attention_weights @ hidden).squeeze(dim=1)

        # Batch of document "scores": num_classes outputs [BxK]
        scores = self.hidden_to_label(document)

        return scores
