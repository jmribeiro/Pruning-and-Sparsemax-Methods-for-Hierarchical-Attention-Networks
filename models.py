import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


# ############################# #
# Main Models (HAN, HPAN, HSAN) #
# ############################# #

class HierarchicalAttentionNetwork(nn.Module):

    """ Original model from https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf"""

    def __init__(self, n_classes, n_words, embeddings, layers, hidden_sizes, dropout, padding_value, eos_value, device, attention_function="softmax", pruned_attention=False, attention_threshold=None):
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
        if attention_function == "sparsemax": self.attention_function = Sparsemax(dim=1, device=device)
        elif attention_function == "softmax": self.attention_function = torch.nn.Softmax(dim=1)
        else: raise ValueError(f"Unregistered attention function {attention_function}. Please pick on of the following: [sparsemax, softmax]")
        self.pruned_attention = pruned_attention
        if self.pruned_attention: self.attention_threshold = attention_threshold
        self.hidden_to_label = nn.Linear(hidden_sizes * 2, n_classes)
        self.device = device
        self.to(device)

    def forward(self, X):
        # B x S x 2H
        X = self.process_words(X)
        # B x S x 2H
        hidden, _ = self.sentence_encoder(X)
        # B x S x 2H
        hidden_representations = self.sentence_hidden_representation(hidden)
        # B x S
        attention_weights = self.attention_function(hidden_representations @ self.sentence_context_vector)
        attention_weights = prune_attentions(attention_weights, self.attention_threshold) if self.pruned_attention else attention_weights.reshape(attention_weights.shape[0], 1, attention_weights.shape[1])
        # B x 2H
        documents = (attention_weights @ hidden).squeeze(dim=1)
        # B x K
        scores = self.hidden_to_label(documents)
        return scores

    def process_words(self, documents):
        sentences = []
        for document in documents:
            # S x L
            words = split_into_sentences(document, self.padding_value, self.end_of_sentence_value)
            # S x L x E
            words = self.embedder(words)
            # S x L x 2H
            hidden, _ = self.word_encoder(words)
            # S x L x 2H
            hidden_representations = self.word_hidden_representation(hidden)
            # S x L
            attention_weights = self.attention_function(hidden_representations @ self.word_context_vector)
            attention_weights = prune_attentions(attention_weights, self.attention_threshold) if self.pruned_attention else attention_weights.reshape(attention_weights.shape[0], 1, attention_weights.shape[1])
            # S x 2H
            sentences.append((attention_weights @ hidden).squeeze(dim=1))
        # B x S x 2H
        sentences = pad_sequence(sentences, batch_first=True)
        return sentences


# ####################### #
# Basic Models (LSTM, HN) #
# ####################### #

class LSTMClassifier(nn.Module):
    """
    Very simple LSTM Classifier to test the datasets.
    """

    def __init__(self, n_classes, n_words, embeddings, layers, hidden_sizes, bidirectional, dropout, padding_value,
                 device):
        super().__init__()

        self.embedder = nn.Embedding(n_words, embeddings.shape[1], padding_idx=padding_value).from_pretrained(
            embeddings, padding_idx=padding_value)
        self.lstm = nn.LSTM(embeddings.shape[1], hidden_sizes, layers, dropout=dropout, batch_first=True,
                            bidirectional=bidirectional)
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

        self.embedder = nn.Embedding(n_words, embeddings.shape[1], padding_idx=padding_value).from_pretrained(
            embeddings, padding_idx=padding_value)

        self.word_encoder = nn.GRU(embeddings.shape[1], hidden_sizes, layers, batch_first=True, bidirectional=True,
                                   dropout=dropout)
        self.sentence_encoder = nn.GRU(hidden_sizes * 2, hidden_sizes, layers, batch_first=True, bidirectional=True,
                                       dropout=dropout)
        self.hidden_to_label = nn.Linear(hidden_sizes * 2, n_classes)

        self.device = device
        self.to(device)

    def forward(self, X):
        documents_as_sentences = []

        for x in X:
            # Sentence batch: L words [SxL]
            document = split_into_sentences(x, self.padding_value, self.end_of_sentence_value)

            # Sentence batch: L words, E embeddings [SxLxE]
            words = self.embedder(document)
            word_encodings = self.word_encoder(words)[1]

            # Document: S sentences of 2H gru-units [1xSx2H]
            sentences = torch.cat((word_encodings[-2], word_encodings[-1]), dim=1)
            documents_as_sentences.append(sentences)

        del X

        # Documents batch: S sentences, 2H gru-units [BxSx2H]
        documents_as_sentences = pad_sequence(documents_as_sentences, batch_first=True)
        sentence_encodings = self.sentence_encoder(documents_as_sentences)[1]

        # Batch of document "features": 2H gru-units [Bx2H]
        document = torch.cat((sentence_encodings[-2], sentence_encodings[-1]), dim=1)

        # Batch of document "scores": num_classes outputs [BxK]
        scores = self.hidden_to_label(document)

        return scores


# ############### #
# Model Utilities #
# ############### #

def split_into_sentences(document, padding_value, eos_value):
    """
        Given a document as sequence (shape L1: total length)
        Returns a document as sentences (shape SxL2)
    """
    ends_of_sentence = (document == eos_value).nonzero()
    sentences = [document[0:eos + 1] if i == 0 else document[ends_of_sentence[i - 1] + 1:eos + 1] for i, eos in enumerate(ends_of_sentence)]
    last = document[ends_of_sentence[-1] + 1:]
    if False in last == padding_value: sentences.append(last)
    document = pad_sequence(sentences, batch_first=True, padding_value=padding_value)
    return document


def prune_attentions(attention_weights, attention_threshold):
    pruned_attention_weights = (attention_weights < attention_threshold).float() * attention_weights
    sums = pruned_attention_weights.sum(dim=1).reshape(attention_weights.shape[0], 1)
    pruned_attentions = pruned_attention_weights / sums
    pruned_attentions[torch.isnan(pruned_attentions)] = 0.0
    pruned_attentions = pruned_attentions.reshape(pruned_attentions.shape[0], 1, pruned_attentions.shape[1])
    return pruned_attentions


class Sparsemax(nn.Module):

    """Sparsemax function.

    Pytorch implementation of Sparsemax function from:
    -- "From https://github.com/KrisKorrel/sparsemax-pytorch:
        -- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
        -- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)
    """

    def __init__(self, device, dim=None):
        """
        Args: dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim
        self.device = device

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=self.device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        dim = 1
        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))
        return self.grad_input
