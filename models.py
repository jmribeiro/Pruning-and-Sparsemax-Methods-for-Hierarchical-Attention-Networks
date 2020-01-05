import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

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
    last = document[ends_of_sentence[-1] + 1:]
    if False in last == padding_value: sentences.append(last)
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
        
        del X

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

        for x in X:

            # S x L
            words = split_into_sentences(x, self.padding_value, self.end_of_sentence_value)

            # S x L x E
            words = self.embedder(words)

            # S x L x 2H
            hidden, _ = self.word_encoder(words)

            # S x L x 2H
            hidden_representations = self.word_hidden_representation(hidden)

            # S x L
            attention_weights = F.softmax(hidden_representations @ self.word_context_vector, dim=1)
            attention_weights = attention_weights.reshape(attention_weights.shape[0], 1, attention_weights.shape[1])

            # S x 2H
            sentences.append((attention_weights @ hidden).squeeze(dim=1))
        
        del X

        # B x S x 2H
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


class HierarchicalPrunedAttentionNetwork(nn.Module):

    def __init__(self, n_classes, n_words, attention_threshold, embeddings, layers, hidden_sizes, dropout, padding_value, eos_value, device):

        super(HierarchicalPrunedAttentionNetwork, self).__init__()

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
            documents = split_into_sentences(x, self.padding_value, self.end_of_sentence_value)

            # S x L x E
            words = self.embedder(documents)

            # S x L x 2H
            hidden, _ = self.word_encoder(words)

            # S x L x 2H
            hidden_representations = self.word_hidden_representation(hidden)

            # S x L
            attention_weights = F.softmax(hidden_representations @ self.word_context_vector, dim=1)
            pruned_attention_weights = self.prune_attentions(attention_weights)

            # S x 2H
            sentences.append((pruned_attention_weights @ hidden).squeeze(dim=1))
        
        del X

        # B x S x 2H]
        sentences = pad_sequence(sentences, batch_first=True)

        # B x S x 2H
        hidden, _ = self.sentence_encoder(sentences)

        # B x S x 2H
        hidden_representations = self.sentence_hidden_representation(hidden)

        # B x S
        attention_weights = F.softmax(hidden_representations @ self.sentence_context_vector, dim=1)
        pruned_attention_weights = self.prune_attentions(attention_weights)
        documents = (pruned_attention_weights @ hidden).squeeze(dim=1)

        # Batch of document "scores": num_classes outputs [BxK]
        scores = self.hidden_to_label(documents)

        return scores

    def prune_attentions(self, attention_weights):
        pruned_attention_weights = (attention_weights < self.attention_threshold).float() * attention_weights
        sums = pruned_attention_weights.sum(dim=1).reshape(attention_weights.shape[0], 1)
        pruned_attentions = pruned_attention_weights / sums
        pruned_attentions[torch.isnan(pruned_attentions)] = 0.0
        pruned_attentions = pruned_attentions.reshape(pruned_attentions.shape[0], 1, pruned_attentions.shape[1])
        return pruned_attentions


class HierarchicalSparsemaxAttentionNetwork(nn.Module):

    def __init__(self, n_classes, n_words, embeddings, layers, hidden_sizes, dropout, padding_value, eos_value, device):
        super(HierarchicalSparsemaxAttentionNetwork, self).__init__()

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

        self.sparsemax = Sparsemax(dim=1, k=None)
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
            attention_weights = self.sparsemax(hidden_representations @ self.word_context_vector)
            attention_weights = attention_weights.reshape(attention_weights.shape[0], 1, attention_weights.shape[1])

            # S x 2H
            sentences.append((attention_weights @ hidden).squeeze(dim=1))
        
        del X

        # B x S x 2H]
        sentences = pad_sequence(sentences, batch_first=True)

        # B x S x 2H
        hidden, _ = self.sentence_encoder(sentences)

        # B x S x 2H
        hidden_representations = self.sentence_hidden_representation(hidden)

        # B x S
        attention_weights = self.sparsemax(hidden_representations.matmul(self.sentence_context_vector))
        attention_weights = attention_weights.reshape(attention_weights.shape[0], 1, attention_weights.shape[1])

        document = (attention_weights @ hidden).squeeze(dim=1)

        # Batch of document "scores": num_classes outputs [BxK]
        scores = self.hidden_to_label(document)

        return scores


class Sparsemax(nn.Module):

    def __init__(self, dim=-1, k=None):
        self.dim = dim
        self.k = k
        self.supp_size = None
        self.output = None
        super(Sparsemax, self).__init__()

    def forward(self, X):
        max_val, _ = X.max(dim=self.dim, keepdim=True)
        X = X - max_val
        tau, supp_size = Sparsemax.threshold(X, dim=self.dim, k=self.k)
        output = torch.clamp(X - tau, min=0)
        self.supp_size = supp_size
        self.output = output
        return output

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input[self.output == 0] = 0
        v_hat = grad_input.sum(dim=self.dim) / self.supp_size.to(self.output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(self.dim)
        grad_input = torch.where(self.output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None

    @staticmethod
    def threshold(input_x, dim, k):
        if k is None or k >= input_x.shape[dim]:  # do full sort
            topk, _ = torch.sort(input_x, dim=dim, descending=True)
        else:
            topk, _ = torch.topk(input_x, k=k, dim=dim)

        topk_cumsum = topk.cumsum(dim) - 1
        rhos = Sparsemax.make_ix_like(topk, dim)
        support = rhos * topk > topk_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = topk_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input_x.dtype)

        if k is not None and k < input_x.shape[dim]:
            unsolved = (support_size == k).squeeze(dim)

            if torch.any(unsolved):
                in_ = Sparsemax.roll_last(input_x, dim)[unsolved]
                tau_, ss_ = Sparsemax.threshold(in_, dim=-1, k=2 * k)
                Sparsemax.roll_last(tau, dim)[unsolved] = tau_
                Sparsemax.roll_last(support_size, dim)[unsolved] = ss_

        return tau, support_size

    @staticmethod
    def roll_last(input_x, dim):
        if dim == -1:
            return input_x
        elif dim < 0:
            dim = input_x.dim() - dim

        perm = [i for i in range(input_x.dim()) if i != dim] + [dim]
        return input_x.permute(perm)

    @staticmethod
    def make_ix_like(input_x, dim):
        d = input_x.size(dim)
        rho = torch.arange(1, d + 1, device=input_x.device, dtype=input_x.dtype)
        view = [1] * input_x.dim()
        view[0] = -1
        return rho.view(view).transpose(0, dim)
