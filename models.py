from torch import nn

"""
If three classes turn out to be very similar, 
we refactor their code in the end
"""


class HierarchicalAttentionNetwork(nn.Module):

    """ Original model from https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf"""

    # TODO

    def __init__(self, device):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.device = device
        self.to(device)

    def forward(self, X):
        """
        X (batch_size (B) x n_features (D)): a batch of training examples
        Z (batch_size (B) x n_classes (K)): a batch of raw logits (no softmax)
        """
        Z = None
        return Z


class PrunedHierarchicalAttentionNetwork(nn.Module):

    # TODO

    def __init__(self, device):
        super(PrunedHierarchicalAttentionNetwork, self).__init__()
        self.device = device
        self.to(device)

    def forward(self, X):
        """
        X (batch_size (B) x n_features (D)): a batch of training examples
        Z (batch_size (B) x n_classes (K)): a batch of raw logits (no softmax)
        """
        Z = None
        return Z


class HierarchicalSparsemaxAttentionNetwork(nn.Module):

    # TODO

    def __init__(self, device):
        super(HierarchicalSparsemaxAttentionNetwork, self).__init__()
        self.device = device
        self.to(device)

    def forward(self, X):
        """
        X (batch_size (B) x n_features (D)): a batch of training examples
        Z (batch_size (B) x n_classes (K)): a batch of raw logits (no softmax)
        """
        Z = None
        return Z


class FeedforwardNetwork(nn.Module):

    """
    Used to test main's skeleton (train-eval loop).
    Dont remove me.
    """

    def __init__(self, n_classes, n_features, hidden_size, layers, activation, dropout, device):

        super(FeedforwardNetwork, self).__init__()

        # Activation parsing
        if activation == "tanh": act_fn = nn.Tanh()
        else: act_fn = nn.ReLU()

        # Input + First Hidden Layer
        layer_list = [nn.Linear(n_features, hidden_size), act_fn, nn.Dropout(dropout)]

        # Hidden Layers
        for _ in range(layers - 1):  # We already added the first hidden layer with the input features
            layer_list.extend((nn.Linear(hidden_size, hidden_size), act_fn, nn.Dropout(dropout)))

        # Output Layer
        layer_list.append(nn.Linear(hidden_size, n_classes))

        self.Z = nn.Sequential(*layer_list)

        self.device = device
        self.to(device)

    def forward(self, X):
        """
        X (batch_size (B) x n_features (D)): a batch of training examples
        """
        Z = self.Z(X)
        return Z
