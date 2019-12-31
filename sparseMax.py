import torch
from torch.autograd import Function
from torch import nn


# Implentation based on  [Author: Vlad Niculae <vlad@vene.ro>]

# Compute threshold function
def threshold(input_x, dim, k):
    if k is None or k >= input_x.shape[dim]:  # do full sort
        topk, _ = torch.sort(input_x, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(input_x, k=k, dim=dim)

    topk_cumsum = topk.cumsum(dim) - 1
    rhos = make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input_x.dtype)

    if k is not None and k < input_x.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            in_ = roll_last(input_x, dim)[unsolved]
            tau_, ss_ = threshold(in_, dim=-1, k=2 * k)
            roll_last(tau, dim)[unsolved] = tau_
            roll_last(support_size, dim)[unsolved] = ss_

    return tau, support_size


def roll_last(input_x, dim):
    if dim == -1:
        return input_x
    elif dim < 0:
        dim = input_x.dim() - dim

    perm = [i for i in range(input_x.dim()) if i != dim] + [dim]
    return input_x.permute(perm)


def make_ix_like(input_x, dim):
    d = input_x.size(dim)
    rho = torch.arange(1, d + 1, device=input_x.device, dtype=input_x.dtype)
    view = [1] * input_x.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


# Sparsemax activation function Implementation
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
        tau, supp_size = threshold(X, dim=self.dim, k=self.k)
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
