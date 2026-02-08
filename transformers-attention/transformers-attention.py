import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-1, -2))
    scores = scores / torch.sqrt(torch.tensor(d_k))
    scores_max = scores.max(dim = -1, keepdims=True).values
    scores = scores - scores_max
    scores_exp = torch.exp(scores)
    scores_softmax = scores_exp / scores_exp.sum(axis = -1, keepdims = True)
    return torch.matmul(scores_softmax, V)