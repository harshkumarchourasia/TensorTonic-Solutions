import torch
def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    X = torch.tensor(X)
    w = torch.tensor(W)
    b = torch.tensor(b)
    res = torch.matmul(X, w) + b
    return res.tolist()
