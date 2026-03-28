import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.

    The loss ensures that the squared Euclidean distance between anchor and
    positive is smaller than that between anchor and negative by at least margin.

    Parameters
    ----------
    anchor : array-like, shape (N, D) or (D,)
        Anchor embeddings.
    positive : array-like, shape (N, D) or (D,)
        Positive embeddings (same class as anchor).
    negative : array-like, shape (N, D) or (D,)
        Negative embeddings (different class from anchor).
    margin : float, default=1.0
        Margin parameter (m) in the loss formula.

    Returns
    -------
    float
        Mean triplet loss over the batch (scalar).
    """
    # Convert inputs to numpy arrays
    a = np.array(anchor)
    p = np.array(positive)
    n = np.array(negative)

    # Ensure inputs are at least 2D (handle single vectors)
    if a.ndim == 1:
        a = a.reshape(1, -1)
        p = p.reshape(1, -1)
        n = n.reshape(1, -1)

    # Compute squared Euclidean distances: d(a,p) and d(a,n)
    d_ap = np.sum((a - p) ** 2, axis=1)   # shape (N,)
    d_an = np.sum((a - n) ** 2, axis=1)   # shape (N,)

    # Triplet loss per sample
    losses = np.maximum(0, d_ap - d_an + margin)  # shape (N,)

    # Return mean loss across batch
    return np.mean(losses).item()