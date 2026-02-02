import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    dot = np.dot(x1, x2)
    mod = np.linalg.norm(x1) * np.linalg.norm(x2)
    if mod == 0:
        dot = 0.0
    else:
        dot /= mod
    if label==1:
        return 1 - dot
    elif label==-1:
        return max(0, dot - margin)
