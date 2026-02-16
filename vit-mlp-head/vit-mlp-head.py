import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Classification head for ViT.
    """
    batch_size, patch_size, embed_size = encoder_output.shape
    x_norm = (encoder_output[:,0,:] - np.mean(encoder_output[:,0,:], axis = 1, keepdims = True)) / np.std(encoder_output[:,0,:], axis = 1, keepdims = True)
    gamma = np.ones((1, embed_size))
    beta = np.zeros((1, embed_size))
    x_ln = gamma * x_norm + beta
    W = np.random.randn(embed_size, num_classes) * 0.02
    b = np.zeros((1, num_classes))
    return x_ln @ W + b