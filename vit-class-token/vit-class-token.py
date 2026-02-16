import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    """
    # YOUR CODE HERE
    batch_size, patch_size, embed_dim = patches.shape
    cls_token = np.random.normal(size = (1, 1, embed_dim))
    cls_token_repeat = np.repeat(cls_token, repeats = batch_size, axis = 0)
    return np.concatenate([cls_token_repeat, patches], axis = 1)
    