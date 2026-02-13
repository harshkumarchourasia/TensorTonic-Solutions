import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int) -> np.ndarray:
    """
    Add learnable position embeddings to patch embeddings.
    """
    # YOUR CODE HERE
    pos_embedding = np.random.normal(
        loc=0.0,      # mean
        scale=0.02,   # standard deviation (σ)
        size=(1, num_patches, embed_dim)
    )

    return patches + pos_embedding