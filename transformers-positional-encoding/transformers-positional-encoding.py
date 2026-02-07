import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pos = np.arange(seq_length).reshape(-1,1)    
    den = np.power(10000, (np.arange(0,d_model,2)/d_model))

    angles = pos/den
    

    positional_embeddings = np.zeros((seq_length, d_model))

    positional_embeddings[:,::2] = np.sin(angles)
    positional_embeddings[:,1::2] = np.cos(angles)
    return positional_embeddings
    