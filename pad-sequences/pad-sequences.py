import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    N = len(seqs)

    if max_len is None:
        L = max(len(seq) for seq in seqs)
    else:
        L = max_len
    
    res = np.full((N, L), pad_value, dtype=int)
    
    for i, seq in enumerate(seqs):
        length = min(len(seq), L)
        res[i, :length] = seq[:length]
    
    return res