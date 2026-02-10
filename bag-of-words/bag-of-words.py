import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
   # Your code here
    vocab_size = len(vocab)
    result = np.zeros(vocab_size, dtype=int)
    word2id = dict()
    for idx, word in enumerate(vocab):
        word2id[word] = idx
    for token in tokens:
        if token not in word2id: continue
        idx = word2id[token]
        result[idx] += 1
    return result