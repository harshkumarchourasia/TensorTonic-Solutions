import numpy as np

def bag_of_words_vector(tokens, vocab):
    vocab_size = len(vocab)
    word2id = {word: i for i, word in enumerate(vocab)}

    indices = [word2id[t] for t in tokens if t in word2id]
    return np.bincount(indices, minlength=vocab_size)
