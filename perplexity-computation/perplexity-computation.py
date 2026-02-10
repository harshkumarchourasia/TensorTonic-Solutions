import numpy as np
def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    log_likelyhood = 0
    for idx, actual_token in enumerate(actual_tokens):
        log_likelyhood += np.log(prob_distributions[idx][actual_token])
    return np.exp(- log_likelyhood / len(actual_tokens))