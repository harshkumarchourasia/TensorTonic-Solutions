def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    import numpy as np
    n = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum(y_true == y_pred)
    return tp/(tp+(n-tp))
