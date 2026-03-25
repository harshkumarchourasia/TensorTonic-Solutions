import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v = np.array(v)
    w = np.array(w)
    v_norm = np.sqrt(np.sum(v*v))
    w_norm = np.sqrt(np.sum(w*w))
    return np.arccos(np.dot(v,w)/(v_norm*w_norm))