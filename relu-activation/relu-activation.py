import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    
    Parameters:
    x: scalar, list, or numpy array
    
    Returns:
    numpy array with ReLU applied element-wise
    """
    # Convert input to numpy array with float type
    x_arr = np.asarray(x, dtype=np.float64)
    
    # Apply ReLU element-wise using np.maximum
    result = np.maximum(0, x_arr)
    
    # If input was a scalar, return 1D array with shape (1,) as per example
    if np.isscalar(x):
        return result.reshape(1)
    
    return result