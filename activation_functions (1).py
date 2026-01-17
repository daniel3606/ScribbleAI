import numpy as np

def relu(x):
    # x will be an input. For a ReLU function to work 
    # Based on the definition of ReLU, can you complete the function? Super easy
    return np.maximum(0, x)

def softmax(x):
    # Based on the definitions, videos, and mathematical notions, can you write this function
    # that returns the value after applying softmax to x?
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)