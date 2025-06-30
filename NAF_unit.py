import math
import numpy as np
import matplotlib.pyplot as plt



def sigmoid(x):
    return 1/(1+np.exp(-x))


def tanh(x):
    return np.tanh(x)


def RELU(x):
    x1=[]
    
    for i in x:
        if i<0:
            x1.append(0)
        else:
            x1.append(i)

    return x1


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def selu(x, alpha = 1.6732, lambda_ = 1.0507):
    return np.where(x > 0, lambda_ * x, lambda_ * alpha * (np.exp(x) - 1))


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x, gamma=None, beta=None, epsilon=1e-5):
    """
    Layer Normalization function

    Args:
        x (np.ndarray): shape = (batch_size, feature_dim)
        gamma (np.ndarray): scale parameter (same shape as feature_dim)
        beta (np.ndarray): shift parameter (same shape as feature_dim)
        epsilon (float): small constant to avoid division by zero

    Returns:
        np.ndarray: normalized result, same shape as x
    """
    # Step 1: compute mean and variance along last dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)

    # Step 2: normalize
    normalized = (x - mean) / np.sqrt(variance + epsilon)

    # Step 3: scale and shift (optional)
    if gamma is not None:
        normalized *= gamma
    if beta is not None:
        normalized += beta

    return normalized



if __name__ == "__main__":
    x = np.linspace(-10, 10)
    plt.plot(x, layer_norm(x))
    plt.axis('tight')
    plt.title('Activation Function :GELU')
    plt.show()