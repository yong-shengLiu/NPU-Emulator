import math
import numpy as np
import matplotlib.pyplot as plt
import struct
import torch


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


def softmax_test_patterns():
    return [
        np.array([1.0, 1.0, 1.0, 1.0]),                  #  1: uniform
        np.array([1.0, 1.01, 1.02, 1.03]),               #  2: small range
        np.array([1.0, 5.0, 30.0, 100.0]),               #  3: large range
        np.array([-10.0, -20.0, -30.0]),                 #  4: underflow test
        np.array([-1.2, 0.0, 1.2]),                      #  5: centered
        np.array([0.0, 0.0, 0.0, 10.0]),                 #  6: one-hot
        np.random.normal(0, 1, 64),                      #  7: normal
        np.random.uniform(-5, 5, 64),                    #  8: uniform
        # np.array([0, 128, 255], dtype=np.uint8),         #  9: quantization range
        np.array([0.00001, 0.00002, 0.00003])            # 10: soft rounding
    ]


def SoftMax(x):

    maximum = np.max(x)

    diff = x - maximum

    exp_diff = np.exp(diff)

    summation = np.sum(exp_diff)

    softmax_x = exp_diff / summation

    return softmax_x


def SoftMax_hard(x):

    maximum = np.max(x)

    diff = x - maximum
    print("diff:", diff)


    # quantize the exponentials


    # divide by the FastInverse Square Root (FISR)

    softmax_x = 0
    return softmax_x

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


def FISR(number, iteration):
    """ Fast Inverse Square Root (FISR) """


    threehalfs = 1.5
    x2 = number * 0.5
    y = number


    # evil floating point bit level hacking
    i = struct.unpack('I', struct.pack('f', y))[0]
    i = 0x5f3759df - (i >> 1)
    y = struct.unpack('f', struct.pack('I', i))[0]


    for _ in range(iteration):
        y = y * (threehalfs - (x2 * y * y))

    result_bits = struct.unpack('I', struct.pack('f', y))[0]
    size = struct.calcsize('I')


    if result_bits < 0 or result_bits >= (1 << (size * 8)):
        raise ValueError('result_bits out of range')

    return struct.unpack('f', struct.pack('I', result_bits))[0]


def MSE(golden, prediction):
    """ Mean Squared Error (MSE) """

    if len(golden) != len(prediction):
        raise ValueError("Length of golden and prediction must match")

    mse = np.mean((golden - prediction) ** 2)

    if mse == 0:
        accuracy = float('inf')  # Perfect match
        print("Perfect match!")
    else:
        accuracy = math.log10(1 / mse)

    return accuracy


if __name__ == "__main__":
    print("=== SFU testbench ===")

    # x = np.linspace(-10, 10)
    # plt.plot(x, layer_norm(x))
    # plt.axis('tight')
    # plt.title('Activation Function :GELU')
    # plt.show()


    test_cases = softmax_test_patterns()

    for idx, x in enumerate(test_cases):
        gold_out = SoftMax(x)
        ref_out  = SoftMax_hard(x)

        # print(f"Test case {idx + 1}: {x}")
        # print(f"SoftMax output: {my_out}\n\n")

    print(torch.rand(5))