import math
import numpy as np
import matplotlib.pyplot as plt
import struct
import torch


def float_to_q4_11(value):
    """
    Convert a float to Q4.11 fixed-point (signed 16-bit)
    """
    scale = 2**11  # fractional resolution
    max_val = 15.9995
    min_val = -16.0

    # Saturate to range
    value = max(min_val, min(max_val, value))

    # Convert to fixed-point integer
    fixed_val = int(round(value * scale))

    # Convert to 16-bit signed int (simulate overflow behavior)
    if fixed_val < 0:
        fixed_val = (1 << 16) + fixed_val  # two's complement

    return fixed_val


def q4_11_to_float(fixed_val):
    """
    Convert Q4.11 fixed-point back to float
    """
    if fixed_val & (1 << 15):  # negative number in two's complement
        fixed_val = fixed_val - (1 << 16)

    return fixed_val / (2**11)


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
        # np.array([-1.2, 0.0, 1.2]),                      #  5: centered
        np.array([0.0, 0.0, 0.0, 10.0]),                 #  6: one-hot
        np.random.normal(0, 1, 64),                      #  7: normal
        np.random.uniform(-5, 5, 64),                    #  8: uniform
        # np.array([0.00001, 0.00002, 0.00003])            #  9: soft rounding
    ]


def SoftMax(x):
    """
    NOTE: Hardware benchmark list
    (1) SoftMax_1: the exponentials used quantization and linear approximation, the summation is rounded to the nearest power of 2
    (2) SoftMax_2: the exponentials used quantization and linear approximation, and the division is done by Fast Inverse Square Root (FISR)
    (3) SoftMax_3: the exponentials used quantization and Cordic approximation, and the division is done by Cordic inverse
    """
    maximum = np.max(x)

    diff = x - maximum

    exp_diff = np.exp(diff)

    summation = np.sum(exp_diff)

    softmax_x = exp_diff / summation

    return softmax_x


def SoftMax_1(x):
    """
    NOTE:
    (1) the exponentials used quantization and linear approximation
    (2) the summation is rounded to the nearest power of 2
    TODO: Need to Consider the datatype
    """
    maximum = np.max(x)

    diff = x - maximum
    # print("diff:", diff)


    # quantize the exponentials
    log2e = 1.5
    int_frac = diff * log2e

    # print("int_frac:", int_frac)

    frac_part, int_part = np.modf(int_frac)
    # print("frac_part:", frac_part, "int_part:", int_part)

    frac_approximated = frac_part / 2 + 1  # linear approximation
    # print("frac_approximated:", frac_approximated)

    exp_approximated = np.power(2, int_part) * frac_approximated
    # print("exp_approximated:", exp_approximated, "exp: ", np.exp(diff))

    # print(f'Error: {MSE(np.exp(diff), exp_approximated)}')


    # Summation of exponentials
    summation = np.sum(exp_approximated)
    print("Exp. summation:", summation)
    
    # rounde the summation to the nearest power of 2
    summation_rounded = 2 ** np.round(np.log2(summation))
    print("Exp. summation_rounded:", summation_rounded)

    # divide by the FastInverse Square Root (FISR)

    softmax_x = exp_approximated / summation_rounded
    return softmax_x

def SoftMax_2(x):
    """
    NOTE:
    (1) the exponentials used quantization and linear approximation
    (2) the division is done by Fast Inverse Square Root (FISR)
    TODO: Need to Consider the datatype
    """
    maximum = np.max(x)

    diff = x - maximum

    # quantize the exponentials
    log2e = 1.5
    int_frac = diff * log2e

    frac_part, int_part = np.modf(int_frac)

    frac_approximated = frac_part / 2 + 1  # linear approximation

    exp_approximated = np.power(2, int_part) * frac_approximated


    # Summation of exponentials
    summation = np.sum(exp_approximated)
    print("Exp. summation:", summation)
    
    # divide by the FastInverse Square Root (FISR)
    softmax_x = exp_approximated * FISR((summation**2), 3)

    return softmax_x

def SoftMax_3(x):
    
    maximum = np.max(x)
    diff = x - maximum
    
    # cordic approximation of exponentials
    log2e = 1.5
    diff_log2 = diff * log2e
    exp_diff = cordic_pow2(diff_log2)

    summation = np.sum(exp_diff)

    # cordic approximation of inverse
    inv_sum = cordic_inverse(summation)
    softmax_x = exp_diff * inv_sum

    return softmax_x

def cordic_pow2(x, iterations=16):
    int_part = np.floor(x).astype(int)
    frac_part = x - int_part

    pow2_lut = [2 ** (2 ** -i) for i in range(iterations)]
    
    result = np.ones_like(x)
    
    for i in range(iterations):
        mask = frac_part >= (2 ** -i)
        result[mask] *= pow2_lut[i]
        frac_part[mask] -= (2 ** -i)
    
    return result * (2.0 ** int_part)

def cordic_inverse(x, iterations=20):
    """
    CORDIC-based inverse (1/x) using multiplicative inverse approach.
    Simulates shift-add CORDIC style, but not optimized for hardware here.
    Assumes x > 0.
    """
    if x <= 0:
        raise ValueError("CORDIC inverse requires x > 0")

    # Normalize x to [0.5, 1) by shifting (simulate as floating point here)
    shift = 0
    while x < 0.5:
        x *= 2
        shift -= 1
    while x >= 1.0:
        x /= 2
        shift += 1

    # Initial estimate z0
    z = 1.5 - x  # or try z = 1 / (1 + x) as better init

    # Iteratively improve z such that z * x ≈ 1
    for _ in range(iterations):
        z = z * (2 - x * z)

    # Adjust scaling due to initial normalization
    z *= 2 ** (-shift)

    return z

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
        loss = 0
    else:
        accuracy = math.log10(1 / mse)
        loss = mse

    return accuracy


if __name__ == "__main__":
    print("=== SFU testbench ===")

    # x = np.linspace(-10, 10)
    # plt.plot(x, SoftMax(x))
    # plt.axis('tight')
    # plt.title('Activation Function :GELU')
    # plt.show()

    # Store results
    mse_1_list = []
    mse_2_list = []
    mse_3_list = []

    test_cases = softmax_test_patterns()

    for idx, x in enumerate(test_cases):
        gold_out = SoftMax(x)
        ref1_out  = SoftMax_1(x)
        ref2_out  = SoftMax_2(x)
        ref3_out  = SoftMax_3(x)
        
        mse1 = MSE(gold_out, ref1_out)
        mse2 = MSE(gold_out, ref2_out)
        mse3 = MSE(gold_out, ref3_out)

        mse_1_list.append(mse1)
        mse_2_list.append(mse2)
        mse_3_list.append(mse3)

        print(f"Test case {idx + 1}: {x}")
        print(f"MSE error 1: {mse1}, MSE error 2: {mse2}, MSE error 3: {mse3}")

        print(f'\n\n')

    # print(torch.rand(5))


    # === Plotting ===
    x_labels = [f"Case {i+1}" for i in range(len(test_cases))]
    x_pos = np.arange(len(test_cases))

    plt.figure(figsize=(10, 6))
    plt.bar(x_pos - 0.3, mse_1_list, width=0.3, label='MSE vs Power of two', color='skyblue')
    plt.bar(x_pos + 0.0, mse_2_list, width=0.3, label='MSE vs Fast Inverse Square Root', color='orange')
    plt.bar(x_pos + 0.3, mse_3_list, width=0.3, label='MSE vs Cordic', color='yellow')

    plt.xticks(x_pos, x_labels, rotation=45)
    plt.ylabel("MSE Accuracy")
    plt.title("SoftMax Approximation Comparison (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.show()

    x= float_to_q4_11(11.56)
    y= q4_11_to_float(x)

    print(f"Float to Q4.11: {x}, Q4.11 to Float: {y}")



    x= float_to_q4_11(-11.56)
    y= q4_11_to_float(x)

    print(f"Float to Q4.11: {x}, Q4.11 to Float: {y}")