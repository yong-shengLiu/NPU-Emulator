import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import struct
import torch

from Cordic import cordic
from Cordic import cordic_sqrt
from Cordic import cordic_reciprocal
from Cordic import cordic_mac
from Cordic import cordic_q8_exp
from Cordic import cordic_q8_reciprocal
from Cordic import float_to_q8

def fixQ8_to_float(x: np.ndarray) -> np.ndarray:
    """
    Convert fixed-point Q8(8bit fraction) to float32.
    """
    return x.astype(np.float32) / 256.0

def float_to_q8_8(x):
    scale = 2**8
    fixed = np.round(x * scale).astype(np.int16)
    return fixed

def float_to_int8(x):
    fixed = torch.round(x).to(torch.int8)  # 存成 8b
    return fixed

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


"""
Softmax Related
"""
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
        np.array([0.00001, 0.00002, 0.00003])            #  9: soft rounding
    ]

def SoftMax_Tensor(x):
    """
    NOTE: Row-wise softmax for 2D tensor
    """

    # Find maximum along row-wise
    maximum, _ = torch.max(x, dim=1, keepdim=True)
    print("Max Golden:\n", maximum)

    # subtract maximum
    diff = x - maximum
    print("Diff Golden:\n", diff)

    # exponentiation
    exp_diff = torch.exp(diff)
    print("Exp. diff Golden:\n", exp_diff)
    fxp_exp_diff = float_to_q8_8(exp_diff)
    print("FxP Exp. diff Golden:\n", fxp_exp_diff)

    # summation  TODO: check the bitwidth of summation integer part
    summation = torch.sum(fxp_exp_diff, dim=1, keepdim=True)
    print("Summation Golden:\n", summation, summation.dtype)

    # divide
    softmax_x = exp_diff / summation
    print("Softmax Golden:\n", softmax_x)
    int8_softmax_x = float_to_int8(softmax_x)
    print("Int8 Softmax Golden:\n", int8_softmax_x)
    

    return int8_softmax_x

def softmax_golden(x):
    """
    NOTE: the softmax without quantization, the input dimension is 4D tensor (batch, headm sequence(col), sequence(row))
    """
    # find row maximum
    row_max = np.max(x, axis=-1, keepdims=True)

    diff = x - row_max

    exp_diff = np.exp(diff)

    summation = exp_diff.sum(axis=-1, keepdims=True)

    softmax_x = exp_diff / summation

    return softmax_x

def softmax_quantized(x, mask):
    """
    NOTE: the quantized Softmax, the input dimension is 4D tensor (batch, headm sequence(col), sequence(row))
          along the last dimension (head_size) to do softmax
    (1) input x dtype is fix(8, 8)
    (2) intermediate ??
    (3) output softmax_result dtype is fix(0, 8)
    (4) the reciprocal use Fast Inverse Square Root (FISR)
    """
    # find row maximum
    row_max = np.max(x, axis=-1, keepdims=True)
    
    # Calculate different
    diff = x - row_max
    
    # take log2e
    int_frac = diff + (diff >> 1) # diff * log2e (log2e = 1.5)
    # int_frac = diff + diff >> 1 # diff * log2e (log2e = 1.5)

    # causal mask
    int_frac[mask] = -2**15
    
    # split integer and fractional part
    integer_part = int_frac.astype(np.int32) >> 8
    decimal_part = int_frac - (integer_part << 8)
    
    # Exponential (2^(x*log2e) = 2^integer * 2^fractional)
    exponential = ( ( np.ones_like(integer_part, dtype=np.int32) << 8) >> (-integer_part) )                      # integer part
    exponential = (exponential * ( (np.ones_like(decimal_part, dtype=np.int32) << 8) + (decimal_part>>1) ) >> 8) # integer part * fractional part  (e^frac ~ 1 + frac/2)
    # print("Normal Exp:", exponential)

    # summation and division
    summation = exponential.sum(axis=-1, keepdims=True)

    reciprocal_q8 = FISR_Q88((summation * summation) >> 8, iterations=3)
    softmax_result = (exponential * reciprocal_q8) >> 8 # Softmax result in fix0_8 format

    return softmax_result

def cordic_exp_wrapper(x_q8):
    exp1, exp2, theta_remain, conv = cordic_q8_exp(x_q8, log2e=1.5, iterations=8)
    return exp1
def cordic_recip_wrapper(x_q8):
    x_cordic, y_remain, div_cordic, convergence_list = cordic_q8_reciprocal(x_q8, float_to_q8(1), 0, iterations=8)
    return div_cordic
def softmax_quantized_cordic(x, mask):
    """
    NOTE: the quantized Cordic Softmax, the input dimension is 4D tensor (batch, headm sequence(col), sequence(row))
          along the last dimension (head_size) to do softmax
    (1) input x dtype is fix(8, 8)
    (2) intermediate ??
    (3) output softmax_result dtype is fix(0, 8)
    """
    # find row maximum
    row_max = np.max(x, axis=-1, keepdims=True)
    
    # Calculate different
    diff = x - row_max
    # print(diff)
    # causal mask
    diff[mask] = -2**15
    # print(fixQ8_to_float(diff))
    
    # cordic approximation of exponentials
    vec_cordic = np.vectorize(cordic_exp_wrapper)
    exponential = vec_cordic(diff)
    # print("Cordic Exp:", exponential)
    

    # # split integer and fractional part
    # integer_part = int_frac.astype(np.int32) >> 8
    # decimal_part = int_frac - (integer_part << 8)
    
    # # Exponential (2^(x*log2e) = 2^integer * 2^fractional)
    # exponential = ( ( np.ones_like(integer_part, dtype=np.int32) << 8) >> (-integer_part) )                      # integer part
    # exponential = (exponential * ( (np.ones_like(decimal_part, dtype=np.int32) << 8) + (decimal_part>>1) ) >> 8) # integer part * fractional part  (e^frac ~ 1 + frac/2)

    # summation and division
    summation = exponential.sum(axis=-1, keepdims=True)
    # print("Cordic Summation:", summation)

    vec_cordic = np.vectorize(cordic_recip_wrapper)
    # print("Cordic reciprocal:", vec_cordic(summation))
    # print("Cordic Exp:", exponential)
    
    # softmax_result = (exponential * vec_cordic(summation) / 256).astype(np.uint8)
    softmax_result = (exponential * vec_cordic(summation) / 256)
    # softmax_result = (exponential * 255 / summation).astype(np.uint8) # Softmax result in fix0_8 format
    return softmax_result

def SoftMax(x):
    """
    NOTE: Hardware benchmark list
    (1) SoftMax_1: the exponentials used quantization and linear approximation, the summation is rounded to the nearest power of 2
    (2) SoftMax_2: the exponentials used quantization and linear approximation, and the division is done by Fast Inverse Square Root (FISR)
    (3) SoftMax_3: the exponentials used quantization and Cordic approximation, and the division is done by Cordic inverse
    """
    maximum = np.max(x)

    diff = x - maximum
    print("Diff Golden:", diff)

    exp_diff = np.exp(diff)
    print("Exp. diff Golden:", exp_diff)

    summation = np.sum(exp_diff)

    softmax_x = exp_diff / summation

    print("Softmax Golden:", softmax_x)

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
    # print("Exp. summation:", summation)
    
    # rounde the summation to the nearest power of 2
    summation_rounded = 2 ** np.round(np.log2(summation))
    # print("Exp. summation_rounded:", summation_rounded)


    softmax_x = exp_approximated / summation_rounded

    print("Softmax power:", softmax_x)
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
    # print("Exp. summation:", summation)
    
    # divide by the FastInverse Square Root (FISR)
    softmax_x = exp_approximated * FISR((summation**2), 3)

    print("Softmax FISR:", softmax_x)

    return softmax_x

def SoftMax_3(x):
    
    maximum = np.max(x)
    diff = x - maximum
    
    # cordic approximation of exponentials
    # log2e = 1.442695
    log2e = 1.5
    diff_frac_part, diff_int_part = np.modf(diff)

    exp_diff_list = []

    for i in range(len(diff)):
        e_int = 2**(diff_int_part[i] * log2e)  # TODO: the odd cannot shift, need to used LUT

        exp_diff, _, _, _ = cordic(e_int, e_int, diff_frac_part[i], m=-1, iterations=32, mode='rotation')

        # exp_diff_list.append(e_int * e_frac)
        exp_diff_list.append(exp_diff)

    exp_diff = np.array(exp_diff_list)
    
    print("diff:", diff)
    print("Exp. diff Cordic:", exp_diff)
    summation = np.sum(exp_diff)

    # cordic approximation of inverse
    softmax_list = []
    for exp_d in exp_diff:
        _, _, div_cordic, _ = cordic(summation, exp_d, 0, m=0, iterations=32, mode='vectoring')
        softmax_list.append(div_cordic)

    print("Softmax Cordic:", softmax_list)
    
    softmax_x = np.array(softmax_list)

    return softmax_x



def GELU(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def selu(x, alpha = 1.6732, lambda_ = 1.0507):
    return np.where(x > 0, lambda_ * x, lambda_ * alpha * (np.exp(x) - 1))


"""
LayerNorm Related
"""
def layernorm_test_patterns():
    return [
        np.array([1.0, 1.0, 1.0, 1.0]),      # 1: uniform input (no variance)
        np.array([1.0, 1.01, 0.99, 1.02]),   # 2: small noise around constant (low variance)
        np.array([1.0, 5.0, 30.0, 100.0]),   # 3: large range values (high variance)
        np.array([-1.0, 0.0, 1.0]),          # 4: centered around zero
        np.array([-10.0, 0.0, 10.0]),        # 5: symmetric but wider range
        np.array([0.0, 0.0, 0.0, 10.0]),     # 6: one-hot-like (one large value among zeros)
        np.array([-5.0, -10.0, -2.0, -1.0]), # 7: negative dominant (simulate bias drift)
        np.array([0.1, 0.2, 0.3, 100.0]),    # 8: outlier case (one huge value)
        np.random.normal(0, 1, 8),           # 9: random normal
        np.random.uniform(-3, 3, 8)          # 10: random uniform
    ]

def LayerNorm(x, gamma=None, beta=None):
    """
    NOTE: Hardware benchmark list
    (1) LayerNorm1: Fasst Inverse Square Root (FISR)
    (2) LayerNorm2: Cordic approximation
    """
    # Step 1: compute mean and variance along last dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)

    # Step 2: normalize
    normalized = (x - mean) / np.sqrt(variance)

    # Step 3: scale and shift (optional)
    if gamma is not None:
        normalized *= gamma
    if beta is not None:
        normalized += beta

    return normalized

def LayerNorm_1(x, gamma=None, beta=None):
    """
    NOTE:
    uses Fast Inverse Square Root (FISR) as divider and 1/sqrt(x)
    """

    # mean
    summation = np.sum(x)
    mean = summation * FISR((x.shape[-1]**2), 3) # assuming last dimension is the feature dimension
    print("Mean:", mean)


    # substract mean
    difference = x - mean
    print("Difference:", difference)


    # variance
    difference_squared = difference ** 2
    variance = np.sum(difference_squared) * FISR((x.shape[-1]**2), 3)  # assuming last dimension is the feature dimension


    # difference * 1/square(variance)
    normalized = difference * FISR(variance, 3)  # using FISR for fast inverse square root


    # scale gamma, bias beta
    if gamma is not None:
        normalized *= gamma
    if beta is not None:
        normalized += beta

    return normalized

def LayerNorm_2(x, gamma=None, beta=None):
    """
    NOTE: uses Cordic
    """

    # mean
    summation = np.sum(x)
    _, _, divide, _ = cordic(x.shape[-1], 1, 0, m=0, iterations=32, mode='vectoring')  # 1 / x.shape[-1]
    _, mean, _, _ = cordic(summation, 0, divide, m=0, iterations=32, mode='rotation')  # mean = summation / x.shape[-1]
    # print(f"Mean: {mean}, shape: {x.shape[-1]}, summation: {summation}")


    # substract mean
    difference = x - mean
    # print("Difference:", difference)


    # variance
    difference_squared_list = []

    for i in range(len(difference)):
        # print(f"Difference[{i}]: {difference[i]}")

        # normalize the difference to avoid out of Cordic range
        if difference[i] != 0:
            abs_difference = abs(difference[i])
            power = int(np.log2(abs_difference))
            diff_norm = abs_difference / (2**power)
        else:
            diff_norm = difference[i]
        
        _, difference_squared, _, _ = cordic(diff_norm, 0, diff_norm, m=0, iterations=32, mode='rotation')
        
        if difference[i] != 0:
            difference_squared *= (2**power) * (2**power)  # compensate the gain
        difference_squared_list.append(difference_squared)

    difference_squared = np.array(difference_squared_list)


    # variance = difference_squared_summation / x.shape[-1]
    difference_squared_summation = np.sum(difference_squared)
    _, _, divide, _ = cordic(x.shape[-1], 1, 0, m=0, iterations=32, mode='vectoring')  # 1 / x.shape[-1]
    _, variance, _, _ = cordic(difference_squared_summation, 0, divide, m=0, iterations=32, mode='rotation')


    # difference * 1/square(variance)
    square_variance, _, _, _ = cordic_sqrt(variance, threshold=16, scale=1024, iterations=32) # square(variance)
    # print(f"Variance: {variance}, square_variance: {square_variance}")

    # _, _, reciprocal, _ = cordic(square_variance, 1, 0, m=0, iterations=32, mode='vectoring')  # 1 / square_variance
    _, _, reciprocal, _ = cordic_reciprocal(square_variance, 1, 0, iterations=32)  # 1 / square_variance
    # print(f"square_variance: {square_variance}, reciprocal: {reciprocal}")
    
    # normalized = difference * reciprocal
    normalized_list = []

    for i in range(len(difference)):
        # print(f"Difference[{i}]: {difference[i]}")
        _, normalized, _, _  = cordic_mac(difference[i], 0, reciprocal, iterations=32)
        normalized_list.append(normalized)
    normalized = np.array(normalized_list)
    
    # print(f"difference: {difference}, Cordic normalized: {normalized}, Golden normalized: {difference * reciprocal}")
    


    # scale gamma, bias beta
    if gamma is not None:
        # normalized *= gamma
        normalized_list = []

        for i in range(len(normalized)):
            # print(f"Normalized[{i}]: {normalized[i]}, gamma: {gamma}, beta: {beta}")
            _, normalized_temp, _, _  = cordic_mac(normalized[i], 0, gamma, iterations=32)
            normalized_list.append(normalized_temp)
        
        normalized = np.array(normalized_list)

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

def FISR_Q88(x_q8, iterations=2):
    """ Fast Inverse Square Root in Q8.8 format """

    # 初始猜測: 使用近似常數 (0x5F3759DF 對應浮點，這裡簡化成經驗初值)
    # 為了 Q8.8，我們可以用 y = 1 / sqrt(x) ≈ 1.0 / sqrt(x)
    # 假設輸入範圍 [0.25, 4.0]
    # 初值可用: y0 = 1.0 (Q8.8 = 256)
    y_q8 = 256  

    one_point_five_q8 = int(1.5 * 256)  # 1.5 in Q8.8
    half_q8 = 128                       # 0.5 in Q8.8

    for _ in range(iterations):
        # y^2
        y2_q8 = (y_q8 * y_q8) >> 8
        # x * y^2
        xy2_q8 = (x_q8 * y2_q8) >> 8
        # 0.5 * x * y^2
        half_xy2_q8 = (half_q8 * xy2_q8) >> 8
        # (1.5 - 0.5*x*y^2)
        correction_q8 = one_point_five_q8 - half_xy2_q8
        # y * correction
        y_q8 = (y_q8 * correction_q8) >> 8

    return np.int32(y_q8)


def MSE(golden, prediction):
    """ Mean Squared Error (MSE) """

    if len(golden) != len(prediction):
        raise ValueError("Length of golden and prediction must match")

    mse = np.mean((prediction - golden) ** 2)

    if mse == 0:
        accuracy = float('inf')  # Perfect match
        print("Perfect match!")
        loss = 0
    else:
        accuracy = math.log10(1 / mse)  # More larger is better
        accuracy = mse                  # More smaller is better
        loss = mse

    return accuracy


if __name__ == "__main__":
    print("=== SFU testbench 2025.10.13 ===")
    
    ### ---------- softmax quantization ---------- ###
    QK   = np.load("SM_input_fix88.npy")
    mask = np.load("SM_mask.npy")
    print(QK.shape, QK.dtype, mask.shape, mask.dtype)  # int16(1, 8, 3, 3)   bool1, 8, 3, 3)

    # QK = np.random.uniform(-8, 8, size=(1, 8, 3, 3)).astype(np.int16)
    # mask = np.random.randint(0, 2, size=(1, 8, 3, 3), dtype=bool)
    # print(QK.shape, QK.dtype, mask.shape, mask.dtype)  # int16(1, 8, 3, 3)   bool1, 8, 3, 3)


    f_QK = fixQ8_to_float(QK)
    softmax_golden_result = softmax_golden(f_QK)
    # softmax_golden_result = float_to_q8_8(softmax_golden_result)
    print(f"Golden: \n{softmax_golden_result}")


    softmax_result = softmax_quantized(QK, mask)
    f_softmax_result = fixQ8_to_float(softmax_result)
    # print(QK)
    # print(f_QK)
    print(f"f_softmax_result: \n{f_softmax_result}")
    mes = MSE(softmax_golden_result, f_softmax_result)
    print(f"MSE: {mes}")


    # print(QK)
    cordic_softmax_result = softmax_quantized_cordic(QK, mask)
    f_cordic_softmax_result = fixQ8_to_float(cordic_softmax_result)
    print(f"f_cordic_softmax_result: \n{f_cordic_softmax_result}")
    mes = MSE(softmax_golden_result, f_cordic_softmax_result)
    print(f"Cordic MSE: {mes}")

    # ### ---------- SoftMax Benchmarking ---------- ###
    # mse_1_list = []
    # mse_2_list = []
    # mse_3_list = []

    # test_cases = softmax_test_patterns()

    # for idx, x in enumerate(test_cases):
    #     gold_out = SoftMax(x)
    #     ref1_out  = SoftMax_1(x)
    #     ref2_out  = SoftMax_2(x)
    #     ref3_out  = SoftMax_3(x)
        
    #     mse1 = MSE(gold_out, ref1_out)
    #     mse2 = MSE(gold_out, ref2_out)
    #     mse3 = MSE(gold_out, ref3_out)

    #     mse_1_list.append(mse1)
    #     mse_2_list.append(mse2)
    #     mse_3_list.append(mse3)

    #     print(f"Test case {idx + 1}: {x}")
    #     print(f"MSE error 1: {mse1}, MSE error 2: {mse2}, MSE error 3: {mse3}")

    #     print(f'\n\n')

    # # === Plotting ===
    # x_labels = [f"Case {i+1}" for i in range(len(test_cases))]
    # x_pos = np.arange(len(test_cases))

    # plt.figure(figsize=(10, 6))
    # plt.bar(x_pos - 0.3, mse_1_list, width=0.3, label='MSE vs Power of two', color='yellow')
    # plt.bar(x_pos + 0.0, mse_2_list, width=0.3, label='MSE vs Fast Inverse Square Root', color='skyblue')
    # plt.bar(x_pos + 0.3, mse_3_list, width=0.3, label='MSE vs Cordic', color='orange')

    # plt.xticks(x_pos, x_labels, rotation=45)
    # plt.ylabel("MSE Accuracy")
    # plt.title("SoftMax Approximation Comparison (MSE)")
    # plt.legend()
    # plt.tight_layout()
    # plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    # plt.show()

    

    ## ---------- LayerNorm Benchmarking ---------- ###
    # mse_1_list = []
    # mse_2_list = []
    # # mse_3_list = []

    # test_cases = layernorm_test_patterns()

    # for idx, x in enumerate(test_cases):
    #     gold_out = LayerNorm(x, gamma=1.03, beta=0.02)
    #     ref1_out  = LayerNorm_1(x, gamma=1.03, beta=0.02)
    #     ref2_out  = LayerNorm_2(x, gamma=1.03, beta=0.02)
    #     # ref3_out  = LayerNorm_3(x)
        
    #     mse1 = MSE(gold_out, ref1_out)
    #     mse2 = MSE(gold_out, ref2_out)
    #     # mse3 = MSE(gold_out, ref3_out)

    #     mse_1_list.append(mse1)
    #     mse_2_list.append(mse2)
    #     # mse_3_list.append(mse3)

    #     print(f"Test case {idx + 1}: {x}")
    #     # print(f"MSE error 1: {mse1}, MSE error 2: {mse2}, MSE error 3: {mse3}")
    #     print(f"MSE error 1: {mse1}, MSE error 2: {mse2}")
    #     print(f"Gold: {gold_out}, ref1: {ref1_out}, ref2: {ref2_out}")

        

    #     print(f'\n\n')


    # === Plotting ===
    # plt.figure(figsize=(10, 6))
    # x_labels = [f"Case {i+1}" for i in range(len(test_cases))]
    # x_pos = np.arange(len(test_cases))

    # for i in range(len(test_cases)):
    #     x = x_pos[i]
        
    #     # mse_1
    #     if np.isfinite(mse_1_list[i]):
    #         plt.bar(x - 0.15, mse_1_list[i], width=0.3, label='MSE vs Fast Inverse Square Root' if i == 0 else "", color='skyblue')
    #     else:
    #         plt.text(x - 0.15, 0.05, 'inf', ha='center', va='bottom', color='blue', rotation=0)

    #     # mse_2
    #     if np.isfinite(mse_2_list[i]):
    #         plt.bar(x + 0.15, mse_2_list[i], width=0.3, label='MSE vs Cordic' if i == 0 else "", color='orange')
    #     else:
    #         plt.text(x + 0.15, 0.05, 'inf', ha='center', va='bottom', color='orange', rotation=0)


    # plt.xticks(x_pos, x_labels, rotation=45)
    # plt.ylabel("MSE Accuracy")
    # plt.title("LayerNorm Approximation Comparison (MSE)")
    # plt.tight_layout()
    # plt.grid(True, axis='y', linestyle='--', alpha=0.5)


    # patch1 = mpatches.Patch(color='skyblue', label='MSE vs Fast Inverse Square Root')
    # patch2 = mpatches.Patch(color='orange', label='MSE vs Cordic')
    # plt.legend(handles=[patch1, patch2])
    # plt.show()





    ## ---------- Garbege test ---------- ###
    # x= float_to_q4_11(11.56)
    # y= q4_11_to_float(x)

    # print(f"Float to Q4.11: {x}, Q4.11 to Float: {y}")


    # x= float_to_q4_11(-11.56)
    # y= q4_11_to_float(x)

    # print(f"Float to Q4.11: {x}, Q4.11 to Float: {y}")


    # x = np.array(
    #     [[1,  2,  3],     # batch 0, time 0
    #     [4,  5,  6]],    # batch 0, time 1
    # )
    # print(x.shape)
    # print(np.sum(x, axis=0))
    # print(np.sum(x, axis=1))


    # x = np.array([
    #     [[1,  2,  3], [4,  5,  6]],

    #     [[7,  8,  9], [10, 11, 12]]
    # ])
    # print(x.shape)  # (2, 2, 3)

    # print(np.sum(x, axis=0))
    # print(np.sum(x, axis=1))
    # print(np.sum(x, axis=2))