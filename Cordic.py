import math
import numpy as np
import matplotlib.pyplot as plt
import struct


"""
Useful conversion functions
"""
FRAC_BITS = 8   # Q8 小數位
SCALE = 1 << FRAC_BITS

def float_to_q8(x: float) -> int:
    return int(round(x * SCALE))

def q8_to_float(x_q8: int) -> float:
    return x_q8 / SCALE

def float_to_q16(x: float) -> int:
    return int(round(x * (1 << 16)))

def q16_to_float(x_q16: int) -> float:
    return x_q16 / (1 << 16)

def near_even(x):
    lower = math.floor(x / 2) * 2   # 向下最近偶數
    upper = lower + 2                # 向上最近偶數

    # 比較哪個更接近 x
    if abs(x - lower) <= abs(x - upper):
        return lower
    else:
        return upper

def normalize_q8(x):
    if x == 0:
        return 0, 0  # 避免除以零
    
    abs_x = abs(x)
    # Q8.8 範圍下，理想希望 x >= 0.5 (128 in Q8.8)
    target = 128  
    if abs_x < target:
        shift = (target.bit_length() - abs_x.bit_length())
        x_scaled = x << shift
    else:
        shift = 0
        x_scaled = x
    return x_scaled, shift


"""
Q88 CORDIC implementation
"""
def cordic_q8(x_q8, y_q8, theta_q8, m, iterations=1, mode='circular'):
    """
    NOTE: the input x, y, theta are all in Q8 format (fixed-point with 8 fractional bits)
    (1) m=1,  rotation mode(V), vectoring mode(V)   (circular)
    (2) m=0,  rotation mode(V), vectoring mode(V)   (linear)
    (3) m=-1, rotation mode(V), vectoring mode(V)   (hyperbolic)
    """
    # threshold = 1e-20
    threshold = 1
    hyperbolic_iteration = [
        1, 2, 3, 4, 4,
        5, 6, 7, 8, 9, 10, 11, 12, 13, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 40
    ]
    if m == -1 and iterations > len(hyperbolic_iteration):
        iterations = len(hyperbolic_iteration)
    convergence_list = []

    # Precompute the arctangent table for each iteration (Q8)
    if m == 1:
        LUT_table = [float_to_q8(math.atan(2**-i)) for i in range(iterations)]  # tan^-1(2^-i)
    elif m == 0:
        LUT_table = [float_to_q8(2**-i) for i in range(iterations)] # 2^-i
    elif m == -1:
        LUT_table = [int(round(math.atanh(2**-i) * 256)) for i in hyperbolic_iteration[:iterations]]
    else:
        raise ValueError("Invalid m parameter")
    # print(f"LUT_table: {LUT_table}")

    
    # CORDIC gain (will shrink, need to compensate at the end)
    K = 1.0
    if m == 1:
        for i in range(iterations):
            K *= 1 / math.sqrt(1 + 2**(-2*i))
    elif m == 0:
        K = 1.0
    elif m == -1:
        for i in hyperbolic_iteration[:iterations]:
            if theta_q8 == 0:
                break
            K *= 1 / math.sqrt(1 - 2**(-2*i))
    else:
        raise ValueError("Invalid m")
    
    K = float_to_q8(K)
            
    # Perform iterative rotation
    if m == -1:
        for i in range(iterations):
            if theta_q8 == 0:
                # print(f"Early stop at iteration {i}, theta almost 0")
                break
            
            idx = hyperbolic_iteration[i]
            if mode == "rotation":
                di = 1 if theta_q8 >= 0 else -1
                convergence_list.append(theta_q8)
            elif mode == "vectoring":
                di = -1 if y_q8 >= 0 else 1
                convergence_list.append(y)
            else:
                raise ValueError("Invalid mode")
            
            x_new = x_q8 + ((di * y_q8) >> idx)
            y_new = y_q8 + ((di * x_q8) >> idx)
            theta_q8 -= di * LUT_table[i]
            # theta -= di * LUT_table[idx]
            x_q8, y_q8 = x_new, y_new
            # print(f'x_q8: {q8_to_float(x_q8)}, y_q8: {q8_to_float(y_q8)}, theta_q8: {q8_to_float(theta_q8)}, idx: {idx}, LUT: {q8_to_float(LUT_table[i])}')
            # print(f"y{i}_{hyperbolic_iteration[i]}: {y_new}")
    else:
        for i in range(iterations):
            # Early termination: y is already close enough to 0
            if mode == "rotation":
                di = 1.0 if theta_q8 >= 0 else -1.0
                convergence_list.append(theta_q8)
                
                if abs(theta_q8) < threshold:
                    print(f"Early stop at iteration {i}, theta almost 0")
                    break
                
            elif mode == "vectoring":
                di = -1.0 if y_q8 >= 0 else 1.0
                convergence_list.append(y_q8)
                
                if abs(y_q8) < threshold:
                    print(f"Early stop at iteration {i}, y almost 0")
                    break

            else:
                raise ValueError("Invalid mode")

            x_new = x_q8 - (int(m * di * y_q8) >> i)
            y_new = y_q8 + (int(di * x_q8)     >> i)
            theta_q8 -= di * LUT_table[i]
            x_q8, y_q8 = x_new, y_new
            # print(f"theta{i}: {theta_q8}, x{i}: {x_q8}, y{i}: {y_q8}")

    # print(f"Final K: {K}")
    # Output with gain compensation
    if m == 1 or m == -1:
        # print(f'Before K compensation: x: {x_q8}, y: {y_q8}, theta: {theta_q8}')
        return int(x_q8 * K)>>8, int(y_q8 * K)>>8, theta_q8, convergence_list
        # return int(x_q8 * K)>>16, int(y_q8 * K)>>16, theta_q8, convergence_list
        # return ((x_q8 << 8) * K)>>24, ((y_q8 << 8) * K)>>24, theta_q8, convergence_list
        # return (x_q8 * K)/256, (y_q8 * K)/256, theta_q8, convergence_list
    elif m == 0:
        return x_q8, y_q8, theta_q8, convergence_list
    
def cordic_q8_exp(x_q8, log2e, iterations=8):
    
    # Q8.8 fixed-point to float
    x = q8_to_float(x_q8)

    int_part = near_even(x)
    frac_part = x - int_part

    # print(f"x_q8={x_q8}, int_part={int_part}, frac_part={frac_part}")
    
    temp = int_part + (int_part >> 1)
    # print(f'temp: {temp}')
    # temp = int_part # accuracy is more higer??
    e_int = int((1<<8) >> (-temp))
    # if temp < 0:
    #     temp = -temp
    #     e_int = int((1<<8) >> (temp))
    # else:
    #     e_int = int((1<<8) << (temp))
    # print(f'e_int: {e_int}, q8_to_float: {q8_to_float(e_int)}')
    # e_int = 2**(int_part * log2e)  # log2(e) ≈ 1.5
    # e_int = float_to_q8(e_int)

    exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic_q8(e_int, e_int, float_to_q8(frac_part), m=-1, iterations=iterations, mode='rotation')
    
    return exp1_cordic, exp2_cordic, theta_remain, convergence_list

def cordic_q8_reciprocal(x, y, z, iterations=8):
    # z + y/x
    # if x < float_to_q8(0.25):
    #     print("x is too small, scale it up")
    #     # scale_x = x * 512
    #     scale_x = x << 6
    # else:
    #     scale_x = x

    scale_x, shift = normalize_q8(x)

    # print(f'scale_x: {scale_x}, y: {y}, z: {z}')
    # Perform CORDIC iterations
    x_cordic, y_remain, div_cordic, convergence_list = cordic_q8(scale_x, y, z, m=0, iterations=iterations, mode='vectoring')

    # Compensate the gain
    # print(f'div_cordic: {div_cordic}')
    # if x < float_to_q8(0.25):
    #     # div_cordic = div_cordic * 512
    #     div_cordic = int(div_cordic) << 6

    div_cordic = int(div_cordic) << shift

    return x_cordic, y_remain, div_cordic, convergence_list


"""
Floating-point CORDIC implementation
"""
def cordic(x, y, theta, m, iterations=1, mode='circular'):
    """
    NOTE
    (1) m=1,  rotation mode(V), vectoring mode(V)   (circular)
    (2) m=0,  rotation mode(V), vectoring mode(V)   (linear)
    (3) m=-1, rotation mode(V), vectoring mode(V)   (hyperbolic)
    """
    threshold = 1e-20
    hyperbolic_iteration = [
        1, 2, 3, 4, 4,
        5, 6, 7, 8, 9, 10, 11, 12, 13, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 40
    ]
    if m == -1 and iterations > len(hyperbolic_iteration):
        iterations = len(hyperbolic_iteration)
    convergence_list = []

    # Precompute the arctangent table for each iteration
    if m == 1:
        LUT_table = [math.atan(2**-i) for i in range(iterations)]  # tan^-1(2^-i)
    elif m == 0:
        LUT_table = [2**-i for i in range(iterations)] # 2^-i
    elif m == -1:
        LUT_table = [math.atanh(2**-i) for i in hyperbolic_iteration[:iterations]]
    else:
        raise ValueError("Invalid m parameter")
    # print(f"LUT_table: {LUT_table}")
    
    # CORDIC gain (will shrink, need to compensate at the end)
    K = 1.0
    if m == 1:
        for i in range(iterations):
            K *= 1 / math.sqrt(1 + 2**(-2*i))
    elif m == 0:
        K = 1.0
    elif m == -1:
        for i in hyperbolic_iteration[:iterations]:
            K *= 1 / math.sqrt(1 - 2**(-2*i))
            
    # Perform iterative rotation
    if m == -1:
        for i in range(iterations):
            idx = hyperbolic_iteration[i]
            if mode == "rotation":
                di = 1.0 if theta >= 0 else -1.0
                convergence_list.append(theta)
            elif mode == "vectoring":
                di = -1.0 if y >= 0 else 1.0
                convergence_list.append(y)
            else:
                raise ValueError("Invalid mode")
            

            x_new = x + di * y * 2**(-idx)
            y_new = y + di * x * 2**(-idx)
            theta -= di * LUT_table[i]
            # theta -= di * LUT_table[idx]
            x, y = x_new, y_new

            print(f'x: {x}, y: {y}, theta: {theta}, idx: {idx}, LUT: {LUT_table[i]}')
            # print(f"y{i}_{hyperbolic_iteration[i]}: {y_new}")
    else:
        for i in range(iterations):
            # Early termination: y is already close enough to 0
            if mode == "rotation":
                di = 1.0 if theta >= 0 else -1.0
                convergence_list.append(theta)
                
                if abs(theta) < threshold:
                    print(f"Early stop at iteration {i}, theta almost 0")
                    break
                
            elif mode == "vectoring":
                di = -1.0 if y >= 0 else 1.0
                convergence_list.append(y)
                
                if abs(y) < threshold:
                    print(f"Early stop at iteration {i}, y almost 0")
                    break
                
            else:
                raise ValueError("Invalid mode")


            x_new = x - m * di * y * 2**(-i)
            y_new = y + di * x * 2**(-i)
            theta -= di * LUT_table[i]
            x, y = x_new, y_new
            # print(f"theta{i}: {theta}, x{i}: {x}, y{i}: {y}")


    # Output with gain compensation
    if m == 1 or m == -1:
        return x * K, y * K, theta, convergence_list
    elif m == 0:
        return x, y, theta, convergence_list

def cordic_exp(value, iterations=32):

    # value = value * 1.4426950408889634
    # value = value * 1.5

    frac_part, int_part = np.modf(value)
    # print(f'int_part: {int_part}, frac_part: {frac_part}')
    # e_int = 2**(int_part * 1.4426950408889634)  # log2(e) ≈ 1.4426950408889634 = 1.5 - 0.057305
    # print(int_part * 1.5)
    e_int = 2**(int_part * 1.5)  # log2(e) ≈ 1.5
    # e_int = 2**(int_part * 1.4375)  # log2(e) ≈ 1.5
    # e_int = 2**(int_part)  # log2(e) ≈ 1.5
    # print(f'e_int: {e_int}, frac_part: {frac_part}')

    exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic(e_int, e_int, frac_part, m=-1, iterations=iterations, mode='rotation')

    return exp1_cordic, exp2_cordic, theta_remain, convergence_list

def cordic_reciprocal(x, y, z, iterations=32):
    # z + y/x
    if x < 0.5:
        print("x is too small, scale it up")
        scale_x = x * 512
    else:
        scale_x = x

    # Perform CORDIC iterations
    x_cordic, y_remain, div_cordic, convergence_list = cordic(scale_x, y, z, m=0, iterations=iterations, mode='vectoring')

    # Compensate the gain
    if x < 0.5:
        div_cordic = div_cordic * 512

    return x_cordic, y_remain, div_cordic, convergence_list

def cordic_sqrt(x, threshold=4, scale=1024, iterations=32):

    if x < 1e-10:
        scale_x = 0
    elif x < (0.25 / threshold):
        print("x is too small, scale it up")
        scale_x = x * 1024
        
    # elif x > (0.25 * threshold) and x < (0.25 * threshold * 4):
    #     print("x is little large, scale it down")
    #     scale_x = x / 16
    elif x >= (0.25 * threshold) and x < (0.25 * threshold ** 2):
        print("x is large, scale it down")
        scale_x = x / 256
    elif x >= (0.25 * threshold ** 2):
        print("x is too large, scale it down")
        scale_x = x / 1024
    else:
        print("x is in normal range, no scaling")
        scale_x = x

    print(f"y/x: {(scale_x-1/4)/(scale_x+1/4)}")
    square_cordic, y_remain, tanh_cordic, convergence_list = cordic( (scale_x+1/4), (scale_x-1/4), 0, m=-1, iterations=iterations, mode='vectoring')
    
    if x < 1e-10:
        square_cordic = np.sqrt(x)
    elif x < (0.25 / threshold):
        square_cordic = square_cordic / np.sqrt(1024)
    # elif x > (0.25 * threshold) and x < (0.25 * threshold * 4):
    #     square_cordic = square_cordic / np.sqrt(16)
    elif x >= (0.25 * threshold) and x < (0.25 * threshold ** 2):
        square_cordic = square_cordic * np.sqrt(256)
    elif x >= (0.25 * threshold ** 2):
        square_cordic = square_cordic * np.sqrt(1024)
    else:
        square_cordic = square_cordic

    return square_cordic, y_remain, tanh_cordic, convergence_list

def cordic_mac(x, y, z, iterations=32):
    # y + xz

    if x > 0:
        power_x = int(np.log2(x))
    else:
        power_x = 0
    if z > 0:
        power_z = int(np.log2(z))
    else:
        power_z = 0

    x_norm = x / (2**power_x)
    z_norm = z / (2**power_z)

    x_cordic, acc_cordic, theta_remain, convergence_list = cordic(x_norm, y, z_norm, m=0, iterations=iterations, mode='rotation')
    
    acc_cordic *= (2**power_x) * (2**power_z)  # compensate the gain


    return x_cordic, acc_cordic, theta_remain, convergence_list


"""
CBI quantized
"""
def exp_quantized(x_q8):
    """
    Quantized exponential approximation for one Q8.8 input.
    Input:  x_q8 (int or numpy int16) -- fixed-point Q8.8
    Output: exp(x) approximated in Q0.8 format (0~1 range)
    """

    # 1. log2(e) ≈ 1.5 → x * log2(e) ≈ x + x/2
    int_frac = x_q8 + (x_q8 >> 1)   # still in Q8.8

    # 2. split into integer and fractional parts
    integer_part = int_frac >> 8
    frac_part = int_frac - (integer_part << 8)  # still Q8.8, range [0..255]

    # 3. compute 2^integer_part
    #    (1 << 8) represents "1.0 in Q0.8"
    if integer_part < 0:
        exp_int = (1 << 8) >> (-integer_part)
    else:
        exp_int = (1 << 8) << integer_part

    # 4. approximate 2^fractional ≈ 1 + frac/2
    exp_frac = (1 << 8) + (frac_part >> 1)  # Q0.8 + Q0.8

    # 5. combine integer and fractional
    exp_out = (exp_int * exp_frac) >> 8  # back to Q0.8

    return exp_out

def fisr_q8(x_q8, iterations=3):
    """ Fast Inverse Square Root in Q8.8 """
    if x_q8 <= 0:
        x_q8 = 1  # clamp 避免 0 導致錯誤
    
    # 轉 float 運算 (維持精度)
    x = x_q8 / 256.0
    
    # 初始值
    threehalfs = 1.5
    y = x
    x2 = x * 0.5

    # bit-level trick
    i = struct.unpack('I', struct.pack('f', y))[0]
    i = 0x5f3759df - (i >> 1)
    y = struct.unpack('f', struct.pack('I', i))[0]

    # Newton-Raphson 疊代
    for _ in range(iterations):
        y = y * (threehalfs - (x2 * y * y))

    # 回轉 Q8.8
    return int(round(y * 256))


"""
Analyze function
"""
def analyze_cordic(mode = 'cos_sin_float'):
    
    if mode == 'exp_compare':
        """
        Compare the approximation error between exp_normal and exp_Q88
        """
        # 測試範圍
        x = np.linspace(-10, 0, 800)

        # --- exp_normal ---
        errors_exp_normal = []
        for t in x:
            exp_out = exp_quantized(float_to_q8(t))
            errors_exp_normal.append(abs(q8_to_float(exp_out) - math.exp(t)))

        # --- exp_Q88 ---
        errors_exp_Q88 = []
        for t in x:
            exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic_q8_exp(float_to_q8(t), log2e=1.5, iterations=8)
            errors_exp_Q88.append(abs(q8_to_float(exp1_cordic) - math.exp(t)))

        # --- plot ---
        plt.plot(x, errors_exp_normal, label="exp_quantized error")
        plt.plot(x, errors_exp_Q88, label="cordic_q8_exp error")
        plt.xlabel("x")
        plt.ylabel("Absolute Error")
        plt.title("exp approximation error comparison")
        plt.legend()
        plt.show()

        # --- print 統計結果 ---
        print("[exp_quantized] max error:", max(errors_exp_normal))
        print("[exp_quantized] avg error:", sum(errors_exp_normal)/len(errors_exp_normal))
        print("[cordic_q8_exp] max error:", max(errors_exp_Q88))
        print("[cordic_q8_exp] avg error:", sum(errors_exp_Q88)/len(errors_exp_Q88))

    if mode == 'exp_normal':
        # convergence range: |theta| < atanh(1) ≈ 0.881
        # x = np.linspace(-0.88, 0.88, 200)
        x = np.linspace(-10, 0, 800)
        # x = np.linspace(-50, 10, 800)
        cordic_exp_list = []
        # true_exp   = []
        errors_exp = []

        for t in x:
            # exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic(1, 1, t, m=-1, iterations=32, mode='rotation')
            print(f'\nCalculating exp({t}):')
            exp_out = exp_quantized(float_to_q8(t))

            cordic_exp_list.append(exp_out)
            # true_sin.append(math.cos(t))
            errors_exp.append(abs(q8_to_float(exp_out) - math.exp(t)))
            print(f't: {t}, exp_cordic: {exp_out}, exp_true: {float_to_q8(math.exp(t))}, error: {errors_exp[-1]}')
        
        plt.plot(x, errors_exp, label="exp error")
        plt.xlabel("theta (rad)")
        plt.ylabel("Absolute Error")
        # plt.title("CORDIC exp error")
        plt.title("exp error")
        plt.legend()
        plt.show()

        print("exp 最大誤差:", max(errors_exp))
        print("exp 平均誤差:", sum(errors_exp)/len(errors_exp))

    if mode == 'exp_Q88':
        # convergence range: |theta| < atanh(1) ≈ 0.881
        # x = np.linspace(-0.88, 0.88, 200)
        x = np.linspace(-10, 0, 800)
        # x = np.linspace(-50, 10, 800)
        cordic_exp_list = []
        # true_exp   = []
        errors_exp = []

        for t in x:
            # exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic(1, 1, t, m=-1, iterations=32, mode='rotation')
            print(f'\nCalculating exp({t}):')
            exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic_q8_exp(float_to_q8(t), log2e=1.5, iterations=8)

            cordic_exp_list.append(exp1_cordic)
            # true_sin.append(math.cos(t))
            errors_exp.append(abs(q8_to_float(exp1_cordic) - math.exp(t)))
            print(f't: {t}, exp_cordic: {exp1_cordic}, exp_true: {float_to_q8(math.exp(t))}, error: {errors_exp[-1]}')
        
        plt.plot(x, errors_exp, label="exp error")
        plt.xlabel("theta (rad)")
        plt.ylabel("Absolute Error")
        # plt.title("CORDIC exp error")
        plt.title("exp error")
        plt.legend()
        plt.show()

        print("exp 最大誤差:", max(errors_exp))
        print("exp 平均誤差:", sum(errors_exp)/len(errors_exp))

    if mode == 'exp_flaot':
        # convergence range: |theta| < atanh(1) ≈ 0.881
        # x = np.linspace(-0.88, 0.88, 200)
        x = np.linspace(-4, 2, 800)
        # x = np.linspace(-50, 10, 800)
        cordic_exp_list = []
        # true_exp   = []
        errors_exp = []

        for t in x:
            # exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic(1, 1, t, m=-1, iterations=32, mode='rotation')
            print(f'\nCalculating exp({t}):')
            exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic_exp(t, iterations=32)

            cordic_exp_list.append(exp1_cordic)
            # true_sin.append(math.cos(t))
            errors_exp.append(abs(exp1_cordic - math.exp(t)))
            print(f't: {t}, exp_cordic: {exp1_cordic}, exp_true: {math.exp(t)}, error: {errors_exp[-1]}')
        
        plt.plot(x, errors_exp, label="exp error")
        plt.xlabel("theta (rad)")
        plt.ylabel("Absolute Error")
        # plt.title("CORDIC exp error")
        plt.title("exp error")
        plt.legend()
        plt.show()

        print("exp 最大誤差:", max(errors_exp))
        print("exp 平均誤差:", sum(errors_exp)/len(errors_exp))


    if mode == 'reciprocal_compare':
        x = np.linspace(0.01, 1, 800)

        errors_fisr = []
        errors_cordic = []

        for t in x:
            # --- FISR ---
            inv_sum_q8 = fisr_q8((float_to_q8(t) * float_to_q8(t)) // 256, iterations=3)
            errors_fisr.append(abs(q8_to_float(inv_sum_q8) - 1/t))

            # --- CORDIC reciprocal ---
            x_cordic, y_remain, div_cordic, convergence_list = cordic_q8_reciprocal(float_to_q8(t), float_to_q8(1), 0, iterations=8)
            errors_cordic.append(abs(q8_to_float(div_cordic) - 1/t))

        # --- Plot comparison ---
        plt.plot(x, errors_fisr, label="FISR reciprocal error", color='blue')
        plt.plot(x, errors_cordic, label="CORDIC reciprocal error", color='red')
        plt.xlabel("x")
        plt.ylabel("Absolute Error")
        plt.title("Reciprocal Approximation Error (Q8.8)")
        plt.legend()
        plt.grid(True)
        plt.show()

        print("FISR 最大誤差:", max(errors_fisr))
        print("FISR 平均誤差:", sum(errors_fisr)/len(errors_fisr))
        print("CORDIC 最大誤差:", max(errors_cordic))
        print("CORDIC 平均誤差:", sum(errors_cordic)/len(errors_cordic))

    if mode == 'reciprocal_normal':
        # x = np.linspace(-0.88, 0.88, 200)
        # x = np.linspace(0.24, 0.26, 100)
        x = np.linspace(0.01, 1, 100)
        # x = np.linspace(0.5, 2, 800)
        div_list = []
        errors_div = []

        for t in x:
            print(f'\nCalculating div({t}):')
            inv_sum_q8  = fisr_q8((float_to_q8(t)**2) >> 8, iterations=3)

            div_list.append(inv_sum_q8)
            # true_sin.append(math.cos(t))
            errors_div.append(abs(q8_to_float(inv_sum_q8) - 1/t))
            print(f't: {t}, inv_sum_q8: {q8_to_float(inv_sum_q8)}, div_true: {1/t}, error: {errors_div[-1]}')
        
        plt.plot(x, errors_div, label="div error")
        plt.xlabel("theta (rad)")
        plt.ylabel("Absolute Error")
        plt.title("div error")
        plt.legend()
        plt.show()

        print("div 最大誤差:", max(errors_div))
        print("div 平均誤差:", sum(errors_div)/len(errors_div))

    if mode == 'reciprocal_Q88':
        # x = np.linspace(-0.88, 0.88, 200)
        # x = np.linspace(0.24, 0.26, 100)
        x = np.linspace(0.01, 1, 100)
        # x = np.linspace(0.5, 2, 800)
        cordic_div_list = []
        errors_div = []

        for t in x:
            print(f'\nCalculating div({t}):')
            x_cordic, y_remain, div_cordic, convergence_list = cordic_q8_reciprocal(float_to_q8(t), float_to_q8(1), 0, iterations=8)

            cordic_div_list.append(div_cordic)
            # true_sin.append(math.cos(t))
            errors_div.append(abs(q8_to_float(div_cordic) - 1/t))
            print(f't: {t}, div_cordic: {q8_to_float(div_cordic)}, div_true: {1/t}, error: {errors_div[-1]}')
        
        plt.plot(x, errors_div, label="div error")
        plt.xlabel("theta (rad)")
        plt.ylabel("Absolute Error")
        plt.title("div error")
        plt.legend()
        plt.show()

        print("div 最大誤差:", max(errors_div))
        print("div 平均誤差:", sum(errors_div)/len(errors_div))

    if mode == 'reciprocal_flaot':
        # convergence range: |theta| < atanh(1) ≈ 0.881
        # x = np.linspace(-0.88, 0.88, 200)
        x = np.linspace(0, 2, 800)
        # x = np.linspace(-50, 10, 800)
        cordic_div_list = []
        errors_div = []

        for t in x:
            print(f'\nCalculating div({t}):')
            x_cordic, y_remain, div_cordic, convergence_list = cordic_reciprocal(t, 1, 0, iterations=32)

            cordic_div_list.append(div_cordic)
            # true_sin.append(math.cos(t))
            errors_div.append(abs(div_cordic - 1/t))
            print(f't: {t}, div_cordic: {div_cordic}, div_true: {1/t}, error: {errors_div[-1]}')
        
        plt.plot(x, errors_div, label="div error")
        plt.xlabel("theta (rad)")
        plt.ylabel("Absolute Error")
        plt.title("div error")
        plt.legend()
        plt.show()

        print("div 最大誤差:", max(errors_div))
        print("div 平均誤差:", sum(errors_div)/len(errors_div))

    if mode == 'cos_sin_float':
        # convergence range: -π/2 ~ π/2
        thetas = np.linspace(-math.pi/2, math.pi/2, 200)
        cordic_sin = []
        # true_sin   = []
        errors_sin = []

        cordic_cos = []
        # true_cos   = []
        errors_cos = []

        for t in thetas:
            cos_cordic, sin_cordic, theta_remain, convergence_list = cordic(1, 0, t, m=1, iterations=32, mode='rotation')

            cordic_sin.append(sin_cordic)
            # true_sin.append(math.sin(t))
            errors_sin.append(abs(sin_cordic - math.sin(t)))

            cordic_cos.append(cos_cordic)
            # true_sin.append(math.cos(t))
            errors_cos.append(abs(cos_cordic - math.cos(t)))
        
        plt.plot(thetas, errors_sin, label="sin error")
        plt.plot(thetas, errors_cos, label="cos error")
        plt.xlabel("theta (rad)")
        plt.ylabel("Absolute Error")
        plt.title("CORDIC sin & cos error")
        plt.legend()
        plt.show()

        print("sin 最大誤差:", max(errors_sin))
        print("sin 平均誤差:", sum(errors_sin)/len(errors_sin))
        print("cos 最大誤差:", max(errors_cos))
        print("cos 平均誤差:", sum(errors_cos)/len(errors_cos))

if __name__ == "__main__":
    print("=== Cordic testbench 2025.10.13 ===")

    # analyze_cordic(mode = 'cos_sin_float')
    # analyze_cordic(mode = 'exp_float')
    # analyze_cordic(mode = 'exp_Q88')
    # analyze_cordic(mode = 'exp_normal')
    analyze_cordic(mode = 'exp_compare')

    # analyze_cordic(mode = 'reciprocal_flaot')
    # analyze_cordic(mode = 'reciprocal_Q88')
    # analyze_cordic(mode = 'reciprocal_normal')
    analyze_cordic(mode = 'reciprocal_compare')



    # q8x = -278
    # x = q8_to_float(q8x)
    # x = -0.5081351689612017
    # print(f'x: {x}')
    # # coshz + sinhz = exp(z)
    # # frac_part, int_part = np.modf(x)
    # # e_int = 2**(int_part * 1.5)  # log2(e) ≈ 1.5

    # # exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic_q8(float_to_q8(e_int), float_to_q8(e_int), float_to_q8(frac_part), m=-1, iterations=8, mode='rotation')
    # exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic_q8_exp(float_to_q8(x), log2e=1.5, iterations=5)
    # print(f"exp1(z): {exp1_cordic}, exp2(z): {exp1_cordic}, theta: {theta_remain}")
    
    # exp_true = math.exp(x)
    # print(f"Floating True exp({x}): {exp_true}, CORDIC exp({x}): {q8_to_float(exp1_cordic)}")
    # print(f"Q8       True exp({x}): {float_to_q8(exp_true)}, CORDIC exp({x}): {exp1_cordic}")

    # exp_error = abs(q8_to_float(exp1_cordic) - exp_true)
    # print(f"Error: exp(z) error={exp_error}")

    # print(f'Cordic_q8 exp({x}) = {exp1_cordic}')

    # ## ---- Circular rotation mode ---- ##
    # print(f"\n\n<Case1> Circular rotation mode")
    # # cos sin
    # cos_cordic, sin_cordic, theta_remain, convergence_list = cordic(1, 0, math.pi/4, m=1, iterations=32, mode='rotation')
    # print(f"cos: {cos_cordic}, sin: {sin_cordic}, theta: {theta_remain}")

    # # True values
    # cos_true = math.cos(math.pi/4)
    # sin_true = math.sin(math.pi/4)


    # # Compute errors
    # cos_error = abs(cos_cordic - cos_true)
    # sin_error = abs(sin_cordic - sin_true)

    # print(f"Error:  cos error={cos_error}, sin error={sin_error}")


    # ## ---- Circular vectoring mode ---- ##
    # print(f"\n\n<Case2> Circular vectoring mode")
    # # sqrt(x^2 + y^2), tan-1(y)
    # square_cordic, y_remain, arctan_cordic, convergence_list = cordic(1, 1, 0, m=1, iterations=32, mode='vectoring')
    # print(f"square: {square_cordic}, y: {y_remain}, arctan: {arctan_cordic}")

    # square_true = math.sqrt(2)  # sqrt(1^2 + 1^2)
    # arctan_true = math.atan(1)      # 0.785... ≈ π/4

    # square_error = abs(square_cordic - square_true)
    # arctan_error = abs(arctan_cordic - arctan_true)
    # print(f"Error:  square error={square_error}, arctan error={arctan_error}")


    # ## ---- Linear rotation mode ---- ##
    # print(f"\n\n<Case3> Linear rotation mode")
    # y + xz
    # x = 89.44271910190582
    # y = 0
    # z = -0.005

    # x_cordic, acc_cordic, theta_remain, convergence_list = cordic_mac(x, y, z, iterations=32)
    # print(f"x: {x_cordic}, acc: {acc_cordic}, theta: {theta_remain}")

    
    # acc_true = y + x * z

    # acc_error = abs(acc_cordic - acc_true)
    # print(f"Error:  acc error={acc_error}")


    ## ---- Linear vectoring mode ---- ##
    print(f"\n\n<Case4> Linear vectoring mode")
    # z + y/x
    # x = 0.01118033988985804
    # x = 0.151980198019802
    x = q8_to_float(378)
    y = 1
    z = 0

    # x_cordic, y_remain, div_cordic, convergence_list = cordic_reciprocal(x, y, z, iterations=32)

    # print(f"x: {x_cordic}, y: {y_remain}, div: {div_cordic}")

    # div_true = z + y / x

    # div_error = abs(div_cordic - div_true)
    # print(f"Error:  div error={div_error}, true div={div_true}, cordic div={div_cordic}")


    print(f'1: {float_to_q8(1)}')
    print(f'x({x}) = {float_to_q8(x)}')
    x_cordic, y_remain, div_cordic, convergence_list = cordic_q8_reciprocal(float_to_q8(x), float_to_q8(y), z, iterations=8)

    print(f"x: {x_cordic}, y: {y_remain}, div: {div_cordic}")

    div_true = z + y / x

    div_error = abs(q8_to_float(div_cordic) - div_true)
    print(f"Error:  div error={div_error}, true div={div_true}, cordic div={q8_to_float(div_cordic)}")


    # ## ---- Hyperbolic rotation mode ---- ##
    # print(f"\n\n<Case5> Hyperbolic rotation mode (EXP)")
    # # coshz + sinhz = exp(z)
    # # x = 0.5
    # # exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic(1, 1, x, m=-1, iterations=32, mode='rotation')
    # # print(f"exp1(z): {exp1_cordic}, exp2(z): {exp1_cordic}, theta: {theta_remain}")
    # # exp_true = math.exp(x)

    # # exp_error = abs(exp1_cordic - exp_true)
    # # print(f"Error: exp(z) error={exp_error}")

    # value = -20.55
    # # value = 9
    # frac_part, int_part = np.modf(value)
    # print(f'int_part: {int_part}, frac_part: {frac_part}')
    # # e_int = 2**(int_part * 1.4426950408889634)  # log2(e) ≈ 1.4426950408889634
    # print(int_part * 1.5)
    # e_int = 2**(int_part * 1.5)  # log2(e) ≈ 1.5
    # print(f'e_int: {e_int}')

    # exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic(e_int, e_int, frac_part, m=-1, iterations=32, mode='rotation')
    # print(f"exp1(z): {exp1_cordic}, exp2(z): {exp2_cordic}, theta: {theta_remain}")

    # exp_true = math.exp(value)
    # error = abs(exp1_cordic - exp_true)

    # print(f"CORDIC exp({value}) = {exp1_cordic}, {float_to_q8(exp1_cordic)} in Q8")
    # print(f"True exp({value})   = {exp_true}, {float_to_q8(exp_true)} in Q8")
    # print(f"Error           = {error}")
    

    # ## ---- Hyperbolic vectoring mode ---- ##
    # # NOTE: y/x must in range [-1, 1]
    # print(f"\n\n<Case6> Hyperbolic vectoring mode")
    # # sqrt(x^2 - y^2), tanh-1(y)
    # square_cordic, y_remain, tanh_cordic, convergence_list = cordic(4, 2, 0, m=-1, iterations=32, mode='vectoring')
    # print(f"square: {square_cordic}, y: {y_remain}, tanh: {tanh_cordic}")

    # square_true = math.sqrt(4**2 - 2**2)  # sqrt(2^2 - 4^2)
    # tanh_true = math.atanh(2/4) # 0.549... ≈ tanh^-1(0.5)

    # square_error = abs(square_cordic - square_true)
    # tanh_error = abs(tanh_cordic - tanh_true)
    # print(f"Error:  square error={square_error}, tanh error={tanh_error}")


    # sqrt(x^2 - y^2),   sqrt(x) = sqrt((x+1/4)^2 - (x-1/4)^2)
    
    # x = 3.0773697200180585

    # print(f"y/x: {(x-1/4)/(x+1/4)}")

    # square_cordic, y_remain, tanh_cordic, convergence_list = cordic_sqrt(x, threshold=4, scale=1024, iterations=32)

    # print(f"square: {square_cordic}, y: {y_remain}, tanh: {tanh_cordic}")

    # square_true = math.sqrt(x)

    # square_error = abs(square_cordic - square_true)
    # print(f"True: {square_true},  Error: square error={square_error}")


    # Plot convergence
    # plt.plot(range(len(convergence_list)), [abs(t) for t in convergence_list], marker='o')
    # plt.yscale('log')
    # plt.xlabel('Iteration')
    # plt.ylabel('Residual θ (rad, log scale)')
    # plt.title('CORDIC Convergence of θ (Target = π/4)')
    # plt.grid(True)
    # plt.show()