import math
import numpy as np
import matplotlib.pyplot as plt


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

def cordic_reciprocal(x, y, z, iterations=32):
    # z + y/x
    if x < 2e-2:
        print("x is too small, scale it up")
        scale_x = x * 256
    else:
        scale_x = x

    # Perform CORDIC iterations
    x_cordic, y_remain, div_cordic, convergence_list = cordic(scale_x, y, z, m=0, iterations=iterations, mode='vectoring')

    # Compensate the gain
    if x < 2e-2:
        div_cordic = div_cordic * 256

    return x_cordic, y_remain, div_cordic, convergence_list

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

if __name__ == "__main__":
    print("=== Cordic testbench ===")

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
    x = 89.44271910190582
    y = 0
    z = -0.005

    x_cordic, acc_cordic, theta_remain, convergence_list = cordic_mac(x, y, z, iterations=32)
    print(f"x: {x_cordic}, acc: {acc_cordic}, theta: {theta_remain}")

    
    acc_true = y + x * z

    acc_error = abs(acc_cordic - acc_true)
    print(f"Error:  acc error={acc_error}")


    ## ---- Linear vectoring mode ---- ##
    # print(f"\n\n<Case4> Linear vectoring mode")
    # z + y/x
    # x = 0.01118033988985804
    # y = 1
    # z = 0

    # x_cordic, y_remain, div_cordic, convergence_list = cordic_reciprocal(x, y, z, iterations=32)

    # print(f"x: {x_cordic}, y: {y_remain}, div: {div_cordic}")

    # div_true = z + y / x

    # div_error = abs(div_cordic - div_true)
    # print(f"Error:  div error={div_error}")


    # ## ---- Hyperbolic rotation mode ---- ##
    # print(f"\n\n<Case5> Hyperbolic rotation mode")
    # # coshz + sinhz = exp(z)
    # exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic(1, 1, 0.4, m=-1, iterations=32, mode='rotation')
    # print(f"exp1(z): {exp1_cordic}, exp2(z): {exp1_cordic}, theta: {theta_remain}")
    # exp_true = math.exp(0.4)

    # exp_error = abs(exp1_cordic - exp_true)
    # print(f"Error: exp(z) error={exp_error}")

    # value = -1.2
    # frac_part, int_part = np.modf(value)
    # e_int = 2**(int_part * 1.4426950408889634)  # log2(e) ≈ 1.4426950408889634

    # exp1_cordic, exp2_cordic, theta_remain, convergence_list = cordic(e_int, e_int, frac_part, m=-1, iterations=32, mode='rotation')
    # print(f"exp1(z): {exp1_cordic}, exp2(z): {exp2_cordic}, theta: {theta_remain}")

    # exp_true = math.exp(value)
    # error = abs(exp1_cordic - exp_true)

    # print(f"CORDIC exp(-70) = {exp1_cordic}")
    # print(f"True exp(-70)   = {exp_true}")
    # print(f"Error           = {error}")
    

    # ## ---- Hyperbolic vectoring mode ---- ##
    # NOTE: y/x must in range [-1, 1]
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
    plt.plot(range(len(convergence_list)), [abs(t) for t in convergence_list], marker='o')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Residual θ (rad, log scale)')
    plt.title('CORDIC Convergence of θ (Target = π/4)')
    plt.grid(True)
    plt.show()

