import math
import numpy as np
import matplotlib.pyplot as plt

def cordic_sin_cos(theta, iterations=16):
    # 簡單範例：只示意，不一定跟你實作完全相同
    K = 0.607252935 # CORDIC scaling factor (for many iterations)
    angles = [math.atan(2**-i) for i in range(iterations)]
    
    x, y, z = K, 0.0, theta
    for i in range(iterations):
        di = 1 if z >= 0 else -1
        x = x - di * y * (2**-i)
        y = y + di * x * (2**-i)
        z = z - di * angles[i]
    return x, y  # cos(theta), sin(theta)

# 測試範圍
thetas = np.linspace(-math.pi/2, math.pi/2, 200)
cordic_sin = []
true_sin = []
errors = []

for t in thetas:
    _, s = cordic_sin_cos(t, iterations=16)
    cordic_sin.append(s)
    true_sin.append(math.sin(t))
    errors.append(abs(s - math.sin(t)))

# 畫誤差圖
plt.plot(thetas, errors)
plt.xlabel("theta (rad)")
plt.ylabel("Absolute Error")
plt.title("CORDIC sin error vs math.sin")
plt.show()

print("最大誤差:", max(errors))
print("平均誤差:", sum(errors)/len(errors))
