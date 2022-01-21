import numpy as np
import matplotlib.pyplot as plt
from sympy import *

c = 1  # 惩罚参数，也就是半径
delta = 0.025  # 网格间距

# ------------目标函数---------------
x = np.arange(-1.0, 2.0, delta)
y = np.arange(-1.0, 2.0, delta)
x, y = np.meshgrid(x, y)
bias = 1
Z1 = np.exp(-(x-bias)**2 - (y-bias)**2)
Z = -Z1 * 2
fig, ax = plt.subplots()
CS = ax.contour(x, y, Z)
plt.scatter(bias, bias, color='r')
ax.clabel(CS, inline=1, fontsize=10)

# --------------圆----------------
theta = np.arange(0, 2*np.pi, delta)
w1 = 0 + c * np.cos(theta)
w2 = 0 + c * np.sin(theta)
ax.plot(w1, w2, color='black')

# ------------解方程--------------
W1 = Symbol("w1")
W2 = Symbol("w2")
L = Symbol("l")
a = -2*exp(-(W1-1)**2-(W2-1)**2)+L*(W1**2+W2**2-c**2)
ans = solve([diff(a, W1), diff(a, W2), W1**2+W2**2-c**2], [W1, W2, L])
print(ans)
for i in ans:
    item = [float(j) for j in i]
    plt.scatter(item[0], item[1], color='black')

plt.axis('scaled')
plt.xlabel('w1')
plt.ylabel('w2')
plt.grid(True)
plt.show()
