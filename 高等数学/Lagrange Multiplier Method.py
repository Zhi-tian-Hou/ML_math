import numpy as np
import matplotlib.pyplot as plt
from sympy import *

c = 1  # 惩罚参数，也就是半径
delta = 0.025  # 网格间距
xbias = 0.5
ybias = 1.5


def cost_function(ax):
    # ------------目标函数---------------
    x = np.arange(-1.0, 2.0, delta)
    y = np.arange(-1.0, 2.0, delta)
    x, y = np.meshgrid(x, y)
    Z1 = np.exp(-(x-xbias)**2 - (y-ybias)**2)
    Z = -Z1 * 2
    CS = ax.contour(x, y, Z)
    plt.scatter(xbias, ybias, color='r')
    ax.clabel(CS, inline=1, fontsize=10)


def l1_normal(ax):
    x = np.linspace(-1.0, 1.0, 1000)
    ax.plot(x, abs(x)-1, color='black')
    ax.plot(x, -abs(x)+1, color='black')


def l2_normal(ax):
    # --------------圆----------------
    theta = np.arange(0, 2*np.pi, delta)
    w1 = 0 + c * np.cos(theta)
    w2 = 0 + c * np.sin(theta)
    ax.plot(w1, w2, color='black')


def solve_formula(ax, l):
    """
    解方程
    :param l: 使用的范数
    :return:
    """
    W1 = Symbol("w1")
    W2 = Symbol("w2")
    L = Symbol("l")
    ans = None
    if l == 1:
        l1_normal(ax)
        a = -2 * exp(-(W1 - xbias) ** 2 - (W2 - ybias) ** 2) + L * (W1 + W2 - c)
        ans = solve([diff(a, W1), diff(a, W2), W1+W2-c], [W1, W2, L])
    else:
        l2_normal(ax)
        a = -2 * exp(-(W1 - xbias) ** 2 - (W2 - ybias) ** 2) + L * (W1 ** 2 + W2 ** 2 - c ** 2)
        ans = solve([diff(a, W1), diff(a, W2), W1 ** 2 + W2 ** 2 - c ** 2], [W1, W2, L])
    print("formula result is:", ans)
    for i in ans:
        item = [float(j) for j in i]
        plt.scatter(item[0], item[1], color='black')


if __name__ == '__main__':
    fig, ax = plt.subplots()
    cost_function(ax)
    solve_formula(ax, 2)

    plt.axis('scaled')
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.grid(True)
    plt.show()
