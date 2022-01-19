import matplotlib.pyplot as plt
import numpy as np


def fx(theta, time_1, time_2):
    """
    二项分布
    :param theta: 待测概率
    :param time_1: 正面朝上的次数
    :param time_2: 背面朝上的次数
    :return: 二项分布值
    """
    return (theta ** time_1) * ((1 - theta) ** time_2)


def gaussian(x, u, d):
    """
    高斯分布
    :param x:变量
    :param u:均值
    :param d:标准差
    :return :p -- 高斯分布值
    """
    d_2 = d * d * 2
    zhishu = -(np.square(x - u) / d_2)
    exp = np.exp(zhishu)
    pi = np.pi
    xishu = 1 / (np.sqrt(2 * pi) * d)
    p = xishu * exp
    return p


def get_max(x, y):
    """
    取最大值
    :param x: x的取值
    :param y: y的取值
    :return: 极值的x,y坐标
    """
    max_x_index = np.argmax(y)
    max_x = x[max_x_index]
    max_y = y[max_x_index]
    return max_x, max_y


x = np.linspace(0, 1, 200)
y1 = fx(x, 700, 300)  # fx(x, 7, 3)
g = gaussian(x, 0.5, 0.1)
y = g * y1
y_list = [y1, g, y1 * g]

# ----------------------绘图部分--------------------------------
plt.figure()
plt.subplots_adjust(left=0.08, right=0.95, wspace=0.25, hspace=0.4)
sub_index = [i for i in range(len(y_list))]
x_label = ['theta' for i in range(len(sub_index))]
y_label = ['likelihood', 'P(theta)', 'posterior']
for i in range(len(y_list)):
    plt.subplot(3, 1, i+1)
    max_x, max_y = get_max(x, y_list[i])
    plt.plot(x, y_list[i])
    plt.text(max_x, max_y, f'{(max_x, max_y)}')
    plt.vlines(max_x, 0, max_y, colors='r', linestyles='dashed')
    plt.hlines(max_y, 0, max_x, colors='r', linestyles='dashed')
    plt.xlabel(x_label[i])
    plt.ylabel(y_label[i])
plt.gcf().set_size_inches(15, 15)
plt.show()
