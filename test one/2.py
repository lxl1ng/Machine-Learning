import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv  # 矩阵求逆
from numpy import dot  # 矩阵点乘
from numpy import mat  # 二维矩阵
import time


x1 = [1, 2, 4, 5, 6, 7, 8, 9, 11, 13]
y1 = [2, 2.4, 2.6, 3.0, 3.2, 3.3, 3.6, 3.7, 4.0, 4.2]

X1 = mat(x1).reshape(10, 1)
Y1 = mat(y1).reshape(10, 1)
b = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
X2 = np.column_stack((X1, b.T))


def one():
    time_start = time.time()  # 开始计时
    size = len(x1);
    i = 0
    sum_xy = 0
    sum_y = 0
    sum_x = 0
    average_xy = 0;
    average_x = 0;
    average_y = 0;
    square_x = 0;
    while i < size:
        sum_xy += x1[i] * y1[i];
        sum_y += y1[i]
        sum_x += x1[i]
        square_x += x1[i] * x1[i]
        i += 1
    average_xy = sum_xy / size
    average_x = sum_x / size
    average_y = sum_y / size
    average_square_x = square_x / size
    k = (average_xy - average_x * average_y) / (average_square_x - average_x * average_x)
    b = average_y - (k * average_x)
    print(k, b)

    # 画样本点
    plt.figure(figsize=(8, 6))
    plt.scatter(x1, y1, color='red', label='Sample data', linewidth=2)

    # 画拟合直线
    x = np.linspace(0, 14, 10)
    y = k * x + b

    # 绘制拟合曲线
    plt.plot(x, y, color='blue', label='Fitting Curve', linewidth=2)
    plt.legend()  # 绘制图例
    plt.title("Least squares")

    plt.show()

    time_end = time.time()  # 结束计时
    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')


def two():
    time_start = time.time()  # 开始计时
    w = dot(dot(inv(dot(X2.T, X2)), X2.T), Y1)  # 最小二乘法公式
    w1 = w.tolist()
    print(w1[0], w1[1])
    # 画样本点
    plt.figure(figsize=(8, 6))
    plt.scatter(x1, y1, color='red', label='Sample data', linewidth=2)

    # 画拟合直线
    x = np.linspace(0, 14, 10)
    y = w1[0] * x + w1[1]

    # 绘制拟合曲线
    plt.plot(x, y, color='blue', label='Fitting Curve', linewidth=2)
    plt.legend()  # 绘制图例
    plt.title("Matrix algorithm")

    plt.show()

    time_end = time.time()  # 结束计时
    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')


one()
two()
