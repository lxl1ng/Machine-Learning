import random
import numpy as np
import matplotlib.pyplot as plt

k = 3  # 要分的簇数
rnd = 0  # 轮次，用于控制迭代次数
ROUND_LIMIT = 17  # 轮次的上限
THRESHOLD = 1e-10  # 单轮改变距离的阈值，若改变幅度小于该阈值，算法终止
points = []  # 点的列表
clusters = []  # 簇的列表，clusters[i]表示第i簇包含的点

f = open('data_to2.txt', 'r')
for line in f:
    # 把字符串转化为numpy中的float64类型
    points.append(np.array(line.split(' '), dtype=np.string_).astype(np.float64))
# random的sample函数从列表中随机挑选出k个样本（不重复）。我们在这里把这些样本作为均值向量
mean_vectors = random.sample(points, k)
while True:
    rnd += 1  # 轮次增加
    change = 0  # 把改变幅度重置为0

    # 清空对簇的划分
    clusters = []
    for i in range(k):
        clusters.append([])
    for point in points:
        # argmin 函数找出容器中最小的下标，在这里这个目标容器是
        # list(map(lambda vec: np.linalg.norm(melon - vec, ord = 2), mean_vectors)),
        # 它表示point与mean_vectors中所有向量的距离列表。
        # (numpy.linalg.norm计算向量的范数,ord = 2即欧几里得范数，或模长)
        c = np.argmin(
            list(map(lambda vec: np.linalg.norm(point - vec, ord=2), mean_vectors))
        )

        clusters[c].append(point)

    for i in range(k):
        # 求每个簇的新均值向量
        new_vector = np.zeros((1, 2))
        for point in clusters[i]:
            new_vector += point
        new_vector /= len(clusters[i])

        # 累加改变幅度并更新均值向量
        change += np.linalg.norm(mean_vectors[i] - new_vector, ord=2)
        mean_vectors[i] = new_vector

    # 若超过设定的轮次或者变化幅度<预先设定的阈值，结束算法
    if rnd > ROUND_LIMIT or change < THRESHOLD:
        break

print('最终迭代%d轮' % rnd)

# 绘制中心点
c_X = []
c_Y = []
for c_point in mean_vectors:
    c_X.append(c_point[0][0])
    c_Y.append(c_point[0][1])

plt.scatter(c_X, c_Y, c='r', s=20)

colors = ['green', 'blue', 'purple']
# 每个簇换一下颜色，同时迭代簇和颜色两个列表
for i, col in zip(range(k), colors):
    for point in clusters[i]:
        # 绘制散点图
        plt.scatter(point[0], point[1], color=col, s=1)

plt.show()
