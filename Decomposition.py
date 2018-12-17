import json
import numpy as np


# 导入数据集
def load_file(path):
    data = np.load(path)
    print('File loaded!')

    return data


# 梯度下降
def gradAscent(data, K):
    dataMat = np.mat(data)
    print(dataMat)
    m, n = np.shape(dataMat)
    p = np.mat(np.random.random((m, K)))
    q = np.mat(np.random.random((K, n)))

    alpha = 0.0002
    beta = 0.02
    maxCycles = 100000

    for step in range(maxCycles):
        for i in range(m):
            for j in range(n):
                if dataMat[i, j] > 0:
                    # print dataMat[i,j]
                    error = dataMat[i, j]
                    for k in range(K):
                        error = error - p[i, k] * q[k, j]
                    for k in range(K):
                        p[i, k] = p[i, k] + alpha * (2 * error * q[k, j] - beta * p[i, k])
                        q[k, j] = q[k, j] + alpha * (2 * error * p[i, k] - beta * q[k, j])

        loss = 0.0
        for i in range(m):
            for j in range(n):
                if dataMat[i, j] > 0:
                    error = 0.0
                    for k in range(K):
                        error = error + p[i, k] * q[k, j]
                    loss = (dataMat[i, j] - error) * (dataMat[i, j] - error)
                    for k in range(K):
                        loss = loss + beta * (p[i, k] * p[i, k] + q[k, j] * q[k, j]) / 2

        if loss < 0.001:
            break
        # print step
        if step % 1000 == 0:
            print(loss)

    return p, q


if __name__ == '__main__':
    data = load_file('user_item.npy')
    p, q = gradAscent(data, 5)

    result = p * q
    # print p
    # print q

    print(result)
    print('program finished!')
