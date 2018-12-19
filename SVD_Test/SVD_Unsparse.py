import numpy as np
from surprise import SVD
from surprise import Dataset, Reader
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# 导入训练集，去掉全零项
origin_data = np.load("top21.npy")
nonzero_row_indice, nonzero_col_indice = origin_data.nonzero()
unique_nonzero_indice = np.unique(nonzero_col_indice)
unsparse_data = origin_data[:, unique_nonzero_indice]
np.save('data_unsparse.npy', unsparse_data)
print('File Saved!')
#test_data = origin_data[:, unique_nonzero_indice]
data = np.load("top21.npy")

[user, item] = data.shape
for i in range(user):
    for j in range(item):
        if data[i, j] == 1:
            data[i, j] = 1
        elif data[i, j] == 0:
            data[i, j] = -1

test_data = data.copy()
train_data = data.copy()


print('新数据集中共有21名用户，{}条动态信息。'.format(train_data.shape[1]))

# 训练集
test_index = np.arange(1700, 1767)
train_data_matrix = train_data
for index in test_index:
    train_data_matrix[20, index] = 0
print(train_data_matrix[20, 1757])

# 测试集
test_data_matrix = test_data
print(test_data_matrix[20, 1757])

print('finished!')


# 计算均方误差
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()

    return sqrt(mean_squared_error(prediction, ground_truth))


# ReLU函数
def relu(x_pred):
    for xi in range(x_pred.shape[0]):
        for xj in range(x_pred.shape[1]):
            if x_pred[xi,xj] <= 0:
                x_pred[xi, xj] = 0
    return x_pred

# 画图
loss = []
for k in range(1, 20):
    u, s, vt = svds(train_data_matrix, k)
    s_diag_matrix = np.diag(s)
    x_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    #x_result = relu(x_pred)
    print('User-based CF MSE: ' + str(rmse(x_pred[20, 1700:1766], test_data_matrix[20, 1700:1766])))
    loss.append(rmse(x_pred[20, 1700:1766], test_data_matrix[20, 1700:1766]))

k = np.arange(1, 20)

plt.figure()
plt.plot(k, loss)
plt.show()
