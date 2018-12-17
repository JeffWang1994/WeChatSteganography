import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.sparse as sp
from scipy.sparse.linalg import svds

'''
# 整理数据集_无差分配
data = []
origin_data = np.load("top21.npy")
for i in range(origin_data.shape[0]):
    for j in range(origin_data.shape[1]):
        data.append([i, j, origin_data[i, j]])

n_users = 21
n_items = 1767
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

train_data, test_data = train_test_split(data, test_size=0.25)

train_data_matrix = np.zeros(origin_data.shape)
for line in train_data:
    train_data_matrix[line[0], line[1]] = line[2]

test_data_matrix = np.zeros(origin_data.shape)
for line in test_data:
    test_data_matrix[line[0], line[1]] = line[2]
'''

# 人为构建训练集和测试集
data = []
origin_data = np.load("top21.npy")
for i in range(origin_data.shape[0]):
    for j in range(origin_data.shape[1]):
        data.append([i, j, origin_data[i, j]])

test_index = np.random.randint(0, 1766, 700,dtype=int)

train_data_matrix = origin_data
for index in test_index:
    train_data_matrix[20, index] = 0

test_data_matrix = origin_data


# 计算均方误差
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()

    return sqrt(mean_squared_error(prediction, ground_truth))

'''
user_similarity = pairwise_distances(train_data_matrix, metric="cosine")
item_similarity = pairwise_distances(train_data_matrix.T, metric="cosine")

def predict(rating, similarity, type='user'):
    if type == 'user':
        mean_user_rating = rating.mean(axis = 1)
        rating_diff = (rating - mean_user_rating[:,np.newaxis])
        pred = mean_user_rating[:,np.newaxis] + similarity.dot(rating_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

print('User based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item based CF RMSe: ' + str(rmse(item_prediction, test_data_matrix)))
print('user based CF result of Hong:\n')
print(user_prediction[8, :])
print('item based CF result:\n')
print(item_prediction[8, :])
print('original Hong:\n')
print(test_data_matrix[8, :])
'''

# 尝试使用SVD来进行矩阵分解。
u, s, vt = svds(train_data_matrix, k=20)
s_diag_matrix = np.diag(s)
x_pred = np.dot(np.dot(u, s_diag_matrix), vt)
x_result = x_pred
#x_result[x_result < 0.5] = 0
#x_result[x_result > 0.5] = 1
print('User-based CF MSE: ' + str(rmse(x_result, test_data_matrix)))
print(x_result[20, :])
print(test_data_matrix[20, :])

print('finished!')



