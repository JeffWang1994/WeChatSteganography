import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# 导入JSON文件
f = open('sns_hong.json','r', encoding='UTF-8')
origin_data = json.load(f)

#print(type(origin_data))
print('==========================')
print('数据集示例:')
print(origin_data[1])
#print(type(origin_data[1]))
#print(origin_data[1]['authorName'])
print("数据集中共有{}条动态信息。".format(len(origin_data)))
print('==========================\n')

# 获取用户列表
userNameList = []
for item in origin_data:
    if not (item['authorId'] in userNameList):
        userNameList.append(item['authorId'])
    for likes_user in item['likes']:
        if not (likes_user['userId'] in userNameList):
            userNameList.append(likes_user['userId'])
#    for comments_user in item['comments']:
#        if not (comments_user['authorName'] in userNameList):
#            userNameList.append(comments_user['authorName'])

print('==========================')
print('用户列表:')
print(userNameList)
print('数据集中共有:{}名用户'.format(len(userNameList)))
print('==========================')

# 建立以item为主索引的item-user列表
itemList = []
SUMM = 0
for item in origin_data:
    itemDist = {}
    itemDist['snsId'] = item['snsId']
    itemDist['authorId'] = item['authorId']
    itemDist['likesSum'] = len(item['likes'])
    summ = 0
    for user in userNameList:
        itemDist[user] = 0
        for likes_user in item['likes']:
            if user == likes_user['userId']:
                itemDist[user] = 1
                summ = summ + 1
            else:
                itemDist[user] = 0
    SUMM = SUMM + summ
    itemList.append(itemDist)

itemDataFrame = pd.DataFrame(itemList)

print("==========================")
print("数据集中共有: {}条动态信息。".format(len(itemList)))
print("动态列表示例:")
print(itemList[1])
LikesSUM = 0
NonLikes = 0
for item in itemList:
    LikesSUM = LikesSUM+item['likesSum']
    #print(item['likesSum'])
    if item['likesSum'] == 0:
        NonLikes = NonLikes+1
print("总共有{}个点赞记录。".format(LikesSUM))
print(SUMM)
print("在{}条动态信息中，有{}条动态没有任何点赞信息".format(len(itemList), NonLikes))
print("==========================")

# 建立以user为主索引的user-item矩阵
user_item = np.zeros((len(userNameList), len(itemList)))

user_index = 0
item_index = 0
for user in userNameList:
    for item in origin_data:
        for item_like in item['likes']:
            if user == item_like['userId']:
                user_item[user_index, item_index] = 1
        item_index = item_index + 1
    user_index = user_index + 1
    item_index = 0

print(user_item.shape)

# 检测user-item矩阵中是否包含了所有的点赞记录
SUMM = 0
for i in range(len(userNameList)):
    for j in range(len(itemList)):
        if user_item[i,j] != 0:
            SUMM = SUMM + 1

print(SUMM)

single_max = []
all_max = []
temp = {}

def takeSecond(elem):
    return elem[1]

for item in itemList:
    Likes_max = (item['likesSum'])  #单条动态获赞数
    Author_max = (item['authorId'])
    if Author_max in temp:
        temp[Author_max] += Likes_max
    else:
        temp[Author_max] = Likes_max
    if Author_max != ('wxid_vyebwqsb5wo21'):
        single_max.append((Author_max,Likes_max)) #单条获赞最多

single_max.sort(key=takeSecond, reverse=True)
print ('单条动态获赞前5（除用户自身外）：')
print (single_max[0:5])

f = zip(temp.values(),temp.keys())
all_max = sorted(f,reverse = True)
print ('获赞总数前20：')
print (all_max[0:21])

#建立互动矩阵 计算每位用户为其他用户所有动态的点赞总数
user_interaction = np.zeros((len(userNameList), len(userNameList)))
user_index = 0
users_index = 0
for item in origin_data:
    for user in userNameList:
        if user == item['authorId']:
            for users in userNameList:
                for item_like in item['likes']:
                    if users == item_like['userId']:
                        user_interaction[user_index, users_index] += 1
                users_index +=1
        user_index +=1
        users_index =0
    user_index =0

my_num = userNameList.index('wxid_vyebwqsb5wo21') #获取我在用户列表中的索引位置
my_interaction = user_interaction[my_num]         #我为其余用户的点赞总数
num = range(0,(len(userNameList))-1)              #添加位置索引
f1 = zip(my_interaction,userNameList,num)
my_top = sorted(f1,reverse = True)
my_top20 = my_top[0:20]                           #被我点赞最多的前20名用户
print ('互动最高（点赞次数最多）：')
print (my_top20)

# 建立以top20为主索引的top20-item矩阵
top20_item = np.zeros((len(my_top20)+1, len(itemList)))
num_index = 0
for top_tuple in my_top20:
    top_index = top_tuple[2]
    top20_item[num_index]=user_item[top_index]
    num_index +=1
top20_item[num_index]=user_item[my_num]           #矩阵最后加上我的点赞记录
print ('top(20+1)*item矩阵')
print (top20_item)

#建立co-occurrence矩阵
co_occurrence = np.zeros((len(userNameList),len(userNameList)))
for item in origin_data:
    for likes_user in item['likes']:
        user_index = userNameList.index(likes_user['userId'])
        for likes_user_others in item['likes']:
            if likes_user_others != likes_user:
                user_others_index = userNameList.index(likes_user_others['userId'])
                co_occurrence[user_index,user_others_index] +=1

print (co_occurrence)

#找出与我点赞行为最相似的用户
my_co_occurrence = co_occurrence[my_num]
f2 = zip(my_co_occurrence,userNameList,num)
my_co_top = sorted(f2,reverse = True)
my_co_top20 = my_co_top[0:20]                           #跟我共同点赞最多的前20名用户
print ('同时点赞次数最多：')
print (my_co_top20)

#删除无点赞信息或只有我点赞的item 构建user_item_arranged矩阵
item_user = np.transpose(user_item)
item_arranged = []
for i in range(0,len(item_user)):
    if sum(item_user[i]) !=0 :      #有点赞记录的item
        if sum(item_user[i]) ==1:
            if item_user[i,my_num] != 1 :  #去除仅被我点赞的item
                item_arranged.append(item_user[i])
        else:
            item_arranged.append(item_user[i])
user_item_arranged = np.transpose(item_arranged)

# 建立以co_top20为主索引的co_top20_item矩阵
co_top20_item = np.zeros((len(my_co_top20)+1, len(item_arranged)))
num_index = 0
for top_tuple in my_co_top20:
    top_index = top_tuple[2]
    co_top20_item[num_index]=user_item_arranged[top_index]
    num_index +=1
co_top20_item[num_index]=user_item_arranged[my_num]           #矩阵最后加上我的点赞记录
print ('co_top(20+1)*item矩阵')
print (co_top20_item)

d = np.transpose(co_top20_item)
#np.random.shuffle(d) #随机乱序
n, m = d.shape
test_num = round(1 * n / 3) #取数据集前1/3为测试集
train_num = n - test_num  #后2/3训练集
train_data = d[0:train_num,0: (m-1)]
train_data = np.c_[train_data, np.ones((train_num,1))] #回归的时候会有常数项，故此处加了一列
train_label = d[0:train_num,m-1].reshape(train_num,1) #python中一维数组默认是行向量，需要reshape函数转换
test_data = d[train_num:n,0: (m-1)]
test_data = np.c_[test_data, np.ones((test_num, 1))]
test_label = d[train_num:n,m-1].reshape(test_num,1)

print ("\n linear regression")
print ("\t training start ...")
threshold = (max(train_label) + min(train_label)) / 2
gamma, eps, max_iter = 0.001, 0.00001, 10000
w = linear_regression(train_data, train_label, 'gd', gamma, eps, max_iter)
print ("\t training done !")
train_y_predict = train_data.dot(w)
test_y_predict = test_data.dot(w)
prediction = autoNorm(test_y_predict)
print ("\t train predict error\t: %f"%(sum( abs( ((train_y_predict > threshold) + 0) - ((train_label > threshold) + 0) ))[0] / (train_num + 0)))
print ("\t test predict error \t: %f"%(sum( abs( ((test_y_predict > threshold) + 0) - ((test_label > threshold) + 0) ))[0] / (test_num + 0)))
prediction_arranged = np.zeros((len(test_y_predict), 1))
length = len(test_y_predict)
j = range(0,length-1)
for i in j:
    if test_y_predict[i] > 0.2:
        prediction_arranged[i] = 1

accuracy = accuracy_score(test_label,prediction_arranged)
precision = precision_score(test_label,prediction_arranged)
recall = recall_score(test_label,prediction_arranged)
fpr,tpr,thresholds = roc_curve(test_label,prediction_arranged)
roc_auc = roc_auc_score(test_label,prediction_arranged)

plt.plot(fpr,tpr,linewidth=2,label="ROC")
plt.xlabel("false presitive rate")
plt.ylabel("true presitive rate")
plt.ylim(0,1.05)
plt.xlim(0,1.05)
plt.legend(loc=4)#图例的位置
plt.show()  #ROC曲线

print('准确率:{}'.format(accuracy))
print('精确率:{}'.format(precision))
print('召回率:{}'.format(recall))
print('召回率:{}'.format(recall))

'''
print ("\nlog regression")
print ("\t training start ...")
min_label, max_label = min(train_label), max(train_label)
train_label = train_label - min_label + 1 #保证label>0，才可以取对数
test_label = test_label - min_label + 1 #保证label>0，才可以取对数
threshold = (np.log(max(train_label)) + np.log(min(train_label))) / 2
gamma, eps, max_iter = 0.001, 0.00001, 10000
w = log_regression(train_data, train_label, 'gd', gamma, eps, max_iter)
train_y_predict = train_data.dot(w)
test_y_predict = test_data.dot(w)
print ("\t training done")
print ("\t train predict error\t: %f"%(sum( abs( ((train_y_predict > threshold) + 0) - ((train_label > threshold) + 0) ))[0] / (train_num + 0.0)))
print ("\t test predict error \t: %f"%(sum( abs( ((test_y_predict > threshold) + 0) - ((test_label > threshold) + 0) ))[0] / (test_num + 0.0)))

print ("\nlogistic regression")
print ("\t training start ...")
min_label, max_label = min(train_label), max(train_label)
train_label = (train_label - min_label) / (max_label - min_label) #将label变为0，1
test_label = (test_label - min_label) / (max_label - min_label) #将label变为0，1
threshold = 0.5
gamma, eps, max_iter = 0.001, 0.00001, 10000
w = logistic_regression(train_data, train_label, 'gd', gamma, eps, max_iter)
print ("\t training done")
train_y_predict = sigmoid(train_data.dot(w))
test_y_predict = sigmoid(test_data.dot(w))
print ("\t train predict error \t: %f"%(sum( abs( ((train_y_predict > threshold) + 0) - ((train_label > threshold) + 0) ))[0] / (train_num + 0.0)))
print ("\t test predict error \t: %f"%(sum( abs( ((test_y_predict > threshold) + 0) - ((test_label > threshold) + 0) ))[0] / (test_num + 0.0)))

'''