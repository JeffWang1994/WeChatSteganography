import json
import numpy as np
import pandas as pd

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
SUM = 0
for item in origin_data:
    itemDist = {}
    itemDist['snsId'] = item['snsId']
    itemDist['authorId'] = item['authorId']
    itemDist['likesSum'] = len(item['likes'])
    sum = 0
    for user in userNameList:
        itemDist[user] = 0
        for likes_user in item['likes']:
            if user == likes_user['userId']:
                itemDist[user] = 1
                sum = sum + 1
            else:
                itemDist[user] = 0
    SUM = SUM + sum
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
print(SUM)
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
SUM = 0
for i in range(len(userNameList)):
    for j in range(len(itemList)):
        if user_item[i,j] != 0:
            SUM = SUM + 1

print(SUM)

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

#for user in userNameList:
  #  if user = ('wxid_vyebwqsb5wo21'):
my_num = userNameList.index('wxid_vyebwqsb5wo21')
my_interaction = user_interaction[my_num]
num = range(0,(len(userNameList))-1)
f1 = zip(my_interaction,userNameList,num)
my_top = sorted(f1,reverse = True)
my_top20 = my_top[0:20]
print ('互动最高（点赞次数最多）：')
print (my_top20)

# 建立以top20为主索引的top20-item矩阵
top20_item = np.zeros((len(my_top20)+1, len(itemList)))
num_index = 0
for top_tuple in my_top20:
    top_index = top_tuple[2]
    top20_item[num_index]=user_item[top_index]
    num_index +=1
top20_item[num_index]=user_item[my_num]
print ('top(20+1)*item矩阵')
print (top20_item)

np.save('top21.npy', top20_item)
print('File saved!')