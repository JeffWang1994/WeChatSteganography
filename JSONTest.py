import json
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist


# 导入JSON文件, 返回origin_data
def Dataset(path):
    f = open(path)
    origin_data = json.load(f)

    #print(type(origin_data))
    print('==========================')
    print('数据集示例:')
    print(origin_data[1])
    #print(type(origin_data[1]))
    #print(origin_data[1]['authorName'])
    print("数据集中共有{}条动态信息。".format(len(origin_data)))
    print('==========================\n')

    return origin_data


# 获取用户列表, 返回userNameList
# 用户列表的形式为:<type: list>
#               ['wxid_aw3jddgrdgw221', 'tangna-12', 'wxid_sa3w1zpucptq22',...]
# 索引指南: userNameList[i]
def userIdList(origin_data):
    userNameList = []
    for item in origin_data:
        if not (item['authorId'] in userNameList):
            userNameList.append(item['authorId'])
        for likes_user in item['likes']:
            if not (likes_user['userId'] in userNameList):
                userNameList.append(likes_user['userId'])
    usernicklist = []
    for item in origin_data:
        if not (item['authorId'] in usernicklist):
            usernicklist.append(item['authorName'])
        for likes_user in item['likes']:
            if not (likes_user['userId'] in usernicklist):
                usernicklist.append(likes_user['userName'])
    #    for comments_user in item['comments']:
    #        if not (comments_user['authorName'] in userNameList):
    #            userNameList.append(comments_user['authorName'])

    print('==========================')
    print('用户列表:')
    print(userNameList)
    print('数据集中共有:{}名用户'.format(len(userNameList)))
    print('==========================')

    return userNameList, usernicklist


# 建立以item为主索引的item-user列表, 返回itemList
# itemList的形式为: <type: dictionary>
#                ['snsId': 12893789, 'authorId':'tangna-12', 'likesSum':23, 'userId1':0或1, ...]
# 索引指南:itemList[i][snsId]
def itemUserList(origin_data):
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

    return itemList

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


# 建立以user为主索引的user-item矩阵, 返回user_item
# user_item的形式为: <type:numpy array>
#   行为用户，列为item，每一单元为用户i对动态j的点赞情况，1为点赞，0为不点赞
def userItemList(origin_data, userNameList, itemList):
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
    return user_item


# 检测user-item矩阵中是否包含了所有的点赞记录
def examUserLike(userNameList, itemList, user_item):
    SUM = 0
    for i in range(len(userNameList)):
        for j in range(len(itemList)):
            if user_item[i,j] != 0:
                SUM = SUM + 1

    print(SUM)


# 建立余弦相似度矩阵
def cos_sim(userNameList, user_item):
    user_cos = np.zeros((len(userNameList), len(userNameList)))

    for user1 in range(len(userNameList)):
        for user2 in range(len(userNameList)):
            num = np.dot(user_item[user1], user_item[user2])
            denom = np.linalg.norm(user_item[user1]) * np.linalg.norm(user_item[user2])
            if denom != 0:
                cos = num / denom
                sim = 0.5+0.5*cos
                user_cos[user1, user2] = cos
            else:
                user_cos[user1, user2] = 0

    print(user_cos)

    return user_cos


# 建立Jaccard相似度矩阵
def Jaccard_sim(userNameList, user_item):
    user_jaccard = np.zeros((len(userNameList), len(userNameList)))

    for user1 in range(len(userNameList)):
        for user2 in range(len(userNameList)):
            matv = np.array([user_item[user1, :], user_item[user2, :]])
            if matv.max() == 0:
                user_jaccard[user1, user2] = 0
            else:
                fenzi = 0
                fenmu = 0
                for i in range(matv.shape[1]):
                    if (matv[0,i] == 1)|(matv[1,i] == 1):
                        fenmu = fenmu + 1
                        if matv[0,i] == matv[1,i]:
                            fenzi = fenzi + 1
                J = fenzi/fenmu
                user_jaccard[user1, user2] = J

    print(user_jaccard)

    return user_jaccard

# 创建测试集
def TestSet(userNameList):
    user_index = 0
    for user in userNameList:
        if user == 'wxid_vyebwqsb5wo21':
            Hong = user_index
        user_index = user_index + 1

    print(Hong)


# 保存numpy数组
def save_file(user_item):
    np.save('user_item.npy', user_item)
    print('File Saved!')


# 主函数
if __name__ == '__main__':
    origin_data = Dataset('sns_hong.json')
    userNameList, usernicklist = userIdList(origin_data)
    itemList = itemUserList(origin_data)
    user_item = userItemList(origin_data, userNameList, itemList)
    examUserLike(userNameList, itemList, user_item)
    user_cos = cos_sim(userNameList, user_item)
    save_file(user_item)
    print('finished!')







