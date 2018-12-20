import json
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist

# 导入JSON文件
f = open('sns_hong.json')
origin_data = json.load(f)

#print(type(origin_data))
print('==========================')
print('数据集示例:')
print(origin_data[1])
#print(type(origin_data[1]))
#print(origin_data[1]['authorName'])
print("数据集中共有{}条动态信息。".format(len(origin_data)))
print('==========================\n')

# 从原始数据中抽取本人点过赞的信息和部分没有点赞的信息,形成测试集
