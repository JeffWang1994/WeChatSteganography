import json
import numpy as np
import pandas as pd

# 导入JSON文件
f = open('sns_jeff.json')
origin_data = json.load(f)

print('==========================')
print('数据集示例:')
print(origin_data[1])
print("数据集中共有{}条动态信息。".format(len(origin_data)))
print('==========================\n')

# 使用pandas转化为DataFrame
data = pd.DataFrame(origin_data)
#print(data)

likes = data['likes']
#print(likes)

SUM = 0

for likes_item in likes:
    one_like = pd.DataFrame(likes_item)
    print(one_like)
    for temp in one_like:
        SUM = SUM + 1
    #SUM = SUM+sum.get_value('userId')

print(SUM)