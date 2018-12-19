import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.model_selection import GridSearchCV
import tensorflow as tf


# 数据清洗，取出无人点赞的条目，整理以item为主索引的item-user样本集
# 数据集示例： user1, user2, user3, ..., user192
#     item1    1,     1,    0,    ...,    1
def data_load(path):
    origin_data = np.load(path)
    [user_num, item_num] = origin_data.shape

    print('原始全部数据中，总共有{}个用户, {}条动态'.format(user_num, item_num))

    # 数据当中有多少item的特征全为0，即没有被任何人点过赞
    data = []
    for item_index in range(item_num):
        if np.sum(origin_data[:, item_index]) != 0:
            temp = (np.array(origin_data[:, item_index])).T
            data.append(temp)
    data = np.array(data)
    item_num = data.shape[0]
    user_num = data.shape[1]

    print('删除了没人点赞的item之后，总共有{}个用户，{}条动态'.format(user_num, item_num))

    # 从数据中提取出hong的数据作为label，将数据集分为feature和label
    feature = np.concatenate((data[:, 0:8], data[:, 9:user_num]), axis=1)
    label = data[:, 8]

    # 从数据集中取出单独只有hong点赞的item
    X = []
    y = []
    num = feature.shape[0]
    for index in range(num):
        if not((np.sum(feature[index, :]) == 0) & (label[index] == 1)):
            X.append(feature[index, :])
            y.append(label[index])
    X = np.array(X)
    y = np.array(y)

    X_item_num = X.shape[0]
    X_user_num = X.shape[1]
    y_item_num = y.shape[0]

    print('删除了只有hong一个人点赞的item之后，数据集中总共有{}个用户，{}条动态'.format(X_item_num, X_user_num))
    print('删除了只有hong一个人点赞的item之后，标签集中总共有1个用户，{}条动态'.format(y_item_num))

    return X, y

# 切分数据集为训练集和测试集。其比例为t
def train_test_split(feature, label, t):
    [item_num, user_num] = feature.shape
    train_num = round(item_num * t)
    train_data = feature[0:train_num, :].copy()
    test_data = feature[train_num+1:item_num, :].copy()
    train_label = label[0:train_num].copy()
    test_label = label[train_num+1:item_num].copy()

    print('训练集中包含了{}个用户, {}条动态'.format(train_data.shape[1], train_data.shape[0]))
    print('测试集中包含了{}个用户, {}条动态'.format(test_data.shape[1], test_data.shape[0]))
    print('训练集标签中包含了1个用户, {}条动态'.format(train_label.shape[0]))
    print('测试集标签中包含了1个用户, {}条动态'.format(test_label.shape[0]))

    return train_data, test_data, train_label, test_label


# 评估函数，计算准确度，精确度，
def Evaluation(test_label, test_esti):

    Acc = accuracy_score(test_label, test_esti)
    Pre = precision_score(test_label, test_esti)
    ReCall = recall_score(test_label, test_esti)

    # ROC曲线
    fpr, tpr, thresholds = roc_curve(test_label, test_esti)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('Receiver operating characteristic example')
    plt.show()

    print('准确率:{}'.format(Acc))
    print('精确率:{}'.format(Pre))
    print('召回率:{}'.format(ReCall))

    return Acc, Pre, ReCall


# 其他分类器
def other_classifier(train_data, train_label, test_data, test_label):

    # 最简单的线性回归
    linear = LogisticRegression().fit(train_data, train_label)
    print('线性模型拟合系数:{}'.format(linear.coef_))
    result_linear = linear.predict(test_data)
    print('线性回归模型得分:{}'.format(linear.score(test_data, test_label)))
    Evaluation(test_label, result_linear)

    # 岭回归
    rc = RidgeClassifier().fit(train_data, train_label)
    print('岭回归训练集得分:{}'.format(rc.score(train_data, train_label)))
    print('岭回归测试集得分:{}'.format(rc.score(test_data, test_label)))
    result_rc = rc.predict(test_data)
    Acc, Pre, ReCall = Evaluation(test_label, result_rc)
    print('================================================================')

    # 逻辑回归
    lr = LogisticRegression().fit(train_data, train_label)
    print('逻辑回归训练集得分:{}'.format(lr.score(train_data, train_label)))
    print('逻辑回归测试集得分:{}'.format(lr.score(test_data, test_label)))
    result_lr = lr.predict(test_data)
    Acc, Pre, ReCall = Evaluation(test_label, result_lr)
    #print('概率结果:{}'.format(lr.predict_proba(test_data)))
    print('================================================================')

    # 随机森林--垃圾
    rf = RandomForestClassifier(n_estimators=10).fit(train_data, train_label)
    print('随机森林训练集得分:{}'.format(rf.score(train_data, train_label)))
    print('随机森林测试集得分:{}'.format(rf.score(test_data, test_label)))
    result_rf = rf.predict(test_data)
    Evaluation(test_label, result_rf)
    print('================================================================')


    # 伯努利贝叶斯分类器
    nb = BernoulliNB().fit(train_data, train_label)
    print('伯努利贝叶斯分类器训练集得分:{}'.format(nb.score(train_data, train_label)))
    print('伯努利贝叶斯分类器测试集得分:{}'.format(nb.score(test_data, test_label)))
    result_nb = nb.predict(test_data)
    Evaluation(test_label, result_nb)
    #print('概率结果:{}'.format(nb.predict_proba(test_data)))
    print('================================================================')

    # 多项式贝叶斯分类器
    mb = MultinomialNB().fit(train_data, train_label)
    print('多项式贝叶斯分类器训练集得分:{}'.format(mb.score(train_data, train_label)))
    print('多项式贝叶斯分类器测试集得分:{}'.format(mb.score(test_data, test_label)))
    result_mb = mb.predict(test_data)
    Evaluation(test_label, result_mb)
    # print('概率结果:{}'.format(mb.predict_proba(test_data)))
    print('================================================================')


# GBDT调参
def GBDT_param(train_data, train_label, test_data, test_label):
    param_test1 = {'n_estimators': range(20, 200, 10)}
    gsearch = GridSearchCV(estimator=GradientBoostingClassifier(),
                           param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
    gsearch.fit(train_data, train_label)
    print(gsearch.best_params_)
    print('经过调整n_estimators参数后，最佳得分为:{}'.format(gsearch.score(test_data, test_label)))
    result_gsearch = gsearch.predict(test_data)

    Evaluation(test_label, result_gsearch)
    print('===============================================================')

    param_test2 = {'max_depth': range(10, 100, 2), 'min_samples_split': range(2, 50)}
    gsearch2 = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=90, min_samples_leaf=20,
                                             max_features='sqrt', subsample=0.8, random_state=10),
        param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    gsearch2.fit(train_data, train_label)
    print(gsearch2.best_params_)
    print('经过调整max_depth和min_samples_split参数后，最佳得分:{}'.format(gsearch2.score(test_data, test_label)))
    result_gsearch2 = gsearch2.predict(test_data)

    Evaluation(test_label, result_gsearch2)
    print('===============================================================')


if __name__ == '__main__':
    X, y = data_load('../DataSet/user_item.npy')
    train_data, test_data, train_label, test_label = train_test_split(X, y, 2/3)
    print('================================================================')

    # 梯度提升树
    gbdt = GradientBoostingClassifier(n_estimators=90, max_depth=10, min_samples_split=2).fit(train_data, train_label)
    print('GBDT训练集得分:{}'.format(gbdt.score(train_data, train_label)))
    print('GBDT测试集得分:{}'.format(gbdt.score(test_data, test_label)))
    result_gbdt = gbdt.predict(test_data)
    result_gbdt_probs = gbdt.predict_proba(test_data)

    Evaluation(test_label, result_gbdt)
    print('================================================================')

    other_classifier(train_data, train_label, test_data, test_label)
