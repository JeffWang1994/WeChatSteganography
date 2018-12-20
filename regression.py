#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np

def linear_regression(X, Y, method = "gd", gamma=0.001, eps=0.0001, max_iter=10000):
    #求使得 ||Xw-Y||^2最小的w
    #X, Y是样本和label,是np.array类型
    #如果method=="closed_form"，则用闭式解方法求出w，不需要后面三个参数
    #如果method=="gd"，则用梯度下降方法求出w
    if method == "closed_form":
        return linear_regression_by_closed_form(X, Y)
    if method == "gd":
        return linear_regression_by_gd(X, Y, gamma, eps, max_iter)
    print ("args error")

def linear_regression_by_closed_form(X, Y): #闭式解求线性回归
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y);
    return w

def linear_regression_by_gd(X, Y, gamma=0.001, eps=0.0001, max_iter=10000): #梯度下降求线性回归
    pre_w = np.array(np.ones((X.shape[1], 1)))
    cur_w = np.array(np.zeros((X.shape[1], 1)))
    count = 1
    while (cur_w - pre_w).T.dot(cur_w - pre_w) > eps and count < max_iter:
        pre_w = cur_w
        cur_w = cur_w - gamma / np.sqrt(count) * X.T.dot( X.dot(cur_w) - Y)
        count += 1
    return cur_w

def log_regression(X, Y, method="gd", gamma=0.001, eps=0.0001, max_iter=10000):
    ln_Y = np.log(Y)
    return linear_regression(X, ln_Y, method, gamma, eps, max_iter)

def logistic_regression(X, Y, method = "gd", gamma=0.001, eps=0.0001, max_iter=10000):
    #X, Y是样本和label,是np.array类型
    #要求，label的取值为0或1
    #如果method=="gd"，则用梯度下降方法求出w
    if method == "gd":
        return logistic_regression_by_gd(X, Y, gamma, eps, max_iter)
    print ("args error")


def logistic_regression_by_gd(X, Y, gamma=0.001, eps=0.0001, max_iter=10000) :
    pre_w = np.array(np.ones((X.shape[1], 1)))
    cur_w = np.array(np.zeros((X.shape[1], 1)))
    count = 1
    while (cur_w - pre_w).T.dot(cur_w - pre_w) > eps and count < max_iter:
        pre_w = cur_w
        gradient = X.T.dot(sigmoid(X.dot(cur_w)) - Y)
        cur_w = cur_w - gamma / np.sqrt(count) * gradient
        count += 1
    return cur_w

def sigmoid(x):
    return 1.0 / (1 + np.exp( - x))


def Evaluation(test_label, test_esti):
    TP, TN, FP, FN = 0, 0, 0, 0
    num = test_esti.shape[0]
    for index in range(num):
        if (test_label[index] == 1)&(test_esti[index] == 1):
            TP = TP + 1
        elif (test_label[index] == 1)&(test_esti[index] == 0):
            FP = FP + 1
        elif (test_label[index] == 0)&(test_esti[index] == 1):
            FN = FN + 1
        elif (test_label[index] == 0)&(test_esti[index] == 0):
            TN = TN + 1

    Acc = (TP+TN)/(TP+TN+FP+FN)
    Pre = TP/(TP+FP)
    ReCall = TP/(TP+FN)

    print('准确率:{}'.format(Acc))
    print('精确率:{}'.format(Pre))
    print('召回率:{}'.format(ReCall))

    return Acc, Pre, ReCall

def autoNorm(data):         #传入一个矩阵
    mins = data.min(0)      #返回data矩阵中每一列中最小的元素，返回一个列表
    maxs = data.max(0)      #返回data矩阵中每一列中最大的元素，返回一个列表
    ranges = maxs - mins    #最大值列表 - 最小值列表 = 差值列表
    normData = np.zeros(np.shape(data))     #生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据
    row = data.shape[0]                     #返回 data矩阵的行数
    normData = data - np.tile(mins,(row,1)) #data矩阵每一列数据都减去每一列的最小值
    normData = normData / np.tile(ranges,(row,1))   #data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）
    return normData
