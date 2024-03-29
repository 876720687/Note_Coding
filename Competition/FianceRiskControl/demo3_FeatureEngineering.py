#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/4/21 20:14

# 特征工程
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

# TODO 小样本
data_train = pd.read_csv('../data/train_data.csv', encoding="gbk", nrows=7500)
data_test_a = pd.read_csv('../data/test_data.csv', encoding="gbk", nrows=2500)

numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea, list(data_train.columns)))
numerical_fea.remove('isDefault')

# -------------------------------------特征交互-----------------------------------
# 构建有效特征变量


# -------------------------------------特征交互-----------------------------------
# 特征交互是为了啥，交互完了之后data_train也变得很奇怪了，为什么要用这样的显示方式？

for col in ['grade', 'subGrade']:
    temp_dict = data_train.groupby([col])['isDefault'].agg(['mean']).reset_index().rename(
        columns={'mean': col + '_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col + '_target_mean'].to_dict()

    data_train[col + '_target_mean'] = data_train[col].map(temp_dict)
    data_test_a[col + '_target_mean'] = data_test_a[col].map(temp_dict)



# 其他衍生变量 mean 和 std
for df in [data_train, data_test_a]:
    for item in ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14']:
        df['grade_to_mean_' + item] = df['grade'] / df.groupby([item])['grade'].transform('mean')
        df['grade_to_std_' + item] = df['grade'] / df.groupby([item])['grade'].transform('std')



# -------------------------------------特征选择---------------------------------
# 特征交互实现了升维，这里要开始降维了
"""
特征选择的方法：
1 Filter
方差选择法
相关系数法（pearson 相关系数）
卡方检验
互信息法
2 Wrapper （RFE）
递归特征消除法
3 Embedded
基于惩罚项的特征选择法
基于树模型的特征选择
"""

# # "纵向用缺失值上面的值替换缺失值"
# # 而且是无限向下填充
# data_train = data_train.fillna(axis=0, method='ffill')

# x_train = data_train.drop(['isDefault', 'id'], axis=1)
x_train = data_train.drop(['isDefault'], axis=1)

# 计算协方差
data_corr = x_train.corrwith(data_train.isDefault)  # 计算相关性
result = pd.DataFrame(columns=['features', 'corr'])
result['features'] = data_corr.index
result['corr'] = data_corr.values

# ---------------------获取有效特征------------------------
features = [f for f in data_train.columns if f not in ['id', 'issueDate', 'isDefault'] and '_outliers' not in f]
x_train = data_train[features]
x_test = data_test_a[features]
y_train = data_train['isDefault']

train_data = pd.concat([x_train, y_train], axis=1)

# ---------------------保存数据---------------------------
train_data.to_csv("../data/train_data_feture.csv", index=0)
x_test.to_csv("../data/test_data_feture.csv", index=0)