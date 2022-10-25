#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/4/21 20:14

# 特征工程
# 和上面代码所使用的 train 和 test_a 不同，这里使用的是data_train 和 data_test_a
import pandas as pd

import numpy as np
import datetime
pd.set_option('display.max_columns', None)

# TODO 小样本
data_train = pd.read_csv('../data/train_data.csv', encoding="gbk", nrows=7500)
data_test_a = pd.read_csv('../data/test_data.csv', encoding="gbk", nrows=2500)




numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea, list(data_train.columns)))
numerical_fea.remove('isDefault')


# --------------------------处理异常值，这个是补齐之后再清洗处理-----------------------------------------
def find_outliers_by_3segama(data, fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    data[fea + '_outliers'] = data[fea].apply(lambda x: str('异常值') if x > upper_rule or x < lower_rule else '正常值')
    return data


# 删除异常值，只处理了训练集
for fea in numerical_fea:
    data_train = find_outliers_by_3segama(data_train, fea)
    print(data_train[fea + '_outliers'].value_counts())
    print(data_train.groupby(fea + '_outliers')['isDefault'].sum())
    print('*' * 10)


# -------------------------------------特征交互-----------------------------------
# 特征交互是为了啥，交互完了之后data_train也变得很奇怪了，为什么要用这样的显示方式？
# 删除异常值
for fea in numerical_fea:
    data_train = data_train[data_train[fea + '_outliers'] == '正常值']
    data_train = data_train.reset_index(drop=True)

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

# 终于要选择特征了、这里要不还是用下因子分析？层次分析法？主成分分析法？

# from sklearn.feature_selection import VarianceThreshold
# #其中参数threshold为方差的阈值
# VarianceThreshold(threshold=3).fit_transform(train,target_train)


# 删除不需要的数据
for data in [data_train, data_test_a]:
    data.drop(['issueDate','id'], axis=1,inplace=True)

# # "纵向用缺失值上面的值替换缺失值"
# # 而且是无限向下填充
data_train = data_train.fillna(axis=0, method='ffill')

# x_train = data_train.drop(['isDefault', 'id'], axis=1)
x_train = data_train.drop(['isDefault'], axis=1)

# 计算协方差
data_corr = x_train.corrwith(data_train.isDefault)  # 计算相关性
result = pd.DataFrame(columns=['features', 'corr'])
result['features'] = data_corr.index
result['corr'] = data_corr.values

# ------------------------相关性可视化--------------------------
# data_numeric = data_train[numerical_fea]
# correlation = data_numeric.corr()
# 
# f, ax = plt.subplots(figsize=(7, 7))
# plt.title('Correlation of Numeric Features with Price', y=1, size=16)
# sns.heatmap(correlation, square=True, vmax=0.8)

# ---------------------获取有效特征------------------------
features = [f for f in data_train.columns if f not in ['id', 'issueDate', 'isDefault'] and '_outliers' not in f]
x_train = data_train[features]
x_test = data_test_a[features]
y_train = data_train['isDefault']

train_data = pd.concat([x_train, y_train], axis=1)



# ---------------------保存数据---------------------------

train_data.to_csv("train_data.csv", index=0)
x_test.to_csv("x_test.csv", index=0)












