#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/4/21 20:14

# 特征工程
# 和上面代码所使用的 train 和 test_a 不同，这里使用的是data_train 和 data_test_a
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime

data_train = pd.read_csv('毕业设计/train.csv', encoding="gbk", nrows=7500)
data_test_a = pd.read_csv('毕业设计/testA.csv', encoding="gbk", nrows=2500)

numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea, list(data_train.columns)))

# 填充前后都最好去查看一下
data_train.isnull().sum()

pd.set_option('display.max_columns', None)
# 这个idDefault貌似是结果！！？！
numerical_fea.remove('isDefault')

# 按照中位数填充数值型特征
data_train[numerical_fea] = data_train[numerical_fea].fillna(data_train[numerical_fea].median())
data_test_a[numerical_fea] = data_test_a[numerical_fea].fillna(data_train[numerical_fea].median())
# 按照众数填充类别型特征
data_train[category_fea] = data_train[category_fea].fillna(data_train[category_fea].mode())
data_test_a[category_fea] = data_test_a[category_fea].fillna(data_train[category_fea].mode())

# 前面做了这么多内容，其实还是为了理解数据，将其补齐。
# 时间格式处理、对象类型特征转化到数值、类别特征处理



# 异常值处理
# 转化成时间格式, 然后再进行处理
for data in [data_train, data_test_a]:
    data['issueDate'] = pd.to_datetime(data['issueDate'], format='%Y-%m-%d')
    startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
    # 构造时间特征
    data['issueDateDT'] = data['issueDate'].apply(lambda x: x - startdate).dt.days

data_train['employmentLength'].value_counts(dropna=False).sort_index()

# 对象类型特征转换到数值
def employmentLength_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])

for data in [data_train, data_test_a]:
    data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
    data['employmentLength'].replace('< 1 year', '0 years', inplace=True)
    data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)

data['employmentLength'].value_counts(dropna=False).sort_index()

# 这个变量需要另外处理？
# data_train['earliesCreditLine'].sample(5)

for data in [data_train, data_test_a]:
    data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:])) # 将年月数据处理为年数据

# 类别特征处理, 部分类别特征,这部分内容？
cate_features = ['grade', 'subGrade', 'employmentTitle', 'homeOwnership', 'verificationStatus', 'purpose', 'postCode',
                 'regionCode',
                 'applicationType', 'initialListStatus', 'title', 'policyCode']
for f in cate_features:
    print(f, '类型数：', data[f].nunique())

# 像等级这种类别特征，是有优先级的可以labelencode或者自映射

for data in [data_train, data_test_a]:
    data['grade'] = data['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})

# 类型数在2之上，又不是高维稀疏的,且纯分类特征.employmentTitle 和 postCode被认为是高维稀疏
# 哑变量，减少一列显示，有意义吗、
for data in [data_train, data_test_a]:
    data = pd.get_dummies(data, columns=['subGrade', 'homeOwnership', 'verificationStatus', 'purpose', 'regionCode'],
                          drop_first=True)


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


# -------------------------------------------特征编码---------------------------------

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# label-encode:subGrade,postCode,title
# 高维类别特征需要进行转换
for col in tqdm(['employmentTitle', 'postCode', 'title', 'subGrade']):
    le = LabelEncoder()
    le.fit(list(data_train[col].astype(str).values) + list(data_test_a[col].astype(str).values))
    data_train[col] = le.transform(list(data_train[col].astype(str).values))
    data_test_a[col] = le.transform(list(data_test_a[col].astype(str).values))
print('Label Encoding 完成')

# # 举例归一化过程
# #伪代码
# for fea in [要归一化的特征列表]：
#     data[fea] = ((data[fea] - np.min(data[fea])) / (np.max(data[fea]) - np.min(data[fea])))

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












