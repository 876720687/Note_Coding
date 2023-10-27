# -*- coding: utf-8 -*-
"""
Created on Sat May  8 18:39:14 2021

@author: 北湾

This part is not important at all.
you can find many from the web.
focus on how the data is handled and the models

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns',None) # 显示所有列

train = pd.read_csv('../data/train.csv',encoding = "gbk",nrows=1000)
testA = pd.read_csv('../data/testA.csv',encoding = "gbk", nrows = 1000)


print(train.info())
print(train.describe())
print(train.head(5).append(train.tail(5)))
print(train.isnull().sum())

# ------------------------缺失值可视化-----------------------
# 数据缺失的比率，并转化为字典
have_null_fea_dict = (train.isnull().sum()/len(train)).to_dict()
# 缺失量过大的数据列
fea_null_moreThanHalf = {}
for key,value in have_null_fea_dict.items():
    if value > 0.5: # 设定缺失值比例
        fea_null_moreThanHalf[key] = value
# 缺失值可视化
missing = train.isnull().sum()/len(train)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


# 只有单状态的变量可以直接认为无效
one_value_fea = [col for col in train.columns if train[col].nunique() <= 1]
one_value_fea_test = [col for col in testA.columns if testA[col].nunique() <= 1]

# 获得数值类型和类别类型
# object是父子关系的顶端，所有的数据类型的父类都是它；type是类型实例关系的顶端，所有对象都是它的实例的
numerical_fea = list(train.select_dtypes(exclude=['object']).columns)

category_fea = list(filter(lambda x: x not in numerical_fea,list(train.columns)))

#过滤数值型类别特征
def get_numerical_serial_fea(data,feas):
    """
    数值类型变量按照数据类别的多少进行分类，认为数据类别大于10的numerical变量为numerical_serial_fea
    数据类别少于10的认为是numerical_noserial_fea
    :param data: 数据集
    :param feas: 数值变量特征名称list
    :return:
    """
    numerical_serial_fea = []
    numerical_noserial_fea = []
    for fea in feas:
        temp = data[fea].nunique()
        if temp <= 10:
            numerical_noserial_fea.append(fea)
            continue
        numerical_serial_fea.append(fea)
    return numerical_serial_fea,numerical_noserial_fea


numerical_serial_fea, numerical_noserial_fea = get_numerical_serial_fea(train, numerical_fea)


# 这个整体图像读取显示还是非常耗时的，但是有利于清晰表达数据的分布情况
f = pd.melt(train, value_vars=numerical_serial_fea)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
#g = g.map(sns.distplot, "value")
g = g.map(sns.distplot, "value")


# ----------------------------------pic1-------------------------------------
# 绘制交易金额分布
plt.figure(figsize=(16,12))
plt.suptitle('Transaction Values Distribution', fontsize=22)
plt.subplot(221)
sub_plot_1 = sns.distplot(train['loanAmnt'])
sub_plot_1.set_title("loanAmnt Distribuition", fontsize=20)
sub_plot_1.set_xlabel("")
sub_plot_1.set_ylabel("Probability", fontsize=15)
plt.subplot(222)
sub_plot_2 = sns.distplot(np.log(train['loanAmnt']))
sub_plot_2.set_title("loanAmnt (Log) Distribuition", fontsize=18)
sub_plot_2.set_xlabel("")
sub_plot_2.set_ylabel("Probability", fontsize=15)
plt.show()

# ----------------------------------pic2-------------------------------------
plt.figure(figsize=(8, 8))
sns.barplot(train["employmentLength"].value_counts(dropna=False)[:20],
            train["employmentLength"].value_counts(dropna=False).keys()[:20])
plt.show()
# ----------------------------------pic3-------------------------------------
train_loan_fr = train.loc[train['isDefault'] == 1]
train_loan_nofr = train.loc[train['isDefault'] == 0]

# 查看类别型变量在不同y值上的分布
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
train_loan_fr.groupby('grade')['grade'].count().plot(kind='barh', ax=ax1, title='Count of grade fraud')
train_loan_nofr.groupby('grade')['grade'].count().plot(kind='barh', ax=ax2, title='Count of grade non-fraud')
train_loan_fr.groupby('employmentLength')['employmentLength'].count().plot(kind='barh', ax=ax3, title='Count of employmentLength fraud')
train_loan_nofr.groupby('employmentLength')['employmentLength'].count().plot(kind='barh', ax=ax4, title='Count of employmentLength non-fraud')
plt.show()
# ----------------------------------pic4-------------------------------------
# 查看连续型变量在不同y值上的分布
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))
train.loc[train['isDefault'] == 1]['loanAmnt'].apply(np.log).plot(kind='hist',
                                                                  bins=100,
                                                                  title='Log Loan Amt - Fraud',
                                                                  color='r',
                                                                  xlim=(-3, 10),
                                                                  ax= ax1)
train.loc[train['isDefault'] == 0]['loanAmnt'].apply(np.log).plot(kind='hist',
                                                                  bins=100,
                                                                  title='Log Loan Amt - Not Fraud',
                                                                  color='b',
                                                                  xlim=(-3, 10),
                                                                  ax=ax2)
plt.show()
# ----------------------------------pic5-------------------------------------
# 不同的统计方式的量的多少？
total = len(train)
total_amt = train.groupby(['isDefault'])['loanAmnt'].sum().sum()

plt.figure(figsize=(12,5))
plt.subplot(121)##1代表行，2代表列，所以一共有2个图，1代表此时绘制第一个图。
plot_tr = sns.countplot(x='isDefault',data=train)#data_train‘isDefault’这个特征每种类别的数量**
plot_tr.set_title("Fraud Loan Distribution \n 0: good user | 1: bad user", fontsize=14)
plot_tr.set_xlabel("Is fraud by count", fontsize=16)
plot_tr.set_ylabel('Count', fontsize=16)
for p in plot_tr.patches:
    height = p.get_height()
    plot_tr.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=15)

percent_amt = (train.groupby(['isDefault'])['loanAmnt'].sum())
percent_amt = percent_amt.reset_index()
plt.subplot(122)
plot_tr_2 = sns.barplot(x='isDefault', y='loanAmnt',  dodge=True, data=percent_amt)
plot_tr_2.set_title("Total Amount in loanAmnt  \n 0: good user | 1: bad user", fontsize=14)
plot_tr_2.set_xlabel("Is fraud by percent", fontsize=16)
plot_tr_2.set_ylabel('Total Loan Amount Scalar', fontsize=16)
for p in plot_tr_2.patches:
    height = p.get_height()
    plot_tr_2.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total_amt * 100),
            ha="center", fontsize=15)     
plt.show()
# ----------------------------------pic6-------------------------------------
# 时间格式查看，如果存在时间序列的话，那需要按照时间序列来进行分割
import datetime
#转化成时间格式  issueDateDT特征表示数据日期离数据集中日期最早的日期（2007-06-01）的天数
train['issueDate'] = pd.to_datetime(train['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
train['issueDateDT'] = train['issueDate'].apply(lambda x: x-startdate).dt.days
#转化成时间格式
testA['issueDate'] = pd.to_datetime(train['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
testA['issueDateDT'] = testA['issueDate'].apply(lambda x: x-startdate).dt.days

plt.hist(train['issueDateDT'], label='train')
plt.hist(testA['issueDateDT'], label='test')
plt.legend()
plt.title('Distribution of issueDateDT dates')
plt.show()
#train 和 test issueDateDT 日期有重叠 所以使用基于时间的分割进行验证是不明智的
#透视图 索引可以有多个，“columns（列）”是可选的，聚合函数aggfunc最后是被应用到了变量“values”中你所列举的项目上。
pivot = pd.pivot_table(train, index=['grade'], columns=['issueDateDT'], values=['loanAmnt'], aggfunc=np.sum)










