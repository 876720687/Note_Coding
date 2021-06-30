# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:50:55 2021

@author: 北湾

【reference】
https://cloud.tencent.com/developer/article/1722766

https://blog.csdn.net/jenyzhang/article/details/52047999
调整分辨率
https://blog.csdn.net/weixin_34613450/article/details/80678522
"""

import pandas as pd


df = pd.read_csv(r'C:\Users\北湾\Desktop\hyx数据可视化作业\sales4.csv', encoding = "gbk")

df = df.groupby(df['产品']).sum()

df['总和'] = df.sum(axis = 1)

#下面的内容就是想办法显示
import matplotlib.pyplot as plt 
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.rcParams['savefig.dpi'] = 1080 #图片像素
plt.rcParams['figure.dpi'] = 1080 #分辨率

X = list(df['总和'])
# python dataframe 获得 列名columns 和行名称 index
labels = df._stat_axis.values.tolist()

fig = plt.figure(figsize=(20, 6.5))

# 绘图
patches,l_text,p_text = plt.pie(X, labels = labels, autopct='%1.2f%%') #画饼图（数据，数据对应的标签，百分数保留两位小数点）

# 设置饼图内文字大小
for t in p_text:
  t.set_size(6)
 
for t in l_text:
  t.set_size(5)


plt.title("Pie chart")

plt.show()