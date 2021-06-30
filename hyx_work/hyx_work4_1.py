# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:58:53 2021

@author: admin
"""

'''

实验作业04 
1. 读取“sales4.csv”中的数据，然后对”产品”按4个季度分别进行分组汇总，
最后分别绘制出每个季度各个产品销售额的火柴杆图（基线为该季度所有产品销售额的平均值）。
要求是在一个图里要绘制出4（2x2）个子图，有相应的子标题，图大小合适，必要的可视化元素齐全 。

2. 基于下列项目计划相关数据，绘制项目计划甘特图。
数据：data = [dict(Task = "调研", Start = '2021-06-01', End = '2021-06-20'),   
           dict(Task = "研发", Start = '2021-06-21', End = '2021-08-15'),   
           dict(Task = "测试", Start = '2021-08-16', End = '2021-09-30'),   
           dict(Task = "试用", Start = '2021-10-01', End = '2021-10-31')]。
（可以考虑将data转换为DataFrame类型）
提交：Python程序代码（.py文件或.ipynb文件）。


dfname._stat_axis.values.tolist() # 行名称
 
dfname.columns.values.tolist()    # 列名称

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv(r'C:\Users\admin\Desktop\hyx数据可视化作业\sales4.csv', encoding = "gbk")

df_1 = df.groupby(df['产品']).sum()

# 设置显示中文
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体 
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号


# 构造数据
x = df_1._stat_axis.values.tolist()
y1 = df_1['第1季度'] 
mean1 = int(df_1['第1季度'].sum()/25)
list1 = [mean1 for x in range(0,25)]

y2 = df_1['第2季度'] 
mean2 = df_1['第2季度'].sum()/25
list2= [mean2 for x in range(0,25)]

y3 = df_1['第3季度'] 
mean3 = df_1['第3季度'].sum()/25
list3 = [mean3 for x in range(0,25)]

y4 = df_1['第4季度'] 
mean4 = df_1['第4季度'].sum()/25
list4 = [mean4 for x in range(0,25)]

# plt.figure(figsize=(15,15))
# plt.figure()

# plt.subplot(221)
# plt.stem(x,y1)


# plt.subplot(222)
# plt.stem(x, y2, linefmt='r--', markerfmt='gD', basefmt='b--', bottom=1)

# plt.subplot(223)
# plt.stem(x,y3,label='3')

# plt.subplot(224)
# markerline, stemlines, baseline = plt.stem(x, y4)
# print(markerline.get_color(),  baseline.get_color())

# plt.show()


x = list(range(0,25))

fig = plt.figure(figsize=(15,15))
fig,ax = plt.subplots(2,2)
ax[0][0].stem(x,y1)
ax[0][0].plot(x,list1,color='red')
ax[0][0].set_title('第1季度')

ax[0][1].stem(x,y2)
ax[0][1].plot(x,list2,color='red')
ax[0][1].set_title('第2季度')

ax[1][0].stem(x,y3)
ax[1][0].plot(x,list3,color='red')
ax[1][0].set_title('第3季度')

ax[1][1].stem(x,y4)
ax[1][1].plot(x,list4,color='red')
ax[1][1].set_title('第4季度')

plt.tight_layout()

