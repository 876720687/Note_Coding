# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:20:58 2021

@author: 北湾
"""
'''

里面遇到的问题？
1、dataframe.sum() 和 axis = 1
2、dataframe.shape[] 是个什么东西？

'''

import pandas as pd


df = pd.read_csv(r'C:\Users\admin\Desktop\hyx数据可视化作业\sales4.csv', encoding = "gbk")

df_1 = df.groupby(df['产品']).sum()

# df['总和'] = df.sum(axis = 1)
df_1['总和'] = df_1.sum(axis = 1)

# dataframe.sum() how to use this function ?

# 并不需要对其进行排序吧？
# df_1 = df_1.sort_values(by='总和')

# 图像部分内容的输出,how ?
import matplotlib.pyplot as plt 
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# import numpy as np
# from pandas import DataFrame


# plt.figure(figsize = (26,8))

# n = df_1.shape[0]


# width = 0.35
# x = np.arange(0, (int(width*4)+1)*n, (int(width*4)+1) ) + 1


'''
plt.bar(x, height, width, bottom, **kwargs)
参数 x，height，width，bottom，align：确定了柱体的位置和宽度
（1）参数（x, height）定义在什么位置上，多高的bar（注意X的值和X轴区别）
（2）参数 width 定义bar的宽度
（3）参数 bottom 定义bar的其实高度（常用于用堆叠的方式展示两组数据）
'''

x
y1
y2

name_list = list(df['产品'].unique())

Y1 = list(df['第1季度'].values)
Y2 = list(df['第2季度'].values)
Y3 = list(df['第3季度'].values)
Y4 = list(df['第4季度'].values)

x = list(range(len(Y1)))

total_width, n = 0.8, 4
width = total_width / n

plt.bar(x, Y1, width = width, label = '第1季度', fc = 'y')

for i in range(len(x)):
    x[i] = x[i] + width
    
plt.bar(x, Y2, width = width, label="第2季度", fc = 'r')
plt.bar(x, Y3, width = width, label="第3季度", fc = 'g')
plt.bar(x, Y4, width = width, label="第4季度", fc = 'b')

plt.legend()
plt.show()












# 想办法将所有内容放置在一张图上

# plt.bar(range(len(Y2)), Y2,color='rgb')

# plt.bar(x, Y1, width = width, label = '第1季度', alpha = 0.7)
# plt.bar(x + width * 1, Y2, width = width, label = '第2季度', alpha = 0.7)
# plt.bar(x + width * 2, Y3, width = width, label = '第3季度', alpha = 0.7)
# plt.bar(x + width * 3, Y4, width = width, label = '第4季度', alpha = 0.7)

# plt.xticks(x + width / 4)

# plt.legend()

# plt.show()




# # 随机生成
# df_2 = pd.DataFrame(np.random.rand(10,4), columns=['a', 'b', 'c', 'd'])
# df_2.plot(kind='bar')
# plt.show()







# # 设置标题
# plt.title("四季度柱状图")

# # 为两条坐标轴设置名称
# # plt.xlabel("季度")
# # plt.ylabel("总量")
# # 显示图例
# # plt.legend()
# plt.show()