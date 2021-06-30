# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:42:43 2021

@author: 北湾

[reference]

https://blog.csdn.net/super_he_pi/article/details/85987543
https://blog.csdn.net/weixin_43790276/article/details/109191533

"""
import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

df = pd.read_csv(r'C:\Users\北湾\Desktop\hyx数据可视化作业\sales4.csv', encoding = "gbk")


df = df.groupby(df['客户']).sum()

labels = df._stat_axis.values.tolist()

Y1 = list(df['第1季度'].values)
Y2 = list(df['第2季度'].values)
Y3 = list(df['第3季度'].values)
Y4 = list(df['第4季度'].values)

x = range(len(labels))

# plt.plot(x, Y1, marker='o', mec='r', mfc='w',label='Fisrt_season')

# plt.plot(x, Y2, marker='*', ms=10,label='Second_season')

# plt.plot(x, Y3, marker='*', ms=10,label='Second_season')

plt.plot(x, Y1, color='green', label='First_season')
plt.plot(x, Y2, color='red', label='Second_season')
plt.plot(x, Y3,  color='skyblue', label='Third_season')
plt.plot(x, Y4, color='blue', label='fourth_season')

# plt.plot(labels, Y1, color='green', label='First_season')
# plt.plot(labels, Y2, color='red', label='Second_season')
# plt.plot(labels, Y3,  color='skyblue', label='Third_season')
# plt.plot(labels, Y4, color='blue', label='fourth_season')

plt.legend(loc='best')

plt.legend()  # 让图例生效