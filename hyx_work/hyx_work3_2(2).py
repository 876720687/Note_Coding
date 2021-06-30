# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:32:39 2021

@author: 北湾
"""
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams["font.size"]=10#设置字体大小

df = pd.read_csv(r'C:\Users\admin\Desktop\hyx数据可视化作业\customers.csv')


plt.boxplot([df.age[df.sex=='男'],
             df.age[df.sex=='女']],
            labels=['male','female'],
            notch = True,
            sym='*',
            vert = False,
            showmeans=True,
            patch_artist=True,
            boxprops = {'color':'orangered','facecolor':'pink'}
            )
plt.title('男女年龄分布')

plt.legend()
# 显示图形
plt.show()
# plt.savefig('gender.png',dpi=400)