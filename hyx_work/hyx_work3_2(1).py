# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:17:06 2021

@author: 北湾

https://blog.csdn.net/p1306252/article/details/107502286
https://blog.csdn.net/qq_41080850/article/details/83829045

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams["font.size"]=10#设置字体大小

df = pd.read_csv(r'C:\Users\admin\Desktop\hyx数据可视化作业\customers.csv')

age_male = df.age[df.sex == '男']
age_female = df.age[df.sex == '女']


sns.distplot(age_male, 
             bins=12, 
             kde = False, 
             hist_kws = {'color':'steelblue'}, 
             label = '男性'
             )

sns.distplot(age_female, 
             bins=12, 
             kde = False, 
             hist_kws = {'color':'purple'}, 
             label = '女性'
             )

plt.title('男女乘客的年龄直方图')
# 显示图例
plt.legend()
# 显示图形
plt.show()

plt.boxplot([df.age[df.sex=='male'],
             df.age[df.sex=='female']],
            labels=['male','female']
            )
plt.title('男女保费分布箱线图')
plt.savefig('gender.png',dpi=400)
