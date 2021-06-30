# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:04:07 2021

@author: 北湾

reference 
https://www.cnblogs.com/LiErRui/articles/11588399.html

https://blog.csdn.net/XU_MAN_/article/details/104671818?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-1&spm=1001.2101.3001.4242

https://blog.csdn.net/weixin_43718211/article/details/85726508
"""
import pandas as pd
import matplotlib.pyplot as plt

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


df = pd.read_csv(r'C:\Users\admin\Desktop\hyx数据可视化作业\customers.csv')
                 # header = None, names=['a','b','c','d','e','f','g','h','i','g','k','l'])
# df.to_csv(r'C:\Users\北湾\Desktop\hyx数据可视化作业\customers.csv', index=false)

# # 获得数据
# age = list(df['age'].values)

# num = list(range(15,81,5))

# # 设定图像的大小
# fig = plt.figure(figsize=(16,9),dpi=72)
# ax1 = fig.add_subplot()

# ax1.bar([x+0.5 for x in range(len(num))], age, width = 1)
# ax1.set_xticks([x for x in range(len(num))])
# ax1.set_xticklabels(num)
# plt.show()


df.age.plot(kind = 'hist',
            bins = 12, 
            color = 'steelblue', 
            edgecolor = 'black'
            )




x = list(range(15,81,5))
plt.xticks(x)


# # 观察是否存在缺失值
# any(df.age.isnull())

# # 如果有缺失值就将其删除
# df.dropna(subset=['age'], inplace = True)





# 添加x轴和y轴标签
plt.xlabel('年龄')
plt.ylabel('数量')
# 添加标题
plt.title('年龄分布')
# 显示图形
plt.show()