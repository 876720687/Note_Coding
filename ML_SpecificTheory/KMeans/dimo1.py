# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 13:54:24 2021

@author: Administrator
"""

import numpy as np

c1x=np.random.uniform(0.5,1.5,(1,10))
c1y=np.random.uniform(0.5,1.5,(1,10))
c2x=np.random.uniform(3.5,4.5,(1,10))
c2y=np.random.uniform(3.5,4.5,(1,10))
x=np.hstack((c1x,c2x))
y=np.hstack((c2y,c2y))
X=np.vstack((x,y)).T
Y=np.hstack((x,y)).T
print(X)


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
 
kemans=KMeans(n_clusters=2)
result=kemans.fit_predict(X) #训练及预测
print(result)   #分类结果
 
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei'] #散点图标签可以显示中文

x=[i[0] for i in X]
y=[i[1] for i in X]

plt.scatter(x,y,c=result,marker='o')

plt.xlabel('x')
plt.ylabel('y')
plt.show()




import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
 
K=range(1,10)
meanDispersions=[]
for k in K:
    kemans=KMeans(n_clusters=k)
    kemans.fit(X)
    #计算平均离差
    m_Disp=sum(np.min(cdist(X,kemans.cluster_centers_,'euclidean'),axis=1))/X.shape[0]
    meanDispersions.append(m_Disp)
 
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei'] #使折线图显示中文
 
plt.plot(K,meanDispersions,'bx-')

plt.xlabel('k')
plt.ylabel('平均离差')
plt.title('用肘部方法选择K值')
plt.show()