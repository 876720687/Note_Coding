# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:59:37 2021

@author: admin
"""

'''

逻辑回归

'''



##  基础函数库
import numpy as np 

## 导入画图库
import matplotlib.pyplot as plt
import seaborn as sns

## 导入逻辑回归模型函数
from sklearn.linear_model import LogisticRegression

##Demo演示LogisticRegression分类

## 构造数据集
x_fearures = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
y_label = np.array([0, 0, 0, 1, 1, 1])

## 调用逻辑回归模型
lr_clf = LogisticRegression()

## 用逻辑回归模型拟合构造的数据集
lr_clf = lr_clf.fit(x_fearures, y_label) #其拟合方程为 y=w0+w1*x1+w2*x2

## 查看其对应模型的w, 称之为系数矩阵
print('the weight of Logistic Regression:',lr_clf.coef_)

## 查看其对应模型的w0
print('the intercept(w0) of Logistic Regression:',lr_clf.intercept_)

x_f = np.array([[1,2],[3,-1],[5,2]])

from sklearn.datasets import load_iris
import pandas as pd
data = load_iris() #得到数据特征
iris_target = data.target #得到数据对应的标签
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names) #利用Pandas转化为DataFrame格式

# a = list(iris_features.columns)

from sklearn.model_selection import train_test_split

## 选择其类别为0和1的样本 （不包括类别为2的样本）
iris_features_part = iris_features.iloc[:100]
iris_target_part = iris_target[:100]

## 测试集大小为20%， 80%/20%分
x_train, x_test, y_train, y_test = train_test_split(iris_features_part, iris_target_part, test_size = 0.2, random_state = 2020)







'''

朴素贝叶斯

'''
import warnings
warnings.filterwarnings('ignore')
import numpy as np
# 加载莺尾花数据集
from sklearn import datasets
# 导入高斯朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# 使用高斯朴素贝叶斯进行计算
clf = GaussianNB(var_smoothing=1e-8)
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
acc = np.sum(y_test == y_pred) / X_test.shape[0]
print("Test Acc : %.3f" % acc)

# 预测
y_proba = clf.predict_proba(X_test[:1])
print(clf.predict(X_test[:1]))
print("预计的概率值:", y_proba)

'''

KNN

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets






















'''

图算法的实现

'''

class DenseGraph:
    def __init__(self, n, directed = False ):
        self.n = n # number of vertex
        self.m = 0 # number of edge
        self.directed = directed
        self.matrix = [[0 for i in range(n)] for i in range(n)] # 用这样的方式定义了一个二维矩阵
       
    def __str__(self):
        for line in self.matrix:
            print(str(line))
        return ''
    
    def getNumberOfEdge(self):
        return self.m
    
    def getNumberOfVertex(self):
        return self.n
    
    def hasEdge(self, v, w):
        if 0 <= v <= self.n and 0 <= w <= self.n:
            return self.matrix[v][w]
        else:
            raise Exception("vertex not in the Graph")
            
    def addEdge(self, v, w):
        if 0 <= v <= self.n and 0 <= w <= self.n:
            if self.hasEdge(v,w):
                return
            self.matrix[v][w] = 1
            if self.directed is False:
                self.matrix[w][v] = 1
            self.m += 1
        else:
            raise Exception("vertex not in the Graph")

import regex as re
def buildGraphFromFile(aGraph,filePath):
    graphList=[]
    with open(filePath,'r',encoding='utf-8') as f:
        for line in f:
            graphList.append([int(x) for x in re.split(r'\s+',line.strip())])
    for i in range(len(graphList)):
        aGraph.addEdge(graphList[i][0],graphList[i][1])


g1=DenseGraph(13)  #必须填入正确的结点个数。。。我真的觉得邻接矩阵不好用
buildGraphFromFile(g1,'E:/CODE/ProgrammingProgect/pythoncoding/testG1.txt')
print(g1)


from random import random, choice
import networkx as nx
import matplotlib.pyplot as plt
 
def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
 
G = nx.Graph()
 
points = [(random(), random()) for _ in range( 8 )]
 
for p1, p2 in zip(points[:-1], points[1:]):
    G.add_edge(p1, p2, weight=dist(p1, p2))
 
 
for _ in range( 8 ):
    p1, p2 = choice(points), choice(points)
    G.add_edge(p1, p2, weight=dist(p1, p2))
    
nx.draw(G)
plt.savefig( 'asd.png' )
plt.show()

from matplotlib.pyplot as plt




