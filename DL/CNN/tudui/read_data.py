#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/5/4 20:44

#逻辑回归实现之前的乳腺癌数据集分类预测
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

cancer = datasets.load_breast_cancer()
cancer_X = cancer.data

print(cancer_X.shape)

cancer_y = cancer.target

print(cancer_y.shape)

X_train,X_test,y_train, y_test=train_test_split(cancer_X,cancer_y,test_size=0.3)

print(cancer_X)
clf = LogisticRegression()
clf.fit(X_train,y_train)
clf_y_predict = clf.predict(X_test)
#print(clf_y_predict)
score=clf.score(X_test,y_test)
print(score)