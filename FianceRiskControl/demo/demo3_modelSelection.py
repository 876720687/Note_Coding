#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/4/23 11:55



"""
ValueError: Input contains NaN,
infinity or a value too large for dtype('float32').

"""
# ------------------模型训练----------------------------
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("train_data.csv")

y_train = train_data['isDefault']
x_train = train_data.drop(columns='isDefault', axis=1)
x_test = pd.read_csv("x_test.csv")

# df_norm = (x_train - x_train.min()) / (x_train.max() - x_train.min())

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train)

"""
# def cv_model()

import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, KFold

lgb_train, lgb_test = lgb_model(x_train, y_train, x_test)
xgb_train, xgb_test = xgb_model(x_train, y_train, x_test)

# 特征交互这个地方似乎有一点问题。导致最终的模型训练不成功。

# 但是这并不影响后面建模的内容。进行第四部分的测试。

a = train.drop(['isDefault'], axis=1)
# b=[a,testA]
b = pd.concat([a, testA], axis=0)
b.to_csv(r"E:\CODE\ProgrammingProgect\pythoncoding\基于数据分析的金融风控预测（毕业设计）\baseline\data_for_model.csv")
"""


def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2020
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i + 1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'min_child_weight': 5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.1,
                'seed': 2020,
                'nthread': 28,
                'n_jobs': 24,
                'silent': True,
                'verbose': -1,
            }

            model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=200, early_stopping_rounds=200)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)

            # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])

        if clf_name == "xgb":
            train_matrix = clf.DMatrix(trn_x, label=trn_y)
            valid_matrix = clf.DMatrix(val_x, label=val_y)

            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.04,
                      'tree_method': 'exact',
                      'seed': 2020,
                      'nthread': 36,
                      "silent": True,
                      }

            watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

            model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200,
                              early_stopping_rounds=200)
            val_pred = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
            test_pred = model.predict(test_x, ntree_limit=model.best_ntree_limit)

        if clf_name == "cat":
            params = {'learning_rate': 0.05, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                      'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

            model = clf(iterations=20000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y), cat_features=[], use_best_model=True, verbose=500)

            val_pred = model.predict(val_x)
            test_pred = model.predict(test_x)

        train[valid_index] = val_pred
        test = test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))

        print(cv_scores)

    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


def lgb_model(x_train, y_train, x_test):
    lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_train, lgb_test


def xgb_model(x_train, y_train, x_test):
    xgb_train, xgb_test = cv_model(xgb, x_train, y_train, x_test, "xgb")
    return xgb_train, xgb_test


def cat_model(x_train, y_train, x_test):
    cat_train, cat_test = cv_model(CatBoostRegressor, x_train, y_train, x_test, "cat")
    return cat_train, cat_test


lgb_train, lgb_test = lgb_model(x_train, y_train, x_test)
xgb_train, xgb_test = xgb_model(x_train, y_train, x_test)
# cat_train, cat_test = cat_model(x_train, y_train, x_test)

import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC  # 支持向量机
from sklearn.naive_bayes import MultinomialNB  # 朴素也贝斯
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.ensemble import RandomForestClassifier  # 随机森铃
from sklearn.ensemble import GradientBoostingClassifier  # GBDT
from xgboost import XGBClassifier  # xgboost


def modelReturn(model, name):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    trueNum = 0
    for i in range(len(y_test)):
        if (y_test[i] == predict[i]):
            trueNum += 1
    print(name, ":", trueNum / len(y_test))




# xgboost 46 50
model = XGBClassifier()
modelReturn(model, "xgboost")
# GBDT 40 -48
model = GradientBoostingClassifier()
modelReturn(model, "GBDT")

# 随机森林  44-46
model = RandomForestClassifier()
modelReturn(model, "RFC")

# 决策树  36-39
model = DecisionTreeClassifier()
modelReturn(model, "决策树")

# 朴素也贝斯 44-51
model = MultinomialNB()
modelReturn(model, "朴素也贝斯")

# 支持向量机  45-48
model = LinearSVC()
modelReturn(model, "支持向量机")

# SVM  48-52
model = SVC()
modelReturn(model, "SVM")

# laoss   68-73%
model = Lasso(alpha=0.005)  # 调节aplha 可以实现对拟合的。的程度
modelReturn(model, "laoss")

"""
model.fit(x_train,y_train);

predict =model.predict(x_test);

trueNum =0;

print(predict)

for i  in range(len(y_test)):
    if ((abs(y_test[i])-abs(predict[i])< 0.5)):
        trueNum += 1;


print(trueNum/len(y_test));
"""
"""
pca = PCA(n_components=27);
xTrainPca = pca.fit_transform(x_train);
xTestPca = pca.fit_transform(x_test);


log =LogisticRegression();
log.fit(xTrainPca,y_train);

print("准确率:",log.score(xTestPca,y_test));
"""

"""
#降到10个维度
pca = PCA(n_components=50);

xTrainPca = pca.fit_transform(x_train);
xTestPca = pca.fit_transform(x_test);

knn = KNeighborsClassifier(n_neighbors=11);
knn.fit(xTrainPca,y_train);

print(knn.score(xTestPca,y_test))
"""
