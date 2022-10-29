#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Leo
# datetime： 2022/4/23 11:55


"""
ValueError: Input contains NaN,
infinity or a value too large for dtype('float32').

"""
import pandas as pd
from heamy.pipeline import ModelsPipeline
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from heamy.dataset import Dataset
from heamy.estimator import Classifier
from tools import *
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

data_train = pd.read_csv('../data/train_data.csv')
data_train = reduce_mem_usage(data_train)
data_test_a = pd.read_csv('../data/test_data.csv')
data_test_a = reduce_mem_usage(data_test_a)
features = [f for f in data_train.columns if f not in ['id','issueDate','isDefault'] and '_outliers' not in f]
x_train = data_train[features]
x_test = data_test_a[features]
y_train = data_train['isDefault']


def xgb_model(X_train, y_train, X_test, y_test=None):
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2)
    train_matrix = xgb.DMatrix(X_train_split, label=y_train_split)
    valid_matrix = xgb.DMatrix(X_val, label=y_val)
    test_matrix = xgb.DMatrix(X_test)

    params = {
        'booster': 'gbtree',
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
        'n_jobs': -1,
        "silent": True,
    }
    watchlist = [(train_matrix, 'train'), (valid_matrix, 'eval')]

    model = xgb.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200,
                      early_stopping_rounds=200)
    """计算在验证集上的得分"""
    val_pred = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
    fpr, tpr, threshold = metrics.roc_curve(y_val, val_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print('调参后xgboost单模型在验证集上的AUC：{}'.format(roc_auc))
    """对测试集进行预测"""
    test_pred = model.predict(test_matrix, ntree_limit=model.best_ntree_limit)

    return test_pred


def lgb_model(X_train, y_train, X_test, y_test=None):
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2)
    train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
    valid_matrix = lgb.Dataset(X_val, label=y_val)

    # 调参后的最优参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'min_child_weight': 0.32,
        'num_leaves': 14,
        'max_depth': 4,
        'feature_fraction': 0.81,
        'bagging_fraction': 0.61,
        'bagging_freq': 9,
        'min_data_in_leaf': 13,
        'min_split_gain': 0.27,
        'reg_alpha': 9.58,
        'reg_lambda': 4.62,
        'seed': 2020,
        'n_jobs': -1,
        'silent': True,
        'verbose': -1,
    }

    model = lgb.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=500,
                      early_stopping_rounds=500)
    """计算在验证集上的得分"""
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    fpr, tpr, threshold = metrics.roc_curve(y_val, val_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print('调参后lightgbm单模型在验证集上的AUC：{}'.format(roc_auc))
    """对测试集进行预测"""
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    return test_pred

model_dataset = Dataset(X_train=x_train, y_train=y_train, X_test=x_test)
model_xgb = Classifier(dataset=model_dataset, estimator=xgb_model, name='xgb', use_cache=False)
model_lgb = Classifier(dataset=model_dataset, estimator=lgb_model, name='lgb', use_cache=False)

pipeline = ModelsPipeline(model_xgb, model_lgb)
# 构建第一层新特征，其中k默认是5，表示5折交叉验证，full_test=True，对全部训练集进行训练得到基学习器，然后用基学习器对测试集预测得到新特征
stack_ds = pipeline.stack(k=5, seed=111, full_test=True)

# 第二层使用逻辑回归进行stack
LogisticRegression(solver='lbfgs')
stacker = Classifier(dataset=stack_ds, estimator=LogisticRegression, parameters={'solver': 'lbfgs'})

# 测试集的预测结果
test_pred = stacker.predict()
"""生成提交格式的DataFrame"""
df_result = pd.DataFrame({'id': data_test_a['id'], 'isDefault': test_pred})
df_result.sort_values(by='id').head(20)
"""保存数据用于预测建模"""
df_result.to_csv('../output/submission_data_stacking_model_20221026_V1_5folds.csv', encoding='gbk', index=False)