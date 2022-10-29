# -*- coding: utf-8 -*- 
# @Time : 2022/10/28 23:29 
# @Author : YeMeng 
# @File : demo4_2_esembelModelSample.py 
# @contact: 876720687@qq.com
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb

data_train = pd.read_csv('../data/train_data.csv')
data_test_a = pd.read_csv('../data/test_data.csv')
features = [f for f in data_train.columns if f not in ['id','issueDate','isDefault'] and '_outliers' not in f]
x_train = data_train[features]
x_test = data_test_a[features]
y_train = data_train['isDefault']

train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size=0.2)

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
            if isinstance(test_x, pd.DataFrame):
                test_x = clf.DMatrix(test_x)
            test_pred = model.predict(test_x, ntree_limit=model.best_ntree_limit)

        if clf_name == "cat":
            params = {'learning_rate': 0.05, 'depth': 5, 'l2_leaf_reg': 10, 'bootstrap_type': 'Bernoulli',
                      'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

            model = CatBoostClassifier(iterations=1000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y), cat_features=[], use_best_model=True, verbose=500)

            val_pred = model.predict(val_x)
            test_pred = model.predict(test_x)


        train[valid_index] = val_pred
        test = test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))

        print("cv_scores : {}".format(cv_scores))

    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test




def lgb_model(x_train, y_train, x_test):
    lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_train, lgb_test


# def xgb_model(x_train, y_train, x_test):
#     xgb_train, xgb_test = cv_model(xgb, x_train, y_train, x_test, "xgb")
#     return xgb_train, xgb_test
#
# def cat_model(x_train, y_train, x_test):
#     cat_train, cat_test = cv_model(CatBoostRegressor, x_train, y_train, x_test, "cat")
#     return cat_train, cat_test

lgb_train, lgb_test = lgb_model(x_train, y_train, x_test)
# xgb_train, xgb_test = xgb_model(x_train, y_train, x_test)
# cat_train, cat_test = cat_model(x_train, y_train, x_test)

# ----------------------超参数优化----------------------
