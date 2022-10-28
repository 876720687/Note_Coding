# -*- coding: utf-8 -*- 
# @Time : 2022/10/28 23:29 
# @Author : YeMeng 
# @File : demo4_2_esembelModelSample.py 
# @contact: 876720687@qq.com



#lgb
train_matrix = lgb.Dataset(train_x, label=train_y)
valid_matrix = lgb.Dataset(val_x, label=val_y)

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

model = lgb.train(train_matrix,
                  num_boost_round=50000,
                  valid_sets=[train_matrix, valid_matrix],
                  verbose_eval=200,
                  early_stopping_rounds=200
                  )

"""计算在验证集上的得分"""
val_pred = model.predict(val_x, num_iteration=model.best_iteration)
fpr, tpr, threshold = metrics.roc_curve(val_y, val_pred)
roc_auc = metrics.auc(fpr, tpr)
print('调参后lightgbm单模型在验证集上的AUC：{}'.format(roc_auc))
"""对测试集进行预测"""
test_pred = model.predict(x_test, num_iteration=model.best_iteration)
