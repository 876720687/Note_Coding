import lightgbm


def select_by_lgb(train_data, train_label, random_state=2020, n_splits=5, metric='auc', num_round=10000,
                  early_stopping_rounds=200):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = train_data.columns
    fold = 0
    for train_idx, val_idx in kfold.split(train_data):
        random_state += 1
        train_x = train_data.loc[train_idx]
        train_y = train_label.loc[train_idx]
        test_x = train_data.loc[val_idx]
        test_y = train_label.loc[val_idx]
        clf = lightgbm
        train_matrix = clf.Dataset(train_x, label=train_y)
        test_matrix = clf.Dataset(test_x, label=test_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'learning_rate': 0.1,
            'metric': metric,
            'seed': 2020,
            'nthread': -1}

        model = clf.train(
            params,
            train_matrix,
            num_round, valid_sets=test_matrix,
            early_stopping_rounds=early_stopping_rounds)

        feature_importances['fold_{}'.format(fold + 1)] = model.feature_importance()
        fold += 1
    feature_importances['averge'] = feature_importances[['fold_{}'.format(i) for i in range(1, n_splits + 1)]].mean(
        axis=1)
    return feature_importances