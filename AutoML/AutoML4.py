# -*- coding: utf-8 -*- 
# @Time : 2022/9/25 10:43 
# @Author : YeMeng 
# @File : AutoML4.py 
# @contact: 876720687@qq.com
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from itertools import combinations


class SelectRowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, comb_idx=[0, ]):
        self.comb_idx = comb_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, self.comb_idx].copy()


if __name__ == '__main__':
    __spec__ = None
    K = 10
    data, label = make_classification(n_samples=200, n_informative=4, n_redundant=0,
                                      random_state=223, n_features=K)
    pipe = Pipeline([
        ('Comb', SelectRowTransformer()),
        ('SVC', LinearSVC()),
    ])
    param = {
        'Comb__comb_idx': [i for j in range(K) for i in combinations(range(K), j + 1)],
        'SVC__C': [2 ** (f - 2) for f in range(5)]
    }
    grid = GridSearchCV(pipe, param, cv=3, verbose=1, n_jobs=-1)
    grid.fit(data, label)
    print('最佳CV得分:{0}, 最佳得分对应的特征组合:{1}, SVC-C:{2}'.format(grid.best_score_,
                                                                         grid.best_params_['Comb__comb_idx'],
                                                                         grid.best_params_['SVC__C']))