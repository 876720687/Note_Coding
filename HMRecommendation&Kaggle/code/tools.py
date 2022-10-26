# -*- coding: utf-8 -*- 
# @Time : 2022/10/26 16:24 
# @Author : YeMeng 
# @File : tools.py 
# @contact: 876720687@qq.com
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
import datetime as dt


def customer_hex_id_to_int(series):
    return series.str[-16:].apply(hex_id_to_int)


def hex_id_to_int(str):
    return int(str[-16:], 16)


def article_id_str_to_int(series):
    return series.astype('int32')


def article_id_int_to_str(series):
    return '0' + series.astype('str')


class Categorize(BaseEstimator, TransformerMixin):
    def __init__(self, min_examples=0):
        self.min_examples = min_examples
        self.categories = []

    def fit(self, X):
        for i in range(X.shape[1]):
            vc = X.iloc[:, i].value_counts()
            self.categories.append(vc[vc > self.min_examples].index.tolist())
        return self

    def transform(self, X):
        data = {X.columns[i]: pd.Categorical(X.iloc[:, i], categories=self.categories[i]).codes for i in
                range(X.shape[1])}
        return pd.DataFrame(data=data)



def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10, return_apks=False):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    # return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
    assert len(actual) == len(predicted)
    apks = [apk(ac, pr, k) for ac, pr in zip(actual, predicted) if 0 < len(ac)]
    if return_apks:
        return apks
    return np.mean(apks)



def iter_to_str(iterable):
    return " ".join(map(lambda x: str(0) + str(x), iterable))


def blend(dt, w=[], k=12):
    if len(w) == 0:
        w = [1] * (len(dt))
    preds = []
    for i in range(len(w)):
        preds.append(dt[i].split())
    res = {}
    for i in range(len(preds)):
        if w[i] < 0:
            continue
        for n, v in enumerate(preds[i]):
            if v in res:
                res[v] += (w[i] / (n + 1))
            else:
                res[v] = (w[i] / (n + 1))
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
    return ' '.join(res[:k])

def prune(pred, ok_set, k=12):
    pred = pred.split()
    post = []
    for item in pred:
        if int(item) in ok_set and not item in post:
            post.append(item)
    return " ".join(post[:k])





# 召回数据集
def recall_rules_cmb(val_week, train_type=True):
    actual = df[df.week == val_week].groupby('customer_id')['article_id'].apply(list).reset_index()
    actual['article_id'] = [iter_to_str(xx) for xx in actual['article_id']]

    if train_type:
        sub_part = sub[sub.customer_id.isin(actual.customer_id.unique())]
    else:
        sub_part = sub.copy()

    ##hot
    sub_actual = pd.merge(sub_part, actual, how='left', on='customer_id')
    sub_actual['prediction'] = iter_to_str(
        df[df.week.isin([val_week - 1])]['article_id'].value_counts().head(24).index.tolist())  # 24

    ##last buy
    sub_actual_2 = sub_actual.copy()
    del sub_actual_2['prediction']
    last_date = df.loc[df.week < val_week].t_dat.max()

    init_date = last_date - dt.timedelta(days=9999)
    df2 = df.loc[(df.t_dat >= init_date) & (df.t_dat <= last_date)].copy()
    df2 = df2.merge(df2.groupby('customer_id').t_dat.max().reset_index().rename(columns={'t_dat': 'l_dat'}),
                    on='customer_id', how='left')
    df2['d_dat'] = (df2.l_dat - df2.t_dat).dt.days
    df2 = df2.loc[df2.d_dat < 14].sort_values(['t_dat'], ascending=False).drop_duplicates(['customer_id', 'article_id'])
    last_buy = df2.groupby('customer_id')['article_id'].apply(list).reset_index()
    # last_buy = df[df.week.isin([val_week-1,val_week-2,val_week-3])].groupby('customer_id')['article_id'].apply(list).reset_index()
    last_buy['article_id'] = [iter_to_str(xx) for xx in last_buy['article_id']]
    last_buy.columns = ['customer_id', 'prediction']
    sub_actual_2 = pd.merge(sub_actual_2, last_buy, how='left', on='customer_id')

    ##same color
    init_date = last_date - dt.timedelta(days=6)
    df3 = df.loc[(df.t_dat >= init_date) & (df.t_dat <= last_date)].copy() \
        .groupby(['article_id']).t_dat.count().reset_index()

    adf = pd.read_parquet('../input/hm-parquets-of-datasets/articles.parquet')
    adf = adf.merge(df3, on='article_id', how='left').rename(columns={'t_dat': 'ct'}) \
        .sort_values('ct', ascending=False).query('ct > 0')

    map_to_col = defaultdict(list)
    for aid in adf.article_id.tolist():
        map_to_col[aid] = list(filter(lambda x: x != aid, adf[adf.product_code == aid // 1000].article_id.tolist()))[:1]

    def map_to_variation(s):
        f = lambda item: iter_to_str(map_to_col[int(item)])
        return ' '.join(map(f, s.split()))

    last_buy['other_colors'] = last_buy['prediction'].fillna('').apply(map_to_variation)

    del last_buy['prediction']
    last_buy.columns = ['customer_id', 'prediction']
    sub_actual_3 = sub_actual.copy()
    del sub_actual_3['prediction']
    sub_actual_3 = pd.merge(sub_actual_3, last_buy, how='left', on='customer_id')

    sub_total = pd.merge(sub_actual[['customer_id', 'article_id', 'prediction']], \
                         sub_actual_2[['customer_id', 'prediction']], \
                         how='left', on='customer_id')
    sub_total = pd.merge(sub_total, sub_actual_3[['customer_id', 'prediction']], \
                         how='left', on='customer_id')

    sub_total.columns = ['customer_id', 'actual', 'hot', 'old_buy', 'other_color']
    sub_total = sub_total.fillna('')

    init_date = last_date - dt.timedelta(days=11)
    sold_set = set(df.loc[(df.t_dat >= init_date) & (df.t_dat <= last_date)].article_id.tolist())
    sub_total['prediction'] = sub_total[['old_buy', 'other_color', 'hot']] \
        .apply(blend, w=[100, 10, 1], axis=1, k=50).apply(prune, ok_set=sold_set, k=48)

    sub_total = sub_total[['customer_id', 'actual', 'prediction']]
    sub_total['article_id'] = [[int(xxx) for xxx in xx.split(' ')] if xx != '' else np.nan for xx in sub_total.actual]
    sub_total['pred'] = [[int(xxx) for xxx in xx.split(' ')] if xx != '' else [] for xx in sub_total.prediction]

    return sub_total


