# -*- coding: utf-8 -*- 
# @Time : 2022/10/26 16:36 
# @Author : YeMeng 
# @File : demo2.py 
# @contact: 876720687@qq.com
from tools import *



df = pd.read_parquet('../input/hm-parquets-of-datasets/transactions_train.parquet')
customers = pd.read_parquet('../input/hm-parquets-of-datasets/customers.parquet')
articles = pd.read_parquet('../input/hm-parquets-of-datasets/articles.parquet')
sub = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')
cid = pd.DataFrame(sub.customer_id.apply(lambda s: int(s[-16:], 16)))
customer_id_dict = dict(zip(sub.customer_id.tolist(),[xx[0] for xx in list(cid.values)]))
customer_id_inv_dict = dict(zip([xx[0] for xx in list(cid.values)],sub.customer_id.tolist()))
sub['customer_id'] = cid




#####验证召回的得分
valset_104 = recall_rules_cmb(104)
ap12 = mapk(valset_104.article_id, valset_104.pred, return_apks=True)
map12 = round(np.mean(ap12), 6)
print(map12)


# 排序模型数据集建立
def get_max_dat(feature_week):
    last_date = df.loc[df.week == feature_week].t_dat.max()
    init_date = last_date - dt.timedelta(days=9999)
    print('init_date:',init_date)
    print('last_date:',last_date)

    df2 = df.loc[(df.t_dat >= init_date) & (df.t_dat <= last_date)].copy()
    df2 = df2.merge(df2.groupby('customer_id').t_dat.max().reset_index().rename(columns={'t_dat':'l_dat'}),
                       on = 'customer_id', how='left')
    df2['d_dat'] = (df2.l_dat - df2.t_dat).dt.days
    df2 = df2.loc[df2.d_dat < 100000].sort_values(['t_dat'], ascending=False).drop_duplicates(['customer_id', 'article_id'])
    return df2[['customer_id', 'article_id','d_dat']]

def split_row_rank(data, column):
    """拆分成行
    :param data: 原始数据
    :param column: 拆分的列名
    :type data: pandas.core.frame.DataFrame
    :type column: str
    """
    row_len = list(map(len, data[column].values))
    rows = []
    for i in data.columns:
        if i == column:
            row = np.concatenate(data[i].values)
        else:
            row = np.repeat(data[i].values, row_len)
        rows.append(row)
    tmp = pd.DataFrame(np.dstack(tuple(rows))[0], columns=data.columns.tolist())
    tmp = tmp.reset_index()
    tmp['recall_rank'] = tmp.groupby('customer_id')['index'].rank()
    del tmp['index']
    tmp = tmp[['customer_id','pred','recall_rank']]
    tmp.columns = ['customer_id','article_id','recall_rank']
    return tmp

def split_row(data, column):
    """拆分成行
    :param data: 原始数据
    :param column: 拆分的列名
    :type data: pandas.core.frame.DataFrame
    :type column: str
    """
    row_len = list(map(len, data[column].values))
    rows = []
    for i in data.columns:
        if i == column:
            row = np.concatenate(data[i].values)
        else:
            row = np.repeat(data[i].values, row_len)
        rows.append(row)
    return pd.DataFrame(np.dstack(tuple(rows))[0], columns=data.columns)


def gen_train_dataset(valset):
    traindata = valset.copy()

    recall_rank_pd = split_row_rank(traindata, 'pred')

    traindata['buy_not'] = [list(set(yy) - set(xx)) for (xx, yy) in zip(traindata['article_id'], traindata['pred'])]
    traindata1 = split_row(traindata, 'article_id')
    traindata1['purchased'] = 1
    traindata1 = traindata1[['customer_id', 'article_id', 'purchased']]
    traindata0 = split_row(traindata, 'buy_not')
    traindata0['purchased'] = 0
    traindata0 = traindata0[['customer_id', 'buy_not', 'purchased']]
    traindata0.columns = ['customer_id', 'article_id', 'purchased']
    traindata1.columns = ['customer_id', 'article_id', 'purchased']
    traindata01 = pd.concat([traindata0, traindata1]).reset_index(drop=True)
    traindata01 = pd.merge(traindata01, recall_rank_pd, how='left', on=['customer_id', 'article_id'])

    return traindata01

def gen_test_dataset(valset):
    traindata = valset.copy()
    recall_rank_pd = split_row_rank(traindata, 'pred')

    traindata0 = split_row(traindata, 'pred')
    traindata0['purchased'] = 0
    traindata0 = traindata0[['customer_id','pred','purchased']]
    traindata0.columns = ['customer_id','article_id','purchased']
    traindata0 = pd.merge(traindata0,recall_rank_pd,how='left',on=['customer_id','article_id'])

    return traindata0


def gen_feature(traindata01, feature_week):
    ## 最近一周购买商品数/type数
    feature_user1 = df[df.week.isin([feature_week])].groupby(['customer_id', ])['price'].agg(
        ['count', 'mean']).reset_index()
    feature_user1.columns = ['customer_id', 'user_last1_week_buy_number', 'user_last1_week_buy_price']

    ## 最近两周购买商品数
    feature_user2 = df[(df.week <= feature_week) & (df.week > feature_week - 3)].groupby(['customer_id', ])[
        'price'].agg(['count', 'mean']).reset_index()
    feature_user2.columns = ['customer_id', 'user_last2_week_buy_number', 'user_last2_week_buy_price']

    ## 最近三周购买商品数
    feature_user3 = df[(df.week <= feature_week) & (df.week > feature_week - 9)].groupby(['customer_id', ])[
        'price'].agg(['count', 'mean']).reset_index()
    feature_user3.columns = ['customer_id', 'user_last3_week_buy_number', 'user_last3_week_buy_price']

    ## 最近一周商品数卖出数量
    feature_article1 = df[df.week.isin([feature_week])].groupby(['article_id'])['price'].agg(
        ['count', 'mean']).reset_index()
    feature_article1.columns = ['article_id', 'article_last1_week_buy_number', 'article_last1_week_buy_price']
    ## 最近两周商品数卖出数量
    feature_article2 = df[(df.week <= feature_week) & (df.week > feature_week - 3)].groupby(['article_id'])[
        'price'].agg(['count', 'mean']).reset_index()
    feature_article2.columns = ['article_id', 'article_last2_week_buy_number', 'article_last2_week_buy_price']
    ## 最近三周商品数卖出数量
    feature_article3 = df[(df.week <= feature_week) & (df.week > feature_week - 9)].groupby(['article_id'])[
        'price'].agg(['count', 'mean']).reset_index()
    feature_article3.columns = ['article_id', 'article_last3_week_buy_number', 'article_last3_week_buy_price']

    ###上一周购买次数
    feature1 = df[df.week.isin([feature_week])].groupby(['customer_id', 'article_id'])['price'].count().reset_index()
    feature1.columns = ['customer_id', 'article_id', 'last1_week_buy']
    ###上两周购买次数
    feature2 = df[df.week.isin([feature_week, feature_week - 1])].groupby(['customer_id', 'article_id'])[
        'price'].count().reset_index()
    feature2.columns = ['customer_id', 'article_id', 'last2_week_buy']
    ###上三周购买次数
    feature3 = \
    df[df.week.isin([feature_week, feature_week - 1, feature_week - 2])].groupby(['customer_id', 'article_id'])[
        'price'].count().reset_index()
    feature3.columns = ['customer_id', 'article_id', 'last3_week_buy']
    ##最近一次购买时间
    part = df[(df.week <= feature_week) & (df.week > feature_week - 9)]
    last_date_train = part.t_dat.max()
    part['day_lag'] = ((last_date_train - part['t_dat']) / np.timedelta64(1, 'D')).astype(int)
    feature4 = part.groupby(['customer_id', 'article_id'])['day_lag'].min().reset_index()
    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    feature4.columns = ['customer_id', 'article_id', 'recent_buy_day_gap']
    feature4['recent_buy_day_gap'] = a / np.sqrt(feature4['recent_buy_day_gap']) + \
                                     b * np.exp(-c * feature4['recent_buy_day_gap']) - d
    feature5 = get_max_dat(feature_week)

    traindataall = pd.merge(traindata01, feature_user1, how='left', on='customer_id')
    traindataall = pd.merge(traindataall, feature_user2, how='left', on='customer_id')
    traindataall = pd.merge(traindataall, feature_article1, how='left', on='article_id')
    traindataall = pd.merge(traindataall, feature_article2, how='left', on='article_id')
    traindataall = pd.merge(traindataall, feature1, how='left', on=['customer_id', 'article_id'])
    traindataall = pd.merge(traindataall, feature2, how='left', on=['customer_id', 'article_id'])
    traindataall = pd.merge(traindataall, feature3, how='left', on=['customer_id', 'article_id'])
    traindataall = pd.merge(traindataall, feature4, how='left', on=['customer_id', 'article_id'])
    traindataall = pd.merge(traindataall, feature5, how='left', on=['customer_id', 'article_id'])

    traindataall = pd.merge(traindataall, customers[['customer_id', 'fashion_news_frequency', 'age']],
                            how='left', on='customer_id')

    traindataall = pd.merge(traindataall, articles[['article_id', 'product_type_no',
                                                    'product_group_name', 'graphical_appearance_no',
                                                    'colour_group_code',
                                                    'perceived_colour_value_id',
                                                    'perceived_colour_master_id',
                                                    'department_no', 'index_code',
                                                    'index_group_no', 'section_no',
                                                    'garment_group_no']],
                            how='left', on='article_id')
    return traindataall


testdata_104 = gen_test_dataset(valset_104)
testdata_104 = gen_feature(testdata_104,104-1)

traindataall = []
# for ii in range(99,104):  ##260
for ii in range(103,104):
    print('-----')
    valset_103 = recall_rules_cmb(ii)
    ap12 = mapk(valset_103.article_id, valset_103.pred, return_apks=True)
    map12 = round(np.mean(ap12), 6)
    print(ii,map12)
    traindata_103 = gen_train_dataset(valset_103)
    traindata_103 = gen_feature(traindata_103,ii-1)
    traindataall.append(traindata_103)
traindataall = pd.concat(traindataall).reset_index(drop=True)

testdata_104.groupby('customer_id')['article_id'].count()
features = testdata_104.columns.tolist()[3:]

# 排序模型训练
from lightgbm.sklearn import LGBMRanker
from lightgbm import LGBMClassifier
modellist = []

traindataall.sort_values(by=['customer_id'], inplace=True)
# traindataall = pd.concat([traindata_100,traindata_101,traindata_102,traindata_103])
testdataall = pd.concat([testdata_104])

y = traindataall['purchased']

train_baskets = traindataall.groupby(['customer_id'])['article_id'].count().values


ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="dart",
    max_depth=7,
    n_estimators=300,
    importance_type='gain',
    verbose=10
)

ranker = ranker.fit(
        traindataall[features],
        y,
        group=train_baskets,
    )

# ranker = LGBMClassifier(random_state=42)
# ranker.fit(traindataall[features],y)
modellist.append(ranker)

def print_gbm_model_feature_importance(model):
    feature_names = features
    importances = list(model.feature_importances_)
    feature_importances = [(feature,round(importance)) for feature,importance in zip(feature_names,importances)]
    feature_importances = sorted(feature_importances,key=lambda x:x[1],reverse=True)
    return feature_importances
import matplotlib.pyplot as plt
import seaborn as sns

feature_importance_df = []
for model in [ranker]:
    feature_importance_df_ = print_gbm_model_feature_importance(model)
    feature_importance_df_ = pd.DataFrame(feature_importance_df_)
    feature_importance_df_.columns = ["feature", "importance"]
    feature_importance_df.append(feature_importance_df_)
feature_importance_df = pd.concat(feature_importance_df)
feature_importance_df = feature_importance_df.groupby('feature')['importance'].mean().reset_index()

sss = feature_importance_df.sort_values(by="importance", ascending=False).head(20)

plt.figure(figsize=(8,8))
sns.barplot(x="importance", y="feature",
            data=sss)
plt.title('LightGBM Features')
plt.tight_layout()


pred = 0
for ranker in modellist:
#     tmp = ranker.predict_proba(testdataall[features])[:,1]
    tmp = ranker.predict(testdataall[features])
    pred+=tmp/len(modellist)

testdataall['pred'] = pred

testdataall_pred = testdataall.sort_values(by='pred',ascending=False).reset_index(drop=True)
testdataall_pred = testdataall_pred.groupby('customer_id')['article_id'].apply(list)

testdataall_pred = testdataall_pred.reset_index()
testdataall_pred.columns = ['customer_id','pred_lgb']


valset_104_lgb = pd.merge(valset_104,testdataall_pred,how='left',on='customer_id')
valset_104_lgb['pred_lgb'] = [[] if str(xx)=='nan' else xx for xx in valset_104_lgb.pred_lgb]

#### 排序后的模型得分
ap12 = mapk(valset_104_lgb.article_id, valset_104_lgb.pred_lgb, return_apks=True)
map12 = round(np.mean(ap12), 6)
print(map12)
ap12 = mapk(valset_104_lgb.article_id, valset_104_lgb.pred, return_apks=True)
map12 = round(np.mean(ap12), 6)
print(map12)

###排序后验证得分从0.02569提高到了0.0280
### 把验证数据也加进到模型里面重新训练；使用99-104周的数据
traindataall = []
for ii in range(99,105):
    print('-----')
    valset_103 = recall_rules_cmb(ii)
    ap12 = mapk(valset_103.article_id, valset_103.pred, return_apks=True)
    map12 = round(np.mean(ap12), 6)
    print(ii,map12)
    traindata_103 = gen_train_dataset(valset_103)
    traindata_103 = gen_feature(traindata_103,ii-1)
    traindataall.append(traindata_103)
traindataall = pd.concat(traindataall).reset_index(drop=True)


from lightgbm.sklearn import LGBMRanker
from lightgbm import LGBMClassifier
modellist = []

traindataall.sort_values(by=['customer_id'], inplace=True)

y = traindataall['purchased']

train_baskets = traindataall.groupby(['customer_id'])['article_id'].count().values


ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="dart",
    max_depth=7,
    n_estimators=300,
    importance_type='gain',
    verbose=10
)

ranker = ranker.fit(
        traindataall[features],
        y,
        group=train_baskets,
    )

modellist.append(ranker)

del traindataall
import gc
gc.collect()



#-------------------## 生成最终的预测提交文件---------------------
def recall_rules(val_week):
    top_sell_person = df[df.week.isin([val_week-1,val_week-2,val_week-3])].groupby('customer_id')['article_id'].apply(lambda x:x.value_counts().index[:12].tolist())
    top_sell_person = top_sell_person.reset_index()
    top_sell_person.columns = ['customer_id','pred']
    valset = df[df.week==val_week].groupby('customer_id')['article_id'].apply(list).reset_index()
    valset = pd.merge(valset,top_sell_person,how='left',on='customer_id')
    top_sell = df[df.week.isin([val_week-1])].article_id.value_counts().index.tolist()[:12]
    valset['pred'] = [top_sell if str(xx)=='nan' else xx for xx in valset.pred]
    valset['pred'] = [xx+top_sell[:(12-len(xx))] if len(xx)<12 else xx for xx in valset.pred]
    return valset

test_week = 105
valset_105 = recall_rules_cmb(test_week,train_type=False)

valset_105['customer_id'] = [customer_id_inv_dict[xx] for xx in valset_105['customer_id']]

del valset_105['article_id']

valset_105['customer_id'].nunique()

testdata_105 = gen_test_dataset(valset_105)

testdata_105.customer_id.nunique()

testdata_105['customer_id'] = [customer_id_dict[xx] for xx in testdata_105['customer_id']]

testdata_105['article_id'] = testdata_105['article_id'].astype(int)
testdata_105.head()

testdata_105 = gen_feature(testdata_105,105-1)

from tqdm import tqdm
pred_test = 0
for ranker in modellist:
    tmp2 = []
    for ii in tqdm(range(int(len(testdata_105)/5000000)+1)):
        tmp = ranker.predict(testdata_105[features][ii*5000000:(ii+1)*5000000])
    #     tmp = ranker.predict_proba(testdata_105[features])[:,1]
        tmp2.append(tmp)
    tmp2 = np.concatenate(tmp2)
    pred_test+=tmp2/len(modellist)


testdata_105['article_id'] = testdata_105['article_id'].astype(int)
testdata_105['pred'] = pred_test

testdata_105 = testdata_105.sort_values(by='pred',ascending=False).reset_index(drop=True)
testdata_105 = testdata_105.groupby('customer_id').head(12)
testdata_105 = testdata_105.groupby('customer_id')['article_id'].apply(list)

testdata_105 = testdata_105.reset_index()
testdata_105.head()

testdata_105['customer_id_trans'] = \
        [customer_id_inv_dict[xx] if xx in customer_id_inv_dict else 'nan' for xx in testdata_105['customer_id']]


sub_save = testdata_105[['customer_id_trans','article_id']]
sub_save['article_id'] = [iter_to_str(xx) for xx in sub_save.article_id]
sub_save.columns = ['customer_id','prediction']

sub_ori = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')


sub_final = pd.merge(sub_ori[['customer_id']],sub_save[['customer_id', 'prediction']],how='left',on='customer_id')

sub_final[['customer_id', 'prediction']].to_csv('submission.csv', index=False)

