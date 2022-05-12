# EDA
# https://www.kaggle.com/code/vanguarde/h-m-eda-first-look 
# https://www.kaggle.com/code/ludovicocuoghi/h-m-sales-and-customers-deep-analysis

# Model
# https://www.kaggle.com/code/hiroshisakiyama/recommending-items-recently-bought/notebook

# With merging and cudf to acc
# https://www.kaggle.com/code/cdeotte/recommend-items-purchased-together-0-021/comments

# Essmeble
# https://www.kaggle.com/code/chaudhariharsh/lb-0-0237-h-m-ensembling-how-to-get-bronze


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

import sys # to find the memory usage

# you need to switch to GPU
import cudf
print('RAPIDS version',cudf.__version__)

pd.set_option('display.max_columns', None)

# Load Transactions, all the process below is to reduce memory usage
train = cudf.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')

# sys.getsizeof(train.customer_id)
train['customer_id'] = train['customer_id'].str[-16:].str.hex_to_int().astype('int64') # how to turn str into int?
train['article_id'] = train['article_id'].astype('int32') # to turn int64 into int32
train['t_dat'] = cudf.to_datetime(train['t_dat']) # turn string into datatime64

train = train[['t_dat','customer_id','article_id']]

train.to_parquet('train.pqt',index=False) # !!!

for i in train.columns:
    print(train[i].nunique())


# --------------starting to form the data we need

# -------------------------------Find Each Customer's Last Week of Purchases
tmp = train.groupby('customer_id')['t_dat'].max().reset_index() # we need to find the customer's last purchases time
tmp.columns = ['customer_id','max_dat']

train = train.merge(tmp,on=['customer_id'],how='left') # so is t_dat the first time of customer's purchase record? 

train['diff_dat'] = (train.max_dat - train.t_dat).dt.days

train = train.loc[train['diff_dat']<=6] # then we can find 500w in 3000w rows purchasing record as the recommendation row.



# (1)-----------------------------Recommend Most Often Previously Purchased Items in one week
tmp = train.groupby(['customer_id','article_id'])['t_dat'].count().reset_index()
tmp.columns = ['customer_id','article_id','ct']

train = train.merge(tmp,on=['customer_id','article_id'],how='left')

# train = train.sort_values(['ct','t_dat'],ascending=False) # sort values by two index

train = train.drop_duplicates(['customer_id','article_id']) # del the same name in the col

# train = train.sort_values(['ct','t_dat'],ascending=False) # which will cause resorting



#  (2) ---------------------------Recommend Items Purchased Together
# https://www.kaggle.com/code/cdeotte/customers-who-bought-this-frequently-buy-this/notebook
# use this to help us to merge cv info into our model
# So from this part we need to use features in cv to improve accuracy
# before running this we need to add infor from the picture

import pandas as pd, numpy as np
train = train.to_pandas()
pairs = np.load('../input/hm-test2/pairs_cudf_test2.npy',allow_pickle=True).item() # to get ans from the numpy, which is provided in another notebook
train['article_id2'] = train['article_id'].map(pairs) # i have no idea why article id2 can not show

# RECOMMENDATION OF PAIRED ITEMS, so the article id is the thing that customer have bought before but not recorded in the cv dataset.
train2 = train[['customer_id','article_id2']].copy()
train2 = train2.loc[train2.article_id2.notnull()]
train2 = train2.drop_duplicates(['customer_id','article_id2']) # unhashable type
train2 = train2.rename({'article_id2':'article_id'},axis=1)

# CONCATENATE PAIRED ITEM RECOMMENDATION AFTER PREVIOUS PURCHASED RECOMMENDATIONS
# by concat all the col together to get more feature
train = train[['customer_id','article_id']]
train = pd.concat([train,train2],axis=0,ignore_index=True)# concat train and train2 to train
train.article_id = train.article_id.astype('int32')
train = train.drop_duplicates(['customer_id','article_id'])

# CONVERT RECOMMENDATIONS INTO SINGLE STRING
# -------------------------So from this place we finally have the string of recommendation of each customer!!!
train.article_id = ' 0' + train.article_id.astype('str')

preds = cudf.DataFrame( train.groupby('customer_id').article_id.sum().reset_index() ) # preds always has one customer id each
preds.columns = ['customer_id','prediction']
preds.head()



# (3)---------------------------------Recommend Last Week's Most Popular Items
train = cudf.read_parquet('train.pqt') # ? what is that and where it from? the begin of this mission
train.t_dat = cudf.to_datetime(train.t_dat)
train = train.loc[train.t_dat >= cudf.to_datetime('2020-09-16')]
top12 = ' 0' + ' 0'.join(train.article_id.value_counts().to_pandas().index.astype('str')[:12])
print("Last week's top 12 popular items:")
print( top12 )




# --------------------------------------Submission
sub = cudf.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')
sub = sub[['customer_id']]# so the sample submission provides with customer id that need to be predicted
sub['customer_id_2'] = sub['customer_id'].str[-16:].str.hex_to_int().astype('int64')
sub = sub.merge(preds.rename({'customer_id':'customer_id_2'},axis=1),on='customer_id_2', how='left').fillna('')
del sub['customer_id_2']
sub.prediction = sub.prediction + top12 # we just find the most famous item and recommend it to the customer
sub.prediction = sub.prediction.str.strip()
sub.prediction = sub.prediction.str[:131]
sub.to_csv(f'submission.csv',index=False)
# sub.head()






















































