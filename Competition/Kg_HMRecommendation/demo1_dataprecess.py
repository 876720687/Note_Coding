# -*- coding: utf-8 -*- 
# @Time : 2022/10/26 16:02 
# @Author : YeMeng 
# @File : demo1_dataprecess.py 
# @contact: 876720687@qq.com

from tools import *
pd.set_option('display.max_columns', None)

transactions = pd.read_csv('../input/transactions_train.csv', dtype={"article_id": "str"}, nrows=10000)
customers = pd.read_csv('../input/customers.csv',nrows=10000)
articles = pd.read_csv('../input/articles.csv', dtype={"article_id": "str"},nrows=10000)

transactions['customer_id'] = customer_hex_id_to_int(transactions['customer_id'])
transactions.t_dat = pd.to_datetime(transactions.t_dat, format='%Y-%m-%d')
transactions['week'] = 104 - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7

# Let's do something about the `article_id` (both here and on `articles`) and let's take a closer look at `price`, `sales_channel_id` and `week`.
transactions.article_id = article_id_str_to_int(transactions.article_id)
articles.article_id = article_id_str_to_int(articles.article_id)

transactions.week = transactions.week.astype('int8')
transactions.sales_channel_id = transactions.sales_channel_id.astype('int8')
transactions.price = transactions.price.astype('float32')

# Well, this stuff will be getting merged with our transactions df at some point, so I guess we can also make this smaller and easier to work with down the road.

customers.customer_id = customer_hex_id_to_int(customers.customer_id)
for col in ['FN', 'Active', 'age']:
    customers[col].fillna(-1, inplace=True)
    customers[col] = customers[col].astype('int8')

customers.club_member_status = Categorize().fit_transform(customers[['club_member_status']]).club_member_status
customers.postal_code = Categorize().fit_transform(customers[['postal_code']]).postal_code
customers.fashion_news_frequency = Categorize().fit_transform(customers[['fashion_news_frequency']]).fashion_news_frequency

for col in articles.columns:
    if articles[col].dtype == 'object':
        articles[col] = Categorize().fit_transform(articles[[col]])[col]

for col in articles.columns:
    if articles[col].dtype == 'int64':
        articles[col] = articles[col].astype('int32')

# And this concludes our raw data preparation step! Let’s now write everything back to disk.
transactions.sort_values(['t_dat', 'customer_id'], inplace=True)

transactions.to_parquet('../input/hm-parquets-of-datasets/transactions_train.parquet')
customers.to_parquet('../input/hm-parquets-of-datasets/customers.parquet')
articles.to_parquet('../input/hm-parquets-of-datasets/articles.parquet')