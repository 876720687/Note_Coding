
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:06:39 2021

@author: 北湾
"""

import os
import json
import gc

from tqdm import tqdm_notebook
from tqdm import tqdm
import lightgbm as lgb
import catboost as cbt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler as std
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import f1_score
import time
import datetime 
from datetime import datetime, timedelta
import gc
from scipy.signal import hilbert
# from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
import scipy.spatial.distance as dist
from collections import Counter 
from statistics import mode 
import warnings
warnings.filterwarnings("ignore")
import json 
import math
from itertools import product
import ast
from sklearn.model_selection import train_test_split #数据分隔出训练集和验证集
import lightgbm as lgb
import numpy as np 
import pandas as pd
#导入精度和召回
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import lightgbm as lgb

import seaborn as sns
pd.set_option('display.max_columns',None)
# %matplotlib inline

# 读入数据
train_data = pd.read_csv(r'E:\CODE\ProgrammingProgect\pythoncoding\基于数据分析的金融风控预测（毕业设计）\baseline\train.csv',encoding = "gbk")

test_data = pd.read_csv(r'E:\CODE\ProgrammingProgect\pythoncoding\基于数据分析的金融风控预测（毕业设计）\baseline\testA.csv',encoding = "gbk")

target=train_data['isDefault']             #label
train_data=train_data.drop(['isDefault'],axis=1)

data=pd.concat([train_data,test_data])  #合并数据

'''
是直接把训练集和测试集先合并到一起，然后统一处理其中的数据。
可以很明显的看到，这个数据集中有三种不同类型的数据变量。
我们需要对这三种不同类型的数据分别进行处理。
train_data (800000,46)
test_data (200000,46)
target (800000,) 这个是01标签
data (1000000,46)
从变量说明中我们可以知道只有在训练集中。给了80万个结果。
而对于测试集的结果并没有给我们而是通过测试的方式。来评价我们的模型。
'''

data.info()
data.isnull().sum() #查看失值

'''
通过上面两个语句。我们可以知道数据的信息和内部的缺失值情况。
下面利三个列表来存储三种不同的数据类型。
然后通过两个for循环。第一个找到类别变量。第二个区分离散变量和连续变量。
最终找到。五个类别变量。离散变量只有八个。连续变量有33个。
说明金融风控里面大多数的变量，都是连续型变量。

'''

#object的变量  ==> objectList
#numerical的变量==>classList
#连续变量 ==>numericalList
objectList=[] # 5
classList=[] # 8
numericalList=[] # 33

for i in train_data.columns:
    if train_data[i].dtype=='O':
        objectList.append(i)
for i in list(train_data.select_dtypes(exclude=['object']).columns):
    temp=train_data[i].unique()
    if len(temp)<=10:
        classList.append(i)
    else:
        numericalList.append(i)

'''
下面的三句话找出了哪些变量,有缺失值.然后对缺失变量进行填补。
miss_fea里面存储的就是存在缺失值的变量名称。
然后将这些变量依据它的类别进行了分类，主要是因为不同类别变量对缺失值的处理方式不同。
对于某些唯一性的数据。或者一些信息性的数据。可以选择将其标为零。

'''
info=pd.DataFrame(data.isnull().sum())
info=info[info[0] !=0]
miss_fea=info.index

miss_objectList=[i for i in miss_fea if i in objectList]
miss_classList=[i for i in miss_fea if i in classList]
miss_numericalList=[i for i in miss_fea if i in numericalList]

# employmentLength 是就业时间 没有的填补为0
data['employmentLength'] = data['employmentLength'].fillna(0)

data['n11'] = data['n11'].fillna(0)
data['n12'] = data['n12'].fillna(0)

data['employmentTitle']=data['employmentTitle'].fillna(data['employmentTitle'].mode()[0])  #就业职称
data['postCode']=data['postCode'].fillna(data['postCode'].mode()[0])  #借款人在贷款申请中提供的邮政编码的前3位数字
data['dti']=data['dti'].fillna(data['postCode'].mean())            #债务收入比
data['pubRecBankruptcies']=data['pubRecBankruptcies'].fillna(data['pubRecBankruptcies'].mean()) #公开记录清除的数量
data['revolUtil']=data['revolUtil'].fillna(data['revolUtil'].mean())                       #循环额度利用率
data['title']=data['title'].fillna(data['title'].mode()[0])            #借款人提供的贷款名称
# 匿名变量
NoNameList=[i for i in miss_numericalList if i.startswith("n")]
for i in NoNameList:
    data[i]=data[i].fillna(data[i].mode()[0])
'''
到这个地方为止，就对三类不同类型变量的缺失值都进行了填补。
接下来的工作就是进行变量的类别转换，转换成计算机能够处理的类型。
上面代码每一空行的两侧都代表两种不同的类型。
这个地方处理完毕之后data里面的数据其实就已经发生了改变。

这个地方你可以输出一下data,所有的数据个数都已经变成100万了。
'''

'''
下面对object里面时间格式的变量进行了处理。同时还定义了一个函数。将处理完毕的。变量数值转化成了int类型。
处理完毕之后，工作年限这个变量的值就变成了0~10之间的数值。
用同样的方式对所有的object变量进行处理。并对分析意义不大的变量直接删除。删除的时候同时对测试集训练集还有组合起来的数据总集都删除。
对于grade和subgrade 这两个类似于成绩的ABCD等级，过了一个LabelEncoder的库来实现到数值的转换。
也就是编码选择
'''

#object 变量处理
data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
data['employmentLength'].replace('< 1 year', '0 years', inplace=True)
data['employmentLength'].replace('0', '0 years', inplace=True)

def employmentLength_to_int(s):
    s=str(s)
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])
    
data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)

data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))
#issuedata 时间对分析意义不大 删除
data=data.drop(['issueDate'],axis=1)
train_data=train_data.drop(['issueDate'],axis=1)
test_data=test_data.drop(['issueDate'],axis=1)

# 使用LabelEncoder对等级变量进行处理
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['grade']=le.fit_transform(data['grade'])
data['subGrade']=le.fit_transform(data['subGrade'])

'''
到这个地方为止。所有不可被可视化的object类型的变量全部被转化成了数值类型。
只有到这个时候才能可视化显示。所有的变量特征。才能够更加直观的看到。
于是接下来我们先把所有的变量，显示出来看一看效果。
col表示的是列。row表示的是行
下面的这个代码运行时间比较长，谨慎运行。是能够成功运行并且显示的。
然后下面那一行的代码就是通过图片里面观察变量的分布情况。来确定是否需要保留这些变量。
并将那些觉得没有意义的变量给删除了，被删除的变量就是名称都被放入到了dropList当中
到删除了变量为止，最终的data显示成为了 (1000000, 37)
而且下面那句话让classList从8个的数量更新到了五个，这个更新是什么意思呢？就是将离散值过少的分布过少的都给删除了，这边限定的是离散值至少是需要大于十个的才能够被留下来
因此classList 最终变成了 ['term', 'grade', 'homeOwnership', 'verificationStatus', 'initialListStatus']

'''


#  以下是自定义的一些特征，用于衡量用户价值和创利能力
data['avg_income'] = data['annualIncome'] / data['employmentLength']
data['total_income'] = data['annualIncome'] * data['employmentLength']
data['avg_loanAmnt'] = data['loanAmnt'] / data['term']
data['mean_interestRate'] = data['interestRate'] / data['term']
data['all_installment'] = data['installment'] * data['term']

data['rest_money_rate'] = data['avg_loanAmnt'] / (data['annualIncome'] + 0.1)  # 287个收入为0
data['rest_money'] = data['annualIncome'] - data['avg_loanAmnt']

data['closeAcc'] = data['totalAcc'] - data['openAcc']
data['ficoRange_mean'] = (data['ficoRangeHigh'] + data['ficoRangeLow']) / 2
# 这个del是删除。
del data['ficoRangeHigh'], data['ficoRangeLow']

data['rest_pubRec'] = data['pubRec'] - data['pubRecBankruptcies']
data['rest_Revol'] = data['loanAmnt'] - data['revolBal']
# data['dis_time'] = data['issueDate_year'] - (2020 - data['earliesCreditLine_year'])



img_cols=4
img_rows=len(data.columns)
plt.figure(figsize=(4*6,4*img_rows))
i=1
for col in data.columns:
    ax=plt.subplot(img_rows,img_cols,i)
    ax=sns.kdeplot(data[:80000][col],color="Red",shade=True)
    ax=sns.kdeplot(data[80000:][col],color="Blue",shade=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax=ax.legend(["train","test"])
    i+=1
plt.show()

# 这个地方只是利用了变量的分布状况就对变量进行了删减 是否合理!?


dropList=['id','ficoRangeHigh','applicationType','policyCode','n3','n11','n12','n13']
data.drop(dropList,axis=1,inplace=True)

classList=[i for i in data.columns if len(data[i].value_counts())<=10]



'''
到这个地方为止将有效变量都删选了出来，然后需要对每个有效变量内部的数据进行筛选
就是需要处理其中的异常值。
为什么需要重新从data里面切割数据放入到train_data当中呢？因为前面的所有操作都是对data的操作，因此到这个地方又需要重新开始对训练集和测试集分开进行操作了
而且处理的仅仅是训练集的异常值
当然，需要干的第一件事情是查看每个变量的异常值状况。这个时候箱型图就可以胜任这个工作。
如果存在过多的异常值肯定要将异常值筛除
另外，当数据倾向于正态分布的时候，机器学习的准确率会有所提高。
因此，另外还需要检测的内容就是，这些变量是否符合正态分布？

'''

# 异常值处理,这里只是对异常值进行了显示
train_data=data[:800000]
column=train_data.columns.tolist()
img_cols=6
img_rows=len(column)
plt.figure(figsize=(4*6,4*len(column)))
i=1
for col in data.columns:
    plt.subplot(img_rows,img_cols,i)
    sns.boxplot(train_data[col],orient="v",width=0.5)
    plt.ylabel(col,fontsize=15)
    i+=1
plt.show()

'''
train_data中另外创建了一列 label 用于存放target中的值，这个时候train_data从(800000,37)变成了(800000,38)
同样在classList中也多出了这一列
而且numlist是第一次出现？一共有32个变量。不同于前面定义的三种类型，因为这个时候object类型变量已经完全的转化了。
因此总共的变量其实也只有两种类型了。也就是连续型变量和离散型变量。
numList 32
classList 6
但是最终的结果显示并没有数值类型的变量能够符合正态分布，因此对这些数据都需要做另外的处理才能够提升足够的准确度

'''

import numpy as np
from scipy.stats import kstest
train_data['label']=target
classList=['term', 'grade', 'homeOwnership', 'verificationStatus', 'initialListStatus','label']
numList=[i for i in train_data.columns if i not in classList]
# 这个需要删除就是上面dropList里面的内容，这部分内容不会对结果产生影响
# xuyaoshanchu = ['id','ficoRangeHigh','ficoRangeHigh','applicationType','policyCode','n3','n11','n12','n13']
# numList = [i for i in numList if i not in xuyaoshanchu]
'''
不知道为啥不用这样，只需要执行那两个图片的显示，然后数据的格式就会自动转化过来
毒瘤数据
'''
# 如果数据服从正太分布 则利用正态分布处理 
for i in numList:
    print(kstest(data[i], 'norm', (data[i].mean(), data[i].std())))
#都不符合正太分布
#pvalue>0.05 为正太分布

percentile=pd.DataFrame()
percentile['columns']=numList

dropList=[]
count=0
for i in tqdm(numList):
    count+=1
    deg=train_data[i]
    mean = np.mean(deg)
    var = np.var(deg)
    percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
    Q1 = percentile[0]#上四分位数
    Q3 = percentile[2]#下四分位数
    IQR = Q3 - Q1#四分位距
    ulim = Q3 + 1.5*IQR#上限 非异常范围内的最大值
    llim = Q1 - 1.5*IQR#下限 非异常范围内的最小值
    new_deg = []
    for i in range(len(deg)):
        if(llim<deg[i] and deg[i]<ulim):
            new_deg.append(deg[i])
    droppre=((len(deg)-len(new_deg))/len(deg))*100
    dropList.append(droppre)
'''
这一串代码干的事情
创建了个pd.DataFrame的类，里面有这些内容：
Empty DataFrame
Columns: []
Index: []
而且像代码这样，可以一次性塞入很多的列，如果有数据会直接变成一个可以使用的DataFrame结构的数据表？
这个四分卫是？
主要获得了一个new_deg 还有个 dropList
然后还有下面这个代码块的作用又是什么？
修改了显示，然后将数据都放入到之前定义的percentile当中，并对第一列的内容作了map操作？
这个是不是减小了变量之间的相关性关系？！
然后放入到了一个judgeList 27 当中
然后接下来的内容是对数据进行正态化删除异常值然后反归一化还原数据
还想知道iloc这个东西是干了什么,哦哦，是取了0:38列的所有行
最后那句话还给它输出了csv的文件方便作为其他软件的接口？！

'''

dropList=[('%.4f' % i) for i in dropList]
percentile=pd.DataFrame([numList,dropList]).T
percentile[1]=percentile[1].map(lambda x:-1 if str(x)=='nan' else float(x))
judgeList=list(percentile[percentile[1]<=10][0])


train_data_1=train_data.copy()

#先数据正太化，再利用3σ原则剔除异常数据，反归一化即可还原数据
#先数据正太化，再利用3σ原则剔除异常数据，反归一化即可还原数据
stdsc=StandardScaler()
drop_index=[]
for i in tqdm(numList):
    new_i="zheng_"+i
    train_data_1[new_i]=stdsc.fit_transform(train_data_1[i].values.reshape(-1,1))
    data_std = np.std(train_data_1[new_i])
    data_mean = np.mean(train_data_1[new_i])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    train_data_1=train_data_1.drop(train_data_1[(train_data_1[new_i])>ulim].index)
    train_data_1=train_data_1.drop(train_data_1[(train_data_1[new_i])<llim].index)
    
train_data_2=train_data_1.iloc[:,:38]
# train_data_2.to_csv('正态分布训练集.csv',index=0)
# 下面这句话还有bug啊！否则会自动生成一个序列
# train_data_2.to_csv(r"E:\CODE\ProgrammingProgect\pythoncoding\基于数据分析的金融风控预测（毕业设计）\baseline\正态分布训练集.csv")
train_data_2.to_csv(r"E:\CODE\ProgrammingProgect\pythoncoding\基于数据分析的金融风控预测（毕业设计）\baseline\正态分布训练集.csv",index = 0)

'''
到这个地方为止所有的数据清洗的工作就完成了，接下来就可以对模型方面的内容进行继续的处理了
将正态化清洗完毕的数据存储起来(这个数据可以直接在后续内容当中使用)

将80w往后的data数据作为测试集，并存储起来.将正态化最终处理完毕的数据作为训练集
取出训练集中的label，并删除，然后又对原来的数据进行拼接。

为啥会有下面这句话呢？因为最终被处理完毕的数据都被重新存储下来了，现在要做的只是读取，就不需要另外重新从头清洗一遍了

而且因为反归一化了所以数据看起来还是和原来一样正常
但是后面用的数据应该都用train_data_zt 和 data的后20w作为测试集

'''
train_data_2 = pd.read_csv(r"E:\CODE\ProgrammingProgect\pythoncoding\基于数据分析的金融风控预测（毕业设计）\baseline\正态分布训练集.csv",encoding = "gbk")




'''
这个地方虽然从原始处理好的数据当中获得测试集，但是后来data又马上被覆盖了
下面这两行的代码。保证了测试集的数据独立性。使得程序不用再重头运行。
而且这两句像不用再运行了呀。
'''

# test_data = data[800000:]
# test_data.to_csv(r"E:\CODE\ProgrammingProgect\pythoncoding\基于数据分析的金融风控预测（毕业设计）\baseline\缺失值测试集.csv",index = 0)
test_data = pd.read_csv(r"E:\CODE\ProgrammingProgect\pythoncoding\基于数据分析的金融风控预测（毕业设计）\baseline\缺失值测试集.csv")




train_data_zt = train_data_2

label = train_data_zt['label']
train_data_zt = train_data_zt.drop('label',axis=1)

# data=pd.concat([train_data_zt,test_data])


x_train_gbdt = train_data_zt

# datattt = x_train_gbdt

y_train_gbdt = label

# 下面这两句话我都没有运行。
# datattt['label'] = y_train_gbdt
# # 下面这句话有什么用？
# datattt = datattt.reset_index(drop=True)

'''
这个地方最关键的一个点。就是获得了x和y。
然后就可以尝试使用不同模型来进行训练。
有些时候都是简单的模型和复杂的模型一起进行尝试的。当简单的识别效果不好。才会尝试使用复杂的模型。
将训练结果比较好的模型留下来。然后进行进一步细致的调整，才有了下面这个内容。

'''

# train_val_x = x_train_gbdt
# train_val_y = y_train_gbdt

# model={}
# model['rfc']=RandomForestClassifier()
# model['gdbt']=GradientBoostingClassifier()
# model['cart']=DecisionTreeClassifier()
# model['knn']=KNeighborsClassifier()
# model['lr']=LogisticRegression()
# # model['bayes']= MultinomialNB() 
# for i in model:
#     model[i].fit(train_val_x,train_val_y)
#     score=cross_val_score(model[i],train_val_x,train_val_y,cv=5,scoring='roc_auc')
#     print('%s的auc为：%.3f'%(i,score.mean()))




import lightgbm
def select_by_lgb(train_data,train_label,random_state=2020,n_splits=5,metric='auc',num_round=10000,early_stopping_rounds=200):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = train_data.columns
    fold=0
    for train_idx, val_idx in kfold.split(train_data):
        random_state+=1
        train_x = train_data.loc[train_idx]
        train_y = train_label.loc[train_idx]
        test_x = train_data.loc[val_idx]
        test_y = train_label.loc[val_idx]
        clf=lightgbm
        train_matrix=clf.Dataset(train_x,label=train_y)
        test_matrix=clf.Dataset(test_x,label=test_y)
       
        params={
                'boosting_type': 'gbdt',  
                'objective': 'binary',
                'learning_rate': 0.1,
                'metric': metric,
                'seed': 2020,
                'nthread':-1 }
        
        model = clf.train(
            params,
            train_matrix,
            num_round,valid_sets = test_matrix,
            early_stopping_rounds = early_stopping_rounds)
        
        feature_importances['fold_{}'.format(fold + 1)] = model.feature_importance()
        fold+=1
    feature_importances['averge']=feature_importances[['fold_{}'.format(i) for i in range(1,n_splits+1)]].mean(axis=1)
    return feature_importances
        
feature_importances=select_by_lgb(x_train_gbdt,y_train_gbdt)

feature_importances['averge']=feature_importances[['fold_{}'.format(i) for i in range(1,6)]].mean(axis=1)
'''
到这个地方为止可以说的内容就已经很多了！

'''
# ##################################################
'''
XGBoost贝叶斯调参
这一部分代码有一个很明显的缺陷。
就是像这种网格搜索。他的变量不能太多。否则会超出内存的限制。
因此下面这一部分代码并没有成功运行。这里只是做一个示范的作用。

'''
# #x_train_gbdt,y_train_gbdt test_data
# from xgboost import XGBClassifier
# import xgboost as xgb
# from sklearn.model_selection import GridSearchCV

# parameters = {
#               'max_depth': [5, 10, 15, 20, 25],
#               'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
#               'n_estimators': [500, 1000, 2000, 3000, 5000],
#               'min_child_weight': [0, 2, 5, 10, 20],
#               'max_delta_step': [0, 0.2, 0.6, 1, 2],
#               'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
#               'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
#               'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
#               'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
#               'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]

# }

# xlf = xgb.XGBClassifier(max_depth=10,
#             learning_rate=0.1,
#             n_estimators=2000,
#             silent=True,
#             objective='binary:logistic',
#             nthread=-1,
#             gamma=0,
#             min_child_weight=1,
#             max_delta_step=0,
#             subsample=0.85,
#             colsample_bytree=0.7,
#             colsample_bylevel=1,
#             reg_alpha=0,
#             reg_lambda=1,
#             scale_pos_weight=1,
#             seed=1440,
#             missing=None)
# # 有了gridsearch我们便不需要fit函数
# gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='roc_auc', cv=5,n_jobs=-1)
# # 下面这句话爆内存了？！
# # gsearch.fit(x_train_gbdt, y_train_gbdt)

# print("Best score: %0.3f" % gsearch.best_score_)
# print("Best parameters set:")
# best_parameters = gsearch.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))
# ##################################################


'''
我们在这里采用另外一种自动调参的方式：lgb贝叶斯优化

csdn参考文章

https://blog.csdn.net/qq_42283960/article/details/88317003
'''
from bayes_opt import BayesianOptimization
train_x, val_x, train_y, val_y = train_test_split(x_train_gbdt,y_train_gbdt, test_size = 0.3, random_state = 5)
'''
train_test_split()函数是用来随机划分样本数据为训练集和测试集的，当然也可以人为的切片划分。

优点：随机客观的划分数据，减少人为因素

完整模板：

train_X,test_X,train_y,test_y = train_test_split(train_data,train_target,test_size=0.3,random_state=5)

参数解释：

train_data：待划分样本数据

train_target：待划分样本数据的结果（标签）

test_size：测试数据占样本数据的比例，若整数则样本数量

random_state：设置随机数种子，保证每次都是同一个随机数。若为0或不填，则每次得到数据都不一样
'''

def LGB_bayesian(
    num_leaves,  # int
    min_data_in_leaf,  # int
    learning_rate,
    min_sum_hessian_in_leaf,    # int  
    feature_fraction,
    lambda_l1,
    lambda_l2,
    min_gain_to_split,
    max_depth):
    
    # LightGBM expects next three parameters need to be integer. So we make them integer
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)

    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int

    param = {
        'num_leaves': num_leaves,
        'max_bin': 63,
        'min_data_in_leaf': min_data_in_leaf,
        'learning_rate': learning_rate,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        
        'feature_fraction': feature_fraction,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'min_gain_to_split': min_gain_to_split,
        'max_depth': max_depth,
        'save_binary': True, 
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'boost_from_average': False,   

    }    
# 函数到这个地方为止，上面的内容全部都是模板。
# 下面的这两个语句进行了修改。
    xg_train = lgb.Dataset(train_x,
                           label=train_y
                           )
    xg_valid = lgb.Dataset(val_x,
                           label=val_y
                           )   

    num_round = 5000
    
    clf = lgb.train(param, xg_train, num_round, valid_sets = [xg_valid], verbose_eval=250, early_stopping_rounds = 50)
    
    predictions = clf.predict(val_x, num_iteration=clf.best_iteration)   

# 
    score = roc_auc_score(val_y, predictions)
    
    
    return score
'''
贝叶斯优化函数很多都是模板。主要修改的内容有：xg_train和xg_valid
然后还有下面的score评分方式（其实这个也是模板。）
相当于主要修改的内容就是LightGBM定义了的trainng和validation数据集。
https://blog.csdn.net/qq_42283960/article/details/88317003


'''

# Bounded region of parameter space
# 参数空间的有界区域
bounds_LGB = {
    'num_leaves': (5, 20), 
    'min_data_in_leaf': (5, 20),  
    'learning_rate': (0.01, 0.3),
    'min_sum_hessian_in_leaf': (0.00001, 0.01),    
    'feature_fraction': (0.05, 0.5),
    'lambda_l1': (0, 5.0), 
    'lambda_l2': (0, 5.0), 
    'min_gain_to_split': (0, 1.0),
    'max_depth':(3,15),
}

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=13)

# 
init_points = 5
n_iter = 30
print('-' * 130)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


# 查看调参数据，依据这个设定下面的probe内容
LGB_BO.max['params']

LGB_BO.probe(
    params={'feature_fraction': 0.403, 
            'lambda_l1': 1.23, 
            'lambda_l2': 0.5436, 
            'learning_rate': 0.0264, 
            'max_depth': 12.647, 
            'min_data_in_leaf': 11.07, 
            'min_gain_to_split': 0.4694, 
            'min_sum_hessian_in_leaf': 0.00736, 
            'num_leaves':14.752},
    lazy = True, # 
)

# 然后再次运行
LGB_BO.maximize(init_points=0, n_iter=0) # remember no init_points or n_iter

# 运行完毕查看运行状况
for i, res in enumerate(LGB_BO.res):
    print("Iteration {}: \n\t{}".format(i, res))

# 查看最准确模型结果以及其设定参数值（其实就是上面设定的内容）
LGB_BO.max['target']

LGB_BO.max['params']

# 到这个地方位置优化和最佳状态都显示出来了

# 
param_lgb = {
        'num_leaves': int(LGB_BO.max['params']['num_leaves']), # remember to int here
        'max_bin': 63,
        'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']), # remember to int here
        'learning_rate': LGB_BO.max['params']['learning_rate'],
        'min_sum_hessian_in_leaf': LGB_BO.max['params']['min_sum_hessian_in_leaf'],
        'bagging_fraction': 1.0, 
        'bagging_freq': 5, 
        'feature_fraction': LGB_BO.max['params']['feature_fraction'],
        'lambda_l1': LGB_BO.max['params']['lambda_l1'],
        'lambda_l2': LGB_BO.max['params']['lambda_l2'],
        'min_gain_to_split': LGB_BO.max['params']['min_gain_to_split'],
        'max_depth': int(LGB_BO.max['params']['max_depth']), # remember to int here
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
        'force_row_wise':True
    }


# dropList = ['grade','subGrade','employmentLength','earliesCreditLine']
# test_data.drop(dropList,axis=1,inplace=True)
#x_train_gbdt,y_train_gbdt test_data
# 这个地方利用了五折交叉验证！
nfold = 5
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)
oof = np.zeros(len(x_train_gbdt))
predictions = np.zeros((len(test_data),nfold))

i = 1
for train_index, valid_index in skf.split(x_train_gbdt, y_train_gbdt):
    print("\nfold {}".format(i))
    xg_train = lgb.Dataset(x_train_gbdt.iloc[train_index].values,
                           label=y_train_gbdt[train_index].values,
                           )
    xg_valid = lgb.Dataset(x_train_gbdt.iloc[valid_index].values,
                           label=y_train_gbdt.iloc[valid_index].values,
                           )   

    
    clf = lgb.train(param_lgb, xg_train, 5000, valid_sets = [xg_valid], verbose_eval=250, early_stopping_rounds = 50)
    oof[valid_index] = clf.predict(x_train_gbdt.iloc[valid_index].values, num_iteration=clf.best_iteration) 
    
    predictions[:,i-1] += clf.predict(test_data, num_iteration=clf.best_iteration)
    i = i + 1

print("\n\nCV AUC: {:<0.2f}".format(roc_auc_score(y_train_gbdt, oof)))


result=pd.DataFrame(predictions)
result['averge']=result[[0,1,2,3,4]].mean(axis=1)
result


# # --------------------------------------------------------------------------#
# # catboost 的尝试
# train_x, val_x, train_y, val_y = train_test_split(
#     x_train_gbdt,
#     y_train_gbdt, 
#     test_size = 0.3, 
#     random_state = 5
# )

# '''
# data_x = x_train_gbdt
# data_y = y_train_gbdt
# '''
# #
# col=['grade','subGrade','employmentTitle','homeOwnership','verificationStatus','purpose','postCode','regionCode',
#      'initialListStatus']
# for i in x_train_gbdt.columns:
#     if i in col:
#         x_train_gbdt[i] = x_train_gbdt[i].astype('str')
# for i in test_data.columns:
#     if i in col:
#         test_data[i] = test_data[i].astype('str')

# # catboost的特性，只能识别字符类型和数值类型的数据，因此在这个地方出现了转换
# from catboost import CatBoostClassifier
# model=CatBoostClassifier(
#             loss_function="Logloss",
#             eval_metric="AUC",
#             task_type="CPU",
#             learning_rate=0.1,
#             iterations=500,
#             random_seed=2020,
#             od_type="Iter",
#             depth=7)

# answers = []
# mean_score = 0
# n_folds = 5
# sk = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2019)
# for train, test in sk.split(x_train_gbdt, y_train_gbdt):
#     x_train = x_train_gbdt.iloc[train]
#     y_train = y_train_gbdt.iloc[train]
#     x_test = x_train_gbdt.iloc[test]
#     y_test = y_train_gbdt.iloc[test]
    
#     clf = model.fit(
#         x_train,y_train, 
#         eval_set=(x_test,y_test),
#         verbose=500,
#         cat_features=col)
    
#     yy_pred_valid=clf.predict(x_test)
    
#     print('cat验证的auc:{}'.format(roc_auc_score(y_test, yy_pred_valid)))
    
#     mean_score += roc_auc_score(y_test, yy_pred_valid) / n_folds
    
#     y_pred_valid = clf.predict(test_data,prediction_type='Probability')[:,-1]
    
#     answers.append(y_pred_valid)

# print('mean valAuc:{}'.format(mean_score))
# cat_pre=sum(answers)/n_folds

# # sub['isDefault']=cat_pre
# # sub.to_csv('金融预测.csv',index=False) 










