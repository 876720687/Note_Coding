import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from fbprophet import Prophet

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


df = pd.read_excel(r'C:\Users\Administrator\Desktop\中国高校大数据挑战赛20211028-1101\2021年中国高校大数据挑战赛赛题\赛题\赛题A\附件1：赛题A数据.xlsx')
df['小区PDCP流量']=df['小区PDCP层所发送的下行数据的总吞吐量比特']+df['小区PDCP层所接收到的上行数据的总吞吐量比特']

kpi = []

# a=['小区内的平均用户数']
# b=['小区PDCP层所发送的下行数据的总吞吐量比特','小区PDCP层所接收到的上行数据的总吞吐量比特']
# c=['平均激活用户数']


def Show_Pic(data):
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['figure.figsize'] = (25, 4.0)  # set figure size

    # # 设置时间戳索引
    # data['时间'] = pd.to_datetime(data['时间'])
    # data.set_index("时间", inplace=True)
    # values = data.values
    #
    # # 保证所有数据都是float32类型
    # values = values.astype('float32')
    # # 变量归一化
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled = scaler.fit_transform(values)

    # data[['小区内的平均用户数', '小区PDCP流量', '平均激活用户数']].plot()
    data['小区PDCP流量'].plot()
    plt.grid(True, linestyle="-", color="green", linewidth="0.5")
    plt.legend()
    plt.title('Pic')

    # plt.gca().spines["top"].set_alpha(0.0)
    # plt.gca().spines["bottom"].set_alpha(0.3)
    # plt.gca().spines["right"].set_alpha(0.0)
    # plt.gca().spines["left"].set_alpha(0.3)

    plt.show()

# 可以显示但是不能归一化
def Show_all(data):
    import pandas_profiling
    pfr = pandas_profiling.ProfileReport(data)
    pfr.to_file("./example.html")


# df['时间'] = pd.to_datetime(df['时间'])
# df.set_index("时间", inplace=True)

a = df[(df['小区编号'] == 26019014)]
a.reset_index(drop=True, inplace=True) # 行标重新排列

# # 小区合并，刚好会处理好时间
# a_1 = df[(df['小区编号'] == 26019014)]
# a_1.reset_index(drop=True, inplace=True) # 行标重新排列
# for i in range(len(df['小区编号'].unique())):
#     b = df[(df['小区编号'] == df['小区编号'].unique()[i])]
#     b.reset_index(drop=True, inplace=True) # 行标重新排列
#     a_1.iloc[:, 4:].update(a_1.iloc[:, 4:]+b.iloc[:, 4:])
#     # a_1.reset_index(drop=True, inplace=True)  # 行标重新排列


a_1 = df[(df['小区编号'] == 26019014)].iloc[:, 4:]
a_1.reset_index(drop=True, inplace=True) # 行标重新排列
for i in range(1,len(df['小区编号'].unique())):
    b = df[(df['小区编号'] == df['小区编号'].unique()[i])].iloc[:, 4:]
    b.reset_index(drop=True, inplace=True) # 行标重新排列
    a_1.update(a_1 + b)
    # a_1.reset_index(drop=True, inplace=True)  # 行标重新排列

a_1=a_1/58
a=pd.concat([a['时间'],a_1],axis = 1)




# ['小区内的平均用户数','小区PDCP层所发送的下行数据的总吞吐量比特'
# ,'小区PDCP层所接收到的上行数据的总吞吐量比特','平均激活用户数']
# b=a_std[['时间','小区PDCP流量']] # Std之后异常值从29变为了24
b=a[['时间','小区PDCP流量']]
b['时间'] = pd.to_datetime(b['时间']).copy()
b.columns = ['ds','y']




"""
Prophet（）的主要参数：

    #设置跟随性： changepoint_prior_scale=0.05 值越大，拟合的跟随性越好，可能会过拟合
    #设置置信区间：interval_width=0.8（默认值）,值越小，上下线的带宽越小。
    #指定预测类型： growth='linear'或growth = "logistic" ，默认应该是linear。
    #马尔科夫蒙特卡洛取样（MCMC）： mcmc_samples=0,会计算很慢。距离意义不清楚
    #设置寻找突变点的比例：changepoint_range=0.9 默认从数据的前90%中寻找异常数据。预测这个正弦曲线，如果不设置changepoint_range=1，预测的结果是不对的，不知道为什么。

make_future_dataframe( ）的主要参数：

    #periods 周期，一般是根据实际意义确定，重点：后续预测的长度是一个周期的长度。
    #freq 我见的有‘MS‘、H、M ，预测sin，要设置H ，个人理解数据如果变化很快，要用H

其他的内置参数：

    yearly_seasonality 是年规律拟合，Prophet模型会描绘出以一年为单位的数据规律，后面的部分会有图示；同理于参数 weekly_seasonality，daily_seasonality。
    
    n_changepoints 是预设转折点数量
    
    changepoint_range 是设定转折点可以存在的范围，.1表示存在于历史数据的前十分之一，.5表示在历史数据的前半部分，其余同理。
    
    changepoint_prior_scale 是设置模型对转折点拟合的灵敏度，值越高越灵活。
    
    changepoints=[] 是指定转折点的具体位置
    
    yearly_seasonality 是年的拟合度，值越高越灵活，同时可以选择True和False来设定是否进行年度的拟合。同理与weekly_seasonality和daily_seasonality。
    
    holidays_prior_scale 是假期的拟合度，同样值越高越灵活，同时前提是你需要有假期信息的加入。
    
    seasonality_mode=‘multiplicative’ 是模型学习的方式，默认情况下为加性的，如果如上所示来设置，则是乘性的(multiplicative)。
"""
# 什么也没改动的预测
def Origin_Show():
    # 拟合模型
    m = Prophet(interval_width=0.95
                # , growth='logistic'
                ,changepoint_prior_scale=0.1
                # ,weekly_seasonality=True
                )
    m.fit(b)

    # 构建待预测日期数据框，periods 代表除历史数据的日期外再往后推
    future = m.make_future_dataframe(periods=24 * 3, freq='H')
    # future.tail()
    forecast = m.predict(future)

    m.plot(forecast)
    m.plot_components(forecast)
    plt.show()

# 第三问评价
def Mape(y,y_hat):
    return np.mean(np.abs((y - y_hat) / y)) * 100


# 第三问输出
df['小区编号'].unique()
c = b['y'][:72]
count_Num=0
for i in sorted(df['小区编号'].unique()):
    a = df[(df['小区编号'] == i)]
    a.reset_index(drop=True, inplace=True)  # 行标重新排列
    b = a[['时间', '平均激活用户数']]
    # b['时间'] = pd.to_datetime(b['时间']).copy()
    b.columns = ['ds', 'y']
    m = Prophet(interval_width=0.95
                ,changepoint_prior_scale=0.1
                ,weekly_seasonality=True
                )
    m.fit(b)
    future = m.make_future_dataframe(periods=24 * 3, freq='H')
    forecast = m.predict(future)
    count_Num += Mape(b['y'],forecast['yhat'][:696])
    tt = forecast['yhat'][696:]
    tt.reset_index(drop=True, inplace=True)  # 行标重新排列
    c = pd.concat([c, tt], axis=1)
    # forecast['yhat'][696:].to_excel('1.xlsx',index=False)

print(count_Num/58)
c.to_excel('平均激活用户数.xlsx',index=False)

tt= df['小区编号'].unique()


## 对该数据建立一个时间序列模型
np.random.seed(1234)  ## 设置随机数种子
model = Prophet(daily_seasonality = True
                ,weekly_seasonality= True
                ,seasonality_mode = 'multiplicative'
                ,interval_width = 0.95   ## 获取95%的置信区间
                )
# model = Prophet(interval_width=0.95)
model = model.fit(b)     # 使用数据拟合模型
forecast = model.predict(b)  # 使用模型对数据进行预测
forecast["y"] = b["y"].reset_index(drop = True)
forecast[["ds","y","yhat","yhat_lower","yhat_upper"]].head()


# 根据模型预测值的置信区间"yhat_lower"和"yhat_upper"判断样本是否为异常值
def outlier_detection(forecast):
    index = np.where((forecast["y"] <= forecast["yhat_lower"])|
                     (forecast["y"] >= forecast["yhat_upper"])
                     ,True # 1
                     ,False # 0
                     ) # 这个地方转化为 1 0 就是二分类问题了
    return index


outlier_index = outlier_detection(forecast)
outlier_df = b[outlier_index] # 异常数据
print("异常值的数量为:",np.sum(outlier_index))

# 可视化异常值的结果
fig, ax = plt.subplots()

# 可视化预测值
forecast.plot(x = "ds",y = "yhat",style = "b-",figsize=(14,7),
              label = "预测值",ax=ax)
# 可视化出置信区间
ax.fill_between(forecast["ds"].values, forecast["yhat_lower"],
                forecast["yhat_upper"],color='b',alpha=.2,
                label = "95%置信区间")
forecast.plot(kind = "scatter",x = "ds",y = "y",c = "k",
              s = 20,label = "原始数据",ax = ax)
# 可视化出异常值的点
outlier_df.plot(x = "ds",y = "y",style = "rs",ax = ax,
                label = "异常值")
plt.legend(loc = 2)
plt.grid()
plt.title("时间序列异常值检测结果")
plt.show()


# # 清洗数据进行可视化！不能归一化！
# tt = []
# for i in a.columns:
#     tt.append(len(list(a[i].unique())))
#
# drop_label = ['时间']
# # drop_label = []
# for i in range(len(tt)):
#     if tt[i] <= 5:
#         drop_label.append(a.columns[i])
#
# scaler = MinMaxScaler()
# # nn = scaler.fit_transform(a.iloc[:24, :])
# a_time = a['时间']
# a_values = a.iloc[:,1:]# 保留时间
# a_values_std=pd.DataFrame(StandardScaler().fit_transform(a_values),columns=a_values.columns)
# a_std = pd.concat([ a_time ,a_values_std ],axis=1)
# # Show_all(pd.DataFrame(nn))
#
#
# # 寻找重要参数！
# a = a.drop(columns=drop_label, axis=1)
# y1 = a[['小区PDCP流量','小区内的平均用户数','平均激活用户数','小区PDCP层所发送的下行数据的总吞吐量比特','小区PDCP层所接收到的上行数据的总吞吐量比特']]
# a = a.drop(columns=['小区PDCP流量','小区内的平均用户数','平均激活用户数','小区PDCP层所发送的下行数据的总吞吐量比特','小区PDCP层所接收到的上行数据的总吞吐量比特'],axis=1)






