import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 创建数据，修改
def creat_data(n):
    np.random.seed(0)
    X = 5*np.random.rand(n,1)
    y = np.sin(X).ravel()
    noise_num = (int)(n/5)
    y[::5] += 3*(0.5-np.random.rand(noise_num))
    return train_test_split(X,y,test_size=0.25,random_state=404)


def test_DecisionTreeRegressor(*data):
    X_train, X_test, y_train, y_test = data
    regr = DecisionTreeRegressor()
    regr.fit(X_train, y_train)

    print('Training Score:', regr.score(X_train, y_train))
    print('Testing Score:', regr.score(X_test, y_test))

    X = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    Y = regr.predict(X)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax = fig.subplot()
    # ax.scatter(X_train, y_train, label='train sample', c='g')
    # ax.scatter(X_test, y_test, label='test sample', c='r')
    # ax.plot(X, Y, label="predict_value", linewidth=2,alpha=0.5)
    ax.plot(X, Y)
    # ax.set_xlabel("data")
    # ax.set_ylabel("target")
    # ax.set_title("Decision Tree Regression")
    # ax.legend(framealpha=0.5)
    plt.show()





def test_DecisionTreeRegressor_splitter(*data):
    '''
    测试 DecisionTreeRegressor 预测性能随划分类型的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    splitters=['best','random']
    for splitter in splitters:
        regr = DecisionTreeRegressor(splitter=splitter)
        regr.fit(X_train, y_train)
        print("Splitter %s"%splitter)
        print("Training score:%f"%(regr.score(X_train,y_train)))
        print("Testing score:%f"%(regr.score(X_test,y_test)))




def test_DecisionTreeRegressor_depth(*data,maxdepth):
    '''
    测试 DecisionTreeRegressor 预测性能随  max_depth 的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的值、测试样本的值
    :param maxdepth: 一个整数，它作为 DecisionTreeRegressor 的 max_depth 参数
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    depths=np.arange(1,maxdepth)
    training_scores=[]
    testing_scores=[]
    for depth in depths:
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(X_train, y_train)
        training_scores.append(regr.score(X_train,y_train))
        testing_scores.append(regr.score(X_test,y_test))

    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(depths,training_scores,label="traing score")
    ax.plot(depths,testing_scores,label="testing score")
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5)
    plt.show()

if __name__ == '__main__':
    # 实例化过程
    X_train,X_test,y_train,y_test = creat_data(100)
    test_DecisionTreeRegressor(X_train,X_test,y_train,y_test)
    # 划分的类型对最终的模型预测的结果基本没有影响
    test_DecisionTreeRegressor_splitter(X_train,X_test,y_train,y_test)
    print(y_train.shape)

    '''
    由数据量可以知道，75的数据的最大划分是2^7=128，此时每一个都被分成了一类
    这就是为什么测试机的准确率下降了
    '''
    test_DecisionTreeRegressor_depth(X_train, X_test, y_train, y_test, maxdepth=10)