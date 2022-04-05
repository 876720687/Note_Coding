import numpy as np
import pandas as pd

def mean(dataList):
    # 样本均值
    #********** Begin **********#
    return np.mean(dataList)


    #********** End **********#

def variance(dataList):
    # 样本方差
    #********** Begin **********#
    # return np.var(dataList)
    return np.var(dataList,ddof = 1)

    #********** End **********#


def K_origin_moment(dataList, k):
    # k 阶原点矩
    #********** Begin **********#
    sum=0
    for i in range(len(dataList)):
        sum += (1/len(dataList))*int(pow(dataList[i],k))
    return sum

def K_central_moment(dataList, k):
    # k 阶中心距
    #********** Begin **********#
    sum=0
    ave = np.mean(dataList)
    for i in range(len(dataList)):
        sum += (1/len(dataList))*int(pow(dataList[i]-ave,k))
    return sum