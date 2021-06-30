# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:47:17 2021

@author: admin
"""

'''

读取“stocks.csv”中的数据，然后对其中的所有年月（2017年1月至2019年12月）
获取绘制月K线所需的股票价格数据（月初开盘价、月末收盘价、月中最低价和最高价，
对应数据中的列下标分别为1、4、3、2），最后利用pyecharts绘制出股票价格的月K线图。

'''

import pandas as pd

df = pd.read_csv(r'C:\Users\admin\Desktop\hyx数据可视化作业\stocks.csv',header=None)

names = ['日期','月初开盘价','最高价','月中最低价','月末收盘价','5','6']

df.columns = names





