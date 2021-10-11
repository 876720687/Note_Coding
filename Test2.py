
import math
import numpy as np
from scipy.stats import chi2, f, t, norm
from matplotlib import pyplot as plt

pltchi2 = plt.figure()
# 生成x轴数据空间
x = np.linspace(0, 100, 100000).tolist()

# 画出自由度n, 20 至 60 的卡方分布概率密度函数图像
# ***** begin ***** #
y=[]
for i in range(len(x)):
    y.append(pow(i,2))
plt.plot(x,y)
plt.show()