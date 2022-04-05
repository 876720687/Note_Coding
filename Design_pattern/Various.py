# -*- coding: utf-8 -*-
"""
@描述：本模型利用Python的动画功能，简单模拟了传染病的传播过程；为了防止疫情的快速传播，
      要求完善该模型，设置相应参数，使之能够具有实际意义，为控制疫情提供参考。
@要求1：添加隔离措施参数，限制生成的每个点的下一次随机位置不能超过某个范围（半径为x的圆内），
       超过系统边界[0,1]的数值置为边界值；
@要求2：添加医院收治被感染者参数，在第x天可以收治n名病人，收治的病人用黄色点表示，
       收治后的病人不具有传染性；
@要求3：添加被感染者的死亡参数，被感染者n天内没有被医院收治，则设置为死亡状态，用黑色点表示；
@要求4：实时统计出相应的数据并打印输出，如当前被感染人数、收治人数、死亡人数等信息，并对比
       采取措施前后对疫情发展的影响，用图表展示；
@要求5：增加防疫措施指数，跟被感染者一定距离内的正常人会有感染风险但不一定感染
@要求6: 增加病毒潜伏期以及患者没有得到有效治疗的最长存活期
@要求7：医院里的患者治疗完毕后会出院，或者治疗失败导致死亡，出院后假定该病不会产生抗体之类的，仍有感染风险
@要求8：增加疫情情况曲线图，显示各日的感染人数、收治人数、死亡人数
@要求n：以上4个要求为模型必须提供的功能；另外，还可以根据实际情况自行设置参数，使模型更加
       完善，如数据点全部被感染后跳出循环；
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 随机样本点总数
SAMPLE_NUM = 500
# # 感染天数，应为2的倍数，表示[x,y]坐标
INFECTION_DAYS = 50
# 感染最小距离
INFECTION_DISTANCE = 0.015
# 生成点最远半径
Move_dis = 0.5

# 随机生成样本点
sample_data = np.random.rand(SAMPLE_NUM, INFECTION_DAYS * 2)

# 医院收治人员标志,未收治标0
Receive = np.zeros(SAMPLE_NUM)
# 死亡人员标志,未死亡标0
Dedper = np.zeros(SAMPLE_NUM)

# 增加防疫措施指数(表示在满足感染距离下的感染百分比)
EpLevel = 80

# 医院每天的收治人数
recen = 50
# 被医院收治的死亡率
depe = 0.05
# 病毒潜伏期
mgs = 7

# 患病时间(距离感染后的时间，即便被收治后达到最长存活时间根据死亡率可能存活或死亡，未被收治就..)
sufftime = np.zeros(SAMPLE_NUM)

# 患病后最长存活时间
ltime = 15
# 患者被医院收治的时间
hossuff = np.zeros(SAMPLE_NUM)


# 获取该点颜色的映射值
def gc(x):
    if Dedper[x] == 1:
        return 3  # 死亡
    elif Receive[x] == 1:
        return 2  # 目前在医院收治
    else:
        return sample_data[x, 0]


# 添加隔离措施
for i in range(SAMPLE_NUM):
    for j in range(INFECTION_DAYS):
        if j == 0:
            continue
        # 隔离 下次生成的点由之前的点决定
        x1 = sample_data[i][(j << 1) - 2]
        y1 = sample_data[i][(j << 1) - 1]
        tmpx = np.random.uniform(max(0, x1 - Move_dis), min(1, x1 + Move_dis))
        tmpl = np.sqrt(Move_dis * Move_dis - (x1 - tmpx) * (x1 - tmpx))
        tmpy = np.random.uniform(max(0, y1 - tmpl), min(1, y1 + tmpl))
        sample_data[i][j << 1] = tmpx
        sample_data[i][j << 1 | 1] = tmpy

# 第1列作为标志位：1表示已感染（红色），0表示未感染（绿色）
sample_flag = np.zeros(len(sample_data))
sample_data = np.insert(sample_data, 0, values=sample_flag, axis=1)
# 随机产生一个感染者，并将标志位设为0（红色）
seed = np.random.randint(0, SAMPLE_NUM, 1)
sample_data[seed, 0] = 1
sufftime[seed] = 1
# 创建一个绘图窗口对象
fig = plt.figure()
# 初始化画散列点函数
scat = plt.scatter(sample_data[:, 1], sample_data[:, 2], s=5, marker='o')

# 设置折线图
# 患病
x = np.array([i for i in range(0, INFECTION_DAYS + 1)])
yy1 = np.zeros(INFECTION_DAYS + 1)
yy1[0] = 0
# 收治
yy2 = np.zeros(INFECTION_DAYS + 1)
yy2[0] = 0
# 死亡
yy3 = np.zeros(INFECTION_DAYS + 1)
yy3[0] = 0

stepp = np.array([i for i in range(SAMPLE_NUM)])


def init():
    # 输出当前感染人数
    print(u"第{}天 => 被感染人数：{}".format(1, 1))
    # 根据不同的标签，显示当前感染者
    map_color = {1: 'r', 0: 'g'}
    color = list(map(lambda c: map_color[c], sample_data[:, 0]))
    scat.set_color(color)
    scat.set_offsets(sample_data[:, 1:3])


# 定义更新数据点的函数
def update(n):
    # 使用全局变量
    global sufftime
    global Dedper
    global Receive
    global hossuff
    global yy1
    global yy2
    global yy3
    # 外层循环为得到当前感染者，获取索引及当前感染者对应的位置坐标
    it = np.nditer(np.argwhere(sample_data == 1), flags=['f_index'])
    while not it.finished:
        if it[0] != 1:
            # 计算该感染者与其他点的距离，两者距离小于设定值表示有可能被感染
            # 两者距离小于设定值时有概率感染，设防疫措施x表示当满足条件时有x%感染，
            for i in range(0, SAMPLE_NUM):
                # 计算两点间距离，感染者坐标 - 其他点坐标
                p = sample_data[it[0], n:n + 2] - sample_data[i, n:n + 2]
                d = math.hypot(p[0], p[1])
                # 距离小于设定阈值表示感染同時运气不太好..，置0
                if d <= INFECTION_DISTANCE and np.random.randint(100) < EpLevel:
                    sample_data[i, 0] = 1
        it.iternext()
    # 开始收治
    # 根据已经超过潜伏期的人员按照患病时间先后收治
    cnt = recen
    # 先根据患病时间排序，保留索引
    temp = np.argsort(sufftime)
    # 取出时间大于潜伏期的
    temp = temp[sufftime[temp] > mgs]
    for item in temp:
        # 能够收治并且未被收治以及未死亡
        if cnt != 0 and Receive[item] == 0 and Dedper[item] == 0:
            cnt = cnt - 1
            Receive[item] = 1
        else:
            break
    # 被收治的患者就当就地隔离好了..，覆盖之后的坐标为当前坐标，出院后再做考虑
    for item in temp[Receive[temp] == 1]:
        for j in range(n, INFECTION_DAYS * 2 + 1):
            sample_data[item][j] = sample_data[item][j - 2]

    tempp1 = np.sum(sample_data[:, 0] == 1)
    tempp2 = np.sum(Receive[:] == 1)
    tempp3 = np.sum(Dedper[:] == 1)
    tday = n // 2 + 1
    yy1[tday] = tempp1
    yy2[tday] = tempp2
    yy3[tday] = tempp3
    # 输出当前感染人数
    print(u"第{}天 => 被感染人数：{} => 被收治总人数:{} =>死亡总人数:{}".format(tday, tempp1, tempp2, tempp3))
    # 根据不同的标签，显示当前感染者
    map_color = {0: 'g', 1: 'r', 2: 'yellow', 3: 'black'}
    color = list(map(lambda c: map_color[c], list(map(gc, stepp))))
    scat.set_color(color)
    scat.set_offsets(sample_data[:, n:n + 2])
    # 一天结束 患病天数达到最大天数且未被收治的患者死亡并将坐标保持不变
    # 先筛选未收治患者且未被标记死亡的患者
    Dlist = np.argwhere(Receive == 0)
    Dlist = Dlist[Dedper[Dlist] == 0]
    # 再筛选患病达最大天数
    Dlist = Dlist[sufftime[Dlist] >= ltime]
    for item in Dlist:
        # 死亡tag标上
        Dedper[item] = 1
        # 将之后的坐标全部标为当前值
        for j in range(n, INFECTION_DAYS * 2 + 1):
            sample_data[item][j] = sample_data[item][j - 2]

    # 患者患病时间增加一天，被收治的患者患病时间满三十天 运气好的出院 不好的去世...出院的患者对于之后的坐标进行重写，不排除被再次感染的可能
    sufftime = sufftime + sample_data[:, 0]
    # 被收治的患者住院时间增加一天
    hossuff = hossuff + np.where(Receive == 1, 1, 0)
    # 记录满足住院时长超过最长存活时长的患者
    temp_hos = np.argwhere(hossuff >= ltime)
    # 生死俄罗斯转盘....
    for i in temp_hos:
        Receive[i[0]] = 0  # 出院了
        hossuff[i[0]] = 0
        sample_data[i[0]][0] = 0
        if np.random.uniform(0, 1) < depe:
            # 即便得到合理的救治也没法的患者们
            Dedper[i[0]] = 1
        else:
            # 痊愈出院的患者 重写之后的坐标
            sufftime[i[0]] = 0
            # SAMPLE_NUM[i[0]][0]=0
            for j in range(n // 2 + 1, INFECTION_DAYS):
                x1 = sample_data[i[0]][(j << 1) - 2]
                y1 = sample_data[i[0]][(j << 1) - 1]
                tmpx = np.random.uniform(max(0, x1 - Move_dis), min(1, x1 + Move_dis))
                tmpl = np.sqrt(Move_dis * Move_dis - (x1 - tmpx) * (x1 - tmpx))
                tmpy = np.random.uniform(max(0, y1 - tmpl), min(1, y1 + tmpl))
                sample_data[i[0]][j << 1] = tmpx
                sample_data[i[0]][j << 1 | 1] = tmpy


# 动画显示

ani = animation.FuncAnimation(fig, update, range(3, (INFECTION_DAYS * 2) + 1, 2),
                              init_func=init, interval=500, repeat=0)

plt.show()

plt.figure()
plt.plot(x,yy1, label="Infected_num")#感染人数
plt.plot(x,yy2, label="Take_num")#收治人数
plt.plot(x,yy3, label="Dead_num")#死亡人数
plt.legend(loc='upper left')
plt.show()
