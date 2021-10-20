import numpy as np

# 问题1
C_matrix = np.array([[13,40,105,1],
            [81,6,132,7],
            [30,1,160,6]]) # 权重矩阵需要你自己定义

weight = np.array([0.5,0.2,0.1,0.2]) # 权重

t = 0.5 # 定义幂次系数

Regret_value = [-0.18,-0.23,0]

score_1 = pow(np.dot(weight,C_matrix.T),t) - Regret_value # 最终结果

# 问题2

p = [0.2,0.8] # 概率

delta = 0.6

C_matrix = np.array([[[20,3,9,3],
                      [41,22,53,17],
                      [58,46,119,14]],

                     [[94,49,132,2],
                      [33,29,61,4],
                      [9,1,29,3]]]) # 权重矩阵需要你自己定义

# 归一化
C_matrix_std = np.zeros((C_matrix.shape[0],C_matrix.shape[1],C_matrix.shape[2]))
for i in range(C_matrix.shape[0]): # 列归一化
    max_list = np.max(C_matrix[i], axis=0)
    min_list = np.min(C_matrix[i], axis=0)
    for j in range(C_matrix.shape[1]):
        for k in range(C_matrix.shape[2]):
            a = abs(max_list[j] - C_matrix[i][j][k]) # 取列最大值 - 对应值 的绝对值
            b = max_list[j] - min_list[j] # 取列最大值 - 取列最小值
            C_matrix_std[i][j][k] = a/b

    # norms = np.linalg.norm(C_matrix[i],axis=1)
    # C_matrix[i] = C_matrix[i]/norms

A_Number = C_matrix.shape[0] # 输入矩阵的个数

weight = np.array([0.1,0.6,0.2,0.1]) # 每个矩阵的权重都一样


ans = []
for i in range(A_Number):
    ans=np.concatenate((ans,np.dot(weight, C_matrix[i].T)),axis =0)
ans=ans.reshape(-1,C_matrix.shape[1]) # 这个是由你输入的矩阵的维度决定的(第i组*A_j)

ans_origin = ans # uaik

# temp = np.dot(p,ans)

for i in range(len(p)): # 对矩阵乘概率后进行更新
    ans[i]=p[i]*ans[i]



# 计算后悔度
a=np.max(ans_origin, axis=1) # 原始
a=a.reshape(a.shape[0],1) # 转置，需要扩展列
a=np.tile(a, (1,C_matrix[0].shape[0])) # 这个3之后要依据A的个数进行置换
b=ans
Regret_value = 1 - np.exp(np.multiply(-delta,a-b))


for i in range(ans.shape[0]):
    for j in range(ans.shape[1]):
        ans[i][j]=pow(ans[i][j],t)

score_2 = ans - Regret_value

# 判断哪个最好
for i in range(score_2.shape[0]): # 矩阵A组数
    flag = 0
    tag = 0
    for j in range(score_2.shape[1]):
        if score_2[i][j] > flag :
            flag = score_2[i][j]
            tag = j
    print("The best is A{number}".format(number = tag+1))

