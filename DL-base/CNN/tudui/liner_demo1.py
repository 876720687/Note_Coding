import tensorflow
import torch
from torch.nn import functional as F

X = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]
                 , dtype=torch.float32)

torch.random.manual_seed(420)
# 实例化
output = torch.nn.Linear(2, 1)
# 连续实现分类
zhat = output(X)
sigma = F.sigmoid(zhat)
y = [int(x) for x in sigma>=0.5] # 在sigmoid外层嵌套实现分类
y1 = torch.sign(zhat)
y2 = F.relu(zhat) # F下的都被认为是对神经网络非常有用的函数
y3 = torch.tanh(zhat)
print(output.bias)
print(output.weight)

# softmax
z = torch.softmax()