import torch
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import nn

# 对一元线性回归进行cuda加速

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 5*x + 8 + torch.rand(x.size())
# 上面这行代码是制造出接近y=5x+8的数据集，后面加上torch.rand()函数制造噪音
 
# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1) # 输入和输出的维度都是1
    def forward(self, x):
        out = self.linear(x)
        return out
 
if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()
 
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
 
num_epochs = 1000
for epoch in range(num_epochs):
    # Convert numpy array to torch Variable with cuda acc
    if torch.cuda.is_available():
        inputs = Variable(x).cuda()
        target = Variable(y).cuda()
    else:
        inputs = Variable(x)
        target = Variable(y)

    # Forward + Backward + Optimize
    # 向前传播
    out = model(inputs)
    loss = criterion(out, target)
 
    # 向后传播
    optimizer.zero_grad() # 注意每次迭代都需要清零
    loss.backward()
    optimizer.step()
 
    if (epoch+1) %200 == 0:
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch+1, num_epochs, loss.item()))

model.eval()
if torch.cuda.is_available():
    predict = model(Variable(x).cuda())
    predict = predict.data.cpu().numpy()
else:
    predict = model(Variable(x))
    predict = predict.data.numpy()


plt.plot(x.numpy(), y.numpy(), 'ro', label='Original Data')
plt.plot(x.numpy(), predict, label='Fitting Line')
plt.show()
