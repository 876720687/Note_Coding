import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
#确定数据、确定优先需要设置的值
lr = 0.15
gamma = 0
epochs = 10
bs = 128


#torch.manual_seed(420)
#X = torch.rand((50000,20),dtype=torch.float32) * 100 #要进行迭代了，增加样本数量
#y = torch.randint(low=0,high=3,size=(50000,1),dtype=torch.float32)
#data = TensorDataset(X,y)
#data_withbatch = DataLoader(data,batch_size=bs, shuffle = True)

#torch.manual_seed(420)
#X = torch.rand((50000,20),dtype=torch.float32) * 100 #要进行迭代了，增加样本数量
#y = torch.randint(low=0,high=3,size=(50000,1),dtype=torch.float32)
#data = TensorDataset(X,y)
#data_withbatch = DataLoader(data,batch_size=bs, shuffle = True)


# import torchvision
# import torchvision.transforms as transforms
# #初次运行时会下载，需要等待较长时间
# mnist = torchvision.datasets.FashionMNIST(
#     root='C:\Pythonwork\DEEP LEARNING\WEEK 3\Datasets\FashionMNIST'
#    , train=True
#    , download=True
#    , transform=transforms.ToTensor())
#
# len(mnist)
# # 查看特征张量
# mnist.data
# #这个张量结构看起来非常常规，可惜的是它与我们要输入到模型的数据结构有差异
#
# #查看标签
# mnist.targets
# #查看标签的类别
# mnist.classes
# #查看图像的模样
# import matplotlib.pyplot as plt
# plt.imshow(mnist[0][0].view((28, 28)).numpy());
# plt.imshow(mnist[1][0].view((28, 28)).numpy());
# #分割batch
# batchdata = DataLoader(mnist,batch_size=bs, shuffle = True)
# #总共多少个batch?
# len(batchdata)
# #查看会放入进行迭代的数据结构
# for x,y in batchdata:
#     print(x.shape)
#     print(y.shape)
#     break
# input_ = mnist.data[0].numel() #特征的数目，一般是第一维之外的所有维度相乘的数
# output_ = len(mnist.targets.unique()) #分类的数目
# #最好确认一下没有错误
# input_
# output_

#==================简洁代码====================
import torchvision
import torchvision.transforms as transforms
mnist = torchvision.datasets.FashionMNIST(root='./DL_Practice/data/FashionMNIST'
                                         , train=True
                                         , download=True
                                         , transform=transforms.ToTensor())

batchdata = DataLoader(mnist, batch_size=bs, shuffle=True)

input_ = mnist.data[0].numel()
output_ = len(mnist.targets.unique())


# 定义神经网路的架构
class Model(nn.Module):
    def __init__(self, in_features=10, out_features=2):
        super().__init__()
        # self.normalize = nn.BatchNorm2d(num_features=1)
        self.linear1 = nn.Linear(in_features, 128, bias=False)
        self.output = nn.Linear(128, out_features, bias=False)

    def forward(self, x):
        # x = self.normalize(x)
        x = x.view(-1, 28 * 28)
        # 需要对数据的结构进行一个改变，这里的“-1”代表，我不想算，请pytorch帮我计算
        sigma1 = torch.relu(self.linear1(x))
        z2 = self.output(sigma1)
        sigma2 = F.log_softmax(z2, dim=1)
        return sigma2


def fit(net, batchdata, lr=0.01, epochs=5, gamma=0):
    criterion = nn.NLLLoss()  # 定义损失函数
    opt = optim.SGD(net.parameters(), lr=lr, momentum=gamma)  # 定义优化算法
    correct = 0
    samples = 0
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(batchdata):
            y = y.view(x.shape[0])
            sigma = net.forward(x)
            loss = criterion(sigma, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            # 求解准确率
            yhat = torch.max(sigma, 1)[1]
            correct += torch.sum(yhat == y)
            samples += x.shape[0]

            if (batch_idx + 1) % 125 == 0 or batch_idx == len(batchdata) - 1:
                print('Epoch{}:[{}/{}({:.0f}%)]\tLoss:{:.6f}\t Accuracy:{: .3f}'.format(
                epoch + 1
                , samples
                , len(batchdata.dataset) * epochs
                , 100 * samples / (len(batchdata.dataset) * epochs)
                , loss.data.item()
                , float(correct * 100) / samples))

torch.manual_seed(420)
net = Model(in_features=input_, out_features=output_)
fit(net, batchdata, lr=lr, epochs=epochs, gamma=gamma)