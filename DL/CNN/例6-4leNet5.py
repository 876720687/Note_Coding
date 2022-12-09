import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

show = ToPILImage()  # 可以把Tensor转成Image，方便可视化

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

## 数据加载以及预处理===================================================
# 第一次运行程序torchvision会自动下载CIFAR-10数据集，
# 大约163M，需花费一定的时间，
# 如果已经下载有CIFAR-10，可通过root参数指定root;如果没有下载，root指定下载数据的文件夹地址

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 训练集
trainset = tv.datasets.CIFAR10(
    root='.\data',
    train=True,
    download=True,
    transform=transform)

trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=0)
# 测试集
testset = tv.datasets.CIFAR10(
    root='.\data',
    train=False,
    download=True,
    transform=transform)
testloader = t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Dataset对象是一个数据集，可以按下标访问，返回形如(data, label)的数据。
(data, label) = trainset[100]
print(classes[label])
show((data + 1) / 2).resize((100, 100))


# # Dataloader是一个可迭代的对象，它将dataset返回的每一条数据拼接成一个batch，并提供多线程加速优化和数据打乱等操作。当程序对dataset的所有数据遍历完一遍之后，相应的对Dataloader也完成了一次迭代。
dataiter = iter(trainloader)
images, labels = dataiter.next() # 返回4张图片及标签
print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images+1)/2)).resize((400,100))

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size())
imshow(tv.utils.make_grid(images))
plt.show()#关掉图片才能往后继续算

##　定义网络===================================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

## 训练网络============================================================
import time

start_time = time.time()
# t.set_num_threads(4)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data  # 输入数据
        optimizer.zero_grad()  # 梯度清零
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' \
                  % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
end_time = time.time()
print('Training duration:', end_time - start_time)

##测试=================================================

# #一部分效果
dataiter = iter(testloader)
images, labels = dataiter.next()
print('True label:',' '.join('%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images/2 - 0.5)).resize((400,100))

outputs = net(images)
_, predicted = t.max(outputs.data,1)
print('Predicted label:',' '.join('%5s'%classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
with t.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = t.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
print('Accuracy in testset of 10000 images: %d %%' % (100 * correct / total))

