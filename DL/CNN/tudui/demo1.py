import time
import torch
import torchvision
from torch import optim, device
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

train_data = torchvision.datasets.CIFAR10(root="./data",
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True
                                          )

test_data = torchvision.datasets.CIFAR10(root="./data",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True
                                         )

# 训练集和测试集长度要相等
"""
shuffle打乱顺序
"""
train_dataloader=DataLoader(train_data, batch_size=64)
test_dataloader=DataLoader(test_data, batch_size=64)


# CIFRA 10 model

# 100 81 能抗过拟合
class Net_1(nn.Module):
    def __init__(self):
        super(Net_1,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128,128, 3,padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1,padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512*4*4,1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024,1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1,512*4*4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x

# 82 66 80epoch best
class Net_2(nn.Module):
    def __init__(self):
        super(Net_2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 实例化
net = Net_1()
net = net.to(device)

# 定义 Loss 函数和优化器
loss = nn.CrossEntropyLoss()
#optimizer = optim.SGD(self.parameters(),lr=0.01)
optimizer = optim.Adam(net.parameters(), lr=0.0001)




# 训练
for epoch in range(100):  # loop over the dataset multiple times

    # 训练开始
    net.train()

    timestart = time.time()

    running_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        # 每一次迭代的梯度都需要初始化
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        l = loss(outputs, labels)
        l.backward()
        optimizer.step()

        # 上面的代码是训练的核心部分，分别是前向传播+后向传播+优化,按照 PyTorch 的官方示例，就这样写。

        # print statistics
        running_loss += l.item()
        # print("i ",i)
        if i % 500 == 499:  # print every 500 mini-batches
            print('[%d, %5d] loss: %.4f' %
                  (epoch, i, running_loss / 500))
            running_loss = 0.0
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the %d tran images: %.3f %%' % (total,
                                                                              100.0 * correct / total))
            total = 0
            correct = 0

    print('epoch %d cost %3f sec' % (epoch, time.time() - timestart))

    # 测试
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.3f %%' % (
            100.0 * correct / total))

# print('Finished Training')




