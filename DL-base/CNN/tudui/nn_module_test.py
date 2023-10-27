import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor()
                                          , download=True
                                          )
train_dataloader=DataLoader(train_data, batch_size=64)


class Model(nn.Module):
    """
    对卷积、池化和激活函数进行了测试
    """
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, (3, 3))
        # self.conv2 = nn.Conv2d(20, 20, 5)

        self.maxpool1 = nn.MaxPool2d(3, ceil_mode=True)

        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # return F.relu(self.conv2(x))
        # x = self.conv1(x)

        # x = self.maxpool1(x)

        x = self.sigmoid1(x)
        return x



model = Model()
writer = SummaryWriter("logs_sigmoid")

step = 0
for data in train_dataloader:
    imgs, targets = data
    output = model(imgs)
    # print(imgs.shape)
    # print(output.shape)

    writer.add_images("input", imgs, step)

    # 卷积因为会改变channel因此需要reshape，池化并不会产生channel的改变
    # output = torch.reshape(output, (-1, 3, 30, 30))

    writer.add_images("output", output, step)

    step = step + 1

writer.close()