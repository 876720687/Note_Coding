import torch
from torch import nn


input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype = torch.float32)

# 为什么变成1155了
input = torch.reshape(input, (-1, 1, 5, 5))


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool(input)
        return output


tudui = Tudui()
output = tudui(input)
# output = nn.MaxPool2d(3, ceil_mode=True)
print(output)