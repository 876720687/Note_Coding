import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import pylab
im = Image.open('./dog.jpg').convert('L') # 读入一张灰度图的图片
im = np.array(im, dtype='float32') # 将其转换为一个矩阵
# 可视化图片
plt.imshow(im.astype('uint8'), cmap='gray')
pylab.show()
# 将图片矩阵转化为 pytorch tensor，并适配卷积输入的要求
im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))

pool1 = nn.MaxPool2d(2, 2)
print('before max pool, image shape: {} x {}'.format(im.shape[2], im.shape[3]))
small_im1 = pool1(Variable(im))
small_im1 = small_im1.data.squeeze().numpy()
print('after max pool, image shape: {} x {} '.format(small_im1.shape[0], small_im1.shape[1]))
plt.imshow(small_im1, cmap='gray')
pylab.show()