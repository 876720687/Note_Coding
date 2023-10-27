#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%% 导入本章所需要的模块
## 导入本章所需要的模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models




#%% 导入从GPU训练好的网络，并在CPU上使用
## 1:定义网络结构

## ResidualBlock残差块的网络结构
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        ## channels:b表示要输入的feature map 数量
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1)
        )

    def forward(self, x):
        return F.relu(self.conv(x) + x)


## 定义图像转换网络
class ImfwNet(nn.Module):
    def __init__(self):
        super(ImfwNet, self).__init__()
        self.downsample = nn.Sequential(
            nn.ReflectionPad2d(padding=4),##使用边界反射填充
            nn.Conv2d(3,32,kernel_size=9,stride=1),
            nn.InstanceNorm2d(32,affine=True),## 在像素值上做归一化
            nn.ReLU(),  ## 3*256*256->32*256*256
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(32,64,kernel_size=3,stride=2),
            nn.InstanceNorm2d(64,affine=True),
            nn.ReLU(),  ## 32*256*256 -> 64*128*128
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(64,128,kernel_size=3,stride=2),
            nn.InstanceNorm2d(128,affine=True),
            nn.ReLU(),  ## 64*128*128 -> 128*64*64
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.unsample = nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.InstanceNorm2d(64,affine=True),
            nn.ReLU(),  ## 128*64*64->64*128*128
            nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.InstanceNorm2d(32,affine=True),
            nn.ReLU(),  ## 64*128*128->32*256*256
            nn.ConvTranspose2d(32,3,kernel_size=9,stride=1,padding=4),## 32*256*256->3*256*256; 
        )
    def forward(self,x):
        x = self.downsample(x) ## 输入像素值－2.1～2.7之间
        x = self.res_blocks(x)
        x = self.unsample(x) ## 输出像素值－2.1～2.7之间
        return x

#fwnet = ImfwNet()
#fwnet


#%% 定义辅助函数，用语可视化图像

## 定义一个读取风格图像或内容图像的函数，并且将图像进行必要转化
def load_image(img_path,shape=None):
    image = Image.open(img_path)
    size = image.size
    ## 如果指定了图像的尺寸，就将图像转化为shape指定的尺寸
    if shape is not None:
        size = shape
    ## 使用transforms将图像转化为张量，并进行标准化
    in_transform = transforms.Compose(
        [transforms.Resize(size), # 图像尺寸变换,
         transforms.ToTensor(), # 数组转化为张量
         ## 图像进行标准化
         transforms.Normalize((0.485, 0.456, 0.406), 
                              (0.229, 0.224, 0.225))])
    # 使用图像的RGB通道，并且添加batch纬度
    image = in_transform(image)[:3,:,:].unsqueeze(dim=0)   
    return image

# 定义一个将标准化后的图像转化为便于利用matplotlib可视化的函数
def im_convert(tensor):
    """ 
    将[1, c, h, w]纬度的张量转化为[ h, w,c]的数组
    因为张量进行了表转化，所以要进行标准化逆变换
    """
    tensor = tensor.cpu() ## 数据转换为CPU
    image = tensor.data.numpy().squeeze() # 去处batch纬度数据
    image = image.transpose(1,2,0) ## 置换数组的纬度[c,h,w]->[h,w,c]
    ## 进行标准化的逆操作
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1) ##  将图像的取值剪切到0～1之间
    return image


#%% 定义辅助函数，用语可视化图像
# 读取内容图像
content= load_image("data/chap9/tar58.png",shape = (256,256))
print("content shape:",content.shape)
## 可视化图像
plt.figure()
plt.imshow(im_convert(content))
plt.show()


#%% 导入训练好的GPU网路
## 导入网络
device = torch.device('cpu')
fwnet = ImfwNet()
fwnet.load_state_dict(torch.load("data/chap9/imfwnet_dict.pkl", map_location=device))

transform_content = fwnet(content)
## 可视化图像
plt.figure()
plt.imshow(im_convert(transform_content))
plt.show()



