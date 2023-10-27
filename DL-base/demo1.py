# -*- coding: utf-8 -*- 
# @Time : 2022/12/12 17:03 
# @Author : YeMeng 
# @File : demo1.py 
# @contact: 876720687@qq.com

def func1():
    import torch

    # Define the model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1),
        torch.nn.Sigmoid()
    )

    # Define some random input and target tensors
    input = torch.randn(128, 10)
    target = torch.randint(0, 2, (128, 1), dtype=torch.float)

    # Compute the loss
    loss = torch.nn.BCEWithLogitsLoss()(model(input), target)

    # Print the loss
    print(loss)


def func2():
    import torch
    from torch import nn

    class MyCustomModule(nn.Module):
        def __init__(self):
            # Initialize the superclass (nn.Module)
            super().__init__()

            # Define the layers in your model, and initialize them
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.ReLU()
            self.layer3 = nn.Linear(20, 30)

        def forward(self, x):
            # Define the forward pass through the model
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x

    # Create an instance of your custom module
    model = MyCustomModule()

    # Generate some input data
    x = torch.randn(1, 10)

    # Use the model to make a prediction
    y_pred = model(x)

    print(y_pred)


# dataloader
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load the data and store it in memory
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                sample = (file_path, class_name)
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, class_name = self.samples[index]
        image = Image.open(file_path)
        if self.transform:
            image = self.transform(image)
        return image, class_name


from torch.utils.data import DataLoader

dataset = MyDataset(root_dir='./data/')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


#
# def func3():
#     pass


if __name__ == '__main__':
    # func1()
    func2()
