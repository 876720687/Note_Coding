import numpy as np
import pandas as pd
import torch as t
from torch.autograd import Variable as var


file_path = r'../data/iris.csv'  #1.iris.csv文件的路径前要加字母r，因为在python字符串中\有转义的含义，所以我们需要采取一些方式使得\不被解读为转义字符
df_iris = pd.read_csv(file_path, sep=",", header="infer")
np_iris = df_iris.values
np.random.shuffle(np_iris)

def normalize(temp):
    temp = 2*(temp - np.mean(temp,axis = 0))/(np.max(temp,axis = 0)-np.min(temp,axis = 0))
    return(temp)

def convert2onehot(data):
    # covert data to onehot representation
    return pd.get_dummies(data)

xs = normalize(np_iris[:,1:5]).astype(np.double)
ys = convert2onehot(np_iris[:,-1]).values


xs = var(t.Tensor(xs))
ys = var(t.Tensor(ys))

class softmax_model(t.nn.Module):

    def __init__(self):
        super(softmax_model,self).__init__()
        self.linear1 = t.nn.Linear(4,64)
        self.relu = t.nn.ReLU()
        self.linear2 = t.nn.Linear(64,16)
        self.linear3 = t.nn.Linear(16,3)
        self.softmax = t.nn.Softmax()

        self.criterion = t.nn.MSELoss()
        self.opt = t.optim.SGD(self.parameters(),lr=0.6)
    def forward(self, input):
        y = self.linear1(input)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.relu(y)
        y = self.linear3(y)
        y = self.softmax(y)
        return y

model = softmax_model()
for e in range(6001):
    y_pre = model(xs[:90,:])

    loss = model.criterion(y_pre,ys[:90,:])
    if(e%200==0):
        print(e,loss.data)
    
    # Zero gradients
    model.opt.zero_grad()
    # perform backward pass
    loss.backward()
    # update weights
    model.opt.step()

result = (np.argmax(model(xs[90:,:]).data.numpy(),axis=1) == np.argmax(ys[90:,:].data.numpy(),axis=1))
print(np.sum(result)/60)
