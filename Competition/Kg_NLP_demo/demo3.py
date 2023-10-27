import torch
import time
from torch import autograd
#GPU加速
print(torch.__version__)
print(torch.cuda.is_available())

a=torch.randn(10000,1000)
b=torch.randn(1000,10000)

t0=time.time()
c=torch.matmul(a,b)
t1=time.time()
print(a.device,t1-t0,c.norm(2))


device=torch.device('cuda')
print(device)

a = a.to(device)
b = b.to(device)

t0 = time.time()
c = torch.matmul(a,b)
t2 = time.time()
print(a.device, t2-t0, c.norm(2))

