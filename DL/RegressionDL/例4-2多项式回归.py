from itertools import count
import torch
import torch.autograd
import torch.nn.functional as F

n = 3  # 定义最高次项系数


def make_features(x):
    """构建一个[x,x^2,x^3]矩阵特征的实例"""
    x = x.unsqueeze(1)  # unqueeze(1)为增加一维
    return torch.cat([x ** i for i in range(1, n + 1)], 1)


# 定义真实函数 y = 1 + 2x + 3x^2 + 4x^3
W_target = torch.FloatTensor([2, 3, 4]).unsqueeze(1)
b_target = torch.FloatTensor([1])


def f(x):
    """模拟函数"""
    return x.mm(W_target) + b_target.item()


def get_batch(batch_size=32):
    """定义批"""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return x, y


# 定义模型
fc = torch.nn.Linear(W_target.size(0), 1)

for batch_idx in count(1):
    # 获取数据
    batch_x, batch_y = get_batch()

    # 重置梯度
    fc.zero_grad()

    # 前向传播
    output = F.smooth_l1_loss(fc(batch_x), batch_y)
    loss = output.item()

    # 后向传播
    output.backward()

    # 应用梯度下降算法
    for param in fc.parameters():
        param.data.add_(-0.1 * param.grad.data)

    # 终止条件
    if loss < 1e-3:
        break


def poly_desc(W, b):
    """建立描述的结果的字符串."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
print('==> Learned function:\t' + poly_desc(fc.weight.view(-1), fc.bias))
print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
