import torch
import math
import numpy as np
import painter
class GaussianEncoder:
    def __init__(self, x_min, x_max, M, device='cpu'):
        """
        脉冲神经网络原理及其应用 108页介绍的高斯调谐曲线编码方式
        """
        self.x_min = x_min  # 要编码的数据的最小值
        self.x_max = x_max
        self.M = M  # 编码使用的神经元数量
        self.mu = torch.zeros(size=[M], dtype=torch.float, device=device)
        self.sigma = 1 / 1.5 * (x_max - x_min) / (M - 2)
        for i in range(M):
            self.mu[i] = x_min + (2 * i - 3) / 2 * (x_max - x_min) / (M - 2)


    def encode(self, x, max_spite_time=10):
        """
        x是shape=[N]的tensor，M个神经元，x中的每个值都被编码成M个神经元的脉冲发放时间，也就是一个[M]的tensor
        因此，x的编码结果为shape=[N, M]的tensor，第j行表示的是x_j的编码结果
        记第i个高斯函数为f_i，则高斯函数作用后结果应该为
        f_0(x_0), f_1(x_0), ...
        f_0(x_1), f_1(x_1),...
        """
        ret = x.repeat([self.M, 1])
        """
        [x0, x1, x2, ...
         x0, x1, x2, ...]
        """
        for i in range(self.M):
            ret[i] = torch.exp(-torch.pow(ret[i] - self.mu[i], 2) / 2 / self.sigma**2)
        """
        [f_0(x0), f_0(x1), ...
         f_1(x0), f_1(x1), ...]
        接下来进行取整，函数值从[1,0]对应脉冲发放时间[0,max_spite_time]，计算时会取整
        认为脉冲发放时间大于max_spite_time*9/10的不会导致激活，设置成inf
        """
        ret = -max_spite_time * ret + max_spite_time
        ret = torch.round(ret)
        ret[ret > max_spite_time * 9 / 10] = 10000
        return ret.t()  # x的编码结果为shape=[N, M]的tensor，第j行表示的是x_j的编码结果。返回的dtype=float32








if __name__ == '__main__':

    x = torch.rand(size=[1])
    print(x)
    ge = GaussianEncoder(0, 1, 4)
    ex = ge.encode(x)
    print(ex)
    painter.plot_spike(ex.squeeze(), 4)


    exit()