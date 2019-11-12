import torch
import math
import numpy as np
from sklearn import datasets
import encoder
def psp_kernel(t, v0, tau, tau_s):
    """
    按照ppt 0025 第2页的K(t-t_i)计算
    """
    tt = t.float()
    tt[tt < 0] = 0
    return v0 * (torch.exp(-tt/tau) - torch.exp(-tt/tau_s))

def calculate_normal_v0(tau, tau_s):
    """
    v0要求使psp_kernel的最大值为1，通过求导计算出最值对应时间
    """
    t_max = (tau * tau_s * math.log(tau / tau_s)) / (tau - tau_s)
    v_max = math.exp(-t_max/tau) - math.exp(-t_max/tau_s)
    return 1 / v_max


class LIFNode:
    def __init__(self, tau, tau_s, v_rest, T, N, device='cpu'):
        self.tau = tau
        self.tau_s = tau_s
        self.v0 = calculate_normal_v0(tau, tau_s)
        self.v = 0
        self.v_rest = v_rest
        self.v_thr = 1 # v0已经归一化，所以阈值直接设置为1
        self.T = T # 运行时长
        self.N = N # 输入的数量
        self.device = device
        self.t = torch.arange(start=0, end=T, device=self.device).float().repeat([N, 1]) # shape=[N, T]
    def calculate_membrane_potentials(self, W, t_spike):
        """
        W和t_spike都是shape=[N]的tensor
        并行计算出一次仿真的所有结果，self.v是一个shape=[T]的tensor
        """
        self.v = torch.sum(W.view(-1, 1) * psp_kernel(self.t - t_spike.view(-1, 1), self.v0, self.tau, self.tau_s), dim=0)

"""
测试代码
    M = 100  # M组数据 一半正一半反
    N = 100  # 输入的数量
    T = 500
    lif_node = LIFNode(15.0, 15.0 / 4, 0, T, N, 'cuda:0')
    W = torch.rand(size=[N]).cuda()
    t_spike = torch.randint(low=0, high=T, size=[M, N]).float().cuda()
    learn_rate = 0.1
    train_times = 0

    while 1:
        m = np.random.randint(low=0, high=M)

        lif_node.calculate_membrane_potentials(W, t_spike[m])
        t_max = lif_node.v.argmax()
        if m < M/2:
            # 正例
            if lif_node.v[t_max] < lif_node.v_thr:
                # 正例，电压没有超过阈值
                W += learn_rate * psp_kernel(t_max - t_spike[m], lif_node.v0, lif_node.tau, lif_node.tau_s)

        else:
            if lif_node.v[t_max] > lif_node.v_thr:
                # 反例，电压超过阈值
                W -= learn_rate * psp_kernel(t_max - t_spike[m], lif_node.v0, lif_node.tau, lif_node.tau_s)

        if train_times % 128 == 0:
            print(train_times)

        if train_times % 16384 == 0:
            test_error_times = 0
            for m in range(M):
                lif_node.calculate_membrane_potentials(W, t_spike[m])
                t_max = lif_node.v.argmax()
                if m < M / 2:
                    # 正例
                    if lif_node.v[t_max] < lif_node.v_thr:
                        # 正例，电压没有超过阈值
                        test_error_times += 1

                else:
                    if lif_node.v[t_max] > lif_node.v_thr:
                        # 反例，电压超过阈值
                        test_error_times += 1

            print("错误率", test_error_times / M)

        train_times += 1
        if train_times == 1000000:
            break
"""


if __name__ == '__main__':
    iris = datasets.load_iris()  # 字典，iris.data为特征，iris.target为类别
    # print(iris['data'].shape)  # [150,4]
    # print(iris['target'].shape) # [150]
    x_train = torch.from_numpy(iris['data']).float().cuda()
    y_train = torch.from_numpy(iris['target']).cuda()
    enc_neuron_num = 512
    x_train_min = x_train.min(0)[0]
    x_train_max = x_train.max(0)[0]
    enc = encoder.GaussianEncoder(x_train_min[0], x_train_max[0], enc_neuron_num, 'cuda:0')

    T = 500
    lif_node = LIFNode(15.0, 15.0 / 4, 0, T, enc_neuron_num, 'cuda:0')
    W = torch.rand(size=[enc_neuron_num]).cuda()
    learn_rate = 0.1
    train_times = 0

    t_spike = torch.zeros(size=[150, enc_neuron_num], dtype=torch.float).cuda()
    # 生成脉冲
    for m in range(150):  # 编码过程非常慢
        print(m)
        t_spike[m] = enc.encode(x_train[m][0], T)[0]  # [1, enc_neuron_num]的tensor
    t_spike = t_spike.view(150, -1)


    while 1:
        m = np.random.randint(low=0, high=150)

        lif_node.calculate_membrane_potentials(W, t_spike[m])
        t_max = lif_node.v.argmax()
        if y_train[m] == 0:
            # 正例
            if lif_node.v[t_max] < lif_node.v_thr:
                # 正例，电压没有超过阈值
                W += learn_rate * psp_kernel(t_max - t_spike[m], lif_node.v0, lif_node.tau, lif_node.tau_s)

        else:
            if lif_node.v[t_max] > lif_node.v_thr:
                # 反例，电压超过阈值
                W -= learn_rate * psp_kernel(t_max - t_spike[m], lif_node.v0, lif_node.tau, lif_node.tau_s)

        if train_times % 128 == 0:
            print(train_times)

        if train_times % 16384 == 0:
            test_error_times = 0
            for m in range(150):
                lif_node.calculate_membrane_potentials(W, t_spike[m])
                t_max = lif_node.v.argmax()
                if y_train[m] == 0:
                    # 正例
                    if lif_node.v[t_max] < lif_node.v_thr:
                        # 正例，电压没有超过阈值
                        test_error_times += 1

                else:
                    if lif_node.v[t_max] > lif_node.v_thr:
                        # 反例，电压超过阈值
                        test_error_times += 1

            print("错误率", test_error_times / 150)

        train_times += 1
        if train_times == 1000000:
            break






