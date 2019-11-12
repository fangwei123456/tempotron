from matplotlib import pyplot
import torch
import numpy as np

def plot_spike(t_spike=torch.randint(low=0, high=10, size=[5]), T=10):
    """
    :param t_spike: shape=[N]的tensor,记录每一个神经元的脉冲发放时间
    :param T: 总时长
    """
    pyplot.figure()

    neural_spike = np.zeros(shape=[t_spike.shape[0], T])

    for i in range(t_spike.shape[0]):
        pyplot.subplot(T, 1, i+1)

        neural_spike[i][t_spike[i].item().__int__()] = 1
        pyplot.bar(np.arange(start=0, stop=T), neural_spike[i])
        pyplot.yticks([])
        if i < t_spike.shape[0] - 1:
            pyplot.xticks([])
        else:
            pyplot.xticks(np.arange(start=0, stop=T))

    pyplot.show()

if __name__ == '__main__':
    plot_spike()