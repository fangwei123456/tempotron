from sklearn import datasets
import encoder
import node
import torch
import numpy as np
# 增加门控学习
if __name__ == '__main__':
    iris = datasets.load_iris()  # 字典，iris.data为特征，iris.target为类别
    # print(iris['data'].shape)  # [150,4]
    # print(iris['target'].shape) # [150]
    x_train = torch.from_numpy(iris['data']).float().cuda()
    y_train = torch.from_numpy(iris['target']).cuda()
    # 4个特征，需要4组编码器
    enc_layer = []
    x_train_min = x_train.min(0)[0]
    x_train_max = x_train.max(0)[0]
    print(x_train_min)
    print(x_train_max)
    enc_neuron_num = 512  # 编码1个特征使用的神经元数量
    for i in range(4):
        enc_layer.append(encoder.GaussianEncoder(x_train_min[i], x_train_max[i], enc_neuron_num, 'cuda:0'))
    # 共有3类，需要3个tempotron
    dec_layer = []
    for i in range(3):
        dec_layer.append(node.LIFNode(tau=15.0, tau_s=15.0/4, v_rest=0, T=500, N=enc_neuron_num*4, device='cuda:0'))  # 由于是全连接，因此enc_layer的所有元素都与这个节点相连

    W = torch.rand(size=[3, enc_neuron_num*4]).cuda()  # W[i]表示enc_layer与dec_layer[i]的连接权重
    learn_rate = 0.1
    train_times = 0
    t_spike = torch.zeros(size=[150, 4, enc_neuron_num], dtype=torch.float).cuda()
    # 生成脉冲
    for m in range(150):  # 编码过程非常慢
        print(m)
        for i in range(4):
            t_spike[m][i] = enc_layer[i].encode(x_train[m][i], 500)[0]  # [1, enc_neuron_num]的tensor
    t_spike = t_spike.view(150, -1)


    while 1:
        m = np.random.randint(low=0, high=150)  # 随机抽取一个数据
        real_class = y_train[m]  # 真实的类别
        for i in range(3):
            # 分别送入dec_layer的每一个class
            dec_layer[i].calculate_membrane_potentials(W[i], t_spike[m])
            t_max = dec_layer[i].v.argmax()
            if i == real_class:
                # 应该放电
                if dec_layer[i].v[t_max] < dec_layer[i].v_thr:
                    # 电压没有超过阈值
                    W[i] += learn_rate * node.psp_kernel(t_max - t_spike[m], dec_layer[i].v0, dec_layer[i].tau, dec_layer[i].tau_s)
            else:
                # 不应该放电
                if dec_layer[i].v[t_max] > dec_layer[i].v_thr:
                    # 电压没有超过阈值
                    W[i] -= learn_rate * node.psp_kernel(t_max - t_spike[m], dec_layer[i].v0, dec_layer[i].tau, dec_layer[i].tau_s)
        if train_times % 128 == 0:
            print(train_times)

        if train_times % 4096 == 0:
            # 测试一次
            error_times = 0
            for m in range(150):
                real_class = y_train[m]  # 真实的类别

                for i in range(3):
                    # 分别送入dec_layer的每一个class
                    dec_layer[i].calculate_membrane_potentials(W[i], t_spike[m])
                    t_max = dec_layer[i].v.argmax()
                    if i == real_class:
                        # 应该放电
                        if dec_layer[i].v[t_max] < dec_layer[i].v_thr:
                            error_times += 1
                            break
                    else:
                        # 不应该放电
                        if dec_layer[i].v[t_max] > dec_layer[i].v_thr:
                            # 电压没有超过阈值
                            error_times += 1
                            break
            print("测试错误率", error_times / 150)


        train_times += 1
        if train_times == 1000000:
            break




