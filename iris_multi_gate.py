from sklearn import datasets
import encoder
import node
import torch
import numpy as np
import math
import torch.nn.functional as F
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
    enc_neuron_num = 4096
    per_class_neuron_num = 6
    # [0, per_class_neuron_num-1]对应第0类，[per_class_neuron_num, 2*per_class_neuron_num-1]对应第1类，[2*per_class_neuron_num, 3*per_class_neuron_num-1]对应第2类
    for i in range(4):
        enc_layer.append(encoder.GaussianEncoder(x_train_min[i], x_train_max[i], enc_neuron_num, 'cuda:0'))
    # 共有3类，需要3个tempotron
    dec_layer = []
    for i in range(3*per_class_neuron_num):
        dec_layer.append(node.LIFNode(tau=15.0, tau_s=15.0/4, v_rest=0, T=500, N=enc_neuron_num*4, device='cuda:0'))  # 由于是全连接，因此enc_layer的所有元素都与这个节点相连

    W = torch.rand(size=[3*per_class_neuron_num, enc_neuron_num*4]).cuda()  # W[i]表示enc_layer与dec_layer[i]的连接权重
    B = torch.rand(size=[3*per_class_neuron_num, enc_neuron_num*4]).cuda()
    learn_rate = 0.1
    t_spike = torch.zeros(size=[150, 4, enc_neuron_num], dtype=torch.float).cuda()
    # 生成脉冲
    for m in range(150):  # 编码过程非常慢
        print(m)
        for i in range(4):
            t_spike[m][i] = enc_layer[i].encode(x_train[m][i], 500)[0]  # [1, enc_neuron_num]的tensor
    t_spike = t_spike.view(150, -1)
    min_error_rate = 1
    train_times = 0
    read_seq = np.random.permutation(150)
    while True:

        m = read_seq[np.random.randint(low=0, high=135)]  # 随机抽取一个数据
        real_class = y_train[m].item()  # 真实的类别

        for i in range(3*per_class_neuron_num):
            # 分别送入dec_layer的每一个class
            neural_class = i // per_class_neuron_num  # 表示这个神经元应该对哪一类响应1
            neural_seq = i % per_class_neuron_num
            Gi = F.sigmoid(B[i])
            dec_layer[i].calculate_membrane_potentials(W[i] * Gi, t_spike[m])
            t_max = dec_layer[i].v.argmax()
            # 训练第i个分类器
            if neural_class == real_class:
                # 应该放电
                if dec_layer[i].v[t_max] < dec_layer[i].v_thr:
                    v_error = node.psp_kernel(t_max - t_spike[m], dec_layer[i].v0, dec_layer[i].tau, dec_layer[i].tau_s)
                    dW = Gi * v_error
                    dB = W[i] * v_error * Gi * (1 - Gi)
                    W[i] += learn_rate * dW
                    B[i] += learn_rate * dB
            else:
                # 不应该放电
                if dec_layer[i].v[t_max] > dec_layer[i].v_thr:
                    v_error = node.psp_kernel(t_max - t_spike[m], dec_layer[i].v0, dec_layer[i].tau, dec_layer[i].tau_s)
                    dW = Gi * v_error
                    dB = W[i] * v_error * Gi * (1 - Gi)
                    W[i] -= learn_rate * dW
                    B[i] -= learn_rate * dB

            # 计算第i个分类器的错误率




        if train_times % 128 == 0:
            # 测试一次
            error_times = 0
            for m in read_seq[135:150]:
                real_class = y_train[m].item()  # 真实的类别
                # 比较每一个分类器的输出结果
                vote_result = torch.zeros([3], dtype=torch.float).cuda()
                # vote_result[0]记录的是此类属于第0类的票数
                for i in range(3*per_class_neuron_num):
                    # 分别送入dec_layer的每一个class
                    dec_layer[i].calculate_membrane_potentials(W[i] * F.sigmoid(B[i]), t_spike[m])
                    t_max = dec_layer[i].v.argmax()
                    if dec_layer[i].v[t_max] > dec_layer[i].v_thr:
                        vote_result[i // per_class_neuron_num] += 1

                pred_class = vote_result.argmax()  # 少数服从多数
                if pred_class != real_class:
                    error_times += 1

            error_rate = error_times / 150
            if error_rate < min_error_rate:
                min_error_rate = error_rate
            print("测试错误率", error_rate)
            print("最小错误率", min_error_rate)
            print(W)
            print(F.sigmoid(B))


        train_times += 1
        if train_times == 1000000:
            break




