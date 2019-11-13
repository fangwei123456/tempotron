
import encoder
import node
import torch
import numpy as np
import math
import torch.nn.functional as F
import sys
'''
pima数据集，768个样本，取前500个
'''

def get_k_fold(i):
    # 将样本分成10份，取第i份为测试集，其余为训练集，返回训练集和测试集的标号
    test_i = torch.arange(i*50, i*50+50)
    train_i = []
    for ii in range(10):
        if ii != i:
            train_i.append(torch.arange(ii*50, ii*50+50))
    train_i = torch.cat(train_i)
    return train_i.tolist(), test_i.tolist()
if __name__ == '__main__':
    pima = np.loadtxt('pima-indians-diabetes.csv', skiprows=1, delimiter=',')  # 最后一列为分类
    data_num = pima.shape[0]
    class_num = 2
    feature_num = pima.shape[1] - 1
    x_train = torch.from_numpy(pima[:, 0: feature_num]).float().cuda()
    y_train = torch.from_numpy(pima[:, feature_num]).cuda()
    # data_num个特征，需要data_num组编码器
    enc_layer = []
    x_train_min = x_train.min(0)[0]
    x_train_max = x_train.max(0)[0]
    print(x_train_min)
    print(x_train_max)
    enc_neuron_num = 512
    per_class_neuron_num = 6

    for i in range(feature_num):
        enc_layer.append(encoder.GaussianEncoder(x_train_min[i], x_train_max[i], enc_neuron_num, 'cuda:0'))
    # 共有class_num类，需要class_num个tempotron
    dec_layer = []
    for i in range(class_num*per_class_neuron_num):
        dec_layer.append(node.LIFNode(tau=15.0, tau_s=15.0/4, v_rest=0, T=500, N=enc_neuron_num*feature_num, device='cuda:0'))  # 由于是全连接，因此enc_layer的所有元素都与这个节点相连

    W = torch.rand(size=[class_num*per_class_neuron_num, enc_neuron_num*feature_num]).cuda()  # W[i]表示enc_layer与dec_layer[i]的连接权重
    B = torch.rand(size=[class_num*per_class_neuron_num, enc_neuron_num*feature_num]).cuda()
    learn_rate = 0.1
    train_times = 0
    try:
        t_spike = torch.load('pima-indians-diabetes-' + str(enc_neuron_num) + '.pkl', map_location=x_train.device)
    except BaseException:
        t_spike = torch.zeros(size=[data_num, feature_num, enc_neuron_num], dtype=torch.float).cuda()
        # 生成脉冲
        for m in range(data_num):  # 编码过程非常慢
            print(m)
            for i in range(feature_num):
                t_spike[m][i] = enc_layer[i].encode(x_train[m][i], 500)[0]  # [1, enc_neuron_num]的tensor
        t_spike = t_spike.view(data_num, -1)
        torch.save(t_spike, 'pima-indians-diabetes-' + str(enc_neuron_num) + '.pkl')

    min_error_rate = 1
    train_i, test_i = get_k_fold(1)

    while 1:
        m = train_i[np.random.randint(low=0, high=450)]  # 随机抽取一个数据
        real_class = y_train[m].item()  # 真实的类别

        for i in range(class_num*per_class_neuron_num):
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
            for m in test_i:
                real_class = y_train[m].item()  # 真实的类别
                # 比较每一个分类器的输出结果
                vote_result = torch.zeros([class_num], dtype=torch.float).cuda()
                # vote_result[0]记录的是此类属于第0类的票数
                for i in range(class_num*per_class_neuron_num):
                    # 分别送入dec_layer的每一个class
                    dec_layer[i].calculate_membrane_potentials(W[i] * F.sigmoid(B[i]), t_spike[m])
                    t_max = dec_layer[i].v.argmax()
                    if dec_layer[i].v[t_max] > dec_layer[i].v_thr:
                        vote_result[i // per_class_neuron_num] += 1

                pred_class = vote_result.argmax()  # 少数服从多数
                if pred_class != real_class:
                    error_times += 1

            error_rate = error_times / 50
            if error_rate < min_error_rate:
                min_error_rate = error_rate
            print(sys.argv)

            print("测试错误率", error_rate)
            print("最小错误率", min_error_rate)
            print(W)
            print(B)


        train_times += 1
        if train_times == 1000000:
            break




