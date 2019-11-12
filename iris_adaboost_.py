from sklearn import datasets
import encoder
import node
import torch
import numpy as np
import math
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
    data_weight = torch.ones([3, 150], dtype=torch.float).cuda() / 150  #初始化数据的权值分布
    classifier_weight = torch.ones([3, per_class_neuron_num], dtype=torch.float).cuda() / per_class_neuron_num  # 初始化分类器的权重
    for i in range(4):
        enc_layer.append(encoder.GaussianEncoder(x_train_min[i], x_train_max[i], enc_neuron_num, 'cuda:0'))
    # 共有3类，需要3个tempotron
    dec_layer = []
    for i in range(3*per_class_neuron_num):
        dec_layer.append(node.LIFNode(tau=15.0, tau_s=15.0/4, v_rest=0, T=500, N=enc_neuron_num*4, device='cuda:0'))  # 由于是全连接，因此enc_layer的所有元素都与这个节点相连

    W = torch.rand(size=[3*per_class_neuron_num, enc_neuron_num*4]).cuda()  # W[i]表示enc_layer与dec_layer[i]的连接权重
    learn_rate = 0.5
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
        real_class = y_train[m].item()  # 真实的类别

        for i in range(3*per_class_neuron_num):
            # 分别送入dec_layer的每一个class
            neural_class = i // per_class_neuron_num  # 表示这个神经元应该对哪一类响应1
            neural_seq = i % per_class_neuron_num
            dec_layer[i].calculate_membrane_potentials(W[i], t_spike[m])
            t_max = dec_layer[i].v.argmax()
            # 训练第i个分类器
            if neural_class == real_class:
                # 应该放电
                if dec_layer[i].v[t_max] < dec_layer[i].v_thr:
                    W[i] += learn_rate * node.psp_kernel(t_max - t_spike[m], dec_layer[i].v0, dec_layer[i].tau, dec_layer[i].tau_s)
            else:
                # 不应该放电
                if dec_layer[i].v[t_max] > dec_layer[i].v_thr:
                    W[i] -= learn_rate * node.psp_kernel(t_max - t_spike[m], dec_layer[i].v0, dec_layer[i].tau, dec_layer[i].tau_s)
            # 计算第i个分类器的错误率
            is_error = torch.zeros([150], dtype=torch.float).cuda()
            for m in range(150):
                real_class = y_train[m].item()  # 真实的类别
                dec_layer[i].calculate_membrane_potentials(W[i], t_spike[m])
                t_max = dec_layer[i].v.argmax()
                if neural_class == real_class:
                    # 应该放电
                    if dec_layer[i].v[t_max] < dec_layer[i].v_thr:
                        is_error[m] = 1
                else:
                    # 不应该放电
                    if dec_layer[i].v[t_max] > dec_layer[i].v_thr:
                        is_error[m] = 1
            error_rate = (is_error * data_weight[neural_class]).sum().item()  # 计算带权错误率
            print("train_times", train_times, "分类器", i, "错误率", is_error.mean().item(), "带权错误率", error_rate)
            # 计算此分类器的系数
            if error_rate >= 1:
                classifier_weight[neural_class][neural_seq] = 0
            else:
                classifier_weight[neural_class][neural_seq] = math.log((1 - error_rate) / error_rate)
            # 更新数据的权重
            data_weight[neural_class] = data_weight[neural_class] * torch.exp(-classifier_weight[neural_class][neural_seq] * is_error)
            # 权重归一化
            data_weight[neural_class] = data_weight[neural_class] / data_weight[neural_class].sum()



        if train_times % 4096 == 0:
            # 测试一次
            error_times = 0
            for m in range(150):
                real_class = y_train[m].item()  # 真实的类别
                # 比较每一个分类器的输出结果
                vote_result = torch.zeros([3], dtype=torch.float).cuda()
                # vote_result[0]记录的是此类属于第0类的票数
                for i in range(3*per_class_neuron_num):
                    # 分别送入dec_layer的每一个class
                    dec_layer[i].calculate_membrane_potentials(W[i], t_spike[m])
                    t_max = dec_layer[i].v.argmax()
                    if dec_layer[i].v[t_max] > dec_layer[i].v_thr:
                        vote_result[i // per_class_neuron_num] += classifier_weight[i // per_class_neuron_num][i % per_class_neuron_num]

                pred_class = vote_result.argmax()  # 少数服从多数
                if pred_class != real_class:
                    error_times += 1




            print("测试错误率", error_times / 150)


        train_times += 1
        if train_times == 1000000:
            break




