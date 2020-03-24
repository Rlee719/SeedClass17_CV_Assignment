import numpy as np
# import minpy.numpy as np
import bpnn
import data_utils
import random
import optimizer
import loss

def shuffle(x, y):
    random_arr = [i for i in range(len(x))]
    random.shuffle(random_arr)
    return x[random_arr], y[random_arr]

def get_acc_avg(y, p):
    # input:
    #   y:  图片标签        (batch_size x 1)
    #   p:  预测概率        (batch_size x 10(或者class_num))
    # output:
    #   平均正确率
    return np.sum(p.argmax(1) == y) / y.shape[0]

def train(classifier, X, y, epoch=10, batch_size=3, reg=1e-3):
    # 训练网络
    # input:
    #   X:  训练图片
    #   y:  训练标签
    #   epoch:  训练次数
    #   lr: 训练速率
    #   batch_size: mini batch 大小 (最好能整除 X 的张数)
    # output:
    #   loss_list、acc_list
    #   横坐标为epoch
    loss_list, acc_list = [], []
    #print(len(classifier.raise_params()))
    loss_func = loss.Loss_Sequential(loss.soft_max_loss(), loss.L1_loss())
    _optimizer = optimizer.MB_SGD(classifier.layers, loss_func)
    batch_num = X.shape[0] // batch_size
    for e in range(epoch):
        X, y = shuffle(X, y)
        acc_sum, loss_sum = 0, 0
        for batch in range(batch_num):
            x_batch = X[batch*batch_size:(batch+1)*batch_size]
            y_batch = y[batch*batch_size:(batch+1)*batch_size]
            p = classifier.forward(x_batch)
            loss_sum += loss.softmax_loss(y_batch, p) + loss.L1_loss(classifier.layers, reg)
            _optimizer.optimize(x_batch, y_batch, p)       
            acc_sum += get_acc_avg(y_batch, p)
        loss_list.append(loss_sum / batch_num)
        acc_list.append(acc_sum / batch_num)

    return loss_list, acc_list

def evaluate(classifier, X, y):
    p = classifier.forward(X)
    return get_acc_avg(y, p)

if __name__ == "__main__":
    '''
    print("Doing: load image data")
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10('../cifar-10-batches-py')

    # 图像左右翻转增加数据集
    print("Doing: image enhance")
    data_utils.create_image_enhance_npy(X_train, X_test, y_train, y_test, "../")
    '''
    
    # 加载 npy 文件
    print("Doing: load npy files")
    X_train, X_test, y_train, y_test = data_utils.load_npy("../")

    X_train = X_train[0:10000]
    y_train = y_train[0:10000]

    # 将单幅图片转成 3072 维的向量
    print("Doing: image data reshape")
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))


    models = [
                [[3072,190,10],[bpnn.relu,bpnn.relu,bpnn.relu,bpnn.relu]],
            ]
    lrs = [1.0]
    for m in models:
        for lr in lrs:
            classifier = bpnn.BPNN(m, 'he')
            optimizer = bpnn.Optimizer(classifier)
            
            optimizer.lr = lr
            optimizer.lr_decay = bpnn.Learning_rate_decay.exp
            optimizer.lr_k = 0.1
            optimizer.reg_type = bpnn.Regularization.L2
            optimizer.momentum_type = bpnn.Momentum.Momentum
            optimizer.mu = 0.8
            optimizer.reg = 0

            classifier.useOpt(optimizer)

            print("Doing: train net")
            train_loss_list, train_acc_list = classifier.train(X_train, y_train,epoch=30, batch_size=200)
            print(train_loss_list[-1], train_acc_list[-1])

    # print(train_loss_list)
    # print("Doing: test net")
    # test_acc = classifier.evaluate(X_test, y_test)

    # print("test accuracy is %f" % (test_acc))