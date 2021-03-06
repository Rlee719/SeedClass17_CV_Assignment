import numpy as np
# import minpy.numpy as np
import bpnn
import data_utils
from plot import *
def act_type():
    
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
                [[3072,200,100,50,10],[bpnn.relu,bpnn.relu,bpnn.relu,bpnn.relu,bpnn.relu]],
                [[3072,200,100,50,10],[bpnn.sigmoid, bpnn.sigmoid, bpnn.sigmoid, bpnn.sigmoid, bpnn.sigmoid]],
                [[3072,200,100,50,10],[bpnn.tanh,bpnn.tanh,bpnn.tanh,bpnn.tanh,bpnn.tanh]],
                [[3072,200,100,50,10],[bpnn.unact,bpnn.unact,bpnn.unact,bpnn.unact,bpnn.unact]]
            ]
    lrs = [1e-2]
    act = ["relu", "sigmoid", "tanh", "unact"]
    for i, m in enumerate(models):
        for lr in lrs:
            classifier = bpnn.BPNN(m, 'xavier')
            optimizer = bpnn.Optimizer(classifier)
            
            optimizer.lr = lr
            optimizer.lr_decay = bpnn.Learning_rate_decay.exp
            optimizer.lr_k = 0
            optimizer.momentum_type = bpnn.Momentum.Adagrad
            optimizer.mu = 0.8
            optimizer.reg_type = bpnn.Regularization.L2
            optimizer.reg = 0.00075

            classifier.useOpt(optimizer)

            print("Doing: train net")
            train_loss_list, train_acc_list = classifier.train(X_train, y_train, epoch=50, batch_size=64)
            print(train_loss_list[-1], train_acc_list[-1])
            draw_plot(range(len(train_loss_list)), train_loss_list, "loss_" + act[i], "epoch", "loss", "loss_" + act[i])
            draw_plot(range(len(train_acc_list)), train_acc_list, "train_acc_" + act[i], "epoch", "train_acc", "train_acc_" + act[i])
            test_acc = classifier.evaluate(X_test, y_test)
            print("test accuracy is %f" % (test_acc))

    #print(train_loss_list)
    #print("Doing: test net")


if __name__ == "__main__":
    act_type()
    '''
    print("Doing: load image data")
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10('../cifar-10-batches-py')

    # 图像左右翻转增加数据集
    print("Doing: image enhance")
    data_utils.create_image_enhance_npy(X_train, X_test, y_train, y_test, "../")
    '''