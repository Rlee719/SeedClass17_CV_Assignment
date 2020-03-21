import numpy as np
# import minpy.numpy as np
import bpnn
import data_utils
import random
import optimizer

def shuffle(x, y):
    random_arr = [i for i in range(len(x))]
    random.shuffle(random_arr)
    return x[random_arr], y[random_arr]
class Loss():
    def __init__(self):
        pass

    @classmethod
    def softmax_loss(cls, y, p):
        # input:
        #   y:  图片标签        (batch_size x 1)
        #   p:  预测概率        (batch_size x 10(或者class_num))
        # output:
        #   avg_loss: 每batch的平均loss     float64
        loss = 0.0
        batch_size = y.shape[0]
        for i, yi in enumerate(y):
            loss -= np.log(p[i][yi])
        loss = loss / batch_size
        return loss

    @classmethod
    def L1_loss(cls, get_params, reg):
        #if self.optimizer.reg_type == Regularization.L1:
        loss = 0.0
        for i, param in get_params():
            print(type(param["value"]))
            loss += reg * (np.sum(np.abs(param["value"])))
        return loss

    @classmethod
    def L2_loss(cls, get_params, reg):
        #if self.optimizer.reg_type == Regularization.L2:
        loss = 0.0
        for i, param in get_params():
            loss += reg / 2 * (np.sum(param["value"] * param["value"]))
        return loss

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
    _optimizer = optimizer.SGD(classifier.raise_params)
    batch_num = X.shape[0] // batch_size
    for e in range(epoch):
        X, y = shuffle(X, y)
        acc_sum, loss_sum = 0, 0
        for batch in range(batch_num):
            x_batch = X[batch*batch_size:(batch+1)*batch_size]
            y_batch = y[batch*batch_size:(batch+1)*batch_size]
            p = classifier.forward(x_batch)
            loss_sum += Loss.softmax_loss(y_batch, p) + Loss.L1_loss(classifier.raise_params, reg)
            #_optimizer.optimize(x_batch, y_batch, p)       
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

    X_train = X_train[0:1000]
    y_train = y_train[0:1000]

    # 将单幅图片转成 3072 维的向量
    print("Doing: image data reshape")
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    classifier = bpnn.BPNN(model_config=[3072,20,10], act_func="_relu")
    #optimizer = bpnn.Optimizer(classifier)
    
    #optimizer.lr = 0.1

    #classifier.useOpt(optimizer)
    classifier.set_train()
    print("Doing: train net")
    train_loss_list, train_acc_list = train(classifier, X_train, y_train, epoch=20, batch_size=10)

    #print(train_loss_list)
    print("Doing: test net")
    classifier.set_eval()
    test_acc = evaluate(classifier, X_test, y_test)

    print("test accuracy is %f" % (test_acc))