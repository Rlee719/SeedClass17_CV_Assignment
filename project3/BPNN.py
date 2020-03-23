'''
    调用示例：
        classifier = bpnn.BPNN([[2, 3, 2, 2],[bpnn.tanh, bpnn.relu, bpnn.sigmoid]])
    可选的激活函数有：
        sigmoid
        tanh
        relu
'''

from enum import IntEnum
import numpy as np
import random


def _unact(x):
    return x

def _unact_derivative(y):
    return np.ones(y.shape)

def _tanh(x):
    return np.tanh(x)

def _tanh_derivative(y):
    return 1 - y * y

def _relu(x):
    return np.where(x < 0, 0, x)

def _relu_derivative(y):
    return np.where(y < 0, 0, 1)

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _sigmoid_derivative(y):
    return y * (1 - y)


unact = {"f":_unact, "f'":_unact_derivative}
tanh = {"f":_tanh, "f'":_tanh_derivative}
relu = {"f":_relu, "f'":_relu_derivative}
sigmoid = {"f":_sigmoid, "f'":_sigmoid_derivative}

class BPNN():
    def __init__(self, model_config, init_type='none'):
        # input:
        #   model_config: [[input.length, layer1.length, layer2.length, ..., output_class_num],
        #                              [act_func1,  act_func2,       ..., act_funcN],
        
        # dy/dx = f'(x) = f'(f^-1(y)) = d_in_func(y)
        self.model_config = model_config
        self.layers, self.w, self.b = [], [], []
        self.act_func_dir = []
        self.layer_num = 0
        self.init_model(init_type)

    def init_model(self, init_type):
        # 初始化 W 和 b 为 0
        for i, layer in enumerate(self.model_config[0]):
            if i == 0:
                continue
            else:
                print(np.sqrt(self.model_config[0][i - 1] / 2))
                self.layers.append(np.zeros(layer))
                if init_type == 'none':
                    self.w.append(np.random.randn(self.model_config[0][i-1], layer))
                    self.b.append(np.random.randn(layer))
                elif init_type == 'xavier':
                    self.w.append(np.random.randn(self.model_config[0][i-1], layer) / np.sqrt(self.model_config[0][i - 1]))
                    self.b.append(np.random.randn(layer) / np.sqrt(self.model_config[0][i - 1]))
                elif init_type == 'he':
                    self.w.append(np.random.randn(self.model_config[0][i-1], layer) / np.sqrt(self.model_config[0][i - 1] / 2))
                    self.b.append(np.random.randn(layer) / np.sqrt(self.model_config[0][i - 1] / 2))
                # 不能初始化为0，否则梯度永远是0
                # self.w.append(np.ones((self.model_config[0][i-1], layer)))
                # self.b.append(np.ones((layer)))
                self.act_func_dir.append(self.model_config[1][i-1])
        self.layer_num = len(self.layers)


    def useOpt(self, optimizer):
        self.optimizer = optimizer


    def shuffle(self, x, y):
        random_arr = [i for i in range(len(x))]
        random.shuffle(random_arr)
        return x[random_arr], y[random_arr]
    

    def softmax(self, vector):
        # input:
        #   vector: 最后一层输出 (batch_size x 10(或者class_num))
        # output:
        #   output: 预测概率     (batch_size x 10(或者class_num))
        m = np.max(vector, axis=1, keepdims=True)
        exp_scores = np.exp(vector-m)
        output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return output


    def softmax_loss(self, y, p):
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
        if self.optimizer.reg_type == Regularization.L1:
            for i, w in enumerate(self.w):
                loss += self.optimizer.reg * (np.sum(np.abs(w)) + np.sum(np.abs(self.b[i])))
        elif self.optimizer.reg_type == Regularization.L2:
            for i, w in enumerate(self.w):
                loss += self.optimizer.reg / 2 * (np.sum(w * w) + np.sum(self.b[i] * self.b[i]))
        return loss


    def get_acc_avg(self, y, p):
        # input:
        #   y:  图片标签        (batch_size x 1)
        #   p:  预测概率        (batch_size x 10(或者class_num))
        # output:
        #   平均正确率
        return np.sum(p.argmax(1) == y) / y.shape[0]


    def forward(self, X):
        # input:
        #   X:  图片数据集矩阵  (batch_size x 3072)
        # output:
        #   p:  预测概率       (batch_size x 10(或者 class_num))
        a = X
        for i in range(self.layer_num):
            self.layers[i] = np.dot(a, self.w[i]) + self.b[i]
            a = self.layers[i] = self.act_func_dir[i]["f"](self.layers[i])
        p = self.softmax(self.layers[-1])
        return p


    def train(self, X, y, epoch=10, batch_size=3):
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

        batch_num = X.shape[0] // batch_size
        for e in range(epoch):
            X, y = self.shuffle(X, y)
            acc_sum, loss_sum = 0, 0
            for batch in range(batch_num):
                x_batch = X[batch*batch_size:(batch+1)*batch_size]
                y_batch = y[batch*batch_size:(batch+1)*batch_size]
                p = self.forward(x_batch)
                self.optimizer.optimize(x_batch, y_batch, p)
                loss_sum += self.softmax_loss(y_batch, p)
                acc_sum += self.get_acc_avg(y_batch, p)
            loss_list.append(loss_sum / batch_num)
            acc_list.append(acc_sum / batch_num)

        return loss_list, acc_list


    def predict(self, X):
        p = self.forward(X)
        return p.argmax(1)

    def evaluate(self, X, y):
        p = self.forward(X)
        return self.get_acc_avg(y, p)


class Momentum(IntEnum):
    none     = 0
    Momentum = 1
    Nesterov = 2
    Adagrad  = 3
    RMSprop  = 4


class Regularization(IntEnum):
    none = 0
    L1   = 1
    L2   = 2


class Learning_rate_decay(IntEnum):
    none = 0
    step = 1
    exp  = 2
    inv  = 3


class Optimizer():
    def __init__(self, classifier):
        self.classifier = classifier
        self.lr = 1
        self.lr_k = 0
        self.lr_decay = Learning_rate_decay.none
        self.reg_type = Regularization.none
        self.reg = 0
        self.momentum_type = Momentum.none
        self.mu = 0

        self.init_opt()
        

    def init_opt(self):
        self.time = 0
        self.cache_dvw = []
        self.cache_dvb = []
        for i in range(self.classifier.layer_num):
            self.cache_dvw.append(np.zeros(self.classifier.w[i].shape))
            self.cache_dvb.append(np.zeros(self.classifier.b[i].shape))


    def update_lr_decay(self):
        if self.lr_decay == Learning_rate_decay.step:
            self.lr /= 2
        elif self.lr_decay == Learning_rate_decay.exp:
            self.lr = self.lr * np.exp(- self.lr_k)
        elif self.lr_decay == Learning_rate_decay.inv:
            self.lr = self.lr * ((1+self.lr_k*(self.time-1)) / (1+self.lr_k*self.time))
        self.time += 1


    def optimize(self, X, y, p):
        d_w, d_b = self.evaluate_analytic_grad(X, y, p)
        if self.momentum_type == Momentum.none:
            for i in range(self.classifier.layer_num):
                d_w[i] = - self.lr * d_w[i]
                d_b[i] = - self.lr * d_b[i]

        elif self.momentum_type == Momentum.Momentum:
            for i in range(self.classifier.layer_num):
                d_w[i] = self.cache_dvw[i] * self.mu - self.lr * d_w[i]
                d_b[i] = self.cache_dvb[i] * self.mu - self.lr * d_b[i]
                self.cache_dvw[i] = d_w[i].copy()
                self.cache_dvb[i] = d_b[i].copy()

        elif self.momentum_type == Momentum.Nesterov:
            for i in range(self.classifier.layer_num):
                v = self.cache_dvw[i] * self.mu - self.lr * d_w[i]
                d_w[i] = self.mu * v - self.lr * d_w[i]
                self.cache_dvw[i] = v.copy()

                v = self.cache_dvb[i] * self.mu - self.lr * d_b[i]
                d_b[i] = self.mu * v - self.lr * d_b[i]
                self.cache_dvb[i] = v.copy()

        elif self.momentum_type == Momentum.Adagrad:
            for i in range(self.classifier.layer_num):
                self.cache_dvw[i] += d_w[i] ** 2
                d_w[i] = - self.lr * d_w[i] / (np.sqrt(self.cache_dvw[i]) + 1e-7)
                self.cache_dvb[i] += d_b[i] ** 2
                d_b[i] = - self.lr * d_b[i] / (np.sqrt(self.cache_dvb[i]) + 1e-7)

        elif self.momentum_type == Momentum.RMSprop:
            for i in range(self.classifier.layer_num):
                self.cache_dvw[i] += self.mu * self.cache_dvw[i] + (1 - self.mu) * d_w[i] ** 2
                d_w[i] = - self.lr * d_w[i] / (np.sqrt(self.cache_dvw[i]) + 1e-7)
                self.cache_dvb[i] += self.mu * self.cache_dvb[i] + (1 - self.mu) * d_b[i] ** 2
                d_b[i] = - self.lr * d_b[i] / (np.sqrt(self.cache_dvb[i]) + 1e-7)

        for i in range(self.classifier.layer_num):
            self.classifier.w[i] += d_w[i]
            self.classifier.b[i] += d_b[i]

        self.update_lr_decay()


    def evaluate_analytic_grad(self, X, y, p):
        # input:
        #   X: 图片数据集矩阵  (batch_size x 3072)
        #   y: 图片标签    (batch_size x 1)
        #   p: 预测概率       (batch_size x 10(或者 class_num))
        batch_size = X.shape[0]
        Y = np.zeros(p.shape)
        for i, label in enumerate(y):
            Y[i][label] = 1
        d_w, d_b = list(range(self.classifier.layer_num)), list(range(self.classifier.layer_num))
        
        delta = (p - Y) * self.classifier.act_func_dir[-1]["f'"](self.classifier.layers[-1])
        for i in range(self.classifier.layer_num-1, 0, -1):
            d_w[i] = np.dot(self.classifier.layers[i-1].T, delta) / batch_size
            d_b[i] = delta.sum(0) / batch_size
            delta = np.dot(delta, self.classifier.w[i].T) * self.classifier.act_func_dir[i-1]["f'"](self.classifier.layers[i-1])
        d_w[0] = np.dot(X.T, delta) / batch_size
        d_b[0] = delta.sum(0) / batch_size

        if self.reg_type == Regularization.L1:
            for i, w in enumerate(self.classifier.w):
                d_w[i] += self.reg * np.sign(w)
                d_b[i] += self.reg * np.sign(self.classifier.b[i])
        elif self.reg_type == Regularization.L2:
            for i, w in enumerate(self.classifier.w):
                d_w[i] += self.reg * w
                d_b[i] += self.reg * self.classifier.b[i]
        return d_w, d_b