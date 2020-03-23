import numpy as np
from module import Optimizer
from collections import Iterable
from enum import IntEnum

class MB_SGD(Optimizer):
    def __init__(self, **input):
        super().__init__(**input)
        self.lr_k = 0
        self.lr_decay = Learning_rate_decay.none
        self.reg_type = Regularization.none
        self.reg = 0
        self.momentum_type = Momentum.none
        self.mu = 0

    def optimize(self):
        upstream_gradient = np.array(self.loss_func.backward())
        #print("optimize!")
        for i, layer in enumerate(reversed(self.layers)):
            upstream_gradient = layer.backward(upstream_gradient)
            #print(upstream_gradient.shape, type(layer))
            
            try:
                local_grad = layer.local_grad(upstream_gradient) * self.lr
                layer.optimize(local_grad)
            except:
                pass
        self.update_lr_decay()



'''
class Momentum(Optimizer):
    def __init__(self):
        super().__init__()

    def optimize(self):
    for i in range(self.classifier.layer_num):
        d_w[i] = self.cache_dvw[i] * self.momentum_mu - self.lr * d_w[i]
        d_b[i] = self.cache_dvb[i] * self.momentum_mu - self.lr * d_b[i]
        self.cache_dvw[i] = d_w[i].copy()
        self.cache_dvb[i] = d_b[i].copy()


        elif self.momentum_type == Momentum.Nesterov:
            for i in range(self.classifier.layer_num):
                v = self.cache_dvw[i] * self.momentum_mu - self.lr * d_w[i]
                d_w[i] = self.momentum_mu * v - self.lr * d_w[i]
                self.cache_dvw[i] = v.copy()

                v = self.cache_dvb[i] * self.momentum_mu - self.lr * d_b[i]
                d_b[i] = self.momentum_mu * v - self.lr * d_b[i]
                self.cache_dvb[i] = v.copy()

        elif self.momentum_type == Momentum.Adagrad:
            for i in range(self.classifier.layer_num):
                self.cache_dvw[i] += d_w[i] ** 2
                d_w[i] = - self.lr * d_w[i] / (np.sqrt(self.cache_dvw[i]) + 1e-7)
                self.cache_dvb[i] += d_b[i] ** 2
                d_b[i] = - self.lr * d_b[i] / (np.sqrt(self.cache_dvb[i]) + 1e-7)

        elif self.momentum_type == Momentum.RMSprop:
            for i in range(self.classifier.layer_num):
                self.cache_dvw[i] += self.momentum_mu * self.cache_dvw[i] + (1 - self.momentum_mu) * d_w[i] ** 2
                d_w[i] = - self.lr * d_w[i] / (np.sqrt(self.cache_dvw[i]) + 1e-7)
                self.cache_dvb[i] += self.momentum_mu * self.cache_dvb[i] + (1 - self.momentum_mu) * d_b[i] ** 2
                d_b[i] = - self.lr * d_b[i] / (np.sqrt(self.cache_dvb[i]) + 1e-7)

        for i in range(self.classifier.layer_num):
            self.classifier.w[i] += d_w[i]
            self.classifier.b[i] += d_b[i]

        self.update_lr_decay()

    def compute_grad(self):
        # input:
        #   X: 图片数据集矩阵  (batch_size x 3072)
        #   y: 图片标签    (batch_size x 1)
        #   p: 预测概率       (batch_size x 10(或者 class_num))
'''


class Learning_rate_decay(IntEnum):
    none = 0
    step = 1
    exp  = 2
    inv  = 3

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
