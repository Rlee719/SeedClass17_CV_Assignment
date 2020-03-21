import numpy as np
from enum import IntEnum

class SGD():
    def __init__(self, get_params):
        self.get_params = get_params
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
        self.cache = []
        #self.cache_dvw = []
        #self.cache_dvb = []
        for param in self.get_params():
            print(param["value"])
            self.cache.append(np.zeros(param["value"].shape))
            #self.cache_dvw.append(np.zeros(self.classifier.w[i].shape))
            #self.cache_dvb.append(np.zeros(self.classifier.b[i].shape))


    def update_lr_decay(self):
        if self.lr_decay == Learning_rate_decay.step:
            self.lr /= 2
        elif self.lr_decay == Learning_rate_decay.exp:
            self.lr = self.lr * np.exp(- self.lr_k)
        elif self.lr_decay == Learning_rate_decay.inv:
            self.lr = self.lr * ((1+self.lr_k*(self.time-1)) / (1+self.lr_k*self.time))
        self.time += 1

    def optimize(self, X, y, p):
        d_w, d_b = self.compute_grad(X, y, p)
        if self.momentum_type == Momentum.none:
            #可进一步修改！
            for i, param in enumerate(self.get_params()):
                #if parma[name] = 
                if param["value"].get_params == "weight":
                    pass
            for i in range(self.classifier.layer_num):
                d_w[i] = - self.lr * d_w[i]
                d_b[i] = - self.lr * d_b[i]

        elif self.momentum_type == Momentum.Momentum:
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