'''
    调用示例：
        classifier = bpnn.BPNN([[2, 3, 2, 2],[bpnn.tanh, bpnn.relu, bpnn.sigmoid]])
    可选的激活函数有：
        sigmoid
        tanh
        relu
'''

import numpy as np
import random


def _unact(x):
    return x

def _unact_derivative(y):
    return np.ones(y.shape)

unact = {"f":_unact, "f'":_unact_derivative}

def _tanh(x):
    return np.tanh(x)

def _tanh_derivative(y):
    return 1 - y * y

tanh = {"f":_tanh, "f'":_tanh_derivative}

def _relu(x):
    return np.where(x < 0, 0, x)

def _relu_derivative(y):
    return np.where(y < 0, 0, 1)

relu = {"f":_relu, "f'":_relu_derivative}

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _sigmoid_derivative(y):
    return y * (1 - y)

sigmoid = {"f":_sigmoid, "f'":_sigmoid_derivative}

class BPNN():
    def __init__(self, model_config):
        # input:
        #   model_config: [[input.length, layer1.length, layer2.length, ..., output_class_num],
        #                              [act_func1,  act_func2,       ..., act_funcN],
        
        # dy/dx = f'(x) = f'(f^-1(y)) = d_in_func(y)
        self.model_config = model_config
        self.layers, self.w, self.b = [], [], []
        self.act_func_dir = []
        self.layer_num = 0
        self.init_model()

    def init_model(self):
        # 初始化 W 和 b 为 0
        for i, layer in enumerate(self.model_config[0]):
            if i == 0:
                continue
            else:
                self.layers.append(np.zeros(layer))
                self.w.append(np.random.rand(self.model_config[0][i-1], layer))
                self.b.append(np.random.rand(layer))
                # 不能初始化为0，否则梯度永远是0
                # self.w.append(np.ones((self.model_config[0][i-1], layer)))
                # self.b.append(np.ones((layer)))
                self.act_func_dir.append(self.model_config[1][i-1])
        self.layer_num = len(self.layers)


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


    def softmax_loss(self, y, p, normalize):
        # input:
        #   y:  图片标签        (batch_size x 1)
        #   p:  预测概率        (batch_size x 10(或者class_num))
        #   normalize: 正则化项 dict
        #       type: 可选 ['none','L1','L2']
        #       reg:  正则化系数
        # output:
        #   avg_loss: 每batch的平均loss     float64
        loss = 0.0
        batch_size = y.shape[0]
        for i, yi in enumerate(y):
            loss -= np.log(p[i][yi])
        loss = loss / batch_size
        if normalize["type"] == 'L1':
            for i, w in enumerate(self.w):
                loss += normalize["reg"] * (np.sum(np.abs(w)) + np.sum(np.abs(self.b[i])))
        elif normalize["type"] == 'L2':
            for i, w in enumerate(self.w):
                loss += normalize["reg"] / 2 * (np.sum(w * w) + np.sum(self.b[i] * self.b[i]))
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


    def train(self, X, y, epoch=10 , lr=0.01, batch_size=3, normalize={"type":'none'}):
        # 训练网络
        # input:
        #   X:  训练图片
        #   y:  训练标签
        #   epoch:  训练次数
        #   lr: 训练速率
        #   batch_size: mini batch 大小 (最好能整除 X 的张数)
        #   normalize: 正则化项 dict
        #       type: 可选 ['none','L1','L2']
        #       reg:  正则化系数
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
                self.optimize(x_batch, y_batch, p, lr, normalize)
                loss_sum += self.softmax_loss(y_batch, p, normalize)
                acc_sum += self.get_acc_avg(y_batch, p)
            loss_list.append(loss_sum / batch_num)
            acc_list.append(acc_sum / batch_num)

        return loss_list, acc_list


    def optimize(self, X, y, p, lr, normalize):
        # input:
        #   X: 图片数据集矩阵  (batch_size x 3072)
        #   y: 图片标签    (batch_size x 1)
        #   p: 预测概率       (batch_size x 10(或者 class_num))
        #   lr: 训练速率
        #   normalize: 正则化项 dict
        #       type: 可选 ['none','L1','L2']
        #       reg:  正则化系数
        d_w, d_b = self.evaluate_analytic_grad(X, y, p, normalize)
        for i in range(self.layer_num):
            self.w[i] -= lr * d_w[i]
            self.b[i] -= lr * d_b[i]


    def evaluate_analytic_grad(self, X, y, p, normalize):
        # input:
        #   X: 图片数据集矩阵  (batch_size x 3072)
        #   y: 图片标签    (batch_size x 1)
        #   p: 预测概率       (batch_size x 10(或者 class_num))
        #   normalize: 正则化项 dict
        #       type: 可选 ['none','L1','L2']
        #       reg:  正则化系数
        batch_size = X.shape[0]
        Y = np.zeros(p.shape)
        for i, label in enumerate(y):
            Y[i][label] = 1
        d_w, d_b = list(range(self.layer_num)), list(range(self.layer_num))
        
        delta = (p - Y) * self.act_func_dir[-1]["f'"](self.layers[-1])
        for i in range(self.layer_num-1, 0, -1):
            d_w[i] = np.dot(self.layers[i-1].T, delta) / batch_size
            d_b[i] = delta.sum(0) / batch_size
            delta = np.dot(delta, self.w[i].T) * self.act_func_dir[i-1]["f'"](self.layers[i-1])
        d_w[0] = np.dot(X.T, delta) / batch_size
        d_b[0] = delta.sum(0) / batch_size
   
        if normalize["type"] == 'L1':
            for i, w in enumerate(self.w):
                d_w[i] += normalize["reg"] * np.sign(w)
                d_b[i] += normalize["reg"] * np.sign(self.b[i])
        elif normalize["type"] == 'L2':
            for i, w in enumerate(self.w):
                d_w[i] += normalize["reg"] * w
                d_b[i] += normalize["reg"] * self.b[i]
        return d_w, d_b

    def evaluate_numerical_gradient(self, X, y, p):
        # input:
        #   X: 图片数据集矩阵  (batch_size x 3072)
        #   y: 图片标签    (batch_size x 1)
        #   p: 预测概率       (batch_size x 10(或者 class_num))
        h = 0.000001
        grad_w, grad_b = [], []
        for i in range(self.layer_num):
            grad_w.append(np.zeros(self.w[i].shape))
            grad_b.append(np.zeros(self.b[i].shape))

        loss = self.softmax_loss(y, p)

        for i, w in enumerate(self.w):
            it = np.nditer(w, flags=["multi_index"])
            while not it.finished:
                iw = it.multi_index
                old_value = w[iw]
                w[iw] += h
                p_h = self.forward(X)
                loss_h = self.softmax_loss(y, p_h)
                w[iw] = old_value
                grad_w[i][iw] = (loss_h - loss) / h
                it.iternext()

        for i, b in enumerate(self.b):
            it = np.nditer(b, flags=["multi_index"])
            while not it.finished:
                ib = it.multi_index
                old_value = b[ib]
                b[ib] += h
                p_h = self.forward(X)
                loss_h = self.softmax_loss(y, p_h)
                b[ib] = old_value
                grad_b[i][ib] = (loss_h - loss) / h
                it.iternext()

        return grad_w, grad_b


    def predict(self, X):
        p = self.forward(X)
        return p.argmax(1)

    def evaluate(self, X, y):
        p = self.forward(X)
        return self.get_acc_avg(y, p)

    