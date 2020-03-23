import numpy as np
import layers
from module import *

import loss
import optimizer

class BPNN(Module):
    def __init__(self, model_config, act_func):
        # input:
        #   model_config: [[input.length, layer1.length, layer2.length, ..., output_class_num],
        #                              [act_func1,  act_func2,       ..., act_funcN],
        
        # dy/dx = f'(x) = f'(f^-1(y)) = d_in_func(y)
        super().__init__()
        self.act_func = act_func
        self.layers = []
        self.layer_num = 0
        self.status = "train"
        self.model_config = model_config
        #self.loss = loss
        self.init_model()

    def init_model(self):
        # 初始化 W 和 b 为 0
        for i, layer in enumerate(self.model_config):
            if i == 0:
                _layer = layer
            else:
                self.layers.append(layers.fc(_layer, layer))
                self.layers.append(getattr(layers, self.act_func)())
                _layer = layer
        self.layers.append(layers.softmax())
        self.layer_num = len(self.layers)

    def forward(self, X):
        # input:
        #   X:  图片数据集矩阵  (batch_size x 3072)
        # output:
        #   p:  预测概率       (batch_size x 10(或者 class_num))
        a = X
        for i in range(self.layer_num):
            a = self.layers[i](a)
            #self.layers[i] = np.dot(a, self.w[i]) + self.b[i]
            #a = self.layers[i] = self.act_func_dir[i]["f"](self.layers[i])
        return a

    def predict(self, X):
        p = self.forward(X)
        return p.argmax(1)

if __name__ == "__main__":
    model = BPNN(model_config=[3072,20,10], act_func="relu")
    loss_func = loss.Loss_Sequential(loss.soft_max_loss(), loss.L2_loss(model.layers, 1))
    input = np.zeros((16, 3072))
    y = np.zeros((16,), dtype=int)
    output = model(input)
    _optimizer = optimizer.MB_SGD(layers=model.layers, loss=loss_func)
    loss = loss_func(y, output)
    _optimizer.optimize()    