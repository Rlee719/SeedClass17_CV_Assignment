'''
    调用示例：
        classifier = bpnn.BPNN([[2, 3, 2, 2],[bpnn.tanh, bpnn.relu, bpnn.sigmoid]])
    可选的激活函数有：
        sigmoid
        tanh
        relu
'''

import numpy as np

class Module(object):
    def __init__(self):
        self.params = []    

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if type(value) is Layer:
            self.params.append({"name": key, "value": value})

    def raise_params(self):
        for param in self.params:
            yield param

class Layer(object):
    def __init__(self):
        super().__init__()  

    def raise_params(self):
        for param in self.params:
            yield param

class Activation(Layer):
    def __init__(self):
        pass

    @classmethod
    def _unact(cls, x):
        return x
    @classmethod
    def _unact_derivative(cls, y):
        return np.ones(y.shape)
    @classmethod
    def _tanh(cls, x):
        return np.tanh(x)
    @classmethod
    def _tanh_derivative(cls, y):
        return 1 - y * y
    @classmethod
    def _relu(cls, x):
        return np.where(x < 0, 0, x)
    @classmethod
    def _relu_derivative(cls, y):
        return np.where(y < 0, 0, 1)
    @classmethod
    def _sigmoid(cls, x):
        return 1 / (1 + np.exp(-x))
    @classmethod
    def _sigmoid_derivative(cls, y):
        return y * (1 - y)

#unact = {"f":_unact, "f'":_unact_derivative}
#tanh = {"f":_tanh, "f'":_tanh_derivative}
#relu = {"f":_relu, "f'":_relu_derivative}
#sigmoid = {"f":_sigmoid, "f'":_sigmoid_derivative}

class fc(Layer):
    def __init__(self, in_size, out_size, activation="_relu"):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = np.random.rand(self.in_size, self.out_size) #name must be weight!
        self.bias = np.random.rand(self.out_size) #name must be bias!
        self.activation = activation

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        self.neuron = np.dot(input, self.weight) + self.bias
        self.neuron = getattr(Activation, self.activation)(self.neuron)
        return self.neuron

    def compute_local_grad(self):
        pass

class batch_norm1d(Layer):
    def __init__(self, in_size, bias=True, eps=1e-5):
        self.in_size = in_size
        self.eps = eps
        self.y = np.random.rand(self.in_size)
        self.b = np.random.rand(self.in_size)

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        x_hat = input.mean(0)
        x_hat = x_hat / (input.var(0) + self.eps)
        self.output = x_hat * self.y + self.b
        return self.output

class xavier(Module):
    def __init__(self, module):
        for param in module.raise_params():
            if param["name"] == "weight" or "bias": 
                array_shape = param["value"].shape()
                param["value"] = np.random.rand(*array_shape) #change init method

class BPNN(Module):
    def __init__(self, model_config, act_func):
        # input:
        #   model_config: [[input.length, layer1.length, layer2.length, ..., output_class_num],
        #                              [act_func1,  act_func2,       ..., act_funcN],
        
        # dy/dx = f'(x) = f'(f^-1(y)) = d_in_func(y)
        super().__init__()
        self.act_func = act_func
        #, self.w, selfmodel_config.b = [], [], []
        self.layers = []
        #self.act_func_dir = []
        self.layer_num = 0
        self.status = "train"
        self.model_config = model_config
        #self.loss = loss
        self.init_model()

    def set_train(self):
        self.status = "train"

    def set_eval(self):
        self.status = "eval"

    #self.layer: forward, grad
    #self.w, self.b: normalization
    def init_model(self):
        # 初始化 W 和 b 为 0
        for i, layer in enumerate(self.model_config):
            if i == 0:
                _layer = layer
            else:
                self.layers.append(fc(_layer, layer, activation=self.act_func))
                _layer = layer
                #self.w.append(np.random.rand(self.model_config[0][i-1], layer))
                #self.b.append(np.random.rand(layer))
                # 不能初始化为0，否则梯度永远是0
                # self.w.append(np.ones((self.model_config[0][i-1], layer)))
                # self.b.append(np.ones((layer)))
                #self.act_func_dir.append(self.model_config[1][i-1])
        self.layer_num = len(self.layers)

    def useOpt(self, optimizer):
        self.optimizer = optimizer    

    def softmax(self, vector):
        # input:
        #   vector: 最后一层输出 (batch_size x 10(或者class_num))
        # output:
        #   output: 预测概率     (batch_size x 10(或者class_num))
        m = np.max(vector, axis=1, keepdims=True)
        exp_scores = np.exp(vector-m)
        output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return output

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
        p = self.softmax(a)
        return p

    def predict(self, X):
        p = self.forward(X)
        return p.argmax(1)