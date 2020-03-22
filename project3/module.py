import numpy as np
from enum import IntEnum

class Module(object):
    def __init__(self):
        pass
    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        pass
    def set_train(self):
        self.status = "train"

    def set_eval(self):
        self.status = "eval"

    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        raise NotImplementedError

class Sequential(Module):
    #Connecting Layers together!
    def __init__(self):
        super().__init__()

class Layer(object):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return self.forward(input)
        
    def forward(self):
        raise NotImplementedError

    def backward(self):
        #Note that upstream_grad and local_grad should be batch gradient!!!
        raise NotImplementedError

    def optimize(self, *input):
        pass

    def local_grad(self, *input):
        pass

class Activation(Layer):
    def __init__(self):
        super().__init__()

class Loss():
    def __init__(self):
        self.loss = 0.0
        pass
    def __call__(self, *input):
        return self.forward(*input)

    def forward(self, *input):
        raise NotImplementedError

class Regularization_loss(Loss):
    def __init__(self, layers, reg):
        super().__init__()
        self.layers = layers
        self.reg = reg

class Optimizer():
    def __init__(self, layers, loss, lr=1e-2):
        self.layers = layers
        self.lr = lr
        self.time = 0
        self.cache = []
        self.loss_func = loss
        # for layer in self.layers:
        #     if type(layer) is fc:
        #         self.cache_dvw.append(np.zeros(layer.weight.shape))
        #         self.cache_dvb.append(np.zeros(layer.bias.shape))

    def update_lr_decay(self):
        if self.lr_decay == Learning_rate_decay.step:
            self.lr /= 2
        elif self.lr_decay == Learning_rate_decay.exp:
            self.lr = self.lr * np.exp(- self.lr_k)
        elif self.lr_decay == Learning_rate_decay.inv:
            self.lr = self.lr * ((1+self.lr_k*(self.time-1)) / (1+self.lr_k*self.time))
        self.time += 1

class Learning_rate_decay(IntEnum):
    none = 0
    step = 1
    exp  = 2
    inv  = 3


#unact = {"f":_unact, "f'":_unact_derivative}
#tanh = {"f":_tanh, "f'":_tanh_derivative}
#relu = {"f":_relu, "f'":_relu_derivative}
#sigmoid = {"f":_sigmoid, "f'":_sigmoid_derivative}

