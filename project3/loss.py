import numpy as np
import layers
import torch
from module import Loss, Regularization_loss

class soft_max_loss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        # input:
        #   y:  图片标签        (batch_size x 1)
        #   p:  预测概率        (batch_size x 10(或者class_num))
        # output:
        #   avg_loss: 每batch的平均loss     float64
        self.loss = 0.0
        self.y, self.probs = input
        batch_size = self.y.shape[0]
        print()
        for i, yi in enumerate(self.y):
            self.loss -= np.log(self.probs[i][yi])
        self.loss = self.loss / batch_size
        return self.loss

    def backward(self, *input):
        self.dprobs = 1 / self.probs
        return self.dprobs

class L1_loss(Regularization_loss):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        #if self.optimizer.reg_type == Regularization.L1:
        self.loss = 0.0
        for layer in self.layers:
            if type(layer) is layers.fc:
                self.loss += self.reg * (np.sum(np.abs(layer.weight)) + np.sum(np.abs(layer.bias)))
        return self.loss

    def backward(self, *input):
        for layer in self.layers:
            if type(layer) is layers.fc:
                layer.d_w += self.reg * np.sign(layer.weight)
                layer.d_b += self.reg * np.sign(layer.bias)
        return 0

class L2_loss(Regularization_loss):
    def __init__(self, *input):
        super().__init__(*input)

    def forward(self, *input):
        #if self.optimizer.reg_type == Regularization.L2:
        self.loss = 0.0
        for layer in self.layers:
            if type(layer) is layers.fc:
                self.loss += self.reg / 2 * (np.sum(layer.weight * layer.weight) + np.sum(layer.bias * layer.bias))
        return self.loss

    def backward(self, *inputs):
        for layer in self.layers:
            if type(layer) is layers.fc:
                layer.d_w += self.reg * layer.weight
                layer.d_b += self.reg * layer.bias
        return 0
    
class Loss_Sequential(Loss):
    def __init__(self, *loss):
        self.adapted_loss = loss
        self.loss = 0.0

    def forward(self, *input):
        self.loss = 0.0
        for loss in self.adapted_loss:
            self.loss += loss.forward(*input)
        return self.loss

    def backward(self, *input): 
        for loss in self.adapted_loss:
            try: 
                dp += loss.backward()
            except:
                dp = loss.backward()
        return dp.mean(0)
