import numpy as np
from module import Layer, Activation

class fc(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = np.random.rand(self.in_size, self.out_size)
        self.bias = np.random.rand(self.out_size)
    
    def init_grad(self):
        self.d_w = np.zeros((self.batch_size, self.in_size, self.out_size))
        self.d_b = np.zeros((self.batch_size, self.out_size))

    def forward(self, input):
        self.input = input
        self.batch_size = input.shape[0]
        self.neuron = np.dot(input, self.weight) + self.bias
        self.init_grad()
        return self.neuron

    def backward(self, upstream_grad):
        self.dneuron = np.dot(self.weight, upstream_grad)
        return self.dneuron

    def local_grad(self, upstream_grad):
        #Note that upstream_grad and local_grad should be batch gradient!!!
        self.d_w = np.zeros((self.out_size, self.in_size, self.out_size))

        #acceleration?
        for i, t in enumerate(self.d_w):
            for j, g in enumerate(t):
                for k, h in enumerate(g):
                    if k == i:
                        self.d_w[i][j][k] = self.input.mean(0)[j]

        self.d_w = np.dot(upstream_grad, self.d_w)
        self.d_b = upstream_grad
        return self.d_w, self.d_b

    def optimize(self, *input):
        self.d_w, self.d_b = input
        self.weight -= self.d_w
        self.bias -= self.d_b

if __name__ == "__main__":
    a = fc(2,3)
    input = np.zeros((16, 2))
    y = np.zeros((1,), dtype=int)
    p = a(input)
    up_grad = np.zeros(3) + 1
    dparam = a.local_grad(up_grad)

class batch_norm1d(Layer):
    def __init__(self, in_size, bias=True, eps=1e-5):
        self.in_size = in_size
        self.eps = eps
        self.y = np.random.rand(self.in_size)
        self.b = np.random.rand(self.in_size)

    def forward(self, input):
        x_hat = input.mean(0)
        x_hat = x_hat / (input.var(0) + self.eps)
        self.output = x_hat * self.y + self.b
        return self.output

class relu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.y = np.where(x < 0, 0, x)
        return self.y

    def backward(self, upstream_grad):
        self.grad = np.where(self.y < 0, 0, 1).mean(0)
        self.grad = np.multiply(self.grad, upstream_grad)
        return self.grad.reshape(-1)

    def optimize(self, *input):
        pass

class softmax(Activation):
    def __init__(self):
        super().__init__()
        self.dscore = np.zeros(0)

    def forward(self, score):
        # input:
        #   vector: 最后一层输出 (batch_size x 10(或者class_num))
        # output:
        #   output: 预测概率     (batch_size x 10(或者class_num))
        self.score = score
        m = np.max(score, axis=1, keepdims=True)
        exp_scores = np.exp(score-m)
        self.prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.prob

    def backward(self, upstream_grad):
        dscore = -np.dot(self.prob.T, self.prob)
        for j in range(dscore.shape[0]):
            dscore[j][j] += self.prob[0][j]
        dscore = np.dot(upstream_grad, dscore)
        return dscore

"""
    def _unact(cls, x):
        return x

    def _unact_derivative(cls, y):
        return np.ones(y.shape)

    def _tanh(cls, x):
        return np.tanh(x)

    def _tanh_derivative(cls, y):
        return 1 - y * y

    def _sigmoid(cls, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(cls, y):
        return y * (1 - y)
"""
