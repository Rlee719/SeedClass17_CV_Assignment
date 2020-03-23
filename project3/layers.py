import numpy as np
from module import Layer, Activation

class fc(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = np.random.rand(self.in_size, self.out_size)
        self.bias = np.random.rand(self.out_size)
        self.batch_size = 1
        #self.init_grad()

    def init_grad(self):
        self.batch_size = self.input.shape[0]
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

        #acceleration?
        for i, t in enumerate(self.d_w):
            for j, g in enumerate(t):
                for k, h in enumerate(g):
                    if k == i:
                        self.d_w[i][j][k] += self.input.mean(0)[j]

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
    def __init__(self, in_size, bias=True, eps=1e-8):
        self.in_size = in_size
        self.eps = eps
        self.y = np.random.rand(self.in_size)
        self.b = np.random.rand(self.in_size)
        self.init_grad()

    def forward(self, x):
        gamma, beta, eps = self.y, self.b, self.eps
        N, D = x.shape
        mu = 1./N * np.sum(x, axis = 0)
        xmu = x - mu
        sq = xmu ** 2
        var = 1./N * np.sum(sq, axis = 0)
        sqrtvar = np.sqrt(var + eps)
        ivar = 1./sqrtvar
        xhat = xmu * ivar
        gammax = gamma * xhat
        self.output = gammax + beta
        self.cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
        return self.output

    def init_grad(self):
        self.dbeta = np.zeros(self.in_size)
        self.dgamma = np.zeros(self.in_size)

    def backward(self, upstream_grad):
        xhat,gamma,xmu,ivar,sqrtvar,var,eps = self.cache
        dout = np.expand_dims(upstream_grad, axis=0)
        N,D = dout.shape
        self.dbeta = np.sum(dout, axis=0)
        dgammax = dout #not necessary, but more understandable
        self.dgamma = np.sum(dgammax*xhat, axis=0)
        dxhat = dgammax * gamma
        divar = np.sum(dxhat*xmu, axis=0)
        dxmu1 = dxhat * ivar
        dsqrtvar = -1. /(sqrtvar**2) * divar
        dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
        dsq = 1. /N * np.ones((N,D)) * dvar
        dxmu2 = 2 * xmu * dsq
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
        dx2 = 1. /N * np.ones((N,D)) * dmu
        self.dx = (dx1 + dx2).sum(0)
        return self.dx

    def local_grad(self, *input):
        return self.dgamma, self.dbeta

    def optimize(self, *input):
        dgamma, dbeta = input
        self.y -= dgamma
        self.b -= dbeta

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
