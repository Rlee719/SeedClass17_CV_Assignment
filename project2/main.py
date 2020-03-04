import data_utils
import numpy as np

class softmax():
    def __init__(self, model_config):
        #model_config = [input_length, layer1_length, layer2_length, ... output_classes_num]
        self.model_config = model_config
        self.layers, self.w, self.b = [], [], []
        self.init_model()

    def init_model(self):
        for i, layer in enumerate(self.model_config):
            if i == 0:
                continue
            else:
                self.layers.append(np.zeros(layer))
                self.w.append(np.zeros((self.model_config[i-1], layer)))
                self.b.append(np.zeros(layer))
        self.layer_num = len(self.layers)

    def softmax(self, vector):
        output = np.exp(vector)/sum(np.exp(vector))
        return output

    def forward(self, x):
        outputs = []
        cache = x
        for i in range(self.layer_num):
            self.layers[i] = np.dot(cache, self.w[i]) + self.b[i]
            cache = self.layers[i]
        output = self.softmax(self.layers[-1])
        outputs.append(output)
        outputs = np.array(outputs).squeeze(0)
        return outputs

    def train(self, x, y, batch_size, epoch, lr, reg):   
        for e in range(epoch):
            for batch in range(int(np.floor(x.shape[0] / batch_size))):
                x_batch = x[batch*batch_size:(batch+1)*batch_size]
                y_batch = y[batch*batch_size:(batch+1)*batch_size]
                output = self.forward(x_batch)
                loss = self.softmax_loss(y_batch, output)
                self.optimize(x_batch, y_batch, output, batch_size, lr, reg)
    
    def softmax_loss(self, label, scores):
        correct_log_probs = -np.log(scores[:, label])
        loss = np.sum(correct_log_probs)
        return loss

    def optimize(self, x_batch, y_batch, scores, batch_size, lr, reg):
        scores[:, y_batch] -= 1
        dsoftmax = np.dot(x_batch.T, scores)
        dsoftmax /= batch_size
        dsoftmax += reg * np.sign(self.w[-1])
        self.w[-1] -= lr*dsoftmax
        self.b[-1] -= lr*scores.sum(0) / batch_size
        #for layer in self.layers:

    def evaluate(self, x, y):
        output = self.forward(x).argmax(1)
        accuracy = np.sum(output == y) / x.shape[0]

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_utils.load_npy()
    softmax_classifier = softmax([3072,10]) #这句参数别改
    #output = softmax_classifier.forward(np.array([[999,2,3], [3,1,2]]))
    softmax_classifier.train(X_train, y_train, batch_size=16, epoch=16, lr=0.01, reg=0.01)
    softmax_classifier.evaluate(X_test, y_test)
