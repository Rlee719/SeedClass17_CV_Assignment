import data_utils
import numpy as np

def img_data_normalization(X_train, X_test):
    return X_train/255 - 0.5, X_test/255 - 0.5

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
        output = np.exp(vector) / np.exp(vector).sum(axis=1).reshape(-1, 1)
        return output


    def softmax_loss(self, label, scores, reg):
        correct_log_probs = -1 * np.log(scores[:, label])
        loss = np.average(correct_log_probs) + reg * (np.sum(np.abs(self.w[-1])) + np.sum(np.abs(self.b[-1])))
        return loss


    def evaluate_numerical_gradient(self, x_batch, y_batch, scores, batch_size, reg):
        h = 0.00001
        grad_w = np.zeros(self.w[-1].shape)
        grad_b = np.zeros(self.b[-1].shape)

        loss = self.softmax_loss(y_batch, scores, reg)

        it = np.nditer(self.w[-1], flags=['multi_index'])
        while not it.finished:
            iw = it.multi_index
            old_value = self.w[-1][iw]
            self.w[-1][iw] += h
            score_h = self.forward(x_batch)
            loss_h = self.softmax_loss(y_batch, score_h, reg)
            self.w[-1][iw] = old_value
            grad_w[iw] = (loss_h - loss) / h
            it.iternext()

        it = np.nditer(self.b[-1], flags=['multi_index'])
        while not it.finished:
            ib = it.multi_index
            old_value = self.b[-1][ib]
            self.b[-1][ib] += h
            score_h = self.forward(x_batch)
            loss_h = self.softmax_loss(y_batch, score_h, reg)
            self.b[-1][ib] = old_value
            grad_b[ib] = (loss_h - loss) / h
            it.iternext()
        
        return grad_w, grad_b

    
    def evaluate_analytic_grad(self, x_batch, y_batch, scores, batch_size, reg):
        scores[:, y_batch] -= 1
        d_w = np.dot(x_batch.T, scores) / batch_size + reg * np.sign(self.w[-1])
        d_b = scores.sum(0) / batch_size + reg * np.sign(self.b[-1])
        scores[:, y_batch] += 1
        return d_w, d_b


    def forward(self, x):
        outputs = []
        cache = x
        for i in range(self.layer_num):
            self.layers[i] = np.dot(cache, self.w[i]) + self.b[i]
            cache = self.layers[i]
        outputs = self.softmax(self.layers[-1])
        #print(self.layers[-1])
        return outputs

    def train(self, x, y, batch_size, epoch, lr, reg):   
        for e in range(epoch):
            batch_num = int(np.floor(x.shape[0] / batch_size))
            for batch in range(batch_num):
                x_batch = x[batch*batch_size:(batch+1)*batch_size]
                y_batch = y[batch*batch_size:(batch+1)*batch_size]
                output = self.forward(x_batch)
                loss = self.softmax_loss(y_batch, output, batch_size)
                print(loss)
                self.optimize(x_batch, y_batch, output, batch_size, lr, reg)
                print("epoch: %d / %d, batch: %d / %d, loss = %f" % (e + 1, epoch,batch,batch_num, loss))
    
    def softmax_loss(self, labels, scores, batch_size):
        loss = 0.0
        for i, label in enumerate(labels):
            #print(scores[i])
            loss +=  -np.log(scores[i][label]) 
        loss /= batch_size
        return loss

    def optimize(self, x_batch, y_batch, scores, batch_size, lr, reg):
        for i, label in enumerate(y_batch):
            scores[i][label] -= 1
        #print(scores.shape)
        #print(x_batch.shape)
        dsoftmax = np.dot(x_batch.T, scores)
        dsoftmax /= batch_size
        #dsoftmax += reg * np.sign(self.w[-1])
        self.w[-1] -= lr * dsoftmax
        #self.b[-1] -= lr * scores.sum(0) / batch_size
        #for layer in self.layers:

    def evaluate(self, x, y):
        output = self.forward(x).argmax(1)
        accuracy = np.sum(output == y) / x.shape[0]
        print(accuracy)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_utils.load_npy()
    X_train = X_train / 255 - 0.5
    softmax_classifier = softmax([3072,10]) #这句参数别改
    #output = softmax_classifier.forward(np.array([[999,2,3], [3,1,2]]))
    softmax_classifier.train(X_train, y_train, batch_size=8, epoch=8, lr=0.0001, reg=0.01)
    softmax_classifier.evaluate(X_test, y_test)
