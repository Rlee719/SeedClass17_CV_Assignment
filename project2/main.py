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


    def softmax_loss(self, labels, scores, batch_size, reg, normalize_type):
        loss = 0.0
        for i, label in enumerate(labels):
            loss -= np.log(scores[i][label]) 
        loss /= batch_size
        if normalize_type == 'none':
            return loss
        elif normalize_type == 'L1':
            return loss + reg * (np.sum(np.abs(self.w[-1])) + np.sum(np.abs(self.b[-1])))
        elif normalize_type == 'L2':
            return loss + reg * (np.sum(self.w[-1] * self.w[-1]) + np.sum(self.b[-1] * self.b[-1]))
        else:
            print("Please choose correct normalize type: (none, L1, L2) ")
            quit()


    def get_acc_avg(self, output, y):
        return  np.sum(output.argmax(1) == y) / y.shape[0]

    def evaluate_numerical_gradient(self, x_batch, y_batch, scores, batch_size, reg, normalize_type):
        h = 0.00001
        grad_w = np.zeros(self.w[-1].shape)
        grad_b = np.zeros(self.b[-1].shape)

        loss = self.softmax_loss(y_batch, scores,batch_size, reg, normalize_type)

        it = np.nditer(self.w[-1], flags=['multi_index'])
        while not it.finished:
            iw = it.multi_index
            old_value = self.w[-1][iw]
            self.w[-1][iw] += h
            score_h = self.forward(x_batch)
            loss_h = self.softmax_loss(y_batch, score_h,batch_size, reg, normalize_type)
            self.w[-1][iw] = old_value
            grad_w[iw] = (loss_h - loss) / h
            it.iternext()

        it = np.nditer(self.b[-1], flags=['multi_index'])
        while not it.finished:
            ib = it.multi_index
            old_value = self.b[-1][ib]
            self.b[-1][ib] += h
            score_h = self.forward(x_batch)
            loss_h = self.softmax_loss(y_batch, score_h,batch_size, reg, normalize_type)
            self.b[-1][ib] = old_value
            grad_b[ib] = (loss_h - loss) / h
            it.iternext()
        
        return grad_w, grad_b

    
    def evaluate_analytic_grad(self, x_batch, y_batch, scores, batch_size, reg, normalize_type):
        for i, label in enumerate(y_batch):
            scores[i][label] -= 1
        d_w = np.dot(x_batch.T, scores) / batch_size
        d_b = scores.sum(0) / batch_size

        if normalize_type == 'L1':
            d_w += reg * np.sign(self.w[-1])
            d_b += reg * np.sign(self.b[-1])
        elif normalize_type == 'L2':
            d_w += 2 * reg * self.w[-1]
            d_b += 2 * reg * self.b[-1]
        elif normalize_type != 'none':
            print("Please choose correct normalize type: (none, L1, L2) ")
            quit()
        
        for i, label in enumerate(y_batch):
            scores[i][label] += 1
        
        return d_w, d_b


    def forward(self, x):
        outputs = []
        cache = x
        for i in range(self.layer_num):
            self.layers[i] = np.dot(cache, self.w[i]) + self.b[i]
            cache = self.layers[i]
        outputs = self.softmax(self.layers[-1])
        return outputs


    def train(self, x, y, batch_size, epoch, lr, reg=0, normalize_type='none'):   
        for e in range(epoch):
            batch_num = int(np.floor(x.shape[0] / batch_size))
            for batch in range(batch_num):
                x_batch = x[batch*batch_size:(batch+1)*batch_size]
                y_batch = y[batch*batch_size:(batch+1)*batch_size]
                output = self.forward(x_batch)
                loss = self.softmax_loss(y_batch, output, batch_size, reg, normalize_type)
                self.optimize(x_batch, y_batch, output, batch_size, lr, reg, normalize_type)
                print("epoch: %d / %d, batch: %d / %d, loss = %f, acc = %f" % (e + 1, epoch,batch+1,batch_num, loss, self.get_acc_avg(output, y_batch)))
    

    def optimize(self, x_batch, y_batch, scores, batch_size, lr, reg, normalize_type):
        d_w, d_b = self.evaluate_analytic_grad(x_batch, y_batch, scores, batch_size, reg, normalize_type)
        # d_w_, d_b_ = self.evaluate_numerical_gradient(x_batch, y_batch, scores, batch_size, reg, normalize_type)
        # print(d_b)
        # print(d_b_)
        # quit()
        self.w[-1] -= lr * d_w
        self.b[-1] -= lr * d_b
        #for layer in self.layers:


    def evaluate(self, x, y):
        output = self.forward(x)
        return self.get_acc_avg(output, y)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_utils.load_npy()
    X_train, X_test = img_data_normalization(X_train, X_test)
    softmax_classifier = softmax([3072,10]) #这句参数别改
    #output = softmax_classifier.forward(np.array([[999,2,3], [3,1,2]]))
    softmax_classifier.train(X_train, y_train, batch_size=32, epoch=2, lr=0.15, reg=0.01, normalize_type='none')
    acc_test = softmax_classifier.evaluate(X_test, y_test)
    print(acc_test)
