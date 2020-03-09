import data_utils
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import plot
import math
import sys

def img_data_normalize(X_train, X_test):
    return X_train/255 - 0.5, X_test/255 - 0.5

def three_loss_plot():
    loss_list = []
    for i in range(3):
        if i == 0:
            softmax_classifier = softmax([3072,10])
            loss = softmax_classifier.train(X_train, y_train, batch_size=256, epoch=50, lr=0.04, reg=0.00002, normalize_type='none')
            loss_list.append(loss)
        elif  i  == 1:
            softmax_classifier = softmax([3072,10])
            loss = softmax_classifier.train(X_train, y_train, batch_size=256, epoch=50, lr=0.04, reg=0.00002, normalize_type='L1')
            loss_list.append(loss)
        else:
            softmax_classifier = softmax([3072,10])
            loss = softmax_classifier.train(X_train, y_train, batch_size=256, epoch=1, lr=0.04, reg=0.00001, normalize_type='L2')
            loss_list.append(loss)
    plot.three_loss(loss_list)


def reg_plot(reg_num, L_method):
    loss_list = []
    for i in range(1, reg_num):
        softmax_classifier = softmax([3072,10])
        loss = softmax_classifier.train(X_train, y_train, batch_size=256, epoch=50, lr=0.04, reg= 1/(math.pow(10, i)), normalize_type=L_method)
        loss_list.append(loss)
    plot.reg_loss(loss_list)

def find_param_plot(batch_size_list, lr_list, L_method):
    loss_list, train_acc_list = [], []
    for batch_size in batch_size_list:
        for lr in lr_list:
            softmax_classifier = softmax([3072,10])
            loss, best_acc = softmax_classifier.train(X_train, y_train, batch_size=batch_size, epoch=10, lr=lr, reg= 1e-5, normalize_type=L_method)
            loss = min(loss)
            loss_list.append(loss)
            train_acc_list.append(best_acc)
    plot.plot_3d(batch_size_list, lr_list, loss_list, ["batch_size", "learning rate", "loss"], "bs-lr-loss")
    plot.plot_3d(batch_size_list, lr_list, train_acc_list, ["batch_size", "learning rate", "train_acc"], "bs-lr-acc")

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


    def shuffle(self, x, y):
        random_arr = [i for i in range(len(x))]
        random.shuffle(random_arr)
        return x[random_arr], y[random_arr]

    def softmax(self, vector):
        output = np.exp(vector) / np.exp(vector).sum(axis=1).reshape(-1, 1)
        return output


    def softmax_loss(self, labels, scores, reg, normalize_type):
        loss = 0.0
        batch_size = labels.shape[0]
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

    def evaluate_numerical_gradient(self, x_batch, y_batch, scores, reg, normalize_type):
        h = 0.00001
        grad_w = np.zeros(self.w[-1].shape)
        grad_b = np.zeros(self.b[-1].shape)

        loss = self.softmax_loss(y_batch, scores, reg, normalize_type)

        it = np.nditer(self.w[-1], flags=['multi_index'])
        while not it.finished:
            iw = it.multi_index
            old_value = self.w[-1][iw]
            self.w[-1][iw] += h
            score_h = self.forward(x_batch)
            loss_h = self.softmax_loss(y_batch, score_h, reg, normalize_type)
            self.w[-1][iw] = old_value
            grad_w[iw] = (loss_h - loss) / h
            it.iternext()

        it = np.nditer(self.b[-1], flags=['multi_index'])
        while not it.finished:
            ib = it.multi_index
            old_value = self.b[-1][ib]
            self.b[-1][ib] += h
            score_h = self.forward(x_batch)
            loss_h = self.softmax_loss(y_batch, score_h, reg, normalize_type)
            self.b[-1][ib] = old_value
            grad_b[ib] = (loss_h - loss) / h
            it.iternext()
        
        return grad_w, grad_b

    def evaluate_analytic_grad(self, x_batch, y_batch, scores, reg, normalize_type):
        batch_size = x_batch.shape[0]
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
        best_acc = 0
        epoch_list, loss_list = [], []
        best_w = copy.deepcopy(self.w)
        best_b = copy.deepcopy(self.b)
        batch_num = x.shape[0] // batch_size
        for e in range(epoch):
            x, y = self.shuffle(x, y)
            acc_sum = 0 
            for batch in range(batch_num):
                x_batch = x[batch*batch_size:(batch+1)*batch_size]
                y_batch = y[batch*batch_size:(batch+1)*batch_size]
                output = self.forward(x_batch)
                loss = self.softmax_loss(y_batch, output, reg, normalize_type)
                acc = self.get_acc_avg(output, y_batch)
                acc_sum += acc
                self.optimize(x_batch, y_batch, output, lr, reg, normalize_type)
                #print("epoch: %d / %d, batch: %d / %d, loss = %f, acc = %f" % (e + 1, epoch,batch+1,batch_num, loss, acc))
                loss_list.append(loss)
            acc_avg = acc_sum / batch_num
            if best_acc < acc_avg:
                best_acc = acc_avg
                best_w = copy.deepcopy(self.w)
                best_b = copy.deepcopy(self.b)
            print("epoch %d / %d: acc = %f" % (e + 1, epoch, acc_avg))
            epoch_list.append(e+1)

        print("Training complete. Best accuracy is ", best_acc)
        self.w = best_w
        self.b = best_b
        return loss_list, best_acc

    def optimize(self, x_batch, y_batch, scores, lr, reg, normalize_type):
        d_w, d_b = self.evaluate_analytic_grad(x_batch, y_batch, scores, reg, normalize_type)
        # d_w_, d_b_ = self.evaluate_numerical_gradient(x_batch, y_batch, scores, batch_size, reg, normalize_type)
        # print(d_b)
        # print(d_b_)
        # quit()
        self.w[-1] -= lr * d_w
        self.b[-1] -= lr * d_b
        #for layer in self.layers:

    def evaluate(self, x, y):
        output = self.forward(x)
        np.save("y", y)
        np.save("output", output)
        plot.roc(y, output)
        return self.get_acc_avg(output, y)

if __name__ == "__main__":
    
    '''
    print("Doing: load image data")
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10('../cifar-10-batches-py')

    # 图像左右翻转增加数据集
    print("Doing: image enhance")
    data_utils.create_image_enhance_npy(X_train, X_test, y_train, y_test)
    '''
    
    # 加载 npy 文件
    print("Doing: load npy files")
    X_train, X_test, y_train, y_test = data_utils.load_npy()

    # 将单幅图片转成 3072 维的向量
    print("Doing: image data reshape")
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    print("Doing: normalize image data")
    X_train, X_test = img_data_normalize(X_train, X_test)
    #sys.exit()
    # print("Doing: set net size parameter")
    # softmax_classifier = softmax([3072,10])

    # print("Doing: train net")
    softmax_classifier.train(X_train, y_train, batch_size=32, epoch=2, lr=0.005, reg=0.00002, normalize_type='none')
    softmax_classifier.evaluate(X_test, y_test)
    ## 画图
    #three_loss_plot()
    #reg_plot(6, 'L2')
    #find_param_plot(batch_size_list=[16,32,64,128], lr_list=[0.005, 0.01, 0.02, 0.03, 0.04, 0.1], 'L2')
    # print("Doing: test net")
    # acc_test = softmax_classifier.evaluate(X_test, y_test)
    # print("test accuracy is ", acc_test)
