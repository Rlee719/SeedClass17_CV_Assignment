import numpy as np
import bpnn
import data_utils
import simple_classification as sc

if __name__ == "__main__":
    X, y = sc.generate_data()
    classifier = bpnn.BPNN([[2,5,5,5,2],[bpnn.tanh,bpnn.tanh,bpnn.tanh,bpnn.unact]])
    train_loss_list, train_acc_list = classifier.train(X, y, lr=1, epoch=41, batch_size=20)

    sc.visualize(X, y, classifier)
