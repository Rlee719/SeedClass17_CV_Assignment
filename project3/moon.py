import numpy as np
import bpnn
import data_utils
import simple_classification as sc
import plot

if __name__ == "__main__":
    X, y = sc.generate_data()

    classifier = bpnn.BPNN([[2,3,3,2],[bpnn.tanh,bpnn.tanh,bpnn.tanh]], 'none')
    optimizer = bpnn.Optimizer(classifier)
    
    optimizer.lr = 1
    optimizer.lr_decay = bpnn.Learning_rate_decay.none
    optimizer.lr_k = 0.1
    optimizer.momentum_type = bpnn.Momentum.none
    optimizer.mu = 0.9
    optimizer.reg_type = bpnn.Regularization.none
    optimizer.reg = 0.005

    classifier.useOpt(optimizer)

    train_loss_list, train_acc_list = classifier.train(X, y, epoch=79,batch_size=50)
    plot.draw_plot(list(range(len(train_acc_list))), train_acc_list, "acc", 'epoch', 'acc', "acc.png")

    sc.visualize(X, y , classifier)

    # xs, acc_list, labels = [], [], []
    # for i in range(5):
    #     classifier = bpnn.BPNN([[2,3 + i,3,2],[bpnn.tanh,bpnn.tanh,bpnn.unact]])
    #     train_loss_list, train_acc_list = classifier.train(X, y, lr=1, epoch=50, batch_size=20)
    #     acc_list.append(train_acc_list)
    #     xs.append([i for i in range(len(train_acc_list))])
    #     labels.append("layer1 node: " + str(i+3))
    # plot.draw_plots(xs, acc_list, labels, ["red", "blue", "green", "yellow", "orange"], "acc", "epoch", "acc", "acc.png")

    # layer1nodes = [i+2 for i in range(10)]
    # lrs = [i/10 for i in range(1, 21)]
    # z = []
    # for i,node in enumerate(layer1nodes):
    #     z.append([])
    #     for lr in lrs:
    #         classifier = bpnn.BPNN([[2,node,3,2],[bpnn.tanh,bpnn.tanh,bpnn.unact]])
    #         train_loss_list, train_acc_list = classifier.train(X, y, lr=lr, epoch=50, batch_size=20)
    #         z[i].append(train_acc_list[-1])
    # plot.draw_plot_3d(layer1nodes, lrs, z, "acc", "layer1 node", "lr", "acc", "acc.png")

    # normallist = ['none', 'L1', 'L2']
    # reg = [0.00001,0.00005,0.0001,0.0005,0.001]
    # z = []
    # for i,t in enumerate(normallist):
    #     z.append([])
    #     for r in reg:
    #         classifier = bpnn.BPNN([[2,3,3,2],[bpnn.tanh,bpnn.tanh,bpnn.tanh,bpnn.unact]])
    #         train_loss_list, train_acc_list = classifier.train(X, y, lr=1, epoch=50, batch_size=20, normalize={"type":t, "reg":r})
    #         z[i].append(train_acc_list[-1])
    # plot.draw_plot_3d([0,1,2], reg, z, "acc", "type", "reg", "acc", "acc.png")

