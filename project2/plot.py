import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from matplotlib import cm


def mean(li):
    return sum(li)/len(li)

def accuracy(x,y):
    plt.figure()
    plt.xlabel("training epoch")
    plt.ylabel("training accuracy")
    plt.plot(x,y)
    plt.savefig("accuracy-epoch")
    plt.cla()

def loss(x, y):
    plt.figure()
    plt.xlabel("training epoch")
    plt.ylabel("loss")
    plt.plot(x,y)
    plt.savefig("loss-epoch")
    plt.cla()

def three_loss(x,y):
    plt.figure()
    plt.xlabel("training batch")
    plt.ylabel("loss")
    plt.plot([mean(loss[0][i:i+1000])  for i in range(len(loss[0])-1000)], color = 'yellow',  label = 'none')
    plt.plot([mean(loss[1][i:i+1000])  for i in range(len(loss[0])-1000)], color = 'red',    label = 'best L1')
    plt.plot([mean(loss[2][i:i+1000])  for i in range(len(loss[0])-1000)], color = 'skyblue',label = 'best L2')
    plt.legend() 
    plt.savefig("three-loss-epoch")
    plt.cla()

def reg_loss(loss):
    plt.figure()
    plt.xlabel("training batch")
    plt.ylabel("loss")
    plt.plot([mean(loss[0][i:i+1000])  for i in range(len(loss[0])-1000)], color = 'yellow',  label = 'reg = 0.1')
    plt.plot([mean(loss[1][i:i+1000])  for i in range(len(loss[0])-1000)], color = 'red',    label = 'reg = 0.01')
    plt.plot([mean(loss[2][i:i+1000])  for i in range(len(loss[0])-1000)], color = 'skyblue',label = 'reg = 0.001')
    plt.plot([mean(loss[3][i:i+1000])  for i in range(len(loss[0])-1000)], color = 'green',label = 'reg = 0.0001')
    plt.plot([mean(loss[4][i:i+100])  for i in range(len(loss[0])-100)], color = 'navy',label = 'reg = 0.00001')
    plt.legend() 
    plt.savefig("reg-loss")
    plt.cla()
    
def plot_3d(X, Y, Z, label, save_name):
    Z = np.array(Z)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    ax.set_zlabel(label[2])
    X, Y = np.meshgrid(np.array(X), np.array(Y))
    ax.plot_surface(X, Y, Z.reshape((X.shape[0], X.shape[1])), rstride = 1, cstride = 1, cmap='rainbow', linewidth=0, antialiased=False)
    #ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.legend() 
    plt.savefig(save_name)
    #plt.show()
    plt.cla()


def roc(Y_test, Y_pred): 
    n_classes = 10
    Y_test = label_binarize(Y_test, classes=[i for i in range(n_classes)])
    Y_pred = label_binarize(Y_pred, classes=[i for i in range(n_classes)])
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'purple', 'yellow', 'skyblue', 'red', 'silver', 'silver', 'maroon'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("roc")
    plt.cla()
