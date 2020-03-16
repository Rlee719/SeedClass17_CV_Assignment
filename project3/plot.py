import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def draw_plot(x, y, title, xlabel, ylabel, savepath):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y)
    plt.savefig(savepath)
    plt.cla()

def draw_plots(xs, ys, labels, colors, title, xlabel, ylabel, savepath):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in range(len(xs)):
        plt.plot(xs[i],ys[i], label=labels[i], color=colors[i])
    plt.legend()
    plt.savefig(savepath)
    plt.cla()

def draw_plot_3d(X, Y, Z, title, xlabel, ylabel, zlabel, savepath):
    Z = np.array(Z)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    X, Y = np.meshgrid(np.array(X), np.array(Y))
    ax.plot_surface(X, Y, Z.reshape((X.shape[0], X.shape[1])), rstride = 1, cstride = 1, cmap='rainbow', linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.title(title)
    plt.savefig(savepath)
    plt.show()