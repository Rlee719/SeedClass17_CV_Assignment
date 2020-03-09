import numpy as np
import plot

plot.plot_3d([0,1,2], [1,2,3], [3,2,1,1,2,2,3,3,1], ["batch_size", "learning rate", "loss"], "test")
xx = np.arange(-5,5,0.5)
yy = np.arange(-5,5,0.5)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(X)+np.cos(Y)