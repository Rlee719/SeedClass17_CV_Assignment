import pickle
import numpy as np
import os

def load_CIFAR_batch(filename):
  with open(filename, 'rb') as f:
    #print(filename)
    datadict = pickle.load(f, encoding='bytes')
    print(datadict.keys())
    X = datadict[b'data']
    Y = datadict[b'labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y



def load_CIFAR10(ROOT):
  xs = []
  ys = []
  for b in range(1, 5):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    print("loading file: %s" % f)
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte