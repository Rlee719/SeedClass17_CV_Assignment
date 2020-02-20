import pickle
import os
import numpy as np
import random

def load_CIFAR_batch(filename):
  with open(filename, 'rb') as f:
    #print(filename)
    datadict = pickle.load(f, encoding='bytes')
    X = datadict[b'data']
    Y = datadict[b'labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y = np.array(Y, dtype="uint8")
    return X, Y


def load_CIFAR10(ROOT):
  xs = []
  ys = []
  for b in range(1, 6):
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

def extract_CIFAR10_samples(X, Y, nums):
  if len(X) != len(Y):
    print("X and Y must have the same length! (X: %d, Y: %d)" % (len(X), len(Y)))
    return
  r = [x for x in range(len(X))]
  random.shuffle(r)
  r = r[:int(nums)]
  _X = []
  _Y = []
  for i in r:
    _X.append(X[i])
    _Y.append(Y[i])
  return np.array(_X), np.array(_Y)
  
def generate_test_npy():
    X_train, y_train, X_test, y_test = load_CIFAR10('cifar-10-batches-py')
    for i in range(50):
        np.save("test_npy/img"+str(i), X_train[i+10])

if __name__ == "__main__":
    generate_test_npy()