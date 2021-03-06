import numpy as np
import pickle
import os
import random

def load_CIFAR_batch(filename):
  with open(filename, 'rb') as f:
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


def create_image_npy(X_train, X_test, y_train, y_test):
  if os.path.isfile("X_train") and os.path.isfile("y_train") and os.path.isfile("X_test") and os.path.isfile("y_test"):
    print("npy files exist")
  else:
    print("create npy files:")
    np.save("X_train", X_train)
    np.save("y_train", y_train)
    np.save("X_test", X_test)
    np.save("y_test", y_test)
    print("create npy files success")


def create_image_enhance_npy(X_train, X_test, y_train, y_test):
  if os.path.isfile("X_train") and os.path.isfile("y_train") and os.path.isfile("X_test") and os.path.isfile("y_test"):
    print("npy files exist")
  else:
    data_size = X_train.shape[0]
    newX, newY = [], []
    print("create npy files:")
    for i in range(data_size):
        newX.append(X_train[i])
        newX.append(np.fliplr(X_train[i]))
        newY.append(y_train[i])
        newY.append(y_train[i])
    X_train = np.array(newX)
    y_train = np.array(newY)
    np.save("X_train", X_train)
    np.save("y_train", y_train)
    np.save("X_test", X_test)
    np.save("y_test", y_test)
    print("create npy files success")


def load_npy():
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")
    return X_train, X_test, y_train, y_test