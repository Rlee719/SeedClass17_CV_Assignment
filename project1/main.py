import numpy as np
from data_utils import load_CIFAR10

cifar10_dir = 'cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

print ('Training data shape: {}'.format(X_train.shape))
print ('Training labels shape: {}'.format(y_train.shape))
print ('Test data shape: {}'.format(X_test.shape))
print ('Test labels shape: {}'.format(y_test.shape))