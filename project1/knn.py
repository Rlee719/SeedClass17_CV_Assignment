import numpy as np

class KNN(object):

  def __init__(self):
    pass


  def train(self, X, y):
    self.X_train = X
    self.y_train = y


  def predict(self, X, k=1, num_loops=0):
    dists = self.compute_distance_no_loops(X)

    return self.predict_labels(dists, k=k)


  def compute_distance_no_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    dists = np.abs(self.X_train - X)
    return dists


  def predict_labels(self, dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      closest_y = []
      closest_y = self.y_train[np.argsort(dists[i])][:k]

      y_pred[i] = np.argmax(np.bincount(closest_y))

    return y_pred