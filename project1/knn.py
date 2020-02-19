import numpy as np

class KNearestNeighbor(object):

  def __init__(self):
    pass


  def train(self, X, y):
    # X 是训练集 图片数据
    # y 是训练集 标签
    self.X_train = X
    self.y_train = y


  def predict(self, X, k=1, dist_m='L1'):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train), dtype="float32")
    if dist_m == 'L1':
      for i in range(num_test):
        dists[i] = np.sum(np.abs(X[i] - self.X_train), axis=1)
    elif dist_m == 'L2':
      dists = np.sqrt((X**2).sum(axis=1, keepdims=True) + (self.X_train**2).sum(axis=1) - 2 * X.dot(self.X_train.T))
    else:
      print('Invalid value %s for dist_m' % dist_m)
    return self.predict_labels(dists, k)


  def predict_labels(self, dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # 排序选择最小 k 个
      closest_y = self.y_train[np.argsort(dists[i])][:k]
      # k 个中数量最多的作为 预测值
      y_pred[i] = np.argmax(np.bincount(closest_y))

    return y_pred