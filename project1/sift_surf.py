import numpy as np
import cv2
from data_utils import load_CIFAR10
from matplotlib import pyplot as plt

def sift_extractor(X_train):
    feature_vectors = np.zeros(0, np.int)
    for img in X_train:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        KeyPoint, describs = sift.detectAndCompute(img, None)
        feature_vector = describs.reshape((-1)).astype(np.int)
        print(feature_vector.shape)
    #print(feature_vector.shape)
    return feature_vector
    #img = cv2.drawKeypoints(img, KeyPoint, img)
    #plt.imshow(img)
    #plt.show() 

def surf_extractor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SURF_create()
    KeyPoint, describs = sift.detectAndCompute(img, None)
    print(KeyPoint)
    cv2.drawKeypoints(img, KeyPoint, img)
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_CIFAR10('cifar-10-batches-py')
    sift_extractor(X_train)

    #print(img)
    #plt.imshow(img)
    #plt.show()
    #cv2.waitKey(0)
