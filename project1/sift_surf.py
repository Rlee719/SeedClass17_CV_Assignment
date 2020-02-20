import numpy as np
import cv2
from matplotlib import pyplot as plt

def sift_extractor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    KeyPoint, describs = sift.detectAndCompute(img, None)
    feature_vector = describs.reshape((-1))
    print(feature_vector.shape)
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
    for i in range(50):
        img = np.load("test_npy/img"+str(i)+".npy").astype(np.uint8)
        sift_extractor(img)

    #print(img)
    #plt.imshow(img)
    #plt.show()
    #cv2.waitKey(0)
