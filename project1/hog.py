import numpy as np
import cv2
import sys
from data_utils import load_CIFAR10

def hog_extractor(imgs, cell_size, bin_num):
# input: one data_img (H, W, C), for example (32, 32, 3)
# output hog feature (M, )
    hists = np.zeros(0, np.int)
    imgs = imgs.astype(np.int16)
    for i, img in enumerate(imgs):
        print(i)
        x = cv2.Sobel(img,cv2.CV_16S,1,0)
        y = cv2.Sobel(img,cv2.CV_16S,0,1)
        sita = np.arctan(y/(x+1))
        cell_num = int(img.shape[0] / cell_size)
        hist = np.zeros(0, np.int)
        for x in range(cell_num):
            for y in range(cell_num):
                hist = np.concatenate((histogram(sita[(x*cell_size):((x+1)*cell_size) \
                                ,(y*cell_size):((y+1)*cell_size),:], bin_num), hist))
        
        if i == 0:
            hists = hist
        else:
            hists = np.vstack((hists, hist))
    return hists

def histogram(cell, bin_num):
    sita_bin_length = np.pi / bin_num / 2
    hist = np.zeros(bin_num, np.int)
    for x in range(cell.shape[0]):
        for y in range(cell.shape[1]):
            for c in range(cell.shape[2]):
                pixel = cell[x][y][c]
                for b in range(bin_num):
                    if - np.pi / 2 + sita_bin_length*b< pixel <= - np.pi / 2 + sita_bin_length * (b+1):
                        hist[b] += 1
                        break
    return hist

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_CIFAR10('cifar-10-batches-py')
    #np.save("img", X_train[300])
    hog_X_train = hog_extractor(X_train, cell_size=4, bin_num=12)
    hog_X_test = hog_extractor(X_test, cell_size=4, bin_num=12)
    np.save("hog_X_train", hog_X_train)
    np.save("hog_X_test", hog_X_test)
    pass