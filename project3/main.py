import numpy as np
# import minpy.numpy as np
import BPNN
import data_utils

if __name__ == "__main__":
    '''
    重新创建npy文件
    数据类型float32，精度10^-8
    做了图像翻转及归一化、标准化
    '''
    # Xtr, Ytr, Xte, Yte = data_utils.load_CIFAR10("../cifar-10-batches-py")
    # data_utils.create_image_enhance_npy(Xtr, Xte , Ytr ,Yte,"../")

    X_train, X_test, y_train, y_test = data_utils.load_npy("../")