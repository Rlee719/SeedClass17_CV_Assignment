import numpy as np
# import minpy.numpy as np
import bpnn
import data_utils

if __name__ == "__main__":
    '''
    print("Doing: load image data")
    X_train, y_train, X_test, y_test = data_utils.load_CIFAR10('../cifar-10-batches-py')

    # 图像左右翻转增加数据集
    print("Doing: image enhance")
    data_utils.create_image_enhance_npy(X_train, X_test, y_train, y_test, "../")
    '''
    
    # 加载 npy 文件
    print("Doing: load npy files")
    X_train, X_test, y_train, y_test = data_utils.load_npy("../")

    X_train = X_train[0:1000]
    y_train = y_train[0:1000]

    # 将单幅图片转成 3072 维的向量
    print("Doing: image data reshape")
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    classifier = bpnn.BPNN([[3072,20,10],[bpnn.relu,bpnn.relu,bpnn.relu]])

    print("Doing: train net")
    train_loss_list, train_acc_list = classifier.train(X_train, y_train,lr=0.1,epoch=20, batch_size=10)

    print(train_loss_list)
    print("Doing: test net")
    test_acc = classifier.evaluate(X_test, y_test)

    print("test accuracy is %f" % (test_acc))