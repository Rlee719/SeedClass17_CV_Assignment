import numpy as np
from data_utils import load_CIFAR10, extract_CIFAR10_samples
from knn import KNearestNeighbor
import time
import matplotlib.pyplot as plt

def load_data_set():
    # 加载数据集
    cifar10_dir = 'cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # 将单幅图片转成 3072 维的向量
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    # 根据课程需要，将训练集缩小为 1/5

    X_train, y_train = extract_CIFAR10_samples(X_train, y_train, X_train.shape[0] / 5)

    return X_train, y_train, X_test, y_test


def get_best_hyperpramamter(k_to_accuracies, filename=""):
    if filename != '':
        myfile = open(filename, "w")
    best_k = 1
    best_m = "L1"
    best_acc = 0
    for dist_m in k_to_accuracies:
        for k in sorted(k_to_accuracies[dist_m]):
            mean = np.mean(k_to_accuracies[dist_m][k])
            print("dist metric %s mean for k=%d is %f" % (dist_m, k, mean))
            if filename != '':
                myfile.writelines("%s %d %f\n" % (dist_m, k, mean))
            if mean > best_acc:
                best_k = k
                best_acc = mean
                best_m = dist_m
    if filename != '':
        myfile.close()
    print("Best Distance Metric is %s, Best k is %d, Best Train Acc is %f" % (best_m, best_k, best_acc))
    return best_k, best_m


def save_dict_to_file(dict, filename):
    myfile = open(filename, "w")
    myfile.write(str(dict))
    myfile.close()


def cross_validation(X_train, y_train, num_folds, k_choices, m_choices):
    num_test = X_train.shape[0] / num_folds

    # 将训练集分成 num_folds 份
    X_train_folds = np.array(np.array_split(X_train, num_folds))
    y_train_folds = np.array(np.array_split(y_train, num_folds))

    # 保存不同 k 的结果
    k_to_accuracies = dict.fromkeys(m_choices)
    for m in k_to_accuracies:
        k_to_accuracies[m] = {}

    # 交叉验证核心运行代码
    for dist_m in m_choices:
        for n in range(num_folds):
            combinat = [x for x in range(num_folds) if x != n]
            x_training_dat = np.concatenate(X_train_folds[combinat])
            y_training_dat = np.concatenate(y_train_folds[combinat])
            classifier_k = KNearestNeighbor()
            classifier_k.train(x_training_dat, y_training_dat)
            ks_y_cross_validation_pred = classifier_k.predict_labels_diffrent_Ks(X_train_folds[n], k_choices, dist_m)
            for k in range(len(k_choices)):
                # y_cross_validation_pred = classifier_k.predict(X_train_folds[n], k=k_choices[k], dist_m=dist_m)
                # num_correct = np.sum(y_cross_validation_pred == y_train_folds[n])
                num_correct = np.sum(ks_y_cross_validation_pred[k] == y_train_folds[n])
                accuracy = float(num_correct) / num_test
                k_to_accuracies[dist_m].setdefault(k_choices[k], []).append(accuracy)
                print("num_folds: %d / %d, dist_m: %s, k: %d, acc: %f" % (
                    n + 1, num_folds, dist_m, k_choices[k], accuracy))
    return k_to_accuracies


def run_test(best_k, best_m, X_train, y_train, X_test, y_test):
    # 选择最好的 k 值，在测试集中测试
    num_test = X_test.shape[0]
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)

    y_test_pred = classifier.predict(X_test, k=best_k, dist_m=best_m)

    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
    return num_correct, num_test, accuracy


def cro_val_plot(k_choices, k_to_accuracies):
    for dist_m in k_to_accuracies:
        # plot the trend line with error bars that correspond to standard deviation
        accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies[dist_m].items())])
        accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies[dist_m].items())])
        plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std, ecolor='lightgray')
        plt.title('Cross-validation on k for dist metric %s' % dist_m)
        plt.xlabel('k')
        plt.ylabel('Cross-von accuracy')
        plt.savefig('Cross-validation-dist-metric-' + dist_m)
        plt.cla()
        # plt.show()


################################################################################
#                                                                              #
#                                main program                                  #
#                                                                              #
################################################################################

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data_set()
    print('Training data shape: {}'.format(X_train.shape))
    print('Training labels shape: {}'.format(y_train.shape))
    print('Test data shape: {}'.format(X_test.shape))
    print('Test labels shape: {}'.format(y_test.shape))

    # 运行训练
    k_choices = [x for x in range(1, 101)]
    start = time.time()
    print("Running Train Set ...")
    k_to_accuracies = cross_validation(X_train, y_train, 10, k_choices, ['L1','L2','L3'])
    print("Execution Train Time: ", time.time() - start, "s")

    # 绘制 cross_validation 曲线图
    cro_val_plot(k_choices, k_to_accuracies)

    # 保存训练结果
    save_dict_to_file(k_to_accuracies, "train_result.txt")

    # 将所有 k 的取值结果打印，得到最佳超参数
    best_k, best_m = get_best_hyperpramamter(k_to_accuracies, "train_result_mean.txt")

    # 运行测试
    start = time.time()
    print("Running Test Set ...")
    run_test(best_k, best_m, X_train, y_train, X_test, y_test)
    print("Execution Test Time: ", time.time() - start, "s")
