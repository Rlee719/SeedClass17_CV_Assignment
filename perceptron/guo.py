import numpy as np
import matplotlib.pyplot as plt
import time
import math

def get_training_data(num_observations=500):
    np.random.seed(12)

    x1 = np.random.multivariate_normal([0,0],[[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1,4],[[1, .75], [.75, 1]], num_observations)

    X = np.vstack((x1, x2)).astype(np.float32)
    Y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

    return X, Y

def draw_scatter_and_divide_line(X, D, W):
    num_observations = int(len(D) / 2)
    
    x1 = X[:num_observations, 0]
    y1 = X[:num_observations, 1]

    x2 = X[num_observations:, 0]
    y2 = X[num_observations:, 1]

    colors1 = '#00CED1'
    colors2 = '#DC143C'

    plt.title('Perceptron Result')
    
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.xlim(xmax=4, xmin=-4)
    plt.ylim(ymax=8, ymin=-4)

    plt.scatter(x1, y1, c=colors1, alpha=0.4)
    plt.scatter(x2, y2, c=colors2, alpha=0.4)

    plt.plot([-4, 4], [compute_line_X_to_Y(-4, W), compute_line_X_to_Y(4, W)], color="#000000", linewidth=1)

    plt.savefig("perceptron.png")
    plt.cla()


def draw_iteration_error(ierrs):
    plt.plot([x for x in range(len(ierrs))], ierrs)
    plt.title('Iteration Error Curve')

    plt.ylim(ymin=0)
    plt.xlabel('epoch')
    plt.ylabel('iteration error')

    plt.savefig("iteration_error.png")
    plt.cla()


def classify_func(x):
    # step function
    return 1 if x > 0 else 0


def compute_line_X_to_Y(x, W):
    return -W[1]/W[2] * x - W[0]/W[2]


def initailize(X):
    # set bias
    X = np.insert(X, 0, values=np.ones(1), axis=1)
    W = np.zeros(3)
    return W, X


def compute_actual_output(X, W):
    Y = np.zeros(len(X))
    for j in range(len(X)):
        Y[j] = classify_func(W.dot(X[j].T))
    return Y


def update_weight(W, X, Y, D, r):
    return (W + r * (D - Y).dot(X))
    

def compute_iteration_error(D, Y):
    return np.mean(np.abs(D - Y))


def train_loop(X, W, D, r, threshold):
    # if training is complete, return True
    ierrs = []
    epoch = 0
    max_epoch = 200
    while epoch < max_epoch:
        Y = compute_actual_output(X, W)
        ierr = compute_iteration_error(D, Y)
        ierrs.append(ierr)
        print("trained %d / %d epoch, iteration error is %f." % (epoch+1, max_epoch, ierr))
        if ierr < threshold:
            break
        W = update_weight(W, X, Y, D, r)
        epoch += 1
    return epoch != max_epoch, W, ierrs


if __name__ == "__main__":
    X, D = get_training_data(500)

    ts = time.time()
    W, nX = initailize(X)
    is_complete, W, ierrs = train_loop(nX, W, D, 0.001, 0.003)

    te = time.time()
    print("Use time: %f s" % (te - ts))

    if is_complete:
        print("Training Success:")
    else:
        print("Training Failed:")
    print("W: ", W)
    draw_scatter_and_divide_line(X, D, W)
    draw_iteration_error(ierrs)