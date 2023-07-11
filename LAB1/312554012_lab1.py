import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)


def ReLU(x):
    x = np.maximum(0.00, x)
    return x


def derivative_ReLU(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


# You need to use the following generate functions to create your inputs x, y.
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title("Ground truth", fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

    plt.subplot(1, 2, 2)
    plt.title("Predict result", fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

    plt.show()


def show_learning_curve(epoch, loss):
    plt.title("Learning Curve", fontsize=18)
    # plt.plot(epoch, loss, 'bo-', label='xxx')
    plt.plot(epoch, loss, "bo-", linewidth=1, markersize=2)
    plt.xlabel("epoch", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    # plt.legend(loc = "best", fontsize=10)
    plt.show()


# TODO: Implement simple neural networks with two hidden layers.


def activation(func, arg):
    if func == "sigmoid":
        return derivative_sigmoid(arg)
    elif func == "ReLU":
        return derivative_ReLU(arg)
    else:
        return 1


def forwarding(func, x, w1, w2, w3):
    a0 = x
    z1 = w1 @ a0
    a1 = activation(func, z1)
    z2 = w2 @ a1
    a2 = activation(func, z2)
    z3 = w3 @ a2
    a3 = activation(func, z3)
    pred_y = z3
    return z3


def backpropagation():
    pass


def weight_update():
    pass


def loss():
    pass


if __name__ == "__main__":
    pass
