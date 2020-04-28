import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


# 手动实现一个简单神经网络
class NerualNetwork:
    def __init__(self):
        self.ws = []
        self.bs = []
        self.input = []
        self.w_gradient = []
        self.b_gradient = []
        self.activate = []
        self.activate_gradient = []

    # 参数初始化
    def inital_parameters(self, m, n):
        res = []
        for i in range(m):
            res.append(np.random.randn(n))
        return np.array(res)

    # 全连接层
    def fully_connected_layer(self, x, layer):
        x, w, b = np.array(x), np.array(self.ws[layer]), np.array(self.bs[layer])
        assert len(x[0]) == len(w[0])
        assert len(w) == len(b[0])
        self.input.append(x)
        pre_activation = np.sum([np.matmul(x, w.T), b], axis=0)
        post_activation = self.activate[layer](pre_activation)
        return post_activation

    # BP算法
    def backward_propagation(self, learning_rate, Y, mini_batch_size):
        # mini_batch gradient_descent
        for i in range(len(self.ws) - 1, -1, -1):
            wi = self.ws[i]
            bi = self.bs[i]

            # 初始化当前层w的梯度
            w_gradients = [[0 for _ in range(len(wi[0]))] for _ in range(len(wi))]
            # 初始化当前层b的梯度
            b_gradients = [[0 for _ in range(len(bi[0]))]]

            # 上一层的w梯度
            w_last_gradient = self.w_gradient[len(self.ws) - i - 1]
            # 上一层b梯度
            b_last_gradient = self.b_gradient[len(self.bs) - i - 1]

            a_f_gra = self.activate_gradient[i]
            # mini_batch sample
            sample = np.random.randint(len(Y), size=mini_batch_size)

            # 计算当前层w的梯度
            for l in range(len(wi)):
                w = wi[l]
                for j in range(len(w)):
                    for k in sample:
                        input = self.input[i][k]
                        for m in range(len(w_last_gradient)):
                            w_gradients[l][j] += learning_rate * w_last_gradient[m][l] * a_f_gra(input, j) * input[j]

            # 计算当前层b的梯度
            for l in range(len(bi)):
                b = bi[l]
                for j in range(len(b)):
                    for k in sample:
                        input = self.input[i][k]
                        for m in range(len(b_last_gradient)):
                            b_gradients[l][j] += learning_rate * b_last_gradient[m][l] * a_f_gra(input, j)

            self.w_gradient.append(np.array(w_gradients))
            self.b_gradient.append(np.array(b_gradients))

        # 把初始梯度删去，然后因为是倒着添加梯度的，所以要倒过来
        self.w_gradient.pop(0)
        self.w_gradient = self.w_gradient[::-1]

        self.b_gradient.pop(0)
        self.b_gradient = self.b_gradient[::-1]

        # w梯度下降
        for i in range(len(self.ws)):
            wi = self.ws[i]
            for j in range(len(wi)):
                w = wi[j]
                for k in range(len(w)):
                    self.ws[i][j][k] -= self.w_gradient[i][j][k]

        # b梯度下降
        for i in range(len(self.bs)):
            bi = self.bs[i]
            for j in range(len(bi)):
                b = bi[j]
                for k in range(len(b)):
                    self.bs[i][j][k] -= self.b_gradient[i][j][k]

        # 一轮梯度下降后清空
        self.w_gradient = []
        self.b_gradient = []
        self.input = []

    # 交叉熵损失函数
    def cross_entropy(self, y_, Y):
        loss = 0
        # 记录初始梯度
        w_gradients = [[0, 0]]
        for i in range(len(y_)):
            predict = y_[i]
            for j in range(len(predict)):
                # +0.001，避免除0
                w_gradients[0][j] -= 1 / (predict[j] + 0.0001)
                if j == Y[i]:
                    loss -= np.log(predict[j])
                    break
        loss /= len(y_)
        w_gradients[0][0] /= len(y_)
        w_gradients[0][1] /= len(y_)
        self.w_gradient.append(np.array(w_gradients))
        self.b_gradient.append(np.array(w_gradients))
        return loss


# 激活函数
class ActivationFunction:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def liner(self, x):
        return x + 1

    def softmax(self, x):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        s = x_exp / x_sum
        return s

    # sigmoid的求导函数
    def sigmoid_gradient(self, x, i):
        return sigmoid(x[i]) * (sigmoid(x[i])-1)

    # softmax的求导函数
    def softmax_gradient(self, x, i):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=0, keepdims=True)
        s = x_exp / x_sum
        return s[i] * (1 - s[i])


if __name__ == '__main__':
    # 读取数据
    filename = 'data/pima_data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names=names)
    array = data.values
    X = np.array(array[:, 0:8])
    Y = array[:, 8]
    X_normed = X / X.max(axis=0)

    X_train = X_normed[:700]
    X_test = X_normed[701:]

    Y_train = Y[:700]
    Y_test = Y[701:]

    nn = NerualNetwork()
    sigmoid = ActivationFunction().sigmoid
    softmax = ActivationFunction().softmax

    sigmoid_gradient = ActivationFunction().sigmoid_gradient
    softmax_gradient = ActivationFunction().softmax_gradient

    w1, b1 = nn.inital_parameters(5, 8), nn.inital_parameters(1, 5)
    w2, b2 = nn.inital_parameters(2, 5), nn.inital_parameters(1, 2)

    # 导入初始化参数
    nn.ws = [w1, w2]
    nn.bs = [b1, b2]
    # 导入激活函数及其导数
    nn.activate = [sigmoid, softmax]
    nn.activate_gradient = [sigmoid_gradient, softmax_gradient]

    # 训练
    epoch = 500
    learning_rate = 0.005
    mini_batch_size = 1
    plot_x = []
    plot_y = []
    for i in range(epoch):
        f_c1 = nn.fully_connected_layer(X_train, 0)
        y_ = nn.fully_connected_layer(f_c1, 1)
        train_loss = nn.cross_entropy(y_, Y_train)
        print('Epoch:', i + 1, ', train_loss=', train_loss)
        plot_x.append(i + 1)
        plot_y.append(train_loss)
        nn.backward_propagation(learning_rate, Y_train, mini_batch_size)

    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(plot_x, plot_y)
    plt.show()

    # 输出最终参数结果
    print(nn.ws)
    print(nn.bs)

    # 测试
    f_c1 = nn.fully_connected_layer(X_test, 0)
    y_ = nn.fully_connected_layer(f_c1, 1)
    hit = 0
    for i in range(len(y_)):
        predict = list(y_[i])
        pr = predict.index(max(predict))
        if pr == Y_test[i]:
            hit += 1
    accuracy = hit / len(Y_test)
    print('accuracy=', accuracy)
