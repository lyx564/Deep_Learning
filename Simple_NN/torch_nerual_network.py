import torch
import torch.nn as nn
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 读取数据
    filename = 'data/pima_data.csv'
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names=names)
    array = data.values
    X = np.array(array[:, 0:8])
    Y = np.array(array[:, 8])
    X_normed = X / X.max(axis=0)

    X_train = X_normed[:700]
    X_test = X_normed[701:]

    Y_train = Y[:700]
    Y_test = Y[701:]

    X_train = torch.tensor(X_train).float()
    Y_train = torch.tensor(Y_train).long()

    X_test = torch.tensor(X_test).float()
    Y_test = torch.tensor(Y_test).long()
    # 搭建网络
    myNet = nn.Sequential(
        nn.Linear(8, 5, bias=True),
        nn.Sigmoid(),
        nn.Linear(5, 2, bias=True),
        nn.Softmax()
    )

    # 设置优化器
    optimzer = torch.optim.SGD(myNet.parameters(), lr=0.005)
    loss_func = nn.CrossEntropyLoss()
    plot_x = []
    plot_y = []
    for epoch in range(500):
        out = myNet(X_train)
        loss = loss_func(out, Y_train)
        print('Epoch=', epoch+1, ' train_loss=', loss)
        plot_x.append(epoch + 1)
        plot_y.append(loss)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(plot_x, plot_y)
    plt.show()

    y_ = np.array(myNet(X_test).data)

    hit = 0
    for i in range(len(y_)):
        predict = list(y_[i])
        pr = predict.index(max(predict))
        if pr == Y_test[i]:
            hit += 1
    accuracy = hit / len(Y_test)
    print('accuracy=', accuracy)

