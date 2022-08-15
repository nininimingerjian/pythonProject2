import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from d2l import torch as d2l

# timer = d2l.Timer()

# a=torch.zeros(256,256)
# b=torch.randn(256,256)
# c=torch.randn(256,256)
# d=timer.start()
# for i in range(256):
#     for j in range(256):
#         a[i,j]=torch.dot(b[i,:],c[:,j])
#
# timer.stop()
# print(timer.sum())
# timer.start()
# for j in range(256):
#     a[:,j]=torch.mv(b,c[:,j])
# timer.stop()
# print(timer.stop())

d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')


def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1] - 1


def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()


def train_ch11(trainer_fn, states, hyperparams, data_iter, features_dim, num_epochs=2):
    w = torch.normal(mean=0.0, std=0.01, size=(features_dim, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    net, loss = lambda x: d2l.linreg(x, w, b), d2l.squared_loss

    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for x, y in data_iter:
            l = loss(net(x), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += x.shape[0]
            if n % 20 == 0:
                timer.stop()
                animator.add(n / x.shape[0] / len(data_iter), (d2l.evaluate_loss(net, data_iter, loss)))
                timer.start()
    print(f'loss:{animator.Y[0][-1]:.3f},{timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]


def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, features_dim = get_data_ch11(batch_size)
    return train_ch11(sgd, None, {'lr': lr}, data_iter, features_dim, num_epochs)


gd_res = train_sgd(1, 1500, 10)
sgd_res = train_sgd(0.005, 1)
mini1_res = train_sgd(0.4, 100)
mini2_res = train_sgd(0.05, 10)
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch_size=100', 'batch_size=10'])
d2l.plt.gca().set_xscale('log')
plt.show()
