import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from d2l import torch as d2l


def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1] - 1


def train_concise_ch11(train_fn, hyperparams, data_iter, num_epochs=4):
    net = nn.Sequential(nn.Linear(5, 1))

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    optimizer = train_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for x, y in data_iter:
            optimizer.zero_grad()
            out = net(x)
            y = y.reshape(out.shape)
            l1 = loss(out, y)
            l1.mean().backward()
            optimizer.step()
            n += x.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n / x.shape[0] / len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss:{animator.Y[0][-1]:.3f},{timer.avg():3f} sec/epoch')


data_iter, _ = get_data_ch11(10)
train = torch.optim.SGD
train_concise_ch11(train, {'lr': 0.01}, data_iter)
plt.show()
