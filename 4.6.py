import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l


def dropout_layer(x, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(x.shape)
    elif dropout == 0:
        return x
    else:
        mask = (torch.randn(x.shape) > dropout).float()
        return mask * x / (1 - dropout)


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.lin1(x.reshape((-1, num_inputs))))
        if self.training:
            h1 = dropout_layer(h1, dropout1)
        h2 = self.relu(self.lin2(h1))
        if self.training:
            h2 = dropout_layer(h2, dropout2)
        h3 = self.lin3(h2)
        return h3


# net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(dropout1),
                    nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout2),
                    nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

plt.show()
