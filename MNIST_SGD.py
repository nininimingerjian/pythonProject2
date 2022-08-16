import torch
import torchvision
import tensorflow

trans = torchvision.transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)


def sgd(params, lr):
    with torch.no_grad():
        for p in params:
            p -= p.grad * lr
            p.grad_zero_()



