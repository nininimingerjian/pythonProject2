import matplotlib.pyplot as plt
import numpy as np
import torch
from d2l import torch as d2l


# def f(x):
#     return x ** 2
#
#
# def f_grad(x):
#     return 2 * x
#
#
# def dg(eta, f_grad):
#     x = 10.0
#     results = [x]
#     for i in range(10):
#         x -= eta * f_grad(x)
#         results.append(x)
#     print(f'epoch10,x:{x:f}')
#     return results


def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = torch.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])


def train_2d(trainer, steps=20, f_grad=None):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
        print(f'epoch{i + 1},x1:{float(x1):f},x2:{float(x2)}:f')
    return results


def show_trace_2d(f, results):
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = torch.meshgrid(torch.arange(-5.5, 1.0, 0.1),
                            torch.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')


def f_2d(x1, x2):
    return x1 ** 2, 2 * x2 ** 2


def f_2d_grad(x1, x2):
    return 2 * x1, 4 * x2


def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)


eta = 0.1

# show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
c = torch.tensor(0.5)


def f(x):
    return torch.cosh(c * x)


def f_grad(x):
    return c * torch.sinh(c * x)


def f_hess(x):
    return c ** 2 * torch.cosh(c * x)


def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch10,x:', x)
    return results


show_trace(newton(), f)

plt.show()
