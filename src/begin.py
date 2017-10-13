#coding: utf-8

from __future__ import print_function
import torch
import numpy as np

def test1():
    x = torch.rand(5, 3)
    print(x)

    y = torch.rand(5, 3)
    print(y)

    result = torch.Tensor(5, 3)

    torch.add(x, y, out=result)

    print(result)

    print(result[:, 1])

def test2():
    a = torch.ones(5)
    print(a)

    b = a.numpy()
    print(b)

    a.add_(1)
    print(a)
    print(b)

    a = np.ones(5)
    b = torch.from_numpy(a)
    np.add(a, 1, out=a)
    print(a)
    print(b)

if __name__ == "__main__":
    test2()