from datasets import *
from torch.utils.data.dataset import Dataset
import torch
import os


def func(flag):
    if flag:
        return get()
    else:
        for i in get():
            pass

def get():
    for i in range(10):
        yield i

generator = func(True)
for _ in generator:
    print(_)