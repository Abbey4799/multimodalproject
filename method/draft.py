import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import seed_everything
import random
import numpy as np


def test(a, b, c=3):
    print(a + b + c)


def show(batch_size, num_epochs, model_name):
    print(batch_size, num_epochs, model_name)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='macbert-base-chinese')

    return parser.parse_args()


if __name__ == '__main__':
    # d = {'a': 1, 'b': 2}
    # test(**d, c=5)
    seed_everything()
    a = torch.rand(5, 10)
    b = torch.rand(5, 10)
    cos = nn.CosineSimilarity(dim=-1)
    outputs = cos(a.unsqueeze(1), b.unsqueeze(0))
    print(cos(a[0], b[1]))
    print(cos(a[0], b[2]))
    print(cos(a[0], b[3]))
    print(cos(a[0], b[4]))
    print(outputs)
    print(F.cosine_similarity(a, b))

    x = torch.rand(1, 5)
    y = torch.rand(5, 5)
    print(F.cosine_similarity(x, y, dim=-1))
    x = x.expand(5, -1)
    print(F.cosine_similarity(x, y, dim=-1))

    a = torch.randn(4)
    print(a)
    val, idx = torch.max(a, dim=-1)
    print(val)
    print(idx)
    vals, idxs = torch.topk(a, min(a.size(0), 3))
    print(vals)
    print(idxs)



