import sys
sys.path.append('../')
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, rmsprop
import numpy as np
import inspect
from functools import wraps
import copy
from copy import deepcopy
from uuid import uuid4
from tqdm import tqdm
from typing import Callable, Union
from torch.utils.data.dataloader import DataLoader
import models
from torchvision import datasets, transforms
from joblib import Parallel, delayed
import training_script
import fire


def money(cuda=False):
    seeds = [i for i in range(3)]
    eigs = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    jacobs = [0, 1e-1, 1e-2, 1e-3, 1e-4]
    eps = [0, 0.1]
    activations = ['tanh']
    dims = [1, (1, 28), (4032, 128), (128, 10)]
    # In[]
    "Run code"
    for eig in eigs:
        for jacob in jacobs:
            for ep in eps:
                for nonlin in activations:
                    Parallel(n_jobs=2)(delayed(training_script.train_bad_boys)(alpha_eig=eig, alpha_jacob=jacob,
                                                                                eps=ep, cuda=cuda, nonlin=nonlin,
                                                                                arch='cnn', max_epochs=100,
                                                                                save_dir='../data/cnn_1/',
                                                                                seed=seed) for seed in seeds)


if __name__ == '__main__':
    fire.Fire(money)
