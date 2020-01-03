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
import os
from torch.utils.data.dataloader import DataLoader
import models
from torchvision import datasets, transforms

# In[]
"Create model"
# kwargs = {"dims": [(28 * 28, 1000), (1000, 10)], "activation": "relu", "architecture": "mlp",
#           "trainer": "vanilla", "regularizer": "eig", 'alpha_spectra': 1e-3, 'optimizer': 'adam',
#           'alpha_jacob': 1e-3, 'lr': 1e-3, 'weight_decay': 1e-5, 'cuda': False}
kwargs = {"dims": [(28 * 28, 1000), (1000, 10)], "activation": "relu", "architecture": "mlp",
          "trainer": "adv", "regularizer": "no", 'alpha_spectra': 1e-3, 'alpha_jacob':1e-4,
          'optimizer': 'adam', 'lr': 1e-3, 'weight_decay': 1e-5, 'cuda': False, 'eps': 0.1,
          'gradSteps': 1, 'noRestarts': 0, 'alpha': 0, 'training_type': 'FGSM'}
model = models.ModelFactory(**kwargs)


# In[]
"Load in MNIST and create data loader"
kwargs = {'num_workers': 4, 'pin_memory': True}
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST(root=os.getcwd(), train=True, download=True, transform=transform)
test_set = datasets.MNIST(root=os.getcwd(), train=False, download=True, transform=transform)
num_train = len(train_set)
indices = list(range(num_train))
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
batch_size = 1_200
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                           **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
full_loader = torch.utils.data.DataLoader(train_set, batch_size=60_000, sampler=train_sampler,
                                           **kwargs)

# In[]
# full_X, full_Y = next(iter(full_loader))

# In[]
"Train that bad boy"
model.train_epoch(train_loader)