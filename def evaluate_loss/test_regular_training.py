import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import inspect
from functools import wraps
import copy
from copy import deepcopy
from uuid import uuid4
from tqdm import tqdm
from typing import Callable, Union

import unittest
import os

from torch.utils.data.dataloader import DataLoader
from torch import save
import model_defs

"Script that will check to see if regular training is working"

# In[]
"Create a model"
kwargs = {"dims": [(28 * 28, 1000), (1000, 10)], "activation": "relu", "architecture": "mlp",
                  "trainer": "vanilla", "regularizer": "no"}
model = model_defs.ModelFactory(**kwargs)


# In[]
"Load in MNIST and create data loader"
kwargs = {'num_workers': 4, 'pin_memory': True}
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST(root=os.getcwd(),train=True, download=True, transform=transform)
test_set = datasets.MNIST(root=os.getcwd(),train=False, download=True, transform=transform)
num_train = len(train_set)
indices = list(range(num_train))
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                           **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
