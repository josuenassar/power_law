from tqdm import tqdm
import numpy as np
import sys
import fire
sys.path.append('..')
from ModelDefs.models import ModelFactory
from DataDefs.data import get_data
import torch
import os
import copy
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
Fine-tune a pre-trained a resnet where the spectral regularizer is put on the last hidden layer.
The width of the last hidden layer is [1000ish, 2000ish, 4000ish] so the batch sizes we will use is 
[1500, 3000, 6000]
"""


# In[]
weight_decay = .0001
dataset = 'CIFAR10Augmented'
seeds = [1000, 2000, 3000]

batch_sizes = []
batch_size = 1024
device = 'cpu'
cuda = False
if torch.cuda.is_available():
    cuda = True
    device = 'cuda'
num_grad_steps = 3_000
# In[]
kwargs = {"dims": [],
          "activation": 'relu',
          "architecture": 'vgg11',
          "trainer": "vanilla",
          "regularizer": "eig",
          'alpha_jacob': 1e-4,
          'bn': False,
          'alpha_spectra': 1,
          'lr': 1E-4,
          'optimizer': 'sgd',
          'weight_decay': weight_decay,
          'cuda': cuda,
          'eps': 0.3,
          'only_last': True,
          'gradSteps': 40,
          'noRestarts': 1,
          'lr_pgd': 1e-2,
          'training_type': 'FGSM',
          'slope': 1.00,
          'eig_start': 10,
          'cp': False,
          'dropout': True}

# pretrained_models = torch.load('vgg11', map_location=torch.device(device))
model_params = []
for j in range(len(seeds)):
    train_loader, test_loader, full_loader = get_data(dataset=dataset, batch_size=batch_size, _seed=seeds[j],
                                                      validate=False, data_dir='data/')
    X_full, _ = next(iter(full_loader))
    X_test, Y_test = next(iter(test_loader))

    torch.manual_seed(seeds[j] + 1)
    model = ModelFactory(**kwargs)
    # state_dict = pretrained_models[j][1]
    # model.load_state_dict(state_dict)

    grad_per_epoch = np.ceil(50_000 / batch_size)
    num_epochs = int(np.ceil(num_grad_steps / grad_per_epoch))
    print(num_epochs)
    "Train"
    for _ in tqdm(range(num_epochs)):
        model.train()
        model.train_epoch(train_loader, X_full, desp=True)

    "Print test accuracy"
    with torch.no_grad():
        model.eval()
        y_hat = model(X_test.to(device))
        _, predicted = torch.max(y_hat.to(device), 1)
        mce = (predicted != Y_test.to(device).data).float().mean().item()
        print((1 - mce) * 100)

    "Save model parameters"
    model_params.append((kwargs, copy.deepcopy(model.state_dict())))
    if cuda:
        torch.cuda.empty_cache()
    del model

torch.save(model_params, 'vgg_fine_tune')
# del pretrained_models



