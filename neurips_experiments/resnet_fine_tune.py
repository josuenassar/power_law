from tqdm import tqdm
import sys

sys.path.append('..')
from ModelDefs.models import ModelFactory
from DataDefs.data import get_data
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
Fine-tune a pre-trained a resnet where the spectral regularizer is put on the last hidden layer.
The width of the last hidden layer is [1000ish, 2000ish, 4000ish] so the batch sizes we will use is 
[1500, 3000, 6000]
"""

# In[]
weight_decay = .0001
num_epochs = 200
dataset = 'CIFAR10Augmented'
no_seeds = 3
seeds = [1000, 2000, 3000]
cuda = False
filters = [4, 8, 16]
batch_sizes = [1_500, 3_000, 6_000]
device = 'cpu'
if torch.cuda.is_available():
    cuda = True
    device = 'cuda'
# In[]
for n_filters, batch_size in zip(filters, batch_sizes):
    kwargs = {"dims": [],
              "activation": 'relu',
              "architecture": 'resnet',
              "trainer": "vanilla",
              "regularizer": "eig",
              'alpha_jacob': 1e-4,
              'bn': False,
              'alpha_spectra': 1.0,
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
              'slope': [1.00],
              'eig_start': 10,
              'n_filters': n_filters}

    pretrained_models = torch.load('resnet_' + str(n_filters))
    models = []
    for j in range(len(seeds)):
        train_loader, test_loader, full_loader = get_data(dataset=dataset, batch_size=batch_size, _seed=seeds[j],
                                                          validate=False, data_dir='data/')
        X_full, _ = next(iter(full_loader))
        X_test, Y_test = next(iter(test_loader))


        torch.manual_seed(seeds[j] + 1)
        model = ModelFactory(**kwargs)
        model.load_state_dict(pretrained_models[j][1])
        models.append(model)
        counter = 0
        for epoch in tqdm(range(num_epochs)):
            models[j].train()
            models[j].train_epoch(train_loader, X_full)

        with torch.no_grad():
            models[j].eval()
            y_hat = models[j](X_test.to(device))
            _, predicted = torch.max(y_hat.to(device), 1)
            mce = (predicted != Y_test.to(device).data).float().mean().item()
            print((1 - mce) * 100)
        if cuda:
            torch.cuda.empty_cache()
    model_params = []
    for idx in range(len(models)):
        model_params.append((kwargs, models[idx].state_dict()))

    torch.save(model_params, 'resnet_fine_tune_' + str(n_filters))
    del pretrained_models

