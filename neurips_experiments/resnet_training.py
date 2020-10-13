from tqdm import tqdm
import sys
sys.path.append('..')
from ModelDefs.models import ModelFactory
from DataDefs.data import get_data
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
Train a ResNet-20 on gray-scale CIFAR-10 following
the extract training details from resnet paper.
"""

# In[]
batch_size = 128
weight_decay = .0001
num_epochs = 200
dataset = 'CIFAR10'
no_seeds = 3
seeds = [100, 200, 300]
cuda = False
device = 'cpu'
if torch.cuda.is_available():
    cuda = True
    device = 'cuda'
kwargs = {"dims": [],
          "activation": 'relu',
          "architecture": 'resnet',
          "trainer": "vanilla",
          "regularizer": "no",
          'alpha_jacob': 1e-4,
          'bn': False,
          'alpha_spectra': 1,
          'lr': 0.1,
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
          'eig_start': 10}


# In[]
models = []
for j in range(len(seeds)):
    train_loader, test_loader, _ = get_data(dataset=dataset, batch_size=batch_size, _seed=seeds[j],
                                            validate=False, data_dir='data/')
    X_test, Y_test = next(iter(test_loader))
    torch.manual_seed(seeds[j] + 1)
    models.append(ModelFactory(**kwargs))
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(models[j].optimizer, milestones=[100, 150])
    counter = 0
    for epoch in tqdm(range(num_epochs)):
        models[j].train()
        models[j].train_epoch(train_loader)
        # lr_scheduler.step()
        with torch.no_grad():
            if epoch % 10 == 0:
                models[j].eval()
                y_hat = models[j](X_test.to(device))
                _, predicted = torch.max(y_hat.to(device), 1)
                mce = (predicted != Y_test.to(device).data).float().mean().item()
                print((1 - mce) * 100)

model_params = []
for idx in range(len(models)):
    model_params.append((kwargs, models[idx].state_dict()))

torch.save(model_params, 'resnet_vanilla_grayscale')
