import sys
from tqdm import tqdm
import fire
from itertools import product
sys.path.append('..')
from ModelDefs.models import ModelFactory
from DataDefs.data import get_data
import torch
import numpy as np
import copy


def bad_boy_vibes(tau=0, activation='tanh', cuda=False, num_epochs=50):
    """
    Runs a single layer MLP  with 2,000 hidden units on MNIST. The user can specify at what point in the eigenvalue
    spectrum should the regularizer start aka tau. The other parameters of the experiment are fixed: batch_size=2500,
    lr=0.001, number of realizations=3, regularization strengths=[0.1, 1, 5], slope values=[1.06, 1.04, 1.02, 1.00].
    :param tau: integer, at what point in the eigenvalue spectrum should we regularize
    """
    slopes = [1.06, 1.04, 1.02, 1.00]
    regularizers_strengths = [0.1, 1., 5.]
    realizations = 3
    batch_size = 2500  # 1.25 times the widest layer in the network
    lr = 1e-3
    stuff_to_loop_over = product(slopes, regularizers_strengths)
    # In[]
    "Load in data loader"
    train_loader, test_loader, full_loader = get_data(dataset='MNIST', batch_size=batch_size, _seed=0,
                                                      validate=False, data_dir='')
    X_full, _ = next(iter(full_loader))  # load in full training set for eigenvectors

    # In[]
    kwargs = {"dims": [(28 * 28, 2000), (2000, 10)],
              "activation": activation, "architecture": "mlp",
              "trainer": "vanilla",
              "regularizer": "eig",
              'alpha_jacob': 1e-4,
              'bn': False,
              'alpha_spectra': 1,
              'optimizer': 'adam',
              'lr': lr,
              'weight_decay': 0,
              'cuda': cuda,
              'eps': 0.3,
              'only_last': True,
              'gradSteps': 40,
              'noRestarts': 1,
              'lr_pgd': 1e-2,
              'training_type': 'FGSM',
              'slope': [1.00, 1.00, 1.0],
              'eig_start': tau}

    counter = 0
    for (slope, reg_strength) in stuff_to_loop_over:
        kwargs['slope'] = slope
        kwargs['alpha_spectra'] = reg_strength
        models = []  # container used to store all the trained models

        for j in range(realizations):
            model = ModelFactory(**kwargs)  # create model
            for epoch in tqdm(range(num_epochs)):
                model.train_epoch(train_loader, X_full)
            models.append(model)

        model_params = []
        for idx in range(len(models)):
            model_params.append((kwargs, models[idx].state_dict()))

        torch.save(model_params, 'experiment_1/tau=' + str(tau) + '_activation=' + activation + '_epochs=' + str(num_epochs))
        counter += 1
        print(str(12 - counter) + " combos left")


if __name__ == '__main__':
    fire.Fire(bad_boy_vibes)
