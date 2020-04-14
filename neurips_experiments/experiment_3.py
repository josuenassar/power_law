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
import os
from joblib import Parallel, delayed
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# def train(model, batch_size, num_epochs, X_full, dataset='MNIST'):
#     train_loader, _, _ = get_data(dataset='MNIST', batch_size=batch_size, _seed=np.random.randint(100),
#                                                       validate=False, data_dir='')
#     for epoch in tqdm(range(num_epochs)):
#         model.train_epoch(train_loader, X_full)
#     return model


def bad_boy(tau=10, activation='tanh', cuda=False, num_epochs=1, vanilla=False, dataset='MNIST', arch='cnn'):
    realizations = 3
    lr = 1e-3
    if arch == 'mlp':
        dims = [(28 * 28, 1_000), (1_000, 1_000), (1_000, 1_000), (1_000, 10)]
        batch_size = 1500
    elif arch == 'cnn':
        if dataset == 'MNIST':
            dims = [2, (1, 32), (32, 64), (1024, 1024), (1024, 10)]
            batch_size = 6912
        else:
            dims = [2, (3, 32), (32, 64), (1600, 1600), (1600, 10)]
            batch_size = 9408
    else:
        print("Doesnt exist!!")
    train_loader, _, full_loader = get_data(dataset=dataset, batch_size=batch_size, _seed=0,
                                            validate=False, data_dir='data/')

    if vanilla:
        kwargs = {"dims": dims,
                  "activation": activation,
                  "architecture": arch,
                  "trainer": "vanilla",
                  "regularizer": "no",
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
                  'slope': [1.00],
                  'eig_start': tau}
        models = [ModelFactory(**kwargs) for j in range(realizations)]
        for j in range(realizations):
            for epoch in tqdm(range(num_epochs)):
                models[j].train_epoch(train_loader)

        model_params = []
        for idx in range(len(models)):
            model_params.append((kwargs, models[idx].state_dict()))

        torch.save(model_params,
                   'experiment_3/' + dataset + '/vanilla_arch=' + arch + '_activation=' + activation + '_epochs=' + str(num_epochs))
    else:

        regularizers_strengths = [0.1, 1., 5.]
        # In[]
        "Load in data loader"
        X_full, _ = next(iter(full_loader))  # load in full training set for eigenvectors

        # In[]
        kwargs = {"dims": dims,
                  "activation": activation,
                  "architecture": arch,
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
                  'slope': [1.00],
                  'eig_start': tau}

        counter = 0
        for reg_strength in regularizers_strengths:
            kwargs['alpha_spectra'] = reg_strength
            models = [ModelFactory(**kwargs) for j in range(realizations)]
            # if not parallel:
            print('no vibes')
            for j in range(realizations):
                for epoch in tqdm(range(num_epochs)):
                    models[j].train_epoch(train_loader, X_full)
            # else:
            #     print('vibes')
            #     del full_loader
            #     models = Parallel(n_jobs=realizations)(delayed(train)(models[j],
            #                                                           batch_size,
            #                                                           num_epochs,
            #                                                           X_full) for j in range(realizations))

            model_params = []
            for idx in range(len(models)):
                model_params.append((kwargs, models[idx].state_dict()))

            torch.save(model_params, 'experiment_3/' + dataset + '/tau=' + str(tau) + '_arch=' + arch + '_activation=' + activation + '_epochs=' + str(
                num_epochs) + '_alpha=' + str(1) + '_beta=' + str(reg_strength))
            counter += 1
            print(str(len(regularizers_strengths) - counter) + " combos left")


if __name__ == '__main__':
    fire.Fire(bad_boy)
