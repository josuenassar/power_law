import sys
from tqdm import tqdm
import fire
sys.path.append('..')
from ModelDefs.models import ModelFactory
from DataDefs.data import get_data
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_network(tau=10, activation='tanh', cuda=False, num_epochs=100, vanilla=False, dataset='MNIST', arch='mlp',
                  realizations=3, flat=False, save_dir='experiment_2/'):
    """
    Training script for running experiments for section 4.2 in paper.
    :param tau: integer, at what point in the eigenvalue spectrum should we regularize
    :param activation: string, what activation function to use (only choices are tanh and relu)
    :param cuda: boolean, flag indicating whether to use GPU or not
    :param num_epochs: integer, number of epochs to train model for
    :param vanilla: boolean, flag indicating whether to use spectral regularizer or not
    :param dataset: string, which dataset to use (choices are MNIST or CIFAR10)
    :param arch: string, whether to use an MLP ('mlp') or a CNN ('cnn')
    :param realizations: integer, number of random seeds to use
    :param save_dir: string, location where to save models
    """
    lr = 1e-4
    if arch == 'mlp':
        if flat:
            arch = 'mlp_flat'
        dims = [(28 * 28, 1_000), (1_000, 1_000), (1_000, 1_000), (1_000, 10)]
        batch_size = 1500
        dataset = 'MNIST'
    elif arch == 'cnn':
        if flat:
            arch = 'cnn_flat'
        lr = 1e-4
        if dataset == 'MNIST':
            dims = [2, (1, 16), (16, 32), (800, 1000), (1000, 10)]
            batch_size = 6000
        else:
            dims = [2, (3, 16), (16, 32), (1152, 1000), (1000, 10)]
            batch_size = 6000
    else:
        print("Doesnt exist!!")
    train_loader, _, full_loader = get_data(dataset=dataset, batch_size=batch_size, _seed=0,
                                            validate=False, data_dir='data/')

    if vanilla:
        train_loader, _, full_loader = get_data(dataset=dataset, batch_size=batch_size, _seed=0,
                                                validate=False, data_dir='data/')
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
                   save_dir + dataset + '/vanilla_arch=' + arch + '_activation=' + activation + '_epochs=' + str(num_epochs))
    else:
        regularizers_strengths = [5., 2., 1.]
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
                  'alpha_spectra': 1.0,
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
                  'slope': 1.00,
                  'eig_start': tau}

        counter = 0
        for reg_strength in regularizers_strengths:
            kwargs['alpha_spectra'] = reg_strength
            models = [ModelFactory(**kwargs) for j in range(realizations)]
            print('no vibes')
            for j in range(realizations):
                for epoch in tqdm(range(num_epochs)):
                    models[j].train_epoch(train_loader, X_full)


            model_params = []
            for idx in range(len(models)):
                model_params.append((kwargs, models[idx].state_dict()))

            torch.save(model_params,
                       save_dir + dataset + '/tau=' + str(tau) + '_arch=' + arch + '_activation=' + activation + '_epochs=' + str(
                num_epochs) + '_alpha=' + str(1) + '_beta=' + str(reg_strength))
            counter += 1
            print(str(len(regularizers_strengths) - counter) + " combos left")


if __name__ == '__main__':
    fire.Fire(train_network)
