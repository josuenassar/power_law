import sys
from tqdm import tqdm
import fire
from itertools import product
sys.path.append('..')
from ModelDefs.models import ModelFactory
from DataDefs.data import get_data
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_network(tau=10, activation='tanh', cuda=False, num_epochs=50, vanilla=False, lr=1e-3,
                  save_dir='experiment_1/'):
    """
    Runs a single layer MLP  with 2,000 hidden units on MNIST. The user can specify at what point in the eigenvalue
    spectrum should the regularizer start aka tau. The other parameters of the experiment are fixed: batch_size=2500,
    lr=0.001, number of realizations=3, regularization strengths=[0.1, 1, 5], slope values=[1.06, 1.04, 1.02, 1.00].
    :param tau: integer, at what point in the eigenvalue spectrum should we regularize
    :param activation: string, what activation function to use (only choices are tanh and relu)
    :param cuda: boolean, flag indicating whether to use GPU or not
    :param num_epochs: integer, number of epochs to train model for
    :param vanilla: boolean, flag indicating whether to use spectral regularizer or not
    :param lr: float, learning rate to use
    :param save_dir: string, location where to save models
    """
    realizations = 3
    batch_size = 3500  # 1.25 times the widest layer in the network
    train_loader, _, full_loader = get_data(dataset='MNIST', batch_size=batch_size, _seed=0,
                                            validate=False, data_dir='')
    if vanilla:
        kwargs = {"dims": [(28 * 28, 2000), (2000, 10)],
                  "activation": activation, "architecture": "mlp",
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
                   save_dir + 'vanilla' + '_activation=' + activation + '_epochs=' + str(num_epochs))

    else:
        slopes = [1.00]
        regularizers_strengths = [1., 2., 5.]
        stuff_to_loop_over = product(slopes, regularizers_strengths)
        # In[]
        "Load in data loader"
        X_full, _ = next(iter(full_loader))  # load in full training set for eigenvectors

        # In[]
        kwargs = {"dims": [(28 * 28, 2000), (2000, 10)],
                  "activation": activation, "architecture": "mlp",
                  "trainer": "vanilla",
                  "regularizer": "eig",
                  'alpha_jacob': 1e-4,
                  'bn': False,  # Note that for results shown in section 4.1, we set bn=True
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
        for (slope, reg_strength) in stuff_to_loop_over:
            kwargs['slope'] = slope
            kwargs['alpha_spectra'] = reg_strength
            models = [ModelFactory(**kwargs) for j in range(realizations)]
            for j in range(realizations):
                for epoch in tqdm(range(num_epochs)):
                    models[j].train_epoch(train_loader, X_full)

            model_params = []
            for idx in range(len(models)):
                model_params.append((kwargs, models[idx].state_dict()))

            torch.save(model_params,
                       save_dir + 'tau=' + str(tau) + '_activation=' + activation + '_epochs=' + str(num_epochs) + '_alpha=' + str(slope) + '_beta=' + str(reg_strength))
            counter += 1
            print(str(len(slopes) * len(regularizers_strengths) - counter) + " combos left")


if __name__ == '__main__':
    fire.Fire(train_network)
