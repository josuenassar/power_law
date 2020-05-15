import sys
from tqdm import tqdm
import fire
sys.path.append('..')
from ModelDefs.models import ModelFactory
from DataDefs.data import get_data
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def bad_boy_vibes(tau=0, activation='tanh', cuda=False, num_epochs=50):
    """
    Runs a single layer MLP  with 2,000 hidden units on MNIST. The user can specify at what point in the eigenvalue
    spectrum should the regularizer start aka tau. The other parameters of the experiment are fixed: batch_size=2500,
    lr=0.001, number of realizations=3, regularization strengths=[0.1, 1, 5], slope values=[1.06, 1.04, 1.02, 1.00].
    :param tau: integer, at what point in the eigenvalue spectrum should we regularize
    """
    realizations = 3
    batch_size = 3500  # 1.25 times the widest layer in the network
    lr = 1e-3
    train_loader, _, full_loader = get_data(dataset='MNIST', batch_size=batch_size, _seed=0,
                                            validate=False, data_dir='')
    weights = [0, 1e-4, 1e-3, 1e-2, 1e-1]
    for w in weights:
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
                  'eig_start': tau,
                  'weight_decay': w}
        models = [ModelFactory(**kwargs) for j in range(realizations)]
        for j in range(realizations):
            for epoch in tqdm(range(num_epochs)):
                models[j].train_epoch(train_loader)

        model_params = []
        for idx in range(len(models)):
            model_params.append((kwargs, models[idx].state_dict()))

        if w == 0:
            torch.save(model_params,
                       'experiment_1/vanilla' + '_activation=' + activation + '_epochs=' + str(num_epochs))
        else:
            torch.save(model_params,
                       'experiment_1/vanilla' + '_activation=' + activation + '_epochs=' + str(num_epochs) + '_w=' + str(w))


if __name__ == '__main__':
    fire.Fire(bad_boy_vibes)
