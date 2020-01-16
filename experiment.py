import torch
from ModelDefs.models import ModelFactory  # Since we don't need the internals
from DataDefs.data import GetDataForModel

# Sacred
from sacred import Experiment
# Imports for Serializing and Saving
from random_words import RandomWords
import random
from uuid import uuid4
import os
# MISC
from tqdm import tqdm

rw = RandomWords()
ex = Experiment(name=rw.random_word() + str(random.randint(0, 100)))


@ex.named_config
def LeNet5():
    """
    To avoid passing lengthy architecture definitions we can use this as a convenience function

    The function only modifies the contents of the baseline configuration in experiment.cfg (below)
    """
    architecture = "cnn"
    dims = [1, (1, 28), (4032, 128), (128, 10)]


@ex.named_config
def MadryMNIST():
    """
    To avoid passing lengthy architecture definitions we can use this as a convenience function

    The function only modifies the contents of the baseline configuration in experiment.cfg (below)
    """
    architecture = "cnn"
    dims = [2, (1, 32), (32, 64), (1024, 1024), (1024, 10)]


@ex.named_config
def SaveDiracErdos():
    save_dir = os.getcwd()
    data_dir = '../..'


@ex.named_config
def SaveAdobe():
    save_dir = os.getcwd()
    data_dir = '../../'

@ex.config
def cfg():
    activation = "tanh"
    alpha_jacob = 0
    alpha_spectra = 0
    architecture = "cnn"
    cuda = torch.cuda.is_available()
    data_dir = '../../'
    dataset = "MNIST"
    dims = [2, (1, 32), (32, 64), (1024, 1024), (1024, 10)]  # same arch as MadryMNIST
    eps = 0.3  # 0.3 for MNIST AND 8/255 for CIFAR10
    gradSteps = 40  # 40 for MNIST
    lr = 1e-4
    lr_pgd = 1e-2  # TODO: add PGD learning rate
    max_epochs = 500
    max_iter = 100_000
    noRestarts = 1  # compare to results in https://arxiv.org/pdf/1706.06083.pdf page 13 (should give  ~ 93% accuracy)
    optimizer = "adam"
    regularizer = "eigjac"
    save_dir = os.getcwd()
    trainer = "vanilla"
    training_type = "PGD"
    hpsearch = True
    weight_decay = 0


@ex.automain
def do_training(activation, alpha_jacob, alpha_spectra, architecture, cuda, data_dir,
                dataset, dims, eps, gradSteps, lr, lr_pgd, max_epochs, max_iter, noRestarts,
                optimizer, regularizer, save_dir, trainer,
                training_type, hpsearch, weight_decay, _seed, _run):

    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(_seed)
    # Create model
    model_kwargs = {"dims": dims, "activation": activation, "architecture": architecture,
                    "trainer": trainer, "regularizer": regularizer, 'alpha_spectra': alpha_spectra,
                    'alpha_jacob': alpha_jacob, 'optimizer': optimizer, 'lr': lr, 'lr_pgd': lr_pgd,
                    'weight_decay': weight_decay, 'noRestarts':noRestarts,
                    'cuda': cuda, 'eps': eps, 'gradSteps': gradSteps,  'training_type': training_type, "max_iter": max_iter}

    model = ModelFactory(**model_kwargs)
    # Store where model should be saved
    if save_dir is not None:
        if os.path.isdir(save_dir):
            model.save_name = os.path.join(save_dir, str(uuid4())[:8])
        else:
            model.save_name = save_dir
    print("Save name is {} \n".format(model.save_name))
    # This ensures that any trained model will save the training loss for each mini-batch
    if _run is not None:
        model.logger = _run
    # Takes a model, dataset and bool indicating whether the model should be evaluated against the test set or a
    # validation subsample of the training set
    # tv_loader is shorthand for train_ or validation_ loader'
    train_loader, tv_loader = GetDataForModel(model, dataset=dataset, _seed=_seed, hpsearch=hpsearch, data_dir=data_dir)
    for epoch in tqdm(range(max_epochs), desc="Epochs", ascii=True, position=0, leave=True):
        """
        Model should be tested for test or validation accuracy & loss; the results should be logged in the logger
        """
        mean_ll, mean_mce = model.evaluate_dataset_test_loss(tv_loader)
        if hpsearch:
            _run.log_scalar("validLoss", float(mean_ll))
            _run.log_scalar("validAccuracy", float(1-mean_mce))
        else:
            _run.log_scalar("testLoss", float(mean_ll))
            _run.log_scalar("testAccuracy", float(1-mean_mce))
        model.train_epoch(train_loader)
        torch.cuda.empty_cache()
        model.save()  # TODO: Error in save function
    save_name_for_db = "model_data.pt"
    _run.add_artifact(model.save_name, save_name_for_db, content_type="application/octet-stream")
    if hpsearch:
        os.remove(model.save_name)

    return float(1-mean_mce)  # holdout accuracy
