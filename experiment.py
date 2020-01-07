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
#
# ex.observers.append(MongoObserver(
#     url='mongodb://powerLawNN:Pareto_a^-b@ackermann.memming.com/admin?authMechanism=SCRAM-SHA-1',
#     db_name='powerLawExpts'))
##


@ex.named_config
def LeNet5():
    """
    To avoid passing lengthy architecture definitions we can use this as a convenience function

    The function only modifies the contents of the baseline configuration in experiment.cfg (below)
    """
    architecture = "cnn"
    dims = [1, (1, 28), (4032, 128), (128, 10)]


@ex.config
def cfg():
    activation = "tanh"
    alpha_jacob = 0
    alpha_spectra = 0
    architecture = "cnn"
    cuda = torch.cuda.is_available()
    dataset = "MNIST"
    dims = [1, (1, 28), (4032, 128), (128, 10)]  # TODO: fix this so it works
    eps = 0
    lr = 1e-3
    max_epochs = 1000
    max_iter = 100_000
    optimizer = "adam"
    regularizer = "no"
    save_dir = ""  # TODO: make sure this argument matches the model property
    trainer = "vanilla"
    training_type = "FGSM"
    validate = False
    weight_decay = 0


@ex.main
def do_training(activation, alpha_jacob, alpha_spectra, architecture, cuda, dataset, dims, eps, lr, max_epochs,
                max_iter, optimizer, regularizer, save_dir, trainer, training_type, validate, weight_decay,
                _seed, _run):

    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(_seed)
    # Create model
    model_kwargs = {"dims": dims, "activation": activation, "architecture": architecture,
                    "trainer": trainer, "regularizer": regularizer, 'alpha_spectra': alpha_spectra,
                    'alpha_jacob': alpha_jacob, 'optimizer': optimizer, 'lr': lr, 'weight_decay': weight_decay,
                    'cuda': cuda, 'eps': eps,  'training_type': training_type, "max_iter": max_iter}

    model = ModelFactory(**model_kwargs)
    # Store where model should be saved
    if save_dir is not None:
        if os.path.isdir(save_dir):
            model.save_name = os.path.join(save_dir, str(uuid4())[:8])
        else:
            model.save_name = save_dir
    # This ensures that any trained model will save the training loss for each mini-batch
    if _run is not None:
        model.logger = _run
    # Takes a model, dataset and bool indicating whether the model should be evaluated against the test set or a
    # validation subsample of the training set
    # tv_loader is shorthand for train_ or validation_ loader
    train_loader, tv_loader = GetDataForModel(model, dataset, validate)

    for epoch in tqdm(range(max_epochs), desc="Epochs", ascii=True, position=0, leave=False):
        """
        Model should be tested for test or validation accuracy & loss; the results should be logged in the logger
        """
        model.train_epoch(train_loader)
        mean_ll, mean_mce = model.evaluate_dataset_test_loss(tv_loader)
        if validate:
            _run.logger.log_scalar("validLoss", float(mean_ll))
            _run.logger.log_scalar("validAccuracy", float(1-mean_mce))
        else:
            _run.logger.log_scalar("testLoss", float(mean_ll))
            _run.logger.log_scalar("testAccuracy", float(1-mean_mce))

    return 1-mean_mce


# def parser():
#     import argparse
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--batch_size', default=200, type=int)
#     argparser.add_argument('--lr', default=1e-3, type=float)
#     argparser.add_argument('--dims', type=str, default=argparse.SUPPRESS)
#     argparser.add_argument('--numEpochs', default=250, type=int)
#     argparser.add_argument('--cuda', default=True, type=bool)
#     argparser.add_argument('--alpha', default=0.01, type=float)
#     argparser.add_argument('--eps', default=0.3, type=float)
#     argparser.add_argument('--gradSteps', default=40, type=int)
#     argparser.add_argument('--noRestarts', default=1, type=int)
#     argparser.add_argument('--pathLoad', default='../simple_regularization/data/mnist.npy', type=str)
#     argparser.add_argument('--pathSave', default='trained_models/data_analysis/mlp/', type=str)
#     argparser.add_argument('--epochSave', default=10, type=int)
#     argparser.add_argument('--activation', default='relu')
#     argparser.add_argument('--modelType',
#                            choices=['mlp', 'cnn', 'autoencoder'],
#                            default='mlp',
#                            help='Type of neural network.')
#     argparser.add_argument('--runName', default=rw.random_word() + str(random.randint(0, 100)),
#                            type=str)
#     argparser.add_argument('--computeEigVectorsOnline', default=False, type=bool)
#     argparser.add_argument('--regularizerFcn', default='default') # added it just to keep track of experiments
#     args = argparser.parse_args()
#     args = vars(args)
#
#     name = args['runName']
#     args.pop('runName', None)
#     args['pathSave'] = os.path.join(args['pathSave'], 'tmp' + str(uuid.uuid4())[:8])
#     return name, args
#
#
# if __name__ == '__main__':
#     name, args = parser()
#     train(name, args)
