import abc
from torch.nn import modules
from torch.nn.functional import cross_entropy
from torch import no_grad
from torch.nn import Tanh as tanh
from torch.nn import ReLU as relu
from torch import save
import torch
from torch import nn
from uuid import uuid4
from torch.utils.data.dataloader import DataLoader
from copy import deepcopy
from functools import wraps
from tqdm import tqdm  #TODO fix this mess
from typing import Callable
#TODO use https://github.com/pytorch/ignite/tree/master/examples
import functools
import json
from models import CNN, MLP, ModelArchitecture
import inspect
from types_of_training import Trainer, Adversary, Regular


def filter_n_eval(func, **kwargs):
    """
    Takes kwargs and passes ONLY the named parameters that are specified in the callable func
    :param func: Callable for which we'll filter the kwargs and then pass them
    :param kwargs:
    :return:
    """
    args = inspect.signature(func)
    right_ones = kwargs.keys() & args.parameters.keys()
    newargs = { key: kwargs[key] for key in right_ones}
    return func(**newargs)


def counter(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # self is an instance of the class
        output = func(self, *args, **kwargs)
        self.no_minibatches+=1
        return output
    return wrapper


class Model(object):
    #TODO add loss field here
    def __init__(self):
        self.no_minibatches = 0
        self._save_name = None
        self.lr = 1
        self.max_iter = 100_000
        #TODO initialize network
        # self.loss = nn.CrossEntropyLoss
        # self.optimizer = torch.optim.Adam(lr=self.lr, amsgrad=True)

    @counter
    def train_batch(self, x, y):
        if self.no_minibatches > self.max_iter:
            return
        else:
            loss,_ = self.evaluate_training_loss(x, y)  #TODO
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return

    def train_epoch(self, X:DataLoader):
        self.evaluate_dataset(self, X, function=self.train_batch)


    def serialize_model_type(self, filename=None):
        self._check_fname(filename)
        #TODO dump model definition into JSON so we can read it easily later on
        raise NotImplementedError

    @property
    def save_name(self):
        self.save_name = self._save_name
    @save_name.setter
    def save_name(self,name):
        self._save_name = name
    @save_name.getter
    def save_name(self):
        return self.save_name

    def evaluate_dataset(self,  X:DataLoader, *, function:Callable):
        """
        Creates a generator that eavluates some function over the entire dataset
        Function could for instance be evaluate_batch  *but* could be something more general
        :param X: a PyTorch data loader
        :param function: a callable function that will be evaluated on the entire dataset specified in the DataLoader X
        to wit: function requires two parameters the input x and target y
        :return: returns a generator with the function evals for each batch
        """
        for _, x, y in enumerate(tqdm(X)):
            yield function(x, y)

    def evaluate_loss(self, x, y):
        y_hat = self.model(x)
        return self.loss(y_hat, y), self.compute_mce(y_hat,y)

    # def evaluate_loss_batch(self,x,y):


    def compute_mce(self, y_hat, y):
        """
        Computes the misclassification error
        :param y_hat: prediction
        :param y: ground truth labels
        :return: torch.float  fraction of misclassified examples
        """
        _, predicted = torch.max(y_hat, 1)
        return (predicted == y.data).float().mean()

    def test(self, X:DataLoader):
        return self.evaluate_dataset(X, function=self.evaluate_loss)

    def _check_fname(self, filename=None):
        if filename is not None and self.save_name() is not None:
            raise AssertionError('Save name has already been defined')
        elif filename is not None and self.save_name is None:
            self.save_name(filename)
        elif filename is None and self.save is not None:
            filename = self.save_name
        else:
            filename = str(uuid4())[:8]
            self._check_fname(filename)

    def save(self, filename=None, other_vars=None):
        self._check_fname(filename)
        #TODO add get cwd
        dummy_model = deepcopy(self.model)
        model_data = {'parameters': dummy_model.cpu().state_dict()}
        if other_vars is not None:
            model_data = {**model_data, **other_vars}
        save(model_data, self.save_name)


class DecoratedModel(object):
    """
    Joins a Model type object with subclasses of ModelArchitecture and Trainer.
    Jointly specify all that's needed for training
    Model contains general primitives for training
    ModelArch contains the architecture specific implementation
    Trainer  contains the training specific implementation
    """

    def __init__(self, *args) -> None:
        super().__init__()

    def __new__(cls, model:Model, arch:ModelArchitecture, training:Trainer):
        mydict = {**model.__dict__.copy(), **arch.__dict__.copy(), **training.__dict__.copy()}
        cls = type('DecoratedModel',
                   (DecoratedModel, model.__class__, arch.__class__, training.__class__),
                   mydict)
        return object.__new__(cls)

    def register_parameters(self):
        self.optimizer(self.architecture.parameters(), self.lr)


def TrainingFactory(*,training_type, **kwargs):
    """
    Factory function that creates objects needed to specify the type of training
    :param training_type: str in ['adversarial', 'penalized', 'regular']
    :param kwargs: dict containing other kwards needed for creating the objects
    :return: Training object
    """
    if training_type == 'adversarial':
        return filter_n_eval(Adversary, **kwargs)
    elif training_type == 'penalized':
        pass
    elif training_type == 'regular':
        return filter_n_eval(Regular, **kwargs)
    else:
        raise NotImplementedError

def NetworkFactory(*,network_type,**kwargs):
    """
    Factory function that creates objects needed to specify the architecture
    :param network_type:  str in ['adversarial', 'penalized', 'regular']
    :param kwargs: dict containing other kwards needed for creating the objects
    :return: Network object
    """
    if network_type == 'cnn':
        return filter_n_eval(CNN, **kwargs)
    elif network_type == 'mlp':
        # return filter_n_eval(MLP, **kwargs)
        return filter_n_eval(MLP, **{"dims": [(28 * 28, 1000), (1000, 10)],
                                     "activation": "relu"})
    else:
        raise NotImplementedError

if __name__ == '__main__':
    model = Model()
    network = NetworkFactory(network_type='mlp')
    training_type = TrainingFactory(training_type='regular')

    decorated_model = DecoratedModel(model, network, training_type)
    print('test')
    # decorated_model.register_parameters()
