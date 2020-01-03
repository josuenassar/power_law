import torch
import torch.nn as nn
from torch.optim import Adam, SGD, rmsprop
import numpy as np
from copy import deepcopy
from uuid import uuid4
from tqdm import tqdm
from typing import Callable
from torch.utils.data.dataloader import DataLoader
from torch import save
from utils import JacobianReg, counter, compute_eig_vectors, eigen_val_regulate
from BatchModfier import BatchModifier


class Trainer(nn.Module):

    def __init__(self, *, decoratee: BatchModifier, optimizer, lr, weight_decay, max_iter=100_000, save_name=None):
        super(Trainer, self).__init__()
        self._batch_modifier = decoratee
        self._save_name = save_name
        self.max_iter = max_iter
        self.no_minibatches = 0
        if optimizer.lower() == 'adam':
            self.optimizer = Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == 'sgd':
            self.optimizer = SGD(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == 'rms':
            self.optimizer = rmsprop(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            print('WOAH THERE BUDDY, THAT ISNT AN OPTION')

    def forward(self, x):
        return self._batch_modifier(x)

    def train_epoch(self, X: DataLoader):
        for _, (x, y) in enumerate(X):
            self.train_batch(x, y)

    @counter
    def train_batch(self, x, y):
        if self.no_minibatches > self.max_iter:
            return
        else:
            loss = self.evaluate_training_loss(x, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss

    @staticmethod
    def evaluate_dataset(X: DataLoader, *, function: Callable):
        """
        Function could for instance be evaluate_batch  *but* could be something more general
        :param X: a PyTorch data loader
        :param function: a callable function that will be evaluated on the entire dataset specified in the DataLoader X
        to wit: function requires two parameters the input x and target y
        :return: returns a generator with the function evals for each batch
        """
        for _, (x, y) in enumerate(tqdm(X)):
            function(x, y)

    def evaluate_dataset_test_loss(self, X: DataLoader):
        loss = 0
        mce = 0
        with torch.no_grad():
            for _, (x, y) in enumerate(tqdm(X)):
                loss_tmp, mce_tmp = self.evaluate_test_loss(x, y)
                loss += loss_tmp
                mce += mce_tmp
        return loss/len(X), mce/len(X)

    def train_epoch(self, X: DataLoader):
        self.evaluate_dataset(X, function=self.train_batch)

    def evaluate_training_loss(self, x, y):
        x, y = self.prepare_batch(x, y)
        h, y_hat = self.bothOutputs(x)
        return self.loss(y_hat, y)

    def evaluate_test_loss(self, x, y):
        y_hat = self.forward(x)
        return self.loss(y_hat, y), self.compute_mce(y_hat, y)

    @staticmethod
    def compute_mce(y_hat, y):
        """
        Computes the misclassification error
        :param y_hat: prediction
        :param y: ground truth labels
        :return: torch.float  fraction of misclassified examples
        """
        _, predicted = torch.max(y_hat, 1)
        return (predicted != y.data).float().mean()

# Utilities for serializing the model
    def serialize_model_type(self, filename=None):
        self._check_fname(filename)
        # TODO dump model definition into JSON so we can read it easily later on
        raise NotImplementedError

    @property
    def save_name(self):
        self.save_name = self._save_name

    @save_name.setter
    def save_name(self, name):
        self._save_name = name

    @save_name.getter
    def save_name(self):
        return self.save_name

    def _check_fname(self, filename=None):
        if filename is not None and self.save_name() is not None:
            raise AssertionError('Save name has already been defined')
        elif filename is not None and self.save_name is None:
            self.save_name(filename)
        elif filename is None and self.save_name is not None:
            return self.save_name
        else:
            filename = str(uuid4())[:8]
            self._check_fname(filename)

    def save(self, filename=None, other_vars=None):
        self._check_fname(filename)
        # TODO: add get cwd
        dummy_model = deepcopy(self.model)
        model_data = {'parameters': dummy_model.cpu().state_dict()}
        if other_vars is not None:
            model_data = {**model_data, **other_vars}
        save(model_data, self.save_name)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self._architecture, item)


class NoRegularization(Trainer):
    """
    No penalty class
    """
    def __init__(self, *,  decoratee: BatchModifier, optimizer, lr, weight_decay, max_iter=100_000, save_name=None):
        super(NoRegularization, self).__init__(decoratee=decoratee, optimizer=optimizer,
                                               lr=lr, weight_decay=weight_decay, max_iter=max_iter,
                                               save_name=save_name)


class JacobianRegularization(Trainer):

    def __init__(self, *, decoratee: BatchModifier, alpha_jacob, n=-1,
                 optimizer, lr, weight_decay, max_iter=100_000, save_name=None):
        super(JacobianRegularization, self).__init__(decoratee=decoratee, optimizer=optimizer,
                                               lr=lr, weight_decay=weight_decay, max_iter=max_iter,
                                               save_name=save_name)
        self.alpha_jacob = alpha_jacob
        self.JacobianReg = JacobianReg(n=n)

    def evaluate_training_loss(self, x, y):
        x, y = self.prepare_batch(x, y)
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        x.requires_grad = True  # this is essential!
        return loss + self.alpha_jacob * self.loss_regularizer(x, y_hat)

    def loss_regularizer(self, x, y_hat):
        return self.JacobianReg(x, y_hat)


class EigenvalueRegularization(Trainer):

    def __init__(self, *, decoratee: BatchModifier, save_name=None, max_iter=100_000, optimizer='adam', lr=1e-3,
                 weight_decay=1e-5, alpha_spectra):
        super(EigenvalueRegularization, self).__init__(decoratee=decoratee, optimizer=optimizer,
                                               lr=lr, weight_decay=weight_decay, max_iter=max_iter,
                                               save_name=save_name)
        self.train_spectra = []  # store the (estimated) spectra of the network at the end of each epoch
        self.train_loss = []  # store the training loss (reported at the end of each epoch on the last batch)
        self.train_regularizer = []  # store the value of the regularizer during training
        self.valLoss = []  # validation loss
        self.val_regularizer = []  # validation regularizer
        self.eig_vec = []
        self.alpha_spectra = alpha_spectra
        self.eig_start = 10

    "Overwrites method in trainer"
    def evaluate_training_loss(self, x, y):
        x, y = self.prepare_batch(x,y)
        hidden, y_hat = self.bothOutputs(x.to(self.device))  # feed data forward
        loss = self.loss(y_hat, y.to(self.device))  # compute loss

        "Compute spectra regularizer"
        return loss + self.alpha_spectra * self.spectra_regularizer(hidden)

    def compute_eig_vectors(self, x, y):
        with torch.no_grad():
            eigVec, loss, spectraTemp, regul = compute_eig_vectors(x, y, self._architecture, self.loss, self.device)
        self.eig_vec = eigVec
        return eigVec, loss, spectraTemp, regul

    def spectra_regularizer(self, hidden):
        "Compute spectra regularizer"
        spectra_regul = torch.zeros(1, device=self.device)
        spectra_temp = []
        for idx in range(len(hidden)):
            spectra, rTemp = eigen_val_regulate(hidden[idx], self.eig_vec, start=self.eig_start, device=self.device)
            with torch.no_grad():
                spectra_temp.append(spectra)
            spectra_regul += rTemp
        return spectra_regul


class EigenvalueAndJacobianRegularization(EigenvalueRegularization):
    def __init__(self, *, decoratee: BatchModifier, save_name=None, max_iter=100_000, optimizer='adam', lr=1e-3,
                 weight_decay=1e-5, alpha_spectra, alpha_jacob, n=-1):
        super(EigenvalueAndJacobianRegularization, self).__init__(decoratee=decoratee, optimizer=optimizer,
                                                       lr=lr, weight_decay=weight_decay, max_iter=max_iter,
                                                       save_name=save_name, alpha_spectra=alpha_spectra)
        self.train_spectra = []  # store the (estimated) spectra of the network at the end of each epoch
        self.train_loss = []  # store the training loss (reported at the end of each epoch on the last batch)
        self.train_regularizer = []  # store the value of the regularizer during training
        self.valLoss = []  # validation loss
        self.val_regularizer = []  # validation regularizer
        self.eig_vec = []
        self.alpha_spectra = alpha_spectra
        self.eig_start = 10
        self.alpha_jacob = alpha_jacob
        self.JacobianReg = JacobianReg(n=n)

    "Overwrites method in trainer"
    def evaluate_training_loss(self, x, y):
        x, y = self.prepare_batch(x, y)
        hidden, y_hat = self.bothOutputs(x.to(self.device))  # feed data forward
        loss = self.loss(y_hat, y.to(self.device))  # compute loss
        "Compute jacobian regularization"
        x.requires_grad = True  # this is essential!
        return loss + self.alpha_spectra * self.spectra_regularizer(hidden) + self.alpha_jacob * \
               self.loss_regularizer(x, y_hat)
