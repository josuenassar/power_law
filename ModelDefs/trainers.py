import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop
from copy import deepcopy
from uuid import uuid4
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch import save
from utils import JacobianReg, counter, eigen_val_regulate, compute_eig_vectors_only
from BatchModfier import BatchModifier


class Trainer(nn.Module):

    def __init__(self, *, decoratee: BatchModifier, optimizer, lr, weight_decay, max_iter=100_000, save_name=None):
        super(Trainer, self).__init__()
        self._batch_modifier = decoratee
        self._save_name = save_name
        self.max_iter = max_iter
        self.no_minibatches = 0
        if optimizer.lower() == 'adam':
            self.optimizer = Adam(params=self.parameters(),
                                  lr=lr, weight_decay=weight_decay,amsgrad=True)
        elif optimizer.lower() == 'sgd':
            self.optimizer = SGD(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer.lower() == 'rms':
            self.optimizer = RMSprop(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            print('WOAH THERE BUDDY, THAT ISNT AN OPTION')
        self.logger = None

    def forward(self, x):
        return self._batch_modifier(x)

    def train_epoch(self, X: DataLoader):
        for _, (x, y) in enumerate(tqdm(X)):
            self.train_batch(x.to(self.device), y.to(self.device))
            x.to('cpu')
            y.to('cpu')

    @counter
    def train_batch(self, x, y):
        if self.no_minibatches > self.max_iter:
            return
        else:
            loss = self.evaluate_training_loss(x, y)
            if self.logger is not None:
                self.logger.log_scalar("trainLoss", float(loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss

    def evaluate_dataset_test_loss(self, X: DataLoader):
        loss = 0
        mce = 0
        with torch.no_grad():
            for _, (x, y) in enumerate(tqdm(X)):
                loss_tmp, mce_tmp = self.evaluate_test_loss(x.to(self.device), y.to(self.device))
                x.to('cpu')
                y.to('cpu')
                loss += loss_tmp
                mce += mce_tmp
        return loss/len(X), mce/len(X)

    def evaluate_training_loss(self, x, y):
        x, y = self.prepare_batch(x, y)
        h, y_hat = self.bothOutputs(x)
        # return self.loss(y_hat, y)
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
            return getattr(self._batch_modifier, item)


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
        x.requires_grad = True  # this is essential!
        y_hat = self(x)
        loss = self.loss(y_hat, y)
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
        self.eig_loader = None

    def add_eig_loader(self, X):
        # x,y = next(iter(tl))
        # import pdb; pdb.set_trace()
        if X.dataset.train:
            self.eig_loader = torch.utils.data.DataLoader(X.dataset, batch_size=len(X.dataset.data), shuffle=False,
                                          num_workers=X.num_workers,
                                          pin_memory=X.pin_memory)

    "Overwrites method in trainer"
    def evaluate_training_loss(self, x, y):
        x, y = self.prepare_batch(x, y)
        hidden, y_hat = self.bothOutputs(x.to(self.device))  # feed data forward
        loss = self.loss(y_hat, y.to(self.device))  # compute loss

        "Compute spectra regularizer"
        return loss + self.alpha_spectra * self.spectra_regularizer(hidden)

    def compute_eig_vectors(self, x, y):
        with torch.no_grad():
            hidden, _ = self.bothOutputs(x)
            eigVec = compute_eig_vectors_only(hidden)
        self.eig_vec = eigVec
        return eigVec

    def spectra_regularizer(self, hidden):
        "Compute spectra regularizer"
        spectra_regul = torch.zeros(1, device=self.device)
        spectra_temp = []
        for idx in range(len(hidden)):
            spectra, rTemp = eigen_val_regulate(hidden[idx], self.eig_vec[idx],
                                                start=self.eig_start, device=self.device)
            with torch.no_grad():
                spectra_temp.append(spectra)
            spectra_regul += rTemp
        return spectra_regul

    def train_epoch(self, X: DataLoader, X_full=None, Y_full=None):

        if X_full is None and Y_full is None and self.eig_loader is None:
            self.add_eig_loader(X)
        elif X_full is None and Y_full is None and self.eig_loader is not None:
            X_full, Y_full = next(iter(self.eig_loader))

        self.compute_eig_vectors(X_full, Y_full)
        X_full.to('cpu')
        Y_full.to('cpu')
        for _, (x, y) in enumerate(tqdm(X)):
            self.train_batch(x.to(self.device), y.to(self.device))
            x.to('cpu')
            y.to('cpu')
        # raise NotImplementedError


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
        x.requires_grad = True  # this is essential!
        hidden, y_hat = self.bothOutputs(x.to(self.device))  # feed data forward
        loss = self.loss(y_hat, y.to(self.device))  # compute loss
        "Compute jacobian regularization"
        return loss + self.alpha_spectra * self.spectra_regularizer(hidden) + self.alpha_jacob * \
               self.loss_regularizer(x, y_hat)

    def loss_regularizer(self, x, y_hat):
        return self.JacobianReg(x, y_hat)
