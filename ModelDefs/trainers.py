import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop
from copy import deepcopy
from uuid import uuid4
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from ModelDefs.utils import JacobianReg, counter, eigen_val_regulate, compute_eig_vectors_only
from ModelDefs.BatchModfier import BatchModifier
import numpy as np


class Trainer(nn.Module):

    def __init__(self, *, decoratee: BatchModifier, optimizer, lr, weight_decay, max_iter=100_000, save_name=None):
        super(Trainer, self).__init__()
        self._batch_modifier = decoratee
        self.save_name = save_name
        self.max_iter = max_iter
        self.no_minibatches = 0
        if optimizer.lower() == 'adam':
            self.optimizer = Adam(params=self.parameters(),
                                  lr=lr, weight_decay=weight_decay, amsgrad=True)
        elif optimizer.lower() == 'sgd':
            self.optimizer = SGD(params=self.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optimizer.lower() == 'rms':
            self.optimizer = RMSprop(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            print('WOAH THERE BUDDY, THAT ISNT AN OPTION')
        self.logger = None

    def forward(self, x):
        return self._batch_modifier(x)

    def train_epoch(self, X: DataLoader):
        # for _, (x, y) in enumerate(tqdm(X, desc="Training Batches", ascii=True, position=1, leave=True)):
        for _, (x, y) in enumerate(X):
            self.train_batch(x.to(self.device), y.to(self.device))
            x.to('cpu', non_blocking=True)
            y.to('cpu', non_blocking=True)

    @counter
    def train_batch(self, x, y):
        if self.no_minibatches > self.max_iter:
            return
        else:
            x, y = self.prepare_batch(x, y)
            self.optimizer.zero_grad()  # zero out gradient to ensure we don't backprop through adv image generation
            loss = self.evaluate_training_loss(x, y)
            # print(torch.cuda.max_memory_cached())
            if self.logger is not None:
                self.logger.log_scalar("trainLoss", float(loss))
            loss.backward()
            self.optimizer.step()
            return loss

    def evaluate_dataset_test_loss(self, X: DataLoader):
        loss = 0
        mce = 0
        with torch.no_grad():
            for _, (x, y) in enumerate(tqdm(X, desc="Testing Batches", ascii=True, position=2, leave=False)):
                loss_tmp, mce_tmp = self.evaluate_test_loss(x.to(self.device), y.to(self.device))
                x.to('cpu')
                y.to('cpu')
                loss += loss_tmp
                mce += mce_tmp
        return loss/len(X), mce/len(X)

    def evaluate_training_loss(self, x, y):
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

    def save(self, filename=None, other_vars=dict()):
        if self.save_name is None and filename is None:
            self.save_name = str(uuid4())[:8]
        elif filename is not None:
            self.save_name = filename
        # TODO: add get cwd
        model_data = {'parameters': self._architecture.cpu().state_dict()}
        if isinstance(self, EigenvalueRegularization):
            if self.eig_vec[0] is not None:
                other_vars = {**other_vars, "eig_vec": [v.cpu() for v in self.eig_vec] }
        if other_vars is not None:
            model_data = {**model_data, **other_vars}
        self._architecture.to(self.device,  non_blocking=True)
        if isinstance(self, EigenvalueRegularization):
            if self.eig_vec[0] is not None:
                [v.to(self.device,  non_blocking=True) for v in self.eig_vec]
        torch.save(model_data, self.save_name)

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

    def __init__(self, *, decoratee: BatchModifier, alpha_jacob, n=1,
                 optimizer, lr, weight_decay, max_iter=100_000, save_name=None):
        super(JacobianRegularization, self).__init__(decoratee=decoratee, optimizer=optimizer,
                                               lr=lr, weight_decay=weight_decay, max_iter=max_iter,
                                               save_name=save_name)
        self.alpha_jacob = alpha_jacob
        self.JacobianReg = JacobianReg(n=n)

    def evaluate_training_loss(self, x, y):
        x.requires_grad = True  # this is essential!
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss + self.alpha_jacob * self.loss_regularizer(x, y_hat)

    def loss_regularizer(self, x, y_hat):
        return self.JacobianReg(x, y_hat)


class EigenvalueRegularization(Trainer):

    def __init__(self, *, decoratee: BatchModifier, save_name=None, max_iter=100_000, optimizer='adam', lr=1e-3,
                 weight_decay=1e-5, alpha_spectra, only_last=False, slope=1, eig_start=10):
        super(EigenvalueRegularization, self).__init__(decoratee=decoratee, optimizer=optimizer,
                                               lr=lr, weight_decay=weight_decay, max_iter=max_iter,
                                               save_name=save_name)
        self.train_spectra = []  # store the (estimated) spectra of the network at the end of each epoch
        self.train_loss = []  # store the training loss (reported at the end of each epoch on the last batch)
        self.train_regularizer = []  # store the value of the regularizer during training
        self.valLoss = []  # validation loss
        self.val_regularizer = []  # validation regularizer
        self.eig_vec = []
        self.alpha_spectra = alpha_spectra  # strength of regularizer
        if not isinstance(slope, list):
            self.slopes = list(slope * np.ones(self.numHiddenLayers))
        else:
            self.slopes = slope
        self.eig_start = eig_start
        self.only_last = bool(only_last)
        self.eig_loader = None

    def add_eig_loader(self, X):
        if X.dataset.train:
            self.eig_loader = torch.utils.data.DataLoader(X.dataset, batch_size=len(X.dataset.data), shuffle=False,
                                          num_workers=X.num_workers,
                                          pin_memory=X.pin_memory)

    "Overwrites method in trainer"
    def evaluate_training_loss(self, x, y):
        hidden, y_hat = self.bothOutputs(x.to(self.device), only_last=self.only_last)  # feed data forward
        loss = self.loss(y_hat, y.to(self.device))  # compute loss

        "Compute spectra regularizer"
        return loss + self.alpha_spectra * self.spectra_regularizer(hidden)

    def compute_eig_vectors(self, x, desp='cpu'):
        with torch.no_grad():
            self.eval()
            self.eig_vec = None
            if self.cuda:
                "Delete old eigen vectors to free up space"
                torch.cuda.empty_cache()
            self.to(desp)
            hidden, _ = self.bothOutputs(x.to(desp), only_last=self.only_last)
            eigVec = compute_eig_vectors_only(hidden, self.only_last)
            self.eig_vec = eigVec
            self.train()
            if self.cuda:
                torch.cuda.empty_cache()
            self.to(self.device)
        return eigVec

    def spectra_regularizer(self, hidden):
        "Compute spectra regularizer"
        spectra_regul = torch.zeros(1, device=self.device)

        if self.only_last:
            spectra, rTemp = eigen_val_regulate(hidden[-1], self.eig_vec.to(self.device),
                                                start=self.eig_start, device=self.device,
                                                slope=self.slopes[-1])
            self.eig_vec.to('cpu')
            spectra_regul += rTemp
            if self.eig_vec.requires_grad:
                import warnings
                warnings.warn("This is not OK!!!")

        else:
            for idx in range(len(hidden)):
                spectra, rTemp = eigen_val_regulate(hidden[idx], self.eig_vec[idx].to(self.device),
                                                    start=self.eig_start, device=self.device, slope=self.slopes[idx])
                self.eig_vec[idx].to('cpu')
                spectra_regul += rTemp
                if self.eig_vec[0].requires_grad:
                    import warnings
                    warnings.warn("This is not OK!!!")
        return spectra_regul

    def train_epoch(self, X: DataLoader, X_full=None, desp=None):

        if X_full is None and self.eig_loader is None:
            self.add_eig_loader(X)
        elif X_full is None and self.eig_loader is not None:
            X_full, _ = next(iter(self.eig_loader))
        if desp is None:
            desp = self.device
        self.compute_eig_vectors(X_full.to(self.device), desp=desp)
        X_full.to('cpu', non_blocking=True)
        torch.cuda.empty_cache()
        # for _, (x, y) in enumerate(tqdm(X, desc="Training Elements", ascii=True, position=1, leave=True)):
        for _, (x, y) in enumerate(X):
            self.train_batch(x.to(self.device), y.to(self.device))
            x.to('cpu', non_blocking=True)
            y.to('cpu', non_blocking=True)


class EigenvalueAndJacobianRegularization(EigenvalueRegularization):
    def __init__(self, *, decoratee: BatchModifier, save_name=None, max_iter=100_000, optimizer='adam', lr=1e-3,
                 weight_decay=1e-5, alpha_spectra, alpha_jacob, only_last=False, n=1, slope=1, eig_start=10):
        super(EigenvalueAndJacobianRegularization, self).__init__(decoratee=decoratee, optimizer=optimizer,
                                                       lr=lr, weight_decay=weight_decay, max_iter=max_iter,
                                                       save_name=save_name, alpha_spectra=alpha_spectra,
                                                                  only_last=only_last, slope=slope, eig_start=eig_start)
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
        x.requires_grad = True  # this is essential!
        hidden, y_hat = self.bothOutputs(x.to(self.device))  # feed data forward
        loss = self.loss(y_hat, y.to(self.device))  # compute loss
        "Compute jacobian regularization"
        return loss + self.alpha_spectra * self.spectra_regularizer(hidden) + self.alpha_jacob * \
               self.loss_regularizer(x, y_hat)

    def loss_regularizer(self, x, y_hat):
        return self.JacobianReg(x, y_hat)
