import torch
import torch.nn as nn
from torch.optim import Adam, SGD, rmsprop
import numpy as np
import inspect
from functools import wraps
import copy
from copy import deepcopy
from uuid import uuid4
from tqdm import tqdm
from typing import Callable, Union

import unittest
import os

from torch.utils.data.dataloader import DataLoader
from torch import save
from utils import JacobianReg
from architectures import MLP, CNN

def filter_n_eval(func, **kwargs):
    """
    Takes kwargs and passes ONLY the named parameters that are specified in the callable func
    :param func: Callable for which we'll filter the kwargs and then pass them
    :param kwargs:
    :return:
    """
    args = inspect.signature(func)
    right_ones = kwargs.keys() & args.parameters.keys()
    newargs = {key: kwargs[key] for key in right_ones}
    return func(**newargs)


def counter(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # self is an instance of the class
        output = func(self, *args, **kwargs)
        self.no_minibatches += 1
        return output
    return wrapper

class BatchModfier(nn.Module):

    def __init__(self, *, decoratee, save_name=None, max_iter=100_000, optimizer='adam', lr=1e-3, weight_decay=1e-5):
        super(BatchModfier, self).__init__()
        self._architecture = decoratee
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
        return self._architecture(x)

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
            for _, (x,y) in enumerate(tqdm(X)):
                loss_tmp, mce_tmp = self.evaluate_test_loss(x,y)
                loss += loss_tmp
                mce += mce_tmp
        return loss/len(X), mce/len(X)

    def train_epoch(self, X: DataLoader):
        self.evaluate_dataset(X, function=self.train_batch)

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


class AdversarialTraining(BatchModfier):
    """
    Class that will be in charge of generating batches of adversarial images
    """
    def __init__(self, *, decoratee, eps, lr, gradSteps, noRestarts, training_type='pgd'):
        # def __init__(self, eps=0.3, lr=0.01, gradSteps=100, noRestarts=0, cuda=False):
        """
        Constructor for a first order adversary
        :param eps: radius of l infinity ball
        :param lr: learning rate
        :param gradSteps: number of gradient steps
        :param noRestarts: number of restarts
        """
        super(AdversarialTraining, self).__init__(decoratee=decoratee)
        self.eps = eps  # radius of l infinity ball
        self.lr = lr.to(decoratee.device)
        self.gradSteps = gradSteps  # number of gradient steps to take
        self.noRestarts = noRestarts  # number of restarts
        if training_type == 'FGSM':
            self._gen_input = self.FGSM
        elif training_type == 'PGD':
            self._gen_input = self.pgd_loop
        elif training_type == 'FREE':
            raise NotImplementedError

    def generate_adv_images(self, x_nat, y):
        return self._gen_input(x_nat, y)

# overwrites method in trainer base
    def evaluate_training_loss(self, x, y):
        # overwrites method in trainer
        x_adv = self.generate_adv_images(x, y)
        x_new = torch.cat((x, x_adv), 0)
        y_new = torch.cat((y, y), 0)
        return super(self).evaluate_training_loss(x_new, y_new)

    def pgd_loop(self, x_nat, y):
        losses = torch.zeros(self.noRestarts)
        xs = []
        for r in range(self.noRestarts):
            perturb = 2 * self.eps * torch.rand(x_nat.shape, device=x_nat.device) - self.eps
            xT, ellT = self.pgd(x_nat, x_nat + perturb, y)  # do pgd
            xs.append(xT)
            losses[r] = ellT
        idx = torch.argmax(losses)
        x = xs[idx]  # choose the one with the largest loss function
        ell = losses[idx]
        return x, ell

    def pgd(self, x_nat, x, y):
        """
        Perform projected gradient descent from Madry et al 2018
        :param x_nat: starting image
        :param x: starting point for optimization
        :param y: true label of image
        :return: x, the maximum found
        """
        for i in range(self.gradSteps):
            jacobian, ell = self.get_jacobian(copy.deepcopy(x), y)  # get jacobian
            x += self.lr * torch.sign(jacobian)  # take gradient step
            xT = x.detach()
            xT = self.clip(xT, x_nat.detach() - self.eps, x_nat.detach() + self.eps)
            xT = torch.clamp(xT, 0, 1)
            x = xT
        ell = self.loss(x, y)
        return x, ell.item()

    def FGSM(self, x_nat, y):
        jacobian, ell = self.get_jacobian(x_nat, y)  # get jacobian
        return x_nat + self.eps * torch.sign(jacobian), ell

    @staticmethod
    def clip(T, Tmin, Tmax):
        """

        :param T: input tensor
        :param Tmin: input tensor containing the minimal elements
        :param Tmax: input tensor containing the maximal elements
        :return:
        """
        return torch.max(torch.min(T, Tmax), Tmin)


class MLTraining(BatchModfier):

    def __init__(self, decoratee):
        super(MLTraining, self).__init__(decoratee=decoratee)


class Regularizer(nn.Module):

    def __init__(self, *, decoratee):
        super(Regularizer, self).__init__()
        self._trainer = decoratee

    def forward(self, x):
        return self._trainer(x)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self._trainer, item)


class NoRegularization(Regularizer):
    """
    No penalty class
    """
    def __init__(self, decoratee):
        super(NoRegularization, self).__init__(decoratee=decoratee)


class JacobianRegularization(Regularizer):

    def __init__(self, *, decoratee: Union[Regularizer, Trainer], alpha_jacob, n=-1):
        super(JacobianRegularization, self).__init__(decoratee=decoratee)
        self.alpha_jacob = alpha_jacob
        self.JacobianReg = JacobianReg(n=n)

    def evaluate_training_loss(self, x, y):
        x.requires_grad = True  # this is essential!
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        return loss + self.alpha_jacob * self.loss_regularizer(x, y_hat)

    def loss_regularizer(self, x, y_hat):
        return self.JacobianReg(x, y_hat)


class EigenvalueRegularization(Regularizer):

    def __init__(self, decoratee: Union[Regularizer, Trainer], *, alpha_spectra):
        # import pdb; pdb.set_trace()
        super(EigenvalueRegularization, self).__init__(decoratee=decoratee)
        self.train_spectra = []  # store the (estimated) spectra of the network at the end of each epoch
        self.train_loss = []  # store the training loss (reported at the end of each epoch on the last batch)
        self.train_regularizer = []  # store the value of the regularizer during training
        self.valLoss = []  # validation loss
        self.val_regularizer = []  # validation regularizer
        self.eig_vec = []
        self.eig_T = None
        self.alpha_spectra = alpha_spectra
        self.eig_start = 10
        self.EigDataLoader = None

# overwrites method in trainer
    def evaluate_training_loss(self, x, y):
        hidden, y_hat = self.bothOutputs(x.to(self.device))  # feed data forward
        loss = self.loss(y_hat, y.to(self.device))  # compute loss

        "Compute spectra regularizer"
        spectra_regul = torch.zeros(1, device=self.device)
        spectra_temp = []
        for idx in range(len(hidden)):
            spectra, rTemp = self.eigen_val_regulate(hidden[idx], self.eig_vec)  # compute spectra for each hidden layer
            with torch.no_grad():
                spectra_temp.append(spectra)
            spectra_regul += rTemp
        return loss + self.alpha_spectra * spectra_regul

    def eigen_val_regulate(self, x, v):
        """
        Function that approximates the eigenvalues of the matrix x, by finding them wrt some pseudo eigenvectors v and then
        penalizes eigenvalues that stray too far away from a power law
        :param x: hidden representations, N by D
        :param v: eigenvectors, D by D (each column is an eigenvector!)
        :param eigT: if the eigenspectra is already estimated, can just be passed in, else it is default as None
        :param start: index that states what eigenvalues to start regulating.
        :return: value of regularizer and the estimated spectra
        """
        eigT = self.eig_T
        start = self.eig_start
        if eigT is None:
            xt = x - torch.mean(x, 0)  # demean the data
            cov = xt.transpose(1, 0) @ xt / (x.shape[0] - 1)  # compute covariance matrix
            cov = (cov + cov.transpose(1, 0)) / 2
            eig = torch.diag(v.transpose(1, 0) @ cov @ v).to(self.device)

        else:
            eig = eigT

        eigs = torch.sort(eig, descending=True)[0]
        regul = torch.zeros(1, device=self.device)
        # slope = -1
        with torch.no_grad():
            alpha = eigs[start] * (start + 1)  # let the the constant be the largest eigenvalue

        for n in range(start + 1, eigs.shape[0]):
            if eigs[n] > 0:  # don't use negative eigenvalues
                regul += (eigs[n] / (alpha / (n + 1)) - 1) ** 2 + torch.relu(eigs[n] / (alpha / (n + 1)) - 1)
        return eigs, regul / x.shape[0]

    def compute_eig_vectors(self, x, y):
        hidden, outputs = self.bothOutputs(x.to(self.device))
        loss = self.loss(outputs, y.to(self.device))

        spectraTemp = []
        eigVec = []
        regul = torch.zeros(1, device=self.device)

        for idx in range(len(hidden)):
            hTemp = hidden[idx] - torch.mean(hidden[idx], 0)
            cov = hTemp.transpose(1, 0) @ hTemp / hTemp.shape[0]  # compute covariance matrix
            cov = cov.double()  # cast as 64bit to mitigate underflow in matrix factorization
            cov = (cov + cov.transpose(1, 0)) / 2
            _, eigTemp, vecTemp = torch.svd(cov, compute_uv=True)  # compute eigenvectors and values
            self.eig_T = eigTemp.float()
            vecTemp = vecTemp.float()
            eigTemp, rT = self.eigen_val_regulate(0, 0)  # compute regularizer
            regul += rT
            spectraTemp.append(eigTemp.cpu())  # save spectra
            self.eig_vec.append(vecTemp)

        return eigVec, loss, spectraTemp, regul.cpu().item()

# overwrites method in trainer
    def train_epoch(self, X: DataLoader):
        x, y = next(iter(self.EigDataLoader))

        with torch.no_grad():
            self.eigVec, loss, spectraTemp, regul = self.compute_eig_vectors(x, y)
            self.trainSpectra.append(spectraTemp)  # store computed eigenspectra
            self.trainLoss.append(loss.cpu().item())  # store training loss
            self.trainRegularizer.append(self.omega * regul)  # store value of regularizer
            self.eig_T = None
        for _, (x, y) in enumerate(tqdm(X)):
            self.train_batch(x, y)

    @staticmethod
    def estimate_slope(x, y):
        """
        y = beta * x^alpha + eps
        Goal is to obtain estimate of alpha and beta using linear regression
        """
        logx = np.log(x)
        logx = np.vstack((logx, np.ones(logx.shape)))
        logy = np.log(y)
        alpha, beta = np.linalg.lstsq(logx.T, logy)[0]
        return alpha, beta


class EigenvalueAndJacobianRegularization(EigenvalueRegularization, JacobianRegularization):

    def __init__(self, decoratee: Union[Regularizer, Trainer], *, alpha_spectra, alpha_jacob):
        EigenvalueRegularization.__init__(self, decoratee=decoratee, alpha_spectra=alpha_spectra)
        JacobianRegularization.__init__(self, decoratee=decoratee, alpha_jacob=alpha_jacob)

    # overwrites method in trainer
    def evaluate_training_loss(self, x, y):
        hidden, y_hat = self.bothOutputs(x.to(self.device))  # feed data forward
        loss = self.loss(y_hat, y.to(self.device))  # compute loss

        # Compute spectra regularizer
        spectra_regul = torch.zeros(1, device=self.device)
        spectra_temp = []
        for idx in range(len(hidden)):
            spectra, rTemp = self.eigen_val_regulate(hidden[idx], self.eig_vec)  # compute spectra for each hidden layer
            with torch.no_grad():
                spectra_temp.append(spectra)
            spectra_regul += rTemp
        return loss + self.alpha_spectra * spectra_regul + self.alpha_jacob * self.loss_regularizer(x, loss)


def ModelFactory(**kwargs):
    classes = {'mlp': MLP,
               'cnn': CNN,
               'adv': AdversarialTraining,
               'vanilla': MLTraining,
               'no': NoRegularization,
               'jac': JacobianRegularization,
               'eig': EigenvalueRegularization,
               'eigjac': EigenvalueAndJacobianRegularization
               }
    arch = filter_n_eval(classes[kwargs["architecture"].lower()], **kwargs)
    trainer = filter_n_eval(classes[kwargs["trainer"].lower()], decoratee=arch, **kwargs)
    model = filter_n_eval(classes[kwargs["regularizer"].lower()], decoratee=trainer, **kwargs)
    return model


class TestModel(unittest.TestCase):

    @staticmethod
    def create_model():
        kwargs = {"dims": [(28 * 28, 1000), (1000, 10)], "activation": "relu", "architecture": "mlp",
                  "trainer": "vanilla", "regularizer": "jac", "alpha_spectra": 1, "alpha_jacob": 1}
        return ModelFactory(**kwargs)

    @staticmethod
    def load_data():
        from torchvision import datasets, transforms
        kwargs = {'num_workers': 4, 'pin_memory': True}
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(root=os.getcwd(), train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=os.getcwd(), train=False, download=True, transform=transform)
        num_train = len(train_set)
        indices = list(range(num_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, sampler=train_sampler,
                                                   **kwargs)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, **kwargs)
        return train_loader, test_loader

    def test_assert(self):
        model = self.create_model()
        self.assertIsInstance(model, Regularizer)

    def test_forward(self):
        model = self.create_model()
        train_loader, test_loader = self.load_data()
        x, y = next(iter(train_loader))
        yhat = model(x)
        loss = model.loss(yhat, y)
        self.assertIsNotNone(loss)

    def test_parameters(self):
        model = self.create_model()
        model.parameters()
        cnt = 0
        for param in model.parameters():
            cnt += param.numel()
        self.assertEqual(28**2*1000+1000+1000*10+10, cnt)

    def test_train_batch(self):
        model = self.create_model()
        train_loader, test_loader = self.load_data()

        x,y = next(iter(train_loader))
        L_pre = model.evaluate_training_loss(x,y)
        model.train_batch(x,y)
        L_post = model.evaluate_training_loss(x,y)
        self.assertLess(L_post.item(), L_pre.item())

    def test_train_epoch(self):
        model = self.create_model()
        train_loader, test_loader = self.load_data()
        L_pre, mce_pre = model.evaluate_dataset_test_loss(test_loader)
        model.train_epoch(train_loader)
        L_post, mce_post = model.evaluate_dataset_test_loss(test_loader)
        self.assertLess(L_post.item(), L_pre.item())
        self.assertLess(mce_post.item(), mce_pre.item())

if __name__ == '__main__':
    unittest.main()
