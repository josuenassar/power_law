import torch
import torch.nn as nn
from torch.optim import SGD, Adam, rmsprop
import numpy as np
import numpy.random as npr
from numpy import newaxis as na
from tqdm import tqdm
import utils


# TODO: First get regular and regularized training working
class Trainer:
    "Class that will handle all things training, OH YEAH!"
    def __init__(self, model, lr=1e-3, optimizer='sgd', batch_size=128, adversary=None, alpha_adversary=0,
                 alpha_regularizer=0, lr_scheduler=None, weight_decay=0, max_epochs=100,
                 path_load='../data/MNIST/mnist.npy', frac_val=0.1, cuda=False):
        self.model = model  # DNN
        self.lr = lr  # (initial) learning rate
        self.batch_size = batch_size  # batch size
        self.adversary = adversary  # Adversary for adversarial training
        self.alpha_adversary = alpha_adversary
        self.alpha_regularizer = alpha_regularizer  # strength of spectra regularizer
        self.max_epochs = max_epochs
        self.path_load = path_load  # path to load in data
        self.frac_val = frac_val  # fraction of data to use for validation
        self.cuda = cuda  # boolean flag that determines whether we should use gpu or not

        if optimizer == 'sgd':
            self.optim = SGD(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        elif optimizer == 'adam':
            self.optim = Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        elif optimizer == 'rms':
            self.optim = rmsprop(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            print('Slow down there buddy, not an option')

    def load_data(self):
        train, val, test = utils.split_mnist(self.path, self.frac_val)
        return train, val, test

    def forward_pass(self, data, labels):
        "Excelsior!"
        loss, regul = utils.compute_loss(self.model, data, labels, self.model.loss, self.model.regularizer,
                                         cuda=self.cuda)
        return loss, regul

    def training(self):
        train, val, test = self.load_data()  # load in data for training
        trainData, trainLabels = train[0], train[1]
        valData, valLabels = val[0], val[1]
        testData, testLabels = test[0], test[1]
        numSamples = trainData.shape[0]  # number of total data points

        for epoch in tqdm(range(self.max_epochs), desc="Epochs", ascii=True, position=0, leave=False):
            loss, regul = self.forward_pass(trainData, trainLabels)  # Compute loss and regularizer
            tot_loss = loss + self.alpha_regularizer * regul
            tot_loss.backward()  # backprop!
            self.optim.step()  # take a gradient step

    def test



