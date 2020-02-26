from torch import nn
import math
import torch
# Architectures

# TODO: Fix batch norm issue

class ModelArchitecture(nn.Module):
    """
    A meta class that defines the forward propagation method of nn.Module
    Additionally, it exposes the hidden layer representations for each forward call
    in the bothOutputs method
    """
    def __init__(self, *, cuda=True):
        super(ModelArchitecture, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.device = 'cuda' if cuda else 'cpu'

    def forward(self, x):
        raise NotImplementedError

    def bothOutputs(self, x):
        raise NotImplementedError

    def get_jacobian(self, x, y):
        x.requires_grad_(True)
        y_hat = self(x.to(self.device))
        ell = self.loss(y_hat, y.to(self.device))
        gradient = torch.autograd.grad(ell, x)[0]
        return gradient.data, ell.item()


class MLP(ModelArchitecture):
    """
    Multilayer perceptron with a variable amount of hidden layers
    """
    def __init__(self, *, dims, activation='relu', bn=False, cuda=False):
        """
        Constructor for MLP with a variable number of hidden layers
        :param dims: A list of N tuples where the first N -1 determine the N - 1 hidden layers and the last tuple
        determines the output layer
        :param activation: a string that determines which activation layer to use. The default is relu
        """
        super().__init__(cuda=cuda)
        self.numHiddenLayers = len(dims[:-1])  # number of hidden layers in the network
        self.bn = bn
        modules = []
        for idx in range(len(dims) - 1):
            modules.append(nn.Linear(dims[idx][0], dims[idx][1]))
            if activation == 'relu':
                modules.append(nn.ReLU())
            else:
                modules.append(nn.Tanh())
            if bn:
                modules.append(nn.BatchNorm1d(dims[idx][1]))
        modules.append(nn.Linear(dims[-1][0], dims[-1][1]))
        self.sequential = nn.Sequential(*modules)
        self.max_neurons = max([dims[n][1] for n in range(self.numHiddenLayers)])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # TODO vectorize inputs
        return self.sequential(x)

    def bothOutputs(self, x):
        hidden = [None] * self.numHiddenLayers
        x = x.view(x.size(0), -1)
        if self.bn:
            ell = 3
        else:
            ell = 2
        for idx in range(self.numHiddenLayers):

            if idx == 0:
                hidden[idx] = self.sequential[ell * idx: ell * idx + ell](x)
            else:
                hidden[idx] = self.sequential[ell * idx: ell * idx + ell](hidden[idx - 1])

        return hidden, self.sequential[-1](hidden[-1])


class Whiten(nn.Module):
    def __init__(self, cuda=False, R=None):
        super().__init__()
        self.device = 'cuda' if cuda else 'cpu'
        self.R = R

    def forward(self, input):
        "Compute covariance"
        if self.R is None:
            temp = input - torch.mean(input, 0)
            cov = temp.transpose(1, 0) @ temp / temp.shape[0]  # compute covariance matrix
            cov = (cov + cov.transpose(1, 0)) / 2 + 1e-5 * torch.eye(cov.shape[0], device=self.device)
            R = torch.cholesky(cov)  # returns the upper cholesky matrix
        else:
            R = self.R
            temp = input - torch.mean(input, 0)
        Y, _ = torch.triangular_solve(temp.transpose(1, 0), R, upper=False)
        return Y.transpose(1, 0)


class Flat(ModelArchitecture):
    "A fully connected, 3 layer network where one of the hidden layers will have a white spectra"
    def __init__(self, *, dims, activation='relu', bn=False, cuda=False, R=None):
        super().__init__(cuda=cuda)
        place = dims[0]
        dims = dims[1:]
        self.numHiddenLayers = len(dims[:-1])  # number of hidden layers in the network
        assert self.numHiddenLayers == 3
        self.bn = bn
        modules = []
        for idx in range(len(dims) - 1):
            modules.append(nn.Linear(dims[idx][0], dims[idx][1]))
            if activation == 'relu':
                modules.append(nn.ReLU())
            else:
                modules.append(nn.Tanh())
            if bn:
                modules.append(nn.BatchNorm1d(dims[idx][1]))
            if idx == place:
                modules.append(Whiten(cuda=cuda, R=R))
        modules.append(nn.Linear(dims[-1][0], dims[-1][1]))
        self.sequential = nn.Sequential(*modules)
        self.max_neurons = max([dims[n][1] for n in range(self.numHiddenLayers)])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # TODO vectorize inputs
        return self.sequential(x)

    def bothOutputs(self, x):
        hidden = [None] * self.numHiddenLayers
        x = x.view(x.size(0), -1)

        if self.bn:
            indices = [0, 3, 7, 10]
        else:
            indices = [0, 2, 5, 7]

        for idx in range(self.numHiddenLayers):
            if idx == 0:
                hidden[idx] = self.sequential[indices[idx]:indices[idx + 1]](x)
            else:
                hidden[idx] = self.sequential[indices[idx]:indices[idx + 1]](hidden[idx - 1])
        return hidden, self.sequential[-1](hidden[-1])


class CNN(ModelArchitecture):
    """
    CNN architecture with a variable number of convolutional layers and a variable number of fully connected layers
    """
    def __init__(self, *, dims, activation='relu', bn=False, cuda=False, h_in=28, w_in=28):
        """
        Constructor for CNN
        :param dims: A list of N tuples where the first element states how many convolutional layers to use
        are defined as (# input channels, kernel size, # output channels)
        :param activation: a string that determines which activation layer to use. The default is relu
        """
        super().__init__(cuda=cuda)
        self.numConvLayers = dims[0]
        dims = dims[1:]
        self.bn = bn
        self.numHiddenLayers = len(dims) - 1  # number of hidden layers in the network
        self.max_neurons = 0

        # Construct convolutional layers
        convModules = []
        for idx in range(self.numConvLayers):
            convModules.append(nn.Conv2d(dims[idx][0], dims[idx][-1], kernel_size=(5, 5)))
            # if bn:
            #     convModules.append(nn.BatchNorm1d(dims[idx][1]))
            if activation == 'relu':
                convModules.append(nn.ReLU())
            else:
                convModules.append(nn.Tanh())
            convModules.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            h_in = math.floor(h_in / 2 - 1)
            w_in = math.floor(w_in / 2 - 1)
            self.max_neurons = max([self.max_neurons, h_in * w_in * dims[idx][-1]])

        self.convSequential = nn.Sequential(*convModules)  # convolution layers

        # Construct fully connected layers
        linModules = []
        for idx in range(self.numConvLayers, len(dims) - 1):
            linModules.append(nn.Linear(dims[idx][0], dims[idx][1]))
            if bn:
                linModules.append(nn.BatchNorm1d(dims[idx][1]))
            if activation == 'relu':
                linModules.append(nn.ReLU())
            else:
                linModules.append(nn.Tanh())
            self.max_neurons = max([self.max_neurons, dims[idx][1]])
        linModules.append(nn.Linear(dims[-1][0], dims[-1][1]))
        self.linSequential = nn.Sequential(*linModules)
        # self.eigVec = [None] * (len(dims) - 1)  # eigenvectors for all hidden layers

    def forward(self, x):
        # TODO remove the reshaping
        hT = self.convSequential(x)
        return self.linSequential(hT.view(-1, hT.shape[1] * hT.shape[2] * hT.shape[3]))

    def bothOutputs(self, x):
        hidden = [None] * self.numHiddenLayers
        convHidden = [None] * self.numConvLayers
        for idx in range(self.numConvLayers):
            if idx == 0:
                convHidden[0] = self.convSequential[3 * idx: 3 * idx + 3](x)
            else:
                convHidden[idx] = self.convSequential[3 * idx: 3 * idx + 3](convHidden[idx - 1])
            hidden[idx] = convHidden[idx].view(-1, convHidden[idx].shape[1] * convHidden[idx].shape[2]
                                               * convHidden[idx].shape[3])
        if self.bn:
            ell = 3
        else:
            ell = 2

        for idx in range(self.numHiddenLayers - self.numConvLayers):
            if idx == 0:
                hidden[self.numConvLayers] = self.linSequential[ell * idx: ell * idx + ell](hidden[self.numConvLayers - 1])
            else:
                hidden[self.numConvLayers + idx] = self.linSequential[ell * idx: ell * idx + ell](hidden[self.numConvLayers
                                                                                                         + idx - 1])

        return hidden, self.linSequential[-1](hidden[-1])

