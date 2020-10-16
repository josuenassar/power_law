from torch import nn
import math
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint_sequential
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

    def bothOutputs(self, x, only_last=False):
        raise NotImplementedError

    def get_jacobian(self, x, y):
        x.requires_grad_(True)
        y_hat = self(x.to(self.device))
        ell = self.loss(y_hat, y.to(self.device))
        gradient = torch.autograd.grad(ell, x)[0]
        return gradient.data, ell.item()


class Linear(ModelArchitecture):
    def __init__(self, dx, dy, cuda=False):
        super().__init__(cuda=cuda)
        self.sequential = nn.Sequential(nn.Linear(dx, dy))
        self.max_neurons = dx

    def forward(self, x):
        return self.sequential(x)

    def bothOutputs(self, x, only_last=False):
        return None, self.forward(x)


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
    def __init__(self, cuda=False, R=None, mu=None, demean=True):
        super().__init__()
        self.device = 'cuda' if cuda else 'cpu'
        self.R = R
        self.demean = demean
        self.mu = mu

    def forward(self, input):
        "Compute covariance"
        if self.demean:
            if self.mu is None:
                temp = input - torch.mean(input, 0)
            else:
                temp = input - self.mu
        else:
            temp = input

        if self.R is None:
            cov = temp.transpose(1, 0) @ temp / temp.shape[0]  # compute covariance matrix
            cov = (cov + cov.transpose(1, 0)) / 2 + 1e-5 * torch.eye(cov.shape[0], device=self.device)
            R = torch.cholesky(cov)  # returns the lower cholesky matrix
        else:
            R = self.R
        Y, _ = torch.triangular_solve(temp.transpose(1, 0), R, upper=False)
        return Y.transpose(1, 0)


class Flat(ModelArchitecture):
    "A fully connected, 3 layer network where one of the hidden layers will have a white spectra"
    def __init__(self, *, dims, activation='relu', bn=False, cuda=False, R=None, demean=False):
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
                modules.append(Whiten(cuda=cuda, R=R, demean=demean))
        modules.append(nn.Linear(dims[-1][0], dims[-1][1]))
        self.sequential = nn.Sequential(*modules)
        self.max_neurons = max([dims[n][1] for n in range(self.numHiddenLayers)])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # TODO vectorize inputs
        return self.sequential(x)

    def bothOutputs(self, x, only_last=False):
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
            convModules.append(nn.Conv2d(dims[idx][0], dims[idx][-1], kernel_size=(3, 3)))
            if bn:
                convModules.append(nn.BatchNorm2d(dims[idx][1]))
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

    def bothOutputs(self, x, only_last=False):
        hidden = [None] * self.numHiddenLayers
        convHidden = [None] * self.numConvLayers
        if self.bn:
            ell = 4
        else:
            ell = 3
        for idx in range(self.numConvLayers):
            if idx == 0:
                convHidden[0] = self.convSequential[ell * idx: ell * idx + ell](x)
            else:
                convHidden[idx] = self.convSequential[ell * idx: ell * idx + ell](convHidden[idx - 1])
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


class CNN_Flat(CNN):
    def __init__(self, *, dims, activation='relu', bn=False, cuda=False, h_in=28, w_in=28, R=None, demean=False):
        super().__init__(dims=dims, activation=activation, bn=bn, cuda=cuda, h_in=h_in, w_in=w_in)
        self.flat = Whiten(cuda=cuda, R=R, demean=demean)

    def forward(self, x):
        # TODO remove the reshaping
        hT = self.convSequential(x)
        return self.linSequential(self.flat(hT.view(-1, hT.shape[1] * hT.shape[2] * hT.shape[3])))

    def bothOutputs(self, x, only_last=False):
        hidden = [None] * self.numHiddenLayers
        convHidden = [None] * self.numConvLayers
        if self.bn:
            ell = 4
        else:
            ell = 3
        for idx in range(self.numConvLayers):
            if idx == 0:
                convHidden[0] = self.convSequential[ell * idx: ell * idx + ell](x)
            else:
                convHidden[idx] = self.convSequential[ell * idx: ell * idx + ell](convHidden[idx - 1])
            temp = convHidden[idx].view(-1, convHidden[idx].shape[1] * convHidden[idx].shape[2]
                                               * convHidden[idx].shape[3])
            if idx + 1 == self.numConvLayers:
                hidden[idx] = self.flat(temp)
            else:
                hidden[idx] = temp
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


"ResNet vibes"


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(ModelArchitecture):
    def __init__(self, block, num_blocks, num_classes=10, cuda=False, n_filters=4, checkpoint=True, nchunks=1):
        super(ResNet, self).__init__(cuda=cuda)
        self.in_planes = n_filters
        self.layer0 = nn.Sequential(nn.Conv2d(3, n_filters, kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.BatchNorm2d(n_filters),nn.ReLU())

        self.layer1 = self._make_layer(block, n_filters, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * n_filters, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * n_filters, num_blocks[2], stride=2)
        self.linear = nn.Linear(int(64 / (16 / n_filters)), num_classes)

        self.apply(_weights_init)
        self.checkpoint = checkpoint
        self.checkpoint = lambda f,inpt, **kv, : checkpoint_sequential(f,nchunks, inpt, **kv) if self.checkpoint \
            else lambda f, inpt, **kv: f(inpt, **kv)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        cp = self.checkpoint
        out = cp(self.layer0, x)
        out = cp(self.layer1, out)
        out = cp(self.layer2, out)
        out = cp(self.layer3, out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def bothOutputs(self, x, only_last=False):
        cp = self.checkpoint
        hiddens = []
        # First block
        out = cp(self.layer0, x)
        out = cp(self.layer1, out)
        if not only_last:
            hiddens.append(out.view(out.size(0), -1))

        # Second block
        out = cp(self.layer2,out)
        if not only_last:
            hiddens.append(out.view(out.size(0), -1))

        # Third block
        out = cp(self.layer3, out)
        # out = self.layer3(out)
        hiddens.append(out.view(out.size(0), -1))

        # Read out
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return hiddens, out


def resnet20(cuda=False, n_filters=4, checkpoint=True):
    return ResNet(BasicBlock, [3, 3, 3], cuda=cuda, n_filters=n_filters, checkpoint=checkpoint)
