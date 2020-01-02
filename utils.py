import numpy as np
import numpy.random as npr
import torch

# from __future__ import division
import torch
import torch.nn as nn
import torch.autograd as autograd


# def get_data(dataset, cuda, batch_size,_seed, h0, data_dir):
    # from ops.transformsParams import CIFAR10, SVHN
    # from torchvision import datasets, transforms
    # from ops.utils import SubsetSequentialSampler
    #
    # kwargs = {'num_workers': 4,op 'pin_memory': True} if cuda else {}
    #
    # if dataset == 'MNIST':
    #     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    #     train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    #     test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    # elif dataset == 'CIFAR10':
    #     transform_train, transform_eval = CIFAR10(h0)
    #     train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    #     stats_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_eval)
    #     test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_eval)
    # elif dataset == 'SVHN':
    #     transform_train, transform_eval = SVHN(h0)
    #     train_set = datasets.SVHN(root=data_dir, split='train', download=True,
    #                               transform=transform_train)
    #     test_set = datasets.SVHN(root=data_dir, split='test', download=True,
    #                              transform=transform_eval)
    #     stats_set = datasets.SVHN(root=data_dir, split='train', download=True,
    #                              transform=transform_eval)
    #
    # num_train = len(train_set)
    # indices = list(range(num_train))
    # np.random.seed(_seed)
    # np.random.shuffle(indices)
    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    # stats_idx = np.array(indices)[np.random.choice(len(indices), 30)]
    # stats_idx = stats_idx.tolist()
    #
    # stats_sampler = SubsetSequentialSampler(stats_idx)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, **kwargs)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
    # stats_loader = torch.utils.data.DataLoader(stats_set, batch_size=30, sampler=stats_sampler)
    # return train_loader, test_loader, stats_loader

def clip(T, Tmin, Tmax):
    """

    :param T: input tensor
    :param Tmin: input tensor containing the minimal elements
    :param Tmax: input tensor containing the maximal elements
    :return:
    """
    return torch.max(torch.min(T, Tmax), Tmin)


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


'''
    Copyright (c) Facebook, Inc. and its affiliates.

    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.
    
    PyTorch implementation of Jacobian regularization described in [1].

    [1] Judy Hoffman, Daniel A. Roberts, and Sho Yaida,
        "Robust Learning with Jacobian Regularization," 2019.
        [arxiv:1908.02729](https://arxiv.org/abs/1908.02729)
'''


class JacobianReg(nn.Module):
    '''
    Loss criterion that computes the trace of the square of the Jacobian.

    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output
            space and projection is non-random and orthonormal, yielding
            the exact result.  For any reasonable batch size, the default
            (n=1) should be sufficient.
    '''
    def __init__(self, n=1):
        """
        creates a module that computes the Frobenius norm of the input-output Jacobian
        :param n: if n == -1 uses the entire Jacobian matrix;
         if n > 0 uses n random-vector Jacobian products to estimate the norm
        """
        assert n == -1 or n > 0
        self.n = n
        super(JacobianReg, self).__init__()

    def forward(self, x, y):
        """
        computes (1/2) tr |dy/dx|^2
        :param x: input
        :param y: output of the network
        :return:
        """
        B,C = y.shape
        if self.n == -1:
            num_proj = C
        else:
            num_proj = self.n
        J2 = 0
        for ii in range(num_proj):
            if self.n == -1:
                # orthonormal vector, sequentially spanned
                v=torch.zeros(B,C)
                v[:,ii]=1
            else:
                # random properly-normalized vector for each sample
                v = self._random_vector(C=C,B=B)
            if x.is_cuda:
                v = v.cuda()
            Jv = self._jacobian_vector_product(y, x, v, create_graph=True)
            J2 += C*torch.norm(Jv)**2 / (num_proj*B)
        R = (1/2)*J2
        return R

    @staticmethod
    def _random_vector(C, B):
        """
        creates a random vector of dimension C with a norm of C^(1/2)
        (as needed for the projection formula to work)

        :param C: int, number of classes
        :param B: int, number of batch elements
        :return:
        """
        if C == 1:
            return torch.ones(B)
        v=torch.randn(B,C)
        arxilirary_zero=torch.zeros(B,C)
        vnorm=torch.norm(v, 2, 1,True)
        v=torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
        return v

    @staticmethod
    def _jacobian_vector_product(y, x, v, create_graph=False):
        '''
        Produce jacobian-vector product dy/dx dot v.

        Note that if you want to differentiate it,
        you need to make create_graph=True
        '''
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        grad_x, = torch.autograd.grad(flat_y, x, flat_v,
                                      retain_graph=True,
                                      create_graph=create_graph)
        return grad_x
