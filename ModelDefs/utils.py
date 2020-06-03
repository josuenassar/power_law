import numpy as np

# from __future__ import division
import torch
import torch.nn as nn
from functools import wraps


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
        """
        Produce jacobian-vector product dy/dx dot v.

        Note that if you want to differentiate it,
        you need to make create_graph=True
        :param y:
        :param x:
        :param v:
        :param create_graph:
        :return:
        """

        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        grad_x, = torch.autograd.grad(flat_y, x, flat_v,
                                      retain_graph=True,
                                      create_graph=create_graph)
        return grad_x


def counter(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # self is an instance of the class
        output = func(self, *args, **kwargs)
        self.no_minibatches += 1
        return output
    return wrapper


def compute_eig_vectors(x, y, model, loss, device):
    hidden, outputs = model.bothOutputs(x.to(device))
    loss = loss(outputs, y.to(device))
    spectraTemp = []
    eigVec = []
    regul = torch.zeros(1, device=device)

    for idx in range(len(hidden)):
        hTemp = hidden[idx] - torch.mean(hidden[idx], 0)
        cov = hTemp.transpose(1, 0) @ hTemp / hTemp.shape[0]  # compute covariance matrix
        cov = cov.double()  # cast as 64bit to mitigate underflow in matrix factorization
        cov = (cov + cov.transpose(1, 0)) / 2
        # _, eigTemp, vecTemp = torch.svd(cov, compute_uv=True)  # compute eigenvectors and values
        eigTemp, vecTemp = torch.symeig(cov, eigenvectors=True)
        eig_T = eigTemp.float()
        vecTemp = vecTemp.float()
        eigTemp, rT = eigen_val_regulate(0, 0, eig_T, device)  # compute regularizer
        regul += rT
        spectraTemp.append(eigTemp.cpu())  # save spectra
        eigVec.append(vecTemp)

    return eigVec, loss, spectraTemp, regul.cpu().item()


def compute_eig_vectors_only(hidden, only_last=False):

    if only_last:
        hTemp = hidden[-1] - torch.mean(hidden[-1], 0)
        cov = hTemp.transpose(1, 0) @ hTemp / hTemp.shape[0]  # compute covariance matrix
        cov = cov.double()  # cast as 64bit to mitigate underflow in matrix factorization
        cov = (cov + cov.transpose(1, 0)) / 2
        _, vecTemp = torch.symeig(cov, eigenvectors=True)
        eigVec = vecTemp.float()
    else:
        eigVec = []
        for idx in range(len(hidden)):
            hTemp = hidden[idx] - torch.mean(hidden[idx], 0)
            cov = hTemp.transpose(1, 0) @ hTemp / hTemp.shape[0]  # compute covariance matrix
            cov = cov.double()  # cast as 64bit to mitigate underflow in matrix factorization
            cov = (cov + cov.transpose(1, 0)) / 2
            # _, _, vecTemp = torch.svd(cov, compute_uv=True)  # compute eigenvectors and values
            _, vecTemp = torch.symeig(cov, eigenvectors=True)
            vecTemp = vecTemp.float()
            eigVec.append(vecTemp.cpu())

    return eigVec


def compute_eig_values_only(hidden, only_last=False):

    if only_last:
        hTemp = hidden[-1] - torch.mean(hidden[-1], 0)
        cov = hTemp.transpose(1, 0) @ hTemp / hTemp.shape[0]  # compute covariance matrix
        cov = cov.double()  # cast as 64bit to mitigate underflow in matrix factorization
        cov = (cov + cov.transpose(1, 0)) / 2
        eigTemp, _ = torch.symeig(cov)
        eigVals = eigTemp.float()
    else:
        eigVals = []
        for idx in range(len(hidden)):
            hTemp = hidden[idx] - torch.mean(hidden[idx], 0)
            cov = hTemp.transpose(1, 0) @ hTemp / hTemp.shape[0]  # compute covariance matrix
            cov = cov.double()  # cast as 64bit to mitigate underflow in matrix factorization
            cov = (cov + cov.transpose(1, 0)) / 2
            # _, _, vecTemp = torch.svd(cov, compute_uv=True)  # compute eigenvectors and values
            eigTemp, _ = torch.symeig(cov)
            eigTemp = eigTemp.float()
            eigVals.append(eigTemp.cpu())
    return eigVals


def eigen_val_regulate(x, v, eigT=None, start=10, device='cpu', slope=1):
    """
    Function that approximates the eigenvalues of the matrix x, by finding them wrt some pseudo eigenvectors v and then
    penalizes eigenvalues that stray too far away from a power law
    :param x: hidden representations, N by D
    :param v: eigenvectors, D by D (each column is an eigenvector!)
    :param eigT: if the eigenspectra is already estimated, can just be passed in, else it is default as None
    :param start: index that states what eigenvalues to start regulating.
    :param device: what device to run things on (cpu or gpu)
    :param slope: n^(-slope),
    :return: value of regularizer and the estimated spectra
    """
    if eigT is None:
        xt = x - torch.mean(x, 0)  # demean the data
        cov = xt.transpose(1, 0) @ xt / (x.shape[0] - 1)  # compute covariance matrix
        cov = (cov + cov.transpose(1, 0)) / 2
        eig = torch.diag(v.transpose(1, 0) @ cov @ v).to(device)

    else:
        eig = eigT

    eigs = torch.sort(eig, descending=True)[0]
    regul = torch.zeros(1, device=device)
    # slope = -1
    with torch.no_grad():
        beta = eigs[start] * (start + 1) ** slope

    for n in range(start + 1, eigs.shape[0]):
        if eigs[n] > 0:  # don't use negative eigenvalues
            gamma = beta / ((n + 1) ** slope)
            regul += (eigs[n] / gamma - 1) ** 2 + torch.relu(eigs[n] / gamma - 1)
    return eigs, regul / x.shape[1]


def jacobian_eig(x, eig_idx, model, layer=-1):
    "Get gradient for images"
    x.requires_grad_(True)
    hiddens, _ = model.bothOutputs(x)
    temp2 = hiddens[layer] - torch.mean(hiddens[layer], 0)
    print(temp2.shape)
    cov = temp2.t() @ temp2 / temp2.shape[0]
    cov = (cov + cov.t()) / 2
    eigs, _ = torch.symeig(cov, eigenvectors=True)
    print(eigs.shape)
    gradient = torch.autograd.grad(eigs[-1 + eig_idx], x)[0]
    return gradient
