import abc
import torch
import utils
from utils import computeEigVectors, compute_loss
import copy
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


class Trainer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def evaluate_training_loss(cls, x, y):
        raise NotImplementedError


class Adversary(Trainer):
    "Class that will be in charge of generating batches of adversarial images"
    def __init__(self, *, eps, lr, gradSteps, noRestarts, cuda, training_type='pgd'):
        # def __init__(self, eps=0.3, lr=0.01, gradSteps=100, noRestarts=0, cuda=False):
        """
        Constructor for a first order adversary
        :param eps: radius of l infinity ball
        :param lr: learning rate
        :param gradSteps: number of gradient steps
        :param noRestarts: number of restarts
        """
        self.eps = eps  # radius of l infinity ball
        self.lr = lr.cuda() if cuda == True else lr.cpu()  # learning rate
        self.gradSteps = gradSteps  # number of gradient steps to take
        self.noRestarts = noRestarts  # number of restarts
        self.cuda = cuda
        if training_type == 'FGSM':
            self._gen_input = self.FGSM
        elif training_type == 'PGD':
            self._gen_input = self.pgd_loop
        elif training_type == 'FREE':
            raise NotImplementedError

    def generateAdvImages(self, x_nat, y):
        return self._gen_input(x_nat, y, self.network, self.loss)

    def evaluate_training_loss(self, x,y):
        x_adv = self.generateAdvImages(x,y)
        return self.evaluate_loss(x_adv,y)

    def pgd_loop(self, network, x_nat, y, loss):
        losses = torch.zeros(self.noRestarts)
        xs = []
        for r in range(self.noRestarts):
            perturb = 2 * self.eps * torch.rand(x_nat.shape, device=x_nat.device) - self.eps
            xT, ellT = self.pgd(x_nat, x_nat + perturb, y, network, loss)  # do pgd
            xs.append(xT)
            losses[r] = ellT
        idx = torch.argmax(losses)
        x = xs[idx]  # choose the one with the largest loss function
        ell = losses[idx]
        return x, ell

    def pgd(self, x_nat, x, y, network, loss):
        """
        Perform projected gradient descent from Madry et al 2018
        :param x_nat: starting image
        :param x: starting point for optimization
        :param y: true label of image
        :param network: network
        :param loss: loss function
        :return: x, the maximum found
        """
        for i in range(self.gradSteps):
            jacobian, ell = utils.get_jacobian(network, copy.deepcopy(x), y, loss, cuda=self.cuda)  # get jacobian
            x += self.lr * torch.sign(jacobian)  # take gradient step
            xT = x.detach()
            xT = utils.clip(xT, x_nat.detach() - self.eps,
                            x_nat.detach() + self.eps)
            xT = torch.clamp(xT, 0, 1)
            x = xT
        ell = loss(x, y)
        return x, ell.item()

    def FGSM(self, network, x_nat, y, loss):
        jacobian, ell = utils.get_jacobian(network, x_nat, y, loss, cuda=self.cuda)  # get jacobian
        return x_nat + self.eps * torch.sign(jacobian), ell


class Regular(Trainer):

    def __init__(self, **kwargs):
        self.test = 'test'

    def evaluate_training_loss(self, x,y):
        return self.evaluate_loss(x,y)


class Penalized(Trainer):

    def __init__(cls, *, omega) -> None:
        super().__init__()
        cls.trainSpectra = []  # store the (estimated) spectra of the network at the end of each epoch
        cls.trainLoss = []  # store the training loss (reported at the end of each epoch on the last batch)
        cls.trainRegularizer = []  # store the value of the regularizer during training
        cls.valLoss = []  # validation loss
        cls.valRegularizer = []  # validation regularizer
        cls.omega = omega

    def evaluate_training_loss(cls, x, y):
        loss, regul = utils.compute_loss(cls, x, y, cls.loss, alpha=cls.omega, cuda=cls.cuda)
        return loss + cls.omega*regul, None

    def computeEigVectors(cls, x, y):
        return utils.computeEigVectors(cls, x, y, cls.loss, alpha=cls.omega, cuda=cls.cuda)

    def train_epoch(cls, X:DataLoader):
        x, y = next(iter(cls.EigDataLoader))

        with torch.no_grad():
            cls.eigVec, loss, spectraTemp, regul = computeEigVectors(x,y)
            cls.trainSpectra.append(spectraTemp)  # store computed eigenspectra
            cls.trainLoss.append(loss.cpu().item())  # store training loss
            cls.trainRegularizer.append(cls.omega * regul)  # store value of regularizer

        for _, x, y in enumerate(tqdm(X)):
            cls.train_batch(x, y)

