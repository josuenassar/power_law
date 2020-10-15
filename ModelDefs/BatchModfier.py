import torch
import torch.nn as nn
import copy


class BatchModifier(nn.Module):

    def __init__(self, *, decoratee):
        super(BatchModifier, self).__init__()
        self._architecture = decoratee

    def forward(self, x):
        return self._architecture(x)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self._architecture, item)


class AdversarialTraining(BatchModifier):
    """
    Class that will be in charge of generating batches of adversarial images
    """
    def __init__(self, *, decoratee, eps, lr_pgd, gradSteps, noRestarts, training_type='PGD', lb=0, ub=1):
        super(AdversarialTraining, self).__init__(decoratee=decoratee)
        self.eps = eps  # radius of l infinity ball
        self.lb = lb
        self.ub = ub
        # self.lr = alpha.to(decoratee.device)
        self.lr = lr_pgd
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
    
    def prepare_batch(self, x, y):
        x_adv, _ = self.generate_adv_images(x, y)
        x_new = torch.cat((x, x_adv), 0).detach()
        y_new = torch.cat((y, y), 0)
        return x_new, y_new

    def pgd_loop(self, x_nat, y):
        losses = torch.zeros(self.noRestarts)
        xs = []
        for r in range(self.noRestarts):
            perturb = 2 * self.eps * torch.rand(x_nat.shape, device=x_nat.device) - self.eps
            xT, ellT = self.pgd(x_nat, torch.clamp(x_nat + perturb, 0, 1), y)  # do pgd
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
            jacobian, ell = self.get_jacobian(x, y)  # get jacobian
            xT = (x + self.lr * torch.sign(jacobian)).detach()
            with torch.no_grad():
                xT = self.clip(xT, x_nat.detach() - self.eps, x_nat.detach() + self.eps)
                if x_nat.shape[1] == 1:
                    # if just one channel, then lb and ub are just numbers
                    xT = torch.clamp(xT, self.lb, self.ub)
                else:
                    # for more than one channel, need channel specific lb and ub
                    xT = self.clip(xT, self.lb, self.ub)
                x = xT
                del xT
            ell = self.loss(self._architecture(x), y)
            return x, ell.item()

    def FGSM(self, x_nat, y):
        perturb = 2 * self.eps * torch.rand(x_nat.shape, device=x_nat.device) - self.eps
        if x_nat.shape[1] == 1:
            # if just one channel, then lb and ub are just numbers
            x_nat = torch.clamp(x_nat + perturb, self.lb, self.ub)
        else:
            # for more than one channel, need channel specific lb and ub
            x_nat = self.clip(x_nat + perturb, self.lb, self.ub)
        jacobian, ell = self.get_jacobian(x_nat, y)  # get jacobian
        x_nat = x_nat.detach()
        with torch.no_grad():
            if x_nat.shape[1] == 1:
                # if just one channel, then lb and ub are just numbers
                x_nat = torch.clamp(x_nat + self.eps * torch.sign(jacobian), self.lb, self.ub).detach()
            else:
                # for more than one channel, need channel specific lb and ub
                x_nat = self.clip(x_nat + self.eps * torch.sign(jacobian), self.lb, self.ub).detach()
            return x_nat, ell

    @staticmethod
    def clip(T, Tmin, Tmax):
        """

        :param T: input tensor
        :param Tmin: input tensor containing the minimal elements
        :param Tmax: input tensor containing the maximal elements
        :return:
        """
        return torch.max(torch.min(T, Tmax), Tmin)


class MLTraining(BatchModifier):

    def __init__(self, decoratee):
        super(MLTraining, self).__init__(decoratee=decoratee)
    
    @staticmethod        
    def prepare_batch(x, y):
        return x, y
