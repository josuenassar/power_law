import torch
import inspect
import unittest
import os

from torch.utils.data.dataloader import DataLoader
from ModelDefs.architectures import MLP, CNN, Flat, CNN_Flat, resnet20, vgg9, vgg11, vgg19
from ModelDefs.BatchModfier import AdversarialTraining, MLTraining
from ModelDefs.trainers import NoRegularization, JacobianRegularization,\
    EigenvalueAndJacobianRegularization, EigenvalueRegularization


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


def ModelFactory(**kwargs):
    classes = {'mlp': MLP,
               'cnn': CNN,
               'flat': Flat,
               'cnn_flat': CNN_Flat,
               'resnet': resnet20,
               'vgg9': vgg9,
               'vgg11': vgg11,
               'vgg19': vgg19,
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
    model.to(arch.device)
    return model


class TestModel(unittest.TestCase):

    @staticmethod
    def create_model():
        kwargs = {"dims": [(28 * 28, 1000), (1000, 10)], "activation": "relu", "architecture": "mlp",
                  "trainer": "vanilla", "regularizer": "jac", "alpha_spectra": 1, "alpha_jacob": 1,
                  "optimizer":"adam","lr":1e-3, "weight_decay":1e-2}
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

    # def test_assert(self):
    #     model = self.create_model()
    #     self.assertIsInstance(model, Regularizer)
    # TODO change this assert once Regularizer is replaced with the new trainer

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
