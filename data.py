from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np
from trainers import EigenvalueRegularization
data_dir = '../'


def get_data(dataset, batch_size, _seed, validate):
    validation_split = 10_000
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if dataset == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        num_train = len(train_set)
        indices = list(range(num_train))
        if not validate:
            train_sampler = sampler.SubsetRandomSampler(indices)
        else:
            np.random.seed(_seed)
            np.random.shuffle(indices)
            train_idx, valid_idx, = indices[0:-validation_split], indices[-validation_split:-1]
            train_sampler = sampler.SubsetRandomSampler(train_idx)
            test_sampler = sampler.SubsetRandomSampler(valid_idx)
            test_loader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, **kwargs)

    elif dataset == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49137255, 0.48235294,
                                                                                     0.44666667),
                             (0.24705882, 0.24352941, 0.26156863))])
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        num_train = len(train_set)
        indices = list(range(num_train))
        if not validate:
            train_sampler = sampler.SubsetRandomSampler(indices)
        else:
            np.random.seed(_seed)
            np.random.shuffle(indices)
            train_idx, valid_idx, = indices[0:-validation_split], indices[-validation_split:-1]
            train_sampler = sampler.SubsetRandomSampler(train_idx)
            test_sampler = sampler.SubsetRandomSampler(valid_idx)
            test_loader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, **kwargs)

    else:
        raise NotImplementedError

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                              **kwargs)
    full_loader = DataLoader(train_set, batch_size=num_train, sampler=train_sampler,
                             **kwargs)

    return train_loader, test_loader, full_loader


def GetDataForModel(model, dataset, _seed, validate=False):
    batch_size = int(1.05 * model.max_neurons)
    train_loader, test_loader, full_loader = get_data(dataset, batch_size, _seed, validate)
    if isinstance(model, EigenvalueRegularization):
        model.eig_loader = full_loader
    return train_loader, test_loader