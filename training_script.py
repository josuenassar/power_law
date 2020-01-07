import torch
from tqdm import tqdm
import ModelDefs.models as models
from torchvision import datasets, transforms
import fire


def train_bad_boys(lr=1e-3, alpha_eig=0, alpha_jacob=0, eps=0.5, seed=0, cuda=False,
                   save_dir='../data/', nonlin='tanh', max_epochs=1000, weight_decay=0,
                   dims=[1, (1, 28), (4032, 128), (128, 10)], arch='cnn'):
    torch.manual_seed(seed)  # seed random number generator

    # In[]
    "Regular or adversarial training"
    if eps > 0:
        trainer = 'adv'
    else:
        trainer = 'vanilla'

    # In[]
    "What type of regularizer"
    regul = ''
    if alpha_eig == 0 and alpha_jacob == 0:
        regul = 'no'
    else:
        if alpha_eig > 0:
            regul += 'eig'
        if alpha_jacob > 0:
            regul += 'jac'

    # In[]
    "Create model"
    model_kwargs = {"dims": dims, "activation": nonlin, "architecture": arch,
              "trainer": trainer, "regularizer": "no", 'alpha_spectra': alpha_eig, 'alpha_jacob': alpha_jacob,
              'optimizer': 'adam', 'lr': lr, 'weight_decay': weight_decay, 'cuda': cuda,
              'eps': eps,  'training_type': 'FGSM', 'lr_pgd': 0, 'gradSteps': 0, 'noRestarts': 0}
    model = models.ModelFactory(**model_kwargs)

    # In[]
    "Set up data loaders"
    kwargs = {'num_workers': 4, 'pin_memory': True}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(root='../', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='../', train=False, download=True, transform=transform)
    num_train = len(train_set)
    indices = list(range(num_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    batch_size = int(1.05 * model.max_neurons)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
    full_loader = torch.utils.data.DataLoader(train_set, batch_size=60_000, sampler=train_sampler,
                                              **kwargs)
    full_X, full_Y = next(iter(full_loader))

    # In[]
    "Train"
    device = 'cpu'
    if cuda:
        device = 'cuda'

    model.train()
    for epoch in tqdm(range(max_epochs)):
        if alpha_eig == 0:
            model.train_epoch(train_loader)
        else:
            model.train_epoch(train_loader, full_X.to(device), full_Y.to(device))

    # In[]
    "Compute test loss"
    test_loss, test_error = model.evaluate_dataset_test_loss(test_loader)
    test_acc = 1 - test_error.to('cpu').item()

    # In[]
    "Save file"
    model_data = {'params': model.cpu().state_dict(),
                  'kwargs': model_kwargs,
                  'test_loss': test_acc}
    save_file_name = save_dir + '_' + nonlin + '_' + 'seed=' + str(seed) + '_' + trainer + '_' + 'alpha_eig' \
                     + '_' + str(alpha_eig) + '_' + 'alpha_jacob' + '_' + str(alpha_jacob)
    torch.save(model_data, save_file_name)

if __name__ == '__main__':
    fire.Fire(train_bad_boys)
