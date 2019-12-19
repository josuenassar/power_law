import numpy as np
import numpy.random as npr
import torch


def get_data(dataset, cuda, batch_size,_seed, h0, data_dir):
    from ops.transformsParams import CIFAR10, SVHN
    from torchvision import datasets, transforms
    from ops.utils import SubsetSequentialSampler

    kwargs = {'num_workers': 4,op 'pin_memory': True} if cuda else {}

    if dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset == 'CIFAR10':
        transform_train, transform_eval = CIFAR10(h0)
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        stats_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_eval)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_eval)
    elif dataset == 'SVHN':
        transform_train, transform_eval = SVHN(h0)
        train_set = datasets.SVHN(root=data_dir, split='train', download=True,
                                  transform=transform_train)
        test_set = datasets.SVHN(root=data_dir, split='test', download=True,
                                 transform=transform_eval)
        stats_set = datasets.SVHN(root=data_dir, split='train', download=True,
                                 transform=transform_eval)

    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.seed(_seed)
    np.random.shuffle(indices)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    stats_idx = np.array(indices)[np.random.choice(len(indices), 30)]
    stats_idx = stats_idx.tolist()

    stats_sampler = SubsetSequentialSampler(stats_idx)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
    stats_loader = torch.utils.data.DataLoader(stats_set, batch_size=30, sampler=stats_sampler)
    return train_loader, test_loader, stats_loader


def eigen_val_regulate(x, v, eigT=None, start=10, cuda=False):
    """
    Function that approximates the eigenvalues of the matrix x, by finding them wrt some pseudo eigenvectors v and then
    penalizes eigenvalues that stray too far away from a power law
    :param x: hidden representations, N by D
    :param v: eigenvectors, D by D (each column is an eigenvector!)
    :param eigT: if the eigenspectra is already estimated, can just be passed in, else it is default as None
    :param start: index that states what eigenvalues to start regulating.
    :return: value of regularizer and the estimated spectra
    """
    device = 'cuda' if cuda == True else 'cpu'
    if eigT is None:
        xt = x - torch.mean(x, 0)  # demean the data
        cov = xt.transpose(1, 0) @ xt / (x.shape[0] - 1) # compute covariance matrix
        cov = (cov + cov.transpose(1, 0)) / 2
        eig = torch.diag(v.transpose(1, 0) @ cov @ v).to(device)

    else:
        eig = eigT

    eigs = torch.sort(eig, descending=True)[0]
    regul = torch.zeros(1, device=device)
    # slope = -1
    with torch.no_grad():
        alpha = eigs[start] * (start + 1)  # let the the constant be the largest eigenvalue

    for n in range(start + 1, eigs.shape[0]):
        if eigs[n] > 0:  # don't use negative eigenvalues
            regul += (eigs[n] / (alpha / (n + 1)) - 1) ** 2 + torch.relu(eigs[n] / (alpha / (n + 1)) - 1)
    return eigs, regul / x.shape[0]


def compute_eig_vectors(model, data, labels, lossFunction, alpha=0, cuda=False):
    device = 'cuda' if cuda == True else 'cpu'
    hidden, outputs = model.bothOutputs(data.to(device))
    loss = lossFunction(outputs, labels.to(device))

    spectraTemp = []
    eigVec = []
    regul = torch.zeros(1, device=device)

    if alpha > 0:
        for idx in range(len(hidden)):
            hTemp = hidden[idx] - torch.mean(hidden[idx], 0)
            cov = hTemp.transpose(1, 0) @ hTemp / hTemp.shape[0]  # compute covariance matrix
            cov = cov.double()  # cast as 64bit to mitigate underflow in matrix factorization
            cov = (cov + cov.transpose(1, 0)) / 2
            _, eigTemp, vecTemp = torch.svd(cov, compute_uv=True)  # compute eigenvectors and values
            eigTemp = eigTemp.float()
            vecTemp = vecTemp.float()
            eigTemp, rT = eigen_val_regulate(0, 0, eigT=eigTemp, cuda=cuda)  # compute regularizer
            regul += rT
            spectraTemp.append(eigTemp.cpu())  # save spectra
            eigVec.append(vecTemp)

    return eigVec, loss, spectraTemp, regul.cpu().item()


def get_jacobian(net, x, target, loss, cuda=False):
    """
    Function for computing the input-output jacobian of a neural network
    :param net: neural network
    :param x: input image that has beeen compressed into an array
    :param target: target label
    :param loss: corresponding loss function
    :param cuda: boolean variable deciding whether to do stuff on the gpu
    :return: jacobian loaded back on the cpu
    """
    device = 'cuda' if cuda == True else 'cpu'
    x.requires_grad_(True)
    y = net(x.to(device))
    ell = loss(y, target.to(device))
    ell.backward()
    return x.grad.data.squeeze(), ell.item()


def generate_adv_images(model, images, targets, lossFunc, eps=50, cuda=False, rand=False):
    """
    Given images, targets and a model will create adversarial images
    :param model: neural network class
    :param images: input images that are to be corrupted
    :param targets: target labels for the images
    :param lossFunc: corresponding loss function
    :param eps: parameter for bad images
    :param cuda: boolean variable for using GPU
    :return:
    """
    jacob, _ = get_jacobian(model, images, targets, lossFunc, cuda=cuda)
    if rand:
        advImages = images + eps * torch.rand(images.shape[0], 1) * torch.sign(jacob)
    else:
        advImages = images + eps * torch.sign(jacob)
    return advImages


# TODO: Use faster jacobian regularization from Hoffman et al
def compute_loss(model, imgs, labels, loss, eigen_vecs=None, alpha_spectra=0, alpha_adv=False, alpha_jacob=0, cuda=False,
                 adversary=None, num_classes=10):
    """
    Function that will compute the loss function PLUS any combination of the following three regularizers:
    1) Eigenvalue regularization, 2) Adversarial training, 3) Jacobian regularization
    :param model: model object
    :param imgs: batch of images
    :param labels: corresponding image labels
    :param loss: loss function being used
    :param eigen_vecs: eigenvectors used for computing the spectra, default is None.
    :param alpha_spectra: weight of the regularizer term on the spectra. If it is 0, don't compute
    :param alpha_adv: boolean flag that will determine whether we should create a batch of images
    :param alpha_jacob: weight of the regularizer term for the jacobian. If it is 0, don't compute.
    :param cuda: Boolean flag that will determine whether we should use the GPU or not. It is assumed that the model is
                 already on the chosen device, so data will be loaded on the specified device
    :param adversary: An object that is in charge of computing the adversarial examples
    :param num_classes: Number of possible classes
    :return: the value of the loss function and the value of the spectra and jacobian regularizer
    """
    device = 'cuda' if cuda else 'cpu'
    if alpha_adv:
        "If true, create adversarial images from batch"
        adv_imgs = adversary.generate_adv_images(model.to(device), imgs.to(device), labels.to(device), loss.to(device))
        imgs = torch.cat((imgs, adv_imgs), 0)  # concatenate images together

    "Compute a forward pass through the network and compute the loss"
    hidden, outputs = model.bothOutputs(imgs.to(device))  # feed data forward
    loss = loss(outputs, labels.to(device))  # compute loss

    "Compute spectra regularizer"
    spectra_regul = torch.zeros(1, device=device)
    spectra_temp = []
    if alpha_spectra > 0:
        for idx in range(len(hidden)):
            spectra, rTemp = eigen_val_regulate(hidden[idx],
                                                eigen_vecs, cuda=cuda)  # compute spectra for each hidden layer
            with torch.no_grad():
                spectra_temp.append(spectra)
            spectra_regul += rTemp

    "Compute jacobian regularizer"
    jacob_regul = torch.zeros(1, device=device)
    if alpha_jacob > 0:
        temp = torch.ones(loss.shape[0], loss.shape[1], device=device)
        for k in range(num_classes):
            jacobian = get_jacobian(model, imgs, k * temp, loss, cuda)  # get input-output jacobian for class k
            jacob_regul += torch.sum(jacobian ** 2)  # take frobenius norm of input-output jacobian

    total_loss = loss + alpha_spectra * spectra_regul + alpha_jacob * jacob_regul / imgs.shape[0]
    return total_loss, loss, spectra_regul, spectra, jacob_regul


def split_mnist(pathLoad='../simple_regularization/data/mnist.npy', fracVal=0.2):
    """
    Generate training, test and validation set for MNIST
    :param pathLoad: a string indicating where MNIST lives
    :param fracVal: fraction of training set to use for validation
    :return: 3 tuples, (trainData, trainLabels), (valData, valLabels), (testData, testLabels)
    """
    assert fracVal >= 0 and fracVal<=1
    mnist_data = np.load(pathLoad, allow_pickle=True)[()]
    trainData, trainLabels = mnist_data['train']  # training set
    testData, testLabels = mnist_data['test']  # test set
    numImgs = trainData.shape[0]
    fracTrain = 1 - fracVal
    if fracVal >= 0:
        idx = npr.permutation(numImgs)
        trainIdx = idx[:int(fracTrain * numImgs)]
        valIdx = idx[int(fracTrain * numImgs):]

        valData = trainData[valIdx, :]
        valLabels = trainLabels[valIdx]

        trainData = trainData[trainIdx, :]
        trainLabels = trainLabels[trainIdx]
    return (trainData, trainLabels), (valData, valLabels), (testData, testLabels)


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
