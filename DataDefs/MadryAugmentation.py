from PIL import Image
import os
import os.path
import unittest
import numpy as np
import pickle
import copy
import random

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10 as baseCIFAR


class CIFAR10Augmented(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, grow_data_by:int = 2):

        super(CIFAR10Augmented, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.grow_data_by = grow_data_by

        random.seed(42)     # seeding the RNG for reproducibility

        if download:
            self.download()
            if self.train:
                self.augment()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = [['augmented_batches', '']]
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:  # TODO: change s/t it's not downloaded_list but some other var
            # containing the ref to the name of the new file
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def augment(self):
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        orig_data = []
        orig_targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                orig_data.append(entry['data'])
                if 'labels' in entry:
                    orig_targets.extend(entry['labels'])
                else:
                    orig_targets.extend(entry['fine_labels'])

        orig_data = np.vstack(orig_data).reshape(-1, 3, 32, 32)
        orig_data = orig_data.transpose((0, 2, 3, 1))  # convert to HWC
        data = copy.deepcopy(orig_data)
        data = np.tile(data, (self.grow_data_by, 1, 1, 1))
        targets = self.grow_data_by * orig_targets
        for index in range(len(orig_data)):
            img = Image.fromarray(data[index])
            img = self._Madry_augment(img)
            data[index] = np.array(img)

        new_filename = os.path.join(self.root, self.base_folder,'augmented_batches')
        l = 'labels' if 'labels' in entry else 'fine_labels'
        dataset = {l : targets, 'data' : data}
        with open(new_filename, 'wb') as output:
            pickle.dump(dataset, output)

    @staticmethod
    def _Madry_augment(img):
        rc = transforms.RandomCrop((32,32), padding=(4,4), pad_if_needed=True, fill=0, padding_mode='constant')
        rf = transforms.RandomHorizontalFlip(p=0.5)
        T = transforms.Compose([rc, rf])
        return T(img)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):  # TODO: change this so that it's just os.path.exists ...
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100Augmented(CIFAR10Augmented):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class TestModel(unittest.TestCase):
    def test_download_and_augment(self):
        cls = CIFAR10Augmented(root=os.getcwd(), download=True)
        base = baseCIFAR(root=os.getcwd(),download=True)

        self.assertEqual(len(cls), 50_000*cls.grow_data_by)


if __name__ == '__main__':
    unittest.main()
