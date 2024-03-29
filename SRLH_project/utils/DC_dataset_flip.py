from __future__ import print_function
from PIL import Image
import os
import os.path
import torch
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F

class DC_CIFAR10(Dataset):
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

    def __init__(self, root, train=True, transform=None, target_transform=None,):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class CIFAR_10(Dataset):
    data_path = '/home/dacheng/PycharmProjects/ADSH_pytorch/data/CIFAR-10/'
    def __init__(self, img_filename, label_filename, transform=None):
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(self.data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(self.data_path, label_filename)
        fp_label = open(label_filepath, 'r')
        labels = [int(x.strip()) for x in fp_label]
        fp_label.close()
        self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.img_filename[index]))
        img = img.convert('RGB')
        img_flip = F.hflip(img)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([self.label[index]])
        if self.transform is not None:
            img_flip = self.transform(img_flip)
        return img, img_flip, label, index

    def __len__(self):
        return len(self.img_filename)

class MirFlickr(Dataset):
    data_path = '/data/dacheng/Datasets/MirFlickr/'
    def __init__(self, img_filename, label_filename, transform=None):
        self.img_filename = img_filename
        self.label_filename = label_filename
        self.transform = transform
        img_filepath = os.path.join(self.data_path, self.img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(self.data_path, self.label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.img_filename[index]))
        img = img.convert('RGB')
        img_flip = F.hflip(img)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        if self.transform is not None:
            img_flip = self.transform(img_flip)
        return img, img_flip, label, index

    def __len__(self):
        return len(self.img_filename)

class NUSWIDE(Dataset):
    data_path = '/data/dacheng/Datasets/NUSWIDE/'
    def __init__(self, img_filename, label_filename, transform=None):
        self.img_filename = img_filename
        self.label_filename = label_filename
        self.transform = transform
        img_filepath = os.path.join(self.data_path, self.img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(self.data_path, self.label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


class COCO(Dataset):
    data_path = '/data/dacheng/Datasets/COCO/'
    def __init__(self, img_filename, label_filename, transform=None):
        self.img_filename = img_filename
        self.label_filename = label_filename
        self.transform = transform
        img_filepath = os.path.join(self.data_path, self.img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(self.data_path, self.label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.img_filename[index]))
        img = img.convert('RGB')
        img_flip = F.hflip(img)
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        if self.transform is not None:
            img_flip = self.transform(img_flip)
        return img, img_flip, label, index
    def __len__(self):
        return len(self.img_filename)

class MNIST(Dataset):
    root = '/home/dacheng/PycharmProjects/ADSH_pytorch/data/'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, train=True, transform=None):
        self.root = os.path.expanduser(self.root)
        self.transform = transform
        self.train = train  # training set or test set


        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        img_flip = F.hflip (img)
        if self.transform is not None:
            img = self.transform (img)
        if self.transform is not None:
            img_flip = self.transform (img_flip)
        return img, img_flip, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

