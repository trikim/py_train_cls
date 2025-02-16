
import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from conf import settings

class CUB_200_2011_Train(Dataset):

    def __init__(self, path, path_txt, transform=None, target_transform=None):

        self.root = path
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = {}
        self.num_train = 0
        self.train_id = []
        self.class_ids = {}
        with open(os.path.join(self.root, path_txt)) as f:
            for line in f:
                self.num_train = self.num_train + 1
                #print("line:",line.strip().split(settings.DATA_SPLIT))

                path, class_id = line.strip().split(settings.DATA_SPLIT)
                self.images_path[self.num_train] = path
                self.class_ids[self.num_train] = class_id
                self.train_id.append(self.num_train)

    def __len__(self):
        return len(self.train_id)
    
    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_id = self.train_id[index]
        class_id = int(self._get_class_by_id(image_id))# - 1
        path = self._get_path_by_id(image_id)
        image = cv2.imread(os.path.join(self.root, path))
        #name = path.rsplit('/')[-1]
        #cv2.imwrite("./"+name,image)
        #image = Image.open(os.path.join(self.root, 'images', path))
        #if image.mode != 'RGB':
        #    image = image.convert('RGB')
        #if len(image.shape) != 3:
        #    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        #image = np.array(image)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            class_id = self.target_transform(class_id)
        return image, class_id

    def _get_path_by_id(self, image_id):

        return self.images_path[image_id]
    
    def _get_class_by_id(self, image_id):

        return self.class_ids[image_id]


class CUB_200_2011_Test(Dataset):

    def __init__(self, path, path_txt, transform=None, target_transform=None):

        self.root = path
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = {}
        self.num_train = 0
        self.class_ids = {}
        self.train_id = []
        with open(os.path.join(self.root, path_txt)) as f:
            for line in f:
                self.num_train = self.num_train + 1
                path, class_id = line.strip().split(settings.DATA_SPLIT)
                self.images_path[self.num_train] = path
                self.class_ids[self.num_train] = class_id
                self.train_id.append(self.num_train)

    def __len__(self):
        return len(self.train_id)
    
    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_id = self.train_id[index]
        class_id = int(self._get_class_by_id(image_id))# - 1
        path = self._get_path_by_id(image_id)
        image = cv2.imread(os.path.join(self.root, path))
        #image = Image.open(os.path.join(self.root, 'images', path))
        #print("path:",path)
        #if image.mode != 'RGB':
        #    image = image.convert('RGB')
        #image = np.array(image)


        #if len(image.shape) != 3:
        #    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            class_id = self.target_transform(class_id)

        return image, class_id#,path

    def _get_path_by_id(self, image_id):

        return self.images_path[image_id]
    
    def _get_class_by_id(self, image_id):

        return self.class_ids[image_id]


def compute_mean_and_std(dataset):
    """Compute dataset mean and std, and normalize it

    Args:
        dataset: instance of CUB_200_2011_Train, CUB_200_2011_Test
    
    Returns:
        return: mean and std of this dataset
    """

    mean_r = 0
    mean_g = 0
    mean_b = 0

    for img, _ in dataset:
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(dataset)
    mean_g /= len(dataset)
    mean_r /= len(dataset)

    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0

    for img, _ in dataset:

        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

        N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)

    mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
    std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
    return mean, std
