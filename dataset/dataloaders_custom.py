import os
import torch
import numpy as np

from conf import settings
# import transforms 
import torchvision.transforms as transforms
from utils import *

def get_custom_train_loader(args, data_path, path_txt,batch_size, num_classes, workers=5, one_hot=False,  _worker_init_fn=None):
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.ToCVImage(),
        # transforms.Resize(int(args.image_size / 0.875)),
        # transforms.Resize(args.image_size),        
        transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.3, hue=0.05),
        # transforms.RandomErasing(),
        #transforms.CutOut(56),
        transforms.ToTensor(),
        transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)])
    train_dataloader = get_train_dataloader(
        data_path,
        path_txt,
        train_transforms,
        batch_size,
        workers)
    return train_dataloader
def get_custom_val_loader(args, data_path, path_txt,batch_size, num_classes, workers=5, one_hot=False, _worker_init_fn=None):
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.ToCVImage(),
        # transforms.Resize(int(args.image_size / 0.875)),
        transforms.Resize((args.image_size, args.image_size)),
        # transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)])
    test_dataloader = get_test_dataloader(
        data_path,
        path_txt,
        test_transforms,
        batch_size,
        workers)
    return test_dataloader
