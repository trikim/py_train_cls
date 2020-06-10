

import os

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.autograd import Variable
from conf import settings
from dataset.dataset import CUB_200_2011_Train, CUB_200_2011_Test
from models.utils import load_state_dict_from_url

def get_network(args):
   
    if args.net == 'resnet18':
        from models.resnet import resnet18
        model_ft = resnet18(pretrained=True)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        model_ft = resnet34(pretrained=True)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        model_ft = resnet101(pretrained=True)
    elif args.net == 'resnext101':
        from models.resnet import resnext101_32x8d
        model_ft = resnext101_32x8d(pretrained=True)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        model_ft = resnet50(pretrained=True)
    elif args.net == 'resnet50_onnx':
        from models.resnet_onnx import resnet50
        model_ft = resnet50(pretrained=True)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        model_ft = resnet152(pretrained=True)
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet_v2
        model_ft = mobilenet_v2(pretrained=True)
    elif args.net == 'mobilenet_onnx':
        from models.mobilenet_onnx import mobilenet_v2
        model_ft = mobilenet_v2(pretrained=True)
    elif args.net == 'mobilenetv3':
        from models.mobilenet_v3 import mobilenetv3_small
        model_ft = mobilenetv3_small()
        # pretrained_url='http://10.9.0.146:8888/group1/M00/00/51/CgkAkl7d9P-Ea4J1AAAAANzZ--g202.pth'
        # model_ft.load_state_dict(load_state_dict_from_url(pretrained_url))
    elif args.net == 'mobilenetv3_onnx':
        from models.mobilenet_v3_onnx import mobilenetv3_small
        model_ft = mobilenetv3_small()
    else:
        print("The %s is not supported..." %(args.net))
        return
    if not args.net.startswith('mobilenet'):
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.num_classes)
    elif args.net.startswith('mobilenetv3'):
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[3] = nn.Linear(num_ftrs, args.num_classes)
        model_ft.classifier[4] = nn.BatchNorm1d(args.num_classes)
    else:
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, args.num_classes)
        
    net = model_ft

    return net

'''
def get_train_dataloader(path,path_txt, transforms, batch_size, num_workers, target_transforms=None):
    """ return training dataloader
    Args:
        path: path to CUB_200_2011 dataset
        transforms: transforms of dataset
        target_transforms: transforms for targets
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
    Returns: train_data_loader:torch dataloader object
    """
    train_dataset = CUB_200_2011_Train(
        path, 
        path_txt,
        transform=transforms,
        target_transform=target_transforms
    )
    train_dataloader =  DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_dataloader
'''

def get_train_dataloader(path,path_txt, transforms, batch_size, num_workers, target_transforms=None):
    """ return training dataloader
    Args:
        path: path to CUB_200_2011 dataset
        transforms: transforms of dataset
        target_transforms: transforms for targets
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
    Returns: train_data_loader:torch dataloader object
    """


    train_dataset = CUB_200_2011_Train(
        path, 
        path_txt,
        transform=transforms,
        target_transform=target_transforms
    )
    ## get sampler weight
    class_ids = list(train_dataset.class_ids.values())
    target = np.array(class_ids, dtype=np.int32)
    class_sample_count = np.array(
    [len(np.where(target == t)[0]) for t in np.unique(target)])

    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()

    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True) 

    train_dataloader =  DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        #shuffle=True,
    )

    return train_dataloader

def get_test_dataloader(path, path_txt, transforms, batch_size, num_workers, target_transforms=None):
    """ return training dataloader
    Args:
        path: path to CUB_200_2011 dataset
        transforms: transforms of dataset
        target_transforms: transforms for targets
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
    Returns: train_data_loader:torch dataloader object
    """
    test_dataset = CUB_200_2011_Test(
        path, 
        path_txt,
        transform=transforms,
        target_transform=target_transforms
    )

    test_dataloader =  DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    return test_dataloader

def init_weights(net):
    """the weights of conv layer and fully connected layers 
    are both initilized with Xavier algorithm, In particular,
    we set the parameters to random values uniformly drawn from [-a, a]
    where a = sqrt(6 * (din + dout)), for batch normalization 
    layers, y=1, b=0, all bias initialized to 0.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return net

def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias, 
    bn weights, bn bias, linear bias)

    Args:
        net: network architecture
    
    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)
        
        else: 
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)
        
    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]

def mixup_data(x, y, alpha=0.2):

    """Returns mixed up inputs pairs of targets and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    index = index.to(x.device)

    lam = max(lam, 1 - lam)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a = y
    y_b = y[index, :]

    return mixed_x, y_a, y_b, lam



    
