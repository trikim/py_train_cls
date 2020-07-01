
"""author 
   baiyu
"""

import argparse
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

#from PIL import Image
import transforms 
#from torchvision import transforms
from tensorboardX import SummaryWriter
from conf import settings
from utils import *
from lr_scheduler import WarmUpLR
from criterion import LSR
import time
from apex import amp
import apex
import cv2

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=5, help='warm up phase')
    parser.add_argument('-gpus', nargs='+', type=int, default=0, help='gpu device')
    parser.add_argument('-num_classes', type=int, default=2, help='num_classes')
    parser.add_argument('-refine', type=str, default=None)
    #parser.add_argument('--loss-scale', type=str, default=None)
    args = parser.parse_args()

    #checkpoint directory




    #get dataloader

    test_transforms = transforms.Compose([
        transforms.ToCVImage(),
        transforms.CenterCrop(settings.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(settings.TRAIN_MEAN, settings.TRAIN_STD)
    ])


    test_dataloader = get_test_dataloader(
        settings.DATA_PATH,
        test_transforms,
        args.b,
        args.w
    )

    net = get_network(args)

    checkpoint = torch.load(args.refine)
    print(args.refine)
    net.load_state_dict(checkpoint['model'])
    #print("net:",net)

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    
    net = net.cuda()

    #visualize the network
    #visualize_network(writer, net.module)

    #cross_entropy = nn.CrossEntropyLoss() 
    lsr_loss = LSR()


    net.eval()
    best_acc = 0.0
    total_loss = 0
    correct = 0
    for images, labels, path in test_dataloader:
        #print("path:",settings.DATA_PATH+path[0])

        images = images.cuda()
        labels = labels.cuda()

        predicts = net(images)
        preds=F.softmax(predicts)
        #_, preds = predicts.max(1)
        print("preds:",preds)
        if os.path.exists:
            image = cv2.imread(settings.DATA_PATH+'/'+path[0])
            cv2.imshow("test",image)
            cv2.waitKey(0)       
        correct += preds.eq(labels).sum().float()

        loss = lsr_loss(predicts, labels)
        total_loss += loss.item()

    test_loss = total_loss / len(test_dataloader)
    acc = correct / len(test_dataloader.dataset)
    print('Test set: loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, acc))
    print()










    


    

