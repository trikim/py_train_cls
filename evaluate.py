
"""author 
   lhy
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
from dataset.dataloaders_custom import * 
#from torchvision import transforms
from tensorboardX import SummaryWriter
from conf import settings
from utils import *
from lr_scheduler import WarmUpLR
from criterion import LSR
import time
import cv2

from sklearn.metrics import roc_curve
from sklearn.metrics import auc,average_precision_score,precision_score,recall_score,f1_score,accuracy_score

import matplotlib.pyplot as plt
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-batch', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-gpus', default=0, type=str, help='gpu device')
    parser.add_argument('-num_classes', type=int, default=2, help='num_classes')
    parser.add_argument('-load_model', type=str, required=True,default=None)
    parser.add_argument('-export_onnx',action='store_true',default=True, help='onnx.')
    parser.add_argument('-imgs_root',type=str,required=True, default=None)
    parser.add_argument('-test_label_txt',type=str,default=None)
    parser.add_argument('-image-size', type=int, default=224, help='training image ')
    parser.add_argument('-thre', type=float, default=0.5, help='F1-score,precision,recall')
    args = parser.parse_args()


    args.gpus_str = args.gpus
    args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
    args.gpus = [i for i in range(len(args.gpus))] if args.gpus[0] >=0 else [-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    args.device = torch.device('cuda' if args.gpus[0] >= 0 else 'cpu')

    test_dataloader = get_custom_val_loader(args, args.imgs_root, args.test_label_txt, 1, args.num_classes, 1)
    iter_per_test = len(test_dataloader)


    net = get_network(args)

    checkpoint = torch.load(args.load_model)
    net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['model'].items()})
    print('Loading model from path {}...'.format(args.load_model))

    
    #net = net.cuda()
    net = net.to(args.device)

    net.eval()
    best_acc = 0.0
    total_loss = 0
    correct = 0
    result = []
    result_label = []
    label = []
    f = open("checkpoints/"+args.net+"/roc.txt","w")    
    for images, labels in test_dataloader:

        images = images.to(args.device)
        labels = labels.to(args.device)
        label.append(labels.data.cpu().numpy()[0])
        preds = net(images)
        preds_cpu = preds.data.cpu().numpy()[0]
        #print("preds_cpu:",preds_cpu)
        n_conf = preds_cpu[0]
        p_conf = preds_cpu[1]
        if p_conf > n_conf and p_conf > args.thre:
            result_label.append(1)
        else:
            result_label.append(0)
        conf, preds_label = preds.max(1)
        result.append(preds[:, 1:].data.cpu().numpy()[0])
        #correct += preds.eq(labels).sum().float()
    baseline_fpr, baseline_tpr, threshold = roc_curve(label, result,pos_label=1)
    f.write('fpr '+'tpr '+ 'thre' + '\n')
    for fpr,tpr,thre in zip(baseline_fpr,baseline_tpr,threshold):
        #print(fpr,tpr,thre)
        f.write(str(round(fpr, 5))+' '+ str(round(tpr, 5)) +' ' + str(round(thre, 5)) +'\n')


    acc = accuracy_score(label, result_label)
    precision = precision_score(label, result_label)
    recall = recall_score(label, result_label)
    f1 = f1_score(label, result_label)
    print("precision,recall,f1:",precision,recall,f1)

    roc_auc = auc( baseline_fpr,  baseline_tpr)

    print('Test Accuracy: {:.4f}'.format(acc))
    print("Test Auc:",roc_auc)
    plt.plot(baseline_fpr, baseline_tpr, lw=1, label='ROC fold (area = %0.2f)' % (roc_auc))

    
    #plt.savefig("./roc.jpg")    
    #plt.show()

    f.close()



    


    

