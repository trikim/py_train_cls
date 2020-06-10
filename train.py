"""author 
   baiyu
"""

import argparse
import os
import glob
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from logger import Logger

from torchvision import transforms
from dataset.dataloaders_custom import *
from conf import settings
from utils import *
from lr_scheduler import WarmUpLR
from lr_scheduler.mixup import NLLMultiLabelSmooth, MixUpWrapper, mixup, rand_bbox
from criterion import LSR, CELoss, FocalLoss
import time
import json
import datetime
from models.utils import load_state_dict_from_url

# import apex

# try:
#     from apex.parallel import DistributedDataParallel as DDP
#     from apex.fp16_utils import *
#     from apex import amp, optimizers
#     from apex.multi_tensor_apply import multi_tensor_applier
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('-batch', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.04, help='initial learning rate')
    parser.add_argument('-lr_step', type=str, default='30,50,60',help='drop learning rate by 10.')
    parser.add_argument('-epoch', type=int, default=72, help='training epoches')
    parser.add_argument('-save_epoch', type=int, default=20, help='save epoches')
    parser.add_argument('-data-backend', metavar='BACKEND', default='custom')#,choices=DATA_BACKEND_CHOICES)
    parser.add_argument('-image-size', type=int, default=224, help='training image ')
    parser.add_argument('-warm', type=int, default=5, help='warm up phase')
    parser.add_argument('-mixup', default=0.0, type=float,metavar='ALPHA', help='mixup alpha')
    parser.add_argument('-cutmix_prob', default=0.0, type=float, help='cutmix probability')
    parser.add_argument('-gpus', default=0, help='gpu device')
    parser.add_argument('-num_classes', type=int, default=7, help='num_classes')
    parser.add_argument('-pretrained',action='store_true',default=False, help='pretrained.')
    parser.add_argument('-resume', action='store_true',default=False, help='resume.')
    parser.add_argument('-refine',type=str,default=None)
    parser.add_argument('-start_epoch',type=int,default=1)
    parser.add_argument('-steps_per_log', type=int, default=100)
    parser.add_argument('-apex', action='store_true',default=False, help='apex mixed precision')
    parser.add_argument('-opt_level', type=str,default="O1")
    parser.add_argument('-keep_batchnorm_fp32', type=str, default=None)
    parser.add_argument('-sync_bn', action='store_true',default=False, help='enabling apex sync BN.')
    parser.add_argument('-imgs_root',type=str, default='/home/data/14')
    parser.add_argument('-label_txt',type=str, default=None)
    parser.add_argument('-test',action='store_true',default=False, help='test.')
    parser.add_argument('-test_label_txt',type=str,default=None)
    parser.add_argument('-steps_per_test', type=int, default=1)
    parser.add_argument('-export_onnx', action='store_true', default=False, help='export onnx model')
    args = parser.parse_args()

    args.gpus_str = args.gpus
    args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
    args.gpus = [i for i in range(len(args.gpus))] if args.gpus[0] >=0 else [-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    args.device = torch.device('cuda' if args.gpus[0] >= 0 else 'cpu')
    cc = torch.cuda.get_device_capability(args.device)
    data = {"cc:":str(cc[0])+str(cc[1])}
    args.lr_step = [int(i) for i in args.lr_step.split(',')]
    #checkpoint directory
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net)
    args.save_dir = checkpoint_path
    logger = Logger(args)
    with open(checkpoint_path+"/cc.json",'w',encoding='utf-8') as json_file:
        json.dump(data,json_file,ensure_ascii=False)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    #tensorboard log directory
    log_path = os.path.join(settings.LOG_DIR, args.net)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # from dataset.dataloaders import *
    # compute class number and weights
    '''
    cls_num = np.zeros(args.num_classes)
    cls_names = ['ea_style', 'ethnic_style', 'jp_style', 'ladylike', 'leisure', 'maid_style', 'punk']
    for folder in os.listdir(args.imgs_root):
        cls_id = cls_names.index(folder)
        cls_num[cls_id] = len(glob.glob(os.path.join(args.imgs_root, folder) + '/*.jpg'))
    cls_rate = cls_num / np.sum(cls_num)
    print('cls_rate:{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(cls_rate[0], cls_rate[1], cls_rate[2], cls_rate[3], cls_rate[4], cls_rate[5], cls_rate[6]))
    cls_weights = torch.zeros(args.num_classes).to(args.device)
    for i in range(args.num_classes):
        cls_weights[i] = math.exp(1 - cls_rate[i])
    print('cls_weights:{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(cls_weights[0], cls_weights[1], cls_weights[2], cls_weights[3], cls_weights[4], cls_weights[5], cls_weights[6]))
    ''' 

    if args.data_backend == "custom":
        train_dataloader = get_custom_train_loader(args, args.imgs_root, args.label_txt, args.batch, args.num_classes, args.workers)
        iter_per_epoch = len(train_dataloader)
        if args.test:
            test_dataloader = get_custom_val_loader(args, args.imgs_root, args.test_label_txt, 1, args.num_classes, 1)
            iter_per_test = len(test_dataloader)
    
    net = get_network(args)
    if not args.pretrained:
        net = init_weights(net)
    net = net.to(args.device)
  
    criterion = LSR(e=0.1, reduction='mean')
    # alpha = np.load('../data/alpha.npy')
    # criterion = CELoss(args.num_classes, alpha=alpha, use_alpha=True)
    # alpha = np.load('../data/alpha.npy')
    # criterion = FocalLoss(args.num_classes, alpha=alpha, use_alpha=True)

    params = split_weights(net) #apply no weight decay on bias
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    if args.apex:
        net, optimizer = amp.initialize(net, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32)
    if args.resume and args.refine:
        # pretrained_mobilenet_url = 'http://10.9.0.146:8888/group1/M00/00/3E/CgkA617Z35iEIzogAAAAAEIdp8Q456.pth'
        # checkpoint = load_state_dict_from_url(pretrained_mobilenet_url, map_location='cuda:0')        
        # model_state = net.state_dict()
        # del checkpoint['model']['classifier.1.weight']
        # del checkpoint['model']['classifier.1.bias']
        # model_state.update({k.replace('module.',''):v for k,v in checkpoint['model'].items()})
        # net.load_state_dict(model_state)
        checkpoint = torch.load(args.refine)
        net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['model'].items()})
        # args.start_epoch = checkpoint['epoch']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # amp.load_state_dict(checkpoint['amp'])

    net = nn.DataParallel(net, device_ids=args.gpus).to(args.device)
    if args.apex and args.sync_bn:
        print("using apex synced BN")
        net = apex.parallel.convert_syncbn_model(net)

    #set up warmup phase learning rate scheduler    
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    #set up training phase learning rate scheduler
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch - args.warm)

    best_acc = 0.0
    for epoch in range(args.start_epoch, args.epoch + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)        
        #training procedure
        net.train()
        if epoch == int(0.5*args.epoch):
            # criterion = LSR(e=0., reduction='mean', weights=cls_weights)
            criterion = LSR(e=0., reduction='mean')

        for batch_index, (images, labels) in enumerate(train_dataloader):
            if epoch <= args.warm:
                warmup_scheduler.step()
            images = images.to(args.device)
            labels = labels.to(args.device)
            if args.mixup != 0.0:
                images, labels = mixup(args.mixup, args.num_classes, images, labels)
            time_start = time.time()
            optimizer.zero_grad()

            # cutmix
            r = np.random.rand(1)
            if r < args.cutmix_prob and epoch < int(0.75*args.epoch):
                # generate mixed sample
                beta = 1.0
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # compute output
                output = net(images)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            else:
                predicts = net(images)
                loss = criterion(predicts, labels)
            if args.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss: 
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            time_end = time.time()
            n_iter = (epoch - 1) * iter_per_epoch + batch_index + 1
            if (batch_index+1) % args.steps_per_log == 0 or (batch_index+1)%len(train_dataloader) == 0:
                time_str = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                print('[{}] Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tlr:{:0.6f}\tLoss: {:0.4f}\tTime Use: {:0.4f}'.format(
                    time_str,
                    optimizer.param_groups[0]['lr'],
                    loss.item(),
                    (time_end - time_start)*1000,
                    epoch=epoch,
                    trained_samples=batch_index * args.batch + len(images),
                    total_samples=len(train_dataloader.dataset),
                ))
                logger.write('[{}]Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tlr:{:0.6f}\tLoss: {:0.4f}\tTime Use: {:0.4f}'.format(
                    time_str,
                    optimizer.param_groups[0]['lr'],
                    loss.item(),
                    (time_end - time_start)*1000,
                    epoch=epoch,
                    trained_samples=batch_index * args.batch + len(images),
                    total_samples=len(train_dataloader.dataset),
                ))
                logger.write('\n')

        if args.test and epoch % args.steps_per_test == 0:
            net.eval()

            total_loss = 0
            correct = np.zeros((args.num_classes, args.num_classes))
            acc_num = 0
            for images, labels in test_dataloader:#path

                images = images.to(args.device)
                labels = labels.to(args.device)

                predicts = net(images)
                loss = criterion(predicts, labels)
                total_loss += loss.item()
                _, preds = predicts.max(1)
                correct[labels, preds] += 1
                acc_num += preds.eq(labels).sum().float()

                

            test_loss = total_loss / iter_per_test
            acc = acc_num / len(test_dataloader.dataset)
            time_str = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
            print('[{}] Test set: loss: {:.4f}, Accuracy: {}'.format(time_str, test_loss, acc))
            # print('\t'.join(['ea_style', 'ethnic_style', 'jp_style', 'ladylike', 'leisure', 'maid_style', 'punk']))
            for i in range(args.num_classes):
                res = []
                for j in range(args.num_classes):
                    res.append(str(correct[i, j]))
                print('class[{}]:{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(i, res[0], res[1], res[2], res[3], res[4], res[5], res[6]))
            print()
            logger.write('[{}] Test set: loss: {:.4f}, Accuracy: {}'.format(time_str, test_loss, acc))
            # logger.write('\t'.join(['ea_style', 'ethnic_style', 'jp_style', 'ladylike', 'leisure', 'maid_style', 'punk']))
            for i in range(args.num_classes):
                res = []
                for j in range(args.num_classes):
                    res.append(str(correct[i, j]))
                logger.write('\t'.join(res))
            logger.write('\n\n')

        #save weights file
        if args.apex:
            checkpoint = {
                'model': net.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict()
            }
        else:
            checkpoint = {
                'model': net.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
        
        if epoch == args.epoch or not epoch % args.save_epoch:
            torch.save(checkpoint, checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
    logger.close()

    # export onnx models
    if args.export_onnx:
        checkpoint = torch.load(checkpoint_path.format(net=args.net, epoch=args.epoch, type='regular'))
        args.net = args.net+'_onnx'
        net = get_network(args)
        net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['model'].items()})
        net = net.cuda()
        net.eval()
        x = Variable(torch.randn(1, 3, args.image_size, args.image_size)).cuda()

        export_onnx_name = '/project/train/models/mobilenet/deploy.onnx'
        torch_out = torch.onnx.export(net,
                                      x,
                                      export_onnx_name,
                                      export_params=True,
                                      input_names=['data'],
                                      output_names=['prob']
                                      )
