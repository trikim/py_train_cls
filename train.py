
"""author 
   baiyu
"""

import argparse
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from logger import Logger
#from PIL import Image
import transforms 
#from torchvision import transforms
#from dataset.dataloaders import *
from dataset.dataloaders_custom import *
from tensorboardX import SummaryWriter
from conf import settings
from utils import *
from lr_scheduler import WarmUpLR
from lr_scheduler.mixup import NLLMultiLabelSmooth, MixUpWrapper, rand_bbox
from criterion import LSR
import time
import json
import apex
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
    parser.add_argument('-workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-batch', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.04, help='initial learning rate')
    parser.add_argument('-lr_step', type=str, default='30,50,60',help='drop learning rate by 10.')
    parser.add_argument('-epoch', type=int, default=72, help='training epoches')
    parser.add_argument('-save_epoch', type=int, default=1, help='save epoches')
    parser.add_argument('-data-backend', metavar='BACKEND', default='custom')#,choices=DATA_BACKEND_CHOICES)
    parser.add_argument('-image-size', type=int, default=224, help='training image ')
    parser.add_argument('-warm', type=int, default=5, help='warm up phase')
    parser.add_argument('-mixup', default=0.0, type=float,metavar='ALPHA', help='mixup alpha')
    parser.add_argument('-cutmix_prob', default=0.0, type=float, help='cutmix probability')
    parser.add_argument('-gpus', default=0, help='gpu device')
    parser.add_argument('-num_classes', type=int, default=2, help='num_classes')
    parser.add_argument('-pretrained',action='store_true',default=False, help='pretrained.')
    parser.add_argument('-resume', action='store_true',default=False, help='resume.')
    parser.add_argument('-refine',type=str,default=None)
    parser.add_argument('-start_epoch',type=int,default=1)
    parser.add_argument('-apex', action='store_true',default=True, help='apex mixed precision')
    parser.add_argument('-opt_level', type=str,default="O1")
    parser.add_argument('-keep_batchnorm_fp32', type=str, default=None)
    parser.add_argument('-sync_bn', action='store_true',default=False, help='enabling apex sync BN.')
    parser.add_argument('-imgs_root',type=str,required=True, default=None)
    parser.add_argument('-label_txt',type=str,required=True, default=None)
    parser.add_argument('-test',action='store_true',default=False, help='test.')
    parser.add_argument('-test_label_txt',type=str,default=None)
    parser.add_argument('-test_epoch', type=int, default=1, help='test epoches')
    parser.add_argument('-export_onnx',action='store_true',default=False,help='onnx.')
    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                        'executing allreduce across workers; it multiplies '
                        'total batch size.')   
    #parser.add_argument('--loss-scale', type=str, default=None)
    args = parser.parse_args()
    args.gpus_str = args.gpus
    args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
    args.gpus = [i for i in range(len(args.gpus))] if args.gpus[0] >=0 else [-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    args.device = torch.device('cuda' if args.gpus[0] >= 0 else 'cpu')

    #cd = torch.cuda.current_device()
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
    writer = SummaryWriter(log_dir=log_path)
    # from dataset.dataloaders import *
    if args.data_backend == "custom":
        train_dataloader = get_custom_train_loader(args, args.imgs_root, args.label_txt, args.batch, args.num_classes, args.workers)
        iter_per_epoch = len(train_dataloader)
        if args.test:
            test_dataloader = get_custom_val_loader(args, args.imgs_root, args.test_label_txt, 1, args.num_classes, 1)
            iter_per_test = len(test_dataloader)
    elif args.data_backend == 'dali-gpu':
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
        train_dataloader, train_loader_len = get_train_loader(args.imgs_root_IMAGENET, args.batch, args.num_classes, args.mixup > 0.0, workers=args.workers)
        test_dataloader,test_loader_len = get_val_loader(args.imgs_root_IMAGENET_VAL, 1, args.num_classes, args.mixup > 0.0, workers=1)
        iter_per_epoch = train_loader_len
        iter_per_test = test_loader_len
        print("iter_per_epoch:",iter_per_epoch)
    elif args.data_backend == 'dali-cpu':
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()
        train_dataloader, train_loader_len = get_train_loader(args.imgs_root_IMAGENET, args.batch, args.num_classes, args.mixup > 0.0, workers=args.workers)
        test_dataloader,test_loader_len = get_val_loader(args.imgs_root_IMAGENET_VAL, 1, args.num_classes, args.mixup > 0.0, workers=1)
        iter_per_epoch = train_loader_len
        iter_per_test = test_loader_len
    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, args.num_classes, train_loader)



    
    net = get_network(args)
    if not args.pretrained:
        net = init_weights(net)
    if args.resume and args.refine:
        checkpoint = torch.load(args.refine)
        # net = get_network(args)
        net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['model'].items()})
        # start_epoch = checkpoint['epoch']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # amp.load_state_dict(checkpoint['amp'])
    net = net.to(args.device)
  
    lsr_loss = nn.CrossEntropyLoss() 
    # lsr_loss = LSR()

    params = split_weights(net) #apply no weight decay on bias
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    if args.apex:
        net, optimizer = amp.initialize(net, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32)
    

    net = nn.DataParallel(net, device_ids=args.gpus).to(args.device)
    if args.apex and args.sync_bn:
        print("using apex synced BN")
        net = apex.parallel.convert_syncbn_model(net)

    #set up warmup phase learning rate scheduler

    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    #set up training phase learning rate scheduler
    #train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step)
    # train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch - args.warm)
    train_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.002, step_size_up=iter_per_epoch)
    time_start_all = time.time()
    best_acc = 0.0
    for epoch in range(args.start_epoch, args.epoch + 1):
        # if epoch > args.warm:
        #     train_scheduler.step(epoch)
        if epoch == int(3*args.epoch / 8):
            lsr_loss = nn.CrossEntropyLoss()
        #training procedure
        net.train()
        for batch_index, (images, labels) in enumerate(train_dataloader):
            if epoch <= args.warm:
                warmup_scheduler.step()
            else:
                train_scheduler.step()
            #print("batch_index:",batch_index)
            images = images.to(args.device)
            labels = labels.to(args.device).to(args.device)
            time_start = time.time()
            optimizer.zero_grad()
            # cutmix
            r = np.random.rand(1)
            if r < args.cutmix_prob and epoch < 5*args.epoch/8:
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
                loss = lsr_loss(output, target_a) * lam + lsr_loss(output, target_b) * (1. - lam)
            else:
                predicts = net(images)
                loss = lsr_loss(predicts, labels)
            if args.apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss: 
                    scaled_loss.backward()
                    # optimizer.synchronize()
                # with optimizer.skip_synchronize():
                #     optimizer.step()
                    #net.zero_grad()
            else:
                loss.backward()
            optimizer.step()
            time_end = time.time()
            n_iter = (epoch - 1) * iter_per_epoch + batch_index + 1
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tlr:{:0.6f}\tLoss: {:0.4f}\tTime Use: {:0.4f}'.format(
                optimizer.param_groups[0]['lr'],
                loss.item(),
                (time_end - time_start)*1000,
                epoch=epoch,
                trained_samples=batch_index * args.batch + len(images),
                #trained_samples=batch_index,
                total_samples=len(train_dataloader.dataset),
                #total_samples=iter_per_epoch,
            ))
            if logger:
              logger.write('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tlr:{:0.6f}\tLoss: {:0.4f}\tTime Use: {:0.4f}'.format(
                  optimizer.param_groups[0]['lr'],
                  loss.item(),
                  (time_end - time_start)*1000,
                  epoch=epoch,
                  trained_samples=batch_index * args.batch + len(images),
                  #trained_samples=batch_index,
                  total_samples=len(train_dataloader.dataset),
                  #total_samples=iter_per_epoch,
              ))
              logger.write('\n')
            #visualization
            if writer:
              visualize_lastlayer(writer, net, n_iter)
              visualize_train_loss(writer, loss.item(), n_iter)
        if writer:
          visualize_learning_rate(writer, optimizer.param_groups[0]['lr'], epoch)
          visualize_param_hist(writer, net, epoch) 

        if args.test and (epoch == args.epoch or not epoch % args.test_epoch):
            net.eval()

            total_loss = 0
            correct = 0
            for images, labels in test_dataloader:#path

                images = images.to(args.device)
                labels = labels.to(args.device)

                predicts = net(images)
                _, preds = predicts.max(1)
                correct += preds.eq(labels).sum().float()

                loss = lsr_loss(predicts, labels)
                total_loss += loss.item()

            test_loss = total_loss / iter_per_test
            #acc = correct / (iter_per_test*args.batch)
            acc = correct / len(test_dataloader.dataset)
            print('Test set: loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, acc))
            print()
            if logger:
              logger.write('Test set: loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, acc))
              logger.write('\n')
            if writer:
              visualize_test_loss(writer, test_loss, epoch)
              visualize_test_acc(writer, acc, epoch)

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

        if args.test and epoch > args.lr_step[1] and best_acc < acc:
            torch.save(checkpoint, checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue
        
        if  epoch == args.epoch or not epoch % args.save_epoch:
            torch.save(checkpoint, checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
    time_end_all = time.time()
    print('All Time Use: {:0.4f}'.format((time_end_all - time_start_all)*1000))
    if logger:
      logger.close()
    if writer:
      writer.close()










    


    

