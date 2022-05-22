
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import torch.utils.data
import torch.utils.data.distributed
import torch_geometric.nn


import glob

import numpy as np
import torch.nn.functional as F
from HGP_SL.models import Model_joint
from torch.utils.data import random_split
from torch_geometric.data import DataLoader,DataListLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Batch

import moco.builder
from augmentation import random_augmentation
from torch.utils.data import ConcatDataset



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers ')
parser.add_argument('--epochs', default=20000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[100,1000], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')


# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=512, type=int,
                    help='queue size; number of negative keys (default: 512)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')


parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# HGP-SL configs:

parser.add_argument('--sample_neighbor', type=bool, default=False, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='all', help='all')
parser.add_argument('--test_dataset',type=str, default='NCI1', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--patience', type=int, default=100, help='early stopping')
loss_list = []
def main():
    args = parser.parse_args()
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. ')


    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu



    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))



    if args.dataset == 'all':

        target_dataset = TUDataset(os.path.join('data', args.test_dataset), name=args.test_dataset, use_node_attr=True, use_edge_attr=False)
        datasets=[]
        for name in ['Mutagenicity','NCI1','NCI109','PROTEINS']: # DD cost much time, add it or not
            if name!=args.test_dataset:
                datasets.append(TUDataset(os.path.join('data', name), name=name, use_node_attr=True, use_edge_attr=False))

        num_training = int(len(target_dataset) * 0.8)
        num_val = int(len(target_dataset) * 0.1)
        num_test = len(target_dataset) - (num_training + num_val)
        print('num_training,num_val,num_test',num_training,num_val,num_test)
        training_set, validation_set, test_set = random_split(target_dataset, [num_training, num_val, num_test])

        train_dataset = ConcatDataset([d for d in datasets])
        train_dataset = ConcatDataset([train_dataset,training_set])
        num_features = 0
        train_dataset_list = []
        validation_set_list=[]
        test_set_list=[]
        for i in range(len(train_dataset)):
            train_dataset_list.append(train_dataset[i])
            if num_features<train_dataset[i].x.shape[1]:
                num_features = train_dataset[i].x.shape[1]
        args.num_features = num_features
        #print('num_features',num_features)
        for i in range(len(validation_set)):
            validation_set_list.append(validation_set[i])
        for i in range(len(test_set)):
            test_set_list.append(test_set[i])

        for graph in train_dataset_list+validation_set_list+test_set_list:
            if graph.x.shape[1]<num_features:
                pad = nn.ZeroPad2d(padding=(0,num_features - graph.x.shape[1],0,0))
                xpad = pad(graph.x)
                graph.x = xpad


    print(args.dataset, len(train_dataset))
    args.num_classes = target_dataset.num_classes

    args.nhid = args.moco_dim
    # create model
    print("=> creating model with HGP-SL")
    HGP_SL = Model_joint(args)
    ###################################################################################################################################
    model = moco.builder.MoCo(HGP_SL,args.moco_dim, args.moco_k, args.moco_m, args.moco_t)


    for name, param in model.named_parameters():
        #print(name,param.requires_grad)
        param.requires_grad=True

    print(model)


    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)



    # define loss function (criterion) and optimizer
    if args.gpu is not None:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss()


    #optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True





    train_sampler = None



    if args.dataset == 'all':
        train_loader = DataListLoader(train_dataset_list, batch_size=args.batch_size, shuffle=(train_sampler is None),num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = DataLoader(validation_set_list, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set_list, batch_size=args.batch_size, shuffle=False)

    min_loss = 1e10
    val_loss_values = []
    print('begin training')
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch

        train(train_loader, val_loader, val_loss_values, model, criterion, optimizer, epoch, args)
        #print(optimizer)
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_model = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        print('patience',patience_cnt)

        if not os.path.exists('./results/CSSL-Reg/'+args.dataset+'/'+args.test_dataset+'/'+str(args.batch_size)):
            os.makedirs('./results/CSSL-Reg/'+args.dataset+'/'+args.test_dataset+'/'+str(args.batch_size))
        if (epoch+1)%1==0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'HGP-SL',
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='./results/CSSL-Reg/'+args.dataset+'/'+args.test_dataset+'/'+str(args.batch_size)+'/checkpoint_{:05d}.pth.tar'.format(epoch))

        if patience_cnt == args.patience or epoch == args.epochs - 1:
            model.load_state_dict(torch.load('./results/CSSL-Reg/'+args.dataset+'/'+args.test_dataset+'/'+str(args.batch_size)+'/checkpoint_{:05d}.pth.tar'.format(best_model))['state_dict'])
            test_acc, test_loss = compute_test(model.encoder_q, test_loader, args)
            print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))
            break


def train(train_loader, val_loader,val_loss_values, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))


    # switch to train mode
    model.train()


    correct = 0

    for i,  images in enumerate(train_loader):
        # measure data loading time
        end = time.time()
        for im in images:
            im.edge_attr=None
        images_cls = Batch.from_data_list(images)
        im_q = Batch.from_data_list(random_augmentation(images))
        im_k = Batch.from_data_list(random_augmentation(images))

        data_time.update(time.time() - end)

        if args.gpu is not None:

            im_q = im_q.to(args.gpu)#, non_blocking=True)
            im_k = im_k.to(args.gpu)#, non_blocking=True)
            images_cls= images_cls.to(args.gpu)

        output, target, q_cls = model(im_q=im_q, im_k=im_k,image = images_cls)

        if args.gpu!=None:
            target = target.to(args.gpu)
        loss = criterion(output, target)
        loss_cls = F.nll_loss(q_cls, images_cls.y)
        #print('moco loss',loss,'cls loss',loss_cls)
        loss = 0.001*loss + loss_cls

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), len(images))
        loss_list.append(loss.item())
        top1.update(acc1[0], len(images))
        top5.update(acc5[0], len(images))

        # compute gradient and do SGD step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        #cls performance
        pred = q_cls.max(dim=1)[1]
        correct += pred.eq(images_cls.y).sum().item()
    acc_train = correct / len(train_loader.dataset)
    loss_train = loss_cls
    acc_val, loss_val = compute_test(model.encoder_q, val_loader,args)
    print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
            'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
            'acc_val: {:.6f}'.format(acc_val))

    val_loss_values.append(loss_val)



def compute_test(model, loader,args):
    model.eval()

    correct = 0.0
    loss_test = 0.0
    number = 0
    for data in loader:
        data = data.to(args.gpu)
        _,  out = model(data)

        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()* (data.y.shape[0])
        number += data.y.shape[0]
    loss_test = loss_test/number
    return correct / len(loader.dataset), loss_test

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('cuurent learning rate',lr)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
