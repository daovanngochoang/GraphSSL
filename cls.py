import argparse
import glob
import os
import time
import sys
sys.path.append('./HGP_SL')
print (sys.path)
import torch
import torch.nn.functional as F
import torch.nn as nn
from models import Model, lincls
from torch.utils.data import random_split
from torch_geometric.data import DataLoader,DataListLoader
from torch_geometric.datasets import TUDataset

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.8, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='NCI1', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=2000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features
max_num_features = 89
if 'all' in args.resume:
    args.num_features = max_num_features
print(args)





model = Model(args).to(args.device)
if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            model.load_state_dict(state_dict, strict=False)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        if 'all' in args.resume:#pretrained with all datasets
            dataset_list = []
            for i in range(len(dataset)):
                dataset_list.append(dataset[i])
            for graph in dataset_list:
                if graph.x.shape[1]<max_num_features:
                    pad = nn.ZeroPad2d(padding=(0,max_num_features - graph.x.shape[1],0,0))
                    xpad = pad(graph.x)
                    graph.x = xpad
        else:
            dataset_list = dataset

num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
print('num_training,num_val,num_test',num_training,num_val,num_test)
training_set, validation_set, test_set = random_split(dataset_list, [num_training, num_val, num_test])
train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

classifier = lincls(args).to(args.device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train():
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.eval()

    for epoch in range(args.epochs):

        classifier.train()
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):

            optimizer.zero_grad()
            data = data.to(args.device)

            _, out = model(data)
            #print(out.shape)
            out = classifier(out)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item() * (data.y.shape[0])

            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        loss_train = loss_train/num_training
        acc_val, loss_val = compute_test(val_loader)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

        val_loss_values.append(loss_val)
        torch.save(classifier.state_dict(), args.dataset+args.resume.split('.')[1].split('/')[-1]+'lincls_epoch={}.pth'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


def compute_test(loader):
    model.eval()
    classifier.eval()
    correct = 0.0
    loss_test = 0.0
    number = 0
    for data in loader:
        data = data.to(args.device)
        _, out = model(data)
        out = classifier(out)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()* (data.y.shape[0])
        number+= data.y.shape[0]
    loss_test = loss_test/number
    return correct / len(loader.dataset), loss_test


if __name__ == '__main__':
    # Model training
    best_model = train()
    # Restore best model for test set
    classifier.load_state_dict(torch.load(args.dataset+args.resume.split('.')[1].split('/')[-1]+'lincls_epoch={}.pth'.format(best_model)))
    test_acc, test_loss = compute_test(test_loader)
    print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))



