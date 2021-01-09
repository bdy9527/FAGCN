import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from dgl import DGLGraph
from utils import accuracy, preprocess_data
from model import FAGCN

# torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='new_squirrel')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--eps', type=float, default=0.3, help='Fixed scalar or learnable weight.')
parser.add_argument('--layer_num', type=int, default=2, help='Number of layers')
parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training set')
parser.add_argument('--patience', type=int, default=200, help='Patience')
args = parser.parse_args()

device = 'cuda'

g, nclass, features, labels, train, val, test = preprocess_data(args.dataset, args.train_ratio)
features = features.to(device)
labels = labels.to(device)
rain = train.to(device)
test = test.to(device)
val = val.to(device)

g = g.to(device)
deg = g.in_degrees().cuda().float().clamp(min=1)
norm = torch.pow(deg, -0.5)
g.ndata['d'] = norm

net = FAGCN(g, features.size()[1], args.hidden, nclass, args.dropout, args.eps, args.layer_num).cuda()

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# main loop
dur = []
los = []
loc = []
counter = 0
min_loss = 100.0
max_acc = 0.0

for epoch in range(args.epochs):
    if epoch >= 3:
        t0 = time.time()

    net.train()
    logp = net(features)

    cla_loss = F.nll_loss(logp[train], labels[train])
    loss = cla_loss
    train_acc = accuracy(logp[train], labels[train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    net.eval()
    logp = net(features)
    test_acc = accuracy(logp[test], labels[test])
    loss_val = F.nll_loss(logp[val], labels[val]).item()
    val_acc = accuracy(logp[val], labels[val])
    los.append([epoch, loss_val, val_acc, test_acc])

    if loss_val < min_loss and max_acc < val_acc:
        min_loss = loss_val
        max_acc = val_acc
        counter = 0
    else:
        counter += 1

    if counter >= args.patience and args.dataset in ['cora', 'citeseer', 'pubmed']:
        print('early stop')
        break

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(
        epoch, loss_val, train_acc, val_acc, test_acc, np.mean(dur)))


if args.dataset in ['cora', 'citeseer', 'pubmed'] or 'syn' in args.dataset:
    los.sort(key=lambda x: x[1])
    acc = los[0][-1]
    print(acc)
else:
    los.sort(key=lambda x: -x[2])
    acc = los[0][-1]
    print(acc)

