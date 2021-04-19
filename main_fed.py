#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, traffic_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, LeNet
from models.Fed import FedAvg
from models.test import test_img
import loading_data as dataset

import torch.distributed as dist
import os
import time
from torch.multiprocessing import Process, Array
from torch.utils.tensorboard import SummaryWriter
from grace_dl.dist.communicator.allgather import Allgather
from grace_dl.dist.compressor.topk import TopKCompressor
from grace_dl.dist.compressor.dgc import DgcCompressor
from grace_dl.dist.memory.residual import ResidualMemory
from grace_dl.dist.memory.dgc import DgcMemory
from grace_dl.dist.compressor.randomk import RandomKCompressor

def partition_dataset(args):
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'traffic':
        dataset_train, dataset_test = get_train_valid_loader(
            '/Users/austineyapp/Documents/REP/Year_4/FYP/FederatedLearning/Federated-Learning-for-Traffic-Sign-Recognition',
            batch_size=32, num_workers=0)
        if args.iid:
            dict_users = traffic_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in Traffic')
    else:
        exit('Error: unrecognized dataset')
    return dataset_train, dataset_test, dict_users

def get_train_valid_loader(data_dir,
                           batch_size,
                           num_workers=0,
                           ):
    # Create Transforms
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])

    # Create Datasets
    dataset_train = dataset.BelgiumTS(
        root_dir=data_dir, train=True,  transform=transform)
    dataset_test = dataset.BelgiumTS(
        root_dir=data_dir, train=False,  transform=transform)

    # Load Datasets
    return dataset_train, dataset_test

def build_model(args):
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'LeNet' and args.dataset == 'traffic':
        net_glob = LeNet(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    return net_glob

def init_processing(rank, size, fn, lost_train, acc_train, dataset_train, idxs_users, net_glob, grc, backend='gloo'):
    """initiale each process by indicate where the master node is located(by ip and port) and run main function
    :parameter
    rank : int , rank of current process
    size : int, overall number of processes
    fn : function, function to run at each node
    backend : string, name of the backend for distributed operations
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend=backend, rank=rank, world_size=size)

    fn(rank, size, loss_train, acc_train, dataset_train, idxs_users, net_glob, grc)

def run(rank, world_size, loss_train, acc_train, dataset_train, idxs_users, net_glob, grc):
    # net_glob.load_state_dict(torch.load('net_state_dict.pt'))
    if rank == 0:
        #compressor, epoch, dgc
        foldername = f'{args.compressor}epoch{args.epochs}ratio{args.gsr}'
        tb = SummaryWriter("runs/" + foldername)
    round = 0
    for i in idxs_users:
        #for each epoch
        idx = dict_users[i[rank]]

        epoch_loss = torch.zeros(1)
        optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)

        local = LocalUpdate(args=args, dataset=dataset_train, idxs=idx) #create LocalUpdate class
        train_loss = local.train(net=net_glob) #train local
        for index, (name, parameter) in enumerate(net_glob.named_parameters()):
                grad = parameter.grad.data
                grc.acc(grad)
                new_tensor = grc.step(grad, name)
                grad.copy_(new_tensor)
        optimizer.step()
        net_glob.zero_grad()
        epoch_loss += train_loss
        dist.reduce(epoch_loss, 0, dist.ReduceOp.SUM)

        net_glob.eval()
        train_acc = torch.zeros(1)
        acc, loss = local.inference(net_glob, dataset_train, idx)
        train_acc += acc
        dist.reduce(train_acc, 0, dist.ReduceOp.SUM)

        if rank == 0:
            torch.save(net_glob.state_dict(), 'net_state_dict.pt')
            epoch_loss /= world_size
            train_acc /= world_size
            loss_train[round] = epoch_loss[0]
            acc_train[round] = train_acc[0]
            tb.add_scalar("Loss", epoch_loss[0], round)
            tb.add_scalar("Accuracy", train_acc[0], round)
            tb.add_scalar("Uncompressed Size", grc.uncompressed_size, round)
            tb.add_scalar("Compressed Size", grc.size, round)
            if round % 50 == 0:
                print('Round {:3d}, Rank {:1d}, Average loss {:.6f}, Average Accuracy {:.2f}%'.format(round, dist.get_rank(), epoch_loss[0], train_acc[0]))
        round+=1
    if rank == 0:
        tb.close()
        print("Printing Compression Stats...")
        grc.printr()

if __name__ == '__main__':
    start_time = time.time()
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load dataset and split users
    dataset_train, dataset_test, dict_users = partition_dataset(args)
    #img_size = dataset_train[0][0].shape

    net_glob = build_model(args).to('cpu')

    # training
    m = max(int(args.frac * args.num_users), 1)
    loss_train = Array('f', args.epochs)
    acc_train = Array('f', args.epochs)
    if args.compressor == 'topk':
        grc = Allgather(TopKCompressor(args.gsr), ResidualMemory(), m)
    elif args.compressor == 'randomk':
        grc = Allgather(RandomKCompressor(args.gsr), ResidualMemory(), m)
    elif args.compressor == 'dgc':
        grc = Allgather(DgcCompressor(args.gsr), DgcMemory(args.momentum, gradient_clipping=False, world_size=m), m)

    idxs_users = [] # size = (epochs * m)
    for _ in range(args.epochs):
        mRand = np.random.choice(range(args.num_users), m, replace=False) #random set of m clients
        idxs_users.append(mRand)

    processes = []

    for i in range(m):
        p = Process(target=init_processing, args=(i, m, run, loss_train, acc_train, dataset_train, idxs_users, net_glob, grc))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # testing
    net_glob.load_state_dict(torch.load('net_state_dict.pt'))
    net_glob.eval()
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Avg Train Accuracy: {:.2f}".format(acc_train[-1]))
    print("Testing Accuracy: {:.2f}".format(acc_test))

    #plot loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(loss_train)), loss_train, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_{}_C{}_iid{}_E{}_B{}_D{}_LR{}_loss.png'.format(args.compressor, args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs, args.gsr, args.lr))

    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(acc_train)), acc_train, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_{}_{}_C{}_iid{}_E{}_B{}_D{}_LR{}_acc.png'.format(args.compressor, args.dataset, args.model, args.epochs, args.frac, args.iid, args.local_ep, args.local_bs, args.gsr, args.lr))

    print("--- %s seconds ---" % (time.time() - start_time))

    # print(net_glob)
    # net_glob.train()

    # # copy weights
    # w_glob = net_glob.state_dict()

    # # training
    # loss_train = []
    # cv_loss, cv_acc = [], []
    # val_loss_pre, counter = 0, 0
    # net_best = None
    # best_loss = None
    # val_acc_list, net_list = [], []

    # if args.all_clients:
    #     print("Aggregation over all clients")
    #     w_locals = [w_glob for i in range(args.num_users)]
    # for iter in range(args.epochs):
    #     loss_locals = []
    #     if not args.all_clients:
    #         w_locals = []
    #     m = max(int(args.frac * args.num_users), 1)
    #     idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    #     for idx in idxs_users:
    #         local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
    #         w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
    #         if args.all_clients:
    #             w_locals[idx] = copy.deepcopy(w)
    #         else:
    #             w_locals.append(copy.deepcopy(w))
    #         loss_locals.append(copy.deepcopy(loss))
    #     # update global weights
    #     w_glob = FedAvg(w_locals)

    #     # copy weight to net_glob
    #     net_glob.load_state_dict(w_glob)

    #     # print loss
    #     loss_avg = sum(loss_locals) / len(loss_locals)
    #     print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
    #     loss_train.append(loss_avg)

    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # # testing
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    # print("Testing accuracy: {:.2f}".format(acc_test))

