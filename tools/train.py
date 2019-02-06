#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tommi Kerola
import os, sys, time
import argparse
import json

import chainer
from chainer import dataset
from chainer import datasets
from chainer import optimizers
from chainer.training import extensions
from chainer.training.updater import ParallelUpdater
import pandas as pd
import numpy as np
import scipy.sparse as ss


sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../')))
from lib import graph
from lib.models import graph_cnn
from lib.make_Laplacian import make_Laplacian
from tools.dataset import BrainSpectDataset, MNISTDataset
from tools.updater import GCNNUpdater


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


# def concat_and_reshape(batch, device=None, padding=None):
#     x, y = dataset.concat_examples(batch, device, padding)
#     return x.reshape(len(x), 1, 784), y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', '-o', type=str,default='results/test',
                        required=True, help='Output directory')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        required=False, help='GPU(s) to use for training')
    # parser.add_argument('--gpus', '-g', type=int, default=0,
    #                     required=True, help='GPU(s) to use for training')

    parser.add_argument('--config', '-c', type=str,default='configs/default.json',
                        required=False, help='Configuration file')
    parser.add_argument('--epoch', '-e', type=int,default=100,
                        required=False, help='Number of epochs to train for')

    parser.add_argument('--val-freq', type=int, default=1,
                        help='Validation frequency')
    parser.add_argument('--snapshot-freq', type=int,
                        default=1, help='Snapshot frequency')
    parser.add_argument('--log-freq', type=int,
                        default=1, help='Log frequency')

    parser.add_argument('--root', '-R', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/preprocessed'),
                        help='Root directory path of input image')
    parser.add_argument('--group','-group', default='datasets/3fold/group1/',
                        help='training list')

    parser.add_argument('--train_list', default='train_list.txt',
                        help='training list')
    # parser.add_argument('--val_list', default='configs/val_list.txt',
    #                     help='validation list')
    parser.add_argument('--val1_list', default='val_list.txt',
                        help='validation list')
    # parser.add_argument('--val2_list', default='val2_list.txt',
    #                     help='validation list')
    parser.add_argument('--num_sampling_point',type=int, default=1024,
                        help='num_sampling_point')

    args = parser.parse_args()
    # path_adjecency_matrix='datasets/adjecency_matrix_{}.csv'.format(str(args.num_sampling_point)),
    path_adjecency_matrix='datasets/adjecency_matrix_'+str(args.num_sampling_point)+'.csv'


    with open(args.config) as f:
        config = json.load(f)

    # A = graph.grid_graph(28)
    # A=make_Laplacian(26)
    A=pd.read_csv(path_adjecency_matrix)
    A=np.array(A.values)
    A=ss.csr.csr_matrix(A)
    print(A)

    model = graph_cnn.GraphCNN(A)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam(alpha=1e-3)
    optimizer.setup(model)
    if 'optimizer' in config:
        optimizer.add_hook(chainer.optimizer.WeightDecay(
            config['optimizer']['weight_decay']))

    devices = {'main': args.gpu}
    # for gid in args.gpus[1:]:
    #     devices['gpu{}'.format(gid)] = gid
    # config['batch_size'] *= len(args.gpu)
    # print(args.gpu[0])
    # train_dataset, val_dataset = datasets.get_mnist()

    path = os.path.join(args.group, args.train_list)
    train_dataset=BrainSpectDataset(root=args.root, group=args.group,list_path=path,num_sampling_point=str(args.num_sampling_point))
    path = os.path.join(args.group, args.val1_list)
    val_dataset=BrainSpectDataset(root=args.root, group=args.group,list_path=path,num_sampling_point=str(args.num_sampling_point))

    train_iter = chainer.iterators.SerialIterator(
        train_dataset, config['batch_size'])
    val_iter = chainer.iterators.SerialIterator(
        val_dataset, batch_size=1, repeat=False, shuffle=False)

    # updater = GCNNUpdater(train_iter, optimizer ={'main':optimizer}, device=args.gpu[0])
    updater = GCNNUpdater(models=model,
                          iterator=train_iter,
                          optimizer ={'main':optimizer},
                          device=args.gpu)

    trainer = chainer.training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.outdir)

    # Extentions
    trainer.extend(
        TestModeEvaluator(val_iter, model, device=args.gpu),
        trigger=(args.val_freq, 'epoch'))
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.epoch}.npz'), trigger=(args.snapshot_freq, 'epoch'))
    trainer.extend(
        extensions.LogReport(trigger=(args.log_freq, 'epoch')))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration',
        'main/loss', 'main/accuracy',
        'validation/main/loss', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__ == '__main__':
    main()
