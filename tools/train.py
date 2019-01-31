#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Tommi Kerola

import argparse
import json

import chainer
from chainer import dataset
from chainer import datasets
from chainer import optimizers
from chainer.training import extensions
from chainer.training.updater import ParallelUpdater

from lib import graph
from lib.models import graph_cnn

import os

class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def concat_and_reshape(batch, device=None, padding=None):
    x, y = dataset.concat_examples(batch, device, padding)
    return x.reshape(len(x), 1, 784), y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,default="C:\\Users\\yambe\\PycharmProjects\\chainer(1.19.0)-graph-cnn-master\\configs\\default.json",
                         help='Configuration file')
    parser.add_argument('--outdir', '-o', type=str,
                        default="C:\\Users\\yambe\\PycharmProjects\\chainer(1.19.0)-graph-cnn-master\\results", help='Output directory')
    parser.add_argument('--epoch', '-e', type=int,default=100,
                         help='Number of epochs to train for')
    parser.add_argument('--gpus', '-g', type=int, nargs="*",default=[-1],
                         help='GPU(s) to use for training')
    # parser.add_argument('--gpu', '-g', type=int, default=-1,
    #                     help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--val-freq', type=int, default=1,
                        help='Validation frequency')
    parser.add_argument('--snapshot-freq', type=int,
                        default=1, help='Snapshot frequency')
    parser.add_argument('--log-freq', type=int,
                        default=1, help='Log frequency')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    A = graph.grid_graph(28)
    model = graph_cnn.GraphCNN(A)

    optimizer = optimizers.Adam(alpha=1e-3)
    optimizer.setup(model)
    if 'optimizer' in config:
        optimizer.add_hook(chainer.optimizer.WeightDecay(
            config['optimizer']['weight_decay']))

    devices = {'main': args.gpus[0]}
    # devices = {'main': args.gpu}

    for gid in args.gpus[1:]:
        devices['gpu{}'.format(gid)] = gid
    config['batch_size'] *= len(args.gpus)

    train_dataset, val_dataset = datasets.get_mnist()

    train_iter = chainer.iterators.MultiprocessIterator(
        train_dataset, config['batch_size'])
    val_iter = chainer.iterators.SerialIterator(
        val_dataset, batch_size=1, repeat=False, shuffle=False)

    updater = ParallelUpdater(train_iter, optimizer, devices=devices,
                              converter=concat_and_reshape)
    trainer = chainer.training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.outdir)

    # Extentions
    trainer.extend(
        TestModeEvaluator(val_iter, model, device=devices['main'],
                          converter=concat_and_reshape),
        trigger=(args.val_freq, 'epoch'))
    # aaa=args.snapshot_freq, 'epoch'
    # print('aaa',aaa)
    trainer.extend(
        extensions.snapshot(),trigger=(args.snapshot_freq, 'epoch'))
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
