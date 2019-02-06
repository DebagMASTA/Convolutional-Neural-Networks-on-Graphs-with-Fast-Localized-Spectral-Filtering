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
import sklearn.metrics.pairwise

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../')))
from lib import graph
from lib.models import graph_cnn
from lib.make_Laplacian import make_Laplacian
from tools.dataset import BrainSpectDataset, MNISTDataset
import utils.ioFunctions as IO
from utils.sampling_method import uniform_subsampling


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

def save_subsampling(fold,group,dat_dir,ZSFM,num_sampling_point,random_seed):
    train_list='datasets/{}/{}/train_list.txt'.format(fold,group)
    val_list='datasets/{}/{}/val_list.txt'.format(fold,group)
    subsampling_data_dir='datasets/{}/{}/subsampling_{}/'.format(fold,group,num_sampling_point)
    if not os.path.exists(subsampling_data_dir):
        os.makedirs(subsampling_data_dir)
    #training data
    f = open(train_list)
    all_cases = f.readlines()
    f.close()
    for case in all_cases:
        case = case.replace('\n', '')
        data_df = IO.read_dat(dat_dir + case + ZSFM + '.dat')[:-1]
        subsampling_data_df = uniform_subsampling(data_df, num_sampling_point, random_seed)
        assert (len(subsampling_data_df) == num_sampling_point)
        subsampling_data_df.to_csv(subsampling_data_dir + case + ZSFM + '.csv',index=False,mode='w')
    #validation data
    f = open(val_list)
    all_cases = f.readlines()
    f.close()
    for case in all_cases:
        case = case.replace('\n', '')
        data_df = IO.read_dat(dat_dir + case + ZSFM + '.dat')[:-1]
        subsampling_data_df = uniform_subsampling(data_df, num_sampling_point, random_seed)
        assert (len(subsampling_data_df) == num_sampling_point)
        subsampling_data_df.to_csv(subsampling_data_dir + case + ZSFM + '.csv',index=False,mode='w')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,default='D:\PycharmProjects\Convolutional-Neural-Networks-on-Graphs-with-Fast-Localized-Spectral-Filtering\configs/default.json',
                        required=False, help='Configuration file')
    parser.add_argument('--outdir', '-o', type=str,default='results/0204',
                        required=True, help='Output directory')
    parser.add_argument('--epoch', '-e', type=int,default=100,
                        required=False, help='Number of epochs to train for')
    parser.add_argument('--gpus', '-g', type=int, nargs="*",default=0,
                        required=True, help='GPU(s) to use for training')
    parser.add_argument('--val-freq', type=int, default=1,
                        help='Validation frequency')
    parser.add_argument('--snapshot-freq', type=int,
                        default=1, help='Snapshot frequency')
    parser.add_argument('--log-freq', type=int,
                        default=1, help='Log frequency')

    parser.add_argument('--root', '-R', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/preprocessed'),
                        help='Root directory path of input image')
    parser.add_argument('--train_list', default='datasets/3fold/train_list.txt',
                        help='training list')
    parser.add_argument('--val_list', default='configs/val_list.txt',
                        help='validation list')

    parser.add_argument('--ZSFM', default='.GLBZSFM',
                        help='ZSFM')
    parser.add_argument('--num_sampling_point',type=int, default=1024,
                        help='num_sampling_point')
    parser.add_argument('--random_seed',type=int, default=0,
                        help='random_seed')
    parser.add_argument('--fold',default='3fold',
                        help='fold of CV')

    dat_dir='C:\\Users\\yambe\\Documents\\Study\\Experiment\\all_data\\'
    args = parser.parse_args()


    with open(args.config) as f:
        config = json.load(f)

    # A = graph.grid_graph(28)
    A=make_Laplacian(26)
    A_A1=graph.SPECT_graph('C:\\Users\\yambe\\Documents\\Study\\Experiment\\all_data\\A-1.dat',
                            num_sampling_point=args.num_sampling_point,random_seed=args.random_seed)
    # A_A3=graph.SPECT_graph('C:\\Users\\yambe\\Documents\\Study\\Experiment\\all_data\\A-3.dat',
    #                         num_sampling_point=args.num_sampling_point,random_seed=0)
    # A_A5=graph.SPECT_graph('C:\\Users\\yambe\\Documents\\Study\\Experiment\\all_data\\A-5.dat',
    #                         num_sampling_point=args.num_sampling_point,random_seed=0)

    # A13=A_A1-A_A3
    # A15=A_A1-A_A5
    A_A1=A_A1.toarray()
    dfA=pd.DataFrame(A_A1)
    dfA.to_csv('datasets/adjecency_matrix_{}.csv'.format(args.num_sampling_point),index=False)
    # print("A13",A13.shape)
    # A13=A13.toarray()
    # df13=pd.DataFrame(A13)
    # df13.to_csv(args.outdir+'/A13.csv',mode='w')
    #
    # A15=A15.toarray()

    # df15=pd.DataFrame(A15)
    # df15.to_csv(args.outdir+'/A15.csv',mode='w')

    print('A.shape',A.shape)
    model = graph_cnn.GraphCNN(A)

    optimizer = optimizers.Adam(alpha=1e-3)
    optimizer.setup(model)
    if 'optimizer' in config:
        optimizer.add_hook(chainer.optimizer.WeightDecay(
            config['optimizer']['weight_decay']))

    devices = {'main': args.gpus[0]}
    for gid in args.gpus[1:]:
        devices['gpu{}'.format(gid)] = gid
    config['batch_size'] *= len(args.gpus)

    save_subsampling(fold=args.fold,group='group1',dat_dir=dat_dir,
                     ZSFM=args.ZSFM,
                     num_sampling_point=args.num_sampling_point,
                     random_seed=args.random_seed)

    save_subsampling(fold=args.fold,group='group2',dat_dir=dat_dir,
                     ZSFM=args.ZSFM,
                     num_sampling_point=args.num_sampling_point,
                     random_seed=args.random_seed)

    save_subsampling(fold=args.fold,group='group3',dat_dir=dat_dir,
                     ZSFM=args.ZSFM,
                     num_sampling_point=args.num_sampling_point,
                     random_seed=args.random_seed)


    # path = os.path.join(args.base, args.train_list)
    # train_dataset=BrainSpectDataset(root=args.root, list_path=path)

if __name__ == '__main__':
    main()
