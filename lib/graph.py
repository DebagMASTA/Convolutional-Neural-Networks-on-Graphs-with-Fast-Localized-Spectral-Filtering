#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as ss
import scipy.spatial.distance
import sklearn.metrics.pairwise
import csv
import sys, os, time
import argparse, yaml, shutil, math
import scipy.sparse.linalg
sys.path.insert(0, '..')
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../..')))
import utils.ioFunctions as IO

from utils.sampling_method import uniform_subsampling
# from utils.mathematical_functions import adjacency, scaled_laplacian

import numpy as np


def create_laplacian(W, normalize=True):
    n = W.shape[0]
    W = ss.csr_matrix(W)
    WW_diag = W.dot(ss.csr_matrix(np.ones((n, 1)))).todense()
    if normalize:
        WWds = np.sqrt(WW_diag)
        # Let the inverse of zero entries become zero.
        WWds[WWds == 0] = np.float("inf")
        WW_diag_invroot = 1. / WWds
        D_invroot = ss.lil_matrix((n, n))
        D_invroot.setdiag(WW_diag_invroot)
        D_invroot = ss.csr_matrix(D_invroot)
        I = scipy.sparse.identity(W.shape[0], format='csr', dtype=W.dtype)
        L = I - D_invroot.dot(W.dot(D_invroot))
    else:
        D = ss.lil_matrix((n, n))
        D.setdiag(WW_diag)
        D = ss.csr_matrix(D)
        L = D - W

    return L.astype(W.dtype)


def grid(m, dtype=np.float32):
    """Return the embedding of a grid graph."""
    # Adapted from https://github.com/mdeff/cnn_graph/blob/master/lib/graph.py
    M = m**2
    x = np.linspace(0, 1, m, dtype=dtype)
    y = np.linspace(0, 1, m, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), dtype)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z


def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    # Adapted from https://github.com/mdeff/cnn_graph/blob/master/lib/graph.py
    print('aaa',z.shape)
    d = sklearn.metrics.pairwise.pairwise_distances(
        z, metric=metric, n_jobs=-2)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k + 1]
    d.sort()
    d = d[:, 1:k + 1]
    return d, idx


def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    # Adapted from https://github.com/mdeff/cnn_graph/blob/master/lib/graph.py
    M, k = dist.shape
    print('dist',dist)
    print(idx)
    print('M,k',M,k)
    print('idx',idx.shape)
    assert M, k == idx.shape
    assert dist.min() >= 0

    # Weights.
    sigma2 = np.mean(dist[:, -1])**2
    dist = np.exp(- dist**2 / sigma2)

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M * k)
    V = dist.reshape(M * k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W


def grid_graph(m):
    # Adapted from https://github.com/mdeff/cnn_graph/blob/master/lib/graph.py
    z = grid(m)
    dist, idx = distance_sklearn_metrics(z, k=8)
    A = adjacency(dist, idx)
    return A

def SPECT_graph(filepath='C:\\Users\\yambe\\Documents\\Study\\Experiment\\all_data\\A-1.dat',
                num_sampling_point=1024,random_seed=0):
    # len = 15964
    # x_min = 34
    # y_min = 25
    # x_size = 60
    # y_size = 77
    # z_size = 59
    # size = x_size * y_size * z_size
    #
    # with open('C:\\Users\\yambe\\Documents\\Study\\Experiment\\all_data\\A-1.dat', 'r') as org:
    #     """学習データについて"""
    #     org = list(csv.reader(org))  # データの末尾に無駄な１行あり
    #     org = np.reshape(org, (len + 1, 4))  # 15965*4のnumpy.arrayに変形
    #     org = np.delete(org, len, 0)  # いらない末尾を削除
    #     value = [n[3] for n in org]  # ４列目を、１次元配列value_trainとする
    #     value = np.array(value, dtype=float)  # float型にする
    #
    # loc = np.delete(org, 3, 1)  # locは15964*3の濃度値がある座標を示す
    # loc = np.array(loc, dtype=int)
    # print('loc.shape',loc.shape)
    # # loc[:,0]=(loc[:,0]-x_min)/x_size
    # # loc[:,1]=(loc[:,1]-y_min)/y_size
    # # loc[:,2]=loc[:,2]/z_size
    #
    # loc[:,0]=(loc[:,0]-x_min)
    # loc[:,1]=(loc[:,1]-y_min)
    # loc[:,2]=loc[:,2]
    # z=loc
#############################


    data_df = IO.read_dat(filepath)[:-1]

    subsampling_data_df = uniform_subsampling(data_df, num_sampling_point, random_seed)
    assert (len(subsampling_data_df) == num_sampling_point)
    # flattenIntensity = subsampling_data_df['intensity'].values.reshape(1, -1)
    # print(flattenIntensity)
    point_coorfinates = subsampling_data_df[['x', 'y', 'z']].values
    print(point_coorfinates.shape)

    dist, idx = distance_sklearn_metrics(point_coorfinates, k=8)
    # print('idx',idx.shape,idx)
    # print('dist',dist.shape,dist)

    A = adjacency(dist, idx)
    return A
