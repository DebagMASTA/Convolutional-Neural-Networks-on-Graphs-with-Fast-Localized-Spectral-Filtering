# coding:utf-8
import os, sys, time
import numpy as np
import chainer
import argparse, glob
import scipy
import sklearn.metrics.pairwise

import utils.ioFunctions as IO

class BrainSpectDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, group,list_path,num_sampling_point):
        """
        """
        with open(list_path) as cases:
            for i,case in enumerate(cases):
                case=case.replace('\n','')
                data_df=IO.read_csv(group+'subsampling_'+num_sampling_point+'/{}.GLBZSFM.csv'.format(case))
                """ZSFMの種類！！！"""
                intensity=data_df[['intensity']].values
                intensity=intensity.reshape(int(num_sampling_point))

                if 'N' in case:
                    label=np.array([1,0,0,0])
                if 'A' in case:
                    label=np.array([0,1,0,0])
                if 'D' in case:
                    label=np.array([0,0,1,0])
                if 'F' in case:
                    label=np.array([0,0,0,1])
                if i == 0:
                    label_dataset = label
                    intensity_dataset = intensity
                else:
                    label_dataset = np.vstack([label_dataset, label])
                    intensity_dataset = np.vstack([intensity_dataset, intensity])
            print('label_dataset',label_dataset.shape,label_dataset)
            print('intensity_dataset',intensity_dataset.shape,intensity_dataset)

            self.label_dataset = label_dataset
            self.intensity_dataset = intensity_dataset.astype(np.float32)
            self.N = len(self.intensity_dataset[1])

        # data_list = IO.read_data_list(list_path)
        # for i, group in enumerate(data_list):
        #     batch_laplacian, batch_intensity, batch_label = IO.read_pickle_data(root, group)
        #     if not isinstance(batch_laplacian, scipy.sparse.coo.coo_matrix) \
        #             or not isinstance(batch_intensity, np.ndarray) \
        #             or not isinstance(batch_label, np.ndarray):
        #         raise NotImplementedError()
        #
        #     if i == 0:
        #         laplacian_dataset = batch_laplacian
        #         label_dataset = batch_label
        #         intensity_dataset =  batch_intensity
        #     else:
        #         laplacian_dataset = scipy.sparse.vstack([laplacian_dataset, batch_laplacian])
        #
        #         label_dataset = np.hstack([label_dataset, batch_label])
        #
        #         intensity_dataset = np.vstack([intensity_dataset, batch_intensity])
        #
        # print(laplacian_dataset.shape)
        # print(label_dataset.shape)
        # print(intensity_dataset.shape)
        #
        # self.laplacian_dataset = laplacian_dataset
        # self.label_dataset = label_dataset
        # self.intensity_dataset = intensity_dataset.astype(np.float32)
        # self.N = len(self.intensity_dataset[1])

    def __len__(self):
        return len(self.label_dataset)

    def get_example(self, i):
        # laplacian = self.laplacian_dataset.tocsr()[i].todense()
        # laplacian = np.array(laplacian, dtype=np.float32).reshape(self.N, self.N)
        intensity = self.intensity_dataset[i].reshape(self.N, -1)
        label = self.label_dataset[i]

        return label, intensity


class MNISTDataset(chainer.dataset.DatasetMixin):
    def __init__(self, datasets):
        print(datasets.shape)

        data=datasets[:,0]
        label=datasets[:,1]



        ###################################################
        def grid(m, dtype=np.float32):
            """Return the embedding of a grid graph."""
            # Adapted from https://github.com/mdeff/cnn_graph/blob/master/lib/graph.py
            M = m ** 2
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
            assert M, k == idx.shape
            assert dist.min() >= 0

            # Weights.
            sigma2 = np.mean(dist[:, -1]) ** 2
            dist = np.exp(- dist ** 2 / sigma2)

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


        ####################################################
        A = grid_graph(28)
        print(data.shape)
        # data=data.reshape(len(data),784)

        print(data[0].shape)
        label=datasets[1]

        """
        """
        data_list = IO.read_data_list(list_path)
        for i, group in enumerate(data_list):
            batch_laplacian, batch_intensity, batch_label = IO.read_pickle_data(root, group)
            if not isinstance(batch_laplacian, scipy.sparse.coo.coo_matrix) \
                    or not isinstance(batch_intensity, np.ndarray) \
                    or not isinstance(batch_label, np.ndarray):
                raise NotImplementedError()

            if i == 0:
                laplacian_dataset = batch_laplacian
                label_dataset = batch_label
                intensity_dataset = batch_intensity
            else:
                laplacian_dataset = scipy.sparse.vstack([laplacian_dataset, batch_laplacian])

                label_dataset = np.hstack([label_dataset, batch_label])

                intensity_dataset = np.vstack([intensity_dataset, batch_intensity])



        self.laplacian_dataset = laplacian_dataset
        self.label_dataset = label_dataset
        self.intensity_dataset = intensity_dataset.astype(np.float32)
        self.N = len(self.intensity_dataset[1])

    def __len__(self):
        return len(self.label_dataset)

    def get_example(self, i):
        laplacian = self.laplacian_dataset.tocsr()[i].todense()
        laplacian = np.array(laplacian, dtype=np.float32).reshape(self.N, self.N)
        intensity = self.intensity_dataset[i].reshape(self.N, -1)
        label = self.label_dataset[i]

        return label, intensity, laplacian
