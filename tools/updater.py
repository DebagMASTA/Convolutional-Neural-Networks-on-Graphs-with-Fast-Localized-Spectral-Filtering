# coding:utf-8
"""
@auther tzw
@date 2018-6-15
"""
import os, sys, time
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
import SimpleITK as sitk
import argparse, yaml, shutil

parser = argparse.ArgumentParser()

parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                    help='base directory path of program files')
parser.add_argument('--out', '-o', default='results\\training',
                    help='Directory to output the result')
args = parser.parse_args()


class GCNNUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.CNN = kwargs.pop("models")
        super(GCNNUpdater, self).__init__(*args, **kwargs)

    def loss_softmax_cross_entropy(self, CNN, predict, ground_truth):
        """
        * @param CNN CNN
        * @param predict Output of CNN
        * @param ground_truth Ground truth label
        """
        # print(predict.shape)
        # print(ground_truth.shape)
        loss = -F.mean(F.log(predict + 1e-16) * ground_truth)

        chainer.report({"loss": loss}, CNN)  # mistery
        return loss

    def jaccard_index(predict, ground_truth):
        JI_numerator = 0.0
        JI_denominator = 0.0

        predict = predict.ravel()
        ground_truth = ground_truth.ravel()
        seg = (predict > 0.5)

        JI_numerator = (seg * ground_truth).sum()
        JI_denominator = (seg + ground_truth > 0).sum()

        return JI_numerator / JI_denominator

    def update_core(self):
        # load optimizer called "CNN"
        CNN_optimizer = self.get_optimizer("main")
        batch = self.get_iterator("main").next()  # iterator

        # iterator
        label, data = self.converter(batch, self.device)

        CNN = self.CNN


        predict = CNN(data)
        # predict, a ,b,c,d= CNN(data)

        CNN_optimizer.update(self.loss_softmax_cross_entropy, CNN, predict, label)
        # b = chainer.backends.cuda.to_cpu(a.data)
        # print('b.shape:',b.shape)
        #
        # img = sitk.GetImageFromArray(b[0][0])
        # img.SetSpacing([1.0, 1.0, 1.0])
        # sitk.WriteImage(img, (os.path.join(args.base, args.out+'\\featuremap.mhd') ))






