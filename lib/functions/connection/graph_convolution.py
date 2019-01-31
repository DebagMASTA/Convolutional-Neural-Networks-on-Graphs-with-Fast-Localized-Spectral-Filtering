#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import six

import chainer
from chainer import backend
from chainer import cuda
from chainer.cuda import cupy
from chainer import function
from chainer.utils import type_check
"""add1213"""
import chainer.functions
from chainer.utils import argument
from chainer.backends import intel64
from chainer import function_node
from chainer import configuration
from chainer.utils import conv

def chebyshev_matvec_cpu(C, x, K, n_batch, LmI):
    # print('C.shape',C.shape)

    C[:, 0] = x.transpose((0, 2, 1))  # (n_batch, N, c_in)
    # NOTE(tommi): scipy.sparse does not support sparse tensordot,
    # so have to use a for loop, although inefficient.
    if K > 1:
        for i in six.moves.range(n_batch):
            C[i, 1] = LmI.dot(C[i, 0])
    for k in six.moves.range(2, K):
        for i in six.moves.range(n_batch):
            C[i, k] = 2 * LmI.dot(C[i, k - 1]) - C[i, k - 2]


if chainer.cuda.available:
    # Computes y = Lx
    # x will be flattened in C-order
    # y will be flattened in C-order
    csr_matvec = cupy.ElementwiseKernel(
        'I p, raw T data, raw I indices, raw I indptr, raw T x',
        'T y',
        '''
            y = 0;
            int n_cols = _ind.size() / p;
            int row_idx = i / n_cols;
            int col_idx = i % n_cols;
            for(I j = indptr[row_idx]; j < indptr[(row_idx+1)]; j++) {
                y += data[j] * x[indices[j] * n_cols + col_idx];
            }
            ''',
        'csr_matvec'
    )

    def chebyshev_matvec_gpu(C, x, K, n_batch,
                             LmI_data, LmI_indices, LmI_indptr):
        C[0] = x.transpose((2, 1, 0))
        N = C.shape[1]
        if K > 1:
            csr_matvec(N, LmI_data, LmI_indices, LmI_indptr, C[0], C[1])
        for k in six.moves.range(2, K):
            csr_matvec(N, LmI_data, LmI_indices, LmI_indptr, C[k - 1], C[k])
            C[k] = 2 * C[k] - C[k - 2]


class GraphConvolutionFunction(function.Function):

    def __init__(self, L, K,**kwargs):
        # NOTE(tommi): It is very important that L
        # is a normalized Graph Laplacian matrix.
        # Otherwise, this will not work.
        dilate, groups = argument.parse_kwargs(
            kwargs, ('dilate', 1), ('groups', 1),
            deterministic="deterministic argument is not supported anymore. "
                          "Use chainer.using_config('cudnn_deterministic', value) context "
                          "where value is either `True` or `False`.",
            requires_x_grad="requires_x_grad argument is not supported "
                            "anymore. Just remove the argument. Note that whether to compute "
                            "the gradient w.r.t. x is automatically decided during "
                            "backpropagation.")

        I = scipy.sparse.identity(L.shape[0], format='csr', dtype=L.dtype)
        self.LmI = L - I
        self.LmI_tuple = (self.LmI.data, self.LmI.indices, self.LmI.indptr)

        self.K = K

        self.groups = groups

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 3,
            w_type.ndim == 3,
            x_type.shape[1] == w_type.shape[1] * self.groups,
        )

        """add1213"""
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

        # # if n_in.eval() == 3:
        # if n_in == 3:
        #     b_type = in_types[2]
        #     type_check.expect(
        #         b_type.dtype == x_type.dtype,
        #         b_type.ndim == 1,
        #         b_type.shape[0] == w_type.shape[0],
        #     )

    def to_cpu(self):
        self.LmI_tuple = tuple(map(cuda.to_cpu, self.LmI_tuple))

    def to_gpu(self, device=None):
        with cuda.get_device(device):
            self.LmI_tuple = tuple(map(cuda.to_gpu, self.LmI_tuple))

    # def forward_cpu(self, inputs):
    #     print('abcde')
    #     x, W = inputs[:2]
    #     n_batch, c_in, N = x.shape#org?
    #     b = inputs[2] if len(inputs) == 3 else None
    #
    #     K = self.K
    #     if x.dtype != self.LmI.dtype:
    #         self.LmI = self.LmI.astype(x.dtype)
    #
    #     C = np.empty((n_batch, K, N, c_in), dtype=x.dtype)
    #     print('abcde')
    #     chebyshev_matvec_cpu(C, x, K, n_batch, self.LmI)
    #     C = C.transpose((0, 3, 1, 2))
    #     self.C = C
    #     y = np.tensordot(C, W, ((1, 2), (1, 2)))
    #
    #     if b is not None:
    #         y += b
    #
    #     return np.rollaxis(y, 2, 1),  # y.shape = (n_batch, c_out, N)
    def forward_cpu(self, inputs):
        # print('abcde')
        # print('inputs',len(inputs),inputs[0].shape,inputs[1].shape,inputs[2].shape,inputs)
        # x, W = inputs[:2]
        self.retain_inputs((0, 1))  # retain only x and W
        if len(inputs) == 2:
            (x, W), b = inputs, None
        else:
            x, W, b = inputs
        n_batch, c_in, N = x.shape#org?
        # b = inputs[2] if len(inputs) == 3 else None

        K = self.K
        if x.dtype != self.LmI.dtype:
            self.LmI = self.LmI.astype(x.dtype)

        C = np.empty((n_batch, K, N, c_in), dtype=x.dtype)
        # print('abcde')
        chebyshev_matvec_cpu(C, x, K, n_batch, self.LmI)
        C = C.transpose((0, 3, 1, 2))
        self.C = C
        # print('W',W.shape,W)
        y = np.tensordot(C, W, ((1, 2), (1, 2)))
        # print('y',y.shape)

        if b is not None:
            y += b

        return np.rollaxis(y, 2, 1),  # y.shape = (n_batch, c_out, N)

    """add1213"""

    def _forward_ideep(self, x, W, b):
        out_c, input_c, kh, kw = W.shape
        n, c, h, w = x.shape

        out_h, out_w = self._get_out_size((x, W))
        pd = (self.sy * (out_h - 1)
              + (kh + (kh - 1) * (self.dy - 1)) - h - self.ph)
        pr = (self.sx * (out_w - 1)
              + (kw + (kw - 1) * (self.dx - 1)) - w - self.pw)
        param = intel64.ideep.convolution2DParam(
            (n, out_c, out_h, out_w),
            self.dy, self.dx,
            self.sy, self.sx,
            self.ph, self.pw,
            pd, pr)
        y = intel64.ideep.convolution2D.Forward(
            intel64.ideep.array(x),
            intel64.ideep.array(W),
            intel64.ideep.array(b) if b is not None else None,
            param)
        return y,


    def forward_gpu(self, inputs):
        x, W = inputs[:2]
        n_batch, c_in, N = x.shape
        b = inputs[2] if len(inputs) == 3 else None
        xp = cuda.get_array_module(x)
        with cuda.get_device(x.data):
            K = self.K
            LmI_data, LmI_indices, LmI_indptr = self.LmI_tuple

            if x.dtype != LmI_data.dtype:
                LmI_data = LmI_data.astype(x.dtype)

            C = xp.empty((K, N, c_in, n_batch), dtype=x.dtype)
            chebyshev_matvec_gpu(C, x, K, n_batch,
                                 LmI_data, LmI_indices, LmI_indptr)

            C = C.transpose((3, 2, 0, 1))
            self.C = C
            y = xp.tensordot(C, W, ((1, 2), (1, 2)))

            if b is not None:
                y += b

            return xp.rollaxis(y, 2, 1),  # y.shape = (n_batch, c_out, N)

    def backward_cpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]

        n_batch, c_in, N = x.shape
        c_out = gy.shape[1]

        gW = np.tensordot(gy, self.C, ((0, 2), (0, 3))
                          ).astype(W.dtype, copy=False)

        K = self.K
        if x.dtype != self.LmI.dtype:
            self.LmI = self.LmI.astype(x.dtype)

        C = np.empty((n_batch, K, N, c_out), dtype=x.dtype)
        chebyshev_matvec_cpu(C, gy, K, n_batch, self.LmI)
        C = C.transpose((0, 3, 1, 2))
        gx = np.tensordot(C, W, ((1, 2), (0, 2)))
        gx = np.rollaxis(gx, 2, 1)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0, 2))
            return gx, gW, gb

    def backward_gpu(self, inputs, grad_outputs):
        x, W = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        gy = grad_outputs[0]
        xp = cuda.get_array_module(x)
        with cuda.get_device(x.data):
            n_batch, c_in, N = x.shape
            c_out = gy.shape[1]

            gW = xp.tensordot(gy, self.C, ((0, 2), (0, 3))
                              ).astype(W.dtype, copy=False)

            K = self.K
            LmI_data, LmI_indices, LmI_indptr = self.LmI_tuple

            if x.dtype != LmI_data.dtype:
                LmI_data = LmI_data.astype(x.dtype)

            C = xp.empty((K, N, c_out, n_batch), dtype=x.dtype)
            chebyshev_matvec_gpu(C, gy, K, n_batch,
                                 LmI_data, LmI_indices, LmI_indptr)
            C = C.transpose((3, 2, 0, 1))
            gx = xp.tensordot(C, W, ((1, 2), (0, 2)))
            gx = xp.rollaxis(gx, 2, 1)

        if b is None:
            return gx, gW
        else:
            gb = gy.sum(axis=(0, 2))
            return gx, gW, gb

"""add1213"""
class Convolution2DGradW(function_node.FunctionNode):

    def __init__(self, conv2d):
        W_node = conv2d.inputs[1]
        self.kh, self.kw = W_node.shape[2:]
        self.sy = conv2d.sy
        self.sx = conv2d.sx
        self.ph = conv2d.ph
        self.pw = conv2d.pw
        self.dy = conv2d.dy
        self.dx = conv2d.dx
        self.cover_all = conv2d.cover_all
        self.W_dtype = W_node.dtype
        self.groups = conv2d.groups
        self._use_ideep = conv2d._use_ideep

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs

        if self.groups > 1:
            return self._forward_grouped_convolution(x, gy)
        else:
            return self._forward_cpu_core(x, gy)

    def _forward_cpu_core(self, x, gy):
        if self._use_ideep:
            return self._forward_ideep(x, gy)

        # NumPy raises an error when the array is not contiguous.
        # See: https://github.com/chainer/chainer/issues/2744
        # TODO(niboshi): Remove this code when NumPy is fixed.
        if (not (gy.flags.c_contiguous or gy.flags.f_contiguous) and
                1 in gy.shape):
            gy = np.ascontiguousarray(gy)

        col = conv.im2col_cpu(
            x, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        gW = np.tensordot(gy, col, ((0, 2, 3), (0, 4, 5))
                             ).astype(self.W_dtype, copy=False)
        return gW,

    def _forward_ideep(self, x, gy):
        n, input_c, h, w = x.shape
        n, out_c, out_h, out_w = gy.shape
        pd = (self.sy * (out_h - 1)
              + (self.kh + (self.kh - 1) * (self.dy - 1))
              - h - self.ph)
        pr = (self.sx * (out_w - 1)
              + (self.kw + (self.kw - 1) * (self.dx - 1))
              - w - self.pw)

        param = intel64.ideep.convolution2DParam(
            (out_c, input_c, self.kh, self.kw),
            self.dy, self.dx,
            self.sy, self.sx,
            self.ph, self.pw,
            pd, pr)
        gW = intel64.ideep.convolution2D.BackwardWeights(
            intel64.ideep.array(x),
            intel64.ideep.array(gy),
            param)
        return gW,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs

        use_cudnn = (
            chainer.should_use_cudnn('>=auto')
            and not self.cover_all
            and x.dtype == self.W_dtype
            and ((self.dy == 1 and self.dx == 1)
                 or (_cudnn_version >= 6000
                     and not configuration.config.cudnn_deterministic))
            and (self.groups <= 1 or _cudnn_version >= 7000)
        )

        if use_cudnn:
            # cuDNN implementation
            return self._forward_cudnn(x, gy)

        elif self.groups > 1:
            return self._forward_grouped_convolution(x, gy)

        else:
            return self._forward_gpu_core(x, gy)

    def _forward_gpu_core(self, x, gy):
        col = conv.im2col_gpu(
            x, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        gW = cuda.cupy.tensordot(gy, col, ((0, 2, 3), (0, 4, 5))
                                 ).astype(self.W_dtype, copy=False)
        return gW,

    def _forward_grouped_convolution(self, x, gy):
        # G: group count
        # N: batch size
        # kH, kW: kernel height, kernel width
        # iC, iH, iW: input channels, input height, input width
        # oC, oH, oW: output channels, output height, output width
        G = self.groups
        N, iC, iH, iW = x.shape
        _, oC, oH, oW = gy.shape  # _ == N
        kH = self.kh
        kW = self.kw
        iCg = iC // G
        oCg = oC // G

        # (N, iC, kH, kW, oH, oW)
        x = conv.im2col(x, kH, kW, self.sy, self.sx, self.ph, self.pw,
                        cover_all=self.cover_all, dy=self.dy, dx=self.dx)

        x = x.transpose(1, 2, 3, 0, 4, 5)  # (iC, kH, kW, N, oH, oW)
        x = x.reshape(G, iCg * kH * kW, N * oH * oW)
        x = x.transpose(0, 2, 1)  # (G, N*oH*oW, iCg*kH*kW)

        gy = gy.transpose(1, 0, 2, 3)  # (oC, N, oH, oW)
        gy = gy.reshape(G, oCg, N * oH * oW)

        # (G, oCg, iCg*kH*kW) = (G, oCg, N*oH*oW) @ (G, N*oH*oW, iCg*kH*kW)
        gW = _matmul(gy, x).astype(self.W_dtype, copy=False)
        gW = gW.reshape(oC, iCg, kH, kW)

        return gW,

    def _forward_cudnn(self, x, gy):
        _, out_c, out_h, out_w = gy.shape
        n, c, h, w = x.shape

        iC = c
        iCg = int(iC / self.groups)
        gW = cuda.cupy.empty((out_c, iCg, self.kh, self.kw),
                             dtype=self.W_dtype)
        pad = (self.ph, self.pw)
        stride = (self.sy, self.sx)
        dilation = (self.dy, self.dx)
        deterministic = configuration.config.cudnn_deterministic
        auto_tune = configuration.config.autotune
        tensor_core = configuration.config.use_cudnn_tensor_core
        cuda.cudnn.convolution_backward_filter(
            x, gy, gW, pad, stride, dilation, self.groups,
            deterministic=deterministic, auto_tune=auto_tune,
            tensor_core=tensor_core)

        return gW,

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ggW, = grad_outputs

        ret = []
        if 0 in indexes:
            xh, xw = x.shape[2:]
            gx = chainer.functions.deconvolution_2d(
                gy, ggW, stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                outsize=(xh, xw), dilate=(self.dy, self.dx),
                groups=self.groups)
            ret.append(gx)
        if 1 in indexes:
            ggy = convolution_2d(
                x, ggW, stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                cover_all=self.cover_all, dilate=(self.dy, self.dx),
                groups=self.groups)
            ret.append(ggy)

        return ret



def graph_convolution(x, W, L, K, b=None,**kwargs):
    """Graph convolution function.

    Graph convolutional layer using Chebyshev polynomials
    in the graph spectral domain.
    This is an implementation the graph convolution described in
    the following paper:

    Defferrard et al. "Convolutional Neural Networks on Graphs
    with Fast Localized Spectral Filtering", NIPS 2016.

    Notation:
    - :math:`n_batch` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output
      channels, respectively.
    - :math:`n_vertices` is the number of vertices in the graph.

    Args:
        x (~chainer.Variable): Input graph signal.
            Its shape is :math:`(n_batch, c_I, n_vertices)`.
        W (~chainer.Variable): Weight variable of shape
            :math:`c_O, c_I, K`.
        L (scipy.sparse.csr_matrix): Normalized graph Laplacian matrix
            that describes the graph.
        K (int): Polynomial order of the Chebyshev approximation.
        b (~chainer.Variable): Bias variable of length :math:`c_O` (optional)

    Returns:
        ~chainer.Variable: Output variable.

    If the bias vector is given, it is added to all spatial locations of the
    output of the graph convolution.

    """
    # func = GraphConvolutionFunction(L, K)
    # if b is None:
    #     return func(x, W)
    # else:
    #     return func(x, W, b)

    dilate, groups = argument.parse_kwargs(
        kwargs, ('dilate', 1), ('groups', 1),
        deterministic="deterministic argument is not supported anymore. "
                      "Use chainer.using_config('cudnn_deterministic', value) "
                      "context where value is either `True` or `False`.")

    fnode = GraphConvolutionFunction(L, K, dilate=dilate,
                                  groups=groups)
    if b is None:
        args = x, W
        # return func(x, W)
    else:
        args = x, W, b
    y, = fnode.apply(args)
    return y

