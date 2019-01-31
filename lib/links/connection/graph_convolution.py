#!/usr/bin/env python
#-*- coding: utf-8 -*-
#
# from chainer import cuda
# from chainer import initializers
# from chainer import link
#
# from lib.functions.connection import graph_convolution
# from lib import graph
# """added1212"""
# from chainer import variable
# from chainer.utils import argument
#
# class GraphConvolution(link.Link):
#     """Graph convolutional layer.
#
#     This link wraps the :func:`graph_convolution` function and holds the filter
#     weight and bias vector as parameters.
#
#     Args:
#         in_channels (int): Number of channels of input arrays. If ``None``,
#             parameter initialization will be deferred until the first forward
#             data pass at which time the size will be determined.
#         out_channels (int): Number of channels of output arrays.
#         A (~ndarray): Weight matrix describing the graph.
#         K (int): Polynomial order of the Chebyshev approximation.
#         wscale (float): Scaling factor of the initial weight.
#         bias (float): Initial bias value.
#         nobias (bool): If ``True``, then this link does not use the bias term.
#         initialW (4-D array): Initial weight value. If ``None``, then this
#             function uses to initialize ``wscale``.
#             May also be a callable that takes ``numpy.ndarray`` or
#             ``cupy.ndarray`` and edits its value.
#         initial_bias (1-D array): Initial bias value. If ``None``, then this
#             function uses to initialize ``bias``.
#             May also be a callable that takes ``numpy.ndarray`` or
#             ``cupy.ndarray`` and edits its value.
#
#     .. seealso::
#        See :func:`graph_convolution` for the definition of
#        graph convolution.
#
#     Attributes:
#         W (~chainer.Variable): Weight parameter.
#         b (~chainer.Variable): Bias parameter.
#
#     Graph convolutional layer using Chebyshev polynomials
#     in the graph spectral domain.
#
#     This link implements the graph convolution described in
#     the paper
#
#     Defferrard et al. "Convolutional Neural Networks on Graphs
#     with Fast Localized Spectral Filtering", NIPS 2016.
#
#     """
#
#     def __init__(self, in_channels, out_channels, A, K,
#                  nobias=False, initialW=None, initial_bias=None, **kwargs):
#         super(GraphConvolution, self).__init__()
#
#         """added1212"""
#         dilate, groups = argument.parse_kwargs(
#             kwargs, ('dilate', 1), ('groups', 1),
#             deterministic="deterministic argument is not supported anymore. "
#                           "Use chainer.using_config('cudnn_deterministic', value) "
#                           "context where value is either `True` or `False`.")
#
#
#         L = graph.create_laplacian(A)
#         print('K:',K)
#
#         self.K = K
#         self.out_channels = out_channels
#         self.groups = int(groups) #added1212
#
#         self._W_initializer = initializers._get_initializer(
#             initialW,
#         )
#
#         """added"""
#
#         with self.init_scope():
#             W_initializer = initializers._get_initializer(initialW)
#             self.W = variable.Parameter(W_initializer)
#             if in_channels is not None:
#                 self._initialize_params(in_channels)
#
#             if nobias:
#                 self.b = None
#             else:
#                 if initial_bias is None:
#                     initial_bias = 0
#                 bias_initializer = initializers._get_initializer(initial_bias)
#                 self.b = variable.Parameter(bias_initializer, out_channels)
#
#         # if in_channels is None:
#         #     self.add_uninitialized_param('W')
#         # else:
#         #     self._initialize_params(in_channels)
#         #
#         # if nobias:
#         #     self.b = None
#         # else:
#         #     if initial_bias is None:
#         #         initial_bias = bias
#         #     bias_initializer = initializers._get_initializer(initial_bias)
#         #     self.add_param('b', out_channels, initializer=bias_initializer)
#
#         self.func = graph_convolution.GraphConvolutionFunction(L, K)
#
#     def to_cpu(self):
#         super(GraphConvolution, self).to_cpu()
#         self.func.to_cpu()
#
#     def to_gpu(self, device=None):
#         with cuda.get_device(device):
#             super(GraphConvolution, self).to_gpu(device)
#             self.func.to_gpu(device)
#
#     """added"""
#
#     def _initialize_params(self, in_channels):
#         # kh, kw = _pair(self.ksize)
#         if self.out_channels % self.groups != 0:
#             raise ValueError('the number of output channels must be'
#                              ' divisible by the number of groups')
#         if in_channels % self.groups != 0:
#             raise ValueError('the number of input channels must be'
#                              ' divisible by the number of groups')
#         # W_shape = (self.out_channels, int(in_channels / self.groups), kh, kw)
#         # self.W.initialize(W_shape)
#
#     # def _initialize_params(self, in_channels):
#     #     W_shape = (self.out_channels, in_channels, self.K)
#     #     self.add_param('W', W_shape, initializer=self._W_initializer)
#
#     def __call__(self, x):
#         """Applies the graph convolutional layer.
#
#         Args:
#             x: (~chainer.Variable): Input graph signal.
#
#         Returns:
#             ~chainer.Variable: Output of the graph convolution.
#         """
#         # if self.has_uninitialized_params:
#
#         # if self.has_uninitialized_param:
#         #     with cuda.get_device(self._device_id):
#         #         self._initialize_params(x.shape[1])
#         # if self.b is None:
#         #     return self.func(x, self.W)
#         print('x.shape',x.shape)
#
#         if self.W.array is None:
#             self._initialize_params(x.shape[1])
#         else:
#             return self.func(x, self.W, self.b)
#
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import cuda
from chainer import initializers
from chainer import link

from lib.functions.connection import graph_convolution
from lib import graph


class GraphConvolution(link.Link):
    """Graph convolutional layer.
    This link wraps the :func:`graph_convolution` function and holds the filter
    weight and bias vector as parameters.
    Args:
        in_channels (int): Number of channels of input arrays. If ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        out_channels (int): Number of channels of output arrays.
        A (~ndarray): Weight matrix describing the graph.
        K (int): Polynomial order of the Chebyshev approximation.
        wscale (float): Scaling factor of the initial weight.
        bias (float): Initial bias value.
        nobias (bool): If ``True``, then this link does not use the bias term.
        initialW (4-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
    .. seealso::
       See :func:`graph_convolution` for the definition of
       graph convolution.
    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.
    Graph convolutional layer using Chebyshev polynomials
    in the graph spectral domain.
    This link implements the graph convolution described in
    the paper
    Defferrard et al. "Convolutional Neural Networks on Graphs
    with Fast Localized Spectral Filtering", NIPS 2016.
    """

    def __init__(self, in_channels, out_channels, A, K, wscale=1, bias=0,
                 nobias=False, initialW=None, initial_bias=None):
        super(GraphConvolution, self).__init__()

        L = graph.create_laplacian(A)

        self.K = K
        self.out_channels = out_channels

        self.wscale = wscale

        self._W_initializer = initializers._get_initializer(
            initialW)

        if in_channels is None:
            self.add_uninitialized_param('W')
        else:
            self._initialize_params(in_channels)

        if nobias:
            self.b = None
        else:
            if initial_bias is None:
                initial_bias = bias
            bias_initializer = initializers._get_initializer(initial_bias)
            self.add_param('b', out_channels, initializer=bias_initializer)

        self.func = graph_convolution.GraphConvolutionFunction(L, K)

    def to_cpu(self):
        super(GraphConvolution, self).to_cpu()
        self.func.to_cpu()

    def to_gpu(self, device=None):
        with cuda.get_device(device):
            super(GraphConvolution, self).to_gpu(device)
            self.func.to_gpu(device)

    def _initialize_params(self, in_channels):
        W_shape = (self.out_channels, in_channels, self.K)
        self.add_param('W', W_shape, initializer=self._W_initializer)

    def __call__(self, x):
        """Applies the graph convolutional layer.
        Args:
            x: (~chainer.Variable): Input graph signal.
        Returns:
            ~chainer.Variable: Output of the graph convolution.
        """
        # if self.has_uninitialized_params:
        #     with cuda.get_device(self._device_id):
        #         self._initialize_params(x.shape[1])
        # if self.b is None:
        #     return self.func(x, self.W)
        # else:
        #     return self.func(x, self.W, self.b)
        if self.W.array is None:
            self._initialize_params(x.shape[1])
        else:
            return self.func(x, self.W, self.b)
