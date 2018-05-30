#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

Original code with Chainer:
https://github.com/soskek/efficient_softmax

"""
from __future__ import division
from __future__ import print_function
import argparse

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainer import function_node
import chainer.functions
from chainer.utils import type_check

import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.functions.activation import log_softmax
from chainer.utils import type_check
from chainer import variable


def _broadcast_to(array, shape):
    if hasattr(numpy, "broadcast_to"):
        return numpy.broadcast_to(array, shape)
    dummy = numpy.empty(shape, array.dtype)
    return numpy.broadcast_arrays(array, dummy)[0]


def _check_class_weight_option(class_weight):
    if class_weight is not None:
        if class_weight.ndim != 1:
            raise ValueError('class_weight.ndim should be 1')
        if class_weight.dtype.kind != 'f':
            raise ValueError('The dtype of class_weight should be \'f\'')
        if isinstance(class_weight, variable.Variable):
            raise ValueError('class_weight should be a numpy.ndarray or '
                             'cupy.ndarray, not a chainer.Variable')


def _check_reduce_option(reduce):
    if reduce not in ('mean', 'no'):
        raise ValueError(
            "only 'mean' and 'no' are valid for 'reduce', but '%s' is "
            'given' % reduce)


def _check_input_values(x, t, ignore_label):
    # Extract the raw ndarray as Variable.__ge__ is not implemented.
    # We assume that t is already an ndarray.
    if isinstance(x, variable.Variable):
        x = x.data

    if not (((0 <= t) &
             (t < x.shape[1])) |
            (t == ignore_label)).all():
        msg = ('Each label `t` need to satisfy '
               '`0 <= t < x.shape[1] or t == %d`' % ignore_label)
        raise ValueError(msg)


class AdaptiveSoftmaxOutput(function.Function):

    normalize = True

    def __init__(self, cutoff, normalize=True,
                 ignore_label=-1, reduce='mean',
                 output_all=False):
        self.cutoff = cutoff
        self.normalize = normalize
        self.ignore_label = ignore_label
        _check_reduce_option(reduce)
        self.reduce = reduce
        self.output_all = output_all

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() >= 4)
        x_type, t_type = in_types[:2]
        rest = len(in_types) - 2
        Ws_types = in_types[2: 2 + (rest - 1) // 2 + 1]
        Rs_types = in_types[2 + (rest - 1) // 2 + 1:]

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype == numpy.int32,
            t_type.ndim == x_type.ndim - 1,

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[1:],
        )
        for i in six.moves.range(len(Ws_types)):
            type_check.expect(
                x_type.dtype == Ws_types[i].dtype,
                x_type.shape[1] >= Ws_types[i].shape[1],
                Ws_types[i].ndim == 2,
            )
            if i != len(Ws_types) - 1:
                type_check.expect(
                    x_type.dtype == Rs_types[i].dtype,
                    x_type.shape[1] == Rs_types[i].shape[1],
                    x_type.shape[1] >= Rs_types[i].shape[0],
                    Rs_types[i].ndim == 2,
                )

    def linear(self, x, W):
        y = x.dot(W.T).astype(x.dtype, copy=False)
        return y

    def backward_linear(self, x, W, gy):
        gx = gy.dot(W).astype(x.dtype, copy=False).reshape(x.shape)
        gW = gy.T.dot(x).astype(W.dtype, copy=False)
        return gx, gW

    def backward_log_softmax(self, x, y, gy):
        if cuda.cudnn_enabled:
            cudnn = cuda.cudnn
            libcudnn = cuda.cuda.cudnn
            _algorithm = libcudnn.CUDNN_SOFTMAX_LOG
            _mode = libcudnn.CUDNN_SOFTMAX_MODE_CHANNEL

        xp = cuda.get_array_module(x)
        if xp is not numpy and chainer.should_use_cudnn('>=auto', 3000):
            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            handle = cudnn.get_handle()
            gx = xp.empty(x.shape, dtype=x.dtype)
            gx_cube = gx.reshape(gx.shape[:2] + (-1, 1))
            desc = cudnn.create_tensor_descriptor(gx_cube)
            libcudnn.softmaxBackward(
                handle, _algorithm, _mode, one.data, desc.value,
                y.data.ptr, desc.value, gy.data.ptr, zero.data,
                desc.value, gx.data.ptr)
        else:
            gx = gy - xp.exp(y) * gy.sum(axis=1, keepdims=True)

        return gx

    def forward(self, inputs):
        x, t = inputs[:2]
        rest = len(inputs) - 2
        head_W, Ws = inputs[2], inputs[3:2 + (rest - 1) // 2 + 1]
        Rs = inputs[2 + (rest - 1) // 2 + 1:]
        n_tails = len(Rs)
        # minus_inf = -1024.
        minus_inf = -numpy.inf
        xp = cuda.get_array_module(x)

        if chainer.is_debug():
            _check_input_values(x, t, self.ignore_label)

        self.retain_inputs(tuple(six.moves.range(len(inputs))))

        cluster_hots = []
        for i in six.moves.range(1, n_tails + 1):
            lower, upper = self.cutoff[i], self.cutoff[i + 1]
            in_cluster = xp.logical_and(lower <= t, t < upper)
            if self.output_all:
                in_cluster = xp.ones(
                    in_cluster.shape, dtype=in_cluster.dtype)
            cluster_hots.append(in_cluster)
        self.cluster_hots = cluster_hots

        self.head = self.linear(x, head_W)
        self.ls_head = log_softmax._log_softmax(self.head)
        self.reduced_xs = []
        self.tails = []
        self.ls_tails = []
        for i, in_cluster in enumerate(cluster_hots, start=1):
            tail_idx = i - 1
            if xp.any(in_cluster):
                reduced_x = self.linear(x[in_cluster], Rs[tail_idx])
                self.reduced_xs.append(reduced_x)
                out = self.linear(reduced_x, Ws[tail_idx])
                self.tails.append(out)
                ls_out = log_softmax._log_softmax(out)
                self.ls_tails.append(ls_out)
            else:
                self.reduced_xs.append(None)
                self.tails.append(None)
                self.ls_tails.append(None)

        n_head_out = head_W.shape[0] - n_tails
        n_out = n_head_out + sum(W.shape[0] for W in Ws)
        shape = (x.shape[0], n_out)

        log_y = xp.full(shape, minus_inf, dtype=x.dtype)

        log_y[:, :n_head_out] = self.ls_head[:, :n_head_out]
        for i, (in_cluster, tail) in enumerate(
                zip(cluster_hots, self.ls_tails), start=1):
            if tail is None:
                continue
            lower, upper = self.cutoff[i], self.cutoff[i + 1]

            tail_main = self.ls_head[:, n_head_out + i - 1]
            tail_main_in = xp.broadcast_to(
                tail_main[in_cluster][:, None], tail.shape)
            log_y[xp.nonzero(in_cluster)[0], lower:upper] = tail_main_in + tail
            not_in_cluster = xp.logical_not(in_cluster)
            log_y[xp.nonzero(not_in_cluster)[0],
                  lower] = tail_main[not_in_cluster]

        return log_y,

    def backward(self, inputs, grad_outputs):
        x, t = inputs[:2]
        g_log_p = grad_outputs[0]
        x, t = inputs[:2]
        rest = len(inputs) - 2
        head_W, Ws = inputs[2], inputs[3:2 + (rest - 1) // 2 + 1]
        Rs = inputs[2 + (rest - 1) // 2 + 1:]
        n_tails = len(Rs)
        xp = cuda.get_array_module(x)

        # add processing
        n_head_out = head_W.shape[0] - n_tails

        g_ls_head_out = g_log_p[:, :n_head_out]
        g_ls_tail_mains = []
        g_Ws = []
        g_Rs = []
        g_xs_from_reduced = []
        for i, (in_cluster, reduced_x, tail, ls_tail, W, R) in enumerate(
                zip(self.cluster_hots, self.reduced_xs,
                    self.tails, self.ls_tails, Ws, Rs), start=1):
            lower, upper = self.cutoff[i], self.cutoff[i + 1]
            if xp.any(in_cluster):
                g_ls_tail_mains.append(
                    g_log_p[:, lower:upper].sum(axis=1, keepdims=True))

                g_ls_tail = g_log_p[xp.nonzero(in_cluster)[0], lower:upper]
                g_tail = self.backward_log_softmax(
                    tail, ls_tail, g_ls_tail)

                g_reduced_x, g_W = self.backward_linear(reduced_x, W, g_tail)
                g_x_from_reduced, g_R = self.backward_linear(
                    x[in_cluster], R, g_reduced_x)
                g_Ws.append(g_W)
                g_Rs.append(g_R)
                g_xs_from_reduced.append(g_x_from_reduced)
            else:
                g_Ws.append(xp.zeros(W.shape, dtype=W.dtype))
                g_Rs.append(xp.zeros(R.shape, dtype=R.dtype))
                g_xs_from_reduced.append(0.)

        g_ls_head = xp.concatenate(
            [g_ls_head_out] + g_ls_tail_mains, axis=1)
        g_head = self.backward_log_softmax(
            self.head, self.ls_head, g_ls_head)
        g_x_from_head, g_head_W = self.backward_linear(x, head_W, g_head)

        g_x = g_x_from_head
        for i, (in_cluster, g_x_from_reduced) in enumerate(
                zip(self.cluster_hots, g_xs_from_reduced), start=1):
            g_x[in_cluster] += g_x_from_reduced
        # This should be kernel at once?
        # g_x = g_x_from_head + in_cluster * g_x_from_reduced + ...
        # in forward too.

        ret = [g_x, None, g_head_W] + g_Ws + g_Rs
        return tuple(ret)
# TOOD error check


class AdaptiveSoftmaxCrossEntropy(AdaptiveSoftmaxOutput):

    """Softmax activation followed by a cross entropy loss."""

    def forward(self, inputs):
        if any(isinstance(x, cuda.ndarray) for x in inputs):
            return self.forward_gpu(inputs)
        else:
            return self.forward_cpu(inputs)

    def backward(self, inputs, grad_outputs):
        if any(isinstance(x, cuda.ndarray) for x in inputs + grad_outputs):
            return self.backward_gpu(inputs, grad_outputs)
        else:
            return self.backward_cpu(inputs, grad_outputs)

    def forward_cpu(self, inputs):
        x, t = inputs[:2]
        log_y = super(AdaptiveSoftmaxCrossEntropy, self).forward(inputs)[0]
        self.y = numpy.exp(log_y)

        log_yd = numpy.rollaxis(log_y, 1)
        log_yd = log_yd.reshape(len(log_yd), -1)
        log_p = log_yd[numpy.maximum(t.ravel(), 0), numpy.arange(t.size)]

        log_p *= (t.ravel() != self.ignore_label)
        if self.reduce == 'mean':
            # deal with the case where the SoftmaxCrossEntropy is
            # unpickled from the old version
            if self.normalize:
                count = (t != self.ignore_label).sum()
            else:
                count = len(x)
            self._coeff = 1.0 / max(count, 1)

            y = log_p.sum(keepdims=True) * (-self._coeff)
            return y.reshape(()),
        else:
            return -log_p.reshape(t.shape),

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, t = inputs[:2]
        log_y = super(AdaptiveSoftmaxCrossEntropy, self).forward(inputs)[0]
        self.y = cupy.exp(log_y)

        if self.normalize:
            coeff = cupy.maximum(1, (t != self.ignore_label).sum())
        else:
            coeff = max(1, len(t))
        self._coeff = cupy.divide(1.0, coeff, dtype=x.dtype)

        log_y = cupy.rollaxis(log_y, 1, log_y.ndim)
        if self.reduce == 'mean':
            ret = cuda.reduce(
                'S t, raw T log_y, int32 n_channel, raw T coeff, '
                'S ignore_label',
                'T out',
                't == ignore_label ? T(0) : log_y[_j * n_channel + t]',
                'a + b', 'out = a * -coeff[0]', '0', 'crossent_fwd'
            )(t, log_y.reduced_view(), log_y.shape[-1],
              self._coeff, self.ignore_label)
        else:
            ret = cuda.elementwise(
                'S t, raw T log_y, int32 n_channel, T ignore', 'T out',
                '''
                if (t == ignore) {
                  out = 0;
                } else {
                  out = -log_y[i * n_channel + t];
                }
                ''',
                'softmax_crossent_no_reduce_fwd'
            )(t, log_y.reduced_view(), log_y.shape[-1], self.ignore_label)
            ret = ret.reshape(t.shape)
        return ret,

    def backward_cpu(self, inputs, grad_outputs):
        x, t = inputs[:2]

        gloss = grad_outputs[0]
        y = self.y.copy()

        g_log_p = y
        g_log_p[numpy.arange(len(t)), numpy.maximum(t, 0)] -= 1

        g_log_p *= (t != self.ignore_label).reshape((len(t), 1))

        if self.reduce == 'mean':
            g_log_p *= gloss * self._coeff
        else:
            g_log_p *= gloss[:, None]

        ret = super(AdaptiveSoftmaxCrossEntropy, self).backward(
            inputs, (g_log_p, ))
        return ret

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, t = inputs[:2]

        y = self.y
        gloss = grad_outputs[0]

        g_log_p = y
        g_log_p[cupy.arange(len(t)), cupy.maximum(t, 0)] -= 1

        g_log_p *= (t != self.ignore_label).reshape((len(t), 1))

        if self.reduce == 'mean':
            g_log_p *= gloss * self._coeff
        else:
            g_log_p *= gloss[:, None]

        ret = super(AdaptiveSoftmaxCrossEntropy, self).backward(
            inputs, (g_log_p, ))
        return ret


def adaptive_softmax_cross_entropy(
        x, t, Ws, Rs, cutoff, normalize=True,
        ignore_label=-1, reduce='mean', enable_double_backprop=False):
    """Computes cross entropy loss for pre-softmax activations.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable holding a multidimensional array whose element indicates
            hidden states: the first axis of the variable
            represents the number of samples, and the second axis represents
            the number of hidden units.
        Ws (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variables of weight matrices for word outputs.
            The first matrix is for the head.
            The rest matrices are for the tails in order.
        Rs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variables of weight matrices for reducing hidden units.
            The matrices are for the tails in order.
            The number of matrices must be ``len(Ws) - 1``.
        t (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable holding an :class:`numpy.int32` vector of ground truth
            labels. If ``t[i] == ignore_label``, corresponding ``x[i]`` is
            ignored.
        cutoff (list of int):
            Cutoff indices of clusters. e.g. [0, 2000, 10000, n_vocab]
        normalize (bool): If ``True``, this function normalizes the cross
            entropy loss across all instances. If ``False``, it only
            normalizes along a batch size.
        ignore_label (int): Label value you want to ignore. Its default value
            is ``-1``. See description of the argument `t`.
        reduce (str): A string that determines whether to reduce the loss
            values. If it is ``'mean'``, it computes the sum of the individual
            cross entropy and normalize it according to ``normalize`` option.
            If it is ``'no'``, this function computes cross entropy for each
            instance and does not normalize it (``normalize`` option is
            ignored). In this case, the loss value of the ignored instance,
            which has ``ignore_label`` as its target value, is set to ``0``.

    Returns:
        ~chainer.Variable: A variable holding a scalar array of the cross
        entropy loss.  If ``reduce`` is ``'mean'``, it is a scalar array.
        If ``reduce`` is ``'no'``, the shape is same as that of ``x``.

    """

    if enable_double_backprop:
        raise NotImplementedError()
    else:
        return AdaptiveSoftmaxCrossEntropy(
            cutoff, normalize=normalize,
            ignore_label=ignore_label,
            reduce=reduce)(
                x, t, *Ws, *Rs)


def adaptive_softmax_output(
        x, t, Ws, Rs, cutoff,
        output_all=False,
        enable_double_backprop=False):

    if enable_double_backprop:
        raise NotImplementedError()
    else:
        return AdaptiveSoftmaxOutput(
            cutoff, output_all=output_all)(x, t, *Ws, *Rs)


class AdaptiveSoftmaxOutputLayer(chainer.Chain):
    def __init__(self, n_units, n_vocab,
                 cutoff=[2000, 10000], reduce_k=4):
        super(AdaptiveSoftmaxOutputLayer, self).__init__()
        assert(all(c < n_vocab - 1 for c in cutoff))
        self.n_clusters = len(cutoff) + 1
        self.n_tails = self.n_clusters - 1

        cutoff.append(n_vocab)
        initializer = chainer.initializers._get_initializer(None)
        with self.init_scope():
            self.head = variable.Parameter(initializer=initializer)
            self.head.initialize((cutoff[0] + self.n_tails, n_units))

            tail_units = n_units
            for i in range(1, self.n_tails + 1):
                tail_units = tail_units // reduce_k
                n_comp_words = cutoff[i] - cutoff[i - 1]
                assert(tail_units > 0)
                assert(n_comp_words > 0)

                self.add_param('reduce{}'.format(i), initializer=initializer)
                getattr(self, 'reduce{}'.format(i)).initialize(
                    (tail_units, n_units))
                self.add_param('tail{}'.format(i), initializer=initializer)
                getattr(self, 'tail{}'.format(i)).initialize(
                    (n_comp_words, tail_units))

            cutoff = self.xp.array([0] + cutoff, dtype=np.int32)
            assert(len(cutoff) == self.n_clusters + 1)
            self.add_param('cutoff', cutoff.shape, dtype='f')
            self.cutoff.data[:] = cutoff

    def output(self, h, t=None):
        Ws = [self.head] + [getattr(self, 'tail{}'.format(i))
                            for i in range(1, self.n_tails + 1)]
        Rs = [getattr(self, 'reduce{}'.format(i))
              for i in range(1, self.n_tails + 1)]
        cutoff = self.cutoff.data.astype('i').tolist()
        # An error happens to cupy when 0-dim array idx is directly used.

        output_all = t is None
        if output_all:
            t = self.xp.zeros((h.shape[0], ), 'i')
        return adaptive_softmax_output(
            h, t, Ws, Rs, cutoff, output_all=output_all)

    def output_and_loss(self, h, t):
        Ws = [self.head] + [getattr(self, 'tail{}'.format(i))
                            for i in range(1, self.n_tails + 1)]
        Rs = [getattr(self, 'reduce{}'.format(i))
              for i in range(1, self.n_tails + 1)]
        cutoff = self.cutoff.data.astype('i').tolist()
        # An error happens to cupy when 0-dim array idx is directly used.
        return adaptive_softmax_cross_entropy(
            h, t, Ws, Rs, cutoff, normalize=False, reduce='mean')
