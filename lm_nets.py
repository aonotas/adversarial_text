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
# from chainer import cuda
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.functions.connection import embed_id

from adaptive_softmax import AdaptiveSoftmaxOutputLayer


def get_normalized_vector(d, xp=None):
    shape = tuple(range(1, len(d.shape)))
    if xp is not None:
        d /= (1e-12 + xp.max(xp.abs(d), shape, keepdims=True))
        d /= xp.sqrt(1e-6 + xp.sum(d ** 2, shape, keepdims=True))
    else:
        d_term = 1e-12 + F.max(F.absolute(d), shape, keepdims=True)
        d /= F.broadcast_to(d_term, d.shape)
        d_term = F.sqrt(1e-6 + F.sum(d ** 2, shape, keepdims=True))
        d /= F.broadcast_to(d_term, d.shape)
    return d


def embed_seq_batch(embed, seq_batch, dropout=0., norm_vecs_one=False):
    batchsize = len(seq_batch)
    embs = F.dropout(embed(F.concat(seq_batch, axis=0)), ratio=dropout)
    if norm_vecs_one:
        embs = get_normalized_vector(embs, None)
    e_seq_batch = F.split_axis(embs, batchsize, axis=0)
    # [(len, ), ] x batchsize
    return e_seq_batch


class NormalOutputLayer(L.Linear):

    def __init__(self, *args, **kwargs):
        super(NormalOutputLayer, self).__init__(*args, **kwargs)

    def output_and_loss(self, h, t):
        logit = self(h)
        return F.softmax_cross_entropy(
            logit, t, normalize=False, reduce='mean')

    def output(self, h, t=None):
        return self(h)


class SharedOutputLayer(chainer.Chain):

    def __init__(self, W, bias=True, scale=True):
        super(SharedOutputLayer, self).__init__()
        self.W = W
        with self.init_scope():
            if bias:
                self.add_param('b', (W.shape[0], ), dtype='f')
                self.b.data[:] = 0.
            else:
                self.b = None
            if scale:
                self.add_param('scale', (1, ), dtype='f')
                self.scale.data[:] = 1.
            else:
                self.scale = None

    def output_and_loss(self, h, t):
        logit = self(h)
        return F.softmax_cross_entropy(
            logit, t, normalize=False, reduce='mean')

    def __call__(self, x):
        out = F.linear(x, self.W, self.b)
        if self.scale is not None:
            out *= F.broadcast_to(self.scale[None], out.shape)
        return out

    def output(self, h, t=None):
        return self(h)


class EmbedIDNormalized(chainer.links.EmbedID):

    ignore_label = -1

    def __init__(self, in_size, out_size, initialW=None, ignore_label=None,
                 vocab_freq=None, norm_to_one=False):
        super(EmbedIDNormalized, self).__init__(in_size, out_size, ignore_label=ignore_label)
        self.ignore_label = ignore_label

        with self.init_scope():
            if initialW is None:
                initialW = chainer.initializers.normal.Normal(1.0)
            self.W = chainer.variable.Parameter(initialW, (in_size, out_size))
        if len(vocab_freq.shape) == 1:
            vocab_freq = vocab_freq[..., None]
        self.vocab_freq = vocab_freq
        self.normalizedW = None
        self.norm_to_one = norm_to_one

    def norm_by_freq(self, freq):
        word_embs = self.W
        mean = F.sum(freq * word_embs, axis=0, keepdims=True)
        mean = F.broadcast_to(mean, word_embs.shape)
        var = F.sum(freq * ((word_embs - mean) ** 2), axis=0, keepdims=True)
        var = F.broadcast_to(var, word_embs.shape)

        stddev = F.sqrt(1e-6 + var)
        word_embs_norm = (word_embs - mean) / stddev
        return word_embs_norm

    def __call__(self, x):
        if self.normalizedW is None:
            if self.norm_to_one:
                self.normalizedW = F.normalize(self.vocab_freq * self.W)
            else:
                self.normalizedW = self.norm_by_freq(self.vocab_freq)

        return embed_id.embed_id(x, self.normalizedW, ignore_label=self.ignore_label)


# Definition of a recurrent net for language modeling
class RNNForLM(chainer.Chain):
    # TODO: nstep LSTM

    def __init__(self, n_vocab, n_units, n_layers=2, dropout=0.5,
                 share_embedding=False,
                 adaptive_softmax=False, vocab_freq=None, norm_to_one=False,
                 n_units_word=256):
        super(RNNForLM, self).__init__()
        with self.init_scope():
            # n_units_word = 256
            if vocab_freq is not None:
                self.embed = EmbedIDNormalized(
                    n_vocab, n_units_word, vocab_freq=vocab_freq, norm_to_one=norm_to_one)
            else:
                self.embed = L.EmbedID(n_vocab, n_units_word)
            self.rnn = L.NStepLSTM(n_layers, n_units_word, n_units, dropout)
            assert(not (share_embedding))
            if share_embedding:
                self.output = SharedOutputLayer(self.embed.W)
            elif adaptive_softmax:
                self.output = AdaptiveSoftmaxOutputLayer(
                    n_units, n_vocab,
                    cutoff=[2000, 10000], reduce_k=4)
            else:
                self.output = NormalOutputLayer(n_units, n_vocab)
        self.dropout = dropout
        self.n_units = n_units
        self.n_layers = n_layers
        self.norm_vecs_one = False

        for name, param in self.namedparams():
            if param.ndim != 1:
                # This initialization is applied only for weight matrices
                param.data[...] = np.random.uniform(
                    -0.1, 0.1, param.data.shape)

        self.loss = 0.
        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def __call__(self, x):
        raise NotImplementedError()

    def call_rnn(self, e_seq_batch):
        batchsize = len(e_seq_batch)
        if self.h is None:
            self.h = self.xp.zeros(
                (self.n_layers, batchsize, self.n_units), 'f')
        if self.c is None:
            self.c = self.xp.zeros(
                (self.n_layers, batchsize, self.n_units), 'f')
        self.h, self.c, y_seq_batch = self.rnn(self.h, self.c, e_seq_batch)
        return y_seq_batch

    def encode_seq_batch(self, x_seq_batch):
        e_seq_batch = embed_seq_batch(
            self.embed, x_seq_batch, dropout=self.dropout,
            norm_vecs_one=self.norm_vecs_one)
        y_seq_batch = self.call_rnn(e_seq_batch)
        return y_seq_batch

    def forward_seq_batch(self, x_seq_batch, t_seq_batch, normalize=None):
        y_seq_batch = self.encode_seq_batch(x_seq_batch)
        loss = self.output_and_loss_from_seq_batch(
            y_seq_batch, t_seq_batch, normalize)
        return loss

    def output_and_loss_from_seq_batch(self, y_seq_batch, t_seq_batch, normalize=None):
        y = F.concat(y_seq_batch, axis=0)
        y = F.dropout(y, ratio=self.dropout)
        t = F.concat(t_seq_batch, axis=0)
        loss = self.output.output_and_loss(y, t)
        if normalize is not None:
            loss *= 1. * t.shape[0] / normalize
        else:
            loss *= t.shape[0]
        return loss

    def output_from_seq_batch(self, y_seq_batch):
        y = F.concat(y_seq_batch, axis=0)
        y = F.dropout(y, ratio=self.dropout)
        return self.output(y)

    def pop_loss(self):
        loss = self.loss
        self.loss = 0.
        return loss
