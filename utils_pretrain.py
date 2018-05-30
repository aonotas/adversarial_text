#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import division
from __future__ import print_function
import argparse
import collections
import io
import json
import os

import numpy as np

import chainer
from chainer import cuda
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import utils


def convert_xt_batch_seq(xt_batch_seq, gpu):
    batchsize = len(xt_batch_seq[0])
    seq_len = len(xt_batch_seq)
    xt_batch_seq = np.array(xt_batch_seq, 'i')
    # (bproplen, batch, 2)
    xt_batch_seq = convert.to_device(gpu, xt_batch_seq)
    xp = cuda.get_array_module(xt_batch_seq)
    x_seq_batch = xp.split(
        xt_batch_seq[:, :, 0].T.reshape(batchsize * seq_len),
        batchsize, axis=0)
    t_seq_batch = xp.split(
        xt_batch_seq[:, :, 1].T.reshape(batchsize * seq_len),
        batchsize, axis=0)
    return x_seq_batch, t_seq_batch


def count_words(dataset, alpha=0.4):
    counts = collections.defaultdict(int)
    for w in dataset:
        counts[w] += 1
    counts = [counts[i] for i in range(len(counts))]
    counts = np.array(counts, 'f')
    counts /= counts.sum()
    counts = counts ** alpha
    counts = counts.tolist()
    return counts


def tokenize_text(file_path, vocab={}, update_vocab=False):
    tokens = []
    unk_id = vocab['<unk>']
    with io.open(file_path, encoding='utf-8') as f:
        for line in f:
            words = line.split() + ['<eos>']
            for word in words:
                if update_vocab:
                    if word not in vocab:
                        vocab[word] = len(vocab)
                tokens.append(vocab.get(word, unk_id))
    return tokens, vocab


def get_wikitext_words_and_vocab(name='wikitext-2', base_dir='datasets', vocab=None):
    assert(name in ['wikitext-2', 'wikitext-103'])
    base_dir2 = os.path.join(base_dir, name)
    predata_path = os.path.join(base_dir2, 'preprocessed_data.json')
    if os.path.exists(predata_path) and vocab is None:
        train, valid, test, vocab = json.load(open(predata_path))
    else:
        prepared_vocab = (vocab is not None)
        if not prepared_vocab:
            vocab = {'<eos>': 0, '<unk>': 1}
        train, vocab = tokenize_text(
            os.path.join(base_dir2, 'wiki.train.tokens'),
            vocab, update_vocab=not prepared_vocab)
        valid, _ = tokenize_text(
            os.path.join(base_dir2, 'wiki.valid.tokens'),
            vocab, update_vocab=False)
        test, _ = tokenize_text(
            os.path.join(base_dir2, 'wiki.test.tokens'),
            vocab, update_vocab=False)
        json.dump([train, valid, test, vocab], open(predata_path, 'w'))
    return train, valid, test, vocab

# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.


class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0
        # use -1 instead of None internally
        self._previous_epoch_detail = -1.

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different position in the original sequence. Each item is
        # represented by a pair of two word IDs. The first word is at the
        # "current" position, while the second word at the next position.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration
        cur_words = self.get_words()
        self._previous_epoch_detail = self.epoch_detail
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def get_words(self):
        # It returns a list of current words.
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.
