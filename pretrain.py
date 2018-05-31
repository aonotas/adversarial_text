
'''
Original code with Chainer:
https://github.com/soskek/efficient_softmax
'''

from __future__ import print_function
import argparse
import copy
import json
import os
import time

import numpy as np

import chainer
from chainer import cuda
from chainer.dataset import convert
import chainer.links as L
from chainer import serializers

import utils
import utils_pretrain
import lm_nets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--n-units-word', type=int, default=256,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--alpha_decay', type=float, default=0.9999)
    parser.add_argument('--share-embedding', action='store_true')
    parser.add_argument('--adaptive-softmax', action='store_true')
    parser.add_argument('--dataset', default='imdb',
                        choices=['imdb', 'elec', 'rotten', 'dbpedia', 'rcv1'])
    parser.add_argument('--vocab')
    parser.add_argument('--log-interval', type=int, default=500)
    parser.add_argument('--validation-interval', '--val-interval',
                        type=int, default=30000)
    parser.add_argument('--decay-if-fail', action='store_true')
    parser.add_argument('--use-full-vocab', action='store_true')
    parser.add_argument('--decay-every', action='store_true')
    parser.add_argument('--random-seed', type=int, default=1234, help='seed')
    parser.add_argument('--save-all', type=int, default=0, help='save_all')
    parser.add_argument('--norm-vecs', action='store_true')

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    if not os.path.isdir(args.out):
        os.mkdir(args.out)

    xp = cuda.cupy if args.gpu >= 0 else np
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        cuda.get_device(args.gpu).use()
        xp.random.seed(1234)

    def evaluate(raw_model, iter):
        model = raw_model.copy()  # to use different state
        model.reset_state()  # initialize state
        sum_perp = 0
        count = 0
        xt_batch_seq = []
        one_pack = args.batchsize * args.bproplen * 2
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            for batch in copy.copy(iter):
                xt_batch_seq.append(batch)
                count += 1
                if len(xt_batch_seq) >= one_pack:
                    x_seq_batch, t_seq_batch = utils_pretrain.convert_xt_batch_seq(
                        xt_batch_seq, args.gpu)
                    loss = model.forward_seq_batch(
                        x_seq_batch, t_seq_batch, normalize=1.)
                    sum_perp += loss.data
                    xt_batch_seq = []
            if xt_batch_seq:
                x_seq_batch, t_seq_batch = utils_pretrain.convert_xt_batch_seq(
                    xt_batch_seq, args.gpu)
                loss = model.forward_seq_batch(
                    x_seq_batch, t_seq_batch, normalize=1.)
                sum_perp += loss.data
        return np.exp(float(sum_perp) / count)

    if args.vocab:
        vocab = json.load(open(args.vocab))
        print('vocab is loaded', args.vocab)
        print('vocab =', len(vocab))
    else:
        vocab = None

    if args.dataset == 'imdb':
        import sys
        sys.path.append('../')
        lower = False
        min_count = 1
        ignore_unk = 1
        vocab_obj, _, lm_data, t_vocab = utils.load_dataset_imdb(
            include_pretrain=True, lower=lower, min_count=min_count,
            ignore_unk=ignore_unk, add_labeld_to_unlabel=True)
        if vocab is None:
            vocab, vocab_count = vocab_obj
        n_class = 2
        (lm_train_dataset, lm_dev_dataset) = lm_data

        train = lm_train_dataset[:]
        val = lm_dev_dataset[:]
        test = lm_dev_dataset[:]
        n_vocab = len(vocab)

    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]
    print('#train tokens =', len(train))
    print('#valid tokens =', len(val))
    print('#test tokens =', len(test))
    print('#vocab =', n_vocab)

    # Create the dataset iterators
    train_iter = utils_pretrain.ParallelSequentialIterator(train, args.batchsize)
    val_iter = utils_pretrain.ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = utils_pretrain.ParallelSequentialIterator(test, 1, repeat=False)

    # Prepare an RNNLM model
    model = lm_nets.RNNForLM(n_vocab, args.unit, args.layer, args.dropout,
                          share_embedding=args.share_embedding,
                          adaptive_softmax=args.adaptive_softmax,
                          n_units_word=args.n_units_word)

    if args.norm_vecs:
        print('#norm_vecs')
        vocab_freq = np.array([float(vocab_count.get(w, 1)) for w, idx in
                               sorted(vocab.items(), key=lambda x: x[1])], dtype=np.float32)
        vocab_freq = vocab_freq / np.sum(vocab_freq)
        vocab_freq = vocab_freq.astype(np.float32)
        vocab_freq = vocab_freq[..., None]
        freq = vocab_freq
        print('freq:')
        print(freq)
        print('#norm_vecs...')
        word_embs = model.embed.W.data
        print('norm(word_embs):')
        print(np.linalg.norm(word_embs, axis=1).reshape(-1, 1))
        mean = np.sum(freq * word_embs, axis=0)
        print('mean:{}'.format(mean.shape))
        var = np.sum(freq * np.power(word_embs - mean, 2.), axis=0)
        stddev = np.sqrt(1e-6 + var)
        print('var:{}'.format(var.shape))
        print('stddev:{}'.format(stddev.shape))

        word_embs_norm = (word_embs - mean) / stddev
        word_embs_norm = word_embs_norm.astype(np.float32)
        print('word_embs_norm:{}'.format(word_embs_norm))
        print(word_embs_norm)
        print('norm(word_embs_norm):')
        print(np.linalg.norm(word_embs_norm, axis=1).reshape(-1, 1))
        model.embed.W.data[:] = word_embs_norm
        print('#done')

    if args.gpu >= 0:
        model.to_gpu()

    # Set up an optimizer
    # optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer = chainer.optimizers.Adam(alpha=args.alpha)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(1e-6))

    sum_perp = 0
    count = 0
    iteration = 0
    is_new_epoch = 0
    best_val_perp = 1000000.
    best_epoch = 0
    start = time.time()

    log_interval = args.log_interval
    validation_interval = args.validation_interval
    print('iter/epoch', len(train) // (args.bproplen * args.batchsize))
    print('Training start')
    while train_iter.epoch < args.epoch:
        iteration += 1
        xt_batch_seq = []
        if np.random.rand() < 0.01:
            model.reset_state()

        for i in range(args.bproplen):
            batch = train_iter.__next__()
            xt_batch_seq.append(batch)
            is_new_epoch += train_iter.is_new_epoch
            count += 1
        x_seq_batch, t_seq_batch = utils_pretrain.convert_xt_batch_seq(
            xt_batch_seq, args.gpu)
        loss = model.forward_seq_batch(
            x_seq_batch, t_seq_batch, normalize=args.batchsize)

        sum_perp += loss.data
        model.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters
        del loss

        if iteration % log_interval == 0:
            time_str = time.strftime('%Y-%m-%d %H-%M-%S')
            mean_speed = (count // args.bproplen) / (time.time() - start)
            print('\ti {:}\tperp {:.3f}\t\t| TIME {:.3f}i/s ({})'.format(
                iteration, np.exp(float(sum_perp) / count), mean_speed, time_str))
            sum_perp = 0
            count = 0
            start = time.time()

        if args.decay_every and args.alpha_decay > 0.0:
            optimizer.hyperparam.alpha *= args.alpha_decay  # 0.9999

        if is_new_epoch:
            # if iteration % validation_interval == 0:
            tmp = time.time()
            if args.save_all:
                model_name = 'iter_{}.model'.format(train_iter.epoch)
                serializers.save_npz(os.path.join(args.out, model_name), model)

            val_perp = evaluate(model, val_iter)
            time_str = time.strftime('%Y-%m-%d %H-%M-%S')
            print('Epoch {:}: val perp {:.3f}\t\t| TIME [{:.3f}s] ({})'.format(
                train_iter.epoch, val_perp, time.time() - tmp, time_str))
            if val_perp < best_val_perp:
                best_val_perp = val_perp
                best_epoch = train_iter.epoch
                serializers.save_npz(os.path.join(
                args.out, 'best.model'), model)
            elif args.decay_if_fail:
                if hasattr(optimizer, 'alpha'):
                    optimizer.alpha *= 0.5
                    optimizer.alpha = max(optimizer.alpha, 1e-7)
                else:
                    optimizer.lr *= 0.5
                    optimizer.lr = max(optimizer.lr, 1e-7)
            start += (time.time() - tmp)
            if not args.decay_if_fail:

                if args.alpha_decay > 0.0:
                    optimizer.hyperparam.alpha *= args.alpha_decay  # 0.9999
                else:
                    if hasattr(optimizer, 'alpha'):
                        optimizer.alpha *= 0.85
                    else:
                        optimizer.lr *= 0.85
            print('\t*lr = {:.8f}'.format(
                optimizer.alpha if hasattr(optimizer, 'alpha') else optimizer.lr))
            is_new_epoch = 0

    # Evaluate on test dataset
    print('test')
    print('load best model at epoch {}'.format(best_epoch))
    print('valid perplexity: {}'.format(best_val_perp))
    serializers.load_npz(os.path.join(args.out, 'best.model'), model)
    test_perp = evaluate(model, test_iter)
    print('test perplexity: {}'.format(test_perp))


if __name__ == '__main__':
    main()
