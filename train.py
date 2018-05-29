#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import utils
import lm_nets

import random
import numpy as np
import pickle

import chainer
from chainer import cuda
from chainer import optimizers
import chainer.functions as F

import logging
logger = logging.getLogger(__name__)

chainer.config.use_cudnn = 'always'
to_cpu = chainer.cuda.to_cpu
to_gpu = chainer.cuda.to_gpu

from chainer import serializers
import net
import lm_nets

def main():

    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=32, help='learning minibatch size')
    parser.add_argument('--batchsize_semi', dest='batchsize_semi', type=int,
                        default=64, help='learning minibatch size')
    parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=30,
                        help='n_epoch')
    parser.add_argument('--pretrained_model', dest='pretrained_model',
                        type=str, default='', help='pretrained_model')
    parser.add_argument('--use_unlabled_to_vocab', dest='use_unlabled_to_vocab',
                        type=int, default=1, help='use_unlabled_to_vocab')
    parser.add_argument('--use_rational', dest='use_rational',
                        type=int, default=0, help='use_rational')
    parser.add_argument('--save_name', dest='save_name', type=str,
                        default='sentiment_model', help='save_name')
    parser.add_argument('--n_layers', dest='n_layers', type=int,
                        default=1, help='n_layers')
    parser.add_argument('--alpha', dest='alpha',
                        type=float, default=0.001, help='alpha')
    parser.add_argument('--alpha_decay', dest='alpha_decay',
                        type=float, default=0.0, help='alpha_decay')
    parser.add_argument('--clip', dest='clip',
                        type=float, default=5.0, help='clip')
    parser.add_argument('--debug_mode', dest='debug_mode',
                        type=int, default=0, help='debug_mode')
    parser.add_argument('--use_exp_decay', dest='use_exp_decay',
                        type=int, default=1, help='use_exp_decay')
    parser.add_argument('--load_trained_lstm', dest='load_trained_lstm',
                        type=str, default='', help='load_trained_lstm')
    parser.add_argument('--freeze_word_emb', dest='freeze_word_emb',
                        type=int, default=0, help='freeze_word_emb')
    parser.add_argument('--dropout', dest='dropout',
                        type=float, default=0.50, help='dropout')
    parser.add_argument('--use_adv', dest='use_adv',
                        type=int, default=0, help='use_adv')
    parser.add_argument('--xi_var', dest='xi_var',
                        type=float, default=1.0, help='xi_var')
    parser.add_argument('--xi_var_first', dest='xi_var_first',
                        type=float, default=1.0, help='xi_var_first')
    parser.add_argument('--lower', dest='lower',
                        type=int, default=1, help='lower')
    parser.add_argument('--nl_factor', dest='nl_factor', type=float,
                        default=1.0, help='nl_factor')
    parser.add_argument('--min_count', dest='min_count', type=int,
                        default=1, help='min_count')
    parser.add_argument('--ignore_unk', dest='ignore_unk', type=int,
                        default=0, help='ignore_unk')
    parser.add_argument('--use_semi_data', dest='use_semi_data',
                        type=int, default=0, help='use_semi_data')
    parser.add_argument('--add_labeld_to_unlabel', dest='add_labeld_to_unlabel',
                        type=int, default=1, help='add_labeld_to_unlabel')
    parser.add_argument('--norm_sentence_level', dest='norm_sentence_level',
                        type=int, default=1, help='norm_sentence_level')
    parser.add_argument('--dataset', default='imdb',
                        choices=['imdb', 'elec', 'rotten', 'dbpedia', 'rcv1'])
    parser.add_argument('--eval', dest='eval', type=int, default=0, help='eval')
    parser.add_argument('--emb_dim', dest='emb_dim', type=int,
                        default=256, help='emb_dim')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int,
                        default=1024, help='hidden_dim')
    parser.add_argument('--hidden_cls_dim', dest='hidden_cls_dim', type=int,
                        default=30, help='hidden_cls_dim')
    parser.add_argument('--adaptive_softmax', dest='adaptive_softmax',
                        type=int, default=1, help='adaptive_softmax')
    parser.add_argument('--random_seed', dest='random_seed', type=int,
                        default=1234, help='random_seed')
    parser.add_argument('--n_class', dest='n_class', type=int,
                        default=2, help='n_class')
    parser.add_argument('--word_only', dest='word_only', type=int,
                        default=0, help='word_only')


    args = parser.parse_args()
    batchsize = args.batchsize
    batchsize_semi = args.batchsize_semi
    print(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["CHAINER_SEED"] = str(args.random_seed)
    os.makedirs("models", exist_ok=True)

    if args.debug_mode:
        chainer.set_debug(True)

    use_unlabled_to_vocab = args.use_unlabled_to_vocab
    lower = args.lower == 1
    n_char_vocab = 1
    n_class = 2
    if args.dataset == 'imdb':
        vocab_obj, dataset, lm_dataset, train_vocab_size = utils.load_dataset_imdb(include_pretrain=use_unlabled_to_vocab, lower=lower,
                              min_count=args.min_count, ignore_unk=args.ignore_unk, use_semi_data=args.use_semi_data,
                              add_labeld_to_unlabel=args.add_labeld_to_unlabel)
        (train_x, train_x_len, train_y,
         dev_x, dev_x_len, dev_y,
         test_x, test_x_len, test_y) = dataset
        vocab, vocab_count = vocab_obj
        n_class = 2

    if args.use_semi_data:
        semi_train_x, semi_train_x_len = lm_dataset

    print('train_vocab_size:', train_vocab_size)

    vocab_inv = dict([(widx, w) for w, widx in vocab.items()])
    print('vocab_inv:', len(vocab_inv))

    xp = cuda.cupy if args.gpu >= 0 else np
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        xp.random.seed(args.random_seed)

    n_vocab = len(vocab)
    model = net.uniLSTM_VAT(n_vocab=n_vocab, emb_dim=args.emb_dim,
                            hidden_dim=args.hidden_dim,
                            use_dropout=args.dropout, n_layers=args.n_layers,
                            hidden_classifier=args.hidden_cls_dim,
                            use_adv=args.use_adv, xi_var=args.xi_var,
                            n_class=n_class, args=args)

    if args.pretrained_model != '':
        # load pretrained LM model
        pretrain_model = lm_nets.RNNForLM(n_vocab, 1024, args.n_layers, 0.50,
                                          share_embedding=False,
                                          adaptive_softmax=args.adaptive_softmax)
        serializers.load_npz(args.pretrained_model, pretrain_model)
        pretrain_model.lstm = pretrain_model.rnn
        model.set_pretrained_lstm(pretrain_model, word_only=args.word_only)

    if args.load_trained_lstm != '':
        serializers.load_hdf5(args.load_trained_lstm, model)

    if args.gpu >= 0:
        model.to_gpu()

    def evaluate(x_set, x_length_set, y_set):
        chainer.config.train = False
        chainer.config.enable_backprop = False
        iteration_list = range(0, len(x_set), batchsize)
        correct_cnt = 0
        total_cnt = 0.0
        predicted_np = []

        for i_index, index in enumerate(iteration_list):
            x = [to_gpu(_x) for _x in x_set[index:index + batchsize]]
            x_length = x_length_set[index:index + batchsize]
            y = to_gpu(y_set[index:index + batchsize])
            output = model(x, x_length)

            predict = xp.argmax(output.data, axis=1)
            correct_cnt += xp.sum(predict == y)
            total_cnt += len(y)

        accuracy = (correct_cnt / total_cnt) * 100.0
        chainer.config.enable_backprop = True
        return accuracy

    def get_unlabled(perm_semi, i_index):
        index = i_index * batchsize_semi
        sample_idx = perm_semi[index:index + batchsize_semi]
        x = [to_gpu(semi_train_x[_i]) for _i in sample_idx]
        x_length = [semi_train_x_len[_i] for _i in sample_idx]
        return x, x_length

    base_alpha = args.alpha
    opt = optimizers.Adam(alpha=base_alpha)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(args.clip))

    if args.freeze_word_emb:
        model.freeze_word_emb()

    prev_dev_accuracy = 0.0
    global_step = 0.0
    adv_rep_num_statics = {}
    adv_rep_pos_statics = {}

    if args.eval:
        dev_accuracy = evaluate(dev_x, dev_x_len, dev_y)
        log_str = ' [dev] accuracy:{}, length:{}'.format(str(dev_accuracy))
        logging.info(log_str)

        # test
        test_accuracy = evaluate(test_x, test_x_len, test_y)
        log_str = ' [test] accuracy:{}, length:{}'.format(str(test_accuracy))
        logging.info(log_str)


    for epoch in range(args.n_epoch):
        logging.info('epoch:' + str(epoch))
        # train
        model.cleargrads()
        chainer.config.train = True
        iteration_list = range(0, len(train_x), batchsize)

        # iteration_list_semi = range(0, len(semi_train_x), batchsize)
        perm = np.random.permutation(len(train_x))
        if args.use_semi_data:
            perm_semi = [np.random.permutation(len(semi_train_x)) for _ in range(2)]
            perm_semi = np.concatenate(perm_semi, axis=0)
            # print 'perm_semi:', perm_semi.shape
        def idx_func(shape):
            return xp.arange(shape).astype(xp.int32)

        sum_loss = 0.0
        sum_loss_z = 0.0
        sum_loss_z_sparse = 0.0
        sum_loss_label = 0.0
        avg_rate = 0.0
        avg_rate_num = 0.0
        correct_cnt = 0
        total_cnt = 0.0
        N = len(iteration_list)
        is_adv_example_list = []
        is_adv_example_disc_list = []
        is_adv_example_disc_craft_list = []
        y_np = []
        predicted_np = []
        save_items = []
        for i_index, index in enumerate(iteration_list):
            global_step += 1.0
            model.set_train(True)
            sample_idx = perm[index:index + batchsize]
            x = [to_gpu(train_x[_i]) for _i in sample_idx]
            x_length = [train_x_len[_i] for _i in sample_idx]

            y = to_gpu(train_y[sample_idx])

            d = None
            d_hidden = None

            # Classification loss
            output = model(x, x_length)
            output_original = output
            loss = F.softmax_cross_entropy(output, y, normalize=True)
            if args.use_adv or args.use_semi_data:
                # Adversarial Training
                if args.use_adv:
                    output = model(x, x_length, first_step=True, d=None)
                    # Adversarial loss (First step)
                    loss_adv_first = F.softmax_cross_entropy(output, y, normalize=True)
                    model.cleargrads()
                    loss_adv_first.backward()

                    if args.use_adv:
                        d = model.d_var.grad
                        attn_d_grad = chainer.Variable(d)
                        attn_d_grad_original = d
                        d_data = d.data if isinstance(d, chainer.Variable) else d
                    output = model(x, x_length, d=d, d_hidden=d_hidden)
                    # Adversarial loss
                    loss_adv = F.softmax_cross_entropy(output, y, normalize=True)
                    loss += loss_adv * args.nl_factor

                # Virtual Adversarial Training
                if args.use_semi_data:
                    x, length = get_unlabled(perm_semi, i_index)
                    output_original = model(x, length)
                    output_vat = model(x, length, first_step=True, d=None)
                    loss_vat_first = net.kl_loss(xp, output_original.data, output_vat)
                    model.cleargrads()
                    loss_vat_first.backward()
                    d_vat = model.d_var.grad

                    output_vat = model(x, length, d=d_vat)
                    loss_vat = net.kl_loss(xp, output_original.data, output_vat)
                    loss += loss_vat

            predict = xp.argmax(output.data, axis=1)
            correct_cnt += xp.sum(predict == y)
            total_cnt += len(y)

            # update
            model.cleargrads()
            loss.backward()
            opt.update()

            if args.alpha_decay > 0.0:
                if args.use_exp_decay:
                    opt.hyperparam.alpha = (base_alpha) * (args.alpha_decay**global_step)
                else:
                    opt.hyperparam.alpha *= args.alpha_decay  # 0.9999

            sum_loss += loss.data

        accuracy = (correct_cnt / total_cnt) * 100.0

        logging.info(' [train] sum_loss: {}'.format(sum_loss / N))
        logging.info(' [train] apha:{}, global_step:{}'.format(opt.hyperparam.alpha, global_step))
        logging.info(' [train] accuracy:{}'.format(accuracy))


        model.set_train(False)
        # dev
        dev_accuracy = evaluate(dev_x, dev_x_len, dev_y)
        log_str = ' [dev] accuracy:{}'.format(str(dev_accuracy))
        logging.info(log_str)

        # test
        test_accuracy = evaluate(test_x, test_x_len, test_y)
        log_str = ' [test] accuracy:{}'.format(str(test_accuracy))
        logging.info(log_str)

        last_epoch_flag = args.n_epoch - 1 == epoch
        if prev_dev_accuracy < dev_accuracy:

            logging.info(' => '.join([str(prev_dev_accuracy), str(dev_accuracy)]))
            result_str = 'dev_acc_' + str(dev_accuracy)
            result_str += '_test_acc_' + str(test_accuracy)
            model_filename = './models/' + '_'.join([args.save_name,
                                                     str(epoch), result_str])
            # if len(sentences_train_list) == 1:
            serializers.save_hdf5(model_filename + '.model', model)

            prev_dev_accuracy = dev_accuracy



if __name__ == '__main__':
    main()
