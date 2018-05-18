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

import process_dataset
from chainer import serializers
import net

def main():

    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=64, help='learning minibatch size')
    parser.add_argument('--batchsize_semi', dest='batchsize_semi', type=int,
                        default=256, help='learning minibatch size')
    parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=100, help='n_epoch')
    parser.add_argument('--pretrained_model', dest='pretrained_model',
                        type=str, default='', help='pretrained_model')
    parser.add_argument('--w2v_model', dest='w2v_model', type=str, default='', help='w2v_model')
    parser.add_argument('--w2v_norm', dest='w2v_norm', type=int, default=0, help='w2v_norm')
    parser.add_argument('--w2v_var', dest='w2v_var', type=float, default=0.0, help='w2v_var')
    parser.add_argument('--use_unlabled', dest='use_unlabled',
                        type=int, default=0, help='use_unlabled')
    parser.add_argument('--use_rational', dest='use_rational',
                        type=int, default=0, help='use_rational')
    parser.add_argument('--save_name', dest='save_name', type=str,
                        default='sentiment_model_', help='save_name')
    parser.add_argument('--n_layers', dest='n_layers', type=int, default=1, help='n_layers')
    parser.add_argument('--trained_model', dest='trained_model',
                        type=str, default='', help='trained_model')
    parser.add_argument('--alpha', dest='alpha',
                        type=float, default=0.001, help='alpha')
    parser.add_argument('--alpha_decay', dest='alpha_decay',
                        type=float, default=0.0, help='alpha_decay')
    parser.add_argument('--clip', dest='clip',
                        type=float, default=5.0, help='clip')
    parser.add_argument('--l2', dest='l2',
                        type=float, default=0.0, help='l2')
    parser.add_argument('--nobias_lstm', dest='nobias_lstm',
                        type=int, default=0, help='nobias_lstm')
    parser.add_argument('--attenton_for_word', dest='attenton_for_word',
                        type=int, default=0, help='attenton_for_word')
    parser.add_argument('--num_attention', dest='num_attention',
                        type=int, default=1, help='num_attention')
    parser.add_argument('--debug', dest='debug',
                        type=int, default=0, help='debug')
    parser.add_argument('--debug_mode', dest='debug_mode',
                        type=int, default=0, help='debug_mode')
    parser.add_argument('--debug_sim', dest='debug_sim',
                        type=int, default=0, help='debug_sim')
    parser.add_argument('--debug_small', dest='debug_small',
                        type=int, default=0, help='debug_small')
    parser.add_argument('--word_only', dest='word_only',
                        type=int, default=0, help='word_only')
    parser.add_argument('--double_backward', dest='double_backward',
                        type=int, default=0, help='double_backward')
    parser.add_argument('--lambda_1', dest='lambda_1',
                        type=float, default=0.0002, help='lambda_1')
    parser.add_argument('--lambda_2', dest='lambda_2',
                        type=float, default=0.0004, help='lambda_2')
    parser.add_argument('--eps_decay', dest='eps_decay',
                        type=float, default=0.0001, help='eps_decay')
    parser.add_argument('--use_exp_decay', dest='use_exp_decay',
                        type=int, default=1, help='use_exp_decay')
    parser.add_argument('--debug_rational', dest='debug_rational',
                        type=int, default=0, help='debug_rational')
    parser.add_argument('--z_small_limit', dest='z_small_limit',
                        type=int, default=0, help='z_small_limit')
    parser.add_argument('--sampling', dest='sampling',
                        type=int, default=0, help='sampling')
    parser.add_argument('--load_trained_lstm', dest='load_trained_lstm',
                        type=str, default='', help='load_trained_lstm')
    parser.add_argument('--use_fast_lstm', dest='use_fast_lstm', type=int, default=1, help='1')
    parser.add_argument('--load_w_only', dest='load_w_only',
                        type=int, default=0, help='load_w_only')
    parser.add_argument('--freeze_word_emb', dest='freeze_word_emb',
                        type=int, default=0, help='freeze_word_emb')
    parser.add_argument('--dropout', dest='dropout',
                        type=float, default=0.50, help='dropout')
    parser.add_argument('--dot_cost_z', dest='dot_cost_z',
                        type=int, default=0, help='dot_cost_z')
    parser.add_argument('--use_z_word', dest='use_z_word',
                        type=int, default=0, help='use_z_word')
    parser.add_argument('--sep_loss', dest='sep_loss',
                        type=int, default=0, help='sep_loss')
    parser.add_argument('--use_salience', dest='use_salience',
                        type=int, default=0, help='use_salience')
    parser.add_argument('--lambda_1_upper', dest='lambda_1_upper',
                        type=int, default=0, help='lambda_1_upper')
    parser.add_argument('--use_rational_top', dest='use_rational_top',
                        type=int, default=0, help='use_rational_top')
    parser.add_argument('--num_rational', dest='num_rational',
                        type=int, default=10, help='num_rational')
    parser.add_argument('--length_decay', dest='length_decay',
                        type=int, default=0, help='length_decay')
    parser.add_argument('--use_just_norm', dest='use_just_norm',
                        type=int, default=0, help='use_just_norm')
    parser.add_argument('--norm_emb', dest='norm_emb',
                        type=int, default=0, help='norm_emb')
    parser.add_argument('--norm_emb_every', dest='norm_emb_every',
                        type=int, default=1, help='norm_emb_every')
    parser.add_argument('--use_adv', dest='use_adv',
                        type=int, default=0, help='use_adv')
    parser.add_argument('--xi_var', dest='xi_var',
                        type=float, default=1.0, help='xi_var')
    parser.add_argument('--xi_var_first', dest='xi_var_first',
                        type=float, default=1.0, help='xi_var_first')
    parser.add_argument('--lower', dest='lower',
                        type=int, default=1, help='lower')
    parser.add_argument('--use_adv_hidden', dest='use_adv_hidden',
                        type=int, default=0, help='use_adv_hidden')
    parser.add_argument('--use_adv_and_nl_loss', dest='use_adv_and_nl_loss',
                        type=int, default=1, help='use_adv_and_nl_loss')
    parser.add_argument('--norm_lambda', dest='norm_lambda',
                        type=float, default=1.0, help='norm_lambda')
    parser.add_argument('--nl_factor', dest='nl_factor', type=float, default=1.0, help='nl_factor')
    parser.add_argument('--bnorm', dest='bnorm', type=int, default=0, help='bnorm')
    parser.add_argument('--bnorm_hidden', dest='bnorm_hidden',
                        type=int, default=0, help='bnorm_hidden')
    parser.add_argument('--limit_length', dest='limit_length',
                        type=int, default=0, help='limit_length')
    parser.add_argument('--min_count', dest='min_count', type=int, default=1, help='min_count')
    parser.add_argument('--ignore_unk', dest='ignore_unk', type=int, default=0, help='ignore_unk')
    parser.add_argument('--nn_mixup', dest='nn_mixup', type=int, default=0, help='nn_mixup')
    parser.add_argument('--update_nearest_epoch', dest='update_nearest_epoch',
                        type=int, default=0, help='update_nearest_epoch')
    parser.add_argument('--mixup_lambda', dest='mixup_lambda',
                        type=float, default=0.5, help='mixup_lambda')
    parser.add_argument('--mixup_prob', dest='mixup_prob',
                        type=float, default=0.5, help='mixup_prob')
    parser.add_argument('--mixup_type', dest='mixup_type',
                        type=str, default='dir', help='mixup_type')
    parser.add_argument('--mixup_dim', dest='mixup_dim',
                        type=int, default=0, help='mixup_dim')
    parser.add_argument('--nn_k', dest='nn_k', type=int, default=15, help='nn_k')
    parser.add_argument('--nn_k_offset', dest='nn_k_offset',
                        type=int, default=1, help='nn_k_offset')
    parser.add_argument('--norm_mean_var', dest='norm_mean_var',
                        type=int, default=0, help='norm_mean_var')
    parser.add_argument('--word_drop', dest='word_drop', type=int, default=0, help='word_drop')
    parser.add_argument('--word_drop_prob', dest='word_drop_prob',
                        type=float, default=0.25, help='word_drop_prob')
    parser.add_argument('--fix_lstm_norm', dest='fix_lstm_norm',
                        type=int, default=0, help='fix_lstm_norm')
    parser.add_argument('--use_semi_data', dest='use_semi_data',
                        type=int, default=0, help='use_semi_data')
    parser.add_argument('--use_semi_vat', dest='use_semi_vat',
                        type=int, default=1, help='use_semi_vat')
    parser.add_argument('--use_semi_pred_adv', dest='use_semi_pred_adv',
                        type=int, default=0, help='use_semi_pred_adv')
    parser.add_argument('--use_af_dropout', dest='use_af_dropout',
                        type=int, default=0, help='use_af_dropout')
    parser.add_argument('--use_nn_term', dest='use_nn_term',
                        type=int, default=0, help='use_nn_term')
    parser.add_argument('--online_nn', dest='online_nn',
                        type=int, default=0, help='online_nn')
    parser.add_argument('--nn_type', dest='nn_type', type=str, default='dir', help='nn_type')
    parser.add_argument('--nn_div', dest='nn_div', type=int, default=1, help='nn_div')
    parser.add_argument('--xi_type', dest='xi_type',
                        type=str, default='fixed', help='xi_type')
    parser.add_argument('--batchsize_nn', dest='batchsize_nn',
                        type=int, default=10, help='batchsize_nn')
    parser.add_argument('--add_labeld_to_unlabel', dest='add_labeld_to_unlabel',
                        type=int, default=1, help='add_labeld_to_unlabel')
    parser.add_argument('--add_dev_to_unlabel', dest='add_dev_to_unlabel',
                        type=int, default=0, help='add_dev_to_unlabel')
    parser.add_argument('--add_fullvocab', dest='add_fullvocab',
                        type=int, default=0, help='add_fullvocab')
    parser.add_argument('--norm_freq', dest='norm_freq',
                        type=int, default=0, help='norm_freq')
    parser.add_argument('--save_flag', dest='save_flag',
                        type=int, default=1, help='save_flag')
    parser.add_argument('--norm_sentence_level', dest='norm_sentence_level',
                        type=int, default=0, help='norm_sentence_level')
    parser.add_argument('--eps_zeros', dest='eps_zeros',
                        type=int, default=0, help='eps_zeros')
    parser.add_argument('--eps_abs', dest='eps_abs',
                        type=int, default=0, help='eps_abs')
    parser.add_argument('--sampling_eps', dest='sampling_eps',
                        type=int, default=0, help='sampling_eps')
    parser.add_argument('--save_last', dest='save_last',
                        type=int, default=0, help='save_last')
    parser.add_argument('--norm_sent_noise', dest='norm_sent_noise',
                        type=int, default=0, help='norm_sent_noise')
    parser.add_argument('--freeze_nn', dest='freeze_nn',
                        type=int, default=0, help='freeze_nn')
    parser.add_argument('--vat_iter', dest='vat_iter',
                        type=int, default=1, help='vat_iter')
    parser.add_argument('--all_eps', dest='all_eps',
                        type=int, default=0, help='all_eps')
    parser.add_argument('--af_xi_var', dest='af_xi_var',
                        type=float, default=1.0, help='af_xi_var')
    parser.add_argument('--reverse_loss', dest='reverse_loss',
                        type=int, default=0, help='reverse_loss')
    parser.add_argument('--loss_eps', dest='loss_eps',
                        type=float, default=1.0, help='loss_eps')
    parser.add_argument('--init_d_with_nn', dest='init_d_with_nn',
                        type=int, default=0, help='init_d_with_nn')
    parser.add_argument('--ignore_fast_sent_norm', dest='ignore_fast_sent_norm',
                        type=int, default=0, help='ignore_fast_sent_norm')
    parser.add_argument('--ignore_norm', dest='ignore_norm',
                        type=int, default=0, help='ignore_norm')
    parser.add_argument('--init_d_type', dest='init_d_type',
                        type=str, default='rand_nn', help='init_d_type')
    parser.add_argument('--use_d_fixed', dest='use_d_fixed',
                        type=int, default=0, help='use_d_fixed')
    parser.add_argument('--nn_term_sq', dest='nn_term_sq',
                        type=int, default=0, help='nn_term_sq')
    parser.add_argument('--nn_term_sq_half', dest='nn_term_sq_half',
                        type=int, default=0, help='nn_term_sq_half')
    parser.add_argument('--init_d_noise', dest='init_d_noise',
                        type=int, default=0, help='init_d_noise')
    parser.add_argument('--eps_scale', dest='eps_scale',
                        type=float, default=1.0, help='eps_scale')
    parser.add_argument('--sim_type', dest='sim_type',
                        type=str, default='cos', help='sim_type')
    parser.add_argument('--use_all_diff', dest='use_all_diff',
                        type=int, default=1, help='use_all_diff')
    parser.add_argument('--use_first_avg', dest='use_first_avg',
                        type=int, default=0, help='use_first_avg')
    parser.add_argument('--use_norm_d', dest='use_norm_d',
                        type=int, default=1, help='use_norm_d')
    parser.add_argument('--eps_min', dest='eps_min', type=float, default=0.0, help='eps_min')
    parser.add_argument('--eps_max', dest='eps_max', type=float, default=0.0, help='eps_max')
    parser.add_argument('--eps_minus', dest='eps_minus', type=float, default=0.0, help='eps_minus')
    parser.add_argument('--use_random_nn', dest='use_random_nn',
                        type=int, default=0, help='use_random_nn')
    parser.add_argument('--use_attn_d', dest='use_attn_d',
                        type=int, default=0, help='use_attn_d')
    parser.add_argument('--use_softmax', dest='use_softmax',
                        type=int, default=0, help='use_softmax')
    parser.add_argument('--use_attn_full', dest='use_attn_full', type=int, default=0, help='use_attn_full')
    parser.add_argument('--use_attn_dot', dest='use_attn_dot', type=int, default=0, help='use_attn_dot')
    parser.add_argument('--dot_k', dest='dot_k', type=int, default=1, help='dot_k')
    parser.add_argument('--up_grad', dest='up_grad', type=int, default=0, help='up_grad')
    parser.add_argument('--up_grad_attn', dest='up_grad_attn', type=int, default=0, help='up_grad_attn')
    parser.add_argument('--no_grad', dest='no_grad', type=int, default=0, help='no_grad')
    parser.add_argument('--use_nn_drop', dest='use_nn_drop',
                        type=int, default=0, help='use_nn_drop')
    parser.add_argument('--use_onehot', dest='use_onehot',
                        type=int, default=0, help='use_onehot')
    parser.add_argument('--init_rand_diff_d', dest='init_rand_diff_d',
                        type=int, default=0, help='init_rand_diff_d')
    parser.add_argument('--norm_diff', dest='norm_diff', type=int, default=0, help='norm_diff')
    parser.add_argument('--norm_diff_sent', dest='norm_diff_sent', type=int, default=0, help='norm_diff_sent')
    parser.add_argument('--norm_diff_sent_first', dest='norm_diff_sent_first', type=int, default=0, help='norm_diff_sent_first')
    parser.add_argument('--auto_scale_eps', dest='auto_scale_eps',
                        type=int, default=0, help='auto_scale_eps')
    parser.add_argument('--use_concat_random_ids', dest='use_concat_random_ids',
                        type=int, default=0, help='use_concat_random_ids')
    parser.add_argument('--use_d_original_most_sim', dest='use_d_original_most_sim',
                        type=int, default=0, help='use_d_original_most_sim')
    parser.add_argument('--use_important_score', dest='use_important_score',
                        type=int, default=0, help='use_important_score')
    parser.add_argument('--eps_zeros_minus', dest='eps_zeros_minus', type=int, default=0, help='eps_zeros_minus')
    parser.add_argument('--use_attn_drop', dest='use_attn_drop', type=int, default=0, help='use_attn_drop')
    parser.add_argument('--imp_type', dest='imp_type', type=int, default=0, help='imp_type')
    parser.add_argument('--eps_diff', dest='eps_diff', type=int, default=0, help='eps_diff')
    parser.add_argument('--use_limit_vocab', dest='use_limit_vocab', type=int, default=0, help='use_limit_vocab')
    parser.add_argument('--use_plus_d', dest='use_plus_d', type=int, default=0, help='use_plus_d')
    parser.add_argument('--double_adv', dest='double_adv', type=int, default=0, help='double_adv')
    parser.add_argument('--init_d_adv', dest='init_d_adv', type=int, default=0, help='init_d_adv')
    parser.add_argument('--adv_mode', dest='adv_mode', type=int, default=0, help='adv_mode')
    parser.add_argument('--analysis_mode', dest='analysis_mode', type=int, default=0, help='analysis_mode')
    parser.add_argument('--scala_plus', dest='scala_plus', type=int, default=0, help='scala_plus')
    parser.add_argument('--use_plus_zeros', dest='use_plus_zeros', type=int, default=0, help='use_plus_zeros')
    parser.add_argument('--use_attn_one', dest='use_attn_one', type=int, default=0, help='use_attn_one')
    parser.add_argument('--init_d_with', dest='init_d_with', type=float, default=0.0, help='init_d_with')
    parser.add_argument('--kmeans', dest='kmeans', type=int, default=0, help='kmeans')
    parser.add_argument('--n_clusters', dest='n_clusters', type=int, default=100, help='n_clusters')
    parser.add_argument('--top_filter_rate', dest='top_filter_rate', type=float, default=0.10, help='top_filter_rate')
    parser.add_argument('--freeze_d_plus', dest='freeze_d_plus', type=int, default=0, help='freeze_d_plus')
    parser.add_argument('--init_grad', dest='init_grad', type=int, default=0, help='init_grad')
    parser.add_argument('--init_d_attn_ones', dest='init_d_attn_ones', type=int, default=0, help='init_d_attn_ones')
    parser.add_argument('--init_d_fact', dest='init_d_fact', type=float, default=1.0, help='init_d_fact')
    parser.add_argument('--init_d_random', dest='init_d_random', type=int, default=0, help='init_d_random')
    parser.add_argument('--norm_diff_all', dest='norm_diff_all', type=int, default=0, help='norm_diff_all')
    parser.add_argument('--norm_sent_attn_scala', dest='norm_sent_attn_scala', type=int, default=0, help='norm_sent_attn_scala')
    parser.add_argument('--rep_sim_noise_word', dest='rep_sim_noise_word', type=int, default=0, help='rep_sim_noise_word')
    parser.add_argument('--ign_noise_eos', dest='ign_noise_eos', type=int, default=0, help='ign_noise_eos')
    parser.add_argument('--search_iters', dest='search_iters', type=int, default=10, help='search_iters')
    parser.add_argument('--adv_type', dest='adv_type', type=int, default=0, help='adv_type')
    parser.add_argument('--use_saliency', dest='use_saliency', type=int, default=0, help='use_saliency')
    parser.add_argument('--adv_iter', dest='adv_iter', type=int, default=1, help='adv_iter')
    parser.add_argument('--fixed_d', dest='fixed_d', type=int, default=0, help='fixed_d')
    parser.add_argument('--max_attn', dest='max_attn', type=int, default=0, help='max_attn')
    parser.add_argument('--max_attn_type', dest='max_attn_type', type=int, default=0, help='max_attn_type')
    parser.add_argument('--use_grad_scale', dest='use_grad_scale', type=int, default=0, help='use_grad_scale')
    parser.add_argument('--scale_type', dest='scale_type', type=int, default=0, help='scale_type')
    parser.add_argument('--print_info', dest='print_info', type=int, default=0, help='print_info')
    parser.add_argument('--soft_int', dest='soft_int', type=float, default=1.0, help='soft_int')
    parser.add_argument('--soft_int_final', dest='soft_int_final', type=float, default=1.0, help='soft_int_final')
    parser.add_argument('--adv_mode_iter', dest='adv_mode_iter', type=int, default=100, help='adv_mode_iter')
    parser.add_argument('--ignore_norm_final', dest='ignore_norm_final', type=int, default=0, help='ignore_norm_final')
    parser.add_argument('--noise_factor', dest='noise_factor', type=float, default=1.0, help='noise_factor')
    parser.add_argument('--use_zero_d', dest='use_zero_d', type=int, default=0, help='use_zero_d')
    parser.add_argument('--use_random_max', dest='use_random_max', type=int, default=0, help='use_random_max')
    parser.add_argument('--analysis_limit', dest='analysis_limit', type=int, default=0, help='analysis_limit')

    parser.add_argument('--use_weight_alpha', dest='use_weight_alpha', type=int, default=0, help='use_weight_alpha')
    parser.add_argument('--weight_type', dest='weight_type', type=int, default=0, help='weight_type')


    parser.add_argument('--dataset', default='imdb',
                        choices=['imdb', 'elec', 'rotten', 'dbpedia', 'rcv1', 'conll_2014', 'fce','ptb', 'wikitext-2', 'wikitext-103'])
    parser.add_argument('--use_seq_labeling', dest='use_seq_labeling', type=int, default=0, help='use_seq_labeling')
    parser.add_argument('--use_seq_labeling_pickle', dest='use_seq_labeling_pickle', type=int, default=0, help='use_seq_labeling_pickle')
    parser.add_argument('--use_bilstm', dest='use_bilstm', type=int, default=0, help='use_bilstm')
    parser.add_argument('--use_bilstm_forget', dest='use_bilstm_forget', type=int, default=0, help='use_bilstm_forget')
    parser.add_argument('--use_crf', dest='use_crf', type=int, default=0, help='use_crf')
    parser.add_argument('--use_all_for_lm', dest='use_all_for_lm', type=int, default=0, help='use_all_for_lm')

    parser.add_argument('--cs', dest='cs', type=int, default=0, help='cs')
    parser.add_argument('--debug_eval', dest='debug_eval', type=int, default=0, help='debug_eval')
    parser.add_argument('--emb_dim', dest='emb_dim', type=int, default=256, help='emb_dim')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=1024, help='hidden_dim')
    parser.add_argument('--hidden_cls_dim', dest='hidden_cls_dim', type=int, default=30, help='hidden_cls_dim')
    parser.add_argument('--adaptive_softmax', dest='adaptive_softmax', type=int, default=1, help='adaptive_softmax')
    parser.add_argument('--use_w2v_flag', dest='use_w2v_flag', type=int, default=0, help='use_w2v_flag')
    parser.add_argument('--sent_loss', dest='sent_loss', type=int, default=0, help='sent_loss')
    parser.add_argument('--sent_loss_usual', dest='sent_loss_usual', type=int, default=0, help='sent_loss_usual')
    parser.add_argument('--use_ortho', dest='use_ortho', type=int, default=0, help='use_ortho')

    parser.add_argument('--analysis_loss', dest='analysis_loss', type=int, default=0, help='analysis_loss')
    parser.add_argument('--analysis_data', dest='analysis_data', type=int, default=2, help='analysis_data')
    parser.add_argument('--fil_type', dest='fil_type', type=int, default=0, help='fil_type')
    parser.add_argument('--analysis_mode_type', dest='analysis_mode_type', type=int, default=0, help='analysis_mode_type')
    parser.add_argument('--tsne_mode', dest='tsne_mode', type=int, default=0, help='tsne_mode')
    parser.add_argument('--bar_mode', dest='bar_mode', type=int, default=0, help='bar_mode')
    parser.add_argument('--attentional_d_mode', dest='attentional_d_mode', type=int, default=0, help='attentional_d_mode')
    parser.add_argument('--div_attn_d', dest='div_attn_d', type=int, default=0, help='div_attn_d')

    parser.add_argument('--use_attn_sent_norm', dest='use_attn_sent_norm', type=int, default=0, help='use_attn_sent_norm')
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=1234, help='random_seed')

    parser.add_argument('--n_class', dest='n_class', type=int, default=2, help='n_class')
    parser.add_argument('--init_scale', dest='init_scale', type=float, default=0.0, help='init_scale')
    parser.add_argument('--train_test_flag', dest='train_test_flag', type=int, default=0, help='train_test_flag')

    args = parser.parse_args()
    batchsize = args.batchsize
    batchsize_semi = args.batchsize_semi
    print(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["CHAINER_SEED"] = str(args.random_seed)

    if args.debug_mode:
        chainer.set_debug(True)

    use_unlabled = args.use_unlabled
    lower = args.lower == 1
    n_char_vocab = 1
    n_class = 2
    if args.dataset == 'imdb':
        vocab_obj, dataset, lm_dataset, train_vocab_size = utils.load_dataset_imdb(include_pretrain=use_unlabled, lower=lower,
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
                                          blackout_counts=None,
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

    if args.debug_eval:
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
        model.reset_statics()
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
        save_flag = args.save_flag or (args.save_last and last_epoch_flag)
        if prev_dev_accuracy < dev_accuracy and save_flag:

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
