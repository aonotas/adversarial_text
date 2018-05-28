'''
uni-LSTM + Virtual Adversarial Training
'''

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable

from six.moves import xrange


def kl_loss(xp, p_logit, q_logit):
    p = F.softmax(p_logit)
    _kl = F.sum(p * (F.log_softmax(p_logit) - F.log_softmax(q_logit)), 1)
    return F.sum(_kl) / xp.prod(xp.array(_kl.shape))


def get_normalized_vector(d, xp=None, shape=None):
    if shape is None:
        shape = tuple(range(1, len(d.shape)))
    d_norm = d
    if xp is not None:
        d_norm = d / (1e-12 + xp.max(xp.abs(d), shape, keepdims=True))
        d_norm = d_norm / xp.sqrt(1e-6 + xp.sum(d_norm ** 2, shape, keepdims=True))
    else:
        d_term = 1e-12 + F.max(F.absolute(d), shape, keepdims=True)
        d_norm = d / F.broadcast_to(d_term, d.shape)
        d_term = F.sqrt(1e-6 + F.sum(d ** 2, shape, keepdims=True))
        d_norm = d / F.broadcast_to(d_term, d.shape)
    return d_norm


class uniLSTM_VAT(chainer.Chain):

    def __init__(self, n_vocab=None, emb_dim=256, hidden_dim=1024,
                 use_dropout=0.50, n_layers=1, hidden_classifier=30,
                 use_adv=0, xi_var=5.0, n_class=2,
                 args=None):
        self.args = args
        super(uniLSTM_VAT, self).__init__(
            word_embed = L.EmbedID(n_vocab, emb_dim, ignore_label=-1),
            hidden_layer=L.Linear(hidden_dim, hidden_classifier),
            output_layer=L.Linear(hidden_classifier, n_class)
        )
        uni_lstm = L.NStepLSTM(n_layers=n_layers, in_size=emb_dim,
                               out_size=hidden_dim, dropout=use_dropout)
        # Forget gate bias => 1.0
        # MEMO: Values 1 and 5 reference the forget gate.
        for w in uni_lstm:
            w.b1.data[:] = 1.0
            w.b5.data[:] = 1.0

        self.add_link('uni_lstm', uni_lstm)

        self.hidden_dim = hidden_dim
        self.train = True
        self.use_dropout = use_dropout
        self.n_layers = n_layers
        self.use_adv = use_adv
        self.xi_var = xi_var
        self.n_vocab = n_vocab
        self.grad_scale = None

    def freeze_word_emb(self):
        self.word_embed.W.update_rule.enabled = False

    def set_pretrained_lstm(self, pretrain_model, word_only=True):
        # set word embeddding
        limit = self.word_embed.W.shape[0]
        # bf_norm = np.average(np.linalg.norm(self.word_embed.W.data, axis=1))
        # af_norm = np.average(np.linalg.norm(pretrain_model.embed.W.data, axis=1))
        self.word_embed.W.data[:] = pretrain_model.embed.W.data[:limit]

        # if self.args.fix_lstm_norm:
        #     # print 'bf_norm:', bf_norm
        #     # print 'af_norm:', af_norm
        #     self.fix_norm(bf_norm, af_norm)

        if word_only:
            return True

        def split_weights(weights):
            input_dim = weights.shape[-1]
            reshape_weights = F.reshape(weights, (-1, 4, input_dim))
            reshape_weights = [reshape_weights[:, i, :] for i in xrange(4)]
            return reshape_weights

        def split_bias(bias):
            reshape_bias = F.reshape(bias, (-1, 4))
            reshape_bias = [reshape_bias[:, i] for i in xrange(4)]
            # reshape_bias = bias
            # reshape_bias = [reshape_bias[i::4] for i in xrange(4)]
            return reshape_bias

        # set lstm params
        pretrain_lstm = pretrain_model.lstm
        for layer_i in xrange(self.args.n_layers):
            w = pretrain_lstm[layer_i]
            source_w = [w.w2, w.w0, w.w1, w.w3, w.w6, w.w4, w.w5, w.w7]
            source_b = [w.b2, w.b0, w.b1, w.b3, w.b6, w.b4, w.b5, w.b7]

            w = self.uni_lstm[layer_i]
            # [NStepLSTM]
            # w0, w4 : input gate   (i)
            # w1, w5 : forget gate  (f)
            # w2, w6 : new memory gate (c)
            # w3, w7 : output gate

            # [Chaner LSTM]
            # a,   :   w2, w6
            # i,   :   w0, w4
            # f,   :   w1, w5
            # o    :   w3, w7
            uni_lstm_w = [w.w2, w.w0, w.w1, w.w3, w.w6, w.w4, w.w5, w.w7]
            uni_lstm_b = [w.b2, w.b0, w.b1, w.b3, w.b6, w.b4, w.b5, w.b7]
            # uni_lstm_b = [w.b0, w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7]

            for uni_w, pre_w in zip(uni_lstm_w, source_w):
                uni_w.data[:] = pre_w.data[:]

            for uni_b, pre_b in zip(uni_lstm_b, source_b):
                uni_b.data[:] = pre_b.data[:]

    def set_train(self, train):
        self.train = train

    def freeze_word_emb(self):
        self.word_embed.W.update_rule.enabled = False

    def output_mlp(self, hy):
        # hy = F.dropout(hy, ratio=self.use_dropout)
        hy = self.hidden_layer(hy)
        hy = F.relu(hy)
        hy = F.dropout(hy, ratio=self.use_dropout)
        output = self.output_layer(hy)
        return output

    def __call__(self, x_data, lengths=None, d=None, first_step=False):
        batchsize = len(x_data)
        h_shape = (self.n_layers, batchsize, self.hidden_dim)
        hx = None
        cx = None

        x_data = self.xp.concatenate(x_data, axis=0)
        xs = self.word_embed(x_data)
        # dropout
        xs = F.dropout(xs, ratio=self.use_dropout)

        adv_flag = self.train and (self.use_adv or self.args.use_semi_data)

        if adv_flag:

            def norm_vec_sentence_level(d, nn_flag=False, include_norm_term=False):
                dim = d.shape[1]
                d_list = F.split_axis(d, np.cumsum(lengths)[:-1], axis=0)
                max_length = np.max(lengths)
                d_pad = F.pad_sequence(d_list, length=max_length, padding=0.0)
                d_flat = F.reshape(get_normalized_vector(d_pad, None), (-1, dim))
                split_size = np.cumsum(np.full(batchsize, max_length))[:-1]
                d_list = F.split_axis(d_flat, split_size, axis=0)
                d_list = [_d[:_length] for _d, _length in zip(d_list, lengths)]
                d = F.concat(d_list, axis=0)
                return d

            if first_step:
                if self.args.use_semi_data:
                    # Vat
                    d = self.xp.random.normal(size=xs.shape, dtype='f')
                else:
                    # Adv
                    d = self.xp.zeros(xs.shape, dtype='f')
                if self.args.ignore_fast_sent_norm:
                    # Normalize at word-level
                    d = get_normalized_vector(d, self.xp)
                else:
                    # Normalize at sentence-level
                    d = norm_vec_sentence_level(d)

                d_var = Variable(d.astype(self.xp.float32))
                self.d_var = d_var
                xs = xs + self.args.xi_var_first * d_var

            elif d is not None:
                d_original = d.data if isinstance(d, Variable) else d
                if self.args.norm_sentence_level:
                    # Normalize at sentence-level
                    d_variable = norm_vec_sentence_level(d, include_norm_term=True)
                    d = d_variable.data
                else:
                    # Normalize at word-level
                    d = get_normalized_vector(d_original, self.xp)
                    xs_noise_final = self.xi_var * d
                    xs = xs + xs_noise_final

        split_size = np.cumsum(lengths)[:-1]
        xs_f = F.split_axis(xs, split_size, axis=0)

        hy_f, cy_f, ys_list = self.uni_lstm(hx=hx, cx=cx, xs=xs_f)

        hy = [_h[-1] for _h in ys_list]
        hy = F.concat(hy, axis=0)
        hy = F.reshape(hy, (batchsize, -1))
        self.hy = hy

        output = self.output_mlp(hy)
        return output
