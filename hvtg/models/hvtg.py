import math
import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
#from tensorflow.contrib.rnn import LSTMBlockCell as lstm
from tensorflow.nn.rnn_cell import LSTMCell as lstm
from tensorflow.python.layers.core import dense, Dense, dropout
from tensorflow.contrib.cudnn_rnn import CudnnLSTM
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python.ops.rnn_cell_impl import *

conv1d  = tf.layers.conv1d
import ipdb



class Att(object):
    def __init__(self, cfg):

        self.c = cfg
        self.is_t = cfg.mode =='train'


    def _bi_rnn_encode(self, seq, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            seq = tf.transpose(seq, [1,0,2])
            rnn_cell = CudnnLSTM(1, self.c.hid, direction=cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION)
            outputs, _ = rnn_cell(seq)
            h_cat = tf.transpose(outputs, [1,0,2])
            ret = dense(h_cat, self.c.hid*2, activation=tf.nn.relu, use_bias=True)
        return ret

    def bi_rnn_encode(self, seq, scope, proj):
        with tf.variable_scope(scope):
            cell_fw, cell_bw = lstm(384), lstm(384)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                         cell_bw,
                                                         seq,
                                                         dtype=tf.float32)
            h_cat = tf.concat(outputs, 2)
            ret = dense(h_cat, proj, activation=tf.nn.relu, use_bias=True)
        return ret

    def attn(self, z, g, scope):
        with tf.variable_scope(scope):
            z_p = dense(z, self.c.hid, use_bias=False)
            g_p = dense(g, self.c.hid, use_bias=True)
            H = tf.tanh(z_p+g_p[:,None,:])
            H_p = dense(H, 1, use_bias=False)
            H_p = tf.squeeze(H_p)
            a = tf.nn.softmax(H_p)
            att_z = tf.reduce_sum(a[:,:,None]*z, axis=1)
        return att_z, a

    def int(self, q, k, v, num_heads=1, scope='int'):
        num_units = q.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            i = multihead_attention(queries=q,
                                    keys=k,
                                    values=v,
                                    num_units=num_units,
                                    num_heads=num_heads,
                                    dropout_rate=0.1,
                                    is_training=self.is_t)
            o = feedforward(i, num_units=[2*num_units, num_units], scope='ff')
        return o

    def loss_sim(self, sim_score_mat):
        B = sim_score_mat.get_shape().as_list()[-1]
        I_2 = tf.diag(tf.constant(-2.0, shape=[B]))
        I = tf.diag(tf.constant(1.0, shape=[B]))
        all_one = tf.constant(1.0, shape=[B, B])
        all_alpha = tf.constant(1.0/B, shape=[B, B])

        mask_mat = tf.add(I_2, all_one)
        para_mat = tf.add(I, all_alpha)

        loss_mat = tf.log(tf.add(all_one, tf.exp(tf.multiply(mask_mat, sim_score_mat))))
        loss_mat = tf.multiply(loss_mat, para_mat)
        loss_align = tf.reduce_mean(loss_mat)

        return loss_align

    def build(self, features):
        self.v = features[0]
        self.s = features[1]
        self.l = features[2]
        self.m = features[3]
        #adj = tf.cast(features[4], tf.float32)

        bs, len_f, n_prop, dim_f = self.v.get_shape().as_list()

        ms_pre = tf.reduce_mean(self.s, axis=1)

        att_proj = Dense(1, use_bias=False, activation=tf.nn.leaky_relu)

        vis = []
        for i in range(len_f):

            v_i = self.v[:,i]

            v_i = self.int(v_i, self.s, self.s, scope='int_vs')

            v_i_r = tf.tile(tf.expand_dims(v_i, 2), (1,1,n_prop,1))
            v_i_c = tf.tile(tf.expand_dims(v_i, 1), (1,n_prop,1,1))
            v_i_rc = tf.concat([v_i_r, v_i_c], axis=-1)

            att_i = tf.nn.softmax(tf.squeeze(att_proj(v_i_rc), axis=-1), axis=-1)
            v_i_att = tf.einsum('bij,bjk->bik', att_i, v_i)

            v_i, _ = self.attn(v_i_att, ms_pre, 'att_vid_i_guide_by_ms_pre')

            vis.append(v_i)

        v = tf.stack(vis, axis=1)
        print(v)

        r_v = self.bi_rnn_encode(v, 'video_bi_rnn', 512)
        r_s = self.bi_rnn_encode(self.s, 'sent_bi_rnn', 256)

        ms = tf.reduce_mean(r_s, axis=1)

        ms_tile = tf.tile(tf.expand_dims(ms, 1), [1,self.c.max_feat_len,1])
        conv_i = r_v

        def conv(x, c, k, layer):
            if c > 1:
                x = tf.concat([x, ms_tile], axis=-1)
                x = conv1d(x, c, k, 1, padding='same', activation=None)
                x = x+chn_int(x, 'layer_%d'%layer)
            else:
                x = conv1d(x, c, k, 1, padding='same', activation=None)

            x = tf.contrib.layers.instance_norm(x, center=True, scale=True)
            x = tf.nn.leaky_relu(x)

            return x

        def chn_int(x, scope):
            #mx = tf.reduce_mean(x, axis=1)
            with tf.variable_scope(scope):
                x_dim = x.get_shape().as_list()[-1]

                x_p = dense(x, x_dim//4)
                p_dim = x_dim//4
                x_r = tf.tile(tf.expand_dims(x_p, 3), (1,1,1,p_dim))
                x_c = tf.tile(tf.expand_dims(x_p, 2), (1,1,p_dim,1))
                scores = -tf.square(x_r-x_c)
                scores = tf.reduce_mean(scores, axis=1)
                scores = tf.nn.softmax(scores, axis=-1)
                x_p = tf.einsum('blj,bij->bli', x_p, scores)
                x = dense(x_p, x_dim)
            return x

        cs = [512, 256, 256, 1]
        ks = [5, 3, 3, 3]
        is_f = [0,0,1,0]
        for i, (c, k, f) in enumerate(zip(cs, ks, is_f)):
            conv_i = conv(conv_i, c, k, i)
            if f == 1:
                conv_f = conv_i

        w = tf.nn.softmax(tf.squeeze(conv_i, axis=2), axis=1)

        pred = dense(w, 2, activation=tf.nn.relu, use_bias=True)

        att_v = tf.reduce_sum(conv_f * w[:,:,None], axis=1)
        bs = att_v.get_shape().as_list()[0]
        vt = tf.tile(tf.expand_dims(att_v, axis=1), [1,bs,1])
        st = tf.tile(tf.expand_dims(ms, axis=0), [bs,1,1])
        sim_vs = tf.reduce_mean(-tf.square(vt - st), axis=-1)
        loss_align = self.loss_sim(sim_vs)

        feat_norm = tf.norm(conv_f, ord=2, axis=-1)
        norm_v = tf.get_variable('norm', dtype=tf.float32,
                                 initializer=tf.constant(10.),
                                 trainable=True)

        l_norm = tf.nn.l2_loss(feat_norm-norm_v)


        l_reg = tf.losses.huber_loss(self.l, pred, reduction=tf.losses.Reduction.NONE)
        l_reg = tf.reduce_sum(l_reg, axis=1)

        eps = 1e-9
        l_cal = -tf.reduce_sum(self.m*tf.log(w+eps), axis=1) / tf.reduce_sum(self.m, axis=1)

        #print l_reg, l_cal
        l_reg = tf.reduce_mean(l_reg)
        l_cal = tf.reduce_mean(l_cal)

        #loss = self.c.alpha * l_reg + self.c.beta * l_cal
        loss = l_reg + 5. * l_cal + l_norm * 0.001 + loss_align * 1.

        if self.is_t:
            tf.summary.scalar('reg', l_reg)
            tf.summary.scalar('cal', l_cal)
            tf.summary.scalar('all', loss)

        return {'all_loss': loss,
                'reg': l_reg,
                'cal': l_cal,
                'preds': pred,
                'att_w': w}


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        #beta= tf.Variable(tf.zeros(params_shape))
        beta= tf.get_variable('beta', initializer=tf.zeros(params_shape), trainable=True)
        #gamma = tf.Variable(tf.ones(params_shape))
        gamma = tf.get_variable('gamma', initializer=tf.ones(params_shape), trainable=True)
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs

def multihead_attention(queries,
                        keys,
                        values,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(values, num_units, activation=tf.nn.relu) # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

        # Multiplication
        true_scope = tf.get_variable_scope().name

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        #key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        #key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        #key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

        #paddings = tf.ones_like(outputs)*(-2**32+1)
        #outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)

        # Query Masking
        #query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        #query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        #query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        #outputs *= query_masks # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Weighted sum

        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)

        outputs += queries

        outputs = normalize(outputs) # (N, T_q, C)

    return outputs
