# encoding = utf-8

"""
1: dynamic_rnn 与　普通的lstm在BP时，到底怎么进行，是否能够记住训练的步数
2: tf.contrib.legacy_seq2seq.sequence_loss_by_example中weights参数的作用
3: tf.clip_by_global_norm

"""
import tensorflow as tf
import numpy as np


class RNN_Model(object):

    def __init__(self, config, is_training=True):

        self.mask = tf.placeholder(dtype=tf.float64, shape=[None])

        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[config.batch_size])

        self.input_data = tf.placeholder(dtype=tf.float64, shape=[config.batch_size, None, config.input_dimension])
        self.targets = tf.placeholder(dtype=tf.int32, shape=[config.batch_size, None])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_neural_size,
                                                 forget_bias=0.0, state_is_tuple=True,
                                                 reuse=not is_training)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob=config.keep_prob)

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*config.num_layers, state_is_tuple=True)

        self.initial_state = multi_rnn_cell.zero_state(config.batch_size, dtype=tf.float64)

        with tf.name_scope("LSTM_layer"):
            outputs, last_states = tf.nn.dynamic_rnn(
                cell=multi_rnn_cell,
                dtype=tf.float64,
                sequence_length=self.sequence_length,
                inputs=self.input_data,
                initial_state=self.initial_state)

        # reshape outputs
        outputs = tf.reshape(outputs, [-1, config.hidden_neural_size])

        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w", [config.hidden_neural_size, config.class_num], dtype=tf.float64)
            softmax_b = tf.get_variable("softmax_b", [config.class_num], dtype=tf.float64)
            self.logits = tf.matmul(outputs, softmax_w) + softmax_b

        with tf.name_scope("loss"):
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                # targets shape [batch, num_steps]
                [tf.reshape(self.targets, [-1])],
                [self.mask]
            )
            self.cost_batch = tf.reduce_sum(self.loss) / config.batch_size
            self.cost_all = tf.reduce_sum(self.loss)

        if not is_training:
            return

        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost_batch, trainable_variables), config.max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        # keep track of gradient values
