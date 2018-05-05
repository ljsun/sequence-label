# encoding = utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import data_helper
from rnn_model import RNN_Model
import time


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("train_batch_size", 64, 'the batch_size of training procedure')
flags.DEFINE_integer("test_batch_size", 100, 'the batch_size of testing procedure')
flags.DEFINE_integer("input_dimension", 39, 'the dimension of input data')
flags.DEFINE_integer("hidden_neural_size", 39, 'the size of hidden_neural')
flags.DEFINE_float("keep_prob", 0.5, 'dropout rate')
flags.DEFINE_integer("num_layers", 2, 'num of rnn')
flags.DEFINE_integer("class_num", 40, 'class num')
flags.DEFINE_integer("max_grad_norm", 5, 'max_grad_norm')
flags.DEFINE_float("learning_rate", 0.1, 'the learning rate')
flags.DEFINE_float("init_scale", 0.1, 'init scale')
flags.DEFINE_integer("num_epoch", 10, 'num of epoch')


class Config(object):

    train_batch_size = FLAGS.train_batch_size
    test_batch_size = FLAGS.test_batch_size
    input_dimension = FLAGS.input_dimension
    hidden_neural_size = FLAGS.hidden_neural_size
    keep_prob = FLAGS.keep_prob
    num_layers = FLAGS.num_layers
    class_num = FLAGS.class_num
    max_grad_norm = FLAGS.max_grad_norm
    learning_rate = FLAGS.learning_rate
    init_scale = FLAGS.init_scale
    num_epoch = FLAGS.num_epoch


def evaluate(model, session, data, verbose=False):

    total_correct_num = 0
    total_num = 0
    for step, (x, y, sequence_length, num_step, mask) in \
            enumerate(data_helper.data_iterator(data, Config.test_batch_size)):

        feed_dict = dict()
        feed_dict[model.mask] = mask
        feed_dict[model.sequence_length] = sequence_length
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y

        fetchs = {"correct_num": model.correct_num,
                  "accuracy": model.accuracy,
                  "cost_all": model.cost_all}

        vars = session.run(fetchs, feed_dict)
        correct_num = vars["correct_num"]
        accuracy = vars["accuracy"]
        cost_all = vars['cost_all']

        total_correct_num += correct_num
        total_num += sum(sequence_length)
        
        assert sum(mask) == sum(sequence_length)

        if verbose:
            print('the %i step, total_num is %i, correct_num is %.f, accuracy is %.3f, '
                  'average cost of one time_step in one sample is %.3f'
                  % (step, sum(sequence_length), correct_num, accuracy, cost_all / sum(sequence_length)))
    final_accuracy = total_correct_num / total_num
    return final_accuracy


# v1.0 无valid_model,　仅仅train
def run_epoch(model, session, data, global_steps, verbose=False):

    for step, (x, y, sequence_length, num_step, mask) in \
            enumerate(data_helper.data_iterator(data, Config.train_batch_size)):

        feed_dict = dict()
        feed_dict[model.mask] = mask
        feed_dict[model.sequence_length] = sequence_length
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y

        fetchs = {"cost_batch": model.cost_batch,
                  "cost_all": model.cost_all,
                  "accuracy": model.accuracy,
                  "train_op": model.train_op}

        # state = session.run(model.initial_state)
        # for i, (c, h) in enumerate(model.initial_state):
        #     feed_dict[c] = state[i].c
        #     feed_dict[h] = state[i].h
        vars = session.run(fetchs, feed_dict)
        cost_batch = vars["cost_batch"]
        cost_all = vars["cost_all"]
        accuracy = vars["accuracy"]

        assert sum(mask) == sum(sequence_length)

        if verbose and step % 10 == 0:
            print('the %i step, batch_size is %i, accuracy is %.3f, average cost of one time_step in one sample is %f' % (step, Config.train_batch_size, accuracy, cost_all / sum(sequence_length)))
        global_steps += 1
    return global_steps


def train_test_step():

    print("loading the dataset")
    train_x_data, train_y_data, test_x_data, test_y_data = data_helper.load_raw_data('./data')

    initializer = tf.random_normal_initializer(-1 * Config.init_scale, 1 * Config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        train_model = RNN_Model(Config, is_training=True)

    with tf.variable_scope("model", reuse=True, initializer=initializer):
        valid_model = RNN_Model(Config, is_training=False)
        test_model = RNN_Model(Config, is_training=False)

    print("begin training")
    with tf.Session() as session:

        # add summary
        # ...

        # initialize variables
        
        tf.global_variables_initializer().run()
        global_steps = 1
        begin_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print(begin_time)

        for i in range(Config.num_epoch):
            print("the %d epoch training..." % (i+1))
            global_steps = run_epoch(train_model, session, (train_x_data, train_y_data), global_steps, verbose=True)

            print("the %d epoch testing..." % (i+1))
            final_accuracy = evaluate(test_model, session, (test_x_data, test_y_data), verbose=True)
            print('the final accuracy is %.3f' % final_accuracy)

            print('-------------------------')
        end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print(end_time)
        print('--------------------------')
        """print('begin test')
        final_accuracy = evaluate(test_model, session, (test_x_data, test_y_data), verbose=True)
        print('-------------------------')
        print('the final accuracy is %.3f' % final_accuracy)"""


def main(_):
    train_test_step()

if __name__ == '__main__':
    tf.app.run(main)
