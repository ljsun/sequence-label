# encoding = utf-8
import os
import pandas as pd
import numpy as np
import copy


def load_raw_data(data_path=None):
    """
    load raw data from data directory "data_path"
    use mfcc feature
    :param data_path:
    :return:
    """

    train_x_path = os.path.join(data_path, 'train.ark')
    train_y_path = os.path.join(data_path, 'train_label.csv')
    test_x_path = os.path.join(data_path, 'test.ark')
    test_y_path = os.path.join(data_path, 'test_label.csv')

    phone_path = os.path.join(data_path, '48_39.map')
    phone_to_id = _build_vocab(phone_path)

    print('load train_x_data')
    train_x_data = _build_x_data(train_x_path)
    print('load train_y_data')
    train_y_data = _build_y_data(train_y_path, phone_to_id)
    print('load test_x_data')
    test_x_data = _build_x_data(test_x_path)
    print('load test_y_data')
    test_y_data = _build_y_data(test_y_path, phone_to_id)

    return train_x_data, train_y_data, test_x_data, test_y_data


def _build_vocab(filename):
    map_frame = pd.read_table(filename, sep='\t', header=None)
    map_frame.columns = ['48', '39']
    # remove duplicate phone and make sure order
    phonelist = set(map_frame['39'])
    phone_to_id = dict(zip(phonelist, range(1, len(phonelist)+1)))
    return phone_to_id


def _build_x_data(filename):
    x_data = []
    x_frame = pd.read_table(filename, sep=' ', header=None)
    featurecurrent = list(x_frame.iloc[0])
    user_feature_list = []
    user_feature_list.append(featurecurrent[1:])
    for feature in x_frame.iterrows():
        featurenext = list(feature[1][:])
        if featurenext[0].split('_')[0] == featurecurrent[0].split('_')[0] \
                and featurenext[0].split('_')[1] == featurecurrent[0].split('_')[1]:
            featurecurrent = featurenext
            user_feature_list.append(featurecurrent[1:])
        else:
            x_data.append(user_feature_list)
            user_feature_list = []
            featurecurrent = featurenext
            user_feature_list.append(featurecurrent[1:])
    # add last user_speech_feature and remove the first duplication
    x_data[0] = x_data[0][1:]
    x_data.append(user_feature_list)
    return x_data


def _build_y_data(filename, phone_to_id):
    y_data = []
    y_frame = pd.read_table(filename, sep=',', header=None)
    targetcurrent = list(y_frame.iloc[0])
    user_target_list = []
    # convert target phone to id
    user_target_list.append(phone_to_id[targetcurrent[1]])
    for target in y_frame.iterrows():
        targetnext = list(target[1][:])
        if targetnext[0].split('_')[0] == targetcurrent[0].split('_')[0] \
                and targetnext[0].split('_')[1] == targetcurrent[0].split('_')[1]:
            targetcurrent = targetnext
            user_target_list.append(phone_to_id[targetcurrent[1]])
        else:
            y_data.append(user_target_list)
            user_target_list = []
            targetcurrent = targetnext
            user_target_list.append(phone_to_id[targetcurrent[1]])
    # add last user_speech_feature and remove first duplication
    y_data[0] = y_data[0][1:]
    y_data.append(user_target_list)
    return y_data


def _find_max_len_and_padding(data_x, data_y):
    """
    find the max_len and padding the corresponding data
    return the initial sequence_length
    :param data_x:
    :param data_y:
    :return: padded_data_x, padded_data_y, sequence_length
    """
    sequence_length = []
    max_len = 0
    for data in data_x:
        sequence_length.append(len(data))
        if max_len < len(data):
            max_len = len(data)
    # padding x depends on max_len
    padded = [0.0] * 39
    padded_data_x = []
    for data in data_x:
        data_copy_x = copy.deepcopy(data)
        if len(data_copy_x) < max_len:
            data_copy_x.extend([padded] * (max_len - len(data)))
        padded_data_x.append(data_copy_x)
        del data_copy_x
    del data_x

    # padding y depends on max_len
    padded_data_y = []
    for data in data_y:
        data_copy_y = copy.deepcopy(data)
        if len(data_copy_y) < max_len:
            data_copy_y.extend([0] * (max_len - len(data)))
        padded_data_y.append(data_copy_y)
        del data_copy_y
    del data_y

    return np.array(padded_data_x), np.array(padded_data_y), np.array(sequence_length)


def data_iterator(data, batch_size):
    # find the max_len in every batch_size and then padding
    data_x, data_y = data
    data_size = len(data_x)
    num_batches_per_epoch = data_size // batch_size
    for batch_index in range(num_batches_per_epoch):
        start_index = batch_index * batch_size
        end_index = (batch_index + 1) * batch_size
        # find the max_len and padding
        padded_data_x, padded_data_y, sequence_length = _find_max_len_and_padding(data_x[start_index:end_index],
                                                                                  data_y[start_index:end_index])

        padded_data_x = np.array(padded_data_x)
        padded_data_y = np.array(padded_data_y)

        # produce num_step
        num_step = max(sequence_length)
        # produce mask
        bool_sequece_length = []
        for i in sequence_length:
            bool_sequece_length.extend([1] * i)
            bool_sequece_length.extend([0] * (max(sequence_length) - i))
        mask = np.array(bool_sequece_length, dtype=np.float64)
        yield (padded_data_x, padded_data_y, sequence_length, num_step, mask)

if __name__ == '__main__':
    # begin test
    train_x_data, train_y_data, test_x_data, test_y_data = load_raw_data('./data')
    # test_x_data = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12]],
    # [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]]
    # test_y_data = [[0, 1, 1], [1], [2, 3, 4, 5]]
    for i in range(3):
        for step, (x, y, sequence_length, num_step, mask) in \
                enumerate(data_iterator((test_x_data, test_y_data), 100)):
            print('step:', step, sum(sequence_length), num_step)
        print('_______________')



