# -*- coding: utf-8 -*-
"""This is a module that collects various util functions."""

import os
import random
import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from PIL import Image

# dataset path
VIDEO_ROOT = '/SSD/DB/Stress_DB_4y'

def check_folder(log_dir):
    """This is a function that check log folder."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    """This is a function that shows all variables."""
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(str_var):
    """This is a function that converts string to bool."""
    return str_var.lower() in 'true'

def load_dataset() :
    """This is a function that loads the train dataset."""

    train_data_list = list()
    test_data_list = list()
    train_label_list = list()
    test_label_list = list()

    subject_test = [1, 8, 33, 16, 30]
    print(subject_test)

    for sub_path in glob.glob(os.path.join(VIDEO_ROOT + '/file_list', "*")):

        subject_num = sub_path.split("-")[0].split("/")[-1]

        if int(subject_num) in subject_test:
            f_sub = open(sub_path, 'r')
            temp_test_paths = f_sub.read().splitlines()
            f_sub.close()

            frame_per_clip = 48
            total_clip_num = len(temp_test_paths) // frame_per_clip

            label_num = sub_path.split("-")[1].split(".")[0]

            if label_num == '0':
                temp_test_label = 0
            elif label_num == '1':
                temp_test_label = 1
            else:
                temp_test_label = 2

            for j in range(total_clip_num):
                test_data_list.append(temp_test_paths[j * frame_per_clip:(j + 1) * frame_per_clip])
                test_label_list.append(temp_test_label)

        else:
            f_sub = open(sub_path, 'r')
            temp_train_paths = f_sub.read().splitlines()
            f_sub.close()

            frame_per_clip = 48
            total_clip_num = len(temp_train_paths) // frame_per_clip

            label_num = sub_path.split("-")[1].split(".")[0]

            if label_num == '0':
                temp_train_label = 0
            elif label_num == '1':
                temp_train_label = 1
            else:
                temp_train_label = 2

            for j in range(total_clip_num):

                pos_sub_num = random.randint(1, 50)
                while pos_sub_num in subject_test or pos_sub_num == int(subject_num):
                    pos_sub_num = random.randint(1, 50)

                if pos_sub_num == 26 and temp_train_label == 2:
                    while pos_sub_num == 26:
                        pos_sub_num = random.randint(1, 50)
                        while pos_sub_num in subject_test:
                            pos_sub_num = random.randint(1, 50)

                pos_path = VIDEO_ROOT + '/file_list/' + '%02d' % pos_sub_num \
                           + '-' + label_num + '.txt'
                f_sub = open(pos_path, 'r')
                pos_train_paths = f_sub.read().splitlines()
                f_sub.close()

                pos_frame_num = len(pos_train_paths)
                pos_frame_index = random.randint(0, pos_frame_num - 48)

                neg_label_num = random.randint(0, 2)
                while neg_label_num == temp_train_label:
                    neg_label_num = random.randint(0, 2)
                if int(subject_num) == 26 and temp_train_label == 0:
                    neg_label_num = 1
                elif int(subject_num) == 26 and temp_train_label == 1:
                    neg_label_num = 0

                neg_path = VIDEO_ROOT + '/file_list/' + subject_num + '-' + \
                           str(neg_label_num) + '.txt'
                f_sub = open(neg_path, 'r')
                neg_train_paths = f_sub.read().splitlines()
                f_sub.close()

                neg_frame_num = len(neg_train_paths)
                neg_frame_index = random.randint(0, neg_frame_num - 48)

                train_data_list.append((temp_train_paths[j * frame_per_clip:
                                                         (j + 1) * frame_per_clip],
                                        pos_train_paths[pos_frame_index:pos_frame_index + 48],
                                        neg_train_paths[neg_frame_index:neg_frame_index + 48]))
                train_label_list.append((temp_train_label, neg_label_num))

    train_data = np.array(train_data_list)
    train_labels = np.array(train_label_list)
    test_data = np.array(test_data_list)
    test_labels = np.array(test_label_list)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)

    return train_data, train_labels, test_data, test_labels

def load_test_dataset() :
    """This is a function that loads the test dataset."""

    train_data_list = list()
    test_data_list = list()
    train_label_list = list()
    test_label_list = list()

    subject_test = [1, 8, 33, 16, 30]
    print(subject_test)

    for sub_path in glob.glob(os.path.join(VIDEO_ROOT + '/file_list', "*")):

        subject_num = sub_path.split("-")[0].split("/")[-1]

        if int(subject_num) in subject_test:
            f_sub = open(sub_path, 'r')
            temp_test_paths = f_sub.read().splitlines()
            f_sub.close()

            frame_per_clip = 48
            total_clip_num = len(temp_test_paths) // frame_per_clip

            label_num = sub_path.split("-")[1].split(".")[0]

            if label_num == '0':
                temp_test_label = 0
            elif label_num == '1':
                temp_test_label = 1
            else:
                temp_test_label = 2

            for j in range(total_clip_num):
                test_data_list.append(temp_test_paths[j * frame_per_clip:(j + 1) * frame_per_clip])
                test_label_list.append(temp_test_label)

    train_data = np.array(train_data_list)
    train_labels = np.array(train_label_list)
    test_data = np.array(test_data_list)
    test_labels = np.array(test_label_list)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_labels)

    return train_data, train_labels, test_data, test_labels

def one_hot_encoding(target):
    """This is a function that converts the number into one hot vector."""
    if len(target) == 0:
        result = np.array([])
    else:
        result = np.eye(3)[target]

    return result

def convert_train_data_batch(data_batch):
    """This is a function that converts train data into batch."""
    anchor_data_list = []
    pos_data_list = []
    neg_data_list = []

    for i in range(data_batch.shape[0]):
        for j in range(data_batch.shape[2]):
            anchor_img = np.array(Image.open(VIDEO_ROOT + '/' + data_batch[i][0][j]))
            anchor_data_list.append(anchor_img)

            pos_img = np.array(Image.open(VIDEO_ROOT + '/' + data_batch[i][1][j]))
            pos_data_list.append(pos_img)

            neg_img = np.array(Image.open(VIDEO_ROOT + '/' + data_batch[i][2][j]))
            neg_data_list.append(neg_img)

    anchor_data = np.array(anchor_data_list)
    pos_data = np.array(pos_data_list)
    neg_data = np.array(neg_data_list)

    anchor_data_r = anchor_data.reshape(-1, 112, 112, 3)
    pos_data_r = pos_data.reshape(-1, 112, 112, 3)
    neg_data_r = neg_data.reshape(-1, 112, 112, 3)

    concat_data_stack = np.concatenate((anchor_data_r, pos_data_r, neg_data_r), axis=1)
    concat_data_stack_r = concat_data_stack.reshape(-1, 112, 112, 3)

    return concat_data_stack_r

def convert_train_label_batch(label_batch):
    """This is a function that converts train label into batch."""
    anchor_label_list = []
    neg_label_list = []

    for i in range(label_batch.shape[0]):
        anchor_label_list.append(label_batch[i][0])
        neg_label_list.append(label_batch[i][1])

    anchor_label = np.array(anchor_label_list)
    neg_label = np.array(neg_label_list)

    anchor_label_r = one_hot_encoding(anchor_label)
    neg_label_r = one_hot_encoding(neg_label)

    return anchor_label_r, neg_label_r

def convert_test_data_batch(data_batch):
    """This is a function that converts test data into batch."""
    anchor_data_list = []
    for i in range(data_batch.shape[0]):
        for j in range(data_batch.shape[1]):
            anchor_img = np.array(Image.open(VIDEO_ROOT + '/' + data_batch[i][j]))
            anchor_data_list.append(anchor_img)

    anchor_data = np.array(anchor_data_list)
    anchor_data_r = anchor_data.reshape(-1, 112, 112, 3)

    return anchor_data_r

def convert_test_label_batch(label_batch):
    """This is a function that converts test label into batch."""
    anchor_label_list = []
    for i in range(label_batch.shape[0]):
        anchor_label_list.append(label_batch[i])

    anchor_label = np.array(anchor_label_list)
    anchor_label_r = one_hot_encoding(anchor_label)

    return anchor_label_r
