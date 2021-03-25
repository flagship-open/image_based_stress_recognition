import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np
import random
import glob
from PIL import Image

# dataset path
video_root = '/SSD/DB/Stress_DB_4y'

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

def load_dataset() :

    train_data_list = list()
    test_data_list = list()
    train_label_list = list()
    test_label_list = list()

    subject_test = [1, 8, 33, 16, 30]
    print(subject_test)

    for a in glob.glob(os.path.join(video_root + '/file_list', "*")):

        subject_num = a.split("-")[0].split("/")[-1]

        if int(subject_num) in subject_test:
            f = open(a, 'r')
            temp_test_paths = f.read().splitlines()
            f.close()

            frame_per_clip = 48
            total_clip_num = len(temp_test_paths) // frame_per_clip

            label_num = a.split("-")[1].split(".")[0]

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
            f = open(a, 'r')
            temp_train_paths = f.read().splitlines()
            f.close()

            frame_per_clip = 48
            total_clip_num = len(temp_train_paths) // frame_per_clip

            label_num = a.split("-")[1].split(".")[0]

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

                pos_path = video_root + '/file_list/' + '%02d' % (pos_sub_num) + '-' + label_num + '.txt'
                f = open(pos_path, 'r')
                pos_train_paths = f.read().splitlines()
                f.close()

                pos_frame_num = len(pos_train_paths)
                pos_frame_index = random.randint(0, pos_frame_num - 48)

                neg_label_num = random.randint(0, 2)
                while neg_label_num == temp_train_label:
                    neg_label_num = random.randint(0, 2)
                if int(subject_num) == 26 and temp_train_label == 0:
                    neg_label_num = 1
                elif int(subject_num) == 26 and temp_train_label == 1:
                    neg_label_num = 0

                neg_path = video_root + '/file_list/' + subject_num + '-' + str(neg_label_num) + '.txt'
                f = open(neg_path, 'r')
                neg_train_paths = f.read().splitlines()
                f.close()

                neg_frame_num = len(neg_train_paths)
                neg_frame_index = random.randint(0, neg_frame_num - 48)

                train_data_list.append((temp_train_paths[j * frame_per_clip:(j + 1) * frame_per_clip],
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

    train_data_list = list()
    test_data_list = list()
    train_label_list = list()
    test_label_list = list()

    subject_test = [1, 8, 33, 16, 30]
    print(subject_test)

    for a in glob.glob(os.path.join(video_root + '/file_list', "*")):

        subject_num = a.split("-")[0].split("/")[-1]

        if int(subject_num) in subject_test:
            f = open(a, 'r')
            temp_test_paths = f.read().splitlines()
            f.close()

            frame_per_clip = 48
            total_clip_num = len(temp_test_paths) // frame_per_clip

            label_num = a.split("-")[1].split(".")[0]

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
    if len(target) == 0:
        return np.array([])
    else:
        num = 3

    return np.eye(num)[target]

def convert_train_data_batch(data_batch):
    anchor_data_list = []
    pos_data_list = []
    neg_data_list = []

    for i in range(data_batch.shape[0]):
        for j in range(data_batch.shape[2]):
            anchor_img = np.array(Image.open(video_root + '/' + data_batch[i][0][j]))
            anchor_data_list.append(anchor_img)

            pos_img = np.array(Image.open(video_root + '/' + data_batch[i][1][j]))
            pos_data_list.append(pos_img)

            neg_img = np.array(Image.open(video_root + '/' + data_batch[i][2][j]))
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
    anchor_data_list = []
    for i in range(data_batch.shape[0]):
        for j in range(data_batch.shape[1]):
            anchor_img = np.array(Image.open(video_root + '/' + data_batch[i][j]))
            anchor_data_list.append(anchor_img)

    anchor_data = np.array(anchor_data_list)
    anchor_data_r = anchor_data.reshape(-1, 112, 112, 3)

    return anchor_data_r

def convert_test_label_batch(label_batch):
    anchor_label_list = []
    for i in range(label_batch.shape[0]):
        anchor_label_list.append(label_batch[i])

    anchor_label = np.array(anchor_label_list)
    anchor_label_r = one_hot_encoding(anchor_label)

    return anchor_label_r