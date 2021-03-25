# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import re
from SR_utils import *
from SR_ResNet import *


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # prepare validate datasets
            train_x, train_y, test_x, test_y = load_dataset()

            # Load the modelc
            load_model(args.model)

            # Get input and output tensors, ignore phase_train_placeholder for it have default value.
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

            feature_maps = tf.get_default_graph().get_tensor_by_name('MobileFaceNet/MobileFaceNet/Conv2d_4_InvResBlock_5/Conv/Conv2D:0')

            feature_maps_r = tf.reshape(feature_maps, [-1, 3, 14, 14, 256])
            anchor_feature_maps = feature_maps_r[:, 0, :, :, :]
            pos_feature_maps = feature_maps_r[:, 1, :, :, :]
            neg_feature_maps = feature_maps_r[:, 2, :, :, :]

            anchor_labels = tf.placeholder(tf.int32, [None, 3], name='anchor_labels')
            neg_labels = tf.placeholder(tf.int32, [None, 3], name='neg_labels')
            test_labels = tf.placeholder(tf.int32, [None, 3], name='test_labels')
            learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            anchor_logits, anchor_feature = network(feature_maps=anchor_feature_maps, keep_prob=0.5)
            pos_logits, pos_feature = network(feature_maps=pos_feature_maps, keep_prob=0.5)
            neg_logits, neg_feature = network(feature_maps=neg_feature_maps, keep_prob=0.5)
            test_logits, _ = network(feature_maps=feature_maps, keep_prob=1.0, is_training=False, reuse=True)
            train_accuracy = calculate_accuracy(logit=anchor_logits, label=anchor_labels, name='train_accuracy')
            test_accuracy = calculate_accuracy(logit=test_logits, label=test_labels, name='test_accuracy')

            with tf.name_scope("retrain_loss"):
                pos_pair_loss = tf.losses.mean_squared_error(anchor_feature, pos_feature)
                temp_neg_pair_loss = tf.losses.mean_squared_error(anchor_feature, neg_feature)
                neg_pair_loss = tf.maximum(0.0, 2.0 - temp_neg_pair_loss)

                anchor_loss = cross_entropy_loss(anchor_logits, anchor_labels)
                pos_loss = cross_entropy_loss(pos_logits, anchor_labels)
                neg_loss = cross_entropy_loss(neg_logits, neg_labels)

                loss = anchor_loss + pos_loss + neg_loss + pos_pair_loss + neg_pair_loss

            with tf.name_scope("retrain_op"):  # not shown in the book
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)  # not shown
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="retrain_net")
                train_op = optimizer.minimize(loss, var_list=train_vars)

            uninitialized_vars = []
            for var in tf.all_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)

            init_new_vars_op = tf.initialize_variables(uninitialized_vars)
            sess.run(init_new_vars_op)

            MODEL_PATH = './checkpoint/SR_model/'
            new_saver = tf.train.Saver()

            batch_size = 7
            n_epochs = 45
            train_iteration = len(train_x) // batch_size
            test_iteration = len(test_x) // batch_size

            # loop for epoch
            best_test_accuracy = 0
            epoch_lr = 0.0005
            for epoch in range(0, n_epochs):
                if epoch == int(epoch * 0.33) or epoch == int(epoch * 0.66):
                    epoch_lr = epoch_lr * 0.1

                # get batch data
                for idx in range(0, train_iteration):
                    batch_x = train_x[idx * batch_size:(idx + 1) * batch_size]
                    batch_y = train_y[idx * batch_size:(idx + 1) * batch_size]

                    concat_data = convert_train_data_batch(batch_x)
                    anchor_label_r, neg_label_r = convert_train_label_batch(batch_y)

                    train_feed_dict = {
                        inputs_placeholder: concat_data,
                        anchor_labels: anchor_label_r,
                        neg_labels: neg_label_r,
                        learning_rate: epoch_lr
                    }

                    _, train_accuracy_v, anchor_loss_v, pos_loss_v, neg_loss_v, pos_pair_loss_v, neg_pair_loss_v =\
                        sess.run([train_op, train_accuracy, anchor_loss, pos_loss, neg_loss, pos_pair_loss,
                                  neg_pair_loss], feed_dict=train_feed_dict)

                    if idx % 10 == 0:
                        # display training status
                        print('Epoch: [%2d][%4d/%4d] Anchor Loss %.4f Pos Loss %.4f Neg Loss %.4f '
                              'Pos Pair Loss %.4f Neg Pair Loss %.4f Prec %.4f\t'
                            % (epoch, idx, train_iteration, anchor_loss_v, pos_loss_v, neg_loss_v, pos_pair_loss_v,
                                    neg_pair_loss_v, train_accuracy_v))

                total_test_accuracy = 0
                for idx in range(test_iteration):
                    batch_x = test_x[idx * batch_size:(idx + 1) * batch_size]
                    batch_y = test_y[idx * batch_size:(idx + 1) * batch_size]

                    batch_x_r = convert_test_data_batch(batch_x)
                    batch_y_r = convert_test_label_batch(batch_y)

                    test_feed_dict = {
                        inputs_placeholder: batch_x_r,
                        test_labels: batch_y_r
                    }

                    test_accuracy_v = sess.run(test_accuracy, feed_dict=test_feed_dict)

                    total_test_accuracy += test_accuracy_v
                total_test_accuracy /= test_iteration

                # display training status
                print("Epoch: [%2d/%2d]\ttest_accuracy: %.2f" \
                    % (epoch, n_epochs, total_test_accuracy))

                # save model
                if best_test_accuracy < total_test_accuracy:
                    best_test_accuracy = total_test_accuracy
                    new_saver.save(sess, os.path.join(MODEL_PATH, 'SR_model_%2.4f.ckpt' % (best_test_accuracy)))


def parse_arguments(argv):
    '''test parameters'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='./checkpoint/FR_model')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))