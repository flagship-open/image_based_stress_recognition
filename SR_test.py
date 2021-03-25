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
            _, _, test_x, test_y = load_test_dataset()

            # Load the modelc
            load_model(args.model)

            # Get input and output tensors, ignore phase_train_placeholder for it have default value.
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

            feature_maps = tf.get_default_graph().get_tensor_by_name('MobileFaceNet/MobileFaceNet/Conv2d_4_InvResBlock_5/Conv/Conv2D:0')

            test_labels = tf.placeholder(tf.int32, [None, 3], name='test_labels')
            test_logits, _ = network(feature_maps=feature_maps, keep_prob=1.0, is_training=False, reuse=tf.AUTO_REUSE)
            test_accuracy = calculate_accuracy(logit=test_logits, label=test_labels, name='test_accuracy')

            MODEL_PATH = './checkpoint/SR_model/'
            restore_saver = tf.train.Saver()
            tf.global_variables_initializer().run()
            restore_saver.restore(sess, os.path.join(MODEL_PATH, 'SR_model_0.6233.ckpt'))

            batch_size = 7
            test_iteration = len(test_x) // batch_size

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
            print("Test Accuracy: %.4f" % (total_test_accuracy))

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