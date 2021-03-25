# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from SR_utils import *
from SR_ResNet import *
import cv2
import mtcnn.mtcnn as mtcnn


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


def read_test_data(target_path):
    mt = mtcnn.Mtcnn()

    cam = cv2.VideoCapture(target_path)
    temp_img = []
    while (True):
        ret, frame = cam.read()

        if ret:
            temp_img.append(frame)
        else:
            break

    total_frame = len(temp_img)

    img_list = []

    temp_x, temp_y, temp_w, temp_h = 0, 0, 0, 0

    if total_frame < 48:
        plus_index = []
        interval = (total_frame - 0.51) / 48
        for i in range(48):
            plus_index.append(round(interval * i))
        cnt = []
        for i in range(total_frame):
            cnt.append(plus_index.count(i))
    elif total_frame == 48:
        cnt = []
        for i in range(total_frame):
            cnt.append(1)
    else:
        plus_index = []
        interval = (total_frame + 0.49) / 48
        for i in range(48):
            plus_index.append(round(interval * i))
        cnt = []
        for i in range(total_frame):
            cnt.append(plus_index.count(i))

    for i in range(total_frame):
        image = temp_img[i]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mt_out, bbox = mt.mtcnn(gray)

        if mt_out is None:
            if (temp_x, temp_y, temp_w, temp_h) == (0, 0, 0, 0):
                continue

            (x, y, w, h) = (temp_x, temp_y, temp_w, temp_h)

            resized_img = cv2.resize(image[y:y + h, x:x + w], (112, 112),
                                     interpolation=cv2.INTER_AREA)

        else:
            (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1]))

            if (temp_x, temp_y, temp_w, temp_h) == (0, 0, 0, 0):
                (temp_x, temp_y, temp_w, temp_h) = (x, y, w, h)

            if w < (temp_w / 1.2) or w > (temp_w / 0.8) or h < (temp_h / 1.2) or h > (
                    temp_h / 0.8) or \
                    x < temp_x - (temp_w / 4) or x > temp_x + (temp_w / 4) or y < temp_y - (
                    temp_h / 4) or y > temp_y + (temp_h / 4):
                (x, y, w, h) = (temp_x, temp_y, temp_w, temp_h)

            temp_x = x
            temp_y = y
            temp_w = w
            temp_h = h

            resized_img = cv2.resize(image[y:y + h, x:x + w], (112, 112),
                                     interpolation=cv2.INTER_AREA)

        face_t = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        for k in range(cnt[i]):
            img_list.append(face_t)

    img_data = np.array(img_list)

    return img_data


def recognize_test_data(img_data):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            FR_MODEL_PATH = './checkpoint/FR_model/'
            SR_MODEL_PATH = './checkpoint/SR_model/'

            # Load the modelc
            load_model(FR_MODEL_PATH)

            # Get input and output tensors, ignore phase_train_placeholder for it have default value.
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

            feature_maps = tf.get_default_graph().get_tensor_by_name('MobileFaceNet/MobileFaceNet/Conv2d_4_InvResBlock_5/Conv/Conv2D:0')
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            test_logits, _ = network(feature_maps=feature_maps, keep_prob=1.0, is_training=False, reuse=tf.AUTO_REUSE)

            restore_saver = tf.train.Saver()
            tf.global_variables_initializer().run()
            restore_saver.restore(sess, os.path.join(SR_MODEL_PATH, 'SR_model_0.6233.ckpt'))

            test_feed_dict = {
                inputs_placeholder: img_data
            }

            emb_array, test_logits_v = sess.run([embeddings, test_logits], feed_dict=test_feed_dict)
            FR_feature = emb_array.mean(axis=0)

            SR_result = np.array(test_logits_v)
            SR_output = SR_result.argmax()
            print("SR Output: {}".format(SR_output))

    return SR_output, FR_feature


def StressRecognition(path):
    img_data = read_test_data(path)
    SR_output, FR_feature = recognize_test_data(img_data)

    return SR_output, FR_feature
