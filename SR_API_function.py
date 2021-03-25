# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from SR_utils import *
from SR_ResNet import *
import cv2
import mtcnn.mtcnn_API as mtcnn


def load_feature_map_model(FEATURE_MAP_MODEL_PATH, config):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    sess = tf.Session(config=config)
    model_exp = os.path.expanduser(FEATURE_MAP_MODEL_PATH)
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
        saver.restore(sess, os.path.join(model_exp, ckpt_file))

    return sess

def load_stress_recognition_model(SR_MODEL_PATH, config):
    sess = tf.Session(config=config)

    features_placeholder = tf.placeholder(tf.float32, [48, 14, 14, 256])

    test_logits, _ = network(feature_maps=features_placeholder, keep_prob=1.0, is_training=False,
                             reuse=tf.AUTO_REUSE)

    restore_saver = tf.train.Saver()
    tf.global_variables_initializer().run(session=sess)
    restore_saver.restore(sess, os.path.join(SR_MODEL_PATH))

    return sess

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


def image_preprocessing(target_path, mtcnn_model):

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
        mt_out, bbox = mtcnn_model.mtcnn(gray)

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


def extract_image_feature_map(sess, img_data):

    # Get input and output tensors, ignore phase_train_placeholder for it have default value.
    inputs_placeholder = sess.graph.get_tensor_by_name("input:0")

    feature_maps = sess.graph.get_tensor_by_name('MobileFaceNet/MobileFaceNet/Conv2d_4_InvResBlock_5/Conv/Conv2D:0')

    test_feed_dict = {
        inputs_placeholder: img_data
    }

    feature_maps_v = sess.run(feature_maps, feed_dict=test_feed_dict)

    return feature_maps_v


def stress_recognition(sess, feature_map):
    features_placeholder = tf.placeholder(tf.float32, [48, 14, 14, 256])

    test_logits, _ = network(feature_maps=features_placeholder, keep_prob=1.0, is_training=False, reuse=tf.AUTO_REUSE)

    test_feed_dict = {
        features_placeholder: feature_map
    }

    test_logits_v = sess.run(test_logits, feed_dict=test_feed_dict)

    SR_result = np.array(test_logits_v)
    SR_output = SR_result.argmax()

    return SR_output


if __name__ == '__main__':
    # Session setting
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4 # gpu 사용 비율

    sample_path = 'APITestData/AT-001-03_23.mp4' # 샘플 비디오 경로
    feature_map_model_path = 'checkpoint/FR_model' # feature map 모델 경로
    stress_recognition_model_path = 'checkpoint/SR_model/SR_model_0.6233.ckpt' # 스트레스 인식 모델 경로

    sess_feature_map = load_feature_map_model(feature_map_model_path, config)
    sess_stress_recognition = load_stress_recognition_model(stress_recognition_model_path, config)
    mtcnn_model = mtcnn.Mtcnn(config) # 전처리 모델

    img_data = image_preprocessing(sample_path, mtcnn_model)
    feature_map = extract_image_feature_map(sess_feature_map, img_data)
    SR_output = stress_recognition(sess_stress_recognition, feature_map)

    print("SR Output: {}".format(SR_output))

    sess_feature_map.close()
    sess_stress_recognition.close()