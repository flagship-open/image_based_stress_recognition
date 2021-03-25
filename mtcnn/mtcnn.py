"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
from PIL import Image
import numpy as np

import os, glob
import time

import tensorflow as tf
import numpy as np
import mtcnn.detect_face as detect_face
import mtcnn.facenet as facenet
import random
from time import sleep
class Mtcnn():
    def __init__(self):
        sleep(random.random())

        # self.random_order = True
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # grow memory usage as required
        config.gpu_options.per_process_gpu_memory_fraction = 0.8  # limit memory usage up to 80%
        self.sess = tf.Session(config=config)
        # with sess.as_default():
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)
        # self.mtcnn(self.input_dir,self.output_dir,self.image_size,self.margin,self.gpu_memory_fraction,self.detect_multiple_faces)

    def detection_return(self):
        return self.detection

    def mtcnn(self,image ,image_size = 182, margin = 44, gpu_memory_fraction = 1.0, detect_multiple_faces = False):
        # output_dir = os.path.expanduser(output_dir)

        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        # Store some git revision info in a text file in the log directory
        # src_path,_ = os.path.split(os.path.realpath(__file__))
        #
        # # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))

        # dataset = facenet.get_dataset(input_dir)

        # for f in glob.glob(os.path.join(input_dir,'*.jpg')):
        #     dataset.append(f)
        # print(dataset)
        # print('Creating networks and loading parameters')
        self.detection = True
        # with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))


        minsize = 20 # minimum size of face
        threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        factor = 0.709 # scale factor

        # Add a random key to the filename to allow alignment using multiple processes
        # random_key = np.random.randint(0, high=99999)
        # bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_temp.txt')
        #
        # with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        # if random_order:
        #     random.shuffle(dataset)
        for i in range(0,1):
            # image_path = dataset

            nrof_images_total += 1
            # filename = os.path.splitext(os.path.split(image_path)[1])[0]
            # print(filename)
            # filename = 'temp'
            # output_filename = os.path.join(output_dir, filename+'.jpg')
            # print(image_path)
            # if not os.path.exists(output_filename):
            # try:
                # img = misc.imread(image_path)
            img = image
            # except (IOError, ValueError, IndexError) as e:
            #     errorMessage = '{}: {}'.format(image_path, e)
            #     print(errorMessage)
            # else:
            # if img.ndim<2:
            #     print('Unable to align "%s"' % image_path)
            #     text_file.write('%s\n' % (output_filename))
            #     continue
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = img[:,:,0:3]

            bounding_boxes, _ = detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces>0:
                det = bounding_boxes[:,0:4]
                det_arr = []
                img_size = np.asarray(img.shape)[0:2]
                if nrof_faces>1:
                    if detect_multiple_faces:
                        for i in range(nrof_faces):
                            det_arr.append(np.squeeze(det[i]))
                    else:
                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                        img_center = img_size / 2
                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                        det_arr.append(det[index,:])
                else:
                    det_arr.append(np.squeeze(det))

                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-margin/2, 0)
                    bb[1] = np.maximum(det[1]-margin/2, 0)
                    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    # scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    scaled = np.array(Image.fromarray(cropped).resize(size=(image_size, image_size)))
                    nrof_successfully_aligned += 1
                    # filename_base, file_extension = os.path.splitext(output_filename)
                    # if detect_multiple_faces:
                    #     output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                    # else:
                    #     output_filename_n = "{}{}".format(filename_base, file_extension)
                    # misc.imsave(output_filename_n, scaled)
                    # text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
            else:
                # print('Unable to align')
                print()

                # text_file.write('%s\n' % (output_filename))

        # print('Total number of images: %d' % nrof_images_total)
        # print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
        if nrof_successfully_aligned != 0:
            self.detection = True
            return scaled, bb
        else:
            return None, None
        # sess.close()

    def __del__(self):
        self.sess.close()
#
#
# MT = Mtcnn('./temp/temp.jpg','./temp',image_size=160,margin=32)
# MT.start()
# sleep(3)
# MT.stop()

#
# def parse_arguments(argv):
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
#     parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
#     parser.add_argument('--image_size', type=int,
#         help='Image size (height, width) in pixels.', default=182)
#     parser.add_argument('--margin', type=int,
#         help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
#     parser.add_argument('--random_order',
#         help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
#     parser.add_argument('--gpu_memory_fraction', type=float,
#         help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
#     parser.add_argument('--detect_multiple_faces', type=bool,
#                         help='Detect and align multiple faces per image.', default=False)
#     return parser.parse_args(argv)
#
# if __name__ == '__main__':
#     main(parse_arguments(sys.argv[1:]))
