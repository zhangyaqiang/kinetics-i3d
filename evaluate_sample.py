# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import glob
import os
import skvideo.io

import i3d

_IMAGE_SIZE = 224
_NUM_CLASSES = 200

_SAMPLE_VIDEO_FRAMES = None
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_MODEL_PATH = "/home/zyq/moment_in_time/code/kinetics-i3d/data/checkpoints/rgb_mmt/model.ckpt-3"

_LABEL_MAP_PATH = "/home/zyq/moment_in_time/code/data/moments_categories.txt"

_VIDEO_PATH = "/home/zyq/moment_in_time/momentsMiniTest"

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'rgb', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')


def get_label(label_file):
    num2class = {}
    class2num = {}
    with open(label_file) as f:
        lines = f.readlines()
        for line in lines:
            class_name = line.split(',')[0]
            class_num = line.split(',')[1]
            class2num[class_name] = class_num
            num2class[class_num] = class_name
    return num2class, class2num

def get_video_frames(video_path):
    videogen = skvideo.io.vreader(video_path)
    frames = []
    for frame in videogen:
        frames.append(frame[16:-16, 16:-16])
    # frames = np.asarray([frame for frame in videogen])
    frames = np.asarray(frames)
    frames = np.divide(frames, 256.0)
    frames = np.subtract(frames, 0.5)
    frames = np.multiply(frames, 2.0)
    return frames

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type
  imagenet_pretrained = FLAGS.imagenet_pretrained


  test_video_list = glob.glob(os.path.join(_VIDEO_PATH, "*mp4"))

  if eval_type not in ['rgb', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')

  # kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]
  num2class, class2num = get_label(_LABEL_MAP_PATH)

  if eval_type in ['rgb', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable
    # rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    rgb_saver = tf.train.Saver()

  if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
    flow_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
    with tf.variable_scope('Flow'):
      flow_model = i3d.InceptionI3d(
          _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      flow_logits, _ = flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)
    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

  if eval_type == 'rgb':
    model_logits = rgb_logits
  elif eval_type == 'flow':
    model_logits = flow_logits
  else:
    model_logits = rgb_logits + flow_logits
  model_predictions = tf.nn.softmax(model_logits)

  with tf.Session() as sess:
    feed_dict = {}
    rgb_saver.restore(sess, _MODEL_PATH)
    # if eval_type in ['rgb', 'joint']:
    #   if imagenet_pretrained:
    #     rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
    #   else:
    #     rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
    #   tf.logging.info('RGB checkpoint restored')
    #   rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
    #   tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
    #   feed_dict[rgb_input] = rgb_sample
    #
    # if eval_type in ['flow', 'joint']:
    #   if imagenet_pretrained:
    #     flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
    #   else:
    #     flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
    #   tf.logging.info('Flow checkpoint restored')
    #   flow_sample = np.load(_SAMPLE_PATHS['flow'])
    #   tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
    #   feed_dict[flow_input] = flow_sample
    with open("/home/zyq/moment_in_time/result.txt", "w") as f:
        for video in test_video_list:
            rgb_sample = get_video_frames(video)
            rgb_sample = np.expand_dims(rgb_sample, 0)
            feed_dict[rgb_input] = rgb_sample
            out_logits, out_predictions = sess.run(
                [model_logits, model_predictions],
                feed_dict=feed_dict)

            out_logits = out_logits[0]
            out_predictions = out_predictions[0]
            sorted_indices = np.argsort(out_predictions)[::-1]

            # print('Norm of logits: %f' % np.linalg.norm(out_logits))
            # print('\nTop classes and probabilities')
            f.write(os.path.basename(video))
            f.write(" ")
            for index in sorted_indices[:5]:
              # print(out_predictions[index], out_logits[index], kinetics_classes[index])
              f.write(str(index))
              f.write(" ")
            print("{} has done." . format(video))
            f.write("\n")

if __name__ == '__main__':
  tf.app.run(main)
