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
from datetime import datetime
import os.path
import re
import time
import collections

import os, glob
import numpy as np
import tensorflow as tf

import i3d
from reader import get_two_stream_input, get_rgb_input, get_flow_input
from input import build_dataset

_IMAGE_SIZE = 224
_NUM_CLASSES = 200
_NUM_EPOCH = 40

max_steps = 50000

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
    'rgb_mmt': 'data/checkpoints/rgb_mmt/model.ckpt',
    'rgb_mmt_2': 'data/checkpoints/rgb_mmt/model.ckpt_0604',
    'flow_mmt': 'data/checkpoints/flow_mmt/model.ckpt'
}

_MODEL_PATH = "/home/zyq/moment_in_time/code/kinetics-i3d/data/checkpoints/rgb_mmt/model.ckpt-4"

_LABEL_MAP_PATH = '/home/zyq/moment_in_time/Moments_in_Time_Mini/moments_categories.txt'
_TRAIN_LIST_PATH = '/home/zyq/moment_in_time/train_list.txt'
_VAl_LIST_PATH = '/home/zyq/moment_in_time/validation_list.txt'
_TFRECORD_PATH = '/home/zyq/moment_in_time/dataset/'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')


def i3d_loss(scope, rgb, flow, label, rgb_model, flow_model, gpu, confidence):
    """
    Builds an I3D model and computes the loss
    """
    cr_rgb = None
    ce_flow = None
    if rgb_model is not None:
        rgb_logits = rgb_model(rgb, is_training=True, dropout_keep_prob=1.0)[0]
        rgb_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label, logits=rgb_logits, name='cross_entropy_rgb')
        #confidence
        # confidence = tf.divide(confidence, 3)
        # confidence = tf.cast(confidence, dtype=tf.float32)
        # rgb_loss = tf.multiply(rgb_loss, confidence)
        ce_rgb = tf.reduce_mean(rgb_loss, name='rgb_ce')
        tf.summary.scalar('rgb_%d' % gpu, ce_rgb)

    if flow_model is not None:
        flow_logits = flow_model(flow, is_training=True, dropout_keep_prob=1.0)[0]
        flow_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label, logits=flow_logits, name='cross_entropy_flow')
        ce_flow = tf.reduce_mean(flow_loss, name='flow_ce')
        tf.summary.scalar('flow_%d' % gpu, ce_flow)

    return ce_rgb, ce_flow

def i3d_accuracy(scope, rgb, flow, label, rgb_model, flow_model, gpu):
    if rgb_model is not None:
        rgb_logits = rgb_model(rgb, is_training=False, dropout_keep_prob=1.0)[0]

    if flow_model is not None:
        flow_logits = flow_model(flow, is_training=False, dropout_keep_prob=1.0)[0]

    if rgb_model is not None and flow_model is not None:
        logits = rgb_logits + flow_logits
    elif rgb_logits is not None:
        logits = rgb_logits
    else:
        logits = flow_logits

    # top1_acc, acc_op = tf.metrics.accuracy(labels=label, predictions=tf.argmax(logits, 1), name="top1_accuracy")
    top1_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=label, k=1), dtype=tf.float32))
    top5 = tf.nn.top_k(logits, 5)
    top5_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=label, k=5), dtype=tf.float32))
    return top1_acc, top5_acc

def average_gradients(grads):
    """
    Averages all the gradients across the GPUs
    """
    average_grads = []
    for grad_and_vars in zip(*grads):
        gr = []
        # print grad_and_vars
        for g, _ in grad_and_vars:
            if g is None:
                continue
            exp_g = tf.expand_dims(g, 0)
            gr.append(exp_g)
        if len(gr) == 0:
            continue
        grad = tf.concat(axis=0, values=gr)
        grad = tf.reduce_mean(grad, 0)

        # remove redundant vars (because they are shared across all GPUs)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

# def eval(batch_size=64, rgb_model=None, flow_model=None, mode='rgb'):
#     if 'joint' in mode or 'rgb' in mode:
#         with tf.variable_scope('RGB'):
#             rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
#     if 'joint' in mode or 'flow' in mode:
#         with tf.variable_scope('Flow'):
#             flow_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')

def train(split=-1, batch_size=10, num_gpus=1, mode='rgb'):

    train_list = glob.glob(os.path.join(_TFRECORD_PATH, '*train*'))
    val_list = glob.glob(os.path.join(_TFRECORD_PATH, '*val*'))

    train_dataset = build_dataset(train_list, batch_size=batch_size, is_training=True)
    val_dataset = build_dataset(val_list, batch_size=batch_size, is_training=False)
    iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    with tf.device('/cpu:0'):
        # count number of train calls
        # global_step = tf.get_variable('global_step', initializer=tf.constant(0), trainable=False)

        # create the networks
        if 'joint' in mode or 'rgb' in mode:
            with tf.variable_scope('RGB'):
                rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
        if 'joint' in mode or 'flow' in mode:
            with tf.variable_scope('Flow'):
                flow_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')

        # initial learning rate
        # lr = tf.Variable(0.01)
        lr = tf.Variable(0.001)

        # create optimizer
        # opt = tf.train.AdamOptimizer()
        opt = tf.train.MomentumOptimizer(lr, 0.9)

        # if 'joint' in mode:
        #     # get dataset
        #     rgbs, flows, labels, confidence = get_two_stream_input(_TRAIN_LIST_PATH, batch_size)
        #     batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([rgbs, flows, labels, confidence], capacity=16 * num_gpus)
        # elif 'rgb' in mode:
        #     rgbs, labels, confidence = get_rgb_input(_TRAIN_LIST_PATH, batch_size)
        #     batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([rgbs, labels, confidence], capacity=16 * num_gpus)
        # else:
        #     flows, labels, confidence = get_flow_input(_TRAIN_LIST_PATH, batch_size)
        #     batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([flows, labels, confidence], capacity=16 * num_gpus)


            # create model on each GPU and get gradients for each
        # all_rgb_grads = []
        # all_flow_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            # for each GPU
            # for i in range(num_gpus):
            i = 0
            with tf.device('gpu:%d' % i):
                with tf.name_scope('%s_%d' % ('I3D', i)) as scope:
                    rgb_batch, flow_batch, label_batch, confidence = iterator.get_next()
                    # dequeue a batch
                    if 'joint' in mode:
                        # rgb_batch, flow_batch, label_batch, confidence = batch_queue.dequeue()
                        # construct I3D while sharing all variables
                        # but compute the loss for each GPU
                        rgb_loss, flow_loss = i3d_loss(scope, rgb_batch, flow_batch, label_batch, rgb_model,
                                                       flow_model, i, confidence)
                        top1_acc, top5_acc = i3d_accuracy(scope, rgb_batch, flow_batch, label_batch, rgb_model,
                                                          flow_model, i)
                    elif 'rgb' in mode:
                        # rgb_batch, label_batch, confidence = batch_queue.dequeue()
                        rgb_loss, flow_loss = i3d_loss(scope, rgb_batch, None, label_batch, rgb_model, None, i, confidence)
                        top1_acc, top5_acc = i3d_accuracy(scope, rgb_batch, None, label_batch, rgb_model, None, i)
                    else:
                        # flow_batch, label_batch, confidence = batch_queue.dequeue()
                        rgb_loss, flow_loss = i3d_loss(scope, None, flow_batch, label_batch, None, flow_model, i, confidence)
                        top1_acc, top5_acc = i3d_accuracy(scope, None, flow_batch, label_batch, None, flow_model, i)

                    # reuse the variables on next GPU
                    tf.get_variable_scope().reuse_variables()

                    # retain summaries
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # calculate gradients for this tower
                    # track all gradients
                    if rgb_loss is not None:
                        grads_rgb = opt.compute_gradients(rgb_loss)
                        # all_rgb_grads.append(grads_rgb)

                    if flow_loss is not None:
                        grads_flow = opt.compute_gradients(flow_loss)
                        # all_flow_grads.append(grads_flow)


        finetune_grads = []
        # sync and average grads
        if 'joint' in mode or 'rgb' in mode:
            # rgb_grads = average_gradients(all_rgb_grads)
            rgb_grads = grads_rgb
        if 'joint' in mode or 'flow' in mode:
            # flow_grads = average_gradients(all_flow_grads)
            flow_grads = grads_flow

        # track lr
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # track grads:
        if 'joint' in mode or 'rgb' in mode:
            for grad, var in rgb_grads:
                if grad is not None and "Logits" in grad.name:
                    finetune_grads.append((grad, var))
                    summaries.append(tf.summary.histogram(var.op.name + '/rgb_gradients', grad))
        # track grads:
        if 'joint' in mode or 'flow' in mode:
            for grad, var in flow_grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/flow_gradients', grad))

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

        # init new layeres
        new_layers = [lr]

        # load pretrained weights
        if 'joint' in mode or 'rgb' in mode:
            rgb_variable_map = {}
            rgb_final_map = {}
            for variable in tf.global_variables():
                if variable.name.split('/')[0] == 'RGB':
                    if 'Logits' not in variable.name:
                        rgb_variable_map[variable.name.replace(':0', '')] = variable
                    else:
                        new_layers.append(variable)
                    rgb_final_map[variable.name.replace(':0', '')] = variable
            # rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
            rgb_saver = tf.train.Saver(var_list=tf.trainable_variables())
            rgb_saver.restore(sess, _MODEL_PATH)
            # rgb_saver = tf.train.Saver(var_list=rgb_final_map, reshape=True)
            rgb_saver = tf.train.Saver()
            # rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_mmt'])

        if 'joint' in mode or 'flow' in mode:
            flow_variable_map = {}
            flow_final_map = {}
            for variable in tf.global_variables():
                if variable.name.split('/')[0] == 'Flow':
                    if 'Logits' not in variable.name:
                        flow_variable_map[variable.name.replace(':0', '')] = variable
                    else:
                        new_layers.append(variable)
                    flow_final_map[variable.name.replace(':0', '')] = variable
            flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
            flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
            # flow_saver = tf.train.Saver(var_list=flow_final_map, reshape=True)
            flow_saver = tf.train.Saver()
            # flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_mmt'])

        # apply gradients
        if 'joint' in mode or 'rgb' in mode:
            # apply_rgb_grad_op = opt.apply_gradients(rgb_grads)  # , global_step=global_step)
            apply_rgb_grad_op = opt.apply_gradients(finetune_grads)  # , global_step=global_step)
        if 'joint' in mode or 'flow' in mode:
            apply_flow_grad_op = opt.apply_gradients(flow_grads)  # , global_step=global_step)

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        if 'joint' in mode:
            train_op = tf.group(apply_rgb_grad_op, apply_flow_grad_op)
        elif 'rgb' in mode:
            train_op = apply_rgb_grad_op
        else:
            train_op = apply_flow_grad_op

        summary_op = tf.summary.merge(summaries)


        # init all momentum vars
        for var in tf.global_variables():
            if 'Momentum' in var.name or 'I3D_0/mean/total' in var.name or "top1_accuracy" in var.name:
            # if 'beta' in var.name or 'Adam' in var.name or "top1_accuracy" in var.name:
                new_layers.append(var)
        # init new layers
        init = tf.variables_initializer(new_layers)
        sess.run(init)
        # sess.run(tf.local_variables_initializer())

        # begin queues
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter('./logs/0605_rgb', sess.graph)

        if 'joint' in mode:
            loss = rgb_loss + flow_loss
        elif 'rgb' in mode:
            loss = rgb_loss
        else:
            loss = flow_loss

        losses = collections.deque(maxlen=10)
        last_loss = None

        #-------------------train-------------------#
        step = 0
        for epoch in range(_NUM_EPOCH):
            print('[Epoch %d] train begin ' % epoch)
            sess.run(train_init_op)
            while True:
                try:
                    start_time = time.time()
                    _, loss_value, lra = sess.run([train_op, loss, lr])
                    duration = time.time() - start_time

                    # update learning rate when loss saturates
                    losses.append(loss_value)

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % 10 == 0 and step > 0:
                        # check if loss is saturated
                        if last_loss is None:
                            last_loss = sum(losses) / 10
                        else:
                            diff = last_loss - sum(losses) / 10
                            print('Diff:', diff)
                            last_loss = sum(losses) / 10
                            # if diff < 0.001:
                            #     lr /= 10.

                        num_examples_per_step = batch_size * num_gpus
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / num_gpus

                        format_str = ('Mode:%s %s: step %d, avg loss = %.2f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print(format_str % (mode, datetime.now(), step, sum(losses) / 10,
                                            examples_per_sec, sec_per_batch))
                        # print(lra)
                    step += 1
                    if step % 100 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
                except:
                    break
            lr /= 5
            if 'joint' in mode or 'rgb' in mode:
                rgb_saver.save(sess, _CHECKPOINT_PATHS['rgb_mmt_2'], global_step=epoch)
            if 'joint' in mode or 'flow' in mode:
                flow_saver.save(sess, _CHECKPOINT_PATHS['flow_mmt'], global_step=epoch)
            print('[Epoch %d] train end ' % epoch)

            print('Epoch %d] eval begin' % epoch)
            sess.run(val_init_op)

            total_acc1 = 0
            total_acc5 = 0
            i = 0
            while True:
                try:
                    acc1, acc5 = sess.run([top1_acc, top5_acc])
                    total_acc1 += acc1
                    total_acc5 += acc5
                    print(acc1)
                    i += 1
                except:
                    break
            total_acc1 /= i
            total_acc5 /= i
            acc_summary = tf.Summary(value=[tf.Summary.Value(tag='acc1', simple_value=total_acc1),
                                            tf.Summary.Value(tag='acc5', simple_value=total_acc5)])
            summary_writer.add_summary(acc_summary, epoch)
            print('Top1 accuracy is : {}, top5 accuracy is : {}'.format(total_acc1, total_acc5))
            print('Epoch %d] eval end' % epoch)

if __name__ == '__main__':
    tf.app.run(train)