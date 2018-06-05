from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

LENGTH = 64
HEIGHT = 256
WEIGHT = 256
CHANNELS = 3

CROP_HEIGHT = 224
CROP_WEIGHT = 224

def build_dataset(filenames, is_training, batch_size=32, buffer_size=2000):
    dataset = tf.data.TFRecordDataset(filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.map(_train_parse_function, num_parallel_calls=24)
    else:
        dataset = dataset.map(_val_parse_function, num_parallel_calls=24)
    dataset = dataset.batch(batch_size)

    return dataset


def _train_parse_function(example_proto):
    context_features = {
        "class": tf.FixedLenFeature([], dtype=tf.int64),
        "confidence": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "flow_xs": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "flow_ys": tf.FixedLenSequenceFeature([], dtype=tf.string)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example_proto,
        context_features=context_features,
        sequence_features=sequence_features
    )

    label = context_parsed["class"]
    confidence = context_parsed["confidence"]
    frames = sequence_parsed["frames"]
    flow_xs = sequence_parsed["flow_xs"]
    flow_ys = sequence_parsed["flow_ys"]

    frames = tf.decode_raw(frames, np.uint8)
    flow_xs = tf.decode_raw(flow_xs, np.uint8)
    flow_ys = tf.decode_raw(flow_ys, np.uint8)

    frames = tf.reshape(frames, (-1, HEIGHT, WEIGHT, CHANNELS))
    flow_xs = tf.reshape(flow_xs, (-1, HEIGHT, WEIGHT, 1))
    flow_ys = tf.reshape(flow_ys, (-1, HEIGHT, WEIGHT, 1))
    flows = tf.concat([flow_xs, flow_ys], axis=3)

    frames = tf.image.convert_image_dtype(frames, dtype=tf.float32)
    flows = tf.image.convert_image_dtype(flows, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.int32)

    norm_frames = None
    norm_flows = None
    for i in range(LENGTH):
        if norm_frames == None:
            frame = frames[i, :, :, :]
            flow = flows[i, :, :, :]
            frame = tf.image.random_flip_left_right(frame)
            flow = tf.image.random_flip_left_right(flow)
            frame = tf.random_crop(frame, [CROP_HEIGHT, CROP_WEIGHT, CHANNELS])
            flow = tf.random_crop(flow, [CROP_HEIGHT, CROP_WEIGHT, 2])
            norm_frames = tf.expand_dims(frame, 0)
            norm_flows = tf.expand_dims(flow, 0)
        else:
            frame = frames[i, :, :, :]
            flow = flows[i, :, :, :]
            frame = tf.image.random_flip_left_right(frame)
            flow = tf.image.random_flip_left_right(flow)
            frame = tf.random_crop(frame, [CROP_HEIGHT, CROP_WEIGHT, CHANNELS])
            flow = tf.random_crop(flow, [CROP_HEIGHT, CROP_WEIGHT, 2])
            frame = tf.expand_dims(frame, 0)
            flow = tf.expand_dims(flow, 0)
            norm_frames = tf.concat([norm_frames, frame], 0)
            norm_flows = tf.concat([norm_flows, flow], 0)

    norm_frames = tf.subtract(norm_frames, 0.5)
    norm_frames = tf.multiply(norm_frames, 2)
    norm_flows = tf.subtract(norm_flows, 0.5)
    norm_flows = tf.multiply(norm_flows, 2)

    return norm_frames, norm_flows, label, confidence

def _val_parse_function(example_proto):
    context_features = {
        "class": tf.FixedLenFeature([], dtype=tf.int64),
        "confidence": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "flow_xs": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "flow_ys": tf.FixedLenSequenceFeature([], dtype=tf.string)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example_proto,
        context_features=context_features,
        sequence_features=sequence_features
    )

    label = context_parsed["class"]
    confidence = context_parsed["confidence"]
    frames = sequence_parsed["frames"]
    flow_xs = sequence_parsed["flow_xs"]
    flow_ys = sequence_parsed["flow_ys"]

    frames = tf.decode_raw(frames, np.uint8)
    flow_xs = tf.decode_raw(flow_xs, np.uint8)
    flow_ys = tf.decode_raw(flow_ys, np.uint8)

    frames = tf.reshape(frames, (-1, HEIGHT, WEIGHT, CHANNELS))
    flow_xs = tf.reshape(flow_xs, (-1, HEIGHT, WEIGHT, 1))
    flow_ys = tf.reshape(flow_ys, (-1, HEIGHT, WEIGHT, 1))
    flows = tf.concat([flow_xs, flow_ys], axis=3)

    frames = tf.image.convert_image_dtype(frames, dtype=tf.float32)
    flows = tf.image.convert_image_dtype(flows, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.int32)

    norm_frames = None
    norm_flows = None
    for i in range(LENGTH):
        if norm_frames == None:
            frame = frames[i, :, :, :]
            flow = flows[i, :, :, :]
            frame = tf.random_crop(frame, [CROP_HEIGHT, CROP_WEIGHT, CHANNELS])
            flow = tf.random_crop(flow, [CROP_HEIGHT, CROP_WEIGHT, 2])
            norm_frames = tf.expand_dims(frame, 0)
            norm_flows = tf.expand_dims(flow, 0)
        else:
            frame = frames[i, :, :, :]
            flow = flows[i, :, :, :]
            frame = tf.random_crop(frame, [CROP_HEIGHT, CROP_WEIGHT, CHANNELS])
            flow = tf.random_crop(flow, [CROP_HEIGHT, CROP_WEIGHT, 2])
            frame = tf.expand_dims(frame, 0)
            flow = tf.expand_dims(flow, 0)
            norm_frames = tf.concat([norm_frames, frame], 0)
            norm_flows = tf.concat([norm_flows, flow], 0)

    norm_frames = tf.subtract(norm_frames, 0.5)
    norm_frames = tf.multiply(norm_frames, 2)
    norm_flows = tf.subtract(norm_flows, 0.5)
    norm_flows = tf.multiply(norm_flows, 2)

    return norm_frames, norm_flows, label, confidence
