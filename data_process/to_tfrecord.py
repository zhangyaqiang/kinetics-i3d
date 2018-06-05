from __future__ import absolute_import, division, print_function

import os, random
import tensorflow as tf
import numpy as np
import glob
from skimage import io, color, transform, img_as_ubyte
import skvideo.io
import random
import threading
from datetime import datetime
import sys

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_list',
                           '/home/zyq/moment_in_time/train_list.txt',
                           'Traing video directory')
tf.app.flags.DEFINE_string('val_list',
                           '/home/zyq/moment_in_time/validation_list.txt',
                           'Validation video directory')

tf.app.flags.DEFINE_string('output_dir',
                           '/home/zyq/moment_in_time/dataset/',
                           'Output directory')

tf.app.flags.DEFINE_integer('train_shards', 1024,
                           'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('val_shards', 128,
                           'Number of shards in training TFRecord files.')

tf.flags.DEFINE_integer('num_threads', 16,
                       'Numbers of threads to preprocess the videos.')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def get_label(label_file):
    dict = {}
    with open(label_file) as f:
        lines = f.readlines()
        for line in lines:
            class_name = line.split(',')[0]
            class_num = line.split(',')[1]
            dict[class_name] = class_num
    return dict

def get_video_frames(video_path):
    videogen = skvideo.io.vreader(video_path)
    frames = np.asarray([frame for frame in videogen])
    return frames

def _to_sequence_example(video, dict, num_frame=64):
    '''Build a SequenceExample proto for an video-label pair.
    Args:
        video_path: Path of video data_process.
        label_path: Path of label data_process.
        vocab: A Vocabulary object.
    Returns:
        A SequenceExample proto.
    '''

    video_path = video.split(' ')[0]
    video_frames = int(video.split(' ')[1])
    label = int(video.split(' ')[2])
    confidence = int(video.split(' ')[3][0])
    frame_list = glob.glob(os.path.join(video_path, '*img*'))
    flow_x_list = glob.glob(os.path.join(video_path, '*flow_x*'))
    flow_y_list = glob.glob(os.path.join(video_path, '*flow_y*'))
    if len(frame_list) < num_frame or len(flow_x_list) < num_frame or len(flow_y_list) < num_frame:
        raise ('{} frames not enough!'.format(video_path))
    assert len(frame_list) == len(flow_x_list)
    assert len(flow_x_list) == len(flow_y_list)

    start = random.randrange(1, 1 + len(frame_list) - num_frame)
    frames = []
    flow_xs = []
    flow_ys = []
    for i in range(num_frame):
        frame = video_path + '/img_' + str(i + start).zfill(5) + '.jpg'
        flow_x = video_path + '/flow_x_' + str(i + start).zfill(5) + '.jpg'
        flow_y = video_path + '/flow_y_' + str(i + start).zfill(5) + '.jpg'
        frames.append(io.imread(frame))
        flow_xs.append(io.imread(flow_x))
        flow_ys.append(io.imread(flow_y))

    frames_byte = [frame.tostring() for frame in frames]
    flow_xs_byte = [frame.tostring() for frame in flow_xs]
    flow_ys_byte = [frame.tostring() for frame in flow_ys]

    example = tf.train.SequenceExample(
        context=tf.train.Features(feature={
            "class": _int64_feature(label),
            "confidence": _int64_feature(confidence)
        }),
        feature_lists=tf.train.FeatureLists(feature_list={
            "frames": _bytes_feature_list(frames_byte),
            "flow_xs": _bytes_feature_list(flow_xs_byte),
            "flow_ys": _bytes_feature_list(flow_ys_byte)
        })
    )
    return example


def process_batch_files(thread_index, ranges, name, num_shards, video_list, dict):
    '''Processes and saves a subset of video as TFRecord files in one thread.
    Args:
        thread_index: 线程序号
        ranges: 将数据集分成了几个部分，A list of pairs
        name: Unique identifier specifying the dataset
        video_dir：视频数据所在的文件夹
        label_dir：文本数据所在的文件夹
        vocab：A Vocabulary object
        num_shards： 数据集最终分成几个TFRecord
    '''
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_video_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        video_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in video_in_shard:
            video = video_list[i]
            sequence_example = _to_sequence_example(video, dict)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_video_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d video-label pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d video-label pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def process_dataset(name, dataset_list, num_shards, dict):
    '''Process a complete data_process set and save it as a TFRecord.
    Args:
        name: 数据集的名称.
        video_dir: 数据集视频所在文件夹.
        label_dir: 标签所在的文件夹.
        vocab: A Vocabulary object.
    '''

    f = open(dataset_list, 'r')
    video_list = f.readlines()
    f.close()
    random.seed(1117)
    random.shuffle(video_list)

    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(video_list), num_threads + 1).astype(int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    coord = tf.train.Coordinator()

    print('Launching %d threads for spacing: %s' % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, num_shards, video_list, dict)
        t = threading.Thread(target=process_batch_files, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)
    print('%s: Finished processing all %d video-caption pairs in data_process set "%s".' %
          (datetime.now(), len(video_list), name))


def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        '''Please make the FALGS.num_threads commensurate with FLAGS.train_shards''')

    assert _is_valid_num_shards(FLAGS.val_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")

    # dict = get_label(FLAGS.label_file)
    dict = None
    process_dataset('train', FLAGS.train_list, FLAGS.train_shards, dict)
    process_dataset('val', FLAGS.val_list, FLAGS.val_shards, dict)


if __name__ == '__main__':
    tf.app.run()