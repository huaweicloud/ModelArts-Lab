# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import time
import numpy as np

PB_INPUT = 'input'
PB_OUTPUTS = ['logits:0']

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

_RESIZE_MIN = 256
INPUT_SIZE = 224
INPUT_DIMENSIONS = (INPUT_SIZE, INPUT_SIZE)
NUM_CLASSES = 1001


def _decode_crop_and_flip(image_buffer, bbox, num_channels):
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Reassemble the bounding box in the format the crop op requires.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    # Use the fused decode and crop op here, which is faster than each in series.
    cropped = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=num_channels)

    # Flip to add a little more random distortion in.
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped


def _central_crop(image, crop_height, crop_width):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def _mean_image_subtraction(image, means, num_channels):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    # We have a 1-D tensor of means; convert to 3-D.
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)
    # image = tf.cast(image, dtype=tf.float32)

    return image - means


def _smallest_size_at_least(height, width, resize_min):
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width


def _aspect_preserving_resize(image, resize_min):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least(height, width, resize_min)
    return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
    return tf.image.resize_images(
        image, [height, width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)


def preprocess_image(image_buffer,
                     output_height,
                     output_width,
                     num_channels=3
                     ):
    # For validation, we want to decode, resize, then just crop the middle.
    image = tf.image.decode_jpeg(image_buffer, channels=num_channels, dct_method='INTEGER_FAST')
    image = _aspect_preserving_resize(image, _RESIZE_MIN)
    image = _central_crop(image, output_height, output_width)
    image.set_shape([output_height, output_width, num_channels])

    return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)


class LoggerHook(tf.train.SessionRunHook):
    """Logs runtime of each iteration"""

    def __init__(self, batch_size, num_records, display_every):
        self.iter_times = []
        self.display_every = display_every
        self.num_steps = (num_records + batch_size - 1) / batch_size
        self.batch_size = batch_size

    def begin(self):
        self.start_time = time.time()

    def after_run(self, run_context, run_values):
        current_time = time.time()
        duration = current_time - self.start_time
        self.start_time = current_time
        self.iter_times.append(duration)
        current_step = len(self.iter_times)
        if current_step % self.display_every == 0:
            print("    step %d/%d, iter_time(ms)=%.4f, images/sec=%d" % (
                current_step, self.num_steps, duration * 1000,
                self.batch_size / self.iter_times[-1]))


def run(frozen_graph, model, data_files, batch_size,
        num_iterations, num_warmup_iterations, display_every=100, run_calibration=False):
    # Define model function for tf.estimator.Estimator
    def model_fn(features, labels, mode):
        logits_out = tf.import_graph_def(frozen_graph,
                                         input_map={PB_INPUT: features},
                                         return_elements=PB_OUTPUTS,
                                         name='')
        logits_out = tf.reshape(logits_out, [-1, 1001])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_out)

        labels = tf.reshape(labels, [-1])
        top5accuracy = tf.nn.in_top_k(predictions=logits_out, targets=labels, k=5, name='acc_op')
        top5accuracy = tf.cast(top5accuracy, tf.int32)
        top5accuracy = tf.metrics.mean(top5accuracy)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                eval_metric_ops={'accuracy': top5accuracy})

    # preprocess function for input data
    preprocess_fn = get_preprocess_fn(model)

    def get_tfrecords_count(files):
        num_records = 0
        for fn in files:
            for record in tf.python_io.tf_record_iterator(fn):
                num_records += 1
        return num_records

    # Define the dataset input function for tf.estimator.Estimator
    def eval_input_fn():
        dataset = tf.data.TFRecordDataset(data_files)
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(map_func=preprocess_fn, batch_size=batch_size, num_parallel_calls=2))
        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        dataset = dataset.repeat(count=1)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    # Evaluate model
    logger = LoggerHook(
        display_every=display_every,
        batch_size=batch_size,
        num_records=get_tfrecords_count(data_files))
    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads = 8
    tf_config.inter_op_parallelism_threads = 8
    tf_config.gpu_options.allow_growth = True
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(session_config=tf_config),
        model_dir='model_dir')
    results = estimator.evaluate(eval_input_fn, steps=num_iterations, hooks=[logger])

    # Gather additional results
    iter_times = np.array(logger.iter_times[num_warmup_iterations:])
    results['latency_mean'] = np.mean(iter_times) * 1000
    return results


def deserialize_image_record(record):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], tf.string, ''),
        'image/class/label': tf.FixedLenFeature([1], tf.int64, -1),
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label = tf.cast(obj['image/class/label'], tf.int32)
        return imgdata, label


def get_preprocess_fn(model, mode='classification'):
    def process(record):
        imgdata, label = deserialize_image_record(record)
        image = preprocess_image(imgdata, INPUT_SIZE, INPUT_SIZE)
        return image, label

    return process


def get_frozen_graph(
        model,
        model_dir=None,
        pb_name=None,
        use_trt=False,
        use_dynamic_op=False,
        precision='fp32',
        batch_size=8,
        minimum_segment_size=2,
        calib_files=None,
        num_calib_inputs=None,
        cache=False,
        max_workspace_size=(1 << 32)):
    num_nodes = {}
    times = {}
    graph_sizes = {}
    frozen_graph = tf.GraphDef()

    # Load from pb file if frozen graph was already created and cached
    if pb_name:
        prebuilt_graph_path = os.path.join(model_dir, pb_name)
    else:
        prebuilt_graph_path = os.path.join(model_dir, 'r50_93.40_trt.pb')

    if cache:
        if os.path.isfile(prebuilt_graph_path):
            print('Loading cached frozen graph from \'%s\'' % prebuilt_graph_path)
            start_time = time.time()
            with tf.gfile.GFile(prebuilt_graph_path, "rb") as f:
                frozen_graph = tf.GraphDef()
                frozen_graph.ParseFromString(f.read())
            times['loading_frozen_graph'] = time.time() - start_time
            num_nodes['loaded_frozen_graph'] = len(frozen_graph.node)
            num_nodes['trt_only'] = len([1 for n in frozen_graph.node if str(n.op) == 'TRTEngineOp'])
            graph_sizes['loaded_frozen_graph'] = len(frozen_graph.SerializeToString())
            return frozen_graph, num_nodes, times, graph_sizes

    num_nodes['native_tf'] = len(frozen_graph.node)
    graph_sizes['native_tf'] = len(frozen_graph.SerializeToString())

    # Convert to TensorRT graph
    if use_trt:
        print("Using TensorRT")
        start_time = time.time()
        frozen_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=PB_OUTPUTS,
            max_batch_size=batch_size,
            max_workspace_size_bytes=max_workspace_size,
            precision_mode=precision,
            minimum_segment_size=minimum_segment_size,
            is_dynamic_op=use_dynamic_op
        )
        times['trt_conversion'] = time.time() - start_time
        num_nodes['tftrt_total'] = len(frozen_graph.node)
        num_nodes['trt_only'] = len([1 for n in frozen_graph.node if str(n.op) == 'TRTEngineOp'])
        graph_sizes['trt'] = len(frozen_graph.SerializeToString())

        if precision == 'int8':
            calib_graph = frozen_graph
            graph_sizes['calib'] = len(calib_graph.SerializeToString())
            # INT8 calibration step
            print('Calibrating INT8...')
            start_time = time.time()
            run(calib_graph, model, calib_files, batch_size,
                num_calib_inputs // batch_size, 0, False, run_calibration=True)
            times['trt_calibration'] = time.time() - start_time

            start_time = time.time()
            frozen_graph = trt.calib_graph_to_infer_graph(calib_graph)

            times['trt_int8_conversion'] = time.time() - start_time
            # This is already set but overwriting it here to ensure the right size
            graph_sizes['trt'] = len(frozen_graph.SerializeToString())

            del calib_graph
            print('INT8 graph created.')

    # Cache graph to avoid long conversions each time
    if cache:
        saved_pb = os.path.join(model_dir, 'r50.pb')
        if not os.path.exists(os.path.dirname(saved_pb)):
            try:
                os.makedirs(os.path.dirname(saved_pb))
            except Exception as e:
                raise e
        start_time = time.time()
        with tf.gfile.GFile(saved_pb, "wb") as f:
            f.write(frozen_graph.SerializeToString())
        times['saving_frozen_graph'] = time.time() - start_time

    return frozen_graph, num_nodes, times, graph_sizes


def benchmark(args):
    if args.precision != 'fp32' and not args.use_trt:
        raise ValueError('TensorRT must be enabled for fp16 or int8 modes (--use_trt).')
    if args.num_iterations is not None and args.num_iterations <= args.num_warmup_iterations:
        raise ValueError('--num_iterations must be larger than --num_warmup_iterations '
                         '({} <= {})'.format(args.num_iterations, args.num_warmup_iterations))
    if args.num_calib_inputs < args.batch_size:
        raise ValueError('--num_calib_inputs must not be smaller than --batch_size'
                         '({} <= {})'.format(args.num_calib_inputs, args.batch_size))

    def get_files(data_dir, filename_pattern):
        if data_dir == None:
            return []
        files = tf.gfile.Glob(os.path.join(data_dir, filename_pattern))
        if files == []:
            raise ValueError('Can not find any files in {} with pattern "{}"'.format(
                data_dir, filename_pattern))
        return files

    validation_files = get_files(args.data_dir, 'validation*')
    if args.calib_data_dir:
        calib_files = get_files(args.calib_data_dir, 'train*')
    else:
        calib_files = None

    # Retreive graph using NETS table in graph.py
    frozen_graph, num_nodes, times, graph_sizes = get_frozen_graph(
        model=args.model,
        model_dir=args.model_dir,
        pb_name=args.pb_name,
        use_trt=args.use_trt,
        use_dynamic_op=args.use_trt_dynamic_op,
        precision=args.precision,
        batch_size=args.batch_size,
        minimum_segment_size=args.minimum_segment_size,
        calib_files=calib_files,
        num_calib_inputs=args.num_calib_inputs,
        cache=args.cache,
        max_workspace_size=args.max_workspace_size)

    def print_dict(input_dict, str='', scale=None):
        for k, v in sorted(input_dict.items()):
            headline = '{}({}): '.format(str, k) if str else '{}: '.format(k)
            v = v * scale if scale else v
            print('{}{}'.format(headline, '%.1f' % v if type(v) == float else v))

    print_dict(vars(args))
    print_dict(num_nodes, str='num_nodes')
    print_dict(graph_sizes, str='graph_size(MB)', scale=1. / (1 << 20))
    print_dict(times, str='time(s)')

    # Evaluate model
    print('running inference...')
    results = run(
        frozen_graph,
        model=args.model,
        data_files=validation_files,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        num_warmup_iterations=args.num_warmup_iterations,
        display_every=args.display_every)

    # Display results
    print('results of {}:'.format(args.model))
    print('    accuracy: %.2f' % (results['accuracy'] * 100))
    print('    latency_mean(ms): %.2f' % results['latency_mean'])
    return results['accuracy'] * 100, results['latency_mean']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model', type=str, default='inception_v4',
                        choices=['mobilenet_v1', 'mobilenet_v2', 'nasnet_mobile', 'nasnet_large',
                                 'resnet_v1_50', 'resnet_v2_50', 'resnet_v2_152', 'vgg_16', 'vgg_19',
                                 'inception_v3', 'inception_v4', 'resnet_v1_34'],
                        help='Which model to use.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing validation set TFRecord files.')
    parser.add_argument('--calib_data_dir', type=str,
                        help='Directory containing TFRecord files for calibrating int8.')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing model checkpoint. If not provided, a ' \
                             'checkpoint may be downloaded automatically and stored in ' \
                             '"{--default_models_dir}/{--model}" for future use.')
    parser.add_argument('--pb_name', type=str, default=None, help="pb file name")
    parser.add_argument('--use_trt', action='store_true',
                        help='If set, the graph will be converted to a TensorRT graph.')
    parser.add_argument('--use_trt_dynamic_op', action='store_true',
                        help='If set, TRT conversion will be done using dynamic op instead of statically.')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'], default='fp32',
                        help='Precision mode to use. FP16 and INT8 only work in conjunction with --use_trt')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of images per batch.')
    parser.add_argument('--minimum_segment_size', type=int, default=2,
                        help='Minimum number of TF ops in a TRT engine.')
    parser.add_argument('--num_iterations', type=int, default=None,
                        help='How many iterations(batches) to evaluate. If not supplied, the whole set will be evaluated.')
    parser.add_argument('--display_every', type=int, default=1000,
                        help='Number of iterations executed between two consecutive display of metrics')
    parser.add_argument('--num_warmup_iterations', type=int, default=100,
                        help='Number of initial iterations skipped from timing')
    parser.add_argument('--num_calib_inputs', type=int, default=500,
                        help='Number of inputs (e.g. images) used for calibration '
                             '(last batch is skipped in case it is not full)')
    parser.add_argument('--max_workspace_size', type=int, default=(1 << 32),
                        help='workspace size in bytes')
    parser.add_argument('--cache', action='store_true',
                        help='If set, graphs will be saved to disk after conversion. If a converted graph is present on disk, it will be loaded instead of building the graph again.')
    args = parser.parse_args()

    benchmark(args)
