"""
A example to use quantize from a moxing model
Copyright 2019 The Huawei Corps. All Rights Reserved.
Modify By h00423091 2019/6/19 11:23
Created By h00423091 2019/6/19 11:23
"""

import os
import tensorflow as tf
import moxing.tensorflow as mox
from tensorflow.examples.tutorials.mnist import input_data

from efficient_ai.enumerate import TfliteVerKeys, PrecisionKeys

tf.flags.DEFINE_string('data_url', '/opt/howe/workplace/FastAI/data/mnist', 'Directory for storing input data')
tf.flags.DEFINE_string('train_url', '/opt/howe/workplace/FastAI/output', 'Directory for output logs')
flags = tf.flags.FLAGS
mnist = input_data.read_data_sets(flags.data_url, one_hot=True)

batch_size = 1


def input_fn():
    num_epochs = 5
    num_batches = num_epochs * mnist.train.num_examples // batch_size

    def gen():
        for _ in range(num_batches):
            yield mnist.test.next_batch(batch_size)

    ds = tf.data.Dataset.from_generator(gen,
                                        output_types=(tf.float32, tf.int64),
                                        output_shapes=(tf.TensorShape([None, 784]), tf.TensorShape([None, 10])))
    x, y_ = ds.make_one_shot_iterator().get_next()
    return x, y_


def weight_variable(name, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, initializer=initial)


# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)


# 定义一个函数，用于构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def model_fn(inputs):
    x, y_ = inputs
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # 转换输入数据shape,以便于用于网络中
    W_conv1 = weight_variable('w1', [5, 5, 1, 32])
    b_conv1 = bias_variable('b1', [32])
    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, W_conv1), b_conv1))  # 第一个卷积层
    h_pool1 = max_pool(h_conv1)  # 第一个池化层
    W_conv2 = weight_variable('w2', [5, 5, 32, 64])
    b_conv2 = bias_variable('b2', [64])
    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_pool1, W_conv2), b_conv2))  # 第二个卷积层
    h_pool2 = max_pool(h_conv2)  # 第二个池化层
    W_fc1 = weight_variable('w3', [7 * 7 * 64, 1024])
    b_fc1 = bias_variable('b3', [1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape成向量
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 第一个全连接层
    W_fc2 = weight_variable('w4', [1024, 10])
    b_fc2 = bias_variable('b4', [10])
    y = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)  # softmax层
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    from efficient_ai.config import CompressorSpec

    compressor_spec = CompressorSpec(logits=y)
    return mox.ModelSpec(loss=cross_entropy, compressor_spec=compressor_spec, log_info={'loss': cross_entropy})


def get_config():
    config = {
        "algorithm": "DEFAULT",
        "inference_engine": "TFLITE",
        "inference_engine_version": TfliteVerKeys.TFLITE_13,
        "precision": PrecisionKeys.INT8,
        "batch_size": 64,
        "calibrate_batch": 64
    }
    return config


def quantize():
    from efficient_ai.compressor import Compressor
    from efficient_ai.models.moxing_model import MoxingModel
    log_dir = flags.train_url
    model = MoxingModel(model_fn, log_dir)
    c = Compressor(model, input_fn)

    export_dir = os.path.join(log_dir, 'tf_quantize_models')
    c = c.quantizing(config=get_config()).export(export_dir)


if __name__ == '__main__':
    quantize()
