import os
import sys
import tensorflow as tf
import moxing.tensorflow as mox

from tensorflow.examples.tutorials.mnist import input_data

# `t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz  train-images-idx3-ubyte.gz
# train-labels-idx1-ubyte.gz` under data_url
tf.flags.DEFINE_string('data_url', '/home/zgz/nfs/mnist/', 'Directory for storing input data')
tf.flags.DEFINE_string('train_url', '/home/zgz/tmp/tensorflow/mnist/output_log', 'Directory for output logs')
flags = tf.flags.FLAGS
mnist = input_data.read_data_sets(flags.data_url, one_hot=True)

batch_size = 16


def input_fn(run_mode, **kwargs):
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


# 定义一个函数，用于初始化所有的权值 W
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


def model_fn(inputs, run_mode, **kwargs):
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


def optimizer_fn():
    return tf.train.GradientDescentOptimizer(0.5)


mox.run(input_fn=input_fn,
        model_fn=model_fn,
        optimizer_fn=optimizer_fn,
        run_mode=mox.ModeKeys.TRAIN,
        log_dir=flags.train_url,
        max_number_of_steps=50, batch_size=12)


def student_model_fn(inputs, run_mode):
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
    W_fc1 = weight_variable('w3', [7 * 7 * 64, 5])
    b_fc1 = bias_variable('b3', [5])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape成向量
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 第一个全连接层
    W_fc2 = weight_variable('w4', [5, 10])
    b_fc2 = bias_variable('b4', [10])
    y = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)  # softmax层

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    from efficient_ai.config import CompressorSpec

    compressor_spec = CompressorSpec(logits=y)
    return mox.ModelSpec(loss=cross_entropy, compressor_spec=compressor_spec, log_info={'loss': cross_entropy})


def distill():
    from efficient_ai.config import DistillCompressorConfig
    log_dir = flags.train_url
    max_number_of_steps = 50
    num_train_samples = mnist.train.num_examples
    new_log_dir = os.path.join(log_dir, 'distill')

    config = DistillCompressorConfig(
        num_batches_per_epoch=num_train_samples / batch_size,
        log_every_n_steps=10,
        log_dir=new_log_dir,
        max_number_of_steps=max_number_of_steps,
        optimizer_fn=optimizer_fn,
    )

    from efficient_ai.compressor import Compressor
    from efficient_ai.models.moxing_model import MoxingModel
    teacher_model = MoxingModel(model_fn, log_dir)
    student_model = MoxingModel(student_model_fn)
    c = Compressor(student_model, input_fn)

    export_dir = os.path.join(log_dir, 'tf_models')
    c = c.distilling(teacher_model, config).export(export_dir)


if __name__ == '__main__':
    distill()
