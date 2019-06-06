import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.flags.DEFINE_string('data_url', '/tmp/tensorflow/mnist/input_data', 'Directory for storing input data')
tf.flags.DEFINE_string('train_url', '/tmp/tensorflow/mnist/output_log', 'Directory for output logs')
flags = tf.flags.FLAGS
mnist = input_data.read_data_sets(flags.data_url, one_hot=True)

import moxing.tensorflow as mox


def input_fn(run_mode, **kwargs):
  num_epochs = 5
  batch_size = 100
  num_batches = num_epochs * mnist.train.num_examples // batch_size

  def gen():
    for _ in range(num_batches):
      yield mnist.test.next_batch(batch_size)

  ds = tf.data.Dataset.from_generator(gen,
                                      output_types=(tf.float32, tf.int64),
                                      output_shapes=(tf.TensorShape([None, 784]), tf.TensorShape([None, 10])))
  x, y_ = ds.make_one_shot_iterator().get_next()
  return x, y_


def model_fn(inputs, run_mode, **kwargs):
  x, y_ = inputs
  W = tf.get_variable(name='W', initializer=tf.zeros([784, 10]))
  b = tf.get_variable(name='b', initializer=tf.zeros([10]))
  y = tf.matmul(x, W) + b
  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  return mox.ModelSpec(loss=cross_entropy, log_info={'loss': cross_entropy})


def optimizer_fn():
  return tf.train.GradientDescentOptimizer(0.5)


mox.run(input_fn=input_fn,
        model_fn=model_fn,
        optimizer_fn=optimizer_fn,
        run_mode=mox.ModeKeys.TRAIN,
        log_dir=flags.train_url,
        max_number_of_steps=sys.maxint)
