import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.flags.DEFINE_string('data_url', '/tmp/tensorflow/mnist/input_data', 'Directory for storing input data')
tf.flags.DEFINE_string('train_url', '/tmp/tensorflow/mnist/output_log', 'Directory for output logs')
flags = tf.flags.FLAGS
mnist = input_data.read_data_sets(flags.data_url, one_hot=True)

import moxing.tensorflow as mox


def input_fn(run_mode, **kwargs):
  batch_size = 100
  num_batches = mnist.test.num_examples // batch_size

  def gen():
    for _ in range(num_batches):
      yield mnist.test.next_batch(batch_size)

  ds = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int64),
                                      output_shapes=(tf.TensorShape([None, 784]), tf.TensorShape([None, 10])))
  return ds.make_one_shot_iterator().get_next()


def model_fn(inputs, run_mode, **kwargs):
  x, y_ = inputs
  W = tf.get_variable(name='W', initializer=tf.zeros([784, 10]))
  b = tf.get_variable(name='b', initializer=tf.zeros([10]))
  y = tf.matmul(x, W) + b
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return mox.ModelSpec(log_info={'accuracy': accuracy})


mox.run(input_fn=input_fn,
        model_fn=model_fn,
        run_mode=mox.ModeKeys.EVAL,
        checkpoint_path=flags.train_url,
        max_number_of_steps=sys.maxint)
