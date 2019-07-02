from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.flags.DEFINE_integer('max_steps', 1000, 'number of training iterations.')
tf.flags.DEFINE_string('data_url', '/home/jnn/nfs/mnist', 'dataset directory.')
tf.flags.DEFINE_string('train_url', '/home/jnn/temp/delete', 'saved model directory.')

FLAGS = tf.flags.FLAGS


def main(*args):
  # Train model
  print('Training model...')
  mnist = input_data.read_data_sets(FLAGS.data_url, one_hot=True)
  sess = tf.InteractiveSession()
  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)
  x = tf.identity(tf_example['x'], name='x')
  y_ = tf.placeholder('float', shape=[None, 10])
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  sess.run(tf.global_variables_initializer())
  y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

  tf.summary.scalar('cross_entropy', cross_entropy)

  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
  tf.summary.scalar('accuracy', accuracy)
  merged = tf.summary.merge_all()
  test_writer = tf.summary.FileWriter(FLAGS.train_url, flush_secs=1)

  for step in range(FLAGS.max_steps):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    if step % 10 == 0:
      summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
      test_writer.add_summary(summary, step)
      print('training accuracy is:', acc)
  print('Done training!')

  builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(FLAGS.train_url, 'model'))

  tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
  tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

  prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'images': tensor_info_x},
          outputs={'scores': tensor_info_y},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          'predict_images':
              prediction_signature,
      },
      main_op=tf.tables_initializer(),
      strip_default_attrs=True)

  builder.save()

  print('Done exporting!')


if __name__ == '__main__':
  tf.app.run(main=main)