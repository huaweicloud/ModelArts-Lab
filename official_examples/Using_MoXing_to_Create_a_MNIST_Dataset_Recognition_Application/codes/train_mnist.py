# Copyright 2018 Deep Learning Service of Huawei Cloud. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Train a user defined model with TensorFlow APIs.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import moxing.tensorflow as mox
import os

tf.flags.DEFINE_string('data_url', None, 'Dir of dataset')
tf.flags.DEFINE_string('train_url', None, 'Train Url')

flags = tf.flags.FLAGS


def check_dataset():
  work_directory = flags.data_url
  filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz',
               't10k-labels-idx1-ubyte.gz']

  for filename in filenames:
    filepath = os.path.join(work_directory, filename)
    if not mox.file.exists(filepath):
      raise ValueError('MNIST dataset file %s not found in %s' % (filepath, work_directory))


def main(*args, **kwargs):
  check_dataset()
  mnist = input_data.read_data_sets(flags.data_url, one_hot=True)


  # define the input dataset, return image and label
  def input_fn(run_mode, **kwargs):
    def gen():
      while True:
        yield mnist.train.next_batch(50)
    ds = tf.data.Dataset.from_generator(
        gen, output_types=(tf.float32, tf.int64),
        output_shapes=(tf.TensorShape([None, 784]), tf.TensorShape([None, 10])))
    return ds.make_one_shot_iterator().get_next()


  # define the model for training or evaling.
  def model_fn(inputs, run_mode, **kwargs):
    x, y_ = inputs
    W = tf.get_variable(name='W', initializer=tf.zeros([784, 10]))
    b = tf.get_variable(name='b', initializer=tf.zeros([10]))
    y = tf.matmul(x, W) + b
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    export_spec = mox.ExportSpec(inputs_dict={'images': x}, outputs_dict={'logits': y}, version='model')
    return mox.ModelSpec(loss=cross_entropy, log_info={'loss': cross_entropy, 'accuracy': accuracy},
                         export_spec=export_spec)


  mox.run(input_fn=input_fn,
          model_fn=model_fn,
          optimizer_fn=mox.get_optimizer_fn('sgd', learning_rate=0.01),
          run_mode=mox.ModeKeys.TRAIN,
          batch_size=50,
          auto_batch=False,
          log_dir=flags.train_url,
          max_number_of_steps=1000,
          log_every_n_steps=10,
          export_model=mox.ExportKeys.TF_SERVING)

if __name__ == '__main__':
  tf.app.run(main=main)
