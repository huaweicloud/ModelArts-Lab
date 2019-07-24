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
# Example to train iceberg model.
# https://www.kaggle.com/c/statoil-iceberg-classifier-challenge

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ.pop('http_proxy', None)
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.keras.layers import Dropout, Flatten, Activation, Concatenate
from moxing.tensorflow.datasets.tfrecord_common import ImageClassificationTFRecordMetadata
from moxing.tensorflow.datasets.tfrecord_common import BaseTFRecordDataset
import moxing.tensorflow as mox
from moxing.tensorflow.utils import tf_util
if tf_util.version_info() > (1, 13, 0):
  from tensorflow.python.layers.layers import Conv2D, MaxPooling2D, Dense
else:
  from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NUM_SAMPLES_TRAIN = 1176
NUM_SAMPLES_EVAL = 295
NUM_SAMPLES_TEST = 8424

tf.flags.DEFINE_integer('batch_size', 16, 'Mini-batch size')
tf.flags.DEFINE_string('data_url', None, 'Dir of dataset')
tf.flags.DEFINE_string('train_url', None, 'Dir of log')
tf.flags.DEFINE_boolean('is_training', True, 'True for train. False for eval and predict.')
flags = tf.flags.FLAGS


def input_fn(run_mode, **kwargs):
  if run_mode == mox.ModeKeys.TRAIN:
    num_samples = NUM_SAMPLES_TRAIN
    num_epochs = None
    shuffle = True
    file_pattern = 'iceberg-train-*.tfrecord'
    num_readers = 16
  else:
    num_epochs = None
    shuffle = False
    num_readers = 1
    if run_mode == mox.ModeKeys.EVAL:
      num_samples = NUM_SAMPLES_EVAL
      file_pattern = 'iceberg-eval-*.tfrecord'
    else:
      num_samples = NUM_SAMPLES_TEST
      file_pattern = 'iceberg-test-*.tfrecord'

  custom_feature_keys = ['band_1', 'band_2', 'angle']
  custom_feature_values = [tf.FixedLenFeature((75 * 75,), tf.float32, default_value=None),
              tf.FixedLenFeature((75 * 75,), tf.float32, default_value=None),
              tf.FixedLenFeature([], tf.float32, default_value=None)]

  if run_mode == mox.ModeKeys.PREDICT:
    custom_feature_keys.append('id')
    custom_feature_values.append(tf.FixedLenFeature([], tf.string, default_value=None))

  else:
    custom_feature_keys.append('label')
    custom_feature_values.append(tf.FixedLenFeature([], tf.int64, default_value=None))

  dataset_meta = ImageClassificationTFRecordMetadata(flags.data_url,
                                                     file_pattern,
                                                     num_samples=num_samples,
                                                     redirect_dir=None)

  class IcebergTFRecordDataset(BaseTFRecordDataset):

    def _build_features(self, band_1, band_2, angle, id_or_label):
      band_1 = tf.reshape(band_1, shape=[75, 75])
      band_2 = tf.reshape(band_2, shape=[75, 75])
      self._add_feature('band_1', band_1)
      self._add_feature('band_2', band_2)
      self._add_feature('angle', angle)
      if 'id' in custom_feature_keys:
        self._add_feature('id', id_or_label)
      else:
        self._add_feature('label', id_or_label)

  dataset = IcebergTFRecordDataset(dataset_meta,
                                   feature_keys=custom_feature_keys,
                                   feature_values=custom_feature_values,
                                   shuffle=shuffle,
                                   num_parallel=num_readers,
                                   num_epochs=num_epochs)

  if run_mode == mox.ModeKeys.PREDICT:
    band_1, band_2, id_or_label, angle = dataset.get(['band_1', 'band_2', 'id', 'angle'])
    # Non-DMA safe string cannot tensor may not be copied to a GPU.
    # So we encode string to a list of integer.
    from moxing.framework.util import compat
    id_or_label = tf.py_func(lambda str: np.array([ord(ch) for ch in compat.as_str(str)]), [id_or_label], tf.int64)
    # We know `id` is a string of 8 alphabets.
    id_or_label = tf.reshape(id_or_label, shape=(8,))
  else:
    band_1, band_2, id_or_label, angle = dataset.get(['band_1', 'band_2', 'label', 'angle'])

  band_3 = band_1 + band_2

  # Rescale the input image to [0, 1]
  def rescale(*args):
    ret_images = []
    for image in args:
      image = tf.cast(image, tf.float32)
      image_min = tf.reduce_min(image)
      image_max = tf.reduce_max(image)
      image = (image - image_min) / (image_max - image_min)
      ret_images.append(image)
    return ret_images

  band_1, band_2, band_3 = rescale(band_1, band_2, band_3)
  image = tf.stack([band_1, band_2, band_3], axis=2)

  # Data augementation
  if run_mode == mox.ModeKeys.TRAIN:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=tf.random_uniform(shape=(), maxval=3, minval=0, dtype=tf.int32))

  return image, id_or_label, angle


def model_v1(images, angles, run_mode):
  is_training = (run_mode == mox.ModeKeys.TRAIN)

  # Conv Layer 1
  x = Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3))(images)
  x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
  x = Dropout(0.2)(x, training=is_training)

  # Conv Layer 2
  x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = Dropout(0.2)(x, training=is_training)

  # Conv Layer 3
  x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = Dropout(0.2)(x, training=is_training)

  # Conv Layer 4
  x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
  x = Dropout(0.2)(x, training=is_training)

  # Flatten the data for upcoming dense layers
  x = Flatten()(x)
  x = Concatenate()([x, angles])

  # Dense Layers
  x = Dense(512)(x)
  x = Activation('relu')(x)
  x = Dropout(0.2)(x, training=is_training)

  # Dense Layer 2
  x = Dense(256)(x)
  x = Activation('relu')(x)
  x = Dropout(0.2)(x, training=is_training)

  # Sigmoid Layer
  logits = Dense(2)(x)

  return logits


def model_fn(inputs, mode, **kwargs):
  # In train or eval, id_or_labels represents labels. In predict, id_or_labels represents id.
  images, id_or_labels, angles = inputs
  # Reshape angles from [batch_size] to [batch_size, 1]
  angles = tf.expand_dims(angles, 1)
  # Apply your version of model
  logits = model_v1(images, angles, mode)

  if mode == mox.ModeKeys.PREDICT:
    logits = tf.nn.softmax(logits)
    # clip logits to get lower loss value.
    logits = tf.clip_by_value(logits, clip_value_min=0.05, clip_value_max=0.95)
    model_spec = mox.ModelSpec(output_info={'id': id_or_labels, 'logits': logits})
  elif mode == mox.ModeKeys.EXPORT:
    predictions = tf.nn.softmax(logits)
    export_spec = mox.ExportSpec(inputs_dict={'images': images, 'angles': angles},
                                 outputs_dict={'predictions': predictions},
                                 version='model')
    model_spec = mox.ModelSpec(export_spec=export_spec)
  else:
    labels_one_hot = slim.one_hot_encoding(id_or_labels, 2)
    loss = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels_one_hot,
      label_smoothing=0.0, weights=1.0)
    model_spec = mox.ModelSpec(loss=loss, log_info={'loss': loss})
  return model_spec


def output_fn(outputs):
  global submission
  submission = pd.DataFrame(columns=['id', 'is_iceberg'])
  for output in outputs:
    for id, logits in zip(output['id'], output['logits']):
      # Decode id from integer list to string.
      id = ''.join([chr(ch) for ch in id])
      # Get the probability of label==1
      is_iceberg = logits[1]
      df = pd.DataFrame([[id, is_iceberg]], columns=['id', 'is_iceberg'])
      submission = submission.append(df)


def main(*args):
  num_gpus = mox.get_flag('num_gpus')
  num_workers = len(mox.get_flag('worker_hosts').split(','))
  steps_per_epoch = int(round(math.ceil(
    float(NUM_SAMPLES_TRAIN) / (flags.batch_size * num_gpus * num_workers))))

  if flags.is_training:
    mox.run(input_fn=input_fn,
            model_fn=model_fn,
            optimizer_fn=mox.get_optimizer_fn(name='adam', learning_rate=0.001),
            run_mode=mox.ModeKeys.TRAIN,
            batch_size=flags.batch_size,
            log_dir=flags.train_url,
            max_number_of_steps=steps_per_epoch * 150,
            log_every_n_steps=20,
            save_summary_steps=50,
            save_model_secs=120,
            export_model=mox.ExportKeys.TF_SERVING)
  else:
    mox.run(input_fn=input_fn,
            model_fn=model_fn,
            run_mode=mox.ModeKeys.EVAL,
            batch_size=5,
            log_every_n_steps=1,
            max_number_of_steps=int(NUM_SAMPLES_EVAL / 5),
            checkpoint_path=flags.train_url)
    mox.run(input_fn=input_fn,
            output_fn=output_fn,
            model_fn=model_fn,
            run_mode=mox.ModeKeys.PREDICT,
            batch_size=24,
            max_number_of_steps=int(NUM_SAMPLES_TEST / 24),
            log_every_n_steps=50,
            output_every_n_steps=int(NUM_SAMPLES_TEST / 24),
            checkpoint_path=flags.train_url)

    # Write results to file. tf.gfile allow writing file to EBS/s3
    submission_file = os.path.join(flags.train_url, 'submission.csv')
    result = submission.to_csv(path_or_buf=None, index=False)
    with tf.gfile.Open(submission_file, 'w') as f:
      f.write(result)


if __name__ == '__main__':
  tf.app.run(main=main)
