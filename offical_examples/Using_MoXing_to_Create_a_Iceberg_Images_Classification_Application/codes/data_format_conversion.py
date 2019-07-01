import os
import numpy as np
import pandas as pd
import tensorflow as tf
import moxing.tensorflow as mox
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
slim = tf.contrib.slim
BASE_PATH = 's3://obs-testdata/iceberg/'

def get_image(df):
  images = []
  for i, row in df.iterrows():
    band_1 = np.array(row['band_1']).reshape(75 * 75)
    band_2 = np.array(row['band_2']).reshape(75 * 75)
    image = np.stack([band_1, band_2], axis=0)
    images.append(image)
  return np.array(images)


def read_train_and_eval_images(data_path):
  data_buf = mox.file.read(data_path, binary=True)
  train = pd.read_json(data_buf)
  src_images = get_image(train)
  src_labels = train.is_iceberg.values
  images, labels, angles = [], [], []
  num_na = 0
  for i in range(len(train.inc_angle)):
    image = src_images[i]
    label = src_labels[i]
    angle = train.inc_angle[i]
    # Some training data miss angle but all testing data has angle
    # So we drop the cases without angle in training data.
    if angle == 'na':
      num_na += 1
    else:
      images.append(image)
      labels.append(label)
      angles.append(angle)
  print('Training data, Angle na: %s/%s' % (num_na, len(train.inc_angle)))
  print('Training data, Band_1 mean: %s' % np.mean(np.array(images)[:, 0, :]))
  print('Training data, Band_2 mean: %s' % np.mean(np.array(images)[:, 1, :]))
  print('Training data, Angle mean: %s' % np.mean(angles))
  (images_train, images_eval, labels_train, labels_eval, angles_train,
   angles_eval) = train_test_split(images, labels, angles, shuffle=True,
                                   test_size=0.2)
  return images_train, images_eval, labels_train, labels_eval, angles_train, angles_eval


def convert_and_encode_to_tfrecord(num_samples, images, labels, angles, output_file):
  with mox.file.File(os.path.join(BASE_PATH, 'labels.txt'), 'w') as f:
    f.write('iceberg' + '\n' + 'ship')
  with tf.python_io.TFRecordWriter(output_file) as tfrecord_writer:
    for j in range(num_samples):
      example = tf.train.Example(features=tf.train.Features(feature={
        'band_1': tf.train.Feature(float_list=tf.train.FloatList(value=images[j][0])),
        'band_2': tf.train.Feature(float_list=tf.train.FloatList(value=images[j][1])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[j]])),
        'angle': tf.train.Feature(float_list=tf.train.FloatList(value=[angles[j]])),
      }))
      tfrecord_writer.write(example.SerializeToString())


def read_and_decode_tfrecord(dataset_dir, file_pattern, num_samples):
  keys_to_features = {
    'band_1': tf.FixedLenFeature((75 * 75,), tf.float32, default_value=None),
    'band_2': tf.FixedLenFeature((75 * 75,), tf.float32, default_value=None),
    'label': tf.FixedLenFeature([1], tf.int64, default_value=None),
    'angle': tf.FixedLenFeature([1], tf.float32, default_value=None),
  }
  items_to_handlers = {
    'band_1': slim.tfexample_decoder.Tensor('band_1', shape=[75, 75]),
    'band_2': slim.tfexample_decoder.Tensor('band_2', shape=[75, 75]),
    'label': slim.tfexample_decoder.Tensor('label', shape=[]),
    'angle': slim.tfexample_decoder.Tensor('angle', shape=[])
  }
  dataset = mox.get_tfrecord(dataset_dir=dataset_dir,
                             file_pattern=file_pattern,
                             num_samples=num_samples,
                             keys_to_features=keys_to_features,
                             items_to_handlers=items_to_handlers,
                             shuffle=False,
                             num_epochs=1)
  band_1, band_2, label, angle = dataset.get(['band_1', 'band_2', 'label', 'angle'])
  band_3 = (band_1 + band_2) / 2
  image = tf.stack([band_1, band_2, band_3], axis=2)
  image_max = tf.reduce_max(image)
  image_min = tf.reduce_min(image)
  image = (image - image_min) / (image_max - image_min)
  sv = tf.train.Supervisor()
  with sv.managed_session() as sess:
    plt.figure()
    for i in range(40):
      subp = plt.subplot(5, 8, i + 1)
      plt.subplots_adjust(hspace=0.6)
      subp.imshow(sess.run(image))
      label_eval = sess.run(label)
      angle_eval = int(sess.run(angle))
      subp.set_title('label=%s, angle=%s' % (label_eval, angle_eval))
    plt.show()


def check_tst_ids(dataset_dir, file_pattern, num_samples):
  keys_to_features = {
    'band_1': tf.FixedLenFeature((75 * 75,), tf.float32, default_value=None),
    'band_2': tf.FixedLenFeature((75 * 75,), tf.float32, default_value=None),
    'id': tf.FixedLenFeature([1], tf.string, default_value=None),
    'angle': tf.FixedLenFeature([1], tf.float32, default_value=None),
  }
  items_to_handlers = {
    'band_1': slim.tfexample_decoder.Tensor('band_1', shape=[75, 75]),
    'band_2': slim.tfexample_decoder.Tensor('band_2', shape=[75, 75]),
    'id': slim.tfexample_decoder.Tensor('id', shape=[]),
    'angle': slim.tfexample_decoder.Tensor('angle', shape=[])
  }
  dataset = mox.get_tfrecord(dataset_dir=dataset_dir,
                             file_pattern=file_pattern,
                             num_samples=num_samples,
                             keys_to_features=keys_to_features,
                             items_to_handlers=items_to_handlers,
                             shuffle=False,
                             num_epochs=1)
  band_1, band_2, id, angle = dataset.get(
    ['band_1', 'band_2', 'id', 'angle'])
#   sv = tf.train.Supervisor()
  id_dict = {}
#   with sv.managed_session() as sess:
  with tf.train.MonitoredTrainingSession() as sess:
    for i in range(num_samples):
      id_eval = sess.run(id)
      tf.logging.info('%s/%s' % (i, num_samples))
      if id_eval not in id_dict:
        id_dict[id_eval] = 1
      else:
        raise ValueError('id: %s has alreay been defined.')
  print('%s test cases checked.' % len(id_dict))


def read_tst_images(data_path):
  data_buf = mox.file.read(data_path, binary=True)
  test = pd.read_json(data_buf)
  images = get_image(test)
  ids = test.id.values
  angles = []
  num_na = 0
  for angle in test.inc_angle:
    if angle == 'na':
      num_na += 1
      angles.append(0.0)
    else:
      angles.append(angle)
  print('Testing data, Angle na: %s/%s' % (num_na, len(angles)))
  print('Testing data, Band_1 mean: %s' % np.mean(np.array(images)[:, 0, :]))
  print('Testing data, Band_2 mean: %s' % np.mean(np.array(images)[:, 1, :]))
  print('Testing data, Angle mean: %s' % np.mean(angles))
  return ids, images, angles


def convert_and_encode_tst_to_tfrecord(num_samples, ids, images, angles, output_file):
  with tf.python_io.TFRecordWriter(output_file) as tfrecord_writer:
    for j in range(num_samples):
      example = tf.train.Example(features=tf.train.Features(feature={
        'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[unicode.encode(ids[j])])),
        'band_1': tf.train.Feature(float_list=tf.train.FloatList(value=images[j][0])),
        'band_2': tf.train.Feature(float_list=tf.train.FloatList(value=images[j][1])),
        'angle': tf.train.Feature(float_list=tf.train.FloatList(value=[angles[j]])),
      }))
      tfrecord_writer.write(example.SerializeToString())


def main():
  tf.logging.info('Reading train-eval data.')
  x_train, x_eval, y_train, y_eval, z_train, z_eval = read_train_and_eval_images(
      data_path=os.path.join(BASE_PATH, 'train.json'))
  num_train = len(x_train)
  num_eval = len(x_eval)
  tf.logging.info('Converting %d images to iceberg-train-%d.tfrecord' % (num_train, num_train))
  convert_and_encode_to_tfrecord(num_train, x_train, y_train, z_train,
                                 os.path.join(BASE_PATH, 'iceberg-train-%d.tfrecord' % num_train))
  tf.logging.info('Converting %d images to iceberg-eval-%d.tfrecord' % (num_eval, num_eval))
  convert_and_encode_to_tfrecord(num_eval, x_eval, y_eval, z_train,
                                 os.path.join(BASE_PATH, 'iceberg-eval-%d.tfrecord' % num_eval))
  tf.logging.info('Reading test data.')
  id_test, x_test, z_test = read_tst_images(data_path=os.path.join(BASE_PATH, 'test.json'))
  num_test = len(x_test)
  tf.logging.info('Converting %d images to iceberg-test-%d.tfrecord' % (num_test, num_test))
  convert_and_encode_tst_to_tfrecord(num_test, id_test, x_test, z_test,
                                     os.path.join(BASE_PATH, 'iceberg-test-%d.tfrecord' % num_test))

  tf.logging.info('Testing data read and decode.')
  read_and_decode_tfrecord(dataset_dir=BASE_PATH,
                           file_pattern='iceberg-train-*.tfrecord',
                           num_samples=num_train + num_eval)
#   tf.reset_default_graph()
#   tf.logging.info('Checking data ids.')
#   check_tst_ids(dataset_dir=BASE_PATH, file_pattern='iceberg-test-*.tfrecord',
#                 num_samples=num_test)


if __name__ == '__main__':
  main()
