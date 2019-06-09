import sys
import tensorflow as tf
from tensorflow.contrib import slim

import moxing.tensorflow as mox
from moxing.tensorflow.optimizer import learning_rate_scheduler

tf.flags.DEFINE_string('data_url',
                       None, 'Necessary. dataset dir')
tf.flags.DEFINE_string('model_name',
                       'resnet_v1_50', 'Necessary. model_name')
tf.flags.DEFINE_string('train_url',
                       None, 'Optional. train_dir')
tf.flags.DEFINE_string('checkpoint_url',
                       None, 'Optional. checkpoint path')
tf.flags.DEFINE_integer('batch_size',
                        64, 'Necessary. batch size')

flags = tf.flags.FLAGS


def main(*args, **kwargs):
  import time
  st = time.time()
  num_gpus = mox.get_flag('num_gpus')
  num_workers = len(mox.get_flag('worker_hosts').split(','))

  exclude_list = ['global_step']
  model_meta = mox.get_model_meta(flags.model_name)
  exclude_list.append(model_meta.default_logits_pattern)
  checkpoint_exclude_patterns = ','.join(exclude_list)
  mox.set_flag('checkpoint_exclude_patterns', checkpoint_exclude_patterns)

  data_meta = mox.ImageClassificationRawMetadata(base_dir=flags.data_url)

  mox.set_flag('loss_scale', 1024.0)

  def input_fn(mode, **kwargs):
    data_augmentation_fn = mox.get_data_augmentation_fn(name=flags.model_name,
                                                        run_mode=mode)

    dataset = mox.ImageClassificationRawDataset(data_meta,
                                                batch_size=flags.batch_size,
                                                num_epochs=20,
                                                augmentation_fn=data_augmentation_fn,
                                                reader_class=mox.AsyncRawGenerator)

    images, labels = dataset.get(['image', 'label'])

    return images, labels

  def model_fn(inputs, mode, **kwargs):
    images, labels = inputs

    mox_model_fn = mox.get_model_fn(
      name=flags.model_name,
      run_mode=mode,
      num_classes=data_meta.num_classes,
      weight_decay=0.00004,
      data_format='NCHW',
      batch_norm_fused=True)

    images = tf.cast(images, tf.float16)
    with mox.var_scope(force_dtype=tf.float32):
      logits, _ = mox_model_fn(images)

    labels_one_hot = slim.one_hot_encoding(labels, data_meta.num_classes)
    loss = tf.losses.softmax_cross_entropy(labels_one_hot, logits=logits)

    regularization_losses = mox.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.add_n(regularization_losses)
    loss = loss + regularization_loss

    logits_fp32 = tf.cast(logits, tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_fp32, labels, 1), tf.float32))

    export_spec = mox.ExportSpec(inputs_dict={'images': images},
                                 outputs_dict={'logits': logits_fp32})

    return mox.ModelSpec(loss=loss,
                         log_info={'loss': loss, 'accuracy': accuracy},
                         export_spec=export_spec)

  def optimizer_fn():
    lr = learning_rate_scheduler.piecewise_lr('10:0.01,20:0.001',
                                              num_samples=data_meta.total_num_samples,
                                              global_batch_size=flags.batch_size * num_gpus * num_workers)
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    return opt

  mox.run(input_fn=input_fn,
          model_fn=model_fn,
          optimizer_fn=optimizer_fn,
          run_mode=mox.ModeKeys.TRAIN,
          log_dir=flags.train_url,
          checkpoint_path=flags.checkpoint_url,
          max_number_of_steps=sys.maxint,
          export_model=mox.ExportKeys.TF_SERVING)

  print(time.time() - st)


if __name__ == '__main__':
  tf.app.run(main=main)
