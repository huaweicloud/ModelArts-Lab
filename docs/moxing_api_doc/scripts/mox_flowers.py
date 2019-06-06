# encoding: utf-8

import sys
import tensorflow as tf
from tensorflow.contrib import slim

import moxing.tensorflow as mox
from moxing.tensorflow.optimizer import learning_rate_scheduler

tf.flags.DEFINE_string('data_url', None, 'Necessary. dataset dir')
tf.flags.DEFINE_string('train_url', None, 'Optional. train_dir')
tf.flags.DEFINE_integer('batch_size', 64, 'Mini-batch-size')

flags = tf.flags.FLAGS


def main(_):
  # 获取当前使用的GPU数量和节点数量
  num_gpus = mox.get_flag('num_gpus')
  num_workers = len(mox.get_flag('worker_hosts').split(','))
  data_meta = mox.ImageClassificationRawMetadata(base_dir=flags.data_url)

  def input_fn(mode):
    # 创建一个数据增强方法，该方法基于resnet50论文实现
    augmentation_fn = mox.get_data_augmentation_fn(name='resnet_v1_50',
                                                   run_mode=mode,
                                                   output_height=224,
                                                   output_width=224)

    # 创建`数据集读取类`，并将数据增强方法传入，最多读取20个epoch
    dataset = mox.ImageClassificationRawDataset(data_meta,
                                                batch_size=flags.batch_size,
                                                num_epochs=20,
                                                augmentation_fn=augmentation_fn)
    image, label = dataset.get(['image', 'label'])
    return image, label

  def model_fn(inputs, mode):
    images, labels = inputs

    # 获取一个resnet50的模型，输入images，输入logits和end_points，这里不关心end_points，仅取logits
    logits, _ = mox.get_model_fn(name='resnet_v1_50',
                                 run_mode=mode,
                                 num_classes=data_meta.num_classes,
                                 weight_decay=0.00004)(images)

    # 计算交叉熵损失值
    labels_one_hot = slim.one_hot_encoding(labels, data_meta.num_classes)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_one_hot)

    # 获取正则项损失值，并加到loss上，这里必须要用mox.get_collection代替tf.get_collection
    regularization_losses = mox.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.add_n(regularization_losses)
    loss = loss + regularization_loss

    # 计算分类正确率
    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))

    # 返回MoXing-TensorFlow用于定义模型的类ModelSpec
    return mox.ModelSpec(loss=loss, log_info={'loss': loss, 'accuracy': accuracy})

  def optimizer_fn():
    # 使用分段式学习率，0-10个epoch为0.01，10-20个epoch为0.001
    lr = learning_rate_scheduler.piecewise_lr('10:0.01,20:0.001',
                                              num_samples=data_meta.total_num_samples,
                                              global_batch_size=flags.batch_size * num_gpus * num_workers)
    return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

  mox.run(input_fn=input_fn,
          model_fn=model_fn,
          optimizer_fn=optimizer_fn,
          run_mode=mox.ModeKeys.TRAIN,
          max_number_of_steps=sys.maxint,
          log_dir=flags.train_url)


if __name__ == '__main__':
  tf.app.run(main=main)
