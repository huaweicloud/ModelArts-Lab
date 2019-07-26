# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#定义最大的训练迭代次数
tf.flags.DEFINE_integer('max_steps', 1000, 'number of training iterations.') 
#定义mnist数据集目录
tf.flags.DEFINE_string('data_url', '/home/jnn/nfs/mnist', 'dataset directory.')
#定义训练输出目录也就是保存模型的目录 
tf.flags.DEFINE_string('train_url', '/home/jnn/temp/delete', 'saved model directory.')

FLAGS = tf.flags.FLAGS


def main(*args):
  # Train model
  print('Training model...')
  mnist = input_data.read_data_sets(FLAGS.data_url, one_hot=True)
  sess = tf.InteractiveSession()
  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}#手写体的图片大小28*28，总共784个像素，每一个像素都是特征值，采用浮点型表示
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)

  #构建训练模型
  x = tf.identity(tf_example['x'], name='x')#定义输入特征值，也就是列长度为784的张量x
  y_ = tf.placeholder('float', shape=[None, 10])#定义标签值为浮点型,长度为10的one-hot向量，n行10列，n取决于训练的样本数量


  w = tf.Variable(tf.zeros([784, 10])) #定义权重参数，因为后面要用到矩阵运算，用x张量乘以w张量，x的列数784要与w的行数相同，所以w是784行、10列的张量，10代表有10个分类
  b = tf.Variable(tf.zeros([10]))#定义值参数  
  
  #计算预测值：输入值X与权重w相乘，再加上偏置值b,得到预测值
  prediction=tf.matmul(x,w)+b
  #采用softmax函数激活输出预测值y
  y = tf.nn.softmax(prediction)

  #将原有的代价函数改为交叉熵代价函数
  #cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y))   

  tf.summary.scalar('cross_entropy', cross_entropy)
  #定义学习率
  learning_rate = 0.01
  #使用梯度下降法找到最小代价损失
  #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
  
  #初始化全局变量
  sess.run(tf.global_variables_initializer()) 

  #将计算结果存放在一个bool列表中
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) #argmax：返回一维张量中最大的值所在的位置，如果位置相等代表预测正确
  #计算精确率
  #tf.cast是把bool型数组转化为float型,True转化为1.0, False转化为0.0.reduce_mean时求float型数组的平均值,即正确的个数与所有个数之比.这个数越大越精准
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float')) #例如correct_prediction:[true,true,true,true，flase]=>[1,1,1,1,0]=>4/5=>80%
  tf.summary.scalar('accuracy', accuracy)
  merged = tf.summary.merge_all()
  test_writer = tf.summary.FileWriter(FLAGS.train_url, flush_secs=1)

  #开启训练模式，先训练个1000次
  for step in range(FLAGS.max_steps):
    batch = mnist.train.next_batch(50)#随机读取50个训练样本
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})#把x和y_喂进去，走起
    if step % 10 == 0:
      summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})#使用测试集数据评估准确率
      test_writer.add_summary(summary, step)
      #print('training accuracy is:', acc)
      print("迭代次数："+str(step)+",准确率(accuracy)："+str(acc))

  print('Done training!')

  #保存模型
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
