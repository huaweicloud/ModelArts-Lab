## 1.3. Getting Started

这是一个使用MoXing实现的手写数字识别的训练代码。数据集可以在此处下载到：http://yann.lecun.com/exdb/mnist/

```python
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

```

在Nvidia-Tesla-K80单卡上训练时间约为：12秒


- import moxing.tensorflow as mox
  ​        导入 `moxing.tensorflow` 包并重命名为 `mox`

- input_fn
  ​        用户在运行脚本中必须定义 `input_fn` 方法。`input_fn` 的返回值是一个输入数据的list，其中每个元素必须是TensorFlow定义的Tensor类型，不可以是python、numpy、panda等类型。

  - 样例

      ```python
      def input_fn(mode, **kwargs): 
          if mode == mox.ModeKeys.TRAIN: 
              inputs = read_training_data() 
          else: 
              inputs = read_evaluation_data() 
          return inputs
      ```

- model_fn
    ​    用户在运行脚本中必须定义 `model_fn` 方法， `model_fn` 方法必须返回一个`mox.ModelSpec` 的实例。
        在训练模式下，`mox.ModelSpec`中需要指定待下降的`loss`，以及需要在控制台打印的指标`log_info`(仅支持rank=0的`tf.Tensor`)

    - 样例

        ```python
        def model_fn(inputs, mode, **kwargs):
          image, label = inputs
          is_training = (mode == mox.ModeKeys.TRAIN)
          loss, accuracy = model(image, label, is_training=is_training)
          return mox.ModelSpec(loss=loss,
                               log_info={'loss': loss, 'accuracy': accuracy})
        ```

- optimizer_fn
    ​    在训练模式下，必须定义 `optimizer_fn` 方法，否则不需要。`optimizer_fn`返回一个`Optimizer`的实例

- mox.run
        定义网络的训练/验证运行过程
        `run_mode`用于指定运行模式，`mox.ModeKeys.TRAIN`为训练
        `log_dir`用于指定模型输出的checkpoint、summary等文件的位置
        `max_number_of_steps`用于指定运行的最大步数，这里可以根据数据集自动适配，所以指定一个大数即可。

### 1.3.1. 将上述脚本保存为[train_tf_mnist.py](scripts/train_tf_mnist.py)，启动单机单卡训练：

```shell
python train_tf_mnist.py \
--data_url=/tmp/tensorflow/mnist/input_data \
--train_url=/tmp/tensorflow/mnist/output_log
```

### 1.3.2. 启动单机4卡训练：

```shell
python train_tf_mnist.py \
--data_url=/tmp/tensorflow/mnist/input_data \
--train_url=/tmp/tensorflow/mnist/output_log \
--num_gpus=4
```

`num_gpus`参数可以被MoXing自动解析，表示训练时使用GPU的数量。

### 1.3.3. 启动分布式2节点，每个节点使用2个gpu：

PS-0:
```shell
python train_tf_mnist.py \
--data_url=/tmp/tensorflow/mnist/input_data \
--train_url=/tmp/tensorflow/mnist/output_log \
--num_gpus=2 \
--ps_hosts=127.0.0.1:2222,127.0.0.1:2223 \
--worker_hosts=127.0.0.1:2224,127.0.0.1:2225 \
--job_name=ps \
--task_index=0
```

PS-1:
```shell
python train_tf_mnist.py \
--data_url=/tmp/tensorflow/mnist/input_data \
--train_url=/tmp/tensorflow/mnist/output_log \
--num_gpus=2 \
--ps_hosts=127.0.0.1:2222,127.0.0.1:2223 \
--worker_hosts=127.0.0.1:2224,127.0.0.1:2225 \
--job_name=ps \
--task_index=1
```

Worker-0:
```shell
python train_tf_mnist.py \
--data_url=/tmp/tensorflow/mnist/input_data \
--train_url=/tmp/tensorflow/mnist/output_log \
--num_gpus=2 \
--ps_hosts=127.0.0.1:2222,127.0.0.1:2223 \
--worker_hosts=127.0.0.1:2224,127.0.0.1:2225 \
--job_name=worker \
--task_index=0
```

Worker-1:
```shell
python train_tf_mnist.py \
--data_url=/tmp/tensorflow/mnist/input_data \
--train_url=/tmp/tensorflow/mnist/output_log \
--num_gpus=2 \
--ps_hosts=127.0.0.1:2222,127.0.0.1:2223 \
--worker_hosts=127.0.0.1:2224,127.0.0.1:2225 \
--job_name=worker \
--task_index=1
```

`job_name`, `task_index`, `ps_hosts`, `worker_hosts`可以被MoXing自动解析，用于定义分布式训练集群

- job_name
    ​    节点的作业名称，可以是 `ps` 或者 `worker`。

- task_index
    ​    作业中的任务索引号。

- ps_hosts
    ​    `ps` 节点主机列表，列表中内容使用逗号分隔，例如：`'localhost:port1,localhost:port2'` 。

- worker_hosts
    ​    `worker` 节点主机列表，列表中内容使用逗号分隔，例如：`'localhost:port3,localhost:port4` 。


### 1.3.4. 对模型进行验证(evaluation)

```python
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

```

在Nvidia-Tesla-K80单卡上训练时间约为：4秒， 精度约为：96.1%

- input_fn
  ​      验证只需要跑一个epoch
        mnist.train替换成了mnist.test

- model_fn
    ​    验证模式下不需要loss，计算accuracy并添加到log_info中

- optimizer_fn
    ​    验证模式下不需要optimizer

- mox.run
        验证时使用验证模式，即`run_mode=mox.ModeKeys.EVAL`
        验证时需要指定载入的checkpoint，训练时checkpoint输出到了`train_url`，所以这里指定`checkpoint_path=flags.train_url`

### 1.3.5. 将上述脚本保存为[eval_tf_mnist.py](scripts/eval_tf_mnist.py)，启动单机单卡验证：

```shell
python eval_tf_mnist.py \
--data_url=/tmp/tensorflow/mnist/input_data \
--train_url=/tmp/tensorflow/mnist/output_log
```