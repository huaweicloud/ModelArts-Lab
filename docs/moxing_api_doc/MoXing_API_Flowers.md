### 1.4. 训练 resnet50-flowers

数据集下载地址：http://download.tensorflow.org/example_images/flower_photos.tgz

将数据集解压到目录`/tmp/tensorflow/`

#### 1.4.1. 定义主函数和超参

引入MoXing-TensorFlow

```python
import moxing.tensorflow as mox
```

通过`mox.get_flag`获取命令行参数`num_gpus`和`worker_hosts`，从而获取当前使用的GPU数量和节点数量

```python
num_gpus = mox.get_flag('num_gpus')
num_workers = len(mox.get_flag('worker_hosts').split(','))
```

flowers数据集的格式如下：

```shell
/tmp/tensorflow/flower_photos
    |-- daisy
        |-- xxx0.jpg
        ...
    |-- dandelion
        |-- xxx1.jpg
        ...
    |-- roses
        |-- xxx2.jpg
        ...
    |-- sunflowers
        |-- xxx3.jpg
        ...
    |-- tulips
        |-- xxx4.jpg
        ...
```

每一个子目录代表一个分类，每个分类下有若干张图片，对于这种类型的数据集，可以使用`mox.ImageClassificationRawMetadata`和`mox.ImageClassificationRawDataset`来读取。MoXing-TensorFlow预置了若干种解析数据集的类，一般会使用`数据集元信息类`+`数据集读取类`的模式来读取。`数据集元信息类`不会创建任何TensorFlow的数据流图，建议在main方法中直接实例化，那么代码的其他地方都能获取数据集的元信息（如样本数量，分类数量）。`数据集读取类`必须在`input_fn`中实例化，该类的实例化会在TensorFlow数据流图中创建节点。


创建一个`数据集元信息类`，`base_dir`即指定`flower_photos`所在目录

```python
data_meta = mox.ImageClassificationRawMetadata(base_dir=flags.data_url)
```

在`input_fn`中创建一个数据增强方法(基于resnet50)和一个`数据集读取类`

```python
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
```

定义model_fn

```python
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
```

定义optimizer_fn

```python

from moxing.tensorflow.optimizer import learning_rate_scheduler

def optimizer_fn():
  # 使用分段式学习率，0-10个epoch为0.01，10-20个epoch为0.001
  lr = learning_rate_scheduler.piecewise_lr('10:0.01,20:0.001',
                                            num_samples=data_meta.total_num_samples,
                                            global_batch_size=flags.batch_size * num_gpus * num_workers)
  return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
```

完整代码请参考：[mox_flowers.py](scripts/mox_flowers.py)

执行训练：

```shell
python mox_flowers.py \
--data_url=/tmp/tensorflow/flower_photos \
--train_url=/tmp/flowers \
--num_gpus=4
```

使用 4 * Nvidia-Tesla-K80 运行时间大约为：698秒，在训练集上的训练精度约为：50%