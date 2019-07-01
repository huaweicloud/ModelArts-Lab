## 接口文档

### 1. 命令行参数和通用

#### mox.get_flag

    moxing.tensorflow.executor.get_flag(name)

  获取MoXing内部定义的某个运行参数的值。

- 参数说明
  
  - name: 参数名称。
  
- 返回值

  返回对应参数的实际取值。

- 示例

```python
worker_hosts = mox.get_flag(flag_name='worker_hosts')
```

#### mox.set_flag
  
    moxing.tensorflow.executor.set_flag(name, value)
  
  设置MoXing内部定义的某个运行参数的值，需要在input_fn, model_fn, optimizer_fn函数之外及mox.run之前设置

- 参数说明
  
  - name：参数名称，由moxing/tensorflow/utils/hyper_param_flags.py定义。
  - value：参数值。
  
- 示例

```python
mox.set_flag('checkpoint_exclude_patterns', 'logits')
```

#### mox.ModeKeys 

    class moxing.tensorflow.executor.ModeKeys

  moxing运行时，`run_mode`和`inter_mode`需要指定运行模式。
```python
>>> import moxing.tensorflow as mox
>>> print(mox.ModeKeys.TRAIN)
TRAIN
>>> print(mox.ModeKeys.EVAL)
EVAL
>>> print(mox.ModeKeys.PREDICT)
PREDICT
>>> print(mox.ModeKeys.EXPORT)
EXPORT
```
- 内置的ModeKeys

> EVAL = 'EVAL'

> EXPORT = 'EXPORT'

> PREDICT = 'PREDICT'

> TRAIN = 'TRAIN'

### 2. 输入数据

#### mox.ImageClassificationRawMetadata

```python
class moxing.tensorflow.datasets.ImageClassificationRawMetadata(base_dir, redirect_dir=None, labels_filename='labels.txt')
```

当图片分类数据集中数据为Raw Data时，可以使用 `ImageClassificationRawMetadata` 构建数据集的元信息。

- 参数

  - base_dir：数据集目录。
  - redirect_dir：数据集的重定向目录。
  - labels_filename：类标文件的名称，默认为 `labels.txt`。

- 样例1

在 `/export/dataset` 目录下的应该有完整数据集，结构如下：

```
base_dir
  |- label_0
    |- 0_0.jpg
    |- 0_1.jpg
    ...
    |- 0_x.jpg
  |- label_1
    |- 1_0.jpg
    |- 1_1.jpg
    ...
    |- 1_y.jpg
  ...
  |- label_m
    |- m_0.jpg
    |- m_1.jpg
    ...
    |- m_z.jpg
  labels.txt (Optional)
```

该数据集结构表示数据集中有m个标签，每个标签的名称是 `label_0`， `label_1`， ...， `label_m`。

```python
>>> import moxing.tensorflow as mox
>>> metadata = ImageClassificationRawMetadata(base_dir='/export/dataset')
>>> print(metadata.total_num_samples)
3301
>>> print(metadata.num_classes)
5
>>> print(metadata.labels_list)
['label_0', 'label_1', ..., 'label_m']
```

- 样例2

如果提供*labels.txt*，则*labels.txt*的格式如下：

```
0: label_i 
1: label_j 
2: label_k 
...
```

在*labels.txt*文件中的所有类标名称应该和 `/export/dataset` 目录下的子目录名称相同，而且*labels.txt*文件中类标的数量应该等于 `/export/dataset` 目录下的子目录的数量。

```python
>>> import moxing.tensorflow as mox
>>>  metadata = ImageClassificationRawMetadata(base_dir='/export/dataset',
                                               labels_filename='labels.txt')
>>> print(metadata.labels_list)
['label_i', 'label_j', 'label_k', ...]
```

#### mox.ImageClassificationRawDataset

```python
class moxing.tensorflow.datasets.ImageClassificationRawDataset(metadata, batch_size=None, num_readers=8, num_epochs=None, shuffle=True, shuffle_buffer_size=16, capacity=256, cache_dir=None, preprocess_fn=None, preprocess_threads=12, image_size=None, remove_cache_on_finish=True, private_num_threads=0, reader_class=None, reader_kwargs=None, drop_remainder=False, augmentation_fn=None, label_one_hot=False)
```

当图片分类数据集中数据为Raw Data时，可以使用 `ImageClassificationRawDataset`构建数据集。

- 参数

  - metadata：ImageClassificationRawMetadata实例 。
  - batch_size：批数据的大小。当batch_size不是None时，返回批量特征。
  - num_readers：读取数据的线程或进程数目。
  - num_epochs：允许读取数据集的epoch数目。如果是None时，则一直读取。
  - shuffle：是否对训练数据进行洗牌。
  - shuffle_buffer_size：`tf.int64 `格式标量，表示洗牌池大小。
  - capacity：数据缓冲区的最大容量，用于从文件系统中预取数据集。
  - preprocess_fn：数据预处理函数，对解码后的数据进行预处理功能。
  - preprocess_threads：对图片进行预处理的进程数目 。
  - image_size：图片大小。如果是None时，不对图片做resize操作 。
  - remove_cache_on_finish：退出时是否删除缓存文件。
  - private_num_threads： 如果大于0，则将私有线程池用于数据集。
  - reader_class：自定义读取数据类，用于启动多进程读取，加快读取速度，默认为`None`。如果从OBS文件系统读取文件，则可以设置为 `AsyncRawGenerator` 。
  - reader_kwargs：自定义读取数据类的关键字参数。
  - drop_remainder：`tf.bool` 类型的Tensor，表示是否应该丢弃最后一批数据，以防止它的大小比预期小；默认不删除较小的批数据。
  - augmentation_fn：数据增强方法。它将不带 `batch_size` 的单张图片作为输入，并且返回增强后的图片。
  - label_one_hot：是否将类标编码为 `one hot` 格式，默认为False。

- 样例1

```python
metadata = ImageClassificationRawMetadata(base_dir='/export/dataset')
dataset = ImageClassificationRawDataset(metadata)
```

在 `/export/dataset` 目录下的应该有完整数据集，结构如下：

```
base_dir
  |- label_0
    |- 0_0.jpg
    |- 0_1.jpg
    ...
    |- 0_x.jpg
  |- label_1
    |- 1_0.jpg
    |- 1_1.jpg
    ...
    |- 1_y.jpg
  ...
  |- label_m
    |- m_0.jpg
    |- m_1.jpg
    ...
    |- m_z.jpg
  labels.txt (Optional)
```

该结构表示数据集中有m个标签，每个标签的名称是 `label_0`， `label_1`， ...， `label_m`。

- 样例2

用户可以获取 `tf.Tensor` 格式的特征，基本写法如下：

```python
image, label, label_desc = dataset.get(['image', 'label', 'label_desc'])
```

用户如果只想获取 `image` 和 `label` ，则基本写法如下：

```python
image, label = dataset.get(['image', 'label'])
```

- 样例3

如果给出 `batch_size` 参数，则参数 `image_size` 和 `preprocess_fn` 必须给出一个。`preprocess_fn` 的输入参数是4个Tensor：`image_name`，`image_bug`，`label`，`label_desc` 。如果给出 `batch_size`，则在参数的第一维上有 `batch_size`，否则没有。如果未给出 `preprocess_fn` 方法，则使用默认方法，它将对图像进行解码，并使用 `NEAREST_NEIGHBOR` 将图片调整为 `image_size` 大小。使用 `batch_size` 和 `proprocess_fn` 的基本写法如下：

```python
metadata = ImageClassificationRawMetadata(base_dir='/export/dataset')

def _decode_and_resize(buf)
  img = tf.image.decode_image(buf, channels=3)
  img.set_shape([None, None, 3])
  img = tf.image.resize_images(img, size=[224, 224])
  img.set_shape([224, 224, 3])
  return img

def preprocess_fn(image_name, image_buf, label, label_desc):
  image = tf.map_fn(_decode_and_resize, image_buf, dtype=tf.float32, back_prop=False)
  return image_name, image, label, label_desc

dataset = ImageClassificationRawDataset(metadata, batch_size=32, preprocess_fn=preprocess_fn)
```

#### mox.ImageClassificationTFRecordMetadata

```python
class moxing.tensorflow.datasets.ImageClassificationTFRecordMetadata(base_dir, file_pattern, num_samples=None, redirect_dir=None, num_classes=0, label_map_file='labels.txt')
```

当图片分类数据集中数据为 `tfrecord` 格式时，可以使用 `ImageClassificationTFRecordMetadata` 构建数据集的元信息。

- 参数
  - base_dir：数据集目录。
  - file_pattern：表示 `tfrecord` 文件的文件通配符的字符串。
  - num_samples： `tfrecord` 文件中样本总数。如果没有给出，样本总数将自动生成，但需要消耗一些时间。
  - redirect_dir：数据集的重定向目录。

- 样例

在 `/tmp/tfrecord` 目录下的应该有完整数据集，结构如下：

```
base_dir
    data_train_0.tfrecord
    data_train_1.tfrecord
    ...
    data_train_m.tfrecord
    data_eval_0.tfrecord
    ...
    data_eval_n.tfrecord
    labels.txt (Optional)
```

下面示例将加载 `/tmp/tfrecord` 文件中符合文件通配符 `*train*.tfrecord` 的所有 `tfrecord` 文件，基本写法如下：

```python
tfrecord_meta = ImageClassificationTFRecordMetadata(
  base_dir='/tmp/tfrecord'
  file_patterns='*train*.tfrecord')
print(tfrecord_meta.num_samples)
```

如果*labels.txt*文件存在：

```py
print(tfrecord_meta.labels_to_names)
```

### mox.ImageClassificationTFRecordDataset

```python
class moxing.tensorflow.datasets.ImageClassificationTFRecordDataset(metadata, batch_size=None, shuffle=True, shuffle_buffer_size=6400, num_parallel=8, num_epochs=None, cache_dir=None, preprocess_fn=None, image_size=None, remove_cache_on_finish=True, private_num_threads=0, custom_feature_keys=None, custom_feature_values=None, drop_remainder=False, augmentation_fn=None, label_one_hot=False)
```

当图片分类数据集中数据为 `tfrecord` 格式时，可以使用 `ImageClassificationTFRecordDataset` 构建数据集。

- 参数

  - metadata：BaseTFRecordMetadata实例 。
  - batch_size：批数据的大小。当batch_size不是None时，返回批量特征。
  - shuffle：是否对打乱数据顺序。
  - shuffle_buffer_size：`tf.int64` 格式标量，表示新数据集将从中采样的数据集中的元素数。
  - num_parallel：读取数据的线程或进程数目。
  - num_epochs：允许读取数据集的epoch数目。如果是None时，则一直读取。
  - cache_dir：数据集的缓存目录，用于将数据集缓存到本地目录。使用 `ENV` 将数据集缓存到 `${DLS_LOCAL_CACHE_PATH}` 指定的目录下。
  - preprocess_fn：数据预处理函数，对解码后的数据进行预处理功能。
  - image_size：图片大小。如果是None时，不对图片做resize操作 。
  - remove_cache_on_finish：退出时是否删除缓存文件。
  - private_num_threads：当参数 `private_num_threads` 大于0时，使用私有线程池读取数据集。
  - custom_feature_keys：
  - custom_feature_values：
  - drop_remainder：`tf.bool` 类型的Tensor，表示是否应该丢弃最后一批数据，以防止它的大小比预期小；默认不删除较小的批数据。
  - augmentation_fn：数据增强方法。它将不带 `batch_size` 的单张图片作为输入，并且返回增强后的图片。
  - label_one_hot：是否将类标编码为 `one hot` 格式，默认为False。

- 样例1

在 `/tmp/tfrecord` 目录下的应该有完整数据集，结构如下：

```
base_dir
    data_train_0.tfrecord
    data_train_1.tfrecord
    ...
    data_train_m.tfrecord
    data_eval_0.tfrecord
    ...
    data_eval_n.tfrecord
    labels.txt (Optional)
```

下面示例将加载 `/tmp/tfrecord` 文件中符合文件通配符 `*train*.tfrecord` 的所有 `tfrecord` 文件，基本写法如下：

```python
tfrecord_meta = ImageClassificationTFRecordMetadata(
  base_dir='/tmp/tfrecord'
  file_patterns='*train*.tfrecord')
tfrecord_dataset = ImageClassificationTFRecordDataset(meta=tfrecord_meta)
image, label = dataset.get(['image', 'label'])
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  for i in range(10):
    image_eval, label_eval = sess.run([image, label])
  coord.request_stop()
  coord.join(threads)
```

#### mox.MultilabelClassificationRawMetadata

```python
class moxing.tensorflow.datasets.MultilabelClassificationRawMetadata(base_dir, images_sub_dir='images', labels_sub_dir='labels', labels_extension='txt', redirect_dir=None, labels_filename='labels.txt')
```

当图片分类数据集中数据为多标签的Raw Data时，可以使用 `MultilabelClassificationRawMetadata` 构建数据集的元信息。

- 参数

  - base_dir：数据集目录。
  - images_sub_dir：存放图片的子目录名称。
  - labels_sub_dir：存放类标文件的子目录名称。
  - labels_extension：类标文件的扩展名。
  - redirect_dir：数据集的重定向目录。
  - labels_filename：类标文件的名称，默认为 `labels.txt`。

- 样例1

在 `/export/dataset` 目录下的应该有完整数据集，结构如下：

```
base_dir
  |- images
    |- 0.jpg
    |- 1.jpg
    ...
    |- n.jpg
  |- labels
    |- 0.jpg.txt
    |- 1.jpg.txt
    ...
    |- n.jpg.txt
labels.txt
```

如果提供*i.jpg.txt*，则*i.jpg.txt*的格式如下：

```
label_i label_j ... label_m
```

数据集结构表示在数据集中有n张图片和多类标文件 `*.jpg.txt`。用户可以获取数据集的元信息，基本写法如下：

```python
>>> import moxing.tensorflow as mox
>>> metadata = MultilabelClassificationRawMetadata(base_dir='/export/dataset')
>>> print(metadata.total_num_samples)
3301
>>> print(metadata.num_classes)
5
```

- 样例2

如果提供*labels.txt*，则*labels.txt*的格式如下：

```
0: label_i 1: label_j 2: label_k ...
```

*labels.txt*文件中类标的数量应该等于所有 `*.jpg.txt` 文件中类标总数。

```python
>>> import moxing.tensorflow as mox
>>> metadata = MultilabelClassificationRawMetadata(base_dir='/export/dataset', labels_filename='labels.txt')
>>> print(metadata.labels_to_names)
{0: 'label_i', 1: 'label_j', 2: 'label_k', ...]
>>> print(metadata.names_to_labels)
{'label_i': 0, 'label_j': 1, 'label_k': 2, ...]
```

#### mox.MultilabelClassificationRawDataset

```python
class moxing.tensorflow.datasets.MultilabelClassificationRawDataset(metadata, batch_size=None, num_readers=8, num_epochs=None, shuffle=True, shuffle_buffer_size=16, capacity=256, cache_dir=None, preprocess_fn=None, preprocess_threads=12, image_size=None, remove_cache_on_finish=True, private_num_threads=0, reader_class=None, reader_kwargs=None, drop_remainder=False, label_sep='n', skip_label_not_found=False, augmentation_fn=None)
```

当图片分类数据集中数据为多标签的Raw Data时，可以使用 `MultilabelClassificationRawDataset` 构建数据集。

- 参数

  - metadata：MultilabelClassificationRawMetadata实例 。
  - batch_size：批数据的大小。当batch_size不是None时，返回批量特征。
  - num_readers：读取数据的线程或进程数目。
  - num_epochs：允许读取数据集的epoch数目。如果是None时，则一直读取。
  - shuffle：是否对打乱数据顺序。
  - shuffle_buffer_size：`tf.int64 `格式标量，表示新数据集将从中采样的数据集中的元素数。
  - capacity：数据缓冲区的最大容量，用于从文件系统中预取数据集。
  - cache_dir：数据集的缓存目录，用于将数据集缓存到本地目录。使用 `ENV` 将数据集缓存到 `${DLS_LOCAL_CACHE_PATH}` 指定的目录下。
  - preprocess_fn：数据预处理函数，对解码后的数据进行预处理功能。
  - preprocess_threads：对图片进行预处理的进程数目 。
  - image_size：图片大小。如果是None时，不对图片做resize操作 。
  - remove_cache_on_finish：退出时是否删除缓存文件。
  - private_num_threads： 如果大于0，则将私有线程池用于数据集。
  - reader_class：自定义读取数据类，默认为 `ParallelReader`。如果从OBS文件系统读取文件，则可以设置为 `AsyncRawGenerator` 。
  - reader_kwargs：自定义读取数据类的关键字参数。
  - drop_remainder：`tf.bool` 类型的Tensor，表示是否应该丢弃最后一批数据，以防止它的大小比预期小；默认不删除较小的批数据。
  - label_sep：类标文件中类标之间使用的分割符，默认为 `\n` 。
  - skip_label_not_found：如果 `skip_label_not_found` 置为True，将跳过没有对应类标文件图片。
  - augmentation_fn：数据增强方法。它将不带 `batch_size` 的单张图片作为输入，并且返回增强后的图片。

- 样例1

```python
metadata = MultilabelClassificationRawMetadata(base_dir='/export/dataset')
dataset = MultilabelClassificationRawDataset(metadata)
```

在 `/export/dataset` 目录下的应该有完整数据集，结构如下：

```
base_dir
  |- images
    |- 0.jpg
    |- 1.jpg
    ...
    |- n.jpg
  |- labels
    |- 0.jpg.txt
    |- 1.jpg.txt
    ...
    |- n.jpg.txt
labels.txt
```

如果提供*i.jpg.txt*，则*i.jpg.txt*的格式如下：

```
label_i label_j ... label_m
```

数据集结构表示在数据集中有n张图片和多类标文件 `*.jpg.txt`。

如果提供*labels.txt*，则*labels.txt*的格式如下：

```
0: label_i 1: label_j 2: label_k ...
```

- 样例2

用户可以获取 `tf.Tensor` 格式的特征，基本写法如下：

```python
image, label, label_desc = dataset.get(['image', 'label'])
```

用户如果只想获取 `image` 和 `label` ，则基本写法如下：

```python
image, label = dataset.get(['image'])
```

#### mox.get_data_augmentation_fn

```
moxing.tensorflow.preprocessing.get_data_augmentation_fn(name, run_mode, output_height=None, output_width=None, **kwargs)
```

获取针对图片分类模型的数据增强方法，该数据增强方法接收单张图片，图片的形状为3维[height, width, channel]。

- 参数

  - name：图片分类模型的名称。
  - run_mode：模型的运行模式，可以设置为 `ModeKeys.TRAIN`， `ModeKeys.EVAL` 或者 `ModeKeys.PREDICT`。
  - output_height：增强后返回图片的高度。
  - output_width：增强后返回图片的宽度。

- 样例1

```
data_augmentation_fn = mox.get_data_augmentation_fn(
          name='resnet_v1_50', run_mode=mox.ModeKeys.TRAIN,
          output_height=224, output_width=224)
image = data_augmentation_fn(image)
```

#### mox.PreprocessingKeys
```
ALEXNET_V2: vgg_preprocessing
CIFARNET: cifarnet_preprocessing
OVERFEAT: vgg_preprocessing
INCEPTION: inception_preprocessing
INCEPTION_V1: inception_preprocessing
INCEPTION_V2: inception_preprocessing
INCEPTION_V3: inception_preprocessing
INCEPTION_V4: inception_preprocessing
INCEPTION_RESNET_V2: inception_preprocessing
LENET: lenet_preprocessing
RESNET_V1_18: vgg_preprocessing
RESNET_V1_50: vgg_preprocessing
RESNET_V1_50_8K: vgg_preprocessing
RESNET_V1_101: vgg_preprocessing
RESNET_V1_152: vgg_preprocessing
RESNET_V1_200: vgg_preprocessing
RESNET_V2_50: vgg_preprocessing
RESNET_V2_101: vgg_preprocessing
RESNET_V2_152: vgg_preprocessing
RESNET_V2_200: vgg_preprocessing
RESNEXT_B_50: vgg_preprocessing
RESNEXT_B_101: vgg_preprocessing
RESNEXT_C_50: vgg_preprocessing
RESNEXT_C_101: vgg_preprocessing
VGG: vgg_preprocessing
VGG_A: vgg_preprocessing
VGG_16: vgg_preprocessing
VGG_19: vgg_preprocessing
VGG_A_BN: vgg_preprocessing
VGG_16_BN: vgg_preprocessing
VGG_19_BN: vgg_preprocessing
PVANET: pvanet_preprocessing
MOBILENET_V1: inception_preprocessing
MOBILENET_V2: inception_preprocessing
MOBILENET_V2_140: inception_preprocessing
MOBILENET_V2_035: inception_preprocessing
RESNET_V1_20: cifarnet_preprocessing
NASNET_CIFAR: cifarnet_preprocessing
RESNET_V1_110: cifarnet_preprocessing
NASNET_MOBILE: inception_preprocessing
NASNET_LARGE: inception_preprocessing
PNASNET_LARGE: inception_preprocessing
PNASNET_MOBILE: inception_preprocessing
```

### 2. 模型

moxing 内置了大量业界典型的模型供用户直接使用

#### mox.get_model_fn

```
moxing.tensorflow.nets.get_model_fn( 
name, 
run_mode, 
num_classes, 
weight_decay=0.0, 
data_format='NHWC', 
batch_norm_fused=False, 
batch_renorm=False, 
image_height=None, 
image_width=None,
preprocess_fn=default_preprocess_fn
)
```
该函数接收一个四阶张量`images`，并获得一个有MoXing定义完备的图像分类模型函数，包含了预处理和神经网络两个部分。

如果`run_mode`为`mox.ModeKeys.PREDICT`，由于导出的PB模型只会保存`model_fn`内部的计算流图，`image_height`和`image_width`必须指定。

- 参数说明

  - name：预置模型名称。模型名称列表参见
  - run_mode：运行模式：mox.ModeKeys.TRAIN(训练模式）、mox.ModeKeys.PREDICT（预测模式）、mox.ModeKeys.EVAL（验证模式）
  - num_classes：分类数量。
  - weight_decay：L2正则项系数。
  - data_format：数据格式。一般情况下，对于GPU，NCHW较快；对于GPU，NHWC较快。
  - batch_norm_fused：设置为True时表示，在批量正则化时使用融合算子。
  - batch_renorm：设置为True时表示使用batch renormalization算子。
  - image_height：只有在run_mode为mox.MokeKeys.PREDICT使用，将输入图片转换为image_height。
  - image_width：只有在run_mode为mox.MokeKeys.PREDICT使用，将输入图片转换为image_weight。
  - preprocess_fn：图片输入模型前调用该预处理函数。默认为系统内置的默认预处理函数，设置为None时表示不需要预处理函数。

-   返回值

    network_fn：返回包含预处理函数的图片计算网络，基于input_fn可以计算得到logits（最后一层的特征图）和endpoints（中间层的特征图）。

-   异常

    ValueError：模型名称未定义。

-   示例

```python
mox_model_fn = mox.get_model_fn( 
    name='resnet_v1_50', 
    run_mode=mox.MokeKeys.TRAIN, 
    num_classes=1000, 
    weight_decay=0.0001, 
    data_format='NHWC', 
    batch_norm_fused=True, 
    batch_renorm=False, 
    image_height=224, 
    image_width=224) 
logits, end_points = mox_model_fn(images)
```
#### mox.get_model_meta

```python
moxing.tensorflow.nets.get_model_meta(name)
```
获取一个由MoXing定义完备的模型的元信息。

- 参数说明

  - name：预置模型名称。模型名称列表参见[networkKeys](#networkKeys)

- 返回值

    namedtuple模型元信息，包含两部分：
  
    - default_image_size：模型使用的默认输入图像大小。例如：对imagenet来说，inception系列模型设置为299；其他模型设置为224。
    - default_labels_offset：模型分类标签偏置。例如：imagenet数据集默认分类数量为1001，resnet这样的模型分类数量为1000，需要减去差值，但inception分类数量为1001，则不需要。所以，对resnet，default_labels_offset值为1，对inception，default_labels_offset值为0。

-   示例

```
>>> import moxing.tensorflow as mox
>>> model_meta = mox.get_model_meta('resnet_v1_50')
>>> print(model_meta.default_image_size)
224
>>> print(model_meta.default_labels_offset)
1
>>> print(model_meta.default_logits_pattern)
logits
```

#### mox.ModelSpec

```
class moxing.tensorflow.executor.ModelSpec
```

- 参数说明
  - loss：指定模型的损失值，一个0阶tf.Tensor，或者0阶tf.Tensor的list，多loss案例参考生成对抗模型GAN，当mode==mox.ModeKey.TRAIN时必须提供。
  - var_scopes：指定从loss中计算出的梯度需要对应的变量范围，只有在var_scope范围内的tf.Variable的梯度才会被计算和更新。如果loss是一个0阶tf.Tensor，则var_scope为str的list，指定一个或多个variable_scope。当loss是0阶tf.Tensor的list时，var_scope为二阶list，list[i]表示loss[i]的variable_scope
  - log_info：一个dict，运行作业时控制台需要打印的指标信息，仅支持0阶tf.Tensor，如{'loss': loss, 'acc': accuracy}，当mode==mox.ModeKey.EVAL时必须提供。
  - output_info：一个dict，运行作业的同时输出tf.Tensor中具体的值到output_fn中，当mode==mox.ModeKey.PREDICT时必须提供
  - export_spec：一个dict，导出PB模型时指定输入输出节点，必须是一个mox.ExportSpec的实例(参考API)，当mode==mox.ModeKey.EXPORT时必须提供(注意mox.ModeKey.EXPORT是无法在mox.run中显示指定的，仅当mox.run参数中export_model为有效值时会自动添加该模式)，参考导出PB模型 当model_fn中的“run_mode”为“mox.ModeKeys.PREDICT”时，必须给定。
  - hooks： 一个list, 每个元素都必须是mox.AggregativeSessionRunHook子类的实例，会被tf.Session()执行的hook。

- 返回值

    mox.ModelSpec实例。

-   示例1

```python
def model_fn(inputs, run_mode, **kwargs): 
    x, y_ = inputs 
    W = tf.get_variable(name='W', initializer=tf.zeros([784, 10])) 
    b = tf.get_variable(name='b', initializer=tf.zeros([10])) 
    y = tf.matmul(x, W) + b 
    cross_entropy = tf.reduce_mean( 
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) 
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    return mox.ModelSpec(loss=cross_entropy, log_info={'loss': cross_entropy, 'accuracy': accuracy})
```

- 示例2

如果`model_fn`中使用了`placeholder`，`ModelSpec`需要定义`hooks`来给`placeholder`输入数据。该hook需要为`mox.Aggregative
SessionRunHook`的示例，并且实现三个函数。

```python
def model_fn(inputs, run_mode, **kwargs):
  del inputs
  x = tf.placeholder(dtype=tf.float32, shape=())
  
  class FeedRunHook(mox.AggregativeSessionRunHook):
    def before_run(self, run_context):
      feed_x = 1.0
      return tf.train.SessionRunArgs(fetches=None, feed_dict={x: feed_x})

    def support_aggregation(self):
      return False
  
    def support_sync_workers(self):
      return False
  
    def run_inter_mode(self):
      return False
      
  return mox.ModelSpec(log_info={'x': x}, hooks=[FeedRunHook()])
```

- 示例3

`ModelSpec`支持多个loss，例如GAN模型。当定义多个loss时，建议在variable scope中指定每一个loss的变量范围。

```python
def model_fn(inputs, mode, **kwargs):

  def generator(x, is_training, reuse):
    with tf.variable_scope('generator', reuse=reuse):
      ...
  
  def discriminator(x, is_training, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
      ...
    
  D_loss = ...
  G_loss = ...

  return mox.ModelSpec(loss=[D_loss, G_loss],
                       var_scopes=[['discriminator'], ['generator']],
                       log_info={'D_loss': D_loss, 'G_loss': G_loss})
```
  在上面例子中，判别器`D_loss`和生成器`G_loss`在一个step中使用同一个输入进行优化。其中`D_loss`的梯度只更新`discriminator`标识的变量，`G_loss`的梯度只更新`generator`标识的变量。

- 示例4

在`model_fn`中的节点都是以`tf.Tensor`的形式构建在流图中，MoXing中可以提供`output_fn`用于获取并输出`model_fn`中的`tf.Tensor`的值。

`output_fn`的基本使用方法：

	def input_fn(mode, **kwargs):
	  ...
	
	def model_fn(inputs, mode, **kwargs):
	  ...
	  predictions = ...
	  ...
	  return mox.ModelSpec(..., output_dict={'predictions': predictions}, ...)
	
	def output_fn(outputs, **kwargs):
	  print(outputs)
	
	mox.run(...
	        output_fn=output_fn,
	        output_every_n_steps=10,
	        ...)

其中，在`model_fn`中的`output_dict`指定输出值对应的`tf.Tensor`，在`mox.run`中注册`output_fn`，当`output_every_n_steps`为10时，每经过10个step（注意在分布式运行中，这里的step指的是local_step），`output_fn`就会被调用一次，并且输入参数`outputs`为一个长度为10的`list`，每个元素为一个`dict: {'predictions': ndarray}`。在这里，`outputs`的值即为：

    [{'predictions': ndarray_step_i}, ..., {'predictions': ndarray_step_i+9}]

注意，如果用户使用了多GPU，则`outputs`每次被调用时的输入参数`outputs`的长度为`GPU数量*output_every_n_steps`，分别表示`[(step-0,GPU-0), (step-0,GPU-1), (step-1,GPU-0), ..., (step-9,GPU-1)]`

案例，用ResNet_v1_50做预测，将`max_number_of_steps`和`output_every_n_steps`的值设置一致，也就是说`output_fn`只会被调用一次，输入参数为所有steps的预测结果`prediction`。然后将预测的结果输出到`DataFrame`中并写到文件里。

	import pandas as pd
	import numpy as np
	import tensorflow as tf
	import moxing.tensorflow as mox
	
	slim = tf.contrib.slim
	
	
	def input_fn(mode, **kwargs):
	  meta = mox.ImageClassificationRawMetadata(base_dir='/export1/flowers/raw/split/eval')
	  dataset = mox.ImageClassificationRawDataset(meta)
	  image = dataset.get(['image'])[0]
	  image.set_shape([None, None, 3])
	  image = tf.expand_dims(image, 0)
	  image = tf.image.resize_images(image, size=[224, 224])
	  image = tf.squeeze(image, 0)
	  return image
	
	
	def model_fn(inputs, run_mode, **kwargs):
	  images = inputs[0]
	  
	  logits, endpoints = mox.get_model_fn(
	    name='resnet_v1_50',
	    run_mode=run_mode,
	    num_classes=1000,
	    weight_decay=0.0001)(images)
	  prediction = tf.argmax(logits, axis=1)
	  
	  return mox.ModelSpec(output_info={'prediction': prediction})
	
	
	def output_fn(outputs):
	  df = pd.DataFrame(np.array(outputs))
	  with mox.file.File('s3://dls-test/outputs.txt', 'w') as f:
	    df.to_csv(f)
	
	
	mox.run(input_fn=input_fn,
	        model_fn=model_fn,
	        output_fn=output_fn,
	        output_every_n_steps=10,
	        batch_size=32,
	        run_mode=mox.ModeKeys.PREDICT,
	        max_number_of_steps=10,
	        checkpoint_path='/tmp/checkpoint_path')

#### mox.ExportSpec

该对象用于在`ModelSpec`中注册导出模型参数。模型只在运行完成后导出一次（checkpoint依据mox.run中配置的`save_model_secs`进行存储）
其中`inputs_dict`和`outputs_dict`用于最终预测模型的输入和输出。

- 参数说明
  - inputs_dict：输入参数，为字典格式 
  - outputs_dict：输出参数，为字典格式
  - Version：版本，用于导出tf_serving格式模型，默认设置为-1，表示自动增长
- 返回值

    返回ExportSpec示例

- 样例

```python
def model_fn(inputs, run_mode, **kwargs):
  images, labels = inputs
  mox.mdoel_fn = mox.get_model_fn(
    name='resnet_v1_50',
    run_mode=run_mode,
    num_classes=1000,
    image_height=224,
    image_width=224)
  logits, end_points = mox.mdoel_fn(images)
  labels_one_hot = slim.one_hot_encoding(labels, num_classes)
  loss = tf.losses.softmax_cross_entropy(
    logits=logits, onehot_labels=labels_one_hot,
    label_smoothing=0.0, weights=1.0)
  accuracy_top_1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
  export_spec = mox.ExportSpec(inputs_dict={'images': images},
                               outputs_dict={'logits': logits},
                               version=1)
  return mox.ModelSpec(loss=loss, log_info=log_info, export_spec=export_spec)
```

#### mox.get_collection

```python
moxing.tensorflow.executor.get_collection(key)
```

获取key指定变量集合。在model_fn中使用这个函数代替tf.get_collection。对于多GPU环境，model_fn会被调用多次，使用tf.get_collection可能会引发异常。

- 参数说明
    - key：collection的key值，tf.GraphKeys类中包含了默认的key值。

- 返回值

    collection列表。
  
- 样例1

```python
regularization_losses = mox.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
```

- 样例2

```python
global_variables = mox.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
```


#### mox.var_scope

```python
moxing.tensorflow.executor.var_scope(force_dtype=None)
```

创建变量范围，在这个变量范围内，变量已原始类型创建，然后强制类型转化为指定类型。
- 参数说明
  - force_dtype：精度类型

- 返回值

    变量范围

- 样例

```python
>>> import tensorflow as tf
>>> import moxing.tensorflow as mox

>>> with mox.var_scope(force_dtype=tf.float32):
>>>   a = tf.get_variable('a', shape=[], dtype=tf.float16)

>>> print(a)
Tensor("Cast:0", shape=(), dtype=float16)
>>> print(tf.global_variables()[0])
<tf.Variable 'a:0' shape=() dtype=float32_ref>
```

#### mox.NetworkKeys

```
ALEXNET_V2 = 'alexnet_v2'
CIFARNET = 'cifarnet'
OVERFEAT = 'overfeat'
VGG_A = 'vgg_a'
VGG_16 = 'vgg_16'
VGG_19 = 'vgg_19'
VGG_A_BN = 'vgg_a_bn'
VGG_16_BN = 'vgg_16_bn'
VGG_19_BN = 'vgg_19_bn'
INCEPTION_V1 = 'inception_v1'
INCEPTION_V2 = 'inception_v2'
INCEPTION_V3 = 'inception_v3'
INCEPTION_V4 = 'inception_v4'
INCEPTION_RESNET_V2 = 'inception_resnet_v2'
LENET = 'lenet'
RESNET_V1_18 = 'resnet_v1_18'
RESNET_V1_50 = 'resnet_v1_50'
RESNET_V1_50_8K = 'resnet_v1_50_8k'
RESNET_V1_101 = 'resnet_v1_101'
RESNET_V1_152 = 'resnet_v1_152'
RESNET_V1_200 = 'resnet_v1_200'
RESNET_V2_50 = 'resnet_v2_50'
RESNET_V2_101 = 'resnet_v2_101'
RESNET_V2_152 = 'resnet_v2_152'
RESNET_V2_200 = 'resnet_v2_200'
RESNEXT_B_50 = 'resnext_b_50'
RESNEXT_B_101 = 'resnext_b_101'
RESNEXT_C_50 = 'resnext_c_50'
RESNEXT_C_101 = 'resnext_c_101'
PVANET = 'pvanet'
MOBILENET_V1 = 'mobilenet_v1'
MOBILENET_V1_075 = 'mobilenet_v1_075'
MOBILENET_V1_050 = 'mobilenet_v1_050'
MOBILENET_V1_025 = 'mobilenet_v1_025'
MOBILENET_V2 = 'mobilenet_v2'
MOBILENET_V2_140 = 'mobilenet_v2_140'
MOBILENET_V2_035 = 'mobilenet_v2_035'
RESNET_V1_20 = 'resnet_v1_20'
RESNET_V1_110 = 'resnet_v1_110'
NASNET_CIFAR = 'nasnet_cifar'
NASNET_MOBILE = 'nasnet_mobile'
NASNET_LARGE = 'nasnet_large'
PNASNET_LARGE = 'pnasnet_large'
PNASNET_MOBILE = 'pnasnet_mobile'
```

### 3. 优化器

#### mox.get_optimizer_fn()

```python
moxing.tensorflow.optimizer.get_optimizer_fn(name, learning_rate, scope=None, *args, **kwargs)
```

获取一个内置的优化器
- 参数说明
    - name: 优化器名称
    - learning_rate: 学习率
    - scope：名字范围
    - args: 优化器定义的其他参数
    - kwargs: 其他参数
    
- 异常：

    ValueError：参数为不支持的优化器

- 返回值：

    返回一个优化器示例。

- 样例

```python
mox.run(..., optimizer_fn=mox.get_optimizer_fn('sgd', learning_rate=0.5), ...)
```

- 内置优化器：

  - adadelta
  - adagrad
  - adam
  - ftrl
  - momentum
  - rmsprop
  - sgd

#### learning_rate_scheduler.piecewise_lr

```python
moxing.tensorflow.optimizer.learning_rate_scheduler.piecewise_lr(lr_strategy, num_samples, global_batch_size)
```

- 参数说明
  
  - lr_strategy：学习策略。
  - num_samples: 总的样本数
  - global_batch_size: 全局batch_size，为batch_size*gpu_num*worker

- 样例
```python
  lr_stg = "5:0.1->0.4,10:0.4=>0.8,15:0.8,20:0.08"
  def optimizer_fn():
    global_batch_size = total_batch_size * num_workers if sync_replica else total_batch_size
    lr = learning_rate_scheduler.piecewise_lr(lr_stg,
                                              num_samples=ori_dataset_meta.total_num_samples,
                                              global_batch_size=global_batch_size)
    if flags.optimizer is None or flags.optimizer == 'sgd':
      opt = mox.get_optimizer_fn('sgd', learning_rate=lr)()
    elif flags.optimizer == 'momentum':
      opt = mox.get_optimizer_fn('momentum', learning_rate=lr, momentum=flags.momentum)()
    elif flags.optimizer == 'adam':
      opt = mox.get_optimizer_fn('adam', learning_rate=lr)()
    else:
      raise ValueError('Unsupported optimizer name: %s' % flags.optimizer)
    return opt
  mox.run(input_fn=input_fn,
          model_fn=model_fn,
          optimizer_fn=optimizer_fn,
          run_mode=flags.run_mode,
          inter_mode=mox.ModeKeys.EVAL if use_eval_data else None,
          log_dir=log_dir,
          batch_size=batch_size_per_device,
          auto_batch=False,
          max_number_of_steps=max_number_of_steps,
          log_every_n_steps=flags.log_every_n_steps,
          save_summary_steps=save_summary_steps,
          save_model_secs=save_model_secs,
          checkpoint_path=flags.checkpoint_url,
          export_model=mox.ExportKeys.TF_SERVING)
```  
样例中lr_stg表示0-5个epoch内，学习率从0.1到0.4线性增长，5-10个epoch内，学习率从0.4到0.8指数增长，15到20个epoch内，学习率为常数0.8，20个epoch服务之后为0.08
其中->表示线性增长，=>为指数增长。

### 4. 运行

#### mox.run
```python
moxing.tensorflow.executor.run(input_fn=None, output_fn=None, model_fn=None, optimizer_fn=None, run_mode='invalid', inter_mode=None, batch_size=None, auto_batch=True, log_dir=None, max_number_of_steps=0, checkpoint_path=None, log_every_n_steps=10, output_every_n_steps=0, save_summary_steps=100, save_model_secs=<object object>, export_model=False, fetch_strategy_fn=None, save_model_steps=<object object>)
```
运行一个训练或者其他类型的任务。当run_mode为mox.ModeKeys.TRAIN时，优先从“log_dir”中载入ckpt，如果“log_dir”中没有ckpt，再从“checkpoint_path”中载入ckpt。当run_mode为mox.ModeKeys.EVAL时，从“checkpoint_path”中载入ckpt，“log_dir”只是用来保存一些日志。当使用python的内存数据作为输入时，我们推荐使用tf.data.Dataset。

- 参数说明

  - input_fn: 用户定义的input_fn。
  - output_fn: 用户定义的output_fn，output_fn的输入参数为ModelSpec指定的output_info。
  - model_fn: 用户定义的model_fn。
  - optimizer_fn: 用户定义的optimizer_fn。
  - run_mode：运行模式。支持mox.ModeKeys.TRAIN、mox.ModeKeys.EVAL、mox.ModeKeys.PREDICT。
  - inter_mode: 中间模式。支持mox.ModeKeys.TRAIN、mox.ModeKeys.EVAL、mox.ModeKeys.PREDICT。如果一个hook支持中间模式（例如mox.EarlyStoppingHook),这个hook会在中间模式运行时执行。
  - batch_size: Mini-batch size。
  - auto_batch: 如果设置为True，会自动增加一个batch_size的维度。默认值为True。
  - log_dir: summary和checkpoint的保存路径。
  - max_number_of_steps: 每一个worker的最大执行步数。
  - checkpoint_path： 载入模型的路径。在run_mode为mox.ModeKeys.TRAIN时无效。
  - log_every_n_steps：打印控制台信息频率，打印内容和用户在ModelSpec中定义的log_info相关。
  - output_every_n_steps：每经过output_every_n_steps，output_fn将会被调用一次。
  - save_summary_steps：保存summary的频率，以步数计算。当“run_mode”设置为“mox.ModeKeys.EVAL”时无效。
  - save_model_secs：保存checkpoint的频率，以秒计算。当“run_mode”设置为“mox.ModeKeys.EVAL”时无效。如果save_model_secs和save_model_steps同时提供，按save_model_secs为准。
  - export_model：表示运行完成后是否导出模型，默认为False
  - save_model_steps：保存checkpoint的频率，以步数计算。如果save_model_secs和save_model_steps同时提供，按save_model_secs为准。
- 异常

    ValueError：“run_mode ”不为“mox.ModeKeys.TRAIN” 或 “mox.ModeKeys.EVAL”。

- 示例1

  获取metadata
```python
batch_size = 50
dataset_meta = mox.get_dataset_meta('mnist')
model_meta = mox.get_model_meta('lenet')
split_name_train, split_name_eval = dataset_meta.splits_to_sizes.keys()
image_size = model_meta.default_image_size
labels_offset = model_meta.default_labels_offset
num_classes = dataset_meta.num_classes - labels_offset
```
使用内部数据集：
  
```python
def input_fn(run_mode, **kwargs):
  dataset = mox.get_dataset(name='mnist', split_name=split_name_train,
                            dataset_dir='s3://bucket/mnist/', capacity=4000)
  image, label = dataset.get(['image', 'label'])
  data_augmentation_fn = mox.get_data_augmentation_fn(
    name='lenet', run_mode=mox.ModeKeys.TRAIN,
    output_height=image_size, output_width=image_size)
  image = data_augmentation_fn(image)
  label -= labels_offset
  return image, label
```  

使用自定义数据集：

```python
from tensorflow.examples.tutorials.mnist import input_data
# Here returns a batch of data with rank=4, so we need to set auto_batch=False
# When calling mox.run(...)
def input_fn(run_mode, **kwargs):
  mnist = input_data.read_data_sets('s3://bucket/mnist/, one_hot=True)
  def gen():
    while True:
      yield mnist.train.next_batch(batch_size)
  ds = tf.data.Dataset.from_generator(
      gen, output_types=(tf.float32, tf.int64),
      output_shapes=(tf.TensorShape([None, 784]), tf.TensorShape([None, 10])))
  return ds.make_one_shot_iterator().get_next()
```  
定义model_fn

```python
def model_fn(inputs, run_mode, **kwargs):
  images, labels = inputs
  mox.mdoel_fn = mox.get_model_fn(
    name='lenet',
    run_mode=mox.ModeKeys.TRAIN,
    num_classes=num_classes,
    weight_decay=0.0001,
    data_format='NHWC',
    batch_norm_fused=True,
    batch_renorm=False,
    image_height=image_size,
    image_width=image_size)
  logits, end_points = mox.mdoel_fn(images)
  labels_one_hot = slim.one_hot_encoding(labels, num_classes)
  loss = tf.losses.softmax_cross_entropy(
    logits=logits, onehot_labels=labels_one_hot,
    label_smoothing=0.0, weights=1.0)
  accuracy_top_1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
  log_info = {'loss': loss,
              'top1': accuracy_top_1}
  top_1_confidence, top_1_label = tf.nn.top_k(tf.nn.softmax(logits), k=1)
  top_1_label += labels_offset
  export_spec = mox.ExportSpec(inputs_dict={'images': images},
                               outputs_dict={'labels': top_1_label,
                                             'confidences': top_1_confidence},
                               version=1)
  return mox.ModelSpec(loss=loss, log_info=log_info, export_spec=export_spec)
```  

启动训练任务：

```python
mox.run(input_fn=input_fn,
        model_fn=model_fn,
        optimizer_fn=mox.get_optimizer_fn('sgd', learning_rate=0.01),
        run_mode=mox.ModeKeys.TRAIN,
        batch_size=batch_size,
        max_number_of_steps=2000)
```  

- 示例2：

当使用支持inter-mode的hook(例如mox.EarlyStoppingHook)时，需要定义inter_mode来支持该hook。
Early Stopping是建立在同时提供训练集和验证集的前提上，当训练的模型在验证数据集上的指标(minotor)趋于稳定时，则停止训练。

样例代码，训练一个ResNet_v1_50，每训练一个epoch就在验证数据集上观察评价指标`accuracy`，当连续3次评价指标`accuracy`没有上升(第一次无法判断上升还是下降，所以至少评价4次)，则停止训练。

	import tensorflow as tf
	import moxing.tensorflow as mox
	slim = tf.contrib.slim
	
	
	def input_fn(mode, **kwargs):
	  if mode == mox.ModeKeys.TRAIN:
	    meta = mox.ImageClassificationRawMetadata(base_dir='/export1/flowers/raw/split/train')
	  else:
	    meta = mox.ImageClassificationRawMetadata(base_dir='/export1/flowers/raw/split/eval')
	  
	  dataset = mox.ImageClassificationRawDataset(meta)
	  
	  image, label = dataset.get(['image', 'label'])
	  image = mox.get_data_augmentation_fn(
	    name='resnet_v1_50',
	    run_mode=mode,
	    output_height=224,
	    output_width=224)(image)
	  return image, label
	
	
	def model_fn(inputs, mode, **kwargs):
	  images, labels = inputs
	  
	  logits, endpoints = mox.get_model_fn(
	    name='resnet_v1_50',
	    run_mode=mode,
	    num_classes=1000)(images)
	  
	  labels_one_hot = slim.one_hot_encoding(labels, 1000)
	  loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_one_hot)
	  
	  accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
	  
	  early_stop_hook = mox.EarlyStoppingHook(
	    monitor_info={'accuracy': accuracy},
	    monitor='accuracy',
	    batch_size=32,
	    samples_in_train=640,
	    samples_in_eval=32,
	    evaluate_every_n_epochs=1,
	    min_delta=0.01,
	    patience=3,
	    prefix='[EarlyStopping]'
	  )
	  
	  return mox.ModelSpec(loss=loss,
	                       log_info={'loss': loss, 'accuracy': accuracy},
	                       hooks=[early_stop_hook])
	
	
	mox.run(input_fn=input_fn,
	        model_fn=model_fn,
	        optimizer_fn=mox.get_optimizer_fn('sgd', learning_rate=0.01),
	        batch_size=32,
	        run_mode=mox.ModeKeys.TRAIN,
	        inter_mode=mox.ModeKeys.EVAL,
	        max_number_of_steps=10000)

控制台输出日志可能会如下：

	INFO:tensorflow:step: 0(global step: 0)	sample/sec: 15.875	loss: 7.753	accuracy: 0.000
	INFO:tensorflow:step: 10(global step: 10)	sample/sec: 42.087	loss: 3.451	accuracy: 0.312
	INFO:tensorflow:[EarlyStopping] step: 19 accuracy: 0.000
	INFO:tensorflow:step: 20(global step: 20)	sample/sec: 40.802	loss: 4.920	accuracy: 0.250
	INFO:tensorflow:step: 30(global step: 30)	sample/sec: 41.427	loss: 4.368	accuracy: 0.281
	INFO:tensorflow:[EarlyStopping] step: 39 accuracy: 0.000
	INFO:tensorflow:step: 40(global step: 40)	sample/sec: 41.678	loss: 2.614	accuracy: 0.281
	INFO:tensorflow:step: 50(global step: 50)	sample/sec: 41.816	loss: 2.788	accuracy: 0.219
	INFO:tensorflow:[EarlyStopping] step: 59 accuracy: 0.000
	INFO:tensorflow:step: 60(global step: 60)	sample/sec: 41.407	loss: 2.861	accuracy: 0.094
	INFO:tensorflow:step: 70(global step: 70)	sample/sec: 41.929	loss: 2.075	accuracy: 0.469
	INFO:tensorflow:[EarlyStopping] step: 79 accuracy: 0.000
	
	Process finished with exit code 0
