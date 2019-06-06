## MoXing-TensorFlow 详细使用手册

[TOC]

## 1 MoXing程序基本结构

	def input_fn(mode, **kwargs):
      ...
      return input_0, input_1, ..., input_n

    def model_fn(inputs, mode, **kwargs):
      inputs_0, inputs_1, ..., inputs_n = inputs
      ...
      return mox.ModelSpec(...)

    mox.run(...)

一个训练ResNet-50的案例：

	import tensorflow as tf
	import moxing.tensorflow as mox
	slim = tf.contrib.slim
	

	def input_fn(mode, **kwargs):
	  meta = mox.ImageClassificationRawMetadata(base_dir='/export1/flowers/raw/split/train')
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
	    num_classes=1000,
	    weight_decay=0.0001)(images)

	  loss = tf.losses.softmax_cross_entropy(
	    logits=logits, onehot_labels=slim.one_hot_encoding(labels, 1000))
	  
	  return mox.ModelSpec(loss=loss)
	
	mox.run(input_fn=input_fn,
	        model_fn=model_fn,
	        optimizer_fn=mox.get_optimizer_fn('sgd', learning_rate=0.01),
	        batch_size=32,
	        run_mode=mox.ModeKeys.TRAIN,
	        max_number_of_steps=100)


`input_fn`定义了模型的输入，`input_i`表示第i个输入（比如在图像分类中，可以是image和label），每个输入都必须是一个[tf.Tensor](https://www.tensorflow.org/programmers_guide/tensors)类型的变量

`model_fn`定义了模型的主体结构，`inputs`是一个`list`，对应了`input_fn`中的所有返回值。

在ResNet-50的案例中，`input_fn`返回的是`image `和`label`，分别是`Tensor(shape=[224, 224, 3])`和`Tensor(shape=[])`，由于batch_size为32，那么在model_fn中的inputs就是一个包含了图像和分类标签的list: `[Tensor(shape=[32, 224, 224, 3]), Tensor(shape=[32])]`

`mox.run`则是将整个作业进程运行起来。

## 2. 运行参数

MoXing没有对运行参数定义特殊的API，用户可以根据自己的习惯定义运行参数，建议使用TensorFlow的flags组件来定义。

### 2.1 用tf.flags定义运行参数（TensorFlow-1.4）

flags是由TensorFlow-1.4提供的一种定义运行参数的组件，[参考地址](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/platform/flags.py)。从TensorFlow-1.5开始，TensorFlow将flags组件替换成了[absl](https://github.com/abseil/abseil-py)

> #### 踩坑 2-1-1 (关键字：flags, FlagValues)
> 
> 如果代码中出现形如这样的错误信息：
> 
> 	  AttributeError: 'module' object has no attribute '_FlagValues'
> 
> 很有可能就是在使用MoXing-TensorFlow-1.4的情况下TensorFlow的版本不是1.4的，通过以下命令查看TensorFlow版本
> 
> 	  pip list | grep tensorflow
> 
> 或者在python中输入
> 
> 	  import tensorflow as tf
> 	  tf.__version__

使用tf.flags定义一个运行参数如下：

	tf.flags.DEFINE_string(name='data_url', default='./', help='Data directory.')
    flags = tf.flags.FLAGS
    print(flags.data_url)

- `flag_name`: 运行参数名，例如外部命令行运行时添加参数--data_url=xxx
- `default_value`: 当外部命令行参数不传入该参数时使用的缺省值
- `docstring`: 命令行帮助文档信息

在TensorFlow-1.4中，tf.flags仅提供了4种数据类型:

- tf.flags.DEFINE_string
- tf.flags.DEFINE_integer
- tf.flags.DEFINE_float
- tf.flags.DEFINE_boolean

> #### 踩坑 2-1-2 (关键字：运行参数, list)
> 
> 如果本来的脚本中定义的运行参数中有list，但是tf.flags不支持list怎么办？
> 
> 用string表达list，然后在代码中解析出来，比如：
> 
>     tf.flags.DEFINE_string(name='int_list', default='0,1,2', help=None)
>     flags = tf.flags.FLAGS
>     int_list = [int(int_item) for int_item in flags.int_list.split(',')]

### 2.2 使用argparse定义运行参数（不推荐）

argparse是python自带的运行参数定义模块，具体使用方法请参考[相关文档](https://docs.python.org/3/library/argparse.html)

> #### 踩坑 2-2-1 (关键字：argparse, unrecognized)
> 
> 在外部命令行给入的参数（或者在DLS服务中输入的`运行参数`）如果在脚本中没有被argparse定义过，则会出现错误：
> 
> 	  error: unrecognized arguments: --data_url=xxx
> 
> 解决办法，用以下方法：
> 
> 	  args, unparsed = parser.parse_known_args()
> 
> 代替
> 
>     args = parser.parse_args()

### 2.3 MoXing定义的默认运行参数

MoXing本身会定义一些默认的运行参数，[具体参考](http://x)，这些参数不需要在用户脚本中额外定义，当用户使用如下导入代码时即生效，直接可以在外部命令行或DLS服务的`运行参数`中传入。

	import moxing.tensorflow as mox

以下列举几个重要的参数：

- `--num_gpus`: 使用GPU的数量，如果使用的是CPU，这项参数不要填写，或者给`1`，缺省值为`1`

以下四个参数是TensorFlow推荐的分布式运行参数，具体可以参考[TensorFlow官方文档](https://www.tensorflow.org/deploy/distributed)。

- `--job_name`: ps或worker
- `--task_index`: ps或worker进程的序号，一般情况下task_index为0的worker为chief worker (也可以认为是master节点，master节点在物理上并不存在，是一个逻辑节点）
- `--ps_hosts`: ps的ip和端口，多个节点以`,`分割。
- `--worker_hosts`: worker的ip和端口，多个节点以`,`分割。

例如，启动一个2个节点的训练作业，每个节点使用4个GPU，参数配置如下：

	# 节点0启动ps进程参数（对应IP地址为192.168.1.100）
	--job_name=ps
    --task_index=0
    --ps_hosts=192.168.1.100:2222,192.168.1.101:2222
    --worker_hosts=192.168.1.100:2223,192.168.1.101:2223

	# 节点0启动worker进程参数（对应IP地址为192.168.1.100）
	--job_name=worker
    --task_index=0
    --ps_hosts=192.168.1.100:2222,192.168.1.101:2222
    --worker_hosts=192.168.1.100:2223,192.168.1.101:2223
    --num_gpus=4

	# 节点1启动ps进程参数（对应IP地址为192.168.1.101）
	--job_name=ps
    --task_index=1
    --ps_hosts=192.168.1.100:2222,192.168.1.101:2222
    --worker_hosts=192.168.1.100:2223,192.168.1.101:2223

	# 节点1启动ps进程参数（对应IP地址为192.168.1.101）
	--job_name=worker
    --task_index=1
    --ps_hosts=192.168.1.100:2222,192.168.1.101:2222
    --worker_hosts=192.168.1.100:2223,192.168.1.101:2223
    --num_gpus=4

MoXing内部定义运行参数的相关API：[mox.get_flag](http://moxing.inhuawei.com/moxing.tensorflow.executor.html#moxing.tensorflow.executor.get_flag), [mox.set_flag](http://moxing.inhuawei.com/moxing.tensorflow.executor.html?highlight=set_flag#moxing.tensorflow.executor.set_flag)


> #### 踩坑 2-3-1 (关键字：分布式waiting, 分布式阻塞)
> 
> 所有分布式进程都启动后，worker进程不断在打印如下信息，没有开始训练。
> 
> 	  2018-04-13 14:01:47.653259: I tensorflow/core/distributed_runtime/master.cc:221] CreateSession still waiting for response from worker: /job:ps/replica:0/task:0
> 	  2018-04-13 14:01:47.653308: I tensorflow/core/distributed_runtime/master.cc:221] CreateSession still waiting for response from worker: /job:ps/replica:0/task:1
> 	  2018-04-13 14:01:47.653315: I tensorflow/core/distributed_runtime/master.cc:221] CreateSession still waiting for response from worker: /job:worker/replica:0/task:1
> 
> 解决办法
> 
> 首先保证你的`job_name`, `task_index`, `ps_hosts`, `worker_hosts`这四个参数都是正确的。
> 
> 考虑以下这种情况是不正确的：
> 
> 在一个IP为`192.168.1.100`的机器上启动ps或worker进程：
> 
> 	  --job_name=worker
> 	  --task_index=1
> 	  --ps_hosts=192.168.1.100:2222,192.168.1.101:2222
> 	  --worker_hosts=192.168.1.100:2223,192.168.1.101:2223
> 
> 因为该进程启动位置是`192.168.1.100`，但是运行参数中指定的`task_index`为`1`，对应的IP地址是`ps_hosts`或`worker_hosts`的第二项（第一项的`task_index`为`0`)，也就是`192.168.1.101`，和进程本身所在机器的IP不一致。
> 
> 另外一种情况也会导致该问题的发生，从TensorFlow-1.4开始，分布式会自动使用环境变量中的代理去连接，如果运行的节点之间不需要代理互连，那么将代理的环境变量移除即可，在脚本的开始位置添加代码：
> 
>     # 注意这段代码必须写在import tensorflow as tf或者import moxing.tensorflow as mox之前
>     import os
>     os.enrivon.pop('http_proxy')
>     os.enrivon.pop('https_proxy')

### 2.4 DLS服务中训练作业的运行参数

<img src="../../../docs/images_moxing_tensorflow/dls_training_job_create.jpg" />

DLS服务-创建训练作业的界面。DLS服务中，用户不需要考虑代理问题。

如果用户使用原生TensorFlow-API的脚本进行训练，用户需要定义DLS服务规定的几项参数，说明如下：

- `训练数据集`中填写的内容会以运行参数`--data_url`的形式传入到`启动文件`指定的脚本中

- 在选择`计算节点规格`和`计算节点个数`时会产生多GPU和分布式相关的5项参数。如，`计算节点规格`为`4*P100`，即使用4个P100的GPU，所以--num_gpus=4，计算节点个数为`2`，则表示分布式运行并使用2个节点，则会使用`启动文件`指定的脚本启动4个进程，每个进程都会按规范填入`job_name`, `task_index`, `ps_hosts`, `worker_hosts`，其中的IP地址和端口是由DLS预先分配指定好的，用户直接在脚本中使用即可。

如果用户使用基于MoXing的脚本进行训练，则不需要定义多GPU和分布式的参数，也不需要编写多GPU和分布式相关的代码。但用户仍然需要定义`--data_url`这个运行参数。

> #### 踩坑 2-4-1 (关键字：路径不存在，OBS)
> 
> 用户在训练脚本中想读取一些文件或者写入一些日志文件，发现找不到文件或路径。
> 
> 如果你想读取OBS上桶名为`dls-test`中的文件`a.txt`，而且使用了如下代码：
> 
>     f=open('/dls-test/a.txt', 'r')
> 
> 则会出现如下错误：
> 
>     IOError: [Errno 2] No such file or directory: '/dls-test/a.txt'
> 
> 在DLS服务中，作业在容器中启动，一般来说对文件的读写都是基于OBS或其他数据服务的，比如打开一个OBS上，桶名为`dls-test`中的文件`a.txt`，那么要使用OBS的路径`s3://dls-test/a.txt`，但是当你发现把代码修改为：
> 
>     f=open('s3://dls-test/a.txt', 'r')
> 
> 依然会出现同样找不到文件的错误，那是因为大多数文件读写的API都不支持s3路径，用户如果涉及文件读写的操作，都必须使用支持s3的文件接口，如`obs-sdk`, `obscmd`, `s3cmd`, `tf.gfile`, `mox.file`
> 
> 以下代码利用`tf.gfile`操作文件：
>     
>     import tensorflow as tf
>     f=tf.gfile.Open('s3://dls-test/a.txt', 'r')
> 
> 在DLS中，用户不需要配置OBS的`ak`, `sk`等信息，这些都已经默认配置好了。

建议用户将输入数据集、代码、输出日志这三者的路径提前规划好，不要放在相同的目录下，由于`代码目录`有10MB的容量大小限制，所以如果输入数据集或者输出日志存放在`代码目录`中可能会导致不能提交作业。

> #### 踩坑 2-4-2 (关键字：No module，import不存在)
> 
> 假设用户将代码结构如下：
> 
> 	  project_dir
>         |- main.py
> 	    |- module_dir
> 	      |- module_file.py
> 	  
> 用户在main.py中有代码
> 
>     from module_dir import module_file
> 
> 发生如下错误：
> 
> 	  Traceback (most recent call last):
> 	    File "project_dir/main.py", line 1, in <module>
> 	      from module_dir import module_file
> 	  ImportError: No module named module_dir
> 
> 这份代码如果在本地运行，需要将project_dir加入到PYTHONPATH或者将整个project_dir安装到site-package中才能运行，但是在DLS中没有办法进行这些操作，所以可以将project_dir加入到sys.path中解决该问题，步骤如下：
> 
> 首先保证被import的module中有`__init__.py`存在，创建`module_dir`的`__init__.py`，代码结构如下：
> 
> 	  project_dir
> 	    |- main.py
> 	    |- module_dir
> 	      |- __init__.py
> 	      |- module_file.py
> 
> 在main.py中将project_dir添加到sys.path中，由于用户不知道project_dir在容器中的位置，所以利用相对路径：
> 
>     import os
>     import sys
>     # __file__为获取当前执行脚本main.py的绝对路径
>     # os.path.dirname(__file__)获取main.py的父目录，即project_dir的绝对路径
>     current_path = os.path.dirname(__file__)
>     sys.path.append(current_path)
>     # 在sys.path.append执行完毕之后再导入其他模块
>     from module_dir import module_file
> 
> 再次提交作业运行即可。

如果在DLS服务中`训练数据集`的值没有填写，脚本依然会收到`--data_url`的参数，参数值为空。

> #### 踩坑 2-4-3 (关键字：默认运行参数，运行参数不正确)
> 
> 在用户脚本中定义了运行参数如下：
> 
> 	  tf.flags.DEFINE_string('data_url', default='s3://xxx', help=None)
> 
> 在DLS中`训练数据集`项不填写，这种情况下用户可能会认为脚本会取运行参数`data_url`的默认值`default_value`，即`'s3://xxx'`，但实际情况是，即使`训练数据集`项不填写，脚本依然会收到`--data_url`的参数，只是参数的值为空，即：`python xxx.py --data_url=''`，所以默认值无法起作用。 
> 
> 如果是用户自定义的`运行参数`，考虑以下3种情况：
> 
> 1) 用户添加了一个`运行参数`: `my_flag = xxx`，则脚本会收到`--my_flag=xxx`
> 
> 2) 用户添加了一个`运行参数`: `my_flag`，但是没有填入任何的值，则脚本会收到`--my_flag=''`
> 
> 3）用户没有添加`运行参数`: `my_flag`，则脚本不会收到`--my_flag`这个运行参数

> 所以`训练数据集`类似这里的第`2)`种情况

## 3. 输入

MoXing将数据的输入定义在input_fn方法中，并在mox.run时注册该方法。

基本方法：

	def input_fn(mode, **kwargs):
	  ...
	  return input_0, input_1, ...

    mox.run(..., input_fn=input_fn, ...)

输入参数：

- `mode`: 当前调用`input_fn`时的运行模式，需要用户在`input_fn`中做好判断使用相应的数据集和数据集增强、预处理方法。`mox.ModeKeys`中的一个，[参考API](http://moxing.inhuawei.com/moxing.tensorflow.executor.html?highlight=modekeys#moxing.tensorflow.executor.ModeKeys)。
- `**kwargs`: 扩展参数的预留位置。

返回值：

- `tf.Tensor`或`tf.Tensor`的`list`

`input_fn`中的返回值包含了2种情况：

1) auto_batch=True

当用户实现的`input_fn`的返回值`input_i`不包含batch_size维度时，在mox.run中用户需要添加参数：

	mox.run(...
            batch_size=32,
            auto_batch=True,
            ...)

MoXing会自动将`input_fn`中的输入以batch为单位聚合，并将含有batch_size维度的`Tensor`输入到`model_fn`中，例(`auto_batch`的缺省值为`True`)：
	
	def input_fn(mode, **kwargs):
      ...
	  return image, label

    def model_fn(inputs, mode, **kwargs):
      images, labels = inputs
      ...

	mox.run(...
	        batch_size=32,
       		...)    

`input_fn`的返回值：`image`是一个`[224, 224, 3]`的`Tensor`，`label`是一个`[1000]`的`Tensor`

`model_fn`的输入参数：`images`是一个`[32, 224, 224, 3]`, `labels`是一个`[32, 1000]`的`Tensor`

2)auto_batch=False

当`auto_batch`为`False`时，用户就需要自己在`input_fn`中将组织batch，**注意：不论auto_batch的值时什么，mox.run中的batch_size都必须填写**（用于计算运行时吞吐量），例：

    def input_fn(mode, **kwargs):
      ...
	  return images, labels

    def model_fn(inputs, mode, **kwargs):
      images, labels = inputs
      ...

	mox.run(...
            auto_batch=False,
	        batch_size=32,
       		...)   

`input_fn`的返回值：`images`是一个`[32, 224, 224, 3]`的`Tensor`，`label`是一个`[32, 1000]`的`Tensor`

`model_fn`的输入参数：`images`是一个`[32, 224, 224, 3]`, `labels`是一个`[32, 1000]`的`Tensor`

### 3.1 读取图像分类数据集Raw Data

基本使用方法：

	def input_fn(mode, **kwargs):
	  meta = mox.ImageClassificationRawMetadata(base_dir='/export1/flowers/raw/split/train')
	  dataset = mox.ImageClassificationRawDataset(meta)
	  image, label = dataset.get(['image', 'label'])
	  # 将图片resize到相同大小并添加shape信息，或者还可以增加一些数据增强方法。
	  image = tf.expand_dims(image, 0)
	  image = tf.image.resize_bilinear(image, [224, 224])
	  image = tf.squeeze(image)
	  image.set_shape([224, 224, 3])
	  return image, label

API参考文档： [ImageClassificationRawMetadata](http://moxing.inhuawei.com/moxing.tensorflow.datasets.html#moxing.tensorflow.datasets.ImageClassificationRawMetadata)， [ImageClassificationRawDataset](http://moxing.inhuawei.com/moxing.tensorflow.datasets.html#moxing.tensorflow.datasets.ImageClassificationRawDataset)

数据集必须是如下目录结构的：

	base_dir
		|- label_0
			|- 0_0.jpg
			|- 0_1.jpg
			…
			|- 0_x.jpg
		|- label_1
			|- 1_0.jpg
			|- 1_1.jpg
			…
			|- 1_y.jpg
		…
		|- label_m
			|- m_0.jpg
			|- m_1.jpg
			…
			|- m_z.jpg
        |- labels.txt

其中`label_0`, `label_1`, ..., `label_m`代表(m+1)个分类。第i个分类的名称即为`label_i`。
labels.txt是一个label_index到label_string的映射，可以提供也可以不提供。labels.txt必须是如下内容：

	0: label_0
    1: label_1
    ...
    m: label_m

也就是当模型输出的label值为`i`时（训练或预测），对应的label名称是`label_i`

> #### 踩坑 3-1-1 (关键字：预测作业，预测标签错误)
> 
> 利用训练好的模型做预测服务时，发现正确率非常低。
> 
> 当使用纯图像文件数据集时，如果labels.txt没有提供，存储数据集的文件系统对分类目录的排序顺序即为label的顺序，比如在用户存储的文件系统中数据集以以下顺序排列（也就是os.listdir得到的list中的顺序）：
> 
>     base_dir
> 		|- label_0
> 		|- label_1
>         |- label_10
>         |- label_11
> 		|- label_2
>         ...
> 
> 则等效于labels.txt中写入内容：
> 
> 	  0: label_0
> 	  1: label_1
> 	  2: label_10
> 	  3: label_11
> 	  4: label_2
>     ...
> 
> 但是有可能在预测服务的客户端中又以另一种完全不同的映射顺序将服务端返回的label_id值转换成label_string，导致预测结果不准确。为了防止这种情况的发生，最好提供labels.txt，用户能更好的掌握服务端返回值和实际预测结果的映射关系。

如果在`input_fn`中涉及多个数据集，如训练集、验证集等，使用`mode`将`input_fn`的返回值做分支判断，MoXing中使用常量`mox.ModeKeys`来定义模式，分别有`训练态`: `mox.ModeKeys.TRAIN`, `验证态`: `mox.ModeKeys.EVAL`, `预测态`: `mox.ModeKeys.PREDICT`。还有一个隐式状态：`导出态`: `mox.ModeKeys.EXPORT`，由MoXing内部使用，在阐述模型部分的章节说明。例：

	def input_fn(mode, **kwargs):
      if mode == mox.ModeKeys.TRAIN:
	    meta = mox.ImageClassificationRawMetadata(base_dir='/export1/flowers/raw/split/train')
      else:
        meta = mox.ImageClassificationRawMetadata(base_dir='/export1/flowers/raw/split/eval')
	  dataset = mox.ImageClassificationRawDataset(meta)
	  image, label = dataset.get(['image', 'label'])
	  ...
	  return image, label

### 3.2 读取tfrecord

读取tfrecord文件和生成tfrecord文件的代码是相关的，tfrecord文件中以键值对的形式存放了数据

例，考虑读取一个`key`值含有`image`和`label`的tfrecord，`image`和`label`都以字节流的形式储存于tfrecord文件中：

    import tensorflow as tf
    import moxing.tensorflow as mox
    slim = tf.contrib.slim

	keys_to_features = {
		'image': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),
		'label': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),
	}
	
	items_to_handlers = {
		'image': slim.tfexample_decoder.Tensor('image'),
		'label': slim.tfexample_decoder.Tensor('label'),
	}
	
	dataset = mox.get_tfrecord(dataset_dir='/xxx', 
                               file_pattern='*.tfrecord',
                               keys_to_features=keys_to_features,
	                           items_to_handlers=items_to_handlers)
	
	image, label = dataset.get(['image', 'label'])

例，考虑读取一个`key`值含有`image/encoded`, `image/format`, `image/class/label`的tfrecord，并同时将image从字节流解码为像素值张量：

    import tensorflow as tf
    import moxing.tensorflow as mox
    slim = tf.contrib.slim

	keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/class/label': tf.FixedLenFeature(
        [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }

    items_to_handlers = {
      'image': slim.tfexample_decoder.Image(shape=[28, 28, 1], channels=1),
      'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
    }

    dataset = mox.get_tfrecord(dataset_dir='/xxx’,
                               file_pattern='*.tfrecord',
                               keys_to_features=keys_to_features,
                               items_to_handlers=items_to_handlers)

    image, label = dataset.get(['image', 'label'])

相关API: [mox.get_tfrecord](http://moxing.inhuawei.com/moxing.tensorflow.datasets.html?highlight=get_tfrecord#moxing.tensorflow.datasets.get_tfrecord)，tfrecord的用法请参考[TensorFlow官方教程](https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data)

### 3.3 利用`tf.data`模块读取任意数据集

用户实现数据集类`my_dataset`，提供`next()`方法获取下一份数据，可以是一个batch的samples也可以是单个sample，用`auto_batch`来做控制。基本写法如下：

    import tensorflow as tf
    import moxing.tensorflow as tf
    import my_dataset

	def input_fn(run_mode, **kwargs):

	  def gen():
	    while True:
	      yield my_dataset.next()

	  ds = tf.data.Dataset.from_generator(
          gen, 
          output_types=(tf.float32, tf.int64),
	      output_shapes=(tf.TensorShape([224, 224, 3]), tf.TensorShape([1000])))

	  return ds.make_one_shot_iterator().get_next()

在使用这种方法时，由于数据的产生顺序完全取决于用户实现的代码，MoXing无法保证数据的shuffle，所以用户必须确保自己提供的`my_dataset.next()`具有数据随机性。

### 3.4 数据增强

MoXing提供了部分的[数据增强方法](http://moxing.inhuawei.com/moxing.tensorflow.preprocessing.html?highlight=preprocessingkeys#moxing.tensorflow.preprocessing.PreprocessingKeys)，这些数据增强方法都是和模型名称绑定，如：

    data_augmentation_fn = mox.get_data_augmentation_fn(
          name='resnet_v1_50', run_mode=mox.ModeKeys.TRAIN,
          output_height=224, output_width=224)
    image = data_augmentation_fn(image)

即获取一个`resnet_v1_50`模型在`训练态`时对应的数据增强方法。

用户也可以自定义数据增强方法：

	def input_fn(mode, **kwargs):
        ...
		image, label = dataset.get(['image', 'label'])
		image = my_data_augmentation_fn(image)
		return image, label

需要注意的是，从dataset.get()中获取的image如果没有shape信息，甚至每张图片的大小不一致，可能会导致后续的算子出现错误，所以推荐在对image操作之前，将image的size统一（当模型有batch_size维度时，要求输入数据的shape必须相同），并将shape信息进行补全，如：

	def input_fn(mode, **kwargs):
        ...
		image, label = dataset.get(['image', 'label'])
        # 将image统一至[224, 224, 3]的大小并补全shape信息
		image = tf.expand_dims(image, 0)
		image = tf.image.resize_bilinear(image, [224, 224])
		image = tf.squeeze(image)
		image.set_shape([224, 224, 3])
        # 调用自定义数据增强方法，如水平翻转
		image = tf.image.flip_left_right(image)
		return image, label

> #### 踩坑 3-4-1 (关键字：训练作业等待，num_samples)
> 
> 运行作业日志提示如下信息，并经过很长时间都没有反应。
> 
> 	  INFO:tensorflow:Find tfrecord files. Using tfrecord files in this job.
> 	  INFO:tensorflow:Automatically extracting num_samples from tfrecord. If the dataset is large, it may take some time. You can also manually specify the num_samples to Dataset to save time.
> 
> 这个现象的原因是用户使用的tfrecord文件作为数据集，MoXing在扫描tfrecord文件并抽取总样本数量的值，如果tfrecord文件所在位置是一个网络文件系统，而该文件系统的IO速度不高，很可能在这一步会停留很久。
> 
> 解决办法：根据用户数据集的实际情况填写tfrecord文件的总样本数量。
> 
> 可能涉及的API：
> 
> 1) mox.get_tfrecord
> 
> 	  mox.get_tfrecord(..., num_samples=1000, ...)
> 
> 2) 所有`BaseTFRecordMetadata`类以及其子类：
> 
> 	  BaseTFRecordMetadata(..., num_samples=1000, ...)
> 
> 3) DLS服务中的预置模型库：
> 
> 当使用的是未划分的单数据集时，即train或eval数据集，手动指定运行参数： `samples_per_epoch`，表示所选数据集中的总样本数量。
> 
> 当使用的是划分好的数据集时，即train和eval数据集，手动指定运行参数： `samples_per_epoch`和`samples_per_epoch_eval`，分别表示所选train数据集和eval数据集中的总样本数量。

-

> #### 踩坑 3-4-2 (关键字：resize，图像损坏)
> 在利用`tf.image.resize_images`对图像进行resize时，默认使用的是`ResizeMethod.BILINEAR`方法，如果输入一张刚解码后的图片（类型为int8），则会导致图片严重失真。
> 考虑以下代码片段：
> 
> 	raw_image = tf.gfile.FastGFile("../xxx.jpg", 'rb').read()
> 	raw_image = tf.image.decode_jpeg(raw_image)
> 	images = tf.image.resize_images(raw_image, [224, 224])
> 	with tf.Session() as session:
> 	  result = session.run(images)
> 	  plt.imshow(images.eval())
> 	  plt.show()
> 
> 此时发现图片失真，如果将resize时使用的`method`变为其他方法，或使用`cv2.resize`都没有问题。
> 解决办法：在resize前将图片转换为float
> 
> 	raw_image = tf.gfile.FastGFile("../xxx.jpg", 'rb').read()
> 	raw_image = tf.image.decode_jpeg(raw_image)
>     tf.image.convert_image_dtype(raw_image, dtype=tf.float32)
> 	images = tf.image.resize_images(raw_image, [224, 224])
> 	with tf.Session() as session:
> 	  result = session.run(images)
> 	  plt.imshow(images.eval())
> 	  plt.show()

## 4. 模型

MoXing将模型定义在model_fn方法中，并在mox.run时注册该方法。

基本方法：

	def model_fn(inputs, mode, **kwargs):
	  ...
	  return mox.ModelSpec(...)

    mox.run(..., model_fn=model_fn, ...)

输入参数：

输入参数：

- `inputs`: 对应`input_fn`返回值的输入数据。
- `mode`: 当前调用`model_fn`时的运行模式，需要用户在`model_fn`中做好判断使用相应的模型。`mox.ModeKeys`中的一个，[参考API](http://moxing.inhuawei.com/moxing.tensorflow.executor.html?highlight=modekeys#moxing.tensorflow.executor.ModeKeys)。如训练态(mox.ModeKeys.TRAIN)和验证态(mox.ModeKeys.EVAL)下的模型是不一样的（如BN层和Dropout层）。
- `**kwargs`: 扩展参数的预留位置。

返回值：

- `mox.ModelSpec`的实例，[API参考](http://moxing.inhuawei.com/moxing.tensorflow.executor.html?highlight=modelspec#moxing.tensorflow.executor.ModelSpec)

> #### 踩坑 4-0-1 (关键字：input_fn返回值，model_fn输入参数)
> 
> 当`input_fn`返回的输入数据只有一项时，`model_fn`的输入参数`inputs`仍然是一个`list`
> 
> 用户的代码可能是如下样例：
> 
> 	  def input_fn(mode, **kwargs):
> 	    ...
> 	    return image
> 	
> 	  def model_fn(inputs, mode, **kwargs):
> 	    images = inputs
> 	    ...
> 
> 代码看似没什么问题，但是当用户在`model_fn`中使用`images`时发现`images`的`shape`和预想的不太一样。可能会出现如下错误信息：
> 
> 	  ValueError: Input must be of size [batch_size, height, width, C>0]
> 
> 即使`input_fn`返回的输入数据只有image，`model_fn`的输入参数`inputs`仍然是一个`list`，为`[images]`，所以如下代码才是正确的用法：
> 
> 	  def input_fn(mode, **kwargs):
> 	    ...
> 	    return image
> 	
> 	  def model_fn(inputs, mode, **kwargs):
> 	    images = inputs[0]
> 	    ...

`model_fn`必须返回`ModelSpec`的实例，根据`model_fn`中的`mode`不同，`ModelSpec`的入参情况为：

- `loss`: 指定模型的损失值，一个0阶`tf.Tensor`，或者0阶`tf.Tensor`的`list`，多loss案例参考[生成对抗模型GAN](#成对抗模型GAN)，当`mode==mox.ModeKey.TRAIN`时必须提供。
- `var_scope`: 指定从`loss`中计算出的梯度需要对应的变量范围，只有在`var_scope`范围内的`tf.Variable`的梯度才会被计算和更新。如果`loss`是一个0阶`tf.Tensor`，则`var_scope`为`str`的`list`，指定一个或多个[variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope)。当`loss`是0阶`tf.Tensor`的`list`时，`var_scope`为二阶`list`，`list[i]`表示`loss[i]`的variable_scope，参考[生成对抗模型GAN](#成对抗模型GAN)
- `log_info`: 一个`dict`，运行作业时控制台需要打印的指标信息，仅支持0阶`tf.Tensor`，如`{'loss': loss, 'acc': accuracy}`，当`mode==mox.ModeKey.EVAL`时必须提供。
- `output_info`: 一个`dict`，运行作业的同时输出`tf.Tensor`中具体的值到`output_fn`中，当`mode==mox.ModeKey.PREDICT`时必须提供，参考[利用output_fn做预测](利用output_fn做预测)
- `export_spec`: 一个`dict`，导出PB模型时指定输入输出节点，必须是一个`mox.ExportSpec`的实例([参考API](http://moxing.inhuawei.com/moxing.tensorflow.executor.html?highlight=exportspec#moxing.tensorflow.executor.ExportSpec))，当`mode==mox.ModeKey.EXPORT`时必须提供(注意`mox.ModeKey.EXPORT`是无法在`mox.run`中显示指定的，仅当`mox.run`参数中`export_model`为有效值时会自动添加该模式)，参考[导出PB模型](导出PB模型)
- `hooks`: 一个`list`, 每个元素都必须是`mox.AggregativeSessionRunHook`子类的实例([参考API](http://moxing.inhuawei.com/moxing.tensorflow.executor.html?highlight=aggregativesessionrunhook#moxing.tensorflow.executor.AggregativeSessionRunHook))，会被`tf.Session()`执行的hook。参考[在model_fn中使用placeholder](在model_fn中使用placeholder)，[训练时打印验证集指标](训练时打印验证集指标)，[使用Early Stopping](使用Early Stopping)


### 4.1 使用MoXing模型库的内置模型

目前MoXing集成了一些[神经网络模型](http://moxing.inhuawei.com/moxing.tensorflow.nets.html?highlight=networkkeys#moxing.tensorflow.nets.NetworkKeys)，用户可以直接使用[mox.get_model_fn](http://moxing.inhuawei.com/moxing.tensorflow.nets.html?highlight=get_model_fn#moxing.tensorflow.nets.get_model_fn)获取这些模型。以及使用[mox.get_model_meta](http://moxing.inhuawei.com/moxing.tensorflow.nets.html?highlight=get_model_meta#moxing.tensorflow.nets.get_model_meta)获取这些模型的元信息。

例，训练一个ResNet_v1_50:

	import tensorflow as tf
	import moxing.tensorflow as mox
	
	slim = tf.contrib.slim
	
	
	def input_fn(mode, **kwargs):
	  meta = mox.ImageClassificationRawMetadata(base_dir='/export1/flowers/raw/split/train')
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
	    num_classes=1000,
	    weight_decay=0.0001)(images)
	  
	  loss = tf.losses.softmax_cross_entropy(
	    logits=logits, onehot_labels=slim.one_hot_encoding(labels, 1000))
	  
	  return mox.ModelSpec(loss=loss, log_info={'loss': loss})
	
	
	mox.run(input_fn=input_fn,
	        model_fn=model_fn,
	        optimizer_fn=mox.get_optimizer_fn('sgd', learning_rate=0.01),
	        batch_size=32,
	        run_mode=mox.ModeKeys.TRAIN,
	        max_number_of_steps=100)

> #### 踩坑 4-1-1 (关键字：导出模型，image_size)
> 
> 当用户导出模型时，考虑以下代码导出一个被TF-Serving使用的模型：
> 
> 	  import tensorflow as tf
> 	  import moxing.tensorflow as mox
> 	  
> 	  slim = tf.contrib.slim
> 	  
> 	  
> 	  def input_fn(mode, **kwargs):
> 	    meta = mox.ImageClassificationRawMetadata(base_dir='/export1/flowers/raw/split/train')
> 	    dataset = mox.ImageClassificationRawDataset(meta)
> 	    image, label = dataset.get(['image', 'label'])
> 	    image = mox.get_data_augmentation_fn(
> 	      name='resnet_v1_50',
> 	      run_mode=mode,
> 	      output_height=224,
> 	      output_width=224)(image)
> 	    return image, label
> 	  
> 	  
> 	  def model_fn(inputs, mode, **kwargs):
> 	    images, labels = inputs
> 	    logits, endpoints = mox.get_model_fn(
> 	      name='resnet_v1_50',
> 	      run_mode=mode,
> 	      num_classes=1000,
> 	      weight_decay=0.0001)(images)
> 	    
> 	    loss = tf.losses.softmax_cross_entropy(
> 	      logits=logits, onehot_labels=slim.one_hot_encoding(labels, 1000))
> 	    
> 	    return mox.ModelSpec(loss=loss, export_spec=mox.ExportSpec(
> 	      inputs_dict={'images': images}, outputs_dict={'logits': logits}))
> 	  
> 	  
> 	  mox.run(input_fn=input_fn,
> 	          model_fn=model_fn,
> 	          optimizer_fn=mox.get_optimizer_fn('sgd', learning_rate=0.01),
> 	          batch_size=32,
> 	          run_mode=mox.ModeKeys.TRAIN,
> 	          max_number_of_steps=1,
> 	          log_dir='/tmp/delete_me/0417_0',
> 	          export_model=mox.ExportKeys.TF_SERVING)
> 
> 可能会遇到如下错误信息：
> 
> 	  ValueError: `image_height` and `image_width` should be given to `mox.get_model_fn` when `run_mode` is `mox.ModeKeys.EXPORT (When `export_model` is specified in `mox.run`).
> 
> 当用户导出模型时，`model_fn`会以`mode=mox.ModeKeys.EXPORT`模式调用，当`mox.get_model_fn`中的`run_mode`为`mode=mox.ModeKeys.EXPORT`时，必须指定输入图像的尺寸。修改以下代码段：
> 
> 	  logits, endpoints = mox.get_model_fn(
> 	      name='resnet_v1_50',
> 	      run_mode=mode,
> 	      num_classes=1000,
> 	      weight_decay=0.0001)(images)
> 
> 正确的用法为：
> 
> 	  model_meta = mox.get_model_meta('resnet_v1_50')
> 	  logits, endpoints = mox.get_model_fn(
> 		  name='resnet_v1_50',
> 		  run_mode=mode,
> 		  num_classes=1000,
> 		  weight_decay=0.0001,
> 		  image_height=model_meta.default_image_size,
> 		  image_width=model_meta.default_image_size)(images)

除了使用MoXing内置的神经网络模型，用户可以自定义任何模型，只需要返回值符合规范。MoXing会自动将`model_fn`中定义的模型使用在多GPU上和分布式上。

> #### 踩坑 4-1-2 (关键字：get_collection, 正则项损失值)
> 
> 在`model_fn`中调用形如`tf.global_variables()`或`tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)`这些方法时，返回值与预期的不符。`tf.global_variables()`等效于`tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)`
> 
> 当使用单GPU时，这些方法的使用没有问题，但当使用多GPU时，使用`mox.get_collection`代替`tf.get_collection`来获取当前GPU上`model_fn`定义的Collection。
> 
> 以下为获取模型正则项损失值代码：
> 
> 	  def model_fn(inputs, mode, **kwargs):
> 	    ...
>       # 错误用法
> 	    # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
>       # 正确用法
>       reg_losses = mox.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
> 	    ...

### 4.2 生成对抗模型GAN

创建并训练一个DCGAN-MNIST模型，由[此开源代码](https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN/blob/master/tensorflow_MNIST_DCGAN.py)转换为MoXing实现方式。

	from tensorflow.examples.tutorials.mnist import input_data
	import tensorflow as tf
	import moxing.tensorflow as mox
	
	tf.flags.DEFINE_string('data_url', '/export1/zzy/mnist/zip/data', '')
	
	flags = tf.flags.FLAGS
	
	batch_size = 50
	
	def input_fn(run_mode, **kwargs):
	  mnist = input_data.read_data_sets(flags.data_url, one_hot=True)
	  def gen():
	    while True:
	      yield mnist.train.next_batch(batch_size)
	  ds = tf.data.Dataset.from_generator(
	      gen, output_types=(tf.float32, tf.int64),
	      output_shapes=(tf.TensorShape([None, 784]), tf.TensorShape([None, 10])))
	  images, labels = ds.make_one_shot_iterator().get_next()
	  images = tf.reshape(images, shape=[-1, 28, 28, 1])
	  images = tf.image.resize_images(images, [64, 64])
	  images = (images - 0.5) / 0.5
	  return images, labels
	
	
	def model_fn(inputs, run_mode, **kwargs):
	  images, labels = inputs
	  isTrain = (run_mode == mox.ModeKeys.TRAIN)
	
	  def lrelu(x, th=0.2):
	    return tf.maximum(th * x, x)
	
	  # G(z)
	  def generator(x, isTrain=True, reuse=False):
	    with tf.variable_scope('generator', reuse=reuse):
	      # 1st hidden layer
	      conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
	      lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)
	      
	      # 2nd hidden layer
	      conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
	      lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
	      
	      # 3rd hidden layer
	      conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
	      lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
	      
	      # 4th hidden layer
	      conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
	      lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
	      
	      # output layer
	      conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
	      o = tf.nn.tanh(conv5)
	      
	      return o
	  
	  # D(x)
	  def discriminator(x, isTrain=True, reuse=False):
	    with tf.variable_scope('discriminator', reuse=reuse):
	      # 1st hidden layer
	      conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
	      lrelu1 = lrelu(conv1, 0.2)
	      
	      # 2nd hidden layer
	      conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
	      lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
	      
	      # 3rd hidden layer
	      conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
	      lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
	      
	      # 4th hidden layer
	      conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
	      lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)
	      
	      # output layer
	      conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
	      o = tf.nn.sigmoid(conv5)
	      
	      return o, conv5
	
	  # networks : generator
	  z = tf.random_normal(shape=[batch_size, 1, 1, 100], mean=0.0, stddev=1.0)
	  G_z = generator(z, isTrain)
	
	  # networks : discriminator
	  D_real, D_real_logits = discriminator(images, isTrain)
	  D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)
	
	  # loss for each network
	  D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits,
	                                                                       labels=tf.ones(
	                                                                         [batch_size, 1, 1, 1])))
	  D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
	                                                                       labels=tf.zeros(
	                                                                         [batch_size, 1, 1, 1])))
	  D_loss = D_loss_real + D_loss_fake
	  G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
	                                                                  labels=tf.ones(
	                                                                    [batch_size, 1, 1, 1])))
	
	  tf.summary.image('G_z', G_z, max_outputs=10)
	
	  return mox.ModelSpec(loss=[D_loss, G_loss],
	                       var_scopes=[['discriminator'], ['generator']],
	                       log_info={'D_loss': D_loss, 'G_loss': G_loss})
	
	if __name__ == '__main__':
	  mox.run(input_fn=input_fn,
	          model_fn=model_fn,
	          optimizer_fn=[mox.get_optimizer_fn(name='adam', learning_rate=0.0002, beta1=0.5),
	                        mox.get_optimizer_fn(name='adam', learning_rate=0.0002, beta1=0.5)],
	          run_mode=mox.ModeKeys.TRAIN,
	          batch_size=batch_size,
	          auto_batch=False,
	          log_dir='/tmp/delete_me/dcgan_mnist',
	          max_number_of_steps=11000,
	          log_every_n_steps=9,
	          save_summary_steps=99,
	          save_model_secs=60)

### 4.3 利用output_fn做预测

在`model_fn`中的节点都是以`tf.Tensor`的形式构建在流图中，MoXing中可以提供`output_fn`用于获取并输出`model_fn`中的`tf.Tensor`的值。

`output_fn`的基本使用方法：

	def input_fn(mode, **kwargs):
	  ...
	
	def model_fn(inputs, mode, **kwargs):
	  ...
	  predictions = ...
	  ...
	  return mox.ModelSpec(..., output_info={'predictions': predictions}, ...)
	
	def output_fn(outputs, **kwargs):
	  print(outputs)
	
	mox.run(...
	        output_fn=output_fn,
	        output_every_n_steps=10,
	        ...)

其中，在`model_fn`中的`output_info`指定输出值对应的`tf.Tensor`，在`mox.run`中注册`output_fn`，当`output_every_n_steps`为10时，每经过10个step（注意在分布式运行中，这里的step指的是local_step），`output_fn`就会被调用一次，并且输入参数`outputs`为一个长度为10的`list`，每个元素为一个`dict: {'predictions': ndarray}`。在这里，`outputs`的值即为：

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

### 4.4 导出PB模型

MoXing在`mox.run`执行完毕后（训练完成或是验证完成），可以导出模型，基本用法为：

	def input_fn(mode, **kwargs):
	  ...
	
	def model_fn(inputs, mode, **kwargs):
	  ...
	  return mox.ModelSpec(..., 
	                       export_spec=mox.ExportSpec(inputs_dict={...}, outputs_dict={...}, ...), 
	                       ...) 
	
	mox.run(...,
	        export_model=mox.ExportKeys.XXX,
	        ...)

其中，[mox.ExportSpec](http://moxing.inhuawei.com/moxing.tensorflow.executor.html?highlight=mox%20exportspec#moxing.tensorflow.executor.ExportSpec)指定了导出模型的输入输出节点，仅能选取`model_fn`内部定义的`tf.Tensor`，[mox.ExportKeys](http://moxing.inhuawei.com/moxing.tensorflow.executor.html?highlight=exportkeys#moxing.tensorflow.executor.ExportKeys)指定了导出模型的类型。

案例，训练一个ResNet_v1_50模型，在训练结束后导出用于TF-Serving的PB模型：

	import tensorflow as tf
	import moxing.tensorflow as mox
	
	slim = tf.contrib.slim
	
	
	def input_fn(mode, **kwargs):
	  meta = mox.ImageClassificationRawMetadata(base_dir='/export1/flowers/raw/split/train')
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
	    num_classes=1000,
	    weight_decay=0.0001,
	    data_format='NCHW',
	    batch_norm_fused=True)(images)
	  
	  labels_one_hot = slim.one_hot_encoding(labels, 1000)
	  loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_one_hot)
	
	  export_spec = mox.ExportSpec(inputs_dict={'images': images}, outputs_dict={'logits': logits})
	  
	  return mox.ModelSpec(loss=loss, export_spec=export_spec)
	
	
	mox.run(input_fn=input_fn,
	        model_fn=model_fn,
	        optimizer_fn=mox.get_optimizer_fn('sgd', learning_rate=0.01),
	        batch_size=32,
	        run_mode=mox.ModeKeys.TRAIN,
	        max_number_of_steps=100,
	        log_dir='/tmp/delete_me/log_0409',
	        export_model=mox.ExportKeys.TF_SERVING)

当训练完成后，`model_fn`将以`mode=mox.ModeKeys.EXPORT`被调用（用户构建导出模型的流图），在此次调用过程中：

1) 当`auto_batch`为`False`时，`inputs`的shape和训练时保持一致，即`images.shape=[32, 224, 224, 3]`, `labels.shape=[32]`。当`auto_batch`为`True`时，`inputs`中`batch_size`的维度会被置为`None`，即`images.shape=[None, 224, 224, 3]`, `labels.shape=[None]`，所以就会导致输出节点`logits`的`batch_size`维度也为`None`，即`logits.shape=[None, 1000]`。

2) 导出的模型中计算节点的`device`信息为空。

DLS服务中`预测作业`使用的即是`mox.ExportKeys.TF_SERVING`类型的PB模型。

> #### 踩坑 4-4-1 (关键字：预测作业，找不到模型)
> 
> 启动`预测作业`，如果提示信息类似如下：
> 
> 	  tensorflow_serving/sources/storage_path/file_system_storage_path_source.cc:268] No versions of servable resnet_v1_50 found under base path s3://dls-test/log/resnet_v1_50/1/
> 
> 说明没有找到可以用于TF-Serving的模型文件。导出的模型应有如下目录结构
> 
> 	  |- log_dir
> 		  |- 1
> 			  |- svaed_model.pb
> 			  |- variables
> 				  |- variables.data-00000-of-00001
> 				  |- variables.index
> 
> 其中`1`表示模型的版本号，启动预测服务需要指定到目录`log_dir`这层，在这个案例中就是`s3://dls-test/log/resnet_v1_50`而不是`s3://dls-test/log/resnet_v1_50/1/`

当导出模型的目录下有多个版本号的模型时，如`1`，`2`，`99`，TF-Serving会自动选取数字最大`99`的模型做预测，当一个作业往该目录下继续输出了模型`100`，TF-Serving预测服务不需要重启，自动切换到`100`的模型上。在MoXing中，`mox.ExportSpec(..., version=x, ...)`，`version`参数就是用来指定该版本号，缺省值为`-1`，表示自动自增，即在输出目录下找到最大的版本号并+1，然后保存。

> #### 踩坑 4-4-2 (关键字：MaxPoolingOp，NHWC)
> 
> 错误信息如下：
> 
> 	  InvalidArgumentError (see above for traceback): Default MaxPoolingOp only supports NHWC.
> 
> 这个错误可能在训练作业、预测作业中遇到。原因是当使用CPU时，模型中的某些算子不支持`NHWC`数据格式。([Data Formats参考](https://www.tensorflow.org/performance/performance_guide#Data%20formats)）。可能的情况如下：
> 
> 1）DLS服务中，使用预置模型库训练模型（使用GPU训练），运行参数有`data_format=NCHW`，训练完成后使用导出的模型启动预测作业（由于目前预测作业仅支持CPU）。预测作业中出现该错误。
> 
> 2）DLS服务中，使用预置模型库训练模型（使用CPU训练），并且数据格式为`NCHW`（即运行参数`data_format=NCHW`。
> 
> 3）本地MoXing开发，模型中有不支持`NCHW`的算子，并且使用CPU训练。

;

> #### 踩坑 4-4-3 (关键字：路径已存在，export directory)
> 
> 错误信息显示如下：
> 
> 	  AssertionError: Export directory already exists. Please specify a different export directory: s3://bucket_name/log/1
> 
> 导出模型时如果在输出日志路径（`train_url`或是`log_dir`）中存在一个`1`的目录，并且还指定了`version=1`，则会出现该错误。指定一个不存在的版本号或者将版本号设置为自增(即`version=-1`)

在`model_fn`中，如果需要新建变量，建议使用`tf.get_variable`而不是`tf.Variable`。

> #### 踩坑 4-4-4 (关键字：tf.Variable，AssertionError)
> 
> 当`model_fn`中的变量使用了`tf.Variable`来创建，并且损失值loss的计算中使用到了该变量，可能会出现如下错误信息：
> 
> 	  v.name in list_allowed_variable_names_with_port())
> 	  AssertionError
> 
> 这是因为`tf.Variable`创建的变量无法被MoXing管理，替换为`tf.get_variable`即可解决。
> 
> 另外，有一些隐藏调用`tf.Variable`的地方，如`tf.train.AdamOptimizer`中创建变量时使用了`tf.Variable`（仅针对TensorFlow-1.4及以下版本，TensorFlow-1.5及以上版本官方已修复），所以如果使用`tf.train.AdamOptimizer`遇到了类似的问题，MoXing提供了等价的API： `mox.get_optimizer_fn('adam', ...)()`

### 4.5 Hook的使用

MoXing提供了允许在[tf.train.MoniteredSession](https://www.tensorflow.org/api_docs/python/tf/train/MonitoredSession)中注册hooks的方法，hooks要求为继承于[tf.train.SessionRunHook](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook)的子类。MoXing中由于兼容了多GPU和分布式，因此要求用户注册的hooks为[mox.AggregativeSessionRunHook](http://moxing.inhuawei.com/moxing.tensorflow.executor.html?highlight=aggregativesessionrunhook#moxing.tensorflow.executor.AggregativeSessionRunHook)的子类。`AggregativeSessionRunHook`继承于`SessionRunHook`，用户可以添加由`SessionRunHook`定义的回调函数`begin`, `after_create_session`, `before_run`, `after_run`, `end`。另外，用户还必须额外实现三个返回布尔值方法，`support_aggregation`，`support_sync_workers`，`run_inter_mode`，基本用法如下：

	import tensorflow as tf
	import moxing.tensorflow as mox
	
	class MyHook(mox.AggregativeSessionRunHook):
	  def __init__(self, ...):
	    ...
	
	  def support_aggregation(self):
	    return ...
	
	  def support_sync_workers(self):
	    return ...
	
	  def run_inter_mode(self):
	    return ...
	  
	  ...
	
	def input_fn(mode, **kwargs):
	  ...
	
	def model_fn(inputs, mode, **kwargs):
	  ...
	  hook = MyHook(...)
	  return mox.ModelSpec(..., hooks=[feed_hook], ...)
	
	mox.run(...)

#### 4.5.1 在model_fn中使用placeholder

用户可以在`model_fn`中利用hook创建并填充placeholder，MoXing提供了最基本的实现类`FeedSessionRunHook`，样例代码如下:

	import numpy as np
	import tensorflow as tf
	import moxing.tensorflow as mox
	from moxing.tensorflow.executor.hooks import FeedSessionRunHook
	slim = tf.contrib.slim
	
	
	def input_fn(run_mode, **kwargs):
	  return [tf.constant(0.0)]
	
	
	def model_fn(inputs, mode, **kwargs):
	  del inputs
	  images = tf.placeholder(dtype=tf.float32, shape=[16, 16, 16, 3])
	  labels = tf.placeholder(dtype=tf.int64, shape=[16])
	  net = slim.flatten(images)
	  logits = slim.fully_connected(net, 10)
	  labels_one_hot = slim.one_hot_encoding(labels, 10)
	  loss = tf.losses.softmax_cross_entropy(
	    logits=logits, onehot_labels=labels_one_hot,
	    label_smoothing=0.0, weights=1.0)
	  
	  feed_images = np.random.random(size=[16, 16, 16, 3])
	  feed_labels = np.random.random_integers(low=0, high=10, size=[16])
	  
	  feed_hook = FeedSessionRunHook(feed_dict={images: feed_images, labels: feed_labels})
	  
	  return mox.ModelSpec(loss=loss, log_info={'loss': loss}, hooks=[feed_hook])
	
	
	mox.run(input_fn=input_fn,
	        model_fn=model_fn,
	        optimizer_fn=mox.get_optimizer_fn('sgd', learning_rate=0.001),
	        run_mode=mox.ModeKeys.TRAIN,
	        auto_batch=False,
	        batch_size=16,
	        log_dir=None,
	        max_number_of_steps=100,
	        log_every_n_steps=10)

`FeedSessionRunHook`的源码非常简单，如下：

	class FeedSessionRunHook(mox.AggregativeSessionRunHook):
	  def __init__(self, feed_dict):
	    super(FeedSessionRunHook, self).__init__()
	    self.feed_dict = feed_dict
	
	  def before_run(self, run_context):
	    if self.feed_dict:
	      run_args = tf.train.SessionRunArgs(fetches=None, feed_dict=self.feed_dict)
	      return run_args
	
	  def support_aggregation(self):
	    return False
	
	  def support_sync_workers(self):
	    return False
	
	  def run_inter_mode(self):
	    return False

#### 4.5.2 训练时打印验证集指标

在启动一个训练作业时，通常在训练时要不断观察模型在验证数据集上的各项指标。训练和验证在输入和模型上都不相同，所以至少要构建2个数据流图，分别为训练时的流图和验证时的流图。这就是`inter_mode`的作用，`inter_mode`允许在`run_mode`以外额外创建一个`中间模式`并在`run_mode`运行时穿插运行。基本用法：

	def input_fn(mode, **kwargs):
	  ...
	
	def model_fn(inputs, mode, **kwargs):
	  ...
	
	mox.run(...,
	        run_mode=mox.ModeKeys.TRAIN,
	        inter_mode=mox.ModeKeys.EVAL,
	        ...)

其中`input_fn`和`model_fn`都会以`mox.ModeKeys.TRAIN`和`inter_mode=mox.ModeKeys.EVAL`这两个模式被调用。

样例，训练一个ResNet_v1_50，使用[mox.LogEvaluationMetricHook](http://moxing.inhuawei.com/moxing.tensorflow.executor.html?highlight=logevaluationmetrichook#moxing.tensorflow.executor.LogEvaluationMetricHook)，每隔一定训练步数在验证数据集上打印`loss`和`accuracy`：

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
	  
	  metric_hook = mox.LogEvaluationMetricHook(
	    monitor_info={'loss': loss, 'accuracy': accuracy},
	    batch_size=32,
	    samples_in_train=1000,
	    samples_in_eval=100,
	    evaluate_every_n_epochs=1,
	    prefix='[VALIDATION METRICS]')
	  
	  return mox.ModelSpec(loss=loss,
	                       log_info={'loss': loss, 'accuracy': accuracy},
	                       hooks=[metric_hook])
	
	
	mox.run(input_fn=input_fn,
	        model_fn=model_fn,
	        optimizer_fn=mox.get_optimizer_fn('sgd', learning_rate=0.01),
	        batch_size=32,
	        run_mode=mox.ModeKeys.TRAIN,
	        inter_mode=mox.ModeKeys.EVAL,
	        max_number_of_steps=100)

控制台输出日志可能会如下：
	
	INFO:tensorflow:step: 0(global step: 0)	sample/sec: 12.271	loss: 8.273	accuracy: 0.000
	INFO:tensorflow:step: 10(global step: 10)	sample/sec: 42.184	loss: 3.977	accuracy: 0.188
	INFO:tensorflow:step: 20(global step: 20)	sample/sec: 42.211	loss: 2.395	accuracy: 0.156
	INFO:tensorflow:step: 30(global step: 30)	sample/sec: 42.284	loss: 2.063	accuracy: 0.250
	INFO:tensorflow:[VALIDATION METRICS] step: 31 loss: 17737.227 accuracy: 0.000
	INFO:tensorflow:step: 40(global step: 40)	sample/sec: 42.088	loss: 2.797	accuracy: 0.312
	INFO:tensorflow:step: 50(global step: 50)	sample/sec: 42.175	loss: 2.335	accuracy: 0.156
	INFO:tensorflow:step: 60(global step: 60)	sample/sec: 41.986	loss: 4.093	accuracy: 0.156
	INFO:tensorflow:[VALIDATION METRICS] step: 63 loss: 99017.656 accuracy: 0.000
	INFO:tensorflow:step: 70(global step: 70)	sample/sec: 41.681	loss: 2.391	accuracy: 0.375
	INFO:tensorflow:step: 80(global step: 80)	sample/sec: 41.361	loss: 1.550	accuracy: 0.531
	INFO:tensorflow:step: 90(global step: 90)	sample/sec: 41.693	loss: 1.992	accuracy: 0.438
	INFO:tensorflow:[VALIDATION METRICS] step: 95 loss: 9779.766 accuracy: 0.000

#### 4.5.3 使用Early Stopping

在Keras-API中提供了[tf.keras.callbacks.EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)的功能，MoXing中也用同样的API，用法和Keras的相似，为[mox.EarlyStoppingHook](http://moxing.inhuawei.com/moxing.tensorflow.executor.html?highlight=earlystopping#moxing.tensorflow.executor.EarlyStoppingHook)

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

除了EarlyStopping，MoXing还提供了当检测到Plateau时自动下降学习率，当检测到多次Plateau并且评价指标没有上升或下降是，则停止训练，参考API：[mox.PlateauLREarlyStoppingHook](http://moxing.inhuawei.com/moxing.tensorflow.executor.html?highlight=plateaulrearlystoppinghook#moxing.tensorflow.executor.PlateauLREarlyStoppingHook)

### 4.6 利用Keras构建模型

MoXing本身除了支持TensorFlow和TensorFlow-slim的API来构建模型以外，还可以使用Keras-API来构建模型。根据Keras官方教程中的一个案例[Multi-input and multi-output models](https://keras.io/getting-started/functional-api-guide/)，将其迁移到MoXing框架中，代码如下：

	import tensorflow as tf
	import moxing.tensorflow as mox
	
	from tensorflow.python.keras.layers import Embedding, LSTM, Dense
	from tensorflow.python.keras.layers import concatenate
	from tensorflow.python.keras.losses import binary_crossentropy
	from tensorflow.python.keras.models import Model
	from tensorflow.python.keras.layers import Input
	
	
	def input_fn(mode, **kwargs):
	  main_input = tf.random_uniform(shape=(100,), minval=1, maxval=10000, dtype=tf.int32, name='main_input')
	  auxiliary_input = tf.random_normal(shape=(5,), name='aux_input')
	  main_labels = tf.random_uniform(shape=(1,))
	  auxiliary_labels = tf.random_uniform(shape=(1,))
	  return main_input, auxiliary_input, main_labels, auxiliary_labels
	
	
	def model_core(main_input, auxiliary_input):
	  x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
	  lstm_out = LSTM(32)(x)
	  auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
	
	  x = concatenate([lstm_out, auxiliary_input])
	  x = Dense(64, activation='relu')(x)
	  x = Dense(64, activation='relu')(x)
	  x = Dense(64, activation='relu')(x)
	  main_output = Dense(1, activation='sigmoid', name='main_output')(x)
	
	  return main_output, auxiliary_output
	
	
	def model_fn(inputs, mode, **kwargs):
	  main_input, auxiliary_input, main_labels, auxiliary_labels = inputs
	  main_output, auxiliary_output = model_core(main_input, auxiliary_input)
	  loss = 1.0 * binary_crossentropy(main_output, main_labels) + \
	         0.2 * binary_crossentropy(auxiliary_output, auxiliary_labels)
	  loss = tf.reduce_mean(loss)
	  return mox.ModelSpec(loss=loss, log_info={'loss': loss})
	
	
	def save_keras_model(save_path):
	  keras_main_input = Input(shape=(100, ))
	  keras_auxiliary_input = Input(shape=(5, ))
	  keras_main_output, keras_auxiliary_output = model_core(keras_main_input, keras_auxiliary_input)
	  keras_model = Model(inputs=[keras_main_input, keras_auxiliary_input],
	                      outputs=[keras_main_output, keras_auxiliary_output])
	  keras_model_json = keras_model.to_json()
	  with tf.gfile.Open(save_path, 'wb') as f:
	    f.write(keras_model_json)
	  
	if __name__ == '__main__':
	  mox.run(input_fn=input_fn,
	          model_fn=model_fn,
	          optimizer_fn=mox.get_optimizer_fn('rmsprop', learning_rate=0.01),
	          run_mode=mox.ModeKeys.TRAIN,
	          batch_size=32,
	          auto_batch=True,
	          log_dir=None,
	          max_number_of_steps=50,
	          log_every_n_steps=10)
	  save_keras_model(save_path='/tmp/delete_me/keras_model.json')

当运行完成后，将模型以`json`的形式保存（不包含模型参数值，仅保存数据流图），利用以下代码可以载入该模型并训练（仅载入数据流图，载入模型参数值需要使用`checkpoint_path`）

	import tensorflow as tf
	import moxing.tensorflow as mox
	
	from tensorflow.python.keras.losses import binary_crossentropy
	from tensorflow.python.keras.models import model_from_json
	
	
	def input_fn(mode, **kwargs):
	  main_input = tf.random_uniform(shape=(100,), minval=1, maxval=10000, dtype=tf.int32, name='main_input')
	  auxiliary_input = tf.random_normal(shape=(5,), name='aux_input')
	  main_labels = tf.random_uniform(shape=(1,))
	  auxiliary_labels = tf.random_uniform(shape=(1,))
	  return main_input, auxiliary_input, main_labels, auxiliary_labels
	
	
	def model_fn(inputs, mode, **kwargs):
	  main_input, auxiliary_input, main_labels, auxiliary_labels = inputs
	
	  with tf.gfile.Open('/tmp/delete_me/keras_model.json', 'r') as f:
	    keras_model_json = f.read()
	    
	  model_croe = model_from_json(keras_model_json)
	  main_output, auxiliary_output = model_croe([main_input, auxiliary_input])
	
	  loss = 1.0 * binary_crossentropy(main_output, main_labels) + \
	         0.2 * binary_crossentropy(auxiliary_output, auxiliary_labels)
	  loss = tf.reduce_mean(loss)
	
	  return mox.ModelSpec(loss=loss, log_info={'loss': loss})
	
	if __name__ == '__main__':
	  mox.run(input_fn=input_fn,
	          model_fn=model_fn,
	          optimizer_fn=mox.get_optimizer_fn('rmsprop', learning_rate=0.01),
	          run_mode=mox.ModeKeys.TRAIN,
	          batch_size=32,
	          auto_batch=True,
	          log_dir=None,
	          max_number_of_steps=1000,
	          log_every_n_steps=10)

### 4.7 将字符串传入model_fn

当MoXing运行在GPU环境中时，`input_fn`在CPU上构建，而`model_fn`在GPU上构建，如果`input_fn`中返回值中有`tf.string`类型的Tensor，由于字符串是无法传入到GPU上的，所以`model_fn`中会因为无法接受字符串而报错。但是在某些情况中，如图像分类，可能需要将图像对应的文件名传入到`model_fn`中，并且作为`output_info`进行输出，此时就需要将字符串进行编码，通过把字符串转换成int数组，传入到`model_fn`中，然后在`output_fn`中再转换回来。具体代码如下：



	def bytes_to_sequence(bts):
	  return np.array([ord(b) for b in bts])
	
	def sequence_to_bytes(seq):
	  return ''.join([chr(i) for i in seq if i > 0])
	
	def input_fn(mode):
	  ...
	  image, label, image_name = dataset.get(['image', 'label', 'image_name'])
      image_name_seq = tf.py_func(bytes_to_sequence, inp=[image_name], Tout=tf.int64)
      image_name_seq.set_shape([None])
	  reutrn image, label, image_name_seq

    def model_fn(inputs, mode):
      images, labels, image_name_seqs = inputs
      ...
      return mox.ModelSpec(...
		                   output_info={'image_name_seqs': image_name_seqs,
		                                'logits': logits},
		                   ...)

    def output_fn(outputs):
	  batch_seqs = outputs[0]['image_name_seqs']
	  batch_logits = outputs[0]['logits']
	  for seq, pd in zip(batch_seqs, batch_logits):
	    print('%s: %s' % (sequence_to_bytes(seq), pd))

	mox.run(input_fn=input_fn,
		    model_fn=model_fn,
		    output_fn=output_fn,
		    output_every_n_steps=1,
		    ...)

### 4.8 混合精度训练

混合精度即为fp32和fp16的混合，神经网络计算时使用fp16(除了BN层以外)，参数存储使用fp32，[论文链接](https://arxiv.org/pdf/1710.03740.pdf)。在Tesla-P100或V100上使用混合精度训练，可以在不损失精度的前提下，提高训练的速度。在TensorFlow中，算子或网络（如resnet）的输入如果是fp16的，那么该算子或网络自动会使用fp16进行计算，MoXing也继承了该特性。另外，在MoXing中，使用API`mox.var_scope`来强制指定参数的类型，达到fp16+fp32混合精度的效果。

样例代码：

	images = tf.random_normal(shape=[1, 224, 224, 3], dtype=tf.float32)
	# Cast the input of neural network to fp16
	images = tf.cast(images, tf.float16)
	# fp16 model only supports with `batch_norm_fused=True`
	resnet50 = mox.get_model_fn(name=mox.NetworkKeys.RESNET_V1_50, run_mode=mode, 
	                            num_classes=1000, data_format='NCHW', batch_norm_fused=True)
	# fp32 store and fp16 computation
	with mox.var_scope(force_dtype=tf.float32):
	  logits, _ = resnet50(images)
	
	# logits should have dtype of tf.float16 but all variables have dtype of tf.float32

代码中输入模型的是fp16类型的，网络会自动启动fp16计算，但是会导致创建的变量也成为fp16的，就需要加上`mox.var_scope`的作用域，在该作用域下创建的variable，返回的类型依然是用户定义的dtype，但是在正真创建变量的时候，会创建一个`force_dtype`指定的类型，然后cast到用户指定的dtype返回给用户，所以用户创建变量时的返回值将是一个`tf.Tensor`而不是`tf.Variable`

考虑以下代码：

	import tensorflow as tf
	import moxing.tensorflow as mox
	
	with mox.var_scope(force_dtype=tf.float32):
	  a = tf.get_variable('a', shape=[], dtype=tf.float16)

    print(a)
    print(tf.global_variables()【0】)

输出将得到：

	Tensor("Cast:0", shape=(), dtype=float16)
	<tf.Variable 'a:0' shape=() dtype=float32_ref>

`tf.get_variable`本来会返回一个`tf.Variable`，但是由于在`mox.var_scope`作用域下，这里会返回一个`tf.Tensor`，并且类型依然是`tf.float16`，但是在`tf.global_variables()`中将会对应有一个`tf.float32`的`tf.Variable`


## 5. 优化器

用户可以使用[mox.get_optimizer_fn](http://moxing.inhuawei.com/moxing.tensorflow.optimizer.html?highlight=get_optimizer_fn#moxing.tensorflow.optimizer.get_optimizer_fn)来获取MoXing内置的Optimizer，也可以使用TensorFlow定义或由用户自己实现的`Optimizer`。此外，MoXing还提供了OptimizerWrapper的用法。[所有支持的Optimizer列表](http://moxing.inhuawei.com/moxing.tensorflow.optimizer.html?highlight=optimizerkeys#moxing.tensorflow.optimizer.OptimizerKeys)。

### 5.1 基础Optimizer

使用内置OPT：

	mox.run(...,
	        optimizer_fn=mox.get_optimizer_fn('momentum', learning_rate=0.01, momentum=0.9),
	        ...)

使用TF定义的OPT：

	mox.run(...,
	        optimizer_fn=lambda: tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9),
	        ...)

使用自定义的OPT：

	def optimizer_fn():
	  ...
	  return my_optimizer()
	
	mox.run(...,
	        optimizer_fn=optimizer_fn,
	        ...)

> #### 踩坑 5-1-1 (关键字：Optimizer，callable)
> 
> `mox.run`中`optimizer_fn`需要传入的是一个返回optimizer的函数，而不是一个optimizer，以下代码的使用方式是错误的：
> 
> 	  mox.run(...,
> 	          optimizer_fn=tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9),
> 	          ...)
> 
> 此时可能会出现如下错误信息：
> 
> 	  TypeError: 'MomentumOptimizer' object is not callable
> 
> 只需要在optimizer上加上`lambda表达式`就能正确
> 
> 	  mox.run(...,
> 	          optimizer_fn=lambda: tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9),
> 	          ...)

### 5.2 封装器OptimizerWrapper

使用[mox.get_optimizer_wrapper_fn](http://moxing.inhuawei.com/moxing.tensorflow.optimizer.html?highlight=get_optimizer_wrapper_fn#moxing.tensorflow.optimizer.get_optimizer_wrapper_fn)可以获取Optimizer的高级应用方法。`OptimizerWrapper`是对optimizer的一层封装，类似[tf.train.SyncReplicasOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer)的用法。并且在允许的范围内，可以使用多层封装。样例代码如下。

使用Batch Gradient Descent，基础OPT为Momentum，每经过8个step的周期提交一次累计梯度。

	def optimizer_fn():
	  opt = mox.get_optimizer_fn('momentum', learning_rate=lr, momentum=0.9)()
	  opt = mox.get_optimizer_wrapper_fn('batch_gradients', opt, num_batches=8, sync_on_apply=True)()
	  return opt
    
    mox.run(..., optimizer_fn=optimizer_fn, ...)

> #### 踩坑 5-2-1 (关键字：OptimizerWrapper，同步异步)
> 
> 当遇到输出信息如下：
> 
> 	  WARNING:tensorflow:Using OptimizerWrapper when sync_replicas is True may cause performance loss.
> 
> 这并不是一个错误，大多数`OptimizerWrapper`都要求在异步模式下使用，如Batch Gradient Descent当没有到通信周期时，分布式的每个worker是异步的，而到了通信周期时，是通过Optimizer本身的`sync_on_apply=True`参数来做同步，所以需要设置运行参数`--sync_replicas=False`来启动一个异步分布式运行，才能发挥Batch Gradient Descent的性能优势。另外类似EASGD这类Optimizer本身就要求在异步模型下运行。

复现bact_size=32k训练ResNet-50，当节点数量不够时，可以通过Batch Gradient Descent等效增加每个节点的batch_size，并且使用[LARS](https://arxiv.org/abs/1708.03888)训练，此时将涉及3层Optimizer的封装：

	def optimizer_fn():
	  lr = config_lr(...)
	  opt = mox.get_optimizer_fn('momentum', learning_rate=lr, momentum=0.9)()
	  opt = mox.get_optimizer_wrapper_fn('lars', opt, ratio=0.001, weight_decay=0.0001)()
	  opt = mox.get_optimizer_wrapper_fn('batch_gradients', opt, num_batches=8, sync_on_apply=True)()

注意：

- 当run_mode为mox.ModeKeys.TRAIN时，optimizer_fn必须填充
- 当run_mode为mox.ModeKeys.EVAL时，optimizer_fn不需要填充

## 6. 运行

MoXing中运行仅需执行一个API，即[mox.run](http://moxing.inhuawei.com/moxing.tensorflow.executor.html?highlight=run#moxing.tensorflow.executor.run)。`mox.run`中`log_dir`主要用来输出TensorBoard的Summary文件和checkpoint文件，`checkpoint_path`用来指定载入checkpoint的路径。`mox.run`对checkpoint文件的载入优先级如下：

- 当`log_dir`中存在checkpoint时，无视`checkpoint_path`，从`log_dir`中载入checkpoint。如果当前模式为mox.ModeKeys.TRAIN，则将新的checkpoint保存在`log_dir`中。
- 当`log_dir`中不存在checkpoint时，从`checkpoint_path`中载入checkpoint。如果当前当前模式为mox.ModeKeys.TRAIN，则将新的checkpoint保存在`log_dir`中。
- 当`log_dir`和`checkpoint_path`中都不存在checkpoint时，如果当前模式为mox.ModeKeys.TRAIN，则初始化所有变量并将新的checkpoint保存在`log_dir`中。如果当前不是mox.ModeKeys.TRAIN，则抛出异常（非训练模式下必须提供checkpoint)

> #### 踩坑 6-0-1 (关键字：运行很快结束，没有训练)
> 
> 启动一个训练作业时，发现很快就结束了，控制台也没有打印任何与loss或是accuracy相关的信息。
> 
> 输出日志信息可能如下：
> 
> 	  INFO:tensorflow:Restoring parameters from s3://bucket_name/log/model.ckpt-xxx
> 	  INFO:tensorflow:Saving checkpoints for xxx into s3://bucket_name/log
> 
> 这是因为训练开始时，在用户指定的输出日志路径（`train_url`或是`log_dir`）中已经存在了checkpoint，例如一个训练到1000步的checkpoint文件：model.ckpt-1000，而用户启动的训练作业指定的训练步数也是1000步，此时就会认为不需要再训练，就直接退出了。
> 
> 如果想在原有checkpoint基础上继续训练，可以将训练步数指定到更大的步数。如果想重新训练，可以将原来的checkpoint文件删除或者指定一个全新的输出日志路径。

-

> #### 踩坑 6-0-2 (关键字：evaluation, validation, 精度浮动)
> 
> 启动一个验证作业，即mode=mox.ModeKeys.EVAL时（假设此时batch_size 1, num_gpus 1)，发现每次运行得到的精度不同。
> 
> 首先保证`input_fn`中数据集的`shuffle`参数为`False`，然后需要增加参数`--log_first_clone=False`, 当该参数为`True`（缺省）时，在控制台仅输出来自第一个GPU的`log_info`(为了节约时间，提高效率), 所以在验证时（多GPU），需要打印所有GPU上的综合值时，就要将该参数设置为`False`

