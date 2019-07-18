# efficient_ai模型压缩工具库使用说明

本文介绍efficient_ai通用功能的使用，不涉及具体压缩策略的细节。第二章到第四章分别介绍量化、剪枝、蒸馏三种压缩的使用方式。



## 目录

- [efficient_ai使用压缩工具库的前提](#1efficient_ai使用压缩工具库的前提)
- [efficient_ai通用功能使用介绍](#2efficient_ai通用功能使用介绍)
- [量化配置参数介绍和样例](#3量化配置参数介绍和样例)
- [剪枝配置参数介绍和样例](#4剪枝配置参数介绍和样例)
- [蒸馏配置参数介绍和样例](#5蒸馏配置参数介绍和样例)



## 1.efficient_ai使用压缩工具库的前提


**版本要求**： tensorflow >= 1.8

**使用剪枝和蒸馏**：moxing

## 2.efficient_ai通用功能使用介绍

### 2.1可以用于输入的模型

efficient_ai现在可以接受三种输入，分别是tensorflow的frozen\_graph,saved\_model和moxing中用来定义模型的model\_fn

TFServingModel是用来引入saved_model模型

- model_dir:模型文件的dir位置
- input_saved_model_tags:选择serving model的saved_model_tags。默认为第一个
- signature_def_key: 选择serving model的signature_def_key，默认为第一个

示例如下:</br>
```
from efficient_ai.models.tf_serving_model import TFServingModel
model=TFServingModel(model_dir=”/opt/model”,input_saved_model_tags=None,signature_def_key=None)
```

TFFrozenGraphModel用于导入freeze_graph模型

- file_name：模型文件的路径
- outputs：模型的输出节点名列表，可以包含多个输出
- inputs：模型的输入节点名列表，可以包含多个输入

示例如下:</br>
```
from efficient_ai.models. tf_frozen_graph_model import TFFrozenGraphModel
model= TFFrozenGraphModel( (file_name=”/opt/model/model.pb”, outputs =[“logits”], inputs =[“placeholder”])
```

MoxingModel用于导入Moxing框架定义的模型

- model_fn：Moxing里面定义结构用的函数，网络结构的定义详细参见[model_fn](#23-用于描述moxingmodel结构的model_fn)
- ckpt_dir：用来定义ckpt文件所在的文件夹

示例如下:</br>
```
from efficient_ai.models.moxing_model import MoxingModel
model = MoxingModel(model_fn, log_dir)
```

### 2.2 用于进行压缩的数据
input_fn 是用来表示输入进行压缩的数据，将返回两个tensor，第一个tensor表示input，第二个表示label
示例如下:</br>

```
def input_fn(run_mode, **kwargs):
	num_epochs = 5
	num_batches = num_epochs * mnist.train.num_examples // batch_size
	def gen():
		for _ in range(num_batches):
			 yield mnist.test.next_batch(batch_size)
	ds = tf.data.Dataset.from_generator(gen,output_types=(tf.float32, tf.int64),
                                      output_shapes=(tf.TensorShape([None, 784]), tf.TensorShape([None,10])))
	x, y_ = ds.make_one_shot_iterator().get_next()
	return x, y_
```

### 2.3 用于描述MoxingModel结构的model_fn

model_fn与Moxing框架中定义model_fn相似

```
def model_fn(inputs, run_mode, **kwargs):
    x, y_ = inputs
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # 转换输入数据shape,以便于用于网络中
    W_conv1 = weight_variable('w1', [5, 5, 1, 32])
    b_conv1 = bias_variable('b1', [32])
    # dcp export only support BiasAdd
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, W_conv1), b_conv1))  # 第一个卷积层
    h_pool1 = max_pool(h_conv1)  # 第一个池化层
    W_conv2 = weight_variable('w2', [5, 5, 32, 64])
    b_conv2 = bias_variable('b2', [64])
    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_pool1, W_conv2), b_conv2))  # 第二个卷积层
    h_pool2 = max_pool(h_conv2)  # 第二个池化层
    W_fc1 = weight_variable('w3', [7 * 7 * 64, 1024])
    b_fc1 = bias_variable('b3', [1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape成向量
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 第一个全连接层
    W_fc2 = weight_variable('w4', [1024, 10])
    b_fc2 = bias_variable('b4', [10])
    y = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)  # softmax层
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    from efficient_ai.config import CompressorSpec
    compressor_spec = CompressorSpec(logits=y)
    return mox.ModelSpec(loss=cross_entropy, compressor_spec=compressor_spec, log_info={'loss': cross_entropy})
```
与一般的model_fn略有不同之处在于需要将模型的输出设置到CompressorSpec的logits中


### 2.4 产生用于压缩的工具包
Compressor 是用来执行具体的量化、剪枝、蒸馏的算法

- model：可以是上面的TFServingModel，TFFrozenGraphModel或者MoxingModel
- input_fn：输入的方法

```
    from efficient_ai.compresss_tool import Conmpressor
    ct = Conmpressor (model,input_fn)
```
产生的Conmpressor包含量化、剪枝、蒸馏等方法

Compressor 中的distilling方法用来对模型进行蒸馏，因为蒸馏方法只对可训练模型有效，因此这里只支持对MoxingModel进行蒸馏

- teacher_models：teacher model的列表
- config：蒸馏算法所需要的配置，config包含和具体蒸馏算法相关的一些配置，类型为DistillCompressorConfig
- 返回值:Conmpressor

示例如下:</br>
```
    from efficient_ai.compressor import Compressor
    from efficient_ai.models.moxing_model import MoxingModel
    teacher_model = MoxingModel(model_fn, log_dir)
    student_model = MoxingModel(student_model_fn)
    c = Conmpressor (student_model,input_fn)
    c.distilling(teacher_model, config)
```

Compressor 中的pruning方法用来对模型进行剪枝，因为剪枝方法只对可训练模型有效，因此这里只支持对MoxingModel进行剪枝

- config：剪枝算法所需要的配置，config包含和具体剪枝算法相关的一些配置，类型为DCPCompressorConfig
- 返回值:Conmpressor

示例如下:</br>
```
    from efficient_ai.compressor import Compressor
    from efficient_ai.models.moxing_model import MoxingModel
    model = MoxingModel(model_fn, ckpt_dir)
    c = Compressor(model, input_fn)
    c = c.pruning(config)
```


Compressor 中的quantizing方法用来对模型进行量化，量化对上面三种模型都有效

- config：量化算法所需要的配置，config包含和具体量化算法相关的一些配置
- 返回值:Conmpressor

示例如下:</br>
```
    from efficient_ai.compressor import Compressor
    from efficient_ai.models.moxing_model import MoxingModel
    model = MoxingModel(model_fn, ckpt_dir)
    c = Compressor(model, input_fn)
    c = c.quantizing(config)
```


Compressor 中的export方法用来导出压缩后的模型

- export_path：压缩后的模型文件和配置存放的文件夹


示例如下:</br>
```
    from efficient_ai.compressor import Compressor
    from efficient_ai.models.moxing_model import MoxingModel
    model = MoxingModel(model_fn, ckpt_dir)
    c = Compressor(model, input_fn)
    c = c.export(export_dir)
```
## 3.量化配置参数介绍和样例

- algorithm:字符串,可以是"DEFAULT"或者"GRID"
- inferece_engine：字符串,压缩后部署的推理引擎,现在支持tflite或者是tensorrt可以是"TFLITE"或"TENSORRT"
- engine_version：字符串,推理引擎的版本，现在tflite是"TFLITE13" tensorrt是"TENSORRT%"
- precision：字符串,压缩后的精度，tensorrt可以是"FP32"，"FP16"，"INT8"，tflite只能是"FP32"，"INT8"
- batch_size：整形，产生的模型的输入的batch_size,tensorrt是max_batch_size
- max_workspace_size_bytes：整形数字，(tensorrt专有)定义tensorrt在gpu上的最大工作空间

量化的使用[样例](./example/quantize_resnet50_example.py)


## 4.剪枝配置参数介绍和样例

- num_classes：整形，模型分类类别数
- prune_ratio：默认值0.5，表示剪枝50%；如果设置0.8，则表示要剪掉80%的通道，只留下20%的通道。
- nb_stages：默认值3，值越大，剪枝速度越慢，剪枝后精度越高；网络层数越多，值应越大；
- nb_iters_block：默认值10000，数据集越大，值应越大，剪枝速度越慢，剪枝后精度越高；
- nb_iters_layer：默认值500，数据集越大，值应越大，剪枝速度越慢，剪枝后精度越高；
- prune_lrn_rate_adam：默认值1e-3，block-wise和layer-wise使用Adam优化器，值太大可能学习不稳定，值太小可能会很慢，需配合参数nb_iters_block和nb_iters_layer2个参数来使用
- **kwargs：在kwargs字典中必须定义optimizer_fn和max_number_of_steps，optimizer_fn用于指定剪枝中微调学习率；max_number_of_steps用于指定剪枝微调的迭代步数。

说明：optimizer_fn、max_number_of_steps、num_classes和prune_ratio这几个参数是用户必须指定的，如果不指定会校验失败。
建议：如果模型运行在GPU上，建议在model_fn中指定模型格式为NCHW，这样会获得更高的性能。


剪枝的使用[样例](./example/dcp_mnist_example.py)

## 5.蒸馏配置参数介绍和样例
- softmax_temperature: 整形.The softmax temperature.
- soft_target_weight: 浮点.软目标的权重占比
- start_lr:浮点.开始的学习率
- num_epochs_per_decay: 整形.经过几次epoch后开始权重衰减
- num_samples_per_epoch: 整形.每个epoch中batch的数量
- rmsprop_decay: 浮点.RMSprop优化器参数
- rmsprop_momentum: 浮点.RMSprop优化器参数
- opt_epsilon: 浮点.优化器参数 
- weight_decay: 浮点. 平方损失函数的权重衰减.
- gradient_clipping: 布尔. 是否将梯度裁剪到[-1,1]范围
- use_labels_for_distillation: 布尔. 是否在蒸馏中使用标签
- quantized_student_model: 布尔. 是否在蒸馏中量化学生模型

蒸馏的使用[样例](./example/distill_mnist_example.py)


