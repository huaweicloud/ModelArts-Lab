# 1. MoXing-TensorFlow

## 1.1. 什么是MoXing

MoXing是华为云ModelArts团队自研的分布式训练加速框架，它构建于开源的深度学习引擎TensorFlow、MXNet、PyTorch、Keras之上。
相对于TensorFlow和MXNet原生API而言，MoXing API让模型代码的编写更加简单，允许用户只需要关心数据输入(input_fn)和模型构建(model_fn)的代码，即可实现任意模型在多GPU和分布式下的高性能运行，降低了TensorFlow和MXNet的使用门槛。

### 1.1.1. MoXing API与原生API对比

下面以手写体数字识别模型代码为例，对分别基于MoXing API与原生API编写的代码进行对比。为了展示清晰，将mnist_softmax.py做了简化，TensorFlow官方代码和使用MoXing的代码对比如图2-1所示，从图中对比可以看出，基于MoXing开发的模型分布式训练代码更简洁。

TensorFlow原生API与MoXing API分布式训练MNIST代码对比
<div align=center><img src="images_moxing_tensorflow/compare_moxing_and_tensorflow_train_api.png" width="800px"/></div>

- TensorFlow官方提供的训练手写体数字识别模型代码的源码链接为https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dist_test/python/mnist_replica.py。
- 采用MoXing API编写的具体代码如下

```
from tensorflow.examples.tutorials.mnist import input_data 
import tensorflow as tf 
tf.flags.DEFINE_string('data_url', '/tmp/tensorflow/mnist/input_data','Directory for storing input data') 
flags = tf.flags.FLAGS 
mnist = input_data.read_data_sets(flags.data_url, one_hot=True) 
import moxing.tensorflow as mox 
def input_fn(run_mode, **kwargs): 
  def gen(): 
      while True: 
      yield mnist.train.next_batch(100) 
  ds = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int64), 
      output_shapes=(tf.TensorShape([None, 784]), tf.TensorShape([None, 10]))) 
  return ds.make_one_shot_iterator().get_next() 
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
        batch_size=100, 
        auto_batch=False, 
        log_dir=flags.train_url, 
        max_number_of_steps=1000)

```
### 1.1.2. moxing文件操作

MoXing中提供了一套文件对象API：[mox.file](http://moxing.inhuawei.com/moxing.framework.file.html)，可以用来读写本地文件，同时也支持OBS文件系统。

#### 1.1.2.1. 一键切换

将以下代码写到启动脚本的最前面，在之后的Python运行中，几乎所有操作本地文件的接口都可以支持s3路径（具体支持的API参考下表）

```
import moxing as mox
mox.file.shift('os', 'mox')
```

Hello World：

```
import os
import moxing as mox

mox.file.shift('os', 'mox')

print(os.listdir('s3://bucket_name'))
with open('s3://bucket_name/hello_world.txt') as f:
  print(f.read())
```

#### 1.1.2.2. API对应关系

| Python          | mox.file                                      | tf.gfile                           |
| --------------- | --------------------------------------------- | ---------------------------------- |
| glob.glob       | mox.file.glob                                 | tf.gfile.Glob                      |
| os.listdir      | mox.file.list_directory(..., recursive=False) | tf.gfile.ListDirectory             |
| os.makedirs     | mox.file.make_dirs                            | tf.gfile.MakeDirs                  |
| os.mkdir        | mox.file.mk_dir                               | tf.gfile.MkDir                     |
| os.path.exists  | mox.file.exists                               | tf.gfile.Exists                    |
| os.path.getsize | mox.file.get_size                             | ×                                  |
| os.path.isdir   | mox.file.is_directory                         | tf.gfile.IsDirectory               |
| os.remove       | mox.file.remove(..., recursive=False)         | tf.gfile.Remove                    |
| os.rename       | mox.file.rename                               | tf.gfile.Rename                    |
| os.scandir      | mox.file.scan_dir                             | ×                                  |
| os.stat         | mox.file.stat                                 | tf.gfile.Stat                      |
| os.walk         | mox.file.walk                                 | tf.gfile.Walk                      |
| open            | mox.file.File                                 | tf.gfile.FastGFile(tf.gfile.Gfile) |
| shutil.copyfile | mox.file.copy                                 | tf.gfile.Copy                      |
| shutil.copytree | mox.file.copy_parallel                        | ×                                  |
| shutil.rmtree   | mox.file.remove(..., recursive=True)          | tf.gfile.DeleteRecursively         |

#### 1.1.2.3. mox.file API

> **所有的文件操作同时支持本地文件和OBS文件**

***class*  moxing.framework.file.File(\*args*, \*\*kwargs*)**

File Object. 文件对象，和python内置文件对象一样的用法

```
# 写文件.
with mox.file.File("s3://dls-test/a.txt", "w") as f:
  f.write("test")

# 读文件.
with mox.file.File("s3://dls-test/a.txt", "r") as f:
  print(f.read())

# 通过pandas写文件
import pandas as pd
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
with mox.file.File("s3://dls-test/b.txt", "w") as f:
  df.to_csv(f)

# 通过pandas读文件
import pandas as pd
with mox.file.File("s3://dls-test/b.txt", "r") as f:
  csv = pd.read_csv(f)
```

**参数**：

- **name**：文件路径或S3路径
- **mode**: “r”, “w”, “a”, “r+”, “w+”, “a+”, “br”, “bw”, “ba”, “br+”, “bw+”, “ba+”中的一个

**文件对象方法：**

- `close`()：关闭文件IO
- `flush`()：刷新Writable文件
- `isatty`()：文件是否连接到tty设备
- `mode`：返回文件打开模式
- `name：`返回文件名字
- `next`()：
- `read`(n=-1)：从文件中读取n个字节
- `readable`()：一个文件是否可读，返回布尔值
- `readline`(size=-1)：从文件中读取下一行
- `readlines`(sizeint=-1)：返回文件所有行
- seek(offset[, *whence*]) → None：参数偏移量是字节数。可选参数whence默认为0（从文件开头偏移，偏移量应> = 0）;其他值为1（相对于当前位置移动，正或负）和2（相对于文件末尾移动，通常是负数，尽管许多平台允许搜索超出文件末尾）。如果文件以文本模式打开，则只有tell（）返回的偏移是合法的。使用其他偏移会导致未定义的行为。请注意，并非所有文件对象都是可搜索的。
- seekable()：so.seek()是否可以被调用，返回布尔值
- `size`(*args, **kwargs)：返回文件大小
- `tell`()：返回文件中当前的位置
- `writable`()：文件是否可写，返回布尔值
- `write`(*data*)：在文件末尾写入数据
- `writelines`(*sequence*)：在文件末尾写入数据

***exception*`moxing.framework.file.MoxFileNameDuplicateException`**

基类: `exceptions.Exception`

当下载的目录和本地Unix OS目录或文件出现相同名称冲突时，会抛出此异常(在对象存储中允许)，提示没有相关的http响应。

```
import moxing as mox

try:
  mox.file.copy_parallel('s3://dls-test', '/cache/')
except mox.file.MoxFileNameDuplicateException as e:
  print(e)
```

***exception*`moxing.framework.file.MoxFileNotExistsException`**

基类: `exceptions.Exception`

当调用mox.file.stat于一个不存在的文件对象时抛出此异常，提示没有相关的http响应。

例子：

```
import moxing as mox

try:
  mox.file.stat('s3://dls-test/not_exists')
except mox.file.MoxFileNotExistsException as e:
  print(e)
```

***exception*`moxing.framework.file.MoxFileReadException`(*\*resp*, *\*\**args*)**

基类: `moxing.framework.file.file_io._MoxFileBaseRespException`

从先前成功建立的http连接的响应流中读取块时引发异常，提示的响应状态码是OK(200)。

```
import moxing as mox

try:
  mox.file.read('s3://dls-test/file.txt')
except mox.file.MoxFileReadException as e:
  print(e.resp)
```

***exception*`moxing.framework.file.MoxFileRespException`(*\*resp*, \*\*args)**

基类: `moxing.framework.file.file_io._MoxFileBaseRespException`

当访问s3存储发生异常时会抛出此异常，并且返回http返回码为>=300

```
import moxing as mox

try:
  mox.file.read('s3://dls-test/xxx')
except mox.file.MoxFileRespException as e:
  print(e.resp)
```

**moxing.frame.file.append(\*args, \*\*kwargs)**

在文件末尾写入数据，和open(url, 'a').write(data)一样的用法

```
import moxing as mox
mox.file.append('s3://bucket/dir/data.bin', b'xxx', binary=True)
```

参数：

- url - 本地路径或s3 url
- data - 写入文件的内容
- binary - 是否以二进制模式写文件

**moxing.frame.file.append_remote(\*args, \*\*kwargs)**

将一个OBS文件写入到另一个OBS文件中，且在末尾追加

```
import moxing as mox
mox.file.append_remote('s3://bucket/dir/data0.bin', 's3://bucket/dir/data1.bin')
```

参数：

- src_url - s3 url
- dsturl - s3 url

**moxing.frame.file.copy(\*args, \*\*kwargs)**

拷贝文件，只能拷贝单个文件，如果想拷贝一个目录，则必须使用mox.file.copy_parallel

```
import moxing as mox
mox.file.copy('/tmp/file1.txt', /tmp/file2.txt')
```

上传一个本地文件到OBS：

```
import moxing as mox
mox.file.copy('/tmp/file.txt', 's3://bucket/dir/file.txt')
```

下载OBS文件到本地：

```
import moxing as mox
mox.file.copy('s3://bucket/dir/file.txt', '/tmp/file.txt')
```

在OBS上拷贝文件：

```
import moxing as mox
mox.file.copy('s3://bucket/dir/file1.txt', s3://bucket/dir/file2.txt')
```

参数：

- src_url - 源路径或s3路径
- dst_url - 目的路径或s3路径
- client_id - id号或指定的obs客户请求id

**moxing.frame.file.copy_parallel(\*args, \*\*kwargs)**

从源地址拷贝所有文件到目的地址，和shutil.cpoytree一样的用法，此方法只能拷贝目录

```
copy_parallel(src_url='/tmp', dst_url='s3://bucket_name/my_data')
```

参数：

- src_url - 源路径或s3路径
- dst_url - 目的路径或s3路径
- file_list - 需要被拷贝的文件列表
- threads - 资源池中进程或线程的数量
- is_processing - 假如为True，将使用多进程，假如为False，将使用多线程
- use_queue - 是否使用队列来管理下载列表

**moxing.frame.file.exists(\*args, \*\*kwargs)**

路径中是否存在相应的文件

```
import moxing as mox
ret = mox.file.exists('s3://bucket/dir')
print(ret)
```

参数：

- url - 本地路径或s3 url

**moxing.frame.file.get_size(\*args, \*\*kwargs)**

获取文件大小

```
import moxing as mox
size = mox.file.get_size('s3://bucket/dir/file.txt')
print(size)
```

参数：

- url - 本地路径或s3 url
- recurisive - 是否列出路径下所有文件，假如路径是本地路径，则recurisive总是为True

**moxing.frame.file.glob(url)**

返回给定路径的文件列表

```
import moxing as mox
ret = mox.file.glob('s3://bucket/dir/*.jpg')
print(ret)
```

参数：

- url - 本地路径或s3 url

返回： 绝对路径的列表

**moxing.frame.file.is_directory(\*args, \*\*kwargs)**

判断给定路径或s3 url是否是目录

```
import moxing as mox
mox.file.is_directory('s3://bucket/dir')
```

参数：

- url - 本地路径或s3 url

**moxing.frame.file.list_directory(\*args, \*\*kwargs)**

列出给定目录下所有文件

```
import moxing as mox
ret = mox.file.list_directory('s3://bucket/dir', recursive=True)
print(ret)
```

参数：

- url - 本地路径或s3 url
- recurisive - 是否列出路径下所有文件，假如路径是本地路径，则recurisive总是为True
- remove_sep - 当OBS地址被解析的时候，是否移除最后一个分隔符字符串
- skip_file -  列出目录时是否跳过文件
- skip_dir -  列出目录时是否跳过文件夹

返回： 路径的列表

***moxing.frame.file.make_dirs(\*args, \*\*kwargs)***

递归创建目录

```
import moxing as mox
mox.file.make_dirs('s3://bucket/new_dir')
```

参数：

- url - 本地路径或s3 url

**moxing.frame.file.mk_dir(\*args, \*\*kwargs)**

创建目录

```
import moxing as mox
mox.file.mk_dir('s3://bucket/new_dir')
```

参数：

- url - 本地路径或s3 url

异常： OSError- 父目录不存在的时候抛出异常

**moxing.frame.file.read(\*args, \*\*kwargs)**

从本地或者OBS读取文件数据

```
import moxing as mox
image_buf = mox.file.read('/home/username/x.jpg', binary=True)
```

参数：

- url - 本地路径或s3 url
- client_id - id号或指定的obs客户请求id
- binary - 是否以二进制方式读取文件

**moxing.frame.file.remove(\*args, \*\*kwargs)**

删除文件或目录

```
# Remove only file 's3://bucket_name/file_name'
mox.file.remove('s3://bucket_name/file_name')

# Remove only directory 's3://bucket_name/dir_name'
mox.file.remove('s3://bucket_name/dir_name', recursive=True)
# Or
mox.file.remove('s3://bucket_name/dir_name/', recursive=True)

# Do nothing whether recursive is True or False
mox.file.remove('s3://bucket_name/file_name/')

# Do nothing whether recursive is True or False
mox.file.remove('s3://bucket_name/dir_name')

# Do nothing whether recursive is True or False
mox.file.remove('s3://bucket_name/dir_name/')

# Remove only file 's3://bucket_name/file_name'
mox.file.remove('s3://bucket_name/file_name', recursive=True)
```

- url - 本地路径或s3 url
- recurisive - 是否递归删除

**moxing.frame.file.rename(\*args, \*\*kwargs)**

重命名文件或目录

参数：

- src_url - 源路径或s3 url
- dst_url - 目的路径或s3 url

**moxing.frame.file.scan_dir(\*args, \*\*kwargs)**

调用操作系统的目录迭代系统来获取给定路径中文件的名称。仅在python 3中使用。与os.scandir的用法相同。

```
import moxing as mox
dir_gen = mox.file.scan_dir('s3://bucket/dir')
for d in dir_gen:
  print(d)
```

参数：

- url - 本地路径或s3 url

返回：DirEntry对象的生成器

**`moxing.framework.file.set_auth`(*ak=None*, *sk=None*, *server=None*, *port=None*, *is_secure=None*, *ssl_verify=None*, *long_conn_mode=None*, *path_style=None*, *retry=None*, *retry_wait=None*, *client_timeout=None*, *list_max_keys=None*)**

设置OBS鉴权信息

```
import moxing as mox
mox.file.set_auth(ak='xxx', sk='xxx')
```

参数：

- ak - Access Key
- sk - Secret Access Key
- server - OBS 服务端
- port - OBS服务器端口
- is_secure - 是否使用https
- ssl_verify - 是否使用ssl验证
- long_conn_mode - 是否使用长连接模式
- path_style - 是否使用路径风格. HEC则设置为True，当为公有云则设置为False
- retry - 总共尝试的时间
- retry_wait - 在每次尝试时所等待的时间
- client_timeout - obs客户端超时时间
- list_max_keys -  每页列出的最大对象数

**moxing.frame.file.stat(\*args, \*\*kwargs)**

返回给定路径的文件统计信息。与os.stat的用法相同

```
import moxing as mox
ret = mox.file.stat('s3://bucket/dir/file.txt')
print(ret)
```

参数：

- src_url - 源路径或s3 url

返回： 包含有关路径信息的文件统计结构

**moxing.frame.file.walk(\*args, \*\*kwargs)**

目录树生成器。与os.walk相同的用法

参数：

- url - 源路径或s3 url
- topdown - s3文件未使用. 其他的则看os.walk
- onerror - s3文件未使用. 其他的则看os.walk
- followlinks - s3文件未使用. 其他的则看os.walk

返回： 目录树生成器

**moxing.frame.file.write(\*args, \*\*kwargs)**

向文件中写入数据，覆盖原始内容，和open(url ,'w').write(data)一样的用法

```
import moxing as mox
mox.file.write('s3://bucket/dir/data.bin', b'xxx', binary=True)
```

参数：

- url - 源路径或s3 url
- data - 写入文件的内容
- binary - 是否以二进制模式写文件

## 1.1.2.4. Example

**读取一个OBS文件：**

```
mox.file.read('s3://bucket_name/obs_file.txt')
```

**写一个OBS文件：**

```
mox.file.write('s3://bucket_name/obs_file.txt', 'Hello World!')
```

**追加一个OBS文件：**

```
mox.file.append('s3://bucket_name/obs_file.txt', '\nend of file.')
```

**列举一个OBS目录的顶层文件(夹)：**

```
mox.file.list_directory('s3://bucket_name')
```

**递归列举一个OBS目录的所有文件(夹)：**

```
mox.file.list_directory('s3://bucket_name', recursive=True)
```

**创建一个OBS目录：**

```
mox.file.make_dirs('s3://bucket_name/sub_dir_0/sub_dir_1')
```

**判断一个OBS文件(夹)是否存在：**

```
mox.file.exists('s3://bucket_name/sub_dir_0/sub_dir_1')
```

**判断一个OBS路径是否为文件夹：**

```
mox.file.is_directory('s3://bucket_name/sub_dir_0/sub_dir_1')
```

**获取一个OBS文件(夹)的大小：**

```
mox.file.get_size('s3://bucket_name/obs_file.txt')
```

**获取一个OBS文件(夹)的元信息：**

```
stat = mox.file.stat('s3://bucket_name/obs_file.txt')
print(stat.length)
print(stat.mtime_nsec)
print(stat.is_directory)
```

**删除一个OBS文件：**

```
mox.file.remove('s3://bucket_name/obs_file.txt', recursive=False)
```

**递归删除一个OBS目录：**

```
mox.file.remove('s3://bucket_name/sub_dir_0/sub_dir_1', recursive=True)
```

**移动一个OBS文件(夹)，s3 -> s3**

```
mox.file.rename('s3://bucket_name/obs_file.txt', 's3://bucket_name/obs_file_2.txt')
```

**移动一个OBS文件(夹)，s3 -> 本地**

```
mox.file.rename('s3://bucket_name/obs_file.txt', '/tmp/obs_file.txt')
```

**移动一个OBS文件(夹)，本地 -> s3**

```
mox.file.rename('/tmp/obs_file.txt', 's3://bucket_name/obs_file.txt')
```

**拷贝一个OBS文件，s3 -> s3**

```
mox.file.copy('s3://bucket_name/obs_file.txt', 's3://bucket_name/obs_file_2.txt')
```

**下载一个OBS文件，s3 -> 本地**

```
mox.file.copy('s3://bucket_name/obs_file.txt', '/tmp/obs_file.txt')
```

**上传一个OBS文件，本地 -> s3**

```
mox.file.copy('/tmp/obs_file.txt', 's3://bucket_name/obs_file.txt')
```

**拷贝一个OBS文件夹，s3 -> s3**

```
mox.file.copy_parallel('s3://bucket_name/sub_dir_0', 's3://bucket_name/sub_dir_1')
```

**下载一个OBS文件夹，s3 -> 本地**

```
mox.file.copy_parallel('s3://bucket_name/sub_dir_0', '/tmp/sub_dir_0')
```

**上传一个OBS文件夹，本地 -> s3**

```
mox.file.copy_parallel('/tmp/sub_dir_0', 's3://bucket_name/sub_dir_0')
```

**以二进制写模式打开一个本地文件：**

```
f = mox.file.File('/tmp/local_file.txt', 'wb')
f.read()
f.close()
```

**以二进制可读可写模式打开一个OBS文件：**

```
f = mox.file.File('s3://bucket_name/obs_file.txt', 'rb+')
f.read()
f.close()
```

**当读取OBS文件时，实际调用的是HTTP连接读去网络流，注意要记得在读取完毕后将文件关闭。为了防止忘记文件关闭操作，推荐使用with语句，在with语句退出时会自动调用`mox.file.File`对象的`close()`方法：**

```
with mox.file.File('s3://bucket_name/obs_file.txt', 'r') as f:
  data = f.readlines()
```

**利用pandas读一个OBS文件**

```
import pandas as pd
import moxing as mox
with mox.file.File("s3://dls-test/b.txt", "r") as f:
  csv = pd.read_csv(f)
```

**利用pandas写一个OBS文件**

```
import pandas as pd
import moxing as mox
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
with mox.file.File("s3://dls-test/b.txt", "w") as f:
  df.to_csv(f)
```

**重定向logging日志到OBS文件**

```
import logging
import moxing as mox

stream = mox.file.File('s3://bucket_name/sub_dir_0/logging.txt', 'w')
logging.basicConfig(level=logging.DEBUG, stream=stream)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
...
logging.info('hello world')
...
stream.close()
```

在最后一定记得将stream关闭，才能保证日志文件正确输出到OBS了，如果想在代码运行过程中不断写入log文件，每次调用：

```
stream.flush()
```

则会将日志文件同步到OBS。

用cv2从OBS读取一张图片：

```
import cv2
import numpy as np
import moxing as mox
img = cv2.imdecode(np.fromstring(mox.file.read('s3://dls-test/xxx.jpg', binary=True), np.uint8), cv2.IMREAD_COLOR)
```

将一个不支持S3路径的API改造成支持S3路径的API：

pandas中对h5的文件读写`to_hdf`和`read_hdf`即不支持S3路径，也不支持输入一个文件对象，考虑以下代码会出现错误

```
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])
df.to_hdf('s3://wolfros-net/hdftest.h5', key='df', mode='w')
pd.read_hdf('s3://wolfros-net/hdftest.h5')
```

通过重写pandas源码API的方式，将该API改造成支持S3路径的形式，

- 写h5到s3 = 写h5到本地缓存 + 上传本地缓存到S3 + 删除本地缓存
- 从s3读h5 = 下载h5到本地缓存 + 读取本地缓存 + 删除本地缓存

即将以下代码写在运行脚本的最前面，就能使运行过程中的`to_hdf`和`read_hdf`支持S3路径

```
import os
import moxing as mox
import pandas as pd
from pandas.io import pytables
from pandas.core.generic import NDFrame

to_hdf_origin = getattr(NDFrame, 'to_hdf')
read_hdf_origin = getattr(pytables, 'read_hdf')


def to_hdf_override(self, path_or_buf, key, **kwargs):
    tmp_dir = '/cache/hdf_tmp'
    file_name = os.path.basename(path_or_buf)
    mox.file.make_dirs(tmp_dir)
    local_file = os.path.join(tmp_dir, file_name)
    to_hdf_origin(self, local_file, key, **kwargs)
    mox.file.copy(local_file, path_or_buf)
    mox.file.remove(local_file)


def read_hdf_override(path_or_buf, key=None, mode='r', **kwargs):
    tmp_dir = '/cache/hdf_tmp'
    file_name = os.path.basename(path_or_buf)
    mox.file.make_dirs(tmp_dir)
    local_file = os.path.join(tmp_dir, file_name)
    mox.file.copy(path_or_buf, local_file)
    result = read_hdf_origin(local_file, key, mode, **kwargs)
    mox.file.remove(local_file)
    return result

setattr(NDFrame, 'to_hdf', to_hdf_override)
setattr(pytables, 'read_hdf', read_hdf_override)
setattr(pd, 'read_hdf', read_hdf_override)
```

## 1.2. Getting Started

- import moxing.tensorflow as mox
  ​        导入 `moxing.tensorflow` 包并重命名为 `mox`

- input_fn
  ​        用户在运行脚本中必须定义 `input_fn` 方法。`input_fn` 的返回值是一个输入数据的list，其中每个元素必须是TensorFlow定义的Tensor类型，不可以是python、numpy、panda等类型。

  - 样例

      ```python
      def input_fn(run_mode, **kwargs): 
          if run_mode == mox.ModeKeys.TRAIN: 
              inputs = read_training_data() 
          else: 
              inputs = read_evaluation_data() 
          return inputs
      ```

- model_fn
    ​    用户在运行脚本中必须定义 `model_fn` 方法， `model_fn` 方法必须返回一个`mox.ModelSpec` 的实例。

    - 样例

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
          return mox.ModelSpec(loss=cross_entropy,
                               log_info={'loss': cross_entropy, 'accuracy': accuracy})
        ```

- mox.run（规避auto_batch）

- mox.ModelSpec
    ​    用户定义的 `model_fn` 中必须返回的类型。

- num_gpus
    ​    训练时使用GPU的数量。

- job_name
    ​    节点的作业名称，可以是 `ps` 或者 `worker`。

- task_index
    ​    作业中的任务索引。

- ps_hosts
    ​    `ps` 节点主机列表，列表中内容使用逗号分隔，例如：`'localhost:port1,localhost:port2'` 。

- worker_hosts
    ​    `worker` 节点主机列表，列表中内容使用逗号分隔，例如：`'localhost:port1,localhost:port` 。

同时给出本地运行和ModelArts运行案例

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import moxing.tensorflow as mox
import os

tf.flags.DEFINE_string('data_url', None, 'Dir of dataset')
tf.flags.DEFINE_string('train_url', None, 'Train Url')

flags = tf.flags.FLAGS


def check_dataset():
  work_directory = flags.data_url
  filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz',
               't10k-labels-idx1-ubyte.gz']

  for filename in filenames:
    filepath = os.path.join(work_directory, filename)
    if not mox.file.exists(filepath):
      raise ValueError('MNIST dataset file %s not found in %s' % (filepath, work_directory))


def main(*args, **kwargs):
  check_dataset()
  mnist = input_data.read_data_sets(flags.data_url, one_hot=True)

  def input_fn(run_mode, **kwargs):
    def gen():
      while True:
        yield mnist.train.next_batch(50)

    ds = tf.data.Dataset.from_generator(
      gen, output_types=(tf.float32, tf.int64),
      output_shapes=(tf.TensorShape([None, 784]), tf.TensorShape([None, 10])))
    return ds.make_one_shot_iterator().get_next()

  def model_fn(inputs, run_mode, **kwargs):
    x, y_ = inputs
    W = tf.get_variable(name='W', initializer=tf.zeros([784, 10]))
    b = tf.get_variable(name='b', initializer=tf.zeros([10]))
    y = tf.matmul(x, W) + b
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    export_spec = mox.ExportSpec(inputs_dict={'images': x}, outputs_dict={'logits': y})
    return mox.ModelSpec(loss=cross_entropy, log_info={'loss': cross_entropy, 'accuracy': accuracy},
                         export_spec=export_spec)

  mox.run(input_fn=input_fn,
          model_fn=model_fn,
          optimizer_fn=mox.get_optimizer_fn('sgd', learning_rate=0.01),
          run_mode=mox.ModeKeys.TRAIN,
          batch_size=50,
          auto_batch=False,
          log_dir=flags.train_url,
          max_number_of_steps=1000,
          log_every_n_steps=10,
          export_model=mox.ExportKeys.TF_SERVING)


if __name__ == '__main__':
  tf.app.run(main=main)
```



### 1.2.2. 迁移学习resnet50-flowers


- 解耦型

- ImageClassificationRawMetadata

     ```pyt
     metadata = ImageClassificationRawMetadata(base_dir='/export/dataset')
     ```

     ​    返回图片分类的数据集的元信息。

- ImageClassificationRawDataset

    ```python
    metadata = ImageClassificationRawMetadata(base_dir='/export/dataset')
    dataset = ImageClassificationRawDataset(metadata)
    ```

    ​    返回图片分类的数据集的数据。

- get_data_augmentation_fn

    ```python
    data_augmentation_fn = mox.get_data_augmentation_fn(
          name='resnet_v1_50', run_mode=mox.ModeKeys.TRAIN,
          output_height=224, output_width=224)
    image = data_augmentation_fn(image)
    ```

    ​    返回图片分类的数据增强方法，该数据增强方法接收维度为 `[height, width, channel]` 的单张图片。

- get_model_meta

    ```python
    >>> import moxing.tensorflow as mox
    >>> model_meta = mox.get_model_meta('resnet_v1_50')
    >>> print(model_meta.default_image_size)
    224
    >>> print(model_meta.default_labels_offset)
    1
    >>> print(model_meta.default_logits_pattern)
    logits
    ```

    ​    返回模型的元信息，包括默认图片大小，以及默认类标偏移等。

- get_model_fn

    ```python
    mox_mdoel_fn = mox.get_model_fn(
      name='resnet_v1_50',
      run_mode=mox.MokeKeys.TRAIN,
      num_classes=1000,
      weight_decay=0.0001,
      data_format='NHWC',
      batch_norm_fused=True,
      batch_renorm=False,
      image_height=224,
      image_width=224)
    logits, end_points = mox_mdoel_fn(images)
    ```

    ​    返回模型函数，用于处理输入数据，返回 `logits` 等结果，例如 `logits, end_points = model_fn(images)`。

- learning_rate_scheduler.piecewise_lr

- 耦合型

- checkpoint_exclude_patterns

    ```python
    exclude_list = ['global_step', model_meta.default_logits_pattern]
    if mox.get_flag('checkpoint_exclude_patterns'):
      exclude_list.append(mox.get_flag('checkpoint_exclude_patterns'))
    checkpoint_exclude_patterns = ','.join(exclude_list)
    mox.set_flag('checkpoint_exclude_patterns', checkpoint_exclude_patterns)
    ```

    ​    变量名模式用于在导入模型时排除符合模式的变量。

- set_flag

    ```python
    mox.set_flag('num_gpus', 1)
    ```

    ​    设置内部参数的值。

- mox.get_collection

    ```python
    regularization_losses = mox.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    global_variables = mox.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    ```

    ​    返回给定 `name` 的集合中值的列表，如果没有向该集合添加任何值，则返回空列表。当使用多GPU时，需要使用 `mox.get_collection` 代替 `tf.get_collection` 来获取当前GPU上model_fn定义的Collection，以防止返回值与预期不符。

- mox.run(checkpoint_path) （解释log_dir和checkpoint_path的区别）（规避auto_batch）
    ​    执行训练、推理等作业。`mox.run` 中 `log_dir` 主要用来输出 `TensorBoard` 的`Summary` 文件和 `checkpoint` 文件，`checkpoint_path` 用来指定载入 `checkpoint` 的路径。`mox.run` 对 `checkpoint` 文件的载入优先级如下：
    （1）  当log_dir中存在checkpoint时，无视checkpoint_path，从log_dir中载入checkpoint。如果当前模式为mox.ModeKeys.TRAIN，则将新的checkpoint保存在log_dir中。
    （2）  当log_dir中不存在checkpoint时，从checkpoint_path中载入checkpoint。如果当前当前模式为mox.ModeKeys.TRAIN，则将新的checkpoint保存在log_dir中。
    （3）  当log_dir和checkpoint_path中都不存在checkpoint时，如果当前模式为mox.ModeKeys.TRAIN，则初始化所有变量并将新的checkpoint保存在log_dir中。如果当前不是mox.ModeKeys.TRAIN，则抛出异常（非训练模式下必须提供checkpoint)。

- mox.ExportSpec

    ```python
    export_spec = mox.ExportSpec(inputs_dict={'images': images},
                                   outputs_dict={'logits': logits},
                                   version=1)
    ```

    ​    定义 `model_fn` 的返回值 `mox.ModelSpec`中参数 `export_spec` 的数据类型，用于指定输出的PB文件中的输入节点，输出节点和版本号。输出的PB模型只会保存 `model_fn` 中定义的计算流图。

同时给出本地运行和ModelArts运行案例

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
  num_gpus = mox.get_flag('num_gpus')
  num_workers = len(mox.get_flag('worker_hosts').split(','))

  model_meta = mox.get_model_meta(flags.model_name)
  exclude_list = ['global_step', model_meta.default_logits_pattern]
  checkpoint_exclude_patterns = ','.join(exclude_list)
  mox.set_flag('checkpoint_exclude_patterns', checkpoint_exclude_patterns)

  data_meta = mox.ImageClassificationRawMetadata(base_dir=flags.data_url)

  mox.set_flag('loss_scale', 1024.0)

  def input_fn(mode, **kwargs):
    data_augmentation_fn = mox.get_data_augmentation_fn(name=flags.model_name,
                                                        run_mode=mode)

    dataset = mox.ImageClassificationRawDataset(data_meta,
                                                batch_size=flags.batch_size,
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

    # Append reg loss
    regularization_losses = mox.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.add_n(regularization_losses)
    loss = loss + regularization_loss

    logits_fp32 = tf.cast(logits, tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_fp32, labels, 1), tf.float32))

    export_spec = mox.ExportSpec(inputs_dict={'images': images},
                                 outputs_dict={'logits': logits},
                                 version='model')

    return mox.ModelSpec(loss=loss,
                         log_info={'loss': loss, 'accuracy': accuracy}, export_spec=export_spec)

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
          batch_size=flags.batch_size,
          auto_batch=False,
          checkpoint_path=flags.checkpoint_url,
          max_number_of_steps=1500)


if __name__ == '__main__':
  tf.app.run(main=main)
```



### 1.2.3. 进阶特性

引入的API:

- 解耦型
- mox.AsyncRawGenerator
    ​    创建异步数据生成器。

- mox.var_scope
    ​    在使用fp16 + fp32格式数据训练时，可以用 `mox.var_scope` 创建一个变量范围，在此范围内创建的所有变量的 `dtype` 被强制改为 `force_dtype`，示例代码如下。

```python
  with mox.var_scope(force_dtype=tf.float32):
    logits = resnet50(images)
```

- 耦合型
- loss_scale
    ​    在计算梯度之前将损失 `mox` 会乘以 `loss_scale`，然后将每个梯度除以 `loss_scale`。在数学上，没有任何影响，但它有助于避免fp16下溢问题。将 `loss_scale` 设置为1，则禁用。

## 1.3. 接口文档

### 1.3.1. 命令行参数和通用


#### 1.3.1.1. mox.get_flag

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

### 1.3.1.2. mox.set_flag
  
    moxing.tensorflow.executor.set_flag(name, value)
  
  设置MoXing内部定义的某个运行参数的值，需要在input_fn, model_fn, optimizer_fn函数之外及mox.run之前设置

- 参数说明
  
  - name：参数名称，由moxing/tensorflow/utils/hyper_param_flags.py定义。
  - value：参数值。
  
- 示例

```python
mox.set_flag('checkpoint_exclude_patterns', 'logits')
```

#### 1.3.1.3. ModeKeys 

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

#### 1.3.1.4. 流程简介

  mox.run为任务主入口，由最外层调用，运行run函数时，会先调用input_fn构建数据队列，然后调用model_fn构建tensorflow流程运行图，对于多GPU的环境，model_fn会被调用多次，最后启动运行时调用optimizer_fn。

### 1.3.2. 输入数据

#### 1.3.2.1. input_fn以及tf.data.Dataset获取数据

  用户在运行脚本中必须定义input_fn方法，示例如下：

```python
  def input_fn(run_mode, **kwargs): 
    if run_mode == mox.ModeKeys.TRAIN: 
      inputs = read_training_data() 
    else: 
      inputs = read_evaluation_data() 
    return inputs
```

其中 `run_mode` 表示当前的运行模式，该值与用户调用 `mox.run` 时给定的入参有关，请用户在 `input_fn` 中做好判断，区分训练数据集和验证数据集。`mox.ModeKeys.TRAIN`：表示当前为训练模式；`mox.ModeKeys.EVAL`：表示当前为验证模式。
`input_fn` 的返回值是一个list，其中每个元素必须是TensorFlow定义的Tensor类型，不可以是python、numpy、panda等类型。

用户实现数据集类 `my_dataset`，提供 `next()` 方法获取下一份数据，则可以用户构建 `input_fn` 方法，基本写法如下：

```python
import tensorflow as tf
import moxing.tensorflow as tf
import my_dataset

def input_fn(run_mode, **kwargs)
  def gen():
    while True:
      yield my_dataset.next()

  ds = tf.data.Dataset.from_generator(
      gen, 
      output_types=(tf.float32, tf.int64),
      output_shapes=(tf.TensorShape([224, 224, 3]), tf.TensorShape([1000])))

  return ds.make_one_shot_iterator().get_next()
```

​    在使用这种方法时，由于数据的产生顺序完全取决于用户实现的代码，MoXing无法保证数据的shuffle，所以用户必须确保自己提供的 `my_dataset.next()` 具有数据随机性。




- 利用tf.data.Dataset.from_generator读取任意数据集，避免使用placeholder

#### 1.3.2.2. ImageClassificationRawMetadata

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



#### 1.3.2.3 ImageClassificationRawDataset

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



#### 1.3.2.4. ImageClassificationTFRecordMetadata

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



### 1.3.2.5. ImageClassificationTFRecordDataset

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



#### 1.3.2.6. MultilabelClassificationRawMetadata

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



#### 1.3.2.7. MultilabelClassificationRawDataset

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



#### 1.3.2.8. get_data_augmentation_fn

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

#### 1.3.2.9. PreprocessingKeys
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

### 1.3.3. 内置模型

moxing 内置了大量业界典型的模型供用户直接使用

#### 1.3.3.1. <span id="networkKeys">NetworkKeys</span>

    class moxing.tensorflow.nets.NetworkKeys

> ALEXNET_V2 = 'alexnet_v2'

> CIFARNET = 'cifarnet'

> OVERFEAT = 'overfeat'
  
> VGG_A = 'vgg_a'
  
> VGG_16 = 'vgg_16'

> VGG_19 = 'vgg_19'

> VGG_A_BN = 'vgg_a_bn'
  
> VGG_16_BN = 'vgg_16_bn'
  
> VGG_19_BN = 'vgg_19_bn'

> INCEPTION_V1 = 'inception_v1'
  
> INCEPTION_V2 = 'inception_v2'
  
> INCEPTION_V3 = 'inception_v3'

> INCEPTION_V4 = 'inception_v4'

> INCEPTION_RESNET_V2 = 'inception_resnet_v2'

> LENET = 'lenet'

> RESNET_V1_18 = 'resnet_v1_18'

> RESNET_V1_50 = 'resnet_v1_50'

> RESNET_V1_50_8K = 'resnet_v1_50_8k'

> RESNET_V1_101 = 'resnet_v1_101'

> RESNET_V1_152 = 'resnet_v1_152'

> RESNET_V1_200 = 'resnet_v1_200'

> RESNET_V2_50 = 'resnet_v2_50'

> RESNET_V2_101 = 'resnet_v2_101'

> RESNET_V2_152 = 'resnet_v2_152'
  
> RESNET_V2_200 = 'resnet_v2_200'
  
> RESNEXT_B_50 = 'resnext_b_50'
  
> RESNEXT_B_101 = 'resnext_b_101'
  
> RESNEXT_C_50 = 'resnext_c_50'
  
> RESNEXT_C_101 = 'resnext_c_101'
  
> PVANET = 'pvanet'

> MOBILENET_V1 = 'mobilenet_v1'

> MOBILENET_V1_075 = 'mobilenet_v1_075'

> MOBILENET_V1_050 = 'mobilenet_v1_050'
  
> MOBILENET_V1_025 = 'mobilenet_v1_025'

> MOBILENET_V2 = 'mobilenet_v2'

> MOBILENET_V2_140 = 'mobilenet_v2_140'

> MOBILENET_V2_035 = 'mobilenet_v2_035'

> RESNET_V1_20 = 'resnet_v1_20'

> RESNET_V1_110 = 'resnet_v1_110'

> NASNET_CIFAR = 'nasnet_cifar'

> NASNET_MOBILE = 'nasnet_mobile'

> NASNET_LARGE = 'nasnet_large'

> PNASNET_LARGE = 'pnasnet_large'
  
> PNASNET_MOBILE = 'pnasnet_mobile'

#### 1.3.3.2. get_model_fn

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
#### 1.3.3.3. get_model_meta

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

### 1.3.3.4. ModelSpec

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

### 1.3.3.4. ExportSpec

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

### 1.3.3.5. mox.get_collection

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


#### 1.3.3.6. 混合精度mox.var_scope

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

### 1.3.4. 优化器

优化器就是各种对于梯度下降算法的优化。
	
#### 1.3.4.1. optimizer_fn
用户可以在运行脚本中定义optimizer_fn,如下：

```python
def optimizer_fn(): 
    return tf.train.GradientDescentOptimizer(0.5)
```

#### 1.3.4.2. mox.get_optimizer_fn()

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

#### 1.3.4.3. learning_rate_scheduler.piecewise_lr

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

### 1.3.5. 运行

### 1.3.5.1. mox.run
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


## 1.4. Benchmarks

### 1.4.1. 基于云道CSB-BMS集群吞吐量测试：

----------

- 环境：云道DLS服务
- TensorFlow: 1.4.0
- GPU: Tesla-P100
- 使用resnet_v1_50模型

<div align=center><img src="images_moxing_tensorflow/csb_p100_multiGPUs_resnet50_benchmarks.png" width="900px"/></div>

云道单机多卡吞吐量测试结果（真实数据）：

|FPS|TensorFlow-benchmarks||||MoXing|
|:---:|:---:|:---:|:---:|:---:|:---:|
||parameter_server+cpu|parameter_server+gpu|replicated+cpu|replicated+gpu|replicated_host+gpu|
|GPU=1|222.66|223.61|222.71|222.64|235.03|
|GPU=4|578.02|577.2|630.8|681.89|798.54|
|GPU=8|826.99|855.19|1278.25|1268.17|1505.26|

|Speedup|TensorFlow-benchmarks||||MoXing|
|:---:|:---:|:---:|:---:|:---:|:---:|
||parameter_server+cpu|parameter_server+gpu|replicated+cpu|replicated+gpu|replicated_host+gpu|
|GPU=1|1|1|1|1|1
|GPU=4|0.700155046|0.699161781|0.764087407|0.825972673|0.869036218
|GPU=8|0.500866079|0.517945395|0.774171471|0.768066525|0.81907322

云道单机多卡吞吐量测试结果（假数据）：

|FPS|TensorFlow-benchmarks||||MoXing|
|:---:|:---:|:---:|:---:|:---:|:---:|
||parameter_server+cpu|parameter_server+gpu|replicated+cpu|replicated+gpu|replicated_host+gpu|
|GPU=1|222.66|223.61|222.71|222.64|235.03
|GPU=4|711.76|687.82|786.55|786.09|887.79
|GPU=8|1385.24|1272.67|1424.61|1424.9|1649.58

|Speedup|TensorFlow-benchmarks||||MoXing|
|:---:|:---:|:---:|:---:|:---:|:---:|
||parameter_server+cpu|parameter_server+gpu|replicated+cpu|replicated+gpu|replicated_host+gpu|
|GPU=1|1|1|1|1|1
|GPU=4|0.795760476|0.768995125|0.87937704|0.878862752|0.944336893
|GPU=8|0.774361612|0.71143397|0.796369796|0.796531908|0.877324171

----------

- 环境：云道DLS服务
- TensorFlow: 1.8.0
- GPU: 8xP100 & 8xV100
- 混合精度：FP16计算+FP32存储
- 使用脚本: distributed_imagenet_benchmarks.py

<div align=center><img src="images_moxing_tensorflow/csb_v100_distributed_resnet50_fp16_fp32_benchmarks.png" width="800px"/></div>

|node|GPUs|batch/sec|batch_size|FPS|speedup|dist_speedup|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|1|3.458988719|256|885.5011121|1|1|
|1|4|3.044783236|1024|3117.858033|0.880252433|1|
|1|8|2.8840776|2048|5906.590924|0.833792138|1|
|2|16|2.711127837|4096|11104.77962|0.783792044|0.940032902|
|4|32|2.663784904|8192|21821.72594|0.770105115|0.923617626|
|8|64|2.648736077|16384|43396.89188|0.765754471|0.918399726|
|16|128|2.624302388|32768|85993.14064|0.758690647|0.909927801|


### 1.4.2. 基于云道CSB-BMS集群的数据读取吞吐量测试

- 环境：云道DLS服务
- TensorFlow: 1.8.0
- GPU: Tesla-V100
- 使用脚本: dataset_benchmarks.py

<div align=center><img src="images_moxing_tensorflow/csb_multiGPUs_benchmarks_dataset.png" width="800px"/></div>

数据集读取方法：

- 图像数据为3671张图像，总大小为222.082MB，从OBS读取图像数据。
- 若读取的是纯图像文件，则采取64子进程的方法并行读取，若读取的是tfrecord，则采用cycle_length为32的parallel_interleave方式读取（TF-1.8之后的特性）
- 读取分为without cache和with cache，即是否利用tf.data.Dataset模块利用本地缓存（读取第一个epoch时在容器内挂载的SSD中缓存数据，从第二个epoch之后直接从SSD读取数据，而不是从OBS）。
- 读取类型分为三种。bytes: 仅读取图像数据字节流，不进行解码。images：读取图像数据并进行解码，如果做cache则缓存解码后数据。tfrecord：读取tfrecord文件，并进行解码，如果做cache则缓存解码后数据。解码后数据较原图像占用空间会增大5-10倍。
- 在计算8xP100和8xV100对数据读取性能需求时，采用resnet_v1_50模型，在8xP100中的吞吐量按1500 images/sec计算，在8xV100中的吞吐量按3100 images/sec计算

|without cache|with cache|without cache|with cache|without cache|with cache|8xV100-FP16|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|619.64 |863.76 |375.15 |402.57 |139.06 |683.51 |383.35 |