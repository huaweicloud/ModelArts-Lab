### 1.2. moxing文件操作

MoXing中提供了一套文件对象API：[mox.file API document](MoXing-API-File-apidoc.md)，可以用来读写本地文件，同时也支持OBS文件系统。

#### 1.2.1. 一键切换

将以下代码写到启动脚本的最前面，在之后的Python运行中，几乎所有操作本地文件的接口都可以支持s3路径（具体支持的API参考下表）

```python
import moxing as mox
mox.file.shift('os', 'mox')
```

Hello World：

```python
import os
import moxing as mox

mox.file.shift('os', 'mox')

print(os.listdir('s3://bucket_name'))
with open('s3://bucket_name/hello_world.txt') as f:
  print(f.read())
```

#### 1.2.2. API对应关系

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


## 1.2.3. 调用实例

**读取一个OBS文件：**

```python
mox.file.read('s3://bucket_name/obs_file.txt')
```

**写一个OBS文件：**

```python
mox.file.write('s3://bucket_name/obs_file.txt', 'Hello World!')
```

**追加一个OBS文件：**

```python
mox.file.append('s3://bucket_name/obs_file.txt', '\nend of file.')
```

**列举一个OBS目录的顶层文件(夹)：**

```python
mox.file.list_directory('s3://bucket_name')
```

**递归列举一个OBS目录的所有文件(夹)：**

```python
mox.file.list_directory('s3://bucket_name', recursive=True)
```

**创建一个OBS目录：**

```python
mox.file.make_dirs('s3://bucket_name/sub_dir_0/sub_dir_1')
```

**判断一个OBS文件(夹)是否存在：**

```python
mox.file.exists('s3://bucket_name/sub_dir_0/sub_dir_1')
```

**判断一个OBS路径是否为文件夹：**

```python
mox.file.is_directory('s3://bucket_name/sub_dir_0/sub_dir_1')
```

**获取一个OBS文件(夹)的大小：**

```python
mox.file.get_size('s3://bucket_name/obs_file.txt')
```

**获取一个OBS文件(夹)的元信息：**

```python
stat = mox.file.stat('s3://bucket_name/obs_file.txt')
print(stat.length)
print(stat.mtime_nsec)
print(stat.is_directory)
```

**删除一个OBS文件：**

```python
mox.file.remove('s3://bucket_name/obs_file.txt', recursive=False)
```

**递归删除一个OBS目录：**

```python
mox.file.remove('s3://bucket_name/sub_dir_0/sub_dir_1', recursive=True)
```

**移动一个OBS文件(夹)，s3 -> s3**

```python
mox.file.rename('s3://bucket_name/obs_file.txt', 's3://bucket_name/obs_file_2.txt')
```

**移动一个OBS文件(夹)，s3 -> 本地**

```python
mox.file.rename('s3://bucket_name/obs_file.txt', '/tmp/obs_file.txt')
```

**移动一个OBS文件(夹)，本地 -> s3**

```python
mox.file.rename('/tmp/obs_file.txt', 's3://bucket_name/obs_file.txt')
```

**拷贝一个OBS文件，s3 -> s3**

```python
mox.file.copy('s3://bucket_name/obs_file.txt', 's3://bucket_name/obs_file_2.txt')
```

**下载一个OBS文件，s3 -> 本地**

```python
mox.file.copy('s3://bucket_name/obs_file.txt', '/tmp/obs_file.txt')
```

**上传一个OBS文件，本地 -> s3**

```python
mox.file.copy('/tmp/obs_file.txt', 's3://bucket_name/obs_file.txt')
```

**拷贝一个OBS文件夹，s3 -> s3**

```python
mox.file.copy_parallel('s3://bucket_name/sub_dir_0', 's3://bucket_name/sub_dir_1')
```

**下载一个OBS文件夹，s3 -> 本地**

```python
mox.file.copy_parallel('s3://bucket_name/sub_dir_0', '/tmp/sub_dir_0')
```

**上传一个OBS文件夹，本地 -> s3**

```python
mox.file.copy_parallel('/tmp/sub_dir_0', 's3://bucket_name/sub_dir_0')
```

**以二进制写模式打开一个本地文件：**

```python
f = mox.file.File('/tmp/local_file.txt', 'wb')
f.read()
f.close()
```

**以二进制可读可写模式打开一个OBS文件：**

```python
f = mox.file.File('s3://bucket_name/obs_file.txt', 'rb+')
f.read()
f.close()
```

**当读取OBS文件时，实际调用的是HTTP连接读去网络流，注意要记得在读取完毕后将文件关闭。为了防止忘记文件关闭操作，推荐使用with语句，在with语句退出时会自动调用`mox.file.File`对象的`close()`方法：**

```python
with mox.file.File('s3://bucket_name/obs_file.txt', 'r') as f:
  data = f.readlines()
```

**利用pandas读一个OBS文件**

```python
import pandas as pd
import moxing as mox
with mox.file.File("s3://dls-test/b.txt", "r") as f:
  csv = pd.read_csv(f)
```

**利用pandas写一个OBS文件**

```python
import pandas as pd
import moxing as mox
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
with mox.file.File("s3://dls-test/b.txt", "w") as f:
  df.to_csv(f)
```

**重定向logging日志到OBS文件**

```python
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

```python
stream.flush()
```

则会将日志文件同步到OBS。

用cv2从OBS读取一张图片：

```python
import cv2
import numpy as np
import moxing as mox
img = cv2.imdecode(np.fromstring(mox.file.read('s3://dls-test/xxx.jpg', binary=True), np.uint8), cv2.IMREAD_COLOR)
```

将一个不支持S3路径的API改造成支持S3路径的API：

pandas中对h5的文件读写`to_hdf`和`read_hdf`即不支持S3路径，也不支持输入一个文件对象，考虑以下代码会出现错误

```python
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])
df.to_hdf('s3://wolfros-net/hdftest.h5', key='df', mode='w')
pd.read_hdf('s3://wolfros-net/hdftest.h5')
```

通过重写pandas源码API的方式，将该API改造成支持S3路径的形式，

- 写h5到s3 = 写h5到本地缓存 + 上传本地缓存到S3 + 删除本地缓存
- 从s3读h5 = 下载h5到本地缓存 + 读取本地缓存 + 删除本地缓存

即将以下代码写在运行脚本的最前面，就能使运行过程中的`to_hdf`和`read_hdf`支持S3路径

```python
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
