### mox.file API document

> **所有的文件操作同时支持本地文件和OBS文件**

***class*  moxing.framework.file.File(\*args*, \*\*kwargs*)**

File Object. 文件对象，和python内置文件对象一样的用法

```python
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
- `mode`：返回文件打开模式
- `name：`返回文件名字
- `read`(n=-1)：从文件中读取n个字节
- `readable`()：一个文件是否可读，返回布尔值
- `readline`(size=-1)：从文件中读取下一行
- `readlines`(sizeint=-1)：返回文件所有行
- `seek`(offset[, *whence*]) → None：参数偏移量是字节数。可选参数whence默认为0（从文件开头偏移，偏移量应> = 0）;其他值为1（相对于当前位置移动，正或负）和2（相对于文件末尾移动，通常是负数，尽管许多平台允许搜索超出文件末尾）。如果文件以文本模式打开，则只有tell（）返回的偏移是合法的。使用其他偏移会导致未定义的行为。请注意，并非所有文件对象都是可搜索的。
- `seekable`()：seek()是否可以被调用，返回布尔值
- `size`(*args, **kwargs)：返回文件大小
- `tell`()：返回文件中当前的位置
- `writable`()：文件是否可写，返回布尔值
- `write`(*data*)：在文件末尾写入数据
- `writelines`(*sequence*)：在文件末尾写入数据


**moxing.frame.file.append(\*args, \*\*kwargs)**

在文件末尾写入数据，和open(url, 'a').write(data)一样的用法

```python
import moxing as mox
mox.file.append('s3://bucket/dir/data.bin', b'xxx', binary=True)
```

参数：

- url - 本地路径或s3 url
- data - 写入文件的内容
- binary - 是否以二进制模式写文件

**moxing.frame.file.append_remote(\*args, \*\*kwargs)**

将一个OBS文件写入到另一个OBS文件中，且在末尾追加

```python
import moxing as mox
mox.file.append_remote('s3://bucket/dir/data0.bin', 's3://bucket/dir/data1.bin')
```

参数：

- src_url - s3 url
- dsturl - s3 url

**moxing.frame.file.copy(\*args, \*\*kwargs)**

拷贝文件，只能拷贝单个文件，如果想拷贝一个目录，则必须使用mox.file.copy_parallel

```python
import moxing as mox
mox.file.copy('/tmp/file1.txt', /tmp/file2.txt')
```

上传一个本地文件到OBS：

```python
import moxing as mox
mox.file.copy('/tmp/file.txt', 's3://bucket/dir/file.txt')
```

下载OBS文件到本地：

```python
import moxing as mox
mox.file.copy('s3://bucket/dir/file.txt', '/tmp/file.txt')
```

在OBS上拷贝文件：

```python
import moxing as mox
mox.file.copy('s3://bucket/dir/file1.txt', s3://bucket/dir/file2.txt')
```

参数：

- src_url - 源路径或s3路径
- dst_url - 目的路径或s3路径
- client_id - id号或指定的obs客户请求id

**moxing.frame.file.copy_parallel(\*args, \*\*kwargs)**

从源地址拷贝所有文件到目的地址，和shutil.cpoytree一样的用法，此方法只能拷贝目录

```python
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

```python
import moxing as mox
ret = mox.file.exists('s3://bucket/dir')
print(ret)
```

参数：

- url - 本地路径或s3 url

**moxing.frame.file.get_size(\*args, \*\*kwargs)**

获取文件大小

```python
import moxing as mox
size = mox.file.get_size('s3://bucket/dir/file.txt')
print(size)
```

参数：

- url - 本地路径或s3 url
- recurisive - 是否列出路径下所有文件，假如路径是本地路径，则recurisive总是为True

**moxing.frame.file.glob(url)**

返回给定路径的文件列表

```python
import moxing as mox
ret = mox.file.glob('s3://bucket/dir/*.jpg')
print(ret)
```

参数：

- url - 本地路径或s3 url

返回： 绝对路径的列表

**moxing.frame.file.is_directory(\*args, \*\*kwargs)**

判断给定路径或s3 url是否是目录

```python
import moxing as mox
mox.file.is_directory('s3://bucket/dir')
```

参数：

- url - 本地路径或s3 url

**moxing.frame.file.list_directory(\*args, \*\*kwargs)**

列出给定目录下所有文件

```python
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

```python
import moxing as mox
mox.file.make_dirs('s3://bucket/new_dir')
```

参数：

- url - 本地路径或s3 url

**moxing.frame.file.mk_dir(\*args, \*\*kwargs)**

创建目录

```python
import moxing as mox
mox.file.mk_dir('s3://bucket/new_dir')
```

参数：

- url - 本地路径或s3 url

异常： OSError- 父目录不存在的时候抛出异常

**moxing.frame.file.read(\*args, \*\*kwargs)**

从本地或者OBS读取文件数据

```python
import moxing as mox
image_buf = mox.file.read('/home/username/x.jpg', binary=True)
```

参数：

- url - 本地路径或s3 url
- client_id - id号或指定的obs客户请求id
- binary - 是否以二进制方式读取文件

**moxing.frame.file.remove(\*args, \*\*kwargs)**

删除文件或目录

```python
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

```python
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

```python
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

```python
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

```python
import moxing as mox
mox.file.write('s3://bucket/dir/data.bin', b'xxx', binary=True)
```

参数：

- url - 源路径或s3 url
- data - 写入文件的内容
- binary - 是否以二进制模式写文件


***exception*`moxing.framework.file.MoxFileNameDuplicateException`**

基类: `exceptions.Exception`

当下载的目录和本地Unix OS目录或文件出现相同名称冲突时，会抛出此异常(在对象存储中允许)，提示没有相关的http响应。

```python
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

```python
import moxing as mox

try:
  mox.file.stat('s3://dls-test/not_exists')
except mox.file.MoxFileNotExistsException as e:
  print(e)
```

***exception*`moxing.framework.file.MoxFileReadException`(*\*resp*, *\*\**args*)**

基类: `moxing.framework.file.file_io._MoxFileBaseRespException`

从先前成功建立的http连接的响应流中读取块时引发异常，提示的响应状态码是OK(200)。

```python
import moxing as mox

try:
  mox.file.read('s3://dls-test/file.txt')
except mox.file.MoxFileReadException as e:
  print(e.resp)
```

***exception*`moxing.framework.file.MoxFileRespException`(*\*resp*, \*\*args)**

基类: `moxing.framework.file.file_io._MoxFileBaseRespException`

当访问s3存储发生异常时会抛出此异常，并且返回http返回码为>=300

```python
import moxing as mox

try:
  mox.file.read('s3://dls-test/xxx')
except mox.file.MoxFileRespException as e:
  print(e.resp)
```
