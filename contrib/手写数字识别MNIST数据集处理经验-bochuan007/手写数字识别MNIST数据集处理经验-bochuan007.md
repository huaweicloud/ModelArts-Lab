在http://yann.lecun.com/exdb/mnist/上可下载公开的手写体数字数据集
该数据集包括有60,000个样本的训练集和10,000个样本的测试集
但解压后的文件格式为idx-utype，主流的图片浏览器不能处理
我希望找出一个方法，将idx-utype文件里的数据分割并转为主流图片格式，如jpg、png、bmp等

[![QQ截图20190617104753](https://user-images.githubusercontent.com/9285301/59575530-76b99200-90ee-11e9-9d6a-3b31a0ff0ab7.png)](https://user-images.githubusercontent.com/9285301/59575530-76b99200-90ee-11e9-9d6a-3b31a0ff0ab7.png)

参考了一些网上已有代码：
https://www.cnblogs.com/zhouyang209117/p/6436751.html
https://blog.csdn.net/qq_20936739/article/details/82011320

idx-utype 文件，实际上用前4个数据（魔数，样本个数，样本行宽，样本列宽）表示了整个文件的图片信息，之后的无符号byte数组就是idx-utype的实际数据

以分割MNIST训练集图片包为例，这里有一个我改写的分割idx-utype文件数据并输出为多个jpg的代码示例，因为是改写的代码，注释掉了很多原代码中我不需要的语句，如果你觉得这些注释影响了你的阅读，请直接删除。（运行环境为anaconda的jupyterLab）

```python
import numpy as np     
import struct    
import matplotlib.pyplot as plt     
from PIL import Image,ImageFont,ImageDraw
import scipy.misc
import imageio

filename = 'train-images.idx3-ubyte'
binfile = open(filename,'rb')#以二进制方式打开 
buf  = binfile.read()

index = 0
magic, numImages, numRows, numColums = struct.unpack_from('>iiii',buf,index)#读取4个32 int    
print(magic,' ',numImages,' ',numRows,' ',numColums)

outputImgDir='train_image_output/'

offset = 0
fmt_header = '>iiii'
offset += struct.calcsize(fmt_header)
image_size = numRows * numRows
fmt_image = '>' + str(image_size) + 'B'
for i in range(numImages):
    im = struct.unpack_from(fmt_image,buf,offset)
    offset += struct.calcsize(fmt_image)
    index += struct.calcsize(fmt_image)
    im = np.array(im)
    im = im.reshape(28,28)
    imgdir=outputImgDir+str(i)+'.jpg'

    imageio.imwrite(imgdir, im)

binfile.close() 
```
