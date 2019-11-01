CRNN（CNN + RNN）
OCR（光学字符识别）由文本本地化+文本识别组成。（文本本地化找到字符的位置，然后文本识别读取字母。）

您可以使用我研究过的文本本地化模型。

执行本地化后，每个文本区域都将被裁剪并用作文本识别的输入。文本识别的示例通常是CRNN

将文本检测器与CRNN结合使用，可以创建端到端运行的OCR引擎。

神经网络
CRNN是一个将CNN和RNN组合在一起以处理包含序列信息（例如字母）的图像的网络。

它主要用于OCR技术，具有以下优点。

端到端的学习是可能的。
由于LSTM的输入和输出序列的大小不受限制，因此可以处理任意长度的序列数据。
不需要检测器或裁剪技术来逐个查找每个字符。
您可以将CRNN用于OCR，车牌识别，文本识别等。这取决于您正在训练的数据。

我使用了原始CRNN模型的略微修改版本。（输入大小：100x30-> 128x64及以上的CNN层）

网络
CRNN网络

![](https://raw.githubusercontent.com/qjadud1994/CRNN-Keras/master/photo/Network.jpg)

卷积层
通过CNN层（VGGNet，ResNet ...）提取特征。

循环层
将要素拆分为一定的大小，然后将其插入到双向LSTM或GRU的输入中。

转录层
使用CTC（连接器时间分类）将特定于特征的预测转换为标签。

使用CRNN的车牌识别
我使用CRNN识别韩国的车牌。

![](https://raw.githubusercontent.com/qjadud1994/CRNN-Keras/master/photo/license%20plate.jpg)

车牌类型

我学习了以下几种韩国车牌。

我为缺少车牌图片的人更新了韩国车牌合成图像生成器。

结果

![](https://raw.githubusercontent.com/qjadud1994/CRNN-Keras/master/photo/result.jpg)

结果

CRNN可以很好地用于车牌识别，如下所示。

训练方法
首先，您需要裁剪很多车牌图像。
在我的情况下，我用图像文件名表示了车牌号。
（牌照号码1234表示为“ 1234.jpg”）。
（如果需要，也可以使用txt或csv文件定义标签。[[ex）0001.jpg“ 1234” \ n 0002.jpg“ 0000” ...）

由于使用韩国车牌，因此我在车牌上用英语表达了韩语。

例 
（示例）A18sk6897 
A：서울 
sk：나

以这种方式创建训练数据后，将其放在“ DB / train”目录中并运行training.py。

文件描述
操作系统：Ubuntu 16.04.4 LTS

GPU：GeForce GTX 1080（8GB）

的Python：3.5.2

Tensorflow：1.5.0

凯拉斯：2.1.3

CUDA，CUDNN：9.0、7.0

文件	描述
型号.py	使用CNN（VGG）+双向LSTM的网络
型号_GRU。py	使用CNN（VGG）+双向GRU的网络
Image_Generator。py	用于训练的图像批处理生成器
参数。py	CRNN中使用的参数
训练。py	CRNN培训
预测。py	CRNN预测
