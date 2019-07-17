# 基于efficientnet的迁移学习提高猫狗识别准确率到0.98

## 1. 关于环境的说明

### 在华为云modelarts开发环境中需要基于Multi-Engine-python3.6引擎，tensorflow版本为1.13.1版本
#### 在notebook中执行以下语句

#### (此代码安装efficientnet库，基于谷歌efficientnet重新实现的Keras版本)

* !pip install https://github.com/qubvel/efficientnet/archive/master.zip 

#### (下面是环境需要改动的库，keras2.24)

* !pip install scikit-image
* !pip install --upgrade keras
* !pip install numpy==1.15.0

## 2. 关于代码的说明

#### 引用代码说明
谷歌EfficientNets开源项目[位置](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
此项目所引用的基于开源项目实现的EfficientNet-Keras[位置](https://github.com/qubvel/efficientnet)

本人之前有直接引用谷歌代码，tensorflow版本需要1.14.0需要cuda10，由于华为云上最高版本现为1.13.1版本使用cuda9
导致本人在华为云上升级tensorflow1.14.0后执行效率很低，故这里改为引用他人所改动的keras版本代码(函数式api所写，
便于打印结构),谷歌原版核心模型代码基于keras的Subclassed Models,[模型结构定义于call内不能直接打印模型结构](https://colab.research.google.com/drive/172D4jishSgE3N7AO6U2OKAA_0wNnrMOq#scrollTo=mJqOn0snzCRy)
可以通过模型中实现的endpoints访问层结构。

#### 迁移学习代码
code文件夹中的了[b0_hwy.ipynb](https://github.com/anselmiao/ModelArts-Lab/blob/master/contrib/%E7%AC%AC%E4%BA%8C%E6%9C%9F%E7%8C%AB%E7%8B%97%E5%87%86%E7%A1%AE%E7%8E%87%E5%88%B098%25%E5%9F%BA%E4%BA%8Eefficientnet%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0-anselmiao/code/b0_hwy.ipynb)(基于efficientnet-b0)为此项目本人所写的迁移训练代码，[b5_hwy.ipynb](https://github.com/anselmiao/ModelArts-Lab/blob/master/contrib/%E7%AC%AC%E4%BA%8C%E6%9C%9F%E7%8C%AB%E7%8B%97%E5%87%86%E7%A1%AE%E7%8E%87%E5%88%B098%25%E5%9F%BA%E4%BA%8Eefficientnet%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0-anselmiao/code/b5_hwy.ipynb)(基于efficientnet-b5)。
本人测试b0的收敛速度很快，b5相比前者很慢，20轮在最后的测试集上准确率还小于b0,b5的finetune参数较多是其收敛速度慢的主因，
不过b5loss值一直在减小，相信最终准确率会更高。

## 3. Efficientnet系列模型简介

EfficientNet是谷歌最新的论文：
[1] Mingxing Tan and Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019. Arxiv link: https://arxiv.org/abs/1905.11946.

谷歌基于AutoML开发了EfficientNets，这是一种新的模型缩放方法。
ImageNet测试中实现了84.1%的准确率，再次刷新了纪录。
虽然准确率只比之前最好的Gpipe提高了0.1%，但是模型更小更快，参数的数量和FLOPS都大大减少，效率提升了10倍！

Efficientnet在较高准确率下有着更少的参数更快的执行效率，使其非常适合部署于终端设备，例如手机等。


