#基于efficientnet-b0的迁移学习提高猫狗识别准确率到0.97
## 1. 关于环境的说明

### 在华为云modelarts开发环境中需要基于TF-1.8.0-PYTHON3.6引擎升级，将tensorflow升级为1.14.0版本
!pip install --upgrade tensorflow-gpu
执行后需要重启笔记本引擎

### 作者本地环境为win10系统，tensorflow-gpu 1.14.0版本,硬件环境e5-2670,16G，nv-750ti

### 华为云和作者环境中都可执行成功

## 2. 关于代码的说明
代码中引用谷歌EfficientNets开源模型EfficientNets-b0以及代码。

## 3. 迁移学习代码
code文件夹中的dog_and_cat_efficientnetb0.ipynb此项目本人所写的迁移训练代码，作者本地环境为win10系统，tensorflow-gpu 1.14.0版本。
