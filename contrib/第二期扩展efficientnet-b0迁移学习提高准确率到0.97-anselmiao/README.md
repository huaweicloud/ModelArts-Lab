# 基于efficientnet-b0的迁移学习提高猫狗识别准确率到0.97
## 1. 关于环境的说明

### 在华为云modelarts开发环境中需要基于TF-1.8.0-PYTHON3.6引擎升级，将tensorflow升级为1.14.0版本
!pip install --upgrade tensorflow-gpu
执行后需要重启笔记本引擎
#### [华为云代码](https://github.com/anselmiao/ModelArts-Lab/blob/master/contrib/%E7%AC%AC%E4%BA%8C%E6%9C%9F%E6%89%A9%E5%B1%95%E5%9F%BA%E4%BA%8Eefficientnet-b0%E7%9A%84%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E6%8F%90%E9%AB%98%E5%87%86%E7%A1%AE%E7%8E%87%E5%88%B00.97/code/dog_and_cat_efficientnetb0-hwy.ipynb)
### 作者本地环境为win10系统，tensorflow-gpu 1.14.0版本,硬件环境e5-2670,16G，nv-750ti
#### [本地代码](https://github.com/anselmiao/ModelArts-Lab/blob/master/contrib/%E7%AC%AC%E4%BA%8C%E6%9C%9F%E6%89%A9%E5%B1%95%E5%9F%BA%E4%BA%8Eefficientnet-b0%E7%9A%84%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E6%8F%90%E9%AB%98%E5%87%86%E7%A1%AE%E7%8E%87%E5%88%B00.97/code/dog_and_cat_efficientnetb0.ipynb)
### 华为云和作者环境中都可执行成功

## 2. 关于代码的说明
代码中引用谷歌EfficientNets开源模型[EfficientNets-b0](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/efficientnet-b0.tar.gz)以及[代码](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

## 3. 迁移学习代码
code文件夹中的dog_and_cat_efficientnetb0.ipynb此项目本人所写的迁移训练代码，作者本地环境为win10系统，tensorflow-gpu 1.14.0版本。
### 目录结构
data					lost+found
dog_and_cat_efficientnetb0.ipynb	main.py
efficientnet-b0				preprocessing.py
efficientnet_builder.py			__pycache__
efficientnet_model.py			README.md
eval_ckpt_example.ipynb			upgrade tensorflow 1.14.0.ipynb
eval_ckpt_main.py			utils.py
imagenet_input.py
