# 图像分类模型参数&网络调优

基于第二期[图像分类任务](../DL_image_recognition/image_recongition.ipynb)，本页面包含了图像分类任务的参数调优和网络优化的相关技巧。



### 数据集

猫狗二分类数据集（25000张图片）



### 主要模型

VGG16, VGG19, ResNet



### 实践案例

[0. epochs和callbacks](./00_epoch_callbacks.ipynb)：介绍训练轮数，拟合问题以及Keras模型训练时的Callback回调函数

[1. 学习率和优化器](./01_lr_opt.ipynb)：介绍学习率和不同优化器在模型训练中的实践

[2. 数据增广](02_data_augumentation.ipynb)：使用Keras DataGenerator预处理数据，实现数据增广

[3. 使用预训练权重](03_pretrained_weights.ipynb)：使用预训练的参数权重文件进行迁移学习
