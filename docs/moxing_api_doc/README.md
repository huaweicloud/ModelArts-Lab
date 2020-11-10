
# 0. 什么是Moxing
- MoXing是华为云ModelArts团队自研的分布式训练加速框架，它构建于开源的深度学习引擎TensorFlow、MXNet、PyTorch、Keras之上。 相对于TensorFlow和MXNet原生API而言，MoXing API让模型代码的编写更加简单，允许用户只需要关心数据输入(input_fn)和模型构建(model_fn)的代码，即可实现任意模型在多GPU和分布式下的高性能运行，降低了TensorFlow和MXNet的使用门槛。另外，MoXing-TensorFlow还将支持自动超参选择和自动模型结构搜索，用户无需关心超参和模型结构，做到模型全自动学习

# 1. MoXing-TensorFlow

### [1.1 MoXing 介绍与性能测试](MoXing_API_Introduction.md)

- MoXing-TensorFlow简介与性能测试报告

### [1.2. MoXing 文件操作](MoXing_API_File.md)

- MoXing中提供了一套文件对象API：[mox.file API document](MoXing_API_File_apidoc.md)，可以用来读写本地文件，同时也支持OBS文件系统。

### [1.3. Getting Started](MoXing_API_GettingStarted.md)

- 基于MoXing-TensorFlow的手写数字识别样例代码

### [1.4. 训练 resnet50-flowers](MoXing_API_Flowers.md)

- 基于MoXing-TensorFlow训练一个resnet50模型的案例

### [1.5. 进阶：迁移学习 resnet50-flowers](MoXing_API_FlowersAdvanced.md)

- 进阶案例，载入预训练的resnet50，在flowers数据集上做迁移学习，并使用新特性加速训练

### [1.6. MoXing超参搜索案例](MoXing_API_Hyperselector_Training.md)

- MoXing超参搜索案例，代码仅供参考

### [1.7. MoXing-TensorFlow 接口文档](MoXing_API_TensorFlow_apidoc.md)

- MoXing-TensorFlow的常用接口文档

### [1.8. MoXing-TensorFlow 更多案例](MoXing_API_MoreExamples.md)

- MoXing-TensorFlow更多案例脚本，代码仅供参考

### [1.9. MoXing-TensorFlow 详细使用手册](MoXing_API_UserInstructions.md)

- MoXing-TensorFlow详细使用手册以及一些常见错误参考
