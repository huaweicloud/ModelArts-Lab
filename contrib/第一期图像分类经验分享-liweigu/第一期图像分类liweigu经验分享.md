[课程介绍](https://shimo.im/docs/FTRMKWzZ0t01ik02/read)


[课程说明](https://github.com/huaweicloud/ModelArts-Lab/issues/49)


学习心得：


第一期实战活动介绍——图像分类（I）

  用git将 https://github.com/huaweicloud/ModelArts-Lab 拷贝（clone）到本地，然后在 /ExeML/ExeML_Flowers_Recognition/data 目录里能找到训练数据（40张）和测试数据（4张）。

  数据可以换成自己的分类数据，比如动物分类、交通标志分类等等，图片分类差异越大、内容干扰信息越少、数据越多，则训练出的模型会更好。


第二期实战活动介绍——图像分类（II）


  猫狗识别训练集是用代码下载到notebook运行环境本地的。

  dogcat_model是模型定义。由于是做二分类，因此激活函数选用'sigmoid'，损失函数使用'binary_crossentropy'。

  防止过拟合，除了使用了early stopping策略，还应考虑扩充训练数据集。
