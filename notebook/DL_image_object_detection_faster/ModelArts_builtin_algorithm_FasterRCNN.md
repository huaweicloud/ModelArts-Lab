# ModelArts预置算法FasterRCNN实践

我们不仅可以在notebook里面自己开发实现FasterRCNN算法，同时，ModelArts也有预置的FasterRCNN目标检测算法。在本次任务里，我们可以体验一下。

模型详细介绍见[《Faster RCNN模型简介》](https://github.com/huaweicloud/ModelArts-Lab/wiki/Faster-RCNN%E6%A8%A1%E5%9E%8B%E7%AE%80%E4%BB%8B)

在第四期的任务中，我们已经体验了ModelArts的预置Yolo V3算法，学会了如何使用ModelArts预置算法，而且本次任务也是目标检测，所以在本次任务中，我们不给出详细的步骤，大家模仿第四期的[Yolo V3预置算法案例](../DL_image_object_detection_yolo/ModelArts物体检测Yolo_V3预置算法案例.md)的操作步骤，完成FasterRCNN预置算法的体验过程。下面给一下简单的提示。

## 准备数据

使用PASCAL VOC2007 数据集

## 创建训练任务

预置算法列表中选择`Faster_RCNN_ResNet_v1_50 ` 

运行参数使用默认的运行参数，我们建议更改`max_epoches`为3，训练3轮，大概需要一个小时左右的训练时间。如果想要提高模型的精度，可以适当增加`max_epoches`。

## 部署服务

推理测试可以自己在网上找一些有趣的图片测试 （物体类别要在PASCAL VOC2007数据集的覆盖范围内）。



