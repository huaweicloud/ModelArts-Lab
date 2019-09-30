# 基于keras Faster R-CNN的机坪飞机图像识别

# 应用价值

长期以来，机场塔台指挥和机坪指挥中，都是通过人眼直接观察以确定飞机在机场或跑道上的位置以进行指挥。然而这存在很多问题，比如发生大雾天等不良天象时难以进行观察并进行有效的指挥，又如仅通过肉眼观察，可能对相对较远的实际的飞机位置缺乏可靠的判断，导致潜在的安全隐患等等。因此，在机场这种广纵深高像素的场景中，急需一种快速有效的识别航空器的手段，以进行航空器位置的分析或者视觉增强。本程序就旨在利用Keras fasterRCNN 实现飞机的在机坪的实时高速识别。本程序所诉场景具有较高的商业应用价值。

# 程序介绍

本程序采用resnet50进行特征提取，其余技术参考:[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, 2015](https://arxiv.org/pdf/1506.01497.pdf) <br/>

# 数据下载

数据可以在华为云中的公共obs获得，公共obs访问域名为：[obs-public01.obs.cn-north-1.myhuaweicloud.com]

# 训练方法
    本程序将tensorflow作为backend，请自行安装好对应的python包。
    首先采用Pascal VOC 数据进行预训练，数据可见[http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar]
    `python train_frcnn.py -o pascal_voc -p path/to/VOCdevkit/`
    然后进行fine tunning, 如下
    `python train_frcnn.py -p ./dataset/train_data.csv --input_weight_path ./model_frcnn.hdf5`
    
    也可以直接训练，命令如下：
    `python train_frcnn.py -p ./dataset/train_data.csv`


# 测试方法

    `python test_frcnn.py -p ./images/test/`
    程序会将预测对应文件夹下的文件，结果输出到results_imgs中

# 开源协议

本程序遵循Apache 2.0和MIT开源协议，程序参考[https://github.com/you359/Keras-FasterRCNN.git] 和 ModelArts-Lab相关内容[https://github.com/huaweicloud/ModelArts-Lab.git]
