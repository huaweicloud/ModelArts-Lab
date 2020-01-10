# Object detection about cosmetic used by Faster RCNN.


## Dataset and Model

Dataset：https://github.com/xuyanbo03/Cosmetic-Object-Detection/tree/master/data/VOCdevkit2007/VOC2007

Model： [百度云](https://pan.baidu.com/s/1z1B-KCaQqsKoHVUFV2-vgQ)

提取码：vjfw

该模型使用Faster RCNN算法，并加载预训练权重 [resnet_v1_50.ckpt](https://github.com/xuyanbo03/Cosmetic-Object-Detection/tree/master/data/imagenet_weights) 进行训练，数据集为VOC2007格式，最终实现效果为识别护肤品（cosmetic）位置。

该model文件可在ModelArts中导入、发布、部署。


## Training
```
mv llib lib

python train-resnet.py
```

## Test
```
python test.py
```

## Results
效果图：
<p align="center">
  <img src="https://github.com/xuyanbo03/Cosmetic-Object-Detection/blob/master/doc/1.jpg">
</p>

ModelArts部署：
<p align="center">
  <img src="https://github.com/xuyanbo03/Cosmetic-Object-Detection/blob/master/doc/2.jpg">
</p>
