# Monkey 识别

## 数据集
[数据集](https://pan.baidu.com/s/139f4tm7Z7L5bwBDahK3hvA) 
提取码：2n5c 

## 文件目录
data压缩文件中含 训练集training 、 验证集validation 和 标签说明monkey_labels.txt 。
[ipynb](https://github.com/xuyanbo03/monkey_classification/)
[py code](https://github.com/xuyanbo03/monkey_classification)

## 模型
使用了ResNet50训练十种类型的猴子，用imageNet作为预训练模型，并且在原有数据上做了一定的增广，训练100轮，可达到99%，验证集为80%以上。

## 运行
python monkey_classification.py