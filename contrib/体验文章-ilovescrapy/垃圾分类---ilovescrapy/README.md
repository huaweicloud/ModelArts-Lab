# huaweicloud_garbage_classify

# nets
包含vgg16，resnet50，senet50

# metrics
包含各种softmax的改进，时间原因无法把前三个用于分类；NormFace可以应对类别数据不平衡，不过目前效果和softmax差不多。

# others
基于baseline的修改，加了一些tricks，比如label smooth，center crop等等，待你们发现
对了，测试的地方增加了random crop，然后集成，能提升1%
