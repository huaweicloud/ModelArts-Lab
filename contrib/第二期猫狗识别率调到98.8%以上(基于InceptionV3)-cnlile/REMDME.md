# 第二期拓展内容-提高猫狗识别率模型的更高精度

    以猫狗识别netebook案例在notebook中调试，进行代码调试。根据附加题提供的思路，1、调整参数；2、采用其它开源神经网络结构。先后用resnet50和IncepitonV3 两种方式进行是了实验，resNet50精度可以到97.1%以上，IncepitonV3精度可以达到98.8%以上。以下以IncepitonV3做说明。

## 调节超参
  batch_size = 20
  learning_rate = 1e-4 
  max_epochs = 50
  以上结果是观察后，得出的不错的选择。epochs 在39进入早停，第一次实验时候设置的是20轮，最后精度是98.5%。

## 神经网络结构IncepitonV3
  InceptionV3 是比较经典网络，我直接按照keras 的文档的推荐的函数InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, classes=1000)，模型输入尺寸调到299x299，权重训练自ImageNet，优化器用RMSprop。这样就差不多了。
  后来还进行了一些改动，证明还是这个推荐的参数比较好，自己改来改去的效果不如kears文档推荐的。

## 总结
  相对于resnet50，IncepitonV3训练时间长很多，轮数也多，当然效果也好。resnet50需要改的参数更多（尤其优化器对resnet50的影响比较大，还要改进激活函数），速度更快，最后结果差别其实也在1%左右。
​    





 

