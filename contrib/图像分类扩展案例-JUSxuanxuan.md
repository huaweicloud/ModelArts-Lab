# 图像分类的实际应用(以sofasofa题目为例)

想直接看结果的朋友可知直接看最后几段。

## [第一个问题](http://sofasofa.io/competition.php?id=6)

第一个题目即为常见的图像二分类：训练集中圆和方，测试集中圆和方，看看模型的识别率。

### 第一次尝试

尝试VGG16、Xception神经网络，结果发现效果一般般，最后VGG的val_acc=99.78，Xception的val_acc == 99.38，当然，从这次尝试有了个新思路(把keras上面那几个自带的模型都跑一下，把概率跑出来，最后求和取值)[但是有点'傻瓜行为'先不这样，等后面尝试不出来的时候再试]

## 第二次尝试
在尝试各个神经网络的时候，我发现每个神经网络的input_shape都有自己的一个最低值，那我尝试在Xception上面把最原始的71,71调到256,256，发现一个很惊奇的效果，val_acc=1 那说明验证集上效果很好~~~试一下，只用Xception，跑一下，交到sofa上试一试看看排名。
下图为Xception在input_shape=(256,256,1)时的结果[input_shape=(71,71,1)时为99.38，忘了截图]

![image](https://user-images.githubusercontent.com/50792908/67395705-04dfe080-f5d9-11e9-9549-2446767141e9.png)

下图为VGG16在input_shape=(256,256,1)时的结果[input_shape=(48,48,1)时为99.78，忘了截图]

![image](https://user-images.githubusercontent.com/50792908/67397036-2f329d80-f5db-11e9-8b2d-0e170c724c5e.png)

两个截图对比发现，vgg16的收敛性要比xception要好一点，没有那么震荡


用两个模型跑出来的结果对比一下，为1，好吧那就不做'傻瓜模型'了
![image](https://user-images.githubusercontent.com/50792908/67397619-05c64180-f5dc-11e9-9f72-36afa4286b31.png)

sofasofa上的排名也不错，百分百[排不到前面，是因为同样分数按照时间排序]，ok，简单的图像二分类完成。

![image](https://user-images.githubusercontent.com/50792908/67489367-0a562d00-f6a4-11e9-88a6-5166a42bb9d1.png)

## [第二个问题](http://sofasofa.io/competition.php?id=9)

第二个问题要难一点：训练集中圆和方，测试集中圆、方以及异形，最后看看模型的识别率。ok，这个题目的难点在于，如何在训练集中没有第三类样本的情况下，对测试集中的第三类做出精准判断，同时这个问题在数据增广方面需要将单通道灰度值转化为RGB图片。

### 第一次尝试

用第一个问题中的模型先跑一跑，假设没有第三类，这个题按照我原来图像二分类去做，结果~~不太行，分数比较低。

![image](https://user-images.githubusercontent.com/50792908/67490012-23131280-f6a5-11e9-99fa-bd5663d1b2de.png)

### 第二次尝试

既然二分类不可以，我就定义为三分类（虽然训练集中没有第三类的样本）。试一下，万一算出来的概率向量为(0.1，0.1，0.8)这样的话不就可以直接定义其为第三类么，但是，结果不行。【ps:第三类没有样本，特征向量都构造不出来，谈何其属于第三类】
【这里用到一个弱分类器，即分辨率不要选择256，256这样的很大的数值，我用的VGG16，那就选择用48*48，因为在接下来的尝试中，我需要这个模型不要分的那么清，我希望用一种模糊预测的思想解决，因此我需要弱一点的分类器】
结果如图所示，第三列为‘others’类的可能性，很明显，都很小。
![image](https://user-images.githubusercontent.com/50792908/67285007-4e60fa80-f509-11e9-8798-b331763561cb.png)

### 第三次尝试 
第三次尝试的思路，就算某照片预测为第三类的概率概率比较小，为0.04，那么其实是不是可以认为它比其余的图片更不像方圆两类？即这一张图片更像是第三类【不规则图形】，ok，那么我设置一下阈值，当第三类的可能性大于某个值的时候即判定为第三类。
嗯~~最后评分结果如图
![image](https://user-images.githubusercontent.com/50792908/67619128-634fcd80-f82a-11e9-886e-ccbe32a459b1.png)

结果不太行，当然还是方法的问题，需要再次改进方法



### 第四次尝试 
用数据增广试一下，训练集为6000张图片，一张变四张，即训练集24000张图片，感觉数量可以了。
这四张分别用高斯模糊、图像锐化、旋转+原图，ok。其中的难点就在于单通道灰度图片转化为三通道灰度图片，不过查一下相应的理论就可以。

通道转化代码如下
![image](https://user-images.githubusercontent.com/50792908/67619227-7adb8600-f82b-11e9-9a8f-801e4dd1aa6e.png)



数据增广如下
![image](https://user-images.githubusercontent.com/50792908/67619231-8169fd80-f82b-11e9-9230-39a2ac111ee4.png)



ok数据增广完毕

## 第五次尝试
###  STEP：一

依靠第三次尝试发现的思路，就算属于第三类的很小，比如0.02但是我是不是可以肯定，这张图片比测试集的其他图片更不像方圆两类？ok，画一下看一下是不是
![image](https://user-images.githubusercontent.com/50792908/67619493-fa1e8900-f82e-11e9-87ce-0e536cdf7a9c.png)

从图像上看的话，效果很好，这张图片不是，那么我把这张图片的label标位‘other’再放入到训练集中去跑，是不是就是对训练集做了一个很好的扩展，即训练集中出现了other类的模板？
把第三类的概率值取前30个最大值，感觉效果很好，基本上都是other类，
如下图所示
![image](https://user-images.githubusercontent.com/50792908/67620081-93e93480-f835-11e9-9e3e-2548313fe2b8.png)

### STEP：二
在下面一轮的训练中将这三十类other类放进去，再取第三类的概率值前50个最大的，下图中最后的10来个判别为0或1无法判断为2，但是很明显，他们属于异类，因此再重新标注label，重进注入到训练集中，再次训练
![image](https://user-images.githubusercontent.com/50792908/67620094-a7949b00-f835-11e9-8b3f-9a61d8a81e7d.png)

### STEP：三
随着不断地给训练集增添other类别的图像，一步步的扩大训练集，效果很棒，就这样手动充填几次，直到出现方圆图形为止不再扩充[至于到底扩增多少，根据实际情况而定]，传到sofasofa上看看结果。结果还不错，提高了4个百分点，但是还可以再次完善。

![image](https://user-images.githubusercontent.com/50792908/67622571-eb939a00-f84d-11e9-9a73-522144935091.png)

### STEP：四
再改进一下，用Imagenet这个权重（还是VGG16这个模型）；这个效果，没谁了

![image](https://user-images.githubusercontent.com/50792908/67629594-ec124c00-f8b2-11e9-9e50-fecce47d54e4.png)

传到sofasofa上，~~~~~ 排名不错，直接提高了10个百分点

![image](https://user-images.githubusercontent.com/50792908/67629607-077d5700-f8b3-11e9-8b55-eb4391137cff.png)

# =======所有的步骤到此结束













# 技术点总结
## 1.给出的灰度图片为单通道灰色图，如何转RGB图像
## 2.cv2中的各种数据增广函数
## 3.重新定义训练集
## 4.others类的判定
## 5.分辨率的作用

# [代码地址](https://github.com/JUSxuaxuan/image_classification_actually/blob/master/%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%AE%9E%E9%99%85%E5%BA%94%E7%94%A8.ipynb)










