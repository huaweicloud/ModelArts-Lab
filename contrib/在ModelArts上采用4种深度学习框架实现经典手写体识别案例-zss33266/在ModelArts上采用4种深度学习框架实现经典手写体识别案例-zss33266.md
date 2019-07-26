**在ModelArts上采用4种深度学习框架实现经典手写体识别案例**  
Author:zss33266  
Date:2019-07-08  
Update:2019-07-11

华为云ModelArts支持多种主流开源的深度学习框架，作为图像识别入门的经典的手写体识别案例，华为云官网帮助文档已经做了不同版本的实现，在本次ModelArts-Lab库official_examples中也有详细的操作文档，具体实现步骤就不在详细描述，以下主要是我对主流的几个深度学习框架和手写体识别案例关键步骤做一个归纳汇总：

**模型训练**  
一、	MXNet
	

> 框架介绍：
MXNet是DMLC（Distributed Machine Learning Community）开发的一款开源的、轻量级、可移植的、灵活的深度学习库，它让用户可以混合使用符号编程模式和指令式编程模式来最大化效率和灵活性，目前已经是AWS官方推荐的深度学习框架。MXNet的很多作者都是中国人，其最大的贡献组织为百度，同时很多作者来自cxxnet、minerva和purine2等深度学习项目，可谓博采众家之长。它是各个框架中率先支持多GPU和分布式的，同时其分布式性能也非常高。

在ModelArts中采用MXNet框架实现手写体识别的几个核心步骤：
1.	准备数据集，可以直接在模型市场导入到自己的数据集中，存储路径需要事先准备一个桶，如图所示： 
![1](https://user-images.githubusercontent.com/52277737/60807291-db30b400-a1b7-11e9-9b2c-c896eea3a3a3.png)
![image2](https://user-images.githubusercontent.com/52277737/60807422-3cf11e00-a1b8-11e9-8e89-b63c3a40a9f5.png)
![image3](https://user-images.githubusercontent.com/52277737/60807432-4c706700-a1b8-11e9-9d6b-07bee35505d9.png)

2.	创建MXNet训练作业需要注意的地方，数据来源就是我们从市场导入的数据集，算法来源选择MXNet和对应的Python版本，可以选择2.x也可以选择3.x,代码目录和启动文件一定要事先在桶中创建好，运行参数num_epochs为最大训练的批次数，可以根据实际情况来填写，默认为10，这里定义为8，训练输出路径也需要在桶中提前创建好，用于存放训练好的模型。 
![image4](https://user-images.githubusercontent.com/52277737/60807446-54c8a200-a1b8-11e9-8929-ec6fb0e8a0a9.png)
![image5](https://user-images.githubusercontent.com/52277737/60807453-58f4bf80-a1b8-11e9-9109-23ef06e0efe7.png)
![image6](https://user-images.githubusercontent.com/52277737/60807457-5b571980-a1b8-11e9-90f1-21403aa976b5.png)
3.	查看训练作业结果，本次创建的训练资源采用的4个GPU,P100的高性能计算实例，训练速度快到飞起来，虽然每小时30元左右，但最终用了不到1分钟的时间，完成了手写体识别模型的训练，花费不足1元钱，通过日志可以看到，最后训练的精度达到了97%左右。
![image7](https://user-images.githubusercontent.com/52277737/60807462-5db97380-a1b8-11e9-9d08-8117952e86c6.png)

以上是我个人实验的一些关键步骤，具体详细的操作点击：
[官方Git操作指南](https://github.com/yepingjoy/ModelArts-Lab/tree/master/offical_examples/Using_MXNet_to_Create_a_MNIST_Dataset_Recognition_Application)|
[官方帮助文档](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0009.html)

二、	Tensorflow

> 框架介绍：
TensorFlow最初是由Google Brain Team的研究人员和工程师开发的。其目的是面向深度神经网络和机器智能研究。自2015年底以来，TensorFlow的库已正式在GitHub上开源。TensorFlow对于快速执行基于图形的计算非常有用。灵活的TensorFlow API可以通过其GPU支持的架构在多个设备之间部署模型。TensorFlow拥有产品级的高质量代码，有Google强大的开发、维护能力的加持，整体架构设计也非常优秀。相比于同样基于Python的老牌对手Theano，TensorFlow更成熟、更完善，同时Theano的很多主要开发者都去了Google开发TensorFlow（例如书籍Deep Learning的作者Ian Goodfellow，他后来去了OpenAI）。Google作为巨头公司有比高校或者个人开发者多得多的资源投入到TensorFlow的研发，可以预见，TensorFlow未来的发展将会是飞速的，可能会把大学或者个人维护的深度学习框架远远甩在身后。

在ModelArts中采用TensorFlow框架实现手写体识别的几个核心步骤：
1.	数据集已通过市场导入，无须再重复导入
2.	创建TensorFlow训练作业需要注意的地方，算法来源选择TF1.8-python2.7,代码目录和启动文件一定要事先在桶中创建好，同样训练输出路径也需要在桶中提前创建好，用于存放训练好的模型。
![image8](https://user-images.githubusercontent.com/52277737/60807542-90fc0280-a1b8-11e9-91aa-ae619e668fd8.png)
![image15](https://user-images.githubusercontent.com/52277737/60807551-93f6f300-a1b8-11e9-9ed2-a43fc599aa30.png)
3.	查看训练作业结果，同样采用4个GPU,P100的高性能计算实例，用时1分24秒，通过日志可以看到，最后训练的精度达到了91%左右，可以通过调整代码算法继续训练提升精度 。 
![image10](https://user-images.githubusercontent.com/52277737/60807647-cc96cc80-a1b8-11e9-88b4-0d71eb927537.png)


具体详细的操作点击：
[官方Git操作指南](https://github.com/yepingjoy/ModelArts-Lab/tree/master/offical_examples/Using_TensorFlow_to_Create_a_MNIST_Dataset_Recognition_Application)|
[官方帮助文档](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0010.html)


三、	Caffe

> 框架介绍：
Caffe 全称为 Convolutional Architecture for Fast Feature Embedding，是一个被广泛使用的开源深度学习框架（在 TensorFlow 出现之前一直是深度学习领域 GitHub star 最多的项目)，目前由伯克利视觉学中心（Berkeley Vision and Learning Center，BVLC）进行维护。Caffe 的创始人是加州大学伯克利的 Ph.D.贾扬清，他同时也是TensorFlow的作者之一，曾工作于 MSRA、NEC 和 Google Brain，目前就职于 Facebook FAIR 实验室。Caffe 的主要优势包括如下几点。
- 容易上手，网络结构都是以配置文件形式定义，不需要用代码设计网络。
- 训练速度快，能够训练 state-of-the-art 的模型与大规模的数据。
- 组件模块化，可以方便地拓展到新的模型和学习任务上。

在ModelArts中采用Caffe框架实现手写体识别的几个核心步骤：
1.	数据集已通过市场导入，无须再重复导入
2.	创建Caffe训练作业需要注意的地方，算法来源选择Caffe1-python2.7,代码目录和启动文件一定要事先在桶中创建好，目录本次需要上传到codes的文件有三个，“train.py”：训练脚本，“lenet_solver.prototxt”：配置训练时参数的prototxt文件和lenet_train_test.prototxt”：定义网络结构的prototxt文件，且必须使用命名为“codes”文件目录。
 ![image11](https://user-images.githubusercontent.com/52277737/60807735-0e277780-a1b9-11e9-96ca-5b0d39220f50.png)
![image12](https://user-images.githubusercontent.com/52277737/60807759-1b446680-a1b9-11e9-9ebb-cbff749cd597.png)

 
3.	查看训练作业结果
 ![image13](https://user-images.githubusercontent.com/52277737/60807771-24cdce80-a1b9-11e9-9d42-ffbfe7b3fddf.png)

具体详细的操作点击：
[官方Git操作指南](https://github.com/huaweicloud/ModelArts-Lab/blob/master/official_examples/Using_Caffe_to_Create_a_MNIST_Dataset_Recognition_Application/)|
[官方帮助文档](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0011.html)



四、	Moxing

> 框架介绍：
MoXing是华为云ModelArts团队自研的分布式训练加速框架，它构建于开源的深度学习引擎TensorFlow、MXNet、PyTorch、Keras之上。 相对于TensorFlow和MXNet原生API而言，MoXing API让模型代码的编写更加简单，允许用户只需要关心数据输入(input_fn)和模型构建(model_fn)的代码，即可实现任意模型在多GPU和分布式下的高性能运行，降低了TensorFlow和MXNet的使用门槛。另外，MoXing-TensorFlow还将支持自动超参选择和自动模型结构搜索，用户无需关心超参和模型结构，做到模型全自动学习。

在ModelArts中采用MoXing框架实现手写体识别的几个核心步骤：
1.	数据集已通过市场导入，无须再重复导入
2.	创建MoXing训练作业，代码目录和启动文件一定要事先在桶中创建好，引擎可以选择TensorFlow，MoXing框架是构建于之上的，支持调用内部各种算法。
![image14](https://user-images.githubusercontent.com/52277737/60808097-1338f680-a1ba-11e9-8ef4-5272d13b6109.png)
![image15](https://user-images.githubusercontent.com/52277737/60808139-251a9980-a1ba-11e9-9278-6bc134ef4c7b.png)
3.	查看训练作业结果，同样采用4个GPU,P100的高性能计算实例，用时1分14秒，比之前的原生TensorFlow引擎要快一点，通过日志可以看到，最后训练的精度达到了88%左右，可以通过调整代码算法继续训练提升精度 。
![image16](https://user-images.githubusercontent.com/52277737/60808155-2e0b6b00-a1ba-11e9-8928-fe4b849925b0.png)
 
具体详细的操作点击：
[官方Git操作指南](https://github.com/huaweicloud/ModelArts-Lab/tree/master/official_examples/Using_MoXing_to_Create_a_MNIST_Dataset_Recognition_Application)|
[官方帮助文档](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0007.html)|[MoXing详细介绍地址](https://github.com/huaweicloud/ModelArts-Lab/blob/master/docs/moxing_api_doc/MoXing_API_Introduction.md)

以上四个采用不同框架的手写体训练案例全部训练完成： 
![image17](https://user-images.githubusercontent.com/52277737/60808269-70cd4300-a1ba-11e9-852d-effd24e7011c.png) 


**模型优化**  
上面的例子中，我们看看能否优化一下训练代码，提升模型的准确度，在这里我们以TensorFlow的训练模型为例，我在原有的训练代码上做了一些简单的调整，并加了注释，方便大家更深入理解这个例子：
![image](https://user-images.githubusercontent.com/52277737/60971973-e6224a80-a357-11e9-9eae-09495b272467.png)
[调整过的训练代码我已经上传到Github,点击查看](https://github.com/zss33266/ModelArts-Lab/blob/master/contrib/%E5%9C%A8ModelArts%E4%B8%8A%E9%87%87%E7%94%A84%E7%A7%8D%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E5%AE%9E%E7%8E%B0%E7%BB%8F%E5%85%B8%E6%89%8B%E5%86%99%E4%BD%93%E8%AF%86%E5%88%AB%E6%A1%88%E4%BE%8B-zss33266/train_mnist_tf_optimized.py)  
模型调优的过程如下：
1. 将原有的代价函数改为交叉熵代价函数，学习率learning_rate=0.5
![image](https://user-images.githubusercontent.com/52277737/60990216-80938580-a37a-11e9-8aa3-e55346302bb2.png)  
最后训练的准确率为90.4%左右
![image](https://user-images.githubusercontent.com/52277737/60988713-c3ebf500-a376-11e9-8a6d-76d0b4cf1d30.png)  
2. 通过增加训练次数，将训练次数'max_steps', 1000，增加到1500
![image](https://user-images.githubusercontent.com/52277737/60988372-0d881000-a376-11e9-8c26-c14246239b16.png)  
最后训练的准确率为91%左右，有一点点提升 
![image](https://user-images.githubusercontent.com/52277737/60990278-a4ef6200-a37a-11e9-9aca-41b7f6ee3c5d.png)
3. 将梯度下降优化器GradientDescentOptimizer改为AdamOptimizer
![image](https://user-images.githubusercontent.com/52277737/60988466-41633580-a376-11e9-9de9-681f326b9c9b.png)  
通过日志发现，准确率上升到50%左右之后，结果不升反而开始下降了，有可能是原有的学习率0.5过高，错过了梯度最低点，最后准确率只有43%
![image](https://user-images.githubusercontent.com/52277737/60990386-e5e77680-a37a-11e9-9a9f-3dd86eed5351.png)  
Note：
#初始化全局变量sess.run(tf.global_variables_initializer()) 这段代码要从前面移到AdamOptimizer之后；
日志如果无法看到最后的训练结果，直接下载到本地记事本中查看。
这里要说明一下的是，不管用哪一种优化器算法 ，我们的优化器目的是为了找到模型的最佳参数w,b，不断的缩小预测值与实际标签值的误差，怎么样让loss值最小，看下面这张图：  
![image](https://user-images.githubusercontent.com/52277737/60990817-d61c6200-a37b-11e9-921a-2b1bbc64a335.png)  

打个比方：小明现在要从一个山顶去往山下，小明需要找到最低点，但是现在他迷路了，他的每一次行动目的都是找到下山最近的路，而每一次寻找都是一个不断试错的过程，为了让这个试错的成本最小化，他需要一个指南针（可以理解为优化器Optimizer)，这个指南针会指引小明到达最低点，他不停的沿着指明的方向前进，大步向前走，或者小步向前走，按照一定的步频往山下走，这个过程可以理解为学习率（learning_rate），每次往前面走一点，小明离最低的位置就越近，直到找到最低点就成功了（如果迈出去的步子大了，有可能错过最低点）  
4. 调整AdamOptimizer的学习率  
我们先看一下Adam优化器算法的参数说明    
![image](https://user-images.githubusercontent.com/52277737/60990526-35c63d80-a37b-11e9-90fb-4e121fca3323.png)  
其中learning_rate的初始值为0.001，我们按照这个学习率再训练一次：    
![image](https://user-images.githubusercontent.com/52277737/60992920-0239e200-a380-11e9-81c7-a755156bce4d.png)  
训练的准确率又回到了90%左右     
![image](https://user-images.githubusercontent.com/52277737/60993849-d61f6080-a381-11e9-849e-7d2df46ee329.png)  
再把学习率调高一个量级试下，learning_rate = 0.01  
![image](https://user-images.githubusercontent.com/52277737/60988422-2a244800-a376-11e9-9f76-08e5c0a92eae.png)  
训练的准确率为92%左右，比上一次有提升  
![image](https://user-images.githubusercontent.com/52277737/60993284-b50a4000-a380-11e9-81de-1f38b5d7ea00.png)  
5. 经验小结  
我们通过尝试调整不同的代价函数算法、训练次数和学习率，最终将模型的精度提升了1%~2%左右，小结如下：
- 可以尝试不同的Optimizer算法来改善模型的精度或者训练时间。
- 通过增加训练迭代次数max_steps，大部分情况下是可以提升模型精度的。
- 学习率learning_rate并不是越大或者越小，模型的精度就越高，要根据具体的模型算法和相关参数来决定，在不熟悉算子的提前下，可以先按照官方的默认值训练一次，然后在根据实际情况进行调整。
- 如果反复调整参数都无法大幅度提升模型的精度，就要尝试改造模型网络结构了，比如加一些卷积层，全连接层，隐藏层或dropout层来提升结果准确率。本例子可以将模型精度提升到96%以上，具体就不详细描述（在猫狗案例中，已有Vgg16和ResNet神经网络的具体实现）

**导入模型**
1.	上传模型配置和推理服务文件config.json、customize_service.py，一定要上传到之前训练作业的输出目录位置，如果训练了多次，会产生多个版本号，选择上传要导出的模型版本目录。
 ![image18](https://user-images.githubusercontent.com/52277737/60808296-7cb90500-a1ba-11e9-8587-5bbf6bc834df.png)

2.	在导入模型中选择刚才上传过配置文件的训练作业，如果推理代码配置选项已经自动生成了地址，证明推理文件已生效。
![image19](https://user-images.githubusercontent.com/52277737/60808302-804c8c00-a1ba-11e9-8ade-752b391be2ed.png)

**部署在线服务**
1.	在模型管理中可以看到刚才创建好的模型，在列表操作栏中选择部署“在线服务”
 ![image20](https://user-images.githubusercontent.com/52277737/60808322-922e2f00-a1ba-11e9-9946-dfa36eb90ddf.png)
2.	部署服务完成之后，上传一张测试图片，成功预测手写体数字7。
![image22](https://user-images.githubusercontent.com/52277737/60808422-d4f00700-a1ba-11e9-8a8e-d0942bb78a8a.png) 

**使用华为云ModelArts做手写体识别模型总结：**
1.	数据管理快：
数据集可以通过市场公开的数据直接导入，数据准备效率大大提升
2.	模型训练快：
华为云ModelArts将常用的深度学习框架，从分布式加速层抽象出来，形成一套通用框架——MoXing。不仅满足了不同方向的AI开发者，同时也能通过华为自研的硬件、软件和算法协同优化来实现训练加速。
3.	模型部署快：
ModelArts 可以一键部署模型，推送模型到所有云、端、边的设备上，云上的部署还支持在线和批量推理，满足大并发和分布式等多种场景需求
