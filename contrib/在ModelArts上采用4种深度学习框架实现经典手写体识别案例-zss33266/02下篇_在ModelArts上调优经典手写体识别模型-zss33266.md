
**在ModelArts上进行经典手写体识别模型调优**  
Author:zss33266  
Date:2019-07-11  

在上一篇中我们在ModelArts实现了4种深度学习框架完成了手写点识别模型，本文我们来看看能否优化一下训练代码，提升模型的准确度，在这里我们以TensorFlow的训练模型为例，我在原有的训练代码上做了一些简单的调整，并加了注释，方便大家更深入理解这个例子：
![image](https://user-images.githubusercontent.com/52277737/60971973-e6224a80-a357-11e9-9eae-09495b272467.png)
[调整过的训练代码文件train_mnist_tf_optimized.py我已经上传到Github,点击查看](https://github.com/zss33266/ModelArts-Lab/blob/master/contrib/%E5%9C%A8ModelArts%E4%B8%8A%E9%87%87%E7%94%A84%E7%A7%8D%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E5%AE%9E%E7%8E%B0%E7%BB%8F%E5%85%B8%E6%89%8B%E5%86%99%E4%BD%93%E8%AF%86%E5%88%AB%E6%A1%88%E4%BE%8B-zss33266/train_mnist_tf_optimized.py)  
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
