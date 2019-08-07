在自定义学习衰减率的过程中，训练出现了严重的欠拟合，导致无法继续训练。
loss/acc基本是一条直线，acc值接近于0，loss值很大。

由于选择了VGG16网络，不方便在网络的深度宽度等方面进行修改。因为要加强自身对DP参数的调节，预训练权重/最优权重是无法达到对参数的最佳理解，因此改变网络要改善这个问题，我从下面几个方面进行了调整：
1. 激活函数
2. 优化器
3. 学习率

在这个案例中网络的初始化权重选择了none，其实，网络的初始化权重也有很多的选择，也可以缓解欠拟合的问题。比如全零初始化 Zeros、全1初始化 Ones、初始化为固定值value  Constant、随机正态分布初始化 RandomNormal、随机均匀分布初始化 RandomUniform、截尾高斯分布初始化 TruncatedNormal、VarianceScaling、用随机正交矩阵初始化Orthogonal、使用单位矩阵初始化 Identiy、LeCun均匀分布初始化方法 lecun_uniform、LeCun正态分布初始化方法 lecun_normal、Glorot正态分布初始化方法 glorot_normal、Glorot均匀分布初始化 glorot_uniform、He正态分布初始化 he_normal、LeCun均匀分布初始化 he_uniform。 这里没有实践。如果想要理解每个的效果，最好还是都试一下，毕竟DP是一个偏工程性的学科。

激活函数：sigmoid、softmax只适用于输出层，其它层建议用ReLu。这个在实践中效果微乎其微。不晓得是不是我运行的代码有问题，为啥别人在源码的基础上啥都没改就可以跑得很好，实在没想明白。

优化器： 尝试各种不同优化器后，可以解决欠拟合的问题，但是又引来了新的问题——收敛缓慢。在测试了sgd优化器后，发现运行到epoch=63时，acc=0.81；改为adadelta后，在epoch=43时，acc就已经达到0.81。
      经过探索发现常见的优化器的收敛效率遵循以下规律：
      sgd<Momentum<NAG<Adagrad<Adaelta<Rmsprop
      ![](https://src.ailemon.me/blog/2018/20180409-opt2.gif)

学习率：设定好初始学习率后，学习率的衰减对拟合过程影响很大，可以设置下面两种方式综合放入callback中，ReduceLROnPlateau+LearningRateScheduler，当学习停止时，模型总是会受益于降低2-10倍的学习率，此时就降低学习率，也可以设置学习率跟轮索引相关的进行衰减，以达到合适的方法。
def step_decay(epoch,lr=lr_base):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate
lr_reducer  = ReduceLROnPlateau(monitor="val_acc", factor=0.1, patience=3, verbose=1, mode="auto", min_lr=0)
lr_scheduler = LearningRateScheduler(step_decay)
callbacks = [es,cp,lr_reducer,lr_scheduler]


在LearningRateScheduler中可以设置多种不同的衰减方式，可以都运行一下，看看对训练收敛的影响有何不同，如下所示为一些自定义方法：
 if mode is 'power_decay':
        # original lr scheduler
        lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    if mode is 'exp_decay':
        # exponential decay
        lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
    # adam default lr
    if mode is 'adam':
        lr = 0.001
    if mode is 'progressive_drops':
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * epochs:
            lr = 0.0001
        elif epoch > 0.75 * epochs:
            lr = 0.001
        elif epoch > 0.5 * epochs:
            lr = 0.01
        else:
            lr = 0.1

