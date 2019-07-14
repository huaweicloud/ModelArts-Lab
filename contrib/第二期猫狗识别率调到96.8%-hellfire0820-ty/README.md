在第二期案例中，通过观察并分析TensorBoard中loss和accuracy的变化曲线，调节batch_size（批大小）、learning_rate（学习率）、max_epochs（最大训练轮数）等超参，训练精度得到提升，达到到84%以上，但开始具有一定的波动性很难再提高，需要尝试其他的方法。
经过多次尝试，并借鉴网上他人的经验，决定采用迁移学习的方法。
迁移学习具有以下优点：
站在巨人的肩膀上：前人花很大精力训练出来的模型在大概率上会比你自己从零开始搭的模型要强悍，没有必要重复造轮子。
训练成本可以很低：后面可以看到，如果采用导出特征向量的方法进行迁移学习，后期的训练成本非常低，用 CPU 都完全无压力，没有深度学习机器也可以做。
适用于小数据集：对于数据集本身很小（几千张图片）的情况，从头开始训练具有几千万参数的大型神经网络是不现实的，因为越大的模型对数据量的要求越大，过拟合无法避免。这时候如果还想用上大型神经网络的超强特征提取能力，只能靠迁移学习。

在本扩展案例中，使用了ResNet50预训练模型，当初始化一个预训练模型时，会自动下载权重到 ~/.keras/models/ 目录下
因为是训练好的，所以我们冻结全部卷积层，这样就可以正确获得bottleneck特征，然后添加自己定制的全连接层
#猫狗分类部分，需要我们根据现有需求来新定义，
x = base_model.output
#添加自己的全链接分类层
# Dense就是常用的全连接层，所实现的运算是output = activation(dot(input, kernel)+bias)。其中activation是逐元素计算的激活函数，
#kernel是本层的权值矩阵，bias为偏置向量，只有当use_bias=True才会添加。
#如果本层的输入数据的维度大于2，则会先被压为与kernel相匹配的大小。

x = Flatten()(x)     #Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
x = Dense(1024, activation='relu')(x)  
#units：大于0的整数，代表该层的输出维度。这里为1024；
#activation：激活函数，为预定义的激活函数名，如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x），这里使用relu激活函数
x = Dropout(0.5)(x)
#为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合。
predictions = Dense(2, activation='softmax')(x)

优化器：采用较小学习率的SGD，设置学习率是1e-4
optimizer  = SGD(lr=1e-4,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
