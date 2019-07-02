​	在本次实践中，将通过resnet预训练模型来进行猫狗图像识别的模型训练。
	相比之前实验中采用的VGG16网络，resnet网络要更加复杂，模型的参数数量要更多，如果仅通过已有的25000张图片来训练resnet网络，是远远不够的，幸运的是，resnet网络也提供了若干预训练模型，来加速网络的训练和提升精度。
	所谓的预训练模型，就是实现使用大型的图像分类数据集完成了对模型的训练，并将训练后的模型参数保存下来。在我们的训练过程中，就可以在创建模型时直接导入这些已经训练好的模型参数，然后再对模型的某些层展开训练（这里一般指的是模型最后的全连接层）。
采用预训练模型，能够充分利用模型在大型数据集上已经提取出来的图像特征，使得即便只有少量的训练数据，也能够达到很高的训练精度，训练速度也大大提高。
接下来。我们就对我们的程序进行改造，具体代码如下：
只需将原有程序中模型创建的这部分代码：

![vgg创建](https://user-images.githubusercontent.com/50704594/60116125-0673d600-97aa-11e9-8cd3-4b4be14b0384.PNG)

​	改为以下代码：

```
# 首先，添加ResNet50依赖

from keras.applications.resnet50 import ResNet50

# 接下来，导入预训练模型

base_model = ResNet50(weights='imagenet', include_top=False, pooling=None, input_shape=(ROWS, COLS, CHANNELS), classes=2)

# 然后，冻结base_model所有层，这样在训练时，这些层的参数不会再参与训练

for layer in base_model.layers:
    layer.trainable = False
x = base_model.output

# 接下来，添加自己的全链接分类层

x = Flatten()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

# 最终，获得最终模型

model = Model(inputs=base_model.input, outputs=predictions)
```


​	在这里，我们采用的是resnet在imagenet这个数据集上训练得到的预训练模型。Imagenet数据集有1400多万幅图片，涵盖2万多个类别。当使用ResNet50()导入模型时，程序会自动联网到github上下载预训练模型（这里我得吐槽一句，为啥我在本机直接从github上下载预训练模型就只有不到15kb的下载速度，而NoteBook就几乎是瞬间下载…）。
程序的其他部分不变，我们再次展开实验

![resnet训练精度](https://user-images.githubusercontent.com/50704594/60116142-0ffd3e00-97aa-11e9-9028-37a3f8a02f2c.PNG)

​	我们可以发现，模型在训练集和验证集上的精度，均得到了极大提高
	最终，在测试集上的精度如下：

 ![resnet测试精度](https://user-images.githubusercontent.com/50704594/60116154-155a8880-97aa-11e9-8671-a2dc8c8f7cfe.PNG)

​	可以看到，最终的模型在测试集上的分类精度高达96%，几乎是完美完成了分类任务。
由此，可以总结如下：采用常用模型的预训练模型，可以大大缩短模型训练的时间，提高模型的精度。在训练数据不充分，计算资源和开发时间有限的情况下，采用预训练模型是我们的最佳选择。