# 华为云垃圾分类大赛心得与案例分享（追加）
赛事链接如下[https://developer.huaweicloud.com/competition/competitions/1000007620/introduction]
## 前言
上一篇且说道基于ResNet架构，通过Resnet152，Resnet50多加几个后接的全连接层，以及全开训练加训练调度的方式在增强数据集上达到88%左右的准确率，然而上一节遗留了一个问题即关于修改架构后出现测试集合准确率低的问题。本节补充主要致力于完善这一点。
### 1. 猜测
从基本的测试结果发现单纯改变网络结构最后得到的结果是训练集合和测试集合上的准确率均在0.03左右，基本等效于随机猜测的结果。当时的初步想法是可能由于figure图片没有进行rescale将图像归一化，导致的训练困难。于是对于`data_gen.py`,`eval.py`,`customize_service.py`三个脚本内的`preprocess_img`函数进行了修改，将图像归一化至0～1，通过除以255.0实现。实现代码如下：
```
def preprocess_img(img_path, img_size):
    """
    image preprocessing
    you can add your special preprocess mothod here
    """
    img = Image.open(img_path)
    img = img.resize((img_size,img_size))
    img = img.convert('RGB')
    img = np.array(img)
    img = img[:, :, ::-1]/255.0
    return img
```
通过上述修正，并进行训练发现训练集合上准确率稳步提高，然而测试准确率仍旧稳定在0.03这个水平，猜测问题是图像处理的不一致，但是多番寻找并未找到问题代码，也因此搁置在了这一步。
### 2. 新的尝试
既然可能是由于图像处理的调度不一致，那能否直接将rescale嵌入在模型当中，而不是放在前处理以实现调度的一致从而解决上述问题呢？基于上述的思考，在新的模型中在base_model之前嵌入了对图像的预处理，主要实现图像的归一化在模型调度中实现。<br>
为了实现上述目的，需要额外引入一个不做训练，而是将输入input_tensor进行elment wise的除法操作的层。通过查找keras的API接口，发现`keras.layers.Lambda`层可以实现上述需求，于是将新模型调整为`Input Layer` + `Lambda Layer` + `Base_model` + `GlobalAveragePooling2D`/`GlobalMaxPooling2D`/`Flatten` + `Dropout` + `Dense Layers`的基准架构，相比于之前的版本将最后的特征提取进行了改进，从原先的抽取特征结果直接`Flatten`改进为通过AvgPooling，MaxPooling，Flatten三种方式提取特征并进行拼接。具体的实现代码如下：
```
inputs = Input((FLAGS.input_size, FLAGS.input_size, 3))
inputs_scale = Lambda(lambda x : x/255.)(inputs)
base_model = Xception(weights=os.path.join(FLAGS.pretrained_weight_local,'xception_weights_tf_dim_ordering_tf_kernels_notop.h5'),
                      include_top=False,
                      pooling=None,
                      input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                      classes=FLAGS.num_classes)
x = base_model(inputs_scale)
x1 = GlobalMaxPooling2D()(x)
x2 = GlobalAveragePooling2D()(x)
x3 = Flatten()(x)
x = Concatenate(axis=-1)([x1, x2, x3])
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
predictions = Dense(FLAGS.num_classes, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
```
通过上述的改进，成功实现了模型的转换，通过将模型改变为InceptionResnetV2,Xception,NASNetLarge等模型对比，结果最终选择Xception作为最后的实现，一方面降低了模型的大小提高训练和推断速度，另一方面提高了准确率。通过上述的结果最后的准确率可以提高至90%左右。
### 3. 测试时改进TTA
通过借鉴群友的思路，在测试阶段引入TTA，增强预测效果。初始的思路是借助ImageDataGenerator通过对于原图像采用平移、缩放、扭曲等增强方式生成8张图像，并进行预测，获得8个预测结果，然后对预测结果采用投票方式进行确定最终结果。然而，实际上发现ModelArts在部署时，不支持引入keras，使用的tensorflow也是早期版本没法直接调用ImageDataGenerator。最后，只能退而求其次，采用手动写函数增强的方式，通过引入噪声，明度变化等方式对图像进行改变。对应的代码如下：
```
img = data[self.input_key_1]
img1 = self.center_img(img, self.input_size)
img2 = self.darker(img)
img3 = self.brighter(img)

img = img[np.newaxis, :, :, :] 
img1 = img1[np.newaxis, :, :, :]
img2 = img2[np.newaxis, :, :, :]
img3 = img3[np.newaxis, :, :, :]

pred_score = self.sess.run([self.output_score], feed_dict={self.input_images: img})
pred_score1 = self.sess.run([self.output_score], feed_dict={self.input_images: img1})
pred_score2 = self.sess.run([self.output_score], feed_dict={self.input_images: img2})
pred_score3 = self.sess.run([self.output_score], feed_dict={self.input_images: img3})
pred_score = pred_score+pred_score1+pred_score2+pred_score3
```
### 4. 总结
本文主要立足于补充之前的文章，立足于回答模型改变的准确率无法提升问题，通过在模型内嵌图片预处理，通过对像素值归一化，解决训练准确率不提高问题。同时，在测试阶段引入测试阶段增强TTA思路，提高预测性能。最终通过上述两项举措，将准确率提高到90.9%，并列垃圾分类第100位解决方案。<br>
最后，本文尝试的代码给出在同目录下，为Xception全开训练+TTA的代码案例。
