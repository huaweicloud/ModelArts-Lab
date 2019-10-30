已修改成本地可以运行。 

修改方法：

1.`save_model.py|train.py|eval.py|run.py|`中`moxing.framework.file`函数全部换成`os.path`和`shutil.copy`函数。因为python里面暂时没有moxing框架。

2.注释掉`run.py`文件里面的下面几行代码：
        
    # FLAGS.tmp = os.path.join(FLAGS.local_data_root, 'tmp/')
    # print(FLAGS.tmp)
    # if not os.path.exists(FLAGS.tmp):
    #     os.mkdir(FLAGS.tmp)

# 运行环境

>python3.6

>tensorflow 1.13.1

>keras 2.24


# garbage_classify
## 赛题背景
比赛链接：[华为云人工智能大赛·垃圾分类挑战杯](https://developer.huaweicloud.com/competition/competitions/1000007620/introduction)

如今，垃圾分类已成为社会热点话题。其实在2019年4月26日，我国住房和城乡建设部等部门就发布了《关于在全国地级及以上城市全面开展生活垃圾分类工作的通知》，决定自2019年起在全国地级及以上城市全面启动生活垃圾分类工作。到2020年底，46个重点城市基本建成生活垃圾分类处理系统。

人工垃圾分类投放是垃圾处理的第一环节，但能够处理海量垃圾的环节是垃圾处理厂。然而，目前国内的垃圾处理厂基本都是采用人工流水线分拣的方式进行垃圾分拣，存在工作环境恶劣、劳动强度大、分拣效率低等缺点。在海量垃圾面前，人工分拣只能分拣出极有限的一部分可回收垃圾和有害垃圾，绝大多数垃圾只能进行填埋，带来了极大的资源浪费和环境污染危险。

随着深度学习技术在视觉领域的应用和发展，让我们看到了利用AI来自动进行垃圾分类的可能，通过摄像头拍摄垃圾图片，检测图片中垃圾的类别，从而可以让机器自动进行垃圾分拣，极大地提高垃圾分拣效率。

因此，华为云面向社会各界精英人士举办了本次垃圾分类竞赛，希望共同探索垃圾分类的AI技术，为垃圾分类这个利国利民的国家大计贡献自己的一份智慧。

## 赛题说明
本赛题采用深圳市垃圾分类标准，赛题任务是对垃圾图片进行分类，即首先识别出垃圾图片中物品的类别（比如易拉罐、果皮等），然后查询垃圾分类规则，输出该垃圾图片中物品属于可回收物、厨余垃圾、有害垃圾和其他垃圾中的哪一种。
模型输出格式示例：
    
    {

        " result ": "可回收物/易拉罐"

    }

## 垃圾种类40类

    {
        "0": "其他垃圾/一次性快餐盒",
        "1": "其他垃圾/污损塑料",
        "2": "其他垃圾/烟蒂",
        "3": "其他垃圾/牙签",
        "4": "其他垃圾/破碎花盆及碟碗",
        "5": "其他垃圾/竹筷",
        "6": "厨余垃圾/剩饭剩菜",
        "7": "厨余垃圾/大骨头",
        "8": "厨余垃圾/水果果皮",
        "9": "厨余垃圾/水果果肉",
        "10": "厨余垃圾/茶叶渣",
        "11": "厨余垃圾/菜叶菜根",
        "12": "厨余垃圾/蛋壳",
        "13": "厨余垃圾/鱼骨",
        "14": "可回收物/充电宝",
        "15": "可回收物/包",
        "16": "可回收物/化妆品瓶",
        "17": "可回收物/塑料玩具",
        "18": "可回收物/塑料碗盆",
        "19": "可回收物/塑料衣架",
        "20": "可回收物/快递纸袋",
        "21": "可回收物/插头电线",
        "22": "可回收物/旧衣服",
        "23": "可回收物/易拉罐",
        "24": "可回收物/枕头",
        "25": "可回收物/毛绒玩具",
        "26": "可回收物/洗发水瓶",
        "27": "可回收物/玻璃杯",
        "28": "可回收物/皮鞋",
        "29": "可回收物/砧板",
        "30": "可回收物/纸板箱",
        "31": "可回收物/调料瓶",
        "32": "可回收物/酒瓶",
        "33": "可回收物/金属食品罐",
        "34": "可回收物/锅",
        "35": "可回收物/食用油桶",
        "36": "可回收物/饮料瓶",
        "37": "有害垃圾/干电池",
        "38": "有害垃圾/软膏",
        "39": "有害垃圾/过期药物"
    }
## efficientNet默认参数

        (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),



## 代码解析
### BaseLine改进
1.使用多种模型进行对比实验，ResNet50, SE-ResNet50, Xeception, SE-Xeception, [efficientNetB5](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)。

2.使用组归一化（GroupNormalization）代替批量归一化（batch_normalization）-解决当Batch_size过小导致的准确率下降。当batch_size小于16时，BN的error率
逐渐上升，`train.py`。
    
    
    for i, layer in enumerate(model.layers):
        if "batch_normalization" in layer.name:
            model.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)

3.NAdam优化器
    
    
    optimizer = Nadam(lr=FLAGS.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

4.自定义学习率-SGDR余弦退火学习率
    
    
    sample_count = len(train_sequence) * FLAGS.batch_size
    epochs = FLAGS.max_epochs
    warmup_epoch = 5
    batch_size = FLAGS.batch_size
    learning_rate_base = FLAGS.learning_rate
    total_steps = int(epochs * sample_count / batch_size)
    warmup_steps = int(warmup_epoch * sample_count / batch_size)

    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=0,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=0,
                                            )

5.数据增强：随机水平翻转、随机垂直翻转、以一定概率随机旋转90°、180°、270°、随机crop(0-10%)等(详细代码请看`aug.py`和`data_gen.py`)

    def img_aug(self, img):
        data_gen = ImageDataGenerator()
        dic_parameter = {'flip_horizontal': random.choice([True, False]),
                         'flip_vertical': random.choice([True, False]),
                         'theta': random.choice([0, 0, 0, 90, 180, 270])
                        }


        img_aug = data_gen.apply_transform(img, transform_parameters=dic_parameter)
        return img_aug


    from imgaug import augmenters as iaa
    import imgaug as ia

    def augumentor(image):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(rotate=(-10, 10)),
                sometimes(iaa.Crop(percent=(0, 0.1), keep_size=True)),
            ],
            random_order=True
        )


        image_aug = seq.augment_image(image)

        return image_aug


6.标签平滑`data_gen.py`
    
    
    def smooth_labels(y, smooth_factor=0.1):
        assert len(y.shape) == 2
        if 0 <= smooth_factor <= 1:
            # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
            y *= 1 - smooth_factor
            y += smooth_factor / y.shape[1]
        else:
            raise Exception(
                'Invalid label smoothing factor: ' + str(smooth_factor))
        return y
        
7.数据归一化：得到所有图像的位置信息`Save_path.py`并计算所有图像的均值和方差`mead_std.py`
    
    
    normMean = [0.56719673 0.5293289  0.48351972]
    normStd = [0.20874391 0.21455203 0.22451781]
    
    
    img = np.asarray(img, np.float32) / 255.0
    mean = [0.56719673, 0.5293289, 0.48351972]
    std = [0.20874391, 0.21455203, 0.22451781]
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    img[..., 0] /= std[0]
    img[..., 1] /= std[1]
    img[..., 2] /= std[2]

## 各部分代码解析

* `deploy_scripts`——推理文件，需要修改
      
      1.self.input_size = 456 
      
      
      2. def _inference(self, data):
      """
      model inference function
      Here are a inference example of resnet, if you use another model, please modify this function
      """
      img = data[self.input_key_1]
      img = img[np.newaxis, :, :, :]  # the input tensor shape of resnet is [?, 224, 224, 3]
      img = np.asarray(img, np.float32) / 255.0
      mean = [0.56719673, 0.5293289, 0.48351972]
      std = [0.20874391, 0.21455203, 0.22451781]
      img[..., 0] -= mean[0]
      img[..., 1] -= mean[1]
      img[..., 2] -= mean[2]
      img[..., 0] /= std[0]
      img[..., 1] /= std[1]
      img[..., 2] /= std[2]
      pred_score = self.sess.run([self.output_score], feed_dict={self.input_images: img})
      if pred_score is not None:
          pred_label = np.argmax(pred_score[0], axis=1)[0]
          result = {'result': self.label_id_name_dict[str(pred_label)]}
      else:
          result = {'result': 'predict score is None'}
      return result


* `aug.py`——图像增强代码(`imgaug`函数）

* `data_gen.py`——数据预处理代码，包括数据增强、标签平滑以及train和val的划分

* `eval.py`——估值函数

* `Groupnormalization.py`——组归一化

* `mean_std.py`——图像均值和方差

* `Network.py`——ResNet50, SE-ResNet50, Xeception, SE-Xeception, efficientNetB5

* `run.py`——运行代码

* `save_model.py`——保存模型

* `Save_path.py`——图像位置信息

* `train.py`——训练网络部分，包括网络，loss, optimizer等

* `warmup_cosine_decay_scheduler.py`——余弦退火学习率

## 使用
### 前期准备
* 克隆此存储库
    
    
    
      git clone https://github.com/wusaifei/garbage_classify.git
    


* [垃圾分类数据集下载地址](https://modelarts-competitions.obs.cn-north-1.myhuaweicloud.com/garbage_classify/dataset/garbage_classify.zip)

* 扩充数据集：链接：https://pan.baidu.com/s/1SulD2MqZx_U891JXeI2-2g 
提取码：epgs


### 运行
* 运行`Save_path.py`得到图像的位置信息
* 运行`mean_std.py`得到图像的均值和方差
* `run.py`——训练
    
    
      python run.py --data_url='./garbage_classify/train_data' --train_url='./model_snapshots' --deploy_script_path='./deploy_scripts'
    
    
* `run.py`——保存为pd
      
      
      
        python run.py --mode=save_pb --deploy_script_path='./deploy_scripts' --freeze_weights_file_path='./model_snapshots/weights_024_0.9470.h5' --num_classes=40



* `run.py`——估值
    
    
    
      python run.py --mode=eval --eval_pb_path='./model_snapshots/model' --test_data_url='./garbage_classify/train_data'
    
    

## 实验结果

* 网络的改进：ResNet50-0.689704，SE-ResNet50-0.0.83259，Xception-0.879003，EfficientNetB5-0.924113（无数据增强）

* 数据增强：由0.924113提升到0.934721

* 标签平滑和数据归一化处理、学习率策略的调整`ReduceLROnPlateau`换成`WarmUpCosineDecayScheduler`，最终准确率在95%左右















