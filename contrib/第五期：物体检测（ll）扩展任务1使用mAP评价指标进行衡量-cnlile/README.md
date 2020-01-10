# 第五期：物体检测（ll）扩展任务1:使用mAP评价指标进行衡量

​         在第五期的扩展任务1：在第五期的notebook案例中，并没有对物体检测模型的精度进行评估，物体检测模型的精度使用mAP评价指标进行衡量。要求在notebook案例中，开发代码，计算出模型的mAP。

## 一、检测精度mAP简介

​		Mean Average Precision（MAP）：平均精度均值。

### 1.1 基本知识

​	P（Precision）精度，正确率。定义为： precision=返回结果中相关文档的数目/返回结果的数目。
​	Rec(Recall)召回率：定义为：Recall=返回结果中相关文档的数目/所有相关文档的数目。
​	数学公式理解:
​	1）True Positive(真正，TP)：将正类预测为正类数 

​	2）True Negative(真负，TN)：将负类预测为负类数 

​	3）False Positive(假正，FP)：将负类预测为正类数误报 (Type I error) 

​	4）False Negative(假负，FN)：将正类预测为负类数→漏报 (Type II error)

​	精确率(precision):P=TP/(TP+FP)（分类后的结果中正类的占比） 
​	召回率（recall）:recall=TP/(TP+FN)(所有正例被分对的比例）



### 1.2 应用于图像中的mAP

​	在图像中，主要计算两个指标：`precision`和`recall`。

​	precision，recall都是选多少个样本k的函数，如果我总共有1000个样本，那么我就可以像这样计算1000对P-R，并且把他们画出来，这就是PR曲线：这里有一个趋势，recall越高，precision越低。
​	平均精度AP（average precision）:就是PR曲线下的面积，这里average，等于是对recall取平均。而mean average precision的mean，是对所有类别取平均（每一个类当做一次二分类任务）。现在的图像分类论文基本都是用mAP作为标准。 

​	AP是把准确率在recall值为Recall = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1}时（总共11个rank水平上），求平均值。

 	mAP是均精度均值：只是把每个类别的AP都算了一遍，再取平均值。

​	**AP是针对单个类别的，mAP是针对所有类别的。**



## 二、应用检测的流程

​	根据本期实验的特点，结合modelart本身的实际情况，制定以下流程：

### 2.1 定义imdb（检测数据集）

​	首先，需要获取测试数据，依然从VOC2007数据里面获取，定义一个imdb（检测数据）。要求这些数据

### 2.2 定义net

​	在测试计算时，还是需要建立神经网络，用来检测数据。这里还是根据案例情况，选择Vgg16网络，并且加载我们生成的训练模型，并且配置正确的参数。

### 2.3 图像检测并且保存结果

​	计算数据集的图像数量，对所有图像进行分类检测，检测数据结果进行保存。

### 2.4 计算AP

​	按照VOC2007的分类结果，计算检测数据的AP值。利用图像检测的结果数据，对每一类的图像AP值进行分别计算。

### 2.5 取得mAP

​	计算AP的平均值，或者mAP。



## 三、mAP的代码解析

​	代码分三部分：

​	1）数据准备

​	2）图像检测和提取

​	3）计算AP和mAP

​    写了2个文件，一个文件是主要执行文件test_mAP_FasterRCNN.ipynb，还有个是计算AP值的文件voc_eval.py。



### 3.1 数据准备

​	这部分代码都写在test_mAP_FasterRCNN.ipynb内。

#### 3.1.1 imdb

​	这个部分主要还是提取VOC2007 的测试数据，作为检测mAP用。也是利用原来的实验部分代码的写法构成。定义了加载函数 def combined_roidb(imdb_names)，里面处理比较常规，选用数据集，对图像进行了反转处理。最后返回两个数据：imdb 和 roidb。

​	imdb的定义方式：imdb = get_imdb(imdb_name)，从imdb本质上来说，它是一个基于pascal_voc.py 的类的实例对象，大家有兴趣可以去看pascal_voc.py 功能函数，里面定义了诸多方法，都是我们要用到的。imdb这里的数据集获取的同时，还做了数据的初始化，获取了pascal_voc.py的各种功能函数，不少功能我们后面都要用到。

​	结果就是，get_imdb(imdb_name)将会返回的就是pascal_voc(split, year)这样一个对象。

​	最后，不要忘记imdb, roidb = combined_roidb(imdb_name)，这行代码是加载测试集数据。	

#### 3.1.2 net

​	定义一个用于计算的net，和第五期实验一样，也是采用了Vgg16的模型。

​	这里还要做以下工作：

​	1）net.create_architecture，构建faster rcnn进行计算图的操作。这里要注意，modelart环境中的参数和其它场景略有差异，是不需要传sess这个参数的。这里的参数中，参数ANCHOR_SCALES必须，参数ANCHOR_RATIOS可以省略，其它都按常规传。

​	2） 加载权重文件，net.load_state_dict方法，主要是加载的文件位于"./models/vgg16-voc0712/vgg16_faster_rcnn_iter_110000.pth"，这个就是训练模型文件。

​	3）选择计算设施：net.to(net._device)。



### 3.2 图像检测和提取

​	这个是功能和代码的核心部分，主要是形成可以被计算mAP的图像文件。这里分以下步骤：

#### 3.2.1 定义目标图像框容器

​	根据每一类文件，每一张图片，都定义个容器，就是定义一个数据集存储单元。

​	代码为：

​		`all_boxes = [[[] for _ in range(num_images)]`

​			`for _ in range(imdb.num_classes)]`

​	这里的只有存储单元，里面都空值。

#### 3.2.1 分类图像数据组织和填充

​	这里首先根据图像分类，按照类别对检测数据进行处理。

​	前期处理：

​	`from model.test import im_detect`

​    `im = cv2.imread(imdb.image_path_at(i))`

​    `scores, boxes = im_detect( net,  im)`

​    作用是返回这张图片中的多个目标和分数，存放在boxes和scores中。

​	主要处理：先获取上述过程中的为每一个图像分配的存储单元的尺寸数据，和图像数据集的数据，检测后，将数据进行填充。这里代码主要是：

```
		inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j*4:(j+1)*4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = nms(torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores), NMS_THRESH)
        cls_dets = cls_dets[keep, :]
        all_boxes [j][i] = cls_dets
```

​	以上步骤完成后，all_boxes 里面已经有了数据。需要注意，这里的计算时间略长，大概要几分钟时间，需要等待。

​	最后处理；，我们要把all_boxes的数据写入文件，这里调用imdb的函数：	

​	`imdb._write_voc_results_file(all_boxes)`

​	 _write_voc_results_file 函数可以在pascal_voc.py的源代码中找到，完成之后，我们打开目录，/data/VOCdevkit2007/result/VOC2007/Main/下面目录，可以看到，每一类文件都做了单独存储，这里要注意，因为是modelart环境，每次产生文件名，里面有个计算机ID代码，每次启动，不一定是一样的。



### 3.3 计算AP和mAP

​	其实，这里有个便捷的方式，就是直接调用imdb.evaluate_detections(all_boxes, output_dir)。就像我最后面代码做的那样，可以忽略一些细节，直接给出mAP。

​	我这里的做法是自己写了个文件voc_eval.py，提供voc_eval的函数供调用，这样更加灵活，也能更加符合我进行实验的目的，可以获得更加丰富的数据。

​	voc_eval函数：

	def voc_eval(detpath,
	        annopath,
	        imagesetfile,
	        classname,
	        cachedir,
	        ovthresh=0.5,
	        use_07_metric=False,
	        use_diff=False):
​	在这里文件里面代码，都有详细的注释。

​	关键处理如下：

​	1）读取文件数据，存入数据字典，这里的文件就是上述保存的文件位置，代码如下：

        recs = {}#生成一个字典
        for i, imagename in enumerate(imagenames): #对于每一张图像进行循环
            recs[imagename] = parse_rec(annopath.format(imagename))#在字典里面放入每个图像缓存的标签路径
            if i % 100 == 0:#在这里输出标签读取进度。
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))#从这里可以看出来imagenames是什么，是一个测试集合的名字列表，这个Print输出进度。
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))#读取的标签保存到一个文件里面
        with open(cachefile, 'wb+') as f:#打开缓存文件
            pickle.dump(recs, f)#dump是序列化保存，load是反序列化解析
​	2）从以下代码开始就是具体计算了：

​	`nd` = len(image_ids)  #统计检测出来的目标数量`
​    `tp = np.zeros(nd)#tp = true positive 就是检测结果中检测对的-检测是A类，结果是A类`
​    fp = np.zeros(nd)#fp = false positive 检测结果中检测错的-检测是A类，结果gt是B类。`

    if BB.shape[0] > 0:#。shape是numpy里面的函数，用于计算矩阵的行数shape[0]和列数shape[1]



   最后，每一类数据都会返回三个值：

   rec： 召回率

   prec：精确度

   ap：平均精度

   最后得出mAP：0.7888660508602544。



  最后，贴一张图，是保存的分类mAP检测结果。

![](https://github.com/cnlile/huawei_modelart_experiments/blob/master/phase5/expands/fifth_1_output.png?raw=true)

​		