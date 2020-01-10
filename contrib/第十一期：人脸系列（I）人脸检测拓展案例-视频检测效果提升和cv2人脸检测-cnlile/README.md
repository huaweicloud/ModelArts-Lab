# 第十一期：人脸系列（I）人脸检测拓展案例-视频检测效果提升和cv2人脸检测

## 1、前言

​		本期内容主要是基于dlib和openCV，从图像和视频两个方向做了人脸检测，提供了详尽的代码案例的实践。在此基础上，根据案例的要求，做了以下两部分内容：

​		1）基于dlib的视频检测效果提升的案例：主要是对案例上的dlib视频检测内容进行了解读，对其中一些细节进行了推敲，根据一些实验和测试，提升了一些检测速度。根据实验的结果，对过程做了说明（包含主要代码的说明），并且阐述了其中原理，并且对一些拓展的知识内容也捎带做了说明。

​		2）openCV的人脸检测：根据一些文档和资料，基于openCV库，实现了人脸的检测。对openCV实现的原理和方式做了说明，对代码主要部分做了描述，并且将在实践中遇到的问题重点基于分析，并且写出了解决思路和方式，对检测结果数据做了一些对比。

## 2、视频检测效果提升

​	基于Dlib 库做的视频效果提升方式，主要对整个原理和过程做了说明，相关代码参考：new_face_dlib_detector.ipynb文件。

### 2.1 Dlib 人脸检测原理的简述

​		Dlib是一个老牌的跨平台的C公共库，支持C语言和Python语言进行开发和学习，主要以C为主。它除了线程支持，网络支持，提供测试以及大量工具等等优点，Dlib还是一个强大的机器学习的C库，包含了许多机器学习常用的算法。同时支持大量的数值算法如矩阵、大整数、随机数运算等等。Dlib同时还包含了大量的图形模型算法。最重要的是Dlib的文档和例子都非常详细，可以访问它的网站[http://dlib.net]来获取相关资料，目前最高的版本是ver.19.18，比modelart平台版本要高一些，不过兼容性挺好。

​		Dlib人脸识别有两种检测器：1、基于特征（HOG）+分类器（SVM）的检测器；2、基于深度学习的卷积神经网方法（resnet_model）的检测器。

#### 2.1.1 HOG检测器

​		该检测器由两部分组成：特征工程部分和分类器，特征工程算法采用了HOG算法，分类器用了SVM分类器。
​		在计算机视觉以及数字图像处理中，梯度方向直方图(**HOG：Histogram of Oriented Gridients**)是一种能对物体进行检测的基于形状边缘特征的描述算子，它的基本思想是利用梯度信息能很好的反映图像目标的边缘信息并通过局部梯度的大小将图像局部的外观和形状特征化。

​	HOG简单来说分为以下几个过程：

​	1）图像预处理。

​	2）计算图像像素点梯度值，得到梯度图(尺寸和原图同等大小)。

​	3）图像划分多个cell，统计cell内梯度直方向方图。

​	4）将2×2个cell联合成一个block,对每个block做块内梯度归一化。

​	5）获取HOG特征值。

​		分类器用了常见的SVM方式，支持向量机（Support Vector Machine, SVM）是一类按监督学习（supervised learning）方式对数据进行二元分类的广义线性分类器（generalized linear classifier），其决策边界是对学习样本求解的最大边距超平面（maximum-margin hyperplane）。

​		HOG+SVM可以说是经典组合，应用广泛，除了人脸识别，近些年来，还有人用它做行人检测、车牌识别等。

#### 2.1.2  卷积神经网方法检测器

 		该方法使用基于CNN（卷积神经网络）的特征，采用最大余量对象检测器（MMOD）。此方法的训练过程非常吃资源，建议采用GPU进行计算。在Dlib官方网站有一个现成的模型，它使用一个由作者davis king手动标记的数据集，由来自不同数据集的图像组成，如imagenet、pascal voc、vgg、wide、face scrub，它包含7220幅图像。

#### 2.1.3 人脸关键点检测
     基础的人脸关键点共有 68 个，分别是人脸各部位的点，如嘴角，眼睛边等。 dlib提供了很友好的检测人脸landmarkers的接口，landmarkers是一种人脸部特征点提取的技术，大致基于ERT（ensemble of regression trees）级联回归算法，即基于梯度提高学习的回归树方法。
     现在，有些商家的人脸关键点实用中往往不止68个，会更多，上次在百度看到一个API提供的关键点有150个之多。

### 2.2 案例过程和代码的分析
​		打开第十一期的案例代码，现在将关键代码给予分析，了解据体代码运行的过程。代码分成两个部分：

#### 2.2.1 对图片进行人脸检测

```python
detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1("./models/detector.dat")
```
​		以上两行代码，第一行定义了一个dlib的人脸检测器，第二行代码加载了CNN检测器的数据模型，数据模型位置在./models/detector.dat下。这样完成了定义了一个可用的人脸检测器。

```python
dets = cnn_face_detector(image, 1)
```
​		上面这行代码，将我们载入的图片加载进入检测器，进行检测后，返回一个检测结果dets。dets是一个多个检测结果的数据集合，包含检测到的每张图片的位置等数据。

```python
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
        i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
Number of faces detected: 1
Detection 0: Left: 443 Top: 119 Right: 613 Bottom: 289 Confidence: 1.0296450853347778
```
上面代码是对dets数据进行解析，利用for循环输出每一张被解析到的数据内容，返回的i和d参数，i是指第N张图片，d是这张图片的数据内容，i从0开始计数。上面打印出来了这张图片的位置信息：左上角坐标（443，119），右下角坐标（613，289）,这个是一个图片绝对坐标值的位置。

```python
res_img = cv2.rectangle(image, (443, 119), (613, 289), 0, 1)
Image.fromarray(res_img)
```
      上面两行代码是图片显示，利用openCV库，在图片上画位置框，位置信息就是上面的绝对坐标值的位置信息，后面两个参数是框的颜色等信息。显示使用的PIL库，进行图片的展示，就如我们在案例中看到的图像。

```python
predictor_kp = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
```
上面这行代码就是加载了一个人脸关键点的检测器，模型文件为shape_predictor_68_face_landmarks.dat。

```python
 shape = predictor_kp(image, d)
    print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                              shape.part(1)))
Part 0: (435, 200), Part 1: (442, 224) ...
```
       上面代码就是用人脸关键点检测器对图像进行关键点解析，d就是用人脸位置识别器识别出来的人脸位置数据。然后打印出来检测的位置信息。

```python
for i in range(68):
    res_img = cv2.circle(res_img,(shape.part(i).x,shape.part(i).y), 1, 255, 4)
Image.fromarray(res_img)
```
     上面代码是显示图片中人脸关键点位置，利用for循环在图片上逐一描绘每一个点，如果你的关键点不是68个，就修改掉68这个数，最后用使用的PIL库，在案例中显示图片。

#### 2.2.2 对视频进行人脸检测

```python
cap = cv2.VideoCapture(video_name)

while True:
    try:
        clear_output(wait=True)
        ret, frame = cap.read()
        if ret:
			res_img = keypoint_detector(frame)
            img = arrayShow(res_img)
            display(img)
            display(img)
            time.sleep(0.05)
        else:
            break
    except KeyboardInterrupt:
        cap.release()
cap.release()
```
​		以上代码是视频检测的主要代码，首先，cap = cv2.VideoCapture(video_name)读取视频文件，将视频文件处理的结果数据集合返回到cap，然后按照一帧一帧的方式循环读取每一帧，每一帧画面作为一张图像，放入自定义keypoint_detector函数中处理，然后将出后返回的图像文件img进行显示。代码time.sleep(0.05)是中间设置间歇的延迟时间，以便于显示（让我们的肉眼可以明显看出来）。

```python
def keypoint_detector(image):
    global res_img
    detector_kp = dlib.get_frontal_face_detector()
    predictor_kp = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
    dets = detector_kp(image, 1)
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        res_img = cv2.rectangle(image,( d.left(), d.top()), (d.right(), d.bottom()), 0, 1)
        shape = predictor_kp(image, d)
        for i in range(68):
            res_img = cv2.circle(image,(shape.part(i).x,shape.part(i).y), 1, 255, 2)
    return res_img
```
​		以上代码是自定义函数keypoint_detector，它传入的参数是一帧图像数据，dlib.get_frontal_face_detector()定义了一个基于dlib的HOG的人脸检测器，dlib.shape_predictor 是载入这个人脸检测器的模型数据，这样就可以使用用了。进行检测后，返回一个dets检测结果的对象。和上次一样，对结果进行循环读取人脸，然后打印出人脸在图像中的绝对位置坐标。还是用predictor_kp(image, d) 方式进行人脸关键点检测，还是熟悉的68个点特征点，使用循环将这些数据在图像中进行描绘。
这段代码就是熟悉的味道，将视频中每一帧的画面当作一张图像进行处理。

### 2.3 视频提升的方式
​		主要还是想办法优化检测的速度，以提升图像的检测速度为目标任务。为了检测结果，我在代码中增加了代码，可以检测出来这个过程耗时，单位是秒。

```python
cap = cv2.VideoCapture(video_name)
start_time = datetime.now() #获得当前开始时间
#............省略各种处理步骤
end_time = datetime.now() #获得当前结束时间
durn = (end_time - start_time ).seconds  #两个时间差，并以秒显示出来
print('cost time:', durn, count)
```
​		基本的案例运行消耗时间：
![](https://cdn.nlark.com/yuque/0/2019/png/316044/1572277196084-736f6fcd-3f6b-4f56-bd24-a321b618afaf.png#align=left&display=inline&height=357&originHeight=357&originWidth=579&search=&size=0&status=done&width=579)
​		这个是基本的消耗时间552秒。


#### 2.3.1 修改图像的灰度
​		首先，想到了图像的灰度处理方式，我们检测人脸，用彩色图片和灰度图片对结果并无太大差异。但是，把原始图像转为灰度之后，减小图像原始数据量，便于后续处理时计算量更少，因为检测处理不一定需要对彩色图像的RGB三个分量都进行处理。增加了如下代码:

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

​		这样，每次输入的图片都从彩色变成了灰色，对比一下结果吧。
![](https://cdn.nlark.com/yuque/0/2019/png/316044/1572277216694-eaa9bbb1-5c57-4aa7-9a16-9afcda0538e3.png#align=left&display=inline&height=301&originHeight=301&originWidth=596&search=&size=0&status=done&width=596)
​		现在耗时527秒，这个点提升程度，微不足道啊。。


#### 2.3.2 代码优化

```python
def keypoint_detector(image):
    global res_img
    detector_kp = dlib.get_frontal_face_detector()
    predictor_kp = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
```

​		观察以上代码，在原始代码中，发现每次调用函数都要重复进行定义人脸模型检测器，如果1000帧都要重新定义1000次模型检测器，这个每次执行都是一模一样的代码，毫无区别，可以做一下优化，把模型检测器放到函数之外，这样可以节省大量的执行时间，如下:

```python
detector_kp = dlib.get_frontal_face_detector()
predictor_kp = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

def keypoint_detector(image):
    global res_img
```
​		并且，添加如下代码，每计算一帧画面就计数器+1，看看默认的视频有多少帧：
```python
count = 0
while True:
 #省略中间代码
            time.sleep(0.01)
            count = count + 1
```

​		看看结果这样修改之后的执行结果：
![](https://cdn.nlark.com/yuque/0/2019/png/316044/1572277311795-018fb172-c929-42a3-9234-53461ac97449.png#align=left&display=inline&height=404&originHeight=404&originWidth=1047&search=&size=0&status=done&width=1047)
​		这下明显有了提升啊，耗时22秒，和刚才比，真心飞跃，这个视频一共369帧图像，也就是说我们节约了368次的模型定义和设置。


#### 2.3.3 重建模型文件
​		要提高检测速度，最根本的还是要优化模型，即你的模型是和你的检测结果有直接的关系。于是，对dlib 的HOG模型进行了重新的训练，构建了一个新的模型文件，进行了测试。
实践步骤和相关代码:

##### 2.3.3.1 下载训练样本图片

​		很不客气的直接从dlib的网站上下载样本图片了，做了一回伸手党，地址：[http://dlib.net/files/data/](http://dlib.net/files/data/)[dlib_face_detector_training_data.tar.gz](http://dlib.net/files/data/dlib_face_detector_training_data.tar.gz)。 下载之后，在notebook的work工作目录中，使用upload进行上传。上传完成之后，在python jupyter编辑器中，进行解压缩，结果如下：
![](https://cdn.nlark.com/yuque/0/2019/png/316044/1572277178137-26a1637e-5284-4799-9b70-7b489b07d8dc.png#align=left&display=inline&height=371&originHeight=371&originWidth=806&search=&size=0&status=done&width=806)
​		这个文件解压缩之后，里面有两个xml的文件，是图型的配置文件，有个images目录，下面是各种人脸图像文件，有兴趣的话，可以打开看看，他们的大小就没有小于80 × 80 的哦！

##### 2.3.3.2 进行参数设置和训练模型

```python
# options用于设置训练的参数和模式
options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
# 支持向量机的C参数，通常默认取为5.自己适当更改参数以达到最好的效果
options.C = 5
# 线程数，
options.num_threads = 8
options.be_verbose = True
```
​		上面代码是设置训练的参数, 首先，定义一个人脸识别器，dlib.simple_object_detector_training_options。然后设置options参数，配置详情：
1）add_left_right_image_flips: 面部左右对称,选择是
2）C：支持向量机SVM的超参数，这个值如果大一些会比较准确，但是会带来过拟合，选择一个居中的值吧，那就选择5。
3）num_threads：线程数，考虑Notebook是在一个8核的cpu虚拟机上，毫不犹豫选了8
4）be_verbose：这个参数似乎是什么提示信息，就写个True吧。

```python
current_path = os.getcwd()
train_folder = current_path + '/dlib_face_detector_training_data/'
train_xml_path = train_folder + 'frontal_faces.xml'
dlib.train_simple_object_detector(train_xml_path, 'detector.svm', options)
```
​		上面代码是加载训练的图形目录和配置文件，就是刚才解压的数据包里面的，用train_simple_object_detector 载入配置参数，申明模型文件detector.svm，构成一个检测训练器，并且进行训练。
```python
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(train_xml_path, "detector.svm")))
```
​		上面代码是将训练完成后的模型结果保存在模型文件之中，detector.svm 这个模型文件大小是 44.7kB。

##### 2.3.3.3 加载自建模型和测试

```python
my_detector = dlib.simple_object_detector("detector.svm")
image = dlib.load_rgb_image("./face_1.jpeg")
dets = my_detector(image, 1)
```
上面代码是将自己训练的模型文件，载入检测器，然后加载本期官方的范例图片，进行检测。
还是拿官方的图片自测一下效果，如下：
![](https://cdn.nlark.com/yuque/0/2019/png/316044/1572278815199-d106f16a-dd93-4e6d-a833-2000700dcaa2.png#align=left&display=inline&height=530&originHeight=530&originWidth=640&search=&size=0&status=done&width=640)
两张人脸检测对比，上面一张是本期官方的图片检测结果，下面这张是自己训练的模型进行检测的结果。

##### 2.3.3.4 结果

​		最后，我们以自己训练模型进行视频检测，结果如下：
![](https://cdn.nlark.com/yuque/0/2019/png/316044/1572278815863-39148505-8af7-4f64-8d18-9672c5d2cea3.png#align=left&display=inline&height=385&originHeight=385&originWidth=1006&search=&size=0&status=done&width=1006)
​		现在耗时17秒，又有所提升。。。看来通用的模型文件不如自己训练过的本地化模型文件执行更快啊！感觉就是大锅饭和开小灶的区别吧。。从22秒减少到17秒，减少了了22.7%的时间，这个可是切切实实的提升。


#### 2.3.4 其它
       其间，还考虑过其它方式，比如改变图片尺寸，变的更小，但是考虑到dlib的识别图像的只有80 * 80，太小的图片会导致小脸就无法识别了。又参考了于仕琪老师的文章《怎么把人脸检测的速度做到极致》，里面有很多好东西啊，可惜建议自己的能力和时间不足，没有去做，有兴趣的可以尝试用LBP做特征，用AdaBoost这种Boosting方法做分类。于仕琪老师写了一个很快很准的人脸检测算法库，以二进制形式免费发布，地址在:[https://github.com/ShiqiYu/libfacedetection。](https://github.com/ShiqiYu/libfacedetection%E3%80%82)里面可惜没有python形式的，没有办法直接采用。

## 3、cv2人脸检测
​		OpenCV是一个基于BSD许可（开源）发行的跨平台计算机视觉库，它轻量级而且高效——由一系列 C 函数和少量 C++ 类构成，同时提供了Python、Ruby、MATLAB等语言的接口，实现了图像处理和计算机视觉方面的很多通用算法。OpenCV用C++语言编写，它的主要接口也是C++语言，现在对Python也有了很好的支持。
cv2是OpenCV官方的一个更新和扩展库，里面含有各种有用的函数以及进程，目前我们基本都是用它来工作。python使用的模块也是基于为cv2，被写成cv2，是因为该模块引入了一个更好的cv2的api接口。
cv2对人脸检测也有两种不同的检测器：1、基于 Haar 级联分类器的检测器；2、基于DNN网络的人脸检测器。
该部分的代码参考face_openCv_detector.ipynb文件。

### 3.1 OpenCV Haar Cascade人脸检测

​		该检测器由特征提取和分类器两个部分组成，现在对原理和代码进行说明。

#### 3.1.1 检测原理简述
​		在此之前要先介绍一下级联分类器CascadeClassifier，CascadeClassifier为OpenCV下用来做目标检测的级联分类器的一个类。该类中封装的目标检测机制，简而言之是滑动窗口机制+级联分类器的方式。级联分类器： 可以理解为将N个单类的分类器串联起来。如果一个事物能属于这一系列串联起来的的所有分类器，则最终结果就是 是，若有一项不符，则判定为否。人脸，它有很多属性(两条眉毛，两只眼睛，一个鼻子等等)，我们将每个属性做一成个分类器，如果一个模型符合了我们定义的人脸的所有属性，则我们人为这个模型就是一个人脸。
​		opencv目前仅支持三种特征的训练检测， HAAR、LBP、HOG，这里我们主要用的是Haar。Haar：从OpenCV1.0以来，一直都是只有用haar特征的级联分类器训练和检测（检测函数称为cvHaarDetectObjects，训练得到的也是特征和node放在一起的xml），在之后当CascadeClassifier出现并统一三种特征到同一种机制和数据结构下时，没有放弃原来的C代码编写的haar检测，仍保留了原来的检测部分。openCV的的Haar分类器是一个监督分类器，首先对图像进行直方图均衡化并归一化到同样大小(例如，30x30)，然后标记里面是否包含要监测的物体。为了检测整副图像，可以在图像中移动搜索窗口，检测每一个位置来确定可能的目标。 为了搜索不同大小的目标物体，分类器被设计为可以进行尺寸改变，这样比改变待检图像的尺寸大小更为有效。所以，为了在图像中检测未知大小的目标物体，扫描程序通常需要用不同比例大小的搜索窗口对图片进行几次扫描。
​		CascadeClassifier检测的基本原理：检测的时候可以简单理解为就是将每个固定size特征（检测窗口）与输入图像的同样大小区域比较，如果匹配那么就记录这个矩形区域的位置，然后滑动窗口，检测图像的另一个区域，重复操作。由于输入的图像中特征大小不定，比如在输入图像中眼睛是50x50的区域，而训练时的是25x25，那么只有当输入图像缩小到一半的时候，才能匹配上，所以这里还有一个逐步缩小图像，也就是制作图像金字塔的流程。这个size由训练的参数而定。如下图：
![](https://cdn.nlark.com/yuque/0/2019/png/316044/1572281113761-d570d299-40f1-4e71-a4d0-74a970b06672.png#align=left&display=inline&height=367&originHeight=367&originWidth=354&search=&size=0&status=done&width=354)




​		由于人脸可能出现在图像的任何位置，在检测时用固定大小的窗口对图像从上到下、从左到右扫描，判断窗口里的子图像是否为人脸，这称为滑动窗口技术（sliding window）。
#### 3.1.2 代码实现
​		首先，先要去下载人脸训练的数据，这个数据可以去openCV在git上的资料中进行下载，数据是一个xml文件，在openCV网站上(https://github.com/opencv/opencv/tree/master/data/haarcascades)，你可以看到各种xml文件，我们这里选取的haarcascade_frontalface_alt2.xml是一个快速检测模型文件。
```python
# 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值github文件
face_cascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_alt2.xml")
```
​		以上代码是模型文件加载到级联分类器内，构成一个检测器对象。
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
```
      以上代码对图片做灰度处理 ，然后利用上面构成的检测器用来探测图片中的人脸。
```python
vis = image.copy()
for x1, y1, x2, y2 in faces: # x2,y2 为x1，y1 的偏离位置
    x2 = x2 + x1
    y2 = y2 + y1
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
    print("Detection : x1: {} y1: {} x2: {} y2: {}".format(
       x1, y1, x2, y2))
Detection : x1: 448 y1: 88 x2: 655 y2: 295
```
这里做了一个检测结果图片的copy，为了让我更灵活的调测代码：）。利用for循环展开检测结果数据，得到图片检测坐标，这里的坐标和dlib的坐标不同，它的x2和y2不是图片的绝对位置，是相对于x1，y1的相对距离位置。
```python
from PIL import Image
Image.fromarray(vis)
```
还是那熟悉的味道，利用PIL库展示图片，这里友情提示一下，cv2本身也自带图片显示的功能，cv2.imshow("Image",image)。。。结果就是notebook执行崩溃。。因为编辑器不支持cv2的imshow。。
图片检测结果如下：
![ImageText](https://github.com/cnlile/huawei_modelart_experiments/blob/master/phase11/expands/cv2_haarcascade_dect_img.png?raw=true)


### 3.2 OpenCV的DNN网络检测

#### 3.2.1 DNN人脸识别简述
​		opencv的DNN主要的算法出自论文《SSD: Single Shot MultiBox Detector》，使用ResNet-10作为骨干网。          openCV3.3以上版本就有该分类器的模型的方式，openCV提供了两种不同的模型。一种是16位浮点数的caffe人脸模型(5.4MB)，另外一种是8bit量化后的tensorflow人脸模型(2.7MB)。量化是指比如可以用0~255表示原来32个bit所表示的精度，通过牺牲精度来降低每一个权值所需要占用的空间。通常情况深度学习模型会有冗余计算量，冗余性决定了参数个数。因此合理的量化网络也可保证精度的情况下减小模型的存储体积，不会对网络的精度造成影响。具体可以看看深度学习fine-tuning的论文。通常这种操作可以稍微降低精度，提高速度，大大减少模型体积。
opencv的了3.4版本，主要增强了dnn模块，特别是添加了对faster-rcnn的支持，并且带有openCL加速，效果还不错，基本上是向MTCNN看齐了。

#### 3.2.2 代码实现

```python
print('version:',cv2.__version__)
from cv2 import dnn
```
​		以上代码首先检测一下opencv版本，我们的版本结果是3.4。


```python
#TF
#modelFile = './models/opencv_face_detector_uint8.pb'
#configFile = './models/opencv_face_detector.pbtxt'
#net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
#CAFF
modelFile = './model/res10_300x300_ssd_iter_140000.caffemodel'
configFile = './model/deploy.prototxt'
net = cv2.dnn.readNetFromCaffe(configFile,modelFile)
```
​		这是两种方式，基于caff和tf的两种写法，我一开始还是先做的tf的方式，结果失败了，原因很简单。。。官方版本已经升级了，不支持3.4的版本了，我也是找不到tf的3.4的模型文件和配置文件啊。
然后，我开始做caff的实现，也是版本问题，幸运的是我在[https://stackoverflow.com/questions/54660069/opencv-deep-learning-face-detection-assertion-error-in-function-cvdnnconvol](https://stackoverflow.com/questions/54660069/opencv-deep-learning-face-detection-assertion-error-in-function-cvdnnconvol) 这里找到了问题根源和对应的模型文件和配置说明。。。如果有兴趣，可以打开看一下opencv在git上的deploy.prototxt文件和这里的文件，他们在配置上有layer里面bottom数据的差异。又是个版本兼容问题。
​		cv2.dnn.readNetFromCaffe方式读取配置文件和模型文件，这样就构成一个检测网络。

```python
inWidth = 300
inHeight = 300
confThreshold = 0.5

image = cv2.imread('./face_x01.jpg')

cols = image.shape[1]
rows = image.shape[0]
```
​		上面代码是加载输入图像并为图像构造输入blob，设置像素为300 × 300。并且读取图像文件的参数。confidence是一个阀值，默认为0.5，就是如果检测区域结果数值（这里的数值应该是一个数学距离）>0.5，确认该区域为人脸区域。

```python
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0,(300,300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
perf_stats = net.getPerfProfile()
print('Inference time, ms: %.2f' % (perf_stats[0] / cv2.getTickFrequency() * 1000))

Inference time, ms: 36.49
```
​		上面的代码，将图像调整到固定的300x300像素，然后将其规格化，设置检测参数，然后将图片装入检测器，对其进行检测。perf_stats是用来做检测时间的计算，最后检测这个图片，耗时36.49ms，也算不错的！

```python
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > confThreshold:
        x1 = int(detections[0, 0, i, 3] * cols)
        y1 = int(detections[0, 0, i, 4] * rows)
        x2 = int(detections[0, 0, i, 5] * cols)
        y2 = int(detections[0, 0, i, 6] * rows)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
 
from PIL import Image
Image.fromarray(image)
```
​		上面代码是将检测结果进行显示，还是利用cv2.rectangle在图像上描绘检测到的区域。检测结果是一组数据在detections.shape[2]内，每一组数据都是一个特征区域的计算结果，只有大于confidence的区域，才会被检测判断为人脸区域。
检测结果图如下：

![ImageText](https://github.com/cnlile/huawei_modelart_experiments/blob/master/phase11/expands/cv2_dnn_dect_img.png?raw=true)


## 4、总结
​		至此，完成了本期的拓展任务部分，感觉好累，踩了不少坑。原定应该早两个礼拜就做完的，结果拖延到现在，总算做好了。写文档说明的时间其实也不少，最后整理出这份文档，已经做了部分精简。其实做下来，个人感觉还是opencb的DNN模型最好，检测的最准确和高效。Dlib问题就是因为训练的图片都是80×80以上，导致不能辨认小脸，opencv的Haar 级联分类器快是快了，但是效果不好啊，如果脸稍微侧一下什么的，就检测失败了，opencv DNN则完全可以识别。
​		上次的mPA的拓展，有训练营的小伙伴说写的太简单扼要了，这次满足你们，原理和代码都覆盖到，代码还逐行讲解，这样你们满意了吧！
