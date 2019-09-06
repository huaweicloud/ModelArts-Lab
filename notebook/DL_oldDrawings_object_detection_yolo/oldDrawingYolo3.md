
# 古画yolov3物体检测

**下载数据集**


```python
!mkdir tarfile
from modelarts.session import Session
sess = Session()
for i in range(100):
    sess.download_data(bucket_path=f"gg-image/task/tarfile/{i}.tar.gz", path=f"tarfile/{i}.tar.gz")
```

    Successfully download file gg-image/task/tarfile/0.tar.gz from OBS to local tarfile/0.tar.gz
    Successfully download file gg-image/task/tarfile/1.tar.gz from OBS to local tarfile/1.tar.gz
    Successfully download file gg-image/task/tarfile/2.tar.gz from OBS to local tarfile/2.tar.gz
    Successfully download file gg-image/task/tarfile/3.tar.gz from OBS to local tarfile/3.tar.gz
    Successfully download file gg-image/task/tarfile/4.tar.gz from OBS to local tarfile/4.tar.gz
    Successfully download file gg-image/task/tarfile/5.tar.gz from OBS to local tarfile/5.tar.gz
    Successfully download file gg-image/task/tarfile/6.tar.gz from OBS to local tarfile/6.tar.gz
    Successfully download file gg-image/task/tarfile/7.tar.gz from OBS to local tarfile/7.tar.gz
    Successfully download file gg-image/task/tarfile/8.tar.gz from OBS to local tarfile/8.tar.gz
    Successfully download file gg-image/task/tarfile/9.tar.gz from OBS to local tarfile/9.tar.gz
    Successfully download file gg-image/task/tarfile/10.tar.gz from OBS to local tarfile/10.tar.gz
    Successfully download file gg-image/task/tarfile/11.tar.gz from OBS to local tarfile/11.tar.gz
    Successfully download file gg-image/task/tarfile/12.tar.gz from OBS to local tarfile/12.tar.gz
    Successfully download file gg-image/task/tarfile/13.tar.gz from OBS to local tarfile/13.tar.gz
    Successfully download file gg-image/task/tarfile/14.tar.gz from OBS to local tarfile/14.tar.gz
    Successfully download file gg-image/task/tarfile/15.tar.gz from OBS to local tarfile/15.tar.gz
    Successfully download file gg-image/task/tarfile/16.tar.gz from OBS to local tarfile/16.tar.gz
    Successfully download file gg-image/task/tarfile/17.tar.gz from OBS to local tarfile/17.tar.gz
    Successfully download file gg-image/task/tarfile/18.tar.gz from OBS to local tarfile/18.tar.gz
    Successfully download file gg-image/task/tarfile/19.tar.gz from OBS to local tarfile/19.tar.gz
    Successfully download file gg-image/task/tarfile/20.tar.gz from OBS to local tarfile/20.tar.gz
    Successfully download file gg-image/task/tarfile/21.tar.gz from OBS to local tarfile/21.tar.gz
    Successfully download file gg-image/task/tarfile/22.tar.gz from OBS to local tarfile/22.tar.gz
    Successfully download file gg-image/task/tarfile/23.tar.gz from OBS to local tarfile/23.tar.gz
    Successfully download file gg-image/task/tarfile/24.tar.gz from OBS to local tarfile/24.tar.gz
    Successfully download file gg-image/task/tarfile/25.tar.gz from OBS to local tarfile/25.tar.gz
    Successfully download file gg-image/task/tarfile/26.tar.gz from OBS to local tarfile/26.tar.gz
    Successfully download file gg-image/task/tarfile/27.tar.gz from OBS to local tarfile/27.tar.gz
    Successfully download file gg-image/task/tarfile/28.tar.gz from OBS to local tarfile/28.tar.gz
    Successfully download file gg-image/task/tarfile/29.tar.gz from OBS to local tarfile/29.tar.gz
    Successfully download file gg-image/task/tarfile/30.tar.gz from OBS to local tarfile/30.tar.gz
    Successfully download file gg-image/task/tarfile/31.tar.gz from OBS to local tarfile/31.tar.gz
    Successfully download file gg-image/task/tarfile/32.tar.gz from OBS to local tarfile/32.tar.gz
    Successfully download file gg-image/task/tarfile/33.tar.gz from OBS to local tarfile/33.tar.gz
    Successfully download file gg-image/task/tarfile/34.tar.gz from OBS to local tarfile/34.tar.gz
    Successfully download file gg-image/task/tarfile/35.tar.gz from OBS to local tarfile/35.tar.gz
    Successfully download file gg-image/task/tarfile/36.tar.gz from OBS to local tarfile/36.tar.gz
    Successfully download file gg-image/task/tarfile/37.tar.gz from OBS to local tarfile/37.tar.gz
    Successfully download file gg-image/task/tarfile/38.tar.gz from OBS to local tarfile/38.tar.gz
    Successfully download file gg-image/task/tarfile/39.tar.gz from OBS to local tarfile/39.tar.gz
    Successfully download file gg-image/task/tarfile/40.tar.gz from OBS to local tarfile/40.tar.gz
    Successfully download file gg-image/task/tarfile/41.tar.gz from OBS to local tarfile/41.tar.gz
    Successfully download file gg-image/task/tarfile/42.tar.gz from OBS to local tarfile/42.tar.gz
    Successfully download file gg-image/task/tarfile/43.tar.gz from OBS to local tarfile/43.tar.gz
    Successfully download file gg-image/task/tarfile/44.tar.gz from OBS to local tarfile/44.tar.gz
    Successfully download file gg-image/task/tarfile/45.tar.gz from OBS to local tarfile/45.tar.gz
    Successfully download file gg-image/task/tarfile/46.tar.gz from OBS to local tarfile/46.tar.gz
    Successfully download file gg-image/task/tarfile/47.tar.gz from OBS to local tarfile/47.tar.gz
    Successfully download file gg-image/task/tarfile/48.tar.gz from OBS to local tarfile/48.tar.gz
    Successfully download file gg-image/task/tarfile/49.tar.gz from OBS to local tarfile/49.tar.gz
    Successfully download file gg-image/task/tarfile/50.tar.gz from OBS to local tarfile/50.tar.gz
    Successfully download file gg-image/task/tarfile/51.tar.gz from OBS to local tarfile/51.tar.gz
    Successfully download file gg-image/task/tarfile/52.tar.gz from OBS to local tarfile/52.tar.gz
    Successfully download file gg-image/task/tarfile/53.tar.gz from OBS to local tarfile/53.tar.gz
    Successfully download file gg-image/task/tarfile/54.tar.gz from OBS to local tarfile/54.tar.gz
    Successfully download file gg-image/task/tarfile/55.tar.gz from OBS to local tarfile/55.tar.gz
    Successfully download file gg-image/task/tarfile/56.tar.gz from OBS to local tarfile/56.tar.gz
    Successfully download file gg-image/task/tarfile/57.tar.gz from OBS to local tarfile/57.tar.gz
    Successfully download file gg-image/task/tarfile/58.tar.gz from OBS to local tarfile/58.tar.gz
    Successfully download file gg-image/task/tarfile/59.tar.gz from OBS to local tarfile/59.tar.gz
    Successfully download file gg-image/task/tarfile/60.tar.gz from OBS to local tarfile/60.tar.gz
    Successfully download file gg-image/task/tarfile/61.tar.gz from OBS to local tarfile/61.tar.gz
    Successfully download file gg-image/task/tarfile/62.tar.gz from OBS to local tarfile/62.tar.gz
    Successfully download file gg-image/task/tarfile/63.tar.gz from OBS to local tarfile/63.tar.gz
    Successfully download file gg-image/task/tarfile/64.tar.gz from OBS to local tarfile/64.tar.gz
    Successfully download file gg-image/task/tarfile/65.tar.gz from OBS to local tarfile/65.tar.gz
    Successfully download file gg-image/task/tarfile/66.tar.gz from OBS to local tarfile/66.tar.gz
    Successfully download file gg-image/task/tarfile/67.tar.gz from OBS to local tarfile/67.tar.gz
    Successfully download file gg-image/task/tarfile/68.tar.gz from OBS to local tarfile/68.tar.gz
    Successfully download file gg-image/task/tarfile/69.tar.gz from OBS to local tarfile/69.tar.gz
    Successfully download file gg-image/task/tarfile/70.tar.gz from OBS to local tarfile/70.tar.gz
    Successfully download file gg-image/task/tarfile/71.tar.gz from OBS to local tarfile/71.tar.gz
    Successfully download file gg-image/task/tarfile/72.tar.gz from OBS to local tarfile/72.tar.gz
    Successfully download file gg-image/task/tarfile/73.tar.gz from OBS to local tarfile/73.tar.gz
    Successfully download file gg-image/task/tarfile/74.tar.gz from OBS to local tarfile/74.tar.gz
    Successfully download file gg-image/task/tarfile/75.tar.gz from OBS to local tarfile/75.tar.gz
    Successfully download file gg-image/task/tarfile/76.tar.gz from OBS to local tarfile/76.tar.gz
    Successfully download file gg-image/task/tarfile/77.tar.gz from OBS to local tarfile/77.tar.gz
    Successfully download file gg-image/task/tarfile/78.tar.gz from OBS to local tarfile/78.tar.gz
    Successfully download file gg-image/task/tarfile/79.tar.gz from OBS to local tarfile/79.tar.gz
    Successfully download file gg-image/task/tarfile/80.tar.gz from OBS to local tarfile/80.tar.gz
    Successfully download file gg-image/task/tarfile/81.tar.gz from OBS to local tarfile/81.tar.gz
    Successfully download file gg-image/task/tarfile/82.tar.gz from OBS to local tarfile/82.tar.gz
    Successfully download file gg-image/task/tarfile/83.tar.gz from OBS to local tarfile/83.tar.gz
    Successfully download file gg-image/task/tarfile/84.tar.gz from OBS to local tarfile/84.tar.gz
    Successfully download file gg-image/task/tarfile/85.tar.gz from OBS to local tarfile/85.tar.gz
    Successfully download file gg-image/task/tarfile/86.tar.gz from OBS to local tarfile/86.tar.gz
    Successfully download file gg-image/task/tarfile/87.tar.gz from OBS to local tarfile/87.tar.gz
    Successfully download file gg-image/task/tarfile/88.tar.gz from OBS to local tarfile/88.tar.gz
    Successfully download file gg-image/task/tarfile/89.tar.gz from OBS to local tarfile/89.tar.gz
    Successfully download file gg-image/task/tarfile/90.tar.gz from OBS to local tarfile/90.tar.gz
    Successfully download file gg-image/task/tarfile/91.tar.gz from OBS to local tarfile/91.tar.gz
    Successfully download file gg-image/task/tarfile/92.tar.gz from OBS to local tarfile/92.tar.gz
    Successfully download file gg-image/task/tarfile/93.tar.gz from OBS to local tarfile/93.tar.gz
    Successfully download file gg-image/task/tarfile/94.tar.gz from OBS to local tarfile/94.tar.gz
    Successfully download file gg-image/task/tarfile/95.tar.gz from OBS to local tarfile/95.tar.gz
    Successfully download file gg-image/task/tarfile/96.tar.gz from OBS to local tarfile/96.tar.gz
    Successfully download file gg-image/task/tarfile/97.tar.gz from OBS to local tarfile/97.tar.gz
    Successfully download file gg-image/task/tarfile/98.tar.gz from OBS to local tarfile/98.tar.gz
    Successfully download file gg-image/task/tarfile/99.tar.gz from OBS to local tarfile/99.tar.gz


**解压数据集**


```python
import os, tarfile
path = 'tarfile'
savefolder = 'data'
for filename in os.listdir(path):
    filepath = os.path.join(path, filename)
    tf = tarfile.open(filepath)
    tf.extractall(savefolder)
    tf.close()
    os.remove(filepath)   
```

**下载代码**


```python
# !rm yolov3.tar.gz
sess.download_data(bucket_path="gg-image/task/notebook/yolov3.tar.gz", path=f"./yolov3.tar.gz")
```

    rm: cannot remove 'yolov3.tar.gz': No such file or directory
    Successfully download file gg-image/task/notebook/yolov3.tar.gz from OBS to local ./yolov3.tar.gz


**解压代码**


```python
!tar -zxf yolov3.tar.gz
!rm yolov3.tar.gz
```

**显示图片**


```python
# 生成标签对应的颜色
from src.util import produceColors, classes, path
colors = produceColors(classes)
print('classes:', classes)
print('colors:', colors)
print('path:', path)
```

    classes: ['person', 'bird', 'flower']
    colors: {'person': (255, 0, 0), 'bird': (0, 255, 0), 'flower': (0, 0, 255)}
    path: /home/ma-user/work/data



```python
import os
from IPython.display import display
from src.util import getFiles, getImageFile, draw_image
# 随机选3张标注文件并显示
for xml_file in getFiles(path, 3):
    img_file = getImageFile(xml_file)
    display(draw_image(img_file, colors))
```


![png](output_11_0.png)



![png](output_11_1.png)



![png](output_11_2.png)


**将数据分成训练集和测试集**


```python
# 数据分成训练集和测试集时并不改变数据源的位置，而是通过建立trainFile和testFile2个文件来区分
# trainFile记录了所有训练集图片的路径和标注结果，testFile记录了所有测试集图片的路径和标注结果
from src.util import divide, saveAsText
trainFile = 'oldDrawingTrain.txt'
testFile = 'oldDrawingTest.txt'
# 在src.util给了个训练集和测试集使用的标准方法divide()
##### 对于需要进行拓展工作的，不满足以divide()默认方式分开训练集和验证集方式的可更改divide()的参数
##### divide接受2个参数；ratio: 训练集所占的比例，seed: 随机种子（一个整数或None），值为None时为随机生成
##### 比如需要取训练集为随机300个图片，剩余为测试集，考虑所有图片数为7939，可以写作
##### train, test = divide(300/7939, None)
train, test = divide()
saveAsText(train, trainFile)
saveAsText(test, testFile)
```

## 2. yolov3训练

**指定训练相关的文件路径等参数**


```python
from oldDrawingTrain import get_anchors
from src.util import classes
annotation_path = 'oldDrawingTrain.txt'
weights_path="./model_data/yolo_weights.h5"
log_dir = 'logs/'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = classes
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
num_anchors = len(anchors)
```

    Using TensorFlow backend.


**构建YOLO模型**


```python
import keras.backend as K
from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from keras.layers import Input, Lambda
from keras.models import Model

K.clear_session()

input_shape = (416,416)
image_input = Input(shape=(None, None, 3))
h, w = input_shape

y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], num_anchors//3, num_classes+5)) 
          for l in range(3)]
model_body = yolo_body(image_input, num_anchors//3, num_classes)

model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
    arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
    [*model_body.output, *y_true])

model = Model([model_body.input, *y_true], model_loss)
```

    WARNING:tensorflow:From /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.


    /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/keras/engine/saving.py:1009: UserWarning: Skipping loading of weights for layer conv2d_59 due to mismatch in shape ((1, 1, 1024, 24) vs (255, 1024, 1, 1)).
      weight_values[i].shape))
    /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/keras/engine/saving.py:1009: UserWarning: Skipping loading of weights for layer conv2d_59 due to mismatch in shape ((24,) vs (255,)).
      weight_values[i].shape))
    /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/keras/engine/saving.py:1009: UserWarning: Skipping loading of weights for layer conv2d_67 due to mismatch in shape ((1, 1, 512, 24) vs (255, 512, 1, 1)).
      weight_values[i].shape))
    /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/keras/engine/saving.py:1009: UserWarning: Skipping loading of weights for layer conv2d_67 due to mismatch in shape ((24,) vs (255,)).
      weight_values[i].shape))
    /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/keras/engine/saving.py:1009: UserWarning: Skipping loading of weights for layer conv2d_75 due to mismatch in shape ((1, 1, 256, 24) vs (255, 256, 1, 1)).
      weight_values[i].shape))
    /home/ma-user/anaconda3/envs/TensorFlow-1.13.1/lib/python3.6/site-packages/keras/engine/saving.py:1009: UserWarning: Skipping loading of weights for layer conv2d_75 due to mismatch in shape ((24,) vs (255,)).
      weight_values[i].shape))


**显示模型**


```python
model.summary()
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, None, None, 3 0                                            
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, None, None, 3 864         input_1[0][0]                    
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, None, None, 3 128         conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)       (None, None, None, 3 0           batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    zero_padding2d_1 (ZeroPadding2D (None, None, None, 3 0           leaky_re_lu_1[0][0]              
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, None, None, 6 18432       zero_padding2d_1[0][0]           
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, None, None, 6 256         conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)       (None, None, None, 6 0           batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, None, None, 3 2048        leaky_re_lu_2[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, None, None, 3 128         conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)       (None, None, None, 3 0           batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, None, None, 6 18432       leaky_re_lu_3[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, None, None, 6 256         conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_4 (LeakyReLU)       (None, None, None, 6 0           batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, None, None, 6 0           leaky_re_lu_2[0][0]              
                                                                     leaky_re_lu_4[0][0]              
    __________________________________________________________________________________________________
    zero_padding2d_2 (ZeroPadding2D (None, None, None, 6 0           add_1[0][0]                      
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, None, None, 1 73728       zero_padding2d_2[0][0]           
    __________________________________________________________________________________________________
    batch_normalization_5 (BatchNor (None, None, None, 1 512         conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_5 (LeakyReLU)       (None, None, None, 1 0           batch_normalization_5[0][0]      
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, None, None, 6 8192        leaky_re_lu_5[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_6 (BatchNor (None, None, None, 6 256         conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_6 (LeakyReLU)       (None, None, None, 6 0           batch_normalization_6[0][0]      
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, None, None, 1 73728       leaky_re_lu_6[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_7 (BatchNor (None, None, None, 1 512         conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_7 (LeakyReLU)       (None, None, None, 1 0           batch_normalization_7[0][0]      
    __________________________________________________________________________________________________
    add_2 (Add)                     (None, None, None, 1 0           leaky_re_lu_5[0][0]              
                                                                     leaky_re_lu_7[0][0]              
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, None, None, 6 8192        add_2[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_8 (BatchNor (None, None, None, 6 256         conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_8 (LeakyReLU)       (None, None, None, 6 0           batch_normalization_8[0][0]      
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, None, None, 1 73728       leaky_re_lu_8[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_9 (BatchNor (None, None, None, 1 512         conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    leaky_re_lu_9 (LeakyReLU)       (None, None, None, 1 0           batch_normalization_9[0][0]      
    __________________________________________________________________________________________________
    add_3 (Add)                     (None, None, None, 1 0           add_2[0][0]                      
                                                                     leaky_re_lu_9[0][0]              
    __________________________________________________________________________________________________
    zero_padding2d_3 (ZeroPadding2D (None, None, None, 1 0           add_3[0][0]                      
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, None, None, 2 294912      zero_padding2d_3[0][0]           
    __________________________________________________________________________________________________
    batch_normalization_10 (BatchNo (None, None, None, 2 1024        conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_10 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_10[0][0]     
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, None, None, 1 32768       leaky_re_lu_10[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_11 (BatchNo (None, None, None, 1 512         conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_11 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_11[0][0]     
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, None, None, 2 294912      leaky_re_lu_11[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_12 (BatchNo (None, None, None, 2 1024        conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_12 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_12[0][0]     
    __________________________________________________________________________________________________
    add_4 (Add)                     (None, None, None, 2 0           leaky_re_lu_10[0][0]             
                                                                     leaky_re_lu_12[0][0]             
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, None, None, 1 32768       add_4[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_13 (BatchNo (None, None, None, 1 512         conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_13 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_13[0][0]     
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, None, None, 2 294912      leaky_re_lu_13[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_14 (BatchNo (None, None, None, 2 1024        conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_14 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_14[0][0]     
    __________________________________________________________________________________________________
    add_5 (Add)                     (None, None, None, 2 0           add_4[0][0]                      
                                                                     leaky_re_lu_14[0][0]             
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, None, None, 1 32768       add_5[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_15 (BatchNo (None, None, None, 1 512         conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_15 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_15[0][0]     
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, None, None, 2 294912      leaky_re_lu_15[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_16 (BatchNo (None, None, None, 2 1024        conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_16 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_16[0][0]     
    __________________________________________________________________________________________________
    add_6 (Add)                     (None, None, None, 2 0           add_5[0][0]                      
                                                                     leaky_re_lu_16[0][0]             
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, None, None, 1 32768       add_6[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_17 (BatchNo (None, None, None, 1 512         conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_17 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_17[0][0]     
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, None, None, 2 294912      leaky_re_lu_17[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_18 (BatchNo (None, None, None, 2 1024        conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_18 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_18[0][0]     
    __________________________________________________________________________________________________
    add_7 (Add)                     (None, None, None, 2 0           add_6[0][0]                      
                                                                     leaky_re_lu_18[0][0]             
    __________________________________________________________________________________________________
    conv2d_19 (Conv2D)              (None, None, None, 1 32768       add_7[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_19 (BatchNo (None, None, None, 1 512         conv2d_19[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_19 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_19[0][0]     
    __________________________________________________________________________________________________
    conv2d_20 (Conv2D)              (None, None, None, 2 294912      leaky_re_lu_19[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_20 (BatchNo (None, None, None, 2 1024        conv2d_20[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_20 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_20[0][0]     
    __________________________________________________________________________________________________
    add_8 (Add)                     (None, None, None, 2 0           add_7[0][0]                      
                                                                     leaky_re_lu_20[0][0]             
    __________________________________________________________________________________________________
    conv2d_21 (Conv2D)              (None, None, None, 1 32768       add_8[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_21 (BatchNo (None, None, None, 1 512         conv2d_21[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_21 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_21[0][0]     
    __________________________________________________________________________________________________
    conv2d_22 (Conv2D)              (None, None, None, 2 294912      leaky_re_lu_21[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_22 (BatchNo (None, None, None, 2 1024        conv2d_22[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_22 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_22[0][0]     
    __________________________________________________________________________________________________
    add_9 (Add)                     (None, None, None, 2 0           add_8[0][0]                      
                                                                     leaky_re_lu_22[0][0]             
    __________________________________________________________________________________________________
    conv2d_23 (Conv2D)              (None, None, None, 1 32768       add_9[0][0]                      
    __________________________________________________________________________________________________
    batch_normalization_23 (BatchNo (None, None, None, 1 512         conv2d_23[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_23 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_23[0][0]     
    __________________________________________________________________________________________________
    conv2d_24 (Conv2D)              (None, None, None, 2 294912      leaky_re_lu_23[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_24 (BatchNo (None, None, None, 2 1024        conv2d_24[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_24 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_24[0][0]     
    __________________________________________________________________________________________________
    add_10 (Add)                    (None, None, None, 2 0           add_9[0][0]                      
                                                                     leaky_re_lu_24[0][0]             
    __________________________________________________________________________________________________
    conv2d_25 (Conv2D)              (None, None, None, 1 32768       add_10[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_25 (BatchNo (None, None, None, 1 512         conv2d_25[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_25 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_25[0][0]     
    __________________________________________________________________________________________________
    conv2d_26 (Conv2D)              (None, None, None, 2 294912      leaky_re_lu_25[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_26 (BatchNo (None, None, None, 2 1024        conv2d_26[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_26 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_26[0][0]     
    __________________________________________________________________________________________________
    add_11 (Add)                    (None, None, None, 2 0           add_10[0][0]                     
                                                                     leaky_re_lu_26[0][0]             
    __________________________________________________________________________________________________
    zero_padding2d_4 (ZeroPadding2D (None, None, None, 2 0           add_11[0][0]                     
    __________________________________________________________________________________________________
    conv2d_27 (Conv2D)              (None, None, None, 5 1179648     zero_padding2d_4[0][0]           
    __________________________________________________________________________________________________
    batch_normalization_27 (BatchNo (None, None, None, 5 2048        conv2d_27[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_27 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_27[0][0]     
    __________________________________________________________________________________________________
    conv2d_28 (Conv2D)              (None, None, None, 2 131072      leaky_re_lu_27[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_28 (BatchNo (None, None, None, 2 1024        conv2d_28[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_28 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_28[0][0]     
    __________________________________________________________________________________________________
    conv2d_29 (Conv2D)              (None, None, None, 5 1179648     leaky_re_lu_28[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_29 (BatchNo (None, None, None, 5 2048        conv2d_29[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_29 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_29[0][0]     
    __________________________________________________________________________________________________
    add_12 (Add)                    (None, None, None, 5 0           leaky_re_lu_27[0][0]             
                                                                     leaky_re_lu_29[0][0]             
    __________________________________________________________________________________________________
    conv2d_30 (Conv2D)              (None, None, None, 2 131072      add_12[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_30 (BatchNo (None, None, None, 2 1024        conv2d_30[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_30 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_30[0][0]     
    __________________________________________________________________________________________________
    conv2d_31 (Conv2D)              (None, None, None, 5 1179648     leaky_re_lu_30[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_31 (BatchNo (None, None, None, 5 2048        conv2d_31[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_31 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_31[0][0]     
    __________________________________________________________________________________________________
    add_13 (Add)                    (None, None, None, 5 0           add_12[0][0]                     
                                                                     leaky_re_lu_31[0][0]             
    __________________________________________________________________________________________________
    conv2d_32 (Conv2D)              (None, None, None, 2 131072      add_13[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_32 (BatchNo (None, None, None, 2 1024        conv2d_32[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_32 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_32[0][0]     
    __________________________________________________________________________________________________
    conv2d_33 (Conv2D)              (None, None, None, 5 1179648     leaky_re_lu_32[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_33 (BatchNo (None, None, None, 5 2048        conv2d_33[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_33 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_33[0][0]     
    __________________________________________________________________________________________________
    add_14 (Add)                    (None, None, None, 5 0           add_13[0][0]                     
                                                                     leaky_re_lu_33[0][0]             
    __________________________________________________________________________________________________
    conv2d_34 (Conv2D)              (None, None, None, 2 131072      add_14[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_34 (BatchNo (None, None, None, 2 1024        conv2d_34[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_34 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_34[0][0]     
    __________________________________________________________________________________________________
    conv2d_35 (Conv2D)              (None, None, None, 5 1179648     leaky_re_lu_34[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_35 (BatchNo (None, None, None, 5 2048        conv2d_35[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_35 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_35[0][0]     
    __________________________________________________________________________________________________
    add_15 (Add)                    (None, None, None, 5 0           add_14[0][0]                     
                                                                     leaky_re_lu_35[0][0]             
    __________________________________________________________________________________________________
    conv2d_36 (Conv2D)              (None, None, None, 2 131072      add_15[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_36 (BatchNo (None, None, None, 2 1024        conv2d_36[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_36 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_36[0][0]     
    __________________________________________________________________________________________________
    conv2d_37 (Conv2D)              (None, None, None, 5 1179648     leaky_re_lu_36[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_37 (BatchNo (None, None, None, 5 2048        conv2d_37[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_37 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_37[0][0]     
    __________________________________________________________________________________________________
    add_16 (Add)                    (None, None, None, 5 0           add_15[0][0]                     
                                                                     leaky_re_lu_37[0][0]             
    __________________________________________________________________________________________________
    conv2d_38 (Conv2D)              (None, None, None, 2 131072      add_16[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_38 (BatchNo (None, None, None, 2 1024        conv2d_38[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_38 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_38[0][0]     
    __________________________________________________________________________________________________
    conv2d_39 (Conv2D)              (None, None, None, 5 1179648     leaky_re_lu_38[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_39 (BatchNo (None, None, None, 5 2048        conv2d_39[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_39 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_39[0][0]     
    __________________________________________________________________________________________________
    add_17 (Add)                    (None, None, None, 5 0           add_16[0][0]                     
                                                                     leaky_re_lu_39[0][0]             
    __________________________________________________________________________________________________
    conv2d_40 (Conv2D)              (None, None, None, 2 131072      add_17[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_40 (BatchNo (None, None, None, 2 1024        conv2d_40[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_40 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_40[0][0]     
    __________________________________________________________________________________________________
    conv2d_41 (Conv2D)              (None, None, None, 5 1179648     leaky_re_lu_40[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_41 (BatchNo (None, None, None, 5 2048        conv2d_41[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_41 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_41[0][0]     
    __________________________________________________________________________________________________
    add_18 (Add)                    (None, None, None, 5 0           add_17[0][0]                     
                                                                     leaky_re_lu_41[0][0]             
    __________________________________________________________________________________________________
    conv2d_42 (Conv2D)              (None, None, None, 2 131072      add_18[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_42 (BatchNo (None, None, None, 2 1024        conv2d_42[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_42 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_42[0][0]     
    __________________________________________________________________________________________________
    conv2d_43 (Conv2D)              (None, None, None, 5 1179648     leaky_re_lu_42[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_43 (BatchNo (None, None, None, 5 2048        conv2d_43[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_43 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_43[0][0]     
    __________________________________________________________________________________________________
    add_19 (Add)                    (None, None, None, 5 0           add_18[0][0]                     
                                                                     leaky_re_lu_43[0][0]             
    __________________________________________________________________________________________________
    zero_padding2d_5 (ZeroPadding2D (None, None, None, 5 0           add_19[0][0]                     
    __________________________________________________________________________________________________
    conv2d_44 (Conv2D)              (None, None, None, 1 4718592     zero_padding2d_5[0][0]           
    __________________________________________________________________________________________________
    batch_normalization_44 (BatchNo (None, None, None, 1 4096        conv2d_44[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_44 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_44[0][0]     
    __________________________________________________________________________________________________
    conv2d_45 (Conv2D)              (None, None, None, 5 524288      leaky_re_lu_44[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_45 (BatchNo (None, None, None, 5 2048        conv2d_45[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_45 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_45[0][0]     
    __________________________________________________________________________________________________
    conv2d_46 (Conv2D)              (None, None, None, 1 4718592     leaky_re_lu_45[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_46 (BatchNo (None, None, None, 1 4096        conv2d_46[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_46 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_46[0][0]     
    __________________________________________________________________________________________________
    add_20 (Add)                    (None, None, None, 1 0           leaky_re_lu_44[0][0]             
                                                                     leaky_re_lu_46[0][0]             
    __________________________________________________________________________________________________
    conv2d_47 (Conv2D)              (None, None, None, 5 524288      add_20[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_47 (BatchNo (None, None, None, 5 2048        conv2d_47[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_47 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_47[0][0]     
    __________________________________________________________________________________________________
    conv2d_48 (Conv2D)              (None, None, None, 1 4718592     leaky_re_lu_47[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_48 (BatchNo (None, None, None, 1 4096        conv2d_48[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_48 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_48[0][0]     
    __________________________________________________________________________________________________
    add_21 (Add)                    (None, None, None, 1 0           add_20[0][0]                     
                                                                     leaky_re_lu_48[0][0]             
    __________________________________________________________________________________________________
    conv2d_49 (Conv2D)              (None, None, None, 5 524288      add_21[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_49 (BatchNo (None, None, None, 5 2048        conv2d_49[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_49 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_49[0][0]     
    __________________________________________________________________________________________________
    conv2d_50 (Conv2D)              (None, None, None, 1 4718592     leaky_re_lu_49[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_50 (BatchNo (None, None, None, 1 4096        conv2d_50[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_50 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_50[0][0]     
    __________________________________________________________________________________________________
    add_22 (Add)                    (None, None, None, 1 0           add_21[0][0]                     
                                                                     leaky_re_lu_50[0][0]             
    __________________________________________________________________________________________________
    conv2d_51 (Conv2D)              (None, None, None, 5 524288      add_22[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_51 (BatchNo (None, None, None, 5 2048        conv2d_51[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_51 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_51[0][0]     
    __________________________________________________________________________________________________
    conv2d_52 (Conv2D)              (None, None, None, 1 4718592     leaky_re_lu_51[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_52 (BatchNo (None, None, None, 1 4096        conv2d_52[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_52 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_52[0][0]     
    __________________________________________________________________________________________________
    add_23 (Add)                    (None, None, None, 1 0           add_22[0][0]                     
                                                                     leaky_re_lu_52[0][0]             
    __________________________________________________________________________________________________
    conv2d_53 (Conv2D)              (None, None, None, 5 524288      add_23[0][0]                     
    __________________________________________________________________________________________________
    batch_normalization_53 (BatchNo (None, None, None, 5 2048        conv2d_53[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_53 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_53[0][0]     
    __________________________________________________________________________________________________
    conv2d_54 (Conv2D)              (None, None, None, 1 4718592     leaky_re_lu_53[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_54 (BatchNo (None, None, None, 1 4096        conv2d_54[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_54 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_54[0][0]     
    __________________________________________________________________________________________________
    conv2d_55 (Conv2D)              (None, None, None, 5 524288      leaky_re_lu_54[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_55 (BatchNo (None, None, None, 5 2048        conv2d_55[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_55 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_55[0][0]     
    __________________________________________________________________________________________________
    conv2d_56 (Conv2D)              (None, None, None, 1 4718592     leaky_re_lu_55[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_56 (BatchNo (None, None, None, 1 4096        conv2d_56[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_56 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_56[0][0]     
    __________________________________________________________________________________________________
    conv2d_57 (Conv2D)              (None, None, None, 5 524288      leaky_re_lu_56[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_57 (BatchNo (None, None, None, 5 2048        conv2d_57[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_57 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_57[0][0]     
    __________________________________________________________________________________________________
    conv2d_60 (Conv2D)              (None, None, None, 2 131072      leaky_re_lu_57[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_59 (BatchNo (None, None, None, 2 1024        conv2d_60[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_59 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_59[0][0]     
    __________________________________________________________________________________________________
    up_sampling2d_1 (UpSampling2D)  (None, None, None, 2 0           leaky_re_lu_59[0][0]             
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, None, None, 7 0           up_sampling2d_1[0][0]            
                                                                     add_19[0][0]                     
    __________________________________________________________________________________________________
    conv2d_61 (Conv2D)              (None, None, None, 2 196608      concatenate_1[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_60 (BatchNo (None, None, None, 2 1024        conv2d_61[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_60 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_60[0][0]     
    __________________________________________________________________________________________________
    conv2d_62 (Conv2D)              (None, None, None, 5 1179648     leaky_re_lu_60[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_61 (BatchNo (None, None, None, 5 2048        conv2d_62[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_61 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_61[0][0]     
    __________________________________________________________________________________________________
    conv2d_63 (Conv2D)              (None, None, None, 2 131072      leaky_re_lu_61[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_62 (BatchNo (None, None, None, 2 1024        conv2d_63[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_62 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_62[0][0]     
    __________________________________________________________________________________________________
    conv2d_64 (Conv2D)              (None, None, None, 5 1179648     leaky_re_lu_62[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_63 (BatchNo (None, None, None, 5 2048        conv2d_64[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_63 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_63[0][0]     
    __________________________________________________________________________________________________
    conv2d_65 (Conv2D)              (None, None, None, 2 131072      leaky_re_lu_63[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_64 (BatchNo (None, None, None, 2 1024        conv2d_65[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_64 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_64[0][0]     
    __________________________________________________________________________________________________
    conv2d_68 (Conv2D)              (None, None, None, 1 32768       leaky_re_lu_64[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_66 (BatchNo (None, None, None, 1 512         conv2d_68[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_66 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_66[0][0]     
    __________________________________________________________________________________________________
    up_sampling2d_2 (UpSampling2D)  (None, None, None, 1 0           leaky_re_lu_66[0][0]             
    __________________________________________________________________________________________________
    concatenate_2 (Concatenate)     (None, None, None, 3 0           up_sampling2d_2[0][0]            
                                                                     add_11[0][0]                     
    __________________________________________________________________________________________________
    conv2d_69 (Conv2D)              (None, None, None, 1 49152       concatenate_2[0][0]              
    __________________________________________________________________________________________________
    batch_normalization_67 (BatchNo (None, None, None, 1 512         conv2d_69[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_67 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_67[0][0]     
    __________________________________________________________________________________________________
    conv2d_70 (Conv2D)              (None, None, None, 2 294912      leaky_re_lu_67[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_68 (BatchNo (None, None, None, 2 1024        conv2d_70[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_68 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_68[0][0]     
    __________________________________________________________________________________________________
    conv2d_71 (Conv2D)              (None, None, None, 1 32768       leaky_re_lu_68[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_69 (BatchNo (None, None, None, 1 512         conv2d_71[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_69 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_69[0][0]     
    __________________________________________________________________________________________________
    conv2d_72 (Conv2D)              (None, None, None, 2 294912      leaky_re_lu_69[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_70 (BatchNo (None, None, None, 2 1024        conv2d_72[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_70 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_70[0][0]     
    __________________________________________________________________________________________________
    conv2d_73 (Conv2D)              (None, None, None, 1 32768       leaky_re_lu_70[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_71 (BatchNo (None, None, None, 1 512         conv2d_73[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_71 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_71[0][0]     
    __________________________________________________________________________________________________
    conv2d_58 (Conv2D)              (None, None, None, 1 4718592     leaky_re_lu_57[0][0]             
    __________________________________________________________________________________________________
    conv2d_66 (Conv2D)              (None, None, None, 5 1179648     leaky_re_lu_64[0][0]             
    __________________________________________________________________________________________________
    conv2d_74 (Conv2D)              (None, None, None, 2 294912      leaky_re_lu_71[0][0]             
    __________________________________________________________________________________________________
    batch_normalization_58 (BatchNo (None, None, None, 1 4096        conv2d_58[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_65 (BatchNo (None, None, None, 5 2048        conv2d_66[0][0]                  
    __________________________________________________________________________________________________
    batch_normalization_72 (BatchNo (None, None, None, 2 1024        conv2d_74[0][0]                  
    __________________________________________________________________________________________________
    leaky_re_lu_58 (LeakyReLU)      (None, None, None, 1 0           batch_normalization_58[0][0]     
    __________________________________________________________________________________________________
    leaky_re_lu_65 (LeakyReLU)      (None, None, None, 5 0           batch_normalization_65[0][0]     
    __________________________________________________________________________________________________
    leaky_re_lu_72 (LeakyReLU)      (None, None, None, 2 0           batch_normalization_72[0][0]     
    __________________________________________________________________________________________________
    conv2d_59 (Conv2D)              (None, None, None, 2 24600       leaky_re_lu_58[0][0]             
    __________________________________________________________________________________________________
    conv2d_67 (Conv2D)              (None, None, None, 2 12312       leaky_re_lu_65[0][0]             
    __________________________________________________________________________________________________
    conv2d_75 (Conv2D)              (None, None, None, 2 6168        leaky_re_lu_72[0][0]             
    __________________________________________________________________________________________________
    input_2 (InputLayer)            (None, 13, 13, 3, 8) 0                                            
    __________________________________________________________________________________________________
    input_3 (InputLayer)            (None, 26, 26, 3, 8) 0                                            
    __________________________________________________________________________________________________
    input_4 (InputLayer)            (None, 52, 52, 3, 8) 0                                            
    __________________________________________________________________________________________________
    yolo_loss (Lambda)              (None, 1)            0           conv2d_59[0][0]                  
                                                                     conv2d_67[0][0]                  
                                                                     conv2d_75[0][0]                  
                                                                     input_2[0][0]                    
                                                                     input_3[0][0]                    
                                                                     input_4[0][0]                    
    ==================================================================================================
    Total params: 61,587,112
    Trainable params: 61,534,504
    Non-trainable params: 52,608
    __________________________________________________________________________________________________


**训练callbacks定义**


```python
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
```

**读取数据**


```python
import numpy as np
val_split = 0.1
with open(annotation_path) as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val
```

**读取数据函数**


```python
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)
```

**训练周期数设定**


```python
import os
os.makedirs(log_dir, exist_ok=True)
frozen_train_epoch = 5
fine_tuning_epoch = 5
```

**冻结时训练参数**


```python
from keras.optimizers import Adam
from yolo3.utils import get_random_data 
if frozen_train_epoch:
    model.compile(optimizer=Adam(lr=1e-3), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    batch_size = 8
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=frozen_train_epoch,
            initial_epoch=0,
            callbacks=[])
    model.save_weights(log_dir + 'trained_weights_stage_1.h5')
```

    Train on 4287 samples, val on 476 samples, with batch size 8.
    Epoch 1/5
    535/535 [==============================] - 358s 669ms/step - loss: 121.2328 - val_loss: 43.5569
    Epoch 2/5
    535/535 [==============================] - 332s 621ms/step - loss: 45.2910 - val_loss: 38.0415
    Epoch 3/5
    535/535 [==============================] - 335s 626ms/step - loss: 41.9102 - val_loss: 37.2522
    Epoch 4/5
    535/535 [==============================] - 339s 634ms/step - loss: 40.9120 - val_loss: 35.4150
    Epoch 5/5
    535/535 [==============================] - 337s 630ms/step - loss: 59.7642 - val_loss: 454135.6431


**fine tuning**


```python
for i in range(len(model.layers)):
    model.layers[i].trainable = True
model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
print('Unfreeze all of the layers.')

batch_size = 2 # note that more GPU memory is required after unfreezing the body
print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
    steps_per_epoch=max(1, num_train//batch_size),
    validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
    validation_steps=max(1, num_val//batch_size),
    epochs=frozen_train_epoch+fine_tuning_epoch,
    initial_epoch=frozen_train_epoch,
    callbacks=[reduce_lr, early_stopping])
model.save_weights(log_dir + 'trained_weights_final.h5')
```

    Unfreeze all of the layers.
    Train on 4287 samples, val on 476 samples, with batch size 2.
    Epoch 6/10
    2143/2143 [==============================] - 401s 187ms/step - loss: 63.9933 - val_loss: 81.5493
    Epoch 7/10
    2143/2143 [==============================] - 378s 176ms/step - loss: 55.0096 - val_loss: 182.4871
    Epoch 8/10
    2143/2143 [==============================] - 378s 176ms/step - loss: 50.0493 - val_loss: 52.6781
    Epoch 9/10
    2143/2143 [==============================] - 379s 177ms/step - loss: 47.5252 - val_loss: 47.6857
    Epoch 10/10
    2142/2143 [============================>.] - ETA: 0s - loss: 45.6222

## 训练结果展示

**展示测试集前几张图片**


```python
# 在预测之前，需要将（可能发生的）之前预测的结果删除
!rm input/detection-results/*
!rm input/ground-truth/*
!rm -rf results
!rm -rf test_result
!mkdir -p input/detection-results
!mkdir -p input/ground-truth
!mkdir test_result
```

    rm: cannot remove 'input/detection-results/*': No such file or directory
    rm: cannot remove 'input/ground-truth/*': No such file or directory



```python
import random, os 
from PIL import Image
from yolo_video_ import detect_img_many, path_construct
class_path = 'model_data/oldDrawingClasses.txt'
log_dir = 'logs/'
anchors_path = 'model_data/yolo_anchors.txt'
h5_file = log_dir + 'trained_weights_final.h5'
# 随机在测试集中选取10张图片进行标注
inputs, outputs, gt_files, dr_files = path_construct(10)
detect_img_many(h5_file, anchors_path, class_path, inputs, outputs, gt_files, dr_files)
# 显示标注结果
from IPython.display import display
for output in outputs:
    display(Image.open(output))
```

    logs/trained_weights_final.h5 model, anchors, and classes loaded.
    (416, 416, 3)
    Found 0 boxes for img
    3.972397169098258
    (416, 416, 3)
    Found 0 boxes for img
    0.055635412223637104
    (416, 416, 3)
    Found 0 boxes for img
    0.05384439788758755
    (416, 416, 3)
    Found 0 boxes for img
    0.04201371408998966
    (416, 416, 3)
    Found 0 boxes for img
    0.04886780818924308
    (416, 416, 3)
    Found 0 boxes for img
    0.04431925993412733
    (416, 416, 3)
    Found 0 boxes for img
    0.08189580822363496
    (416, 416, 3)
    Found 0 boxes for img
    0.053870758041739464
    (416, 416, 3)
    Found 1 boxes for img
    person 0.38 (255, 453) (313, 586)
    0.05450587905943394
    (416, 416, 3)
    Found 0 boxes for img
    0.04383068485185504



![png](output_36_1.png)



![png](output_36_2.png)



![png](output_36_3.png)



![png](output_36_4.png)



![png](output_36_5.png)



![png](output_36_6.png)



![png](output_36_7.png)



![png](output_36_8.png)



![png](output_36_9.png)



![png](output_36_10.png)



```python

```
