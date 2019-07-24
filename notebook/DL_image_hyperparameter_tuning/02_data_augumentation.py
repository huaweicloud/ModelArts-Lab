import subprocess

subprocess.run(['pip', 'install', '--upgrade', 'keras_applications==1.0.6', 'keras==2.2.4'])

import os
if os.path.exists('./data') == False:
    from modelarts.session import Session
    session = Session()

    session.download_data(
    bucket_path="modelarts-labs/notebook/DL_image_recognition/image_recognition.tar.gz",
    path="./image_recognition.tar.gz")

    # 使用tar命令解压资源包
    subprocess.run(['tar', 'xf', './image_recognition.tar.gz'])

    # 清理压缩包
    subprocess.run(['rm', '-f', './image_recognition.tar.gz'])
from keras.preprocessing.image import ImageDataGenerator
from glob import glob

ROWS = 224
COLS = 224

if os.path.exists('./data-for-gen') == False:
    subprocess.run(['mkdir', '-p', 'data-for-gen/train/dog'])
    subprocess.run(['mkdir', '-p', 'data-for-gen/train/cat'])
    subprocess.run(['mkdir', '-p', 'data-for-gen/val/dog'])
    subprocess.run(['mkdir', '-p', 'data-for-gen/val/cat'])

    DATA_DIR = './data-for-gen/' # 数据集路径
    dog_glob = glob('./data/dog*.jpg')
    cat_glob = glob('./data/cat*.jpg')

    # 我们使用25%的数据作为验证集：
    val_split = 0.25


    index = int(len(dog_glob) * val_split)

    def gen_lnk_cmds(class_name, class_glob):
        cmds = ''
        for i in range(len(class_glob)):
            filename = os.path.basename(class_glob[i])

            src_path = os.path.realpath(class_glob[i])
            sample_type = 'train' if i > index else 'val'
            lnk_path = os.path.realpath('./data-for-gen/{}/{}/{}'.format(sample_type, class_name, filename))
            cmds = cmds + 'ln -s {} {}\n'.format(src_path, lnk_path)

        return cmds

    # 准备dog图片
    print('prepare dog images for data augumentation')
    with open('./tmp_gen_dogs.sh', 'w') as f:
        link_cmds = gen_lnk_cmds('dog', dog_glob)
        f.write(link_cmds)
        subprocess.run(['sh', './tmp_gen_dogs.sh'])
        subprocess.run(['rm', './tmp_gen_dogs.sh'])

    # 准备cat图片
    print('prepare cat images for data augumentation')
    with open('./tmp_gen_cats.sh', 'w') as f:
        link_cmds = gen_lnk_cmds('cat', cat_glob)
        f.write(link_cmds)
        subprocess.run(['sh', './tmp_gen_cats.sh'])
        subprocess.run(['rm', './tmp_gen_cats.sh'])

# Need this to plot without X-server
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import matplotlib.pyplot as plt
import numpy as np

train_datagen = temp_val_datagen = ImageDataGenerator(
    rescale=1.0/255,  # ImageDataGenerator使用[0-1]表示RGB色值，加入rescale以正常显示图片
    zoom_range=0.1, # 图片缩放范围
    horizontal_flip=False) # 是否随机水平翻转图片

val_datagen = temp_val_datagen = ImageDataGenerator(
    rescale=1.0/255,
    zoom_range=0.1,
    horizontal_flip=False)
train_generator = train_datagen.flow_from_directory('./data-for-gen/train', 
                                                    target_size=(ROWS, COLS), batch_size=16, class_mode='binary')

val_generator = val_datagen.flow_from_directory('./data-for-gen/val', 
                                                target_size=(ROWS, COLS), batch_size=16, class_mode='binary')
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import RMSprop

optimizer = RMSprop(lr=1e-4, decay=1e-6) # 优化器使用RMSprop, 设置学习率是1e-4
objective = 'binary_crossentropy' # loss 函数使用交叉熵

base_model = VGG16(weights=None, include_top=False, input_shape=(COLS, ROWS, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)

output = Dense(1, activation='sigmoid')(x)

model = Model(input=base_model.input, output=output)
model.summary()

model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
epochs = 100 # 训练轮数

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# early stopping策略
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')   
mcp = ModelCheckpoint('aug.weights.{epoch:03d}_{acc:.4f}_{val_acc:.4f}.h5', 
                      monitor='val_acc', save_best_only=True, save_weights_only=True, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=1e-9, verbose=1)
# 开始训练
hist = model.fit_generator(
    epochs=2,
    verbose=1,

    generator=train_generator,
    steps_per_epoch=int(0.2*len(train_generator)),
    validation_data=val_generator,
    validation_steps=int(0.2*len(val_generator)),
    shuffle=True,
    callbacks=[early_stopping, mcp, reduce_lr])
import matplotlib.pyplot as plt
# 绘制训练 & 验证的准确率值
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('./data-aug-plot.png')
