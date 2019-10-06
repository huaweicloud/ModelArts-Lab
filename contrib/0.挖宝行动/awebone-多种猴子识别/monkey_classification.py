# from modelarts.session import Session
# session = Session()

# bucket_path="ai-awe-001/wabao/data.zip"

# session.download_data(bucket_path=bucket_path,path='./data.zip')

import os
import cv2
import glob
import h5py
import shutil
import imgaug as aug
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential, Model, load_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import backend as K
import tensorflow as tf

training_data = Path('./training/') 
validation_data = Path('./validation/') 
labels_path = Path('./monkey_labels.txt')

labels_info = []

lines = labels_path.read_text().strip().splitlines()[1:]
for line in lines:
    line = line.split(',')
    line = [x.strip(' \n\t\r') for x in line]
    line[3], line[4] = int(line[3]), int(line[4])
    line = tuple(line)
    labels_info.append(line)
    
labels_info = pd.DataFrame(labels_info, columns=['Label', 'Latin Name', 'Common Name', 
                                                 'Train Images', 'Validation Images'], index=None)
print(labels_info.head(10))

labels_dict= {'n0':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'n5':5, 'n6':6, 'n7':7, 'n8':8, 'n9':9}

names_dict = dict(zip(labels_dict.values(), labels_info["Common Name"]))
print(names_dict)

train_df = []
for folder in os.listdir(training_data):
    imgs_path = training_data / folder
    
    imgs = sorted(imgs_path.glob('*.jpg'))
    
    for img_name in imgs:
        train_df.append((str(img_name), labels_dict[folder]))


train_df = pd.DataFrame(train_df, columns=['image', 'label'], index=None)
train_df = train_df.sample(frac=1.).reset_index(drop=True)

valid_df = []
for folder in os.listdir(validation_data):
    imgs_path = validation_data / folder
    imgs = sorted(imgs_path.glob('*.jpg'))
    for img_name in imgs:
        valid_df.append((str(img_name), labels_dict[folder]))

        
valid_df = pd.DataFrame(valid_df, columns=['image', 'label'], index=None)

valid_df = valid_df.sample(frac=1.).reset_index(drop=True)

print("Number of traininng samples: ", len(train_df))
print("Number of validation samples: ", len(valid_df))

print("\n",train_df.head(), "\n")
print("\n", valid_df.head())

img_rows, img_cols, img_channels = 224,224,3
 
batch_size=8

nb_classes=10

def get_base_model():
    base_model = ResNet50(input_shape=(img_rows, img_cols, img_channels), weights='imagenet', include_top=True)
    return base_model

base_model = get_base_model()
base_model_output = base_model.layers[-2].output
x = Dropout(0.7,name='drop2')(base_model_output)
output = Dense(10, activation='softmax', name='fc3')(x)
model = Model(base_model.input, output)

for layer in base_model.layers[:-1]:
    layer.trainable=False

optimizer = RMSprop(0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
seq = iaa.OneOf([
    iaa.Fliplr(), 
    iaa.Affine(rotate=20), 
    iaa.Multiply((1.2, 1.5))]) 

def data_generator(data, batch_size, is_validation_data=False):
    n = len(data)
    nb_batches = int(np.ceil(n/batch_size))

    indices = np.arange(n)
    batch_data = np.zeros((batch_size, img_rows, img_cols, img_channels), dtype=np.float32)
    batch_labels = np.zeros((batch_size, nb_classes), dtype=np.float32)
    
    while True:
        if not is_validation_data:
            np.random.shuffle(indices)
            
        for i in range(nb_batches):
            next_batch_indices = indices[i*batch_size:(i+1)*batch_size]
            for j, idx in enumerate(next_batch_indices):
                img = cv2.imread(data.iloc[idx]["image"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                label = data.iloc[idx]["label"]
                
                if not is_validation_data:
                    img = seq.augment_image(img)
                
                img = cv2.resize(img, (img_rows, img_cols)).astype(np.float32)
                batch_data[j] = img
                batch_labels[j] = to_categorical(label,num_classes=nb_classes)
            
            batch_data = preprocess_input(batch_data)
            yield batch_data, batch_labels
 
train_data_gen = data_generator(train_df, batch_size)

valid_data_gen = data_generator(valid_df, batch_size, is_validation_data=True)
es = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

chkpt = ModelCheckpoint(filepath="model1", save_best_only=True)

nb_train_steps = int(np.ceil(len(train_df)/batch_size))
nb_valid_steps = int(np.ceil(len(valid_df)/batch_size))

history = model.fit_generator(train_data_gen, 
                              epochs=100, 
                              steps_per_epoch=nb_train_steps, 
                              validation_data=valid_data_gen, 
                              validation_steps=nb_valid_steps,
                              callbacks=[es,chkpt])

train_acc = history.history['acc']
valid_acc = history.history['val_acc']

train_loss = history.history['loss']
valid_loss = history.history['val_loss']

xvalues = np.arange(len(train_acc))

f,ax = plt.subplots(1,2, figsize=(10,5))
ax[0].plot(xvalues, train_loss)
ax[0].plot(xvalues, valid_loss)
ax[0].set_title("Loss curve")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("loss")
ax[0].legend(['train', 'validation'])

ax[1].plot(xvalues, train_acc)
ax[1].plot(xvalues, valid_acc)
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("accuracy")
ax[1].legend(['train', 'validation'])

plt.show()

valid_loss, valid_acc = model.evaluate_generator(valid_data_gen, steps=nb_valid_steps)
print(f"Final validation accuracy: {valid_acc*100:.2f}%")
outputs = [layer.output for layer in model.layers[1:]]

vis_model = Model(model.input, outputs)

vis_model.summary()
layer_names = []
for layer in outputs:
    layer_names.append(layer.name.split("/")[0])

    
print("Layers going to be used for visualization: ")
print(layer_names)

def show_random_sample(idx):
    sample_image = cv2.imread(valid_df.iloc[idx]['image'])
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    sample_image = cv2.resize(sample_image, (img_rows, img_cols))
    sample_label = valid_df.iloc[idx]["label"]
    
    sample_image_processed = np.expand_dims(sample_image, axis=0)
    sample_image_processed = preprocess_input(sample_image_processed)
    
    activations = vis_model.predict(sample_image_processed)
    
    pred_label = np.argmax(model.predict(sample_image_processed), axis=-1)[0]

    plt.imshow(sample_image)
    print(f"True label: {sample_label} \n Predicted label: {pred_label}")

    plt.show()
    
    return activations

from random import randint
n = randint(1,300)
activations= show_random_sample(n)

