import os, cv2, random
import numpy as np

from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.callbacks import Callback, EarlyStopping, TensorBoard
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.utils import np_utils

import tensorflow as tf
from moxing.framework import file
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import backend as K

logging.basicConfig(level=logging.INFO)
K.set_image_data_format('channels_last')

tf.flags.DEFINE_integer('max_epochs', 10, 'number of training iterations.')
tf.flags.DEFINE_string('data_url', '/cache/data', 'dataset directory.')
tf.flags.DEFINE_string('train_url', '/cache/output', 'saved model directory.')
tf.flags.DEFINE_integer('batch_size', 32, 'number of training iterations.')
tf.flags.DEFINE_integer('num_gpus', 1, 'gpu nums.')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'learning rate.')

FLAGS = tf.flags.FLAGS

data_url = FLAGS.data_url
train_url = FLAGS.train_url
max_epochs = FLAGS.max_epochs
batch_size = FLAGS.batch_size
learning_rate = FLAGS.learning_rate

# local path
local_data_path = '/cache/data/'
local_output_path = '/cache/output/'
model_output_path = os.path.join(local_output_path, "model")

if not os.path.exists(local_data_path):
    os.makedirs(local_data_path)

if not os.path.exists(local_output_path):
    os.makedirs(local_output_path)

# copy data to local
file.copy_parallel(data_url, local_data_path)

# untar data
train_file = os.path.join(local_data_path, "dog_and_cat_25000.tar.gz")
os.system('tar xf %s -C %s' % (train_file, local_data_path))

dogcat_data_path = os.path.join(local_data_path, 'data')

ROWS = 128
COLS = 128
CHANNELS = 3

image_file_names = [os.path.join(dogcat_data_path, i) for i in os.listdir(dogcat_data_path)]

random.shuffle(image_file_names)

def read_image(file_path):
    # cv2.IMREAD_GRAYSCALE
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
    return data


# read data
images_data = prep_data(image_file_names)

# get labels

num_train_samples = len(image_file_names)
num_classes = 2
labels = []

index = 0
for filename in image_file_names:
    if 'dog' in filename:
        labels.append(1)
    elif 'cat' in filename:
        labels.append(0)

labels = np_utils.to_categorical(labels, num_classes)
train_data, test_data, train_label, test_label = train_test_split(images_data, labels, test_size=0.25, random_state=10)

optimizer = RMSprop(lr=learning_rate)
objective = 'binary_crossentropy'

def load_model():
    base_model = VGG16(include_top=False, weights=None, input_shape=(ROWS, COLS, CHANNELS), pooling='avg')
    prediction_layer = Dense(2, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=prediction_layer)
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model

model = load_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
tensorBoard = TensorBoard(log_dir=local_output_path)

def run_train():
    model.fit(
        train_data,
        train_label,
        batch_size=batch_size,
        epochs=max_epochs,
        validation_split=0.25,
        verbose=2,
        shuffle=True,
        callbacks=[early_stopping, tensorBoard])

# run training
run_train()

# calculate model accuracy

def calculate_acc(predictions, test_label):
    predictions_test_array = []
    test_label_array = []

    for p in predictions:
        if round(p[1]) == 1:
            predictions_test_array.append(1)
        else:
            predictions_test_array.append(0)

    for t in test_label:
        if int(t[1]) == 1:
            test_label_array.append(1)
        else:
            test_label_array.append(0)

    return accuracy_score(test_label_array, predictions_test_array)


predictions = model.predict(test_data, verbose=0)

acc = calculate_acc(predictions, test_label)

# write acc to file
metric_file_name = os.path.join(local_output_path, 'metric.json')
metric_file_content = '{"total_metric": {"total_metric_values": {"accuracy": %0.4f}}}' % acc
with open(metric_file_name, "w") as f:
    f.write(metric_file_content + '\n')

# convert Keras model to TF model
def save_model_to_serving(model, export_path):
    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'images': model.input}, outputs={'logits': model.output})
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'segmentation': signature,
        },
        legacy_init_op=legacy_init_op)
    builder.save()

save_model_to_serving(model, model_output_path)

# copy output
file.copy_parallel(local_output_path, train_url)

logging.info('training done!')


