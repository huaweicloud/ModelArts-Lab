import os, cv2, random
import numpy as np

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, Callback, EarlyStopping
import tensorflow as tf
from moxing.framework import file
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import backend as K

logging.basicConfig(level=logging.INFO)
K.set_image_data_format('channels_last')

tf.flags.DEFINE_integer('max_epochs', 20, 'number of training iterations.')
tf.flags.DEFINE_string('data_url', '/cache/data', 'dataset directory.')
tf.flags.DEFINE_string('train_url', '/cache/output', 'saved model directory.')
tf.flags.DEFINE_integer('batch_size', 16, 'number of training iterations.')
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

ROWS = 64
COLS = 64
CHANNELS = 3

images = [os.path.join(dogcat_data_path, i) for i in os.listdir(dogcat_data_path)]
random.shuffle(images)

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
images_data = prep_data(images)

# get labels
labels = []
for i in images:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)

train_data, test_data, train_label, test_label = train_test_split(images_data, labels, test_size=0.25, random_state=10)

optimizer = RMSprop(lr=learning_rate)
objective = 'binary_crossentropy'

# define model architecture
def model_fn():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(ROWS, COLS, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_last"))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model


model = model_fn()


# Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
tensorBoard = TensorBoard(log_dir=local_output_path)

def run_catdog():
    history = LossHistory()
    model.fit(
        train_data,
        train_label,
        batch_size=batch_size,
        epochs=max_epochs,
        validation_split=0.25,
        verbose=2,
        shuffle=True,
        callbacks=[history, early_stopping, tensorBoard])
    return history


# run training
history = run_catdog()

# calculate model accuracy
predictions = model.predict(test_data, verbose=0)
predictions_test = []
for p in predictions:
    predictions_test.append(int(round(p[0])))

acc = accuracy_score(test_label, predictions_test)

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



