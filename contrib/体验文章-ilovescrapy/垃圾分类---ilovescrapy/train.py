# -*- coding: utf-8 -*-
import os
import multiprocessing
from glob import glob

import numpy as np
from keras import backend
from keras.models import Model
from keras.optimizers import adam
from keras.layers import Flatten, Dense, Input
from keras.callbacks import TensorBoard, Callback
from keras import regularizers
from moxing.framework import file

from data_gen import data_flow
from models.resnet50 import ResNet50
from metrics import NormFace
from nets import SENET50
from keras.applications.inception_v3 import InceptionV3

backend.set_image_data_format('channels_last')


def model_fn(FLAGS, objective, optimizer, metrics):
    """
    pre-trained resnet50 model
    """
    base_model = ResNet50(weights="imagenet",
                          include_top=False,
                          pooling='avg',
                          input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                          classes=FLAGS.num_classes)
    # base_model = SENET50(weights=None,
    #                       include_top=False,
    #                       pooling='avg',
    #                       input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
    #                       classes=FLAGS.num_classes)
    for layer in base_model.layers:
        layer.trainable = True
    x = base_model.output
    predictions = Dense(FLAGS.num_classes, activation='softmax')(x)
    #predictions=NormFace(FLAGS.num_classes)(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model


class LossHistory(Callback):
    def __init__(self, FLAGS):
        super(LossHistory, self).__init__()
        self.FLAGS = FLAGS

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        save_path = os.path.join(self.FLAGS.train_local, 'weights_%03d_%.4f.h5' % (epoch, logs.get('val_acc',0)))
        self.model.save_weights(save_path)
        if self.FLAGS.train_url.startswith('s3://'):
            save_url = os.path.join(self.FLAGS.train_url, 'weights_%03d_%.4f.h5' % (epoch, logs.get('val_acc',0)))
            file.copy(save_path, save_url)
        print('save weights file', save_path)

        if self.FLAGS.keep_weights_file_num > -1:
            weights_files = glob(os.path.join(self.FLAGS.train_local, '*.h5'))
            if len(weights_files) >= self.FLAGS.keep_weights_file_num:
                weights_files.sort(key=lambda file_name: os.stat(file_name).st_ctime, reverse=True)
                for file_path in weights_files[self.FLAGS.keep_weights_file_num:]:
                    os.remove(file_path)  # only remove weights files on local path


def train_model(FLAGS):
    # data flow generator
    train_sequence, validation_sequence = data_flow(FLAGS.data_local, FLAGS.batch_size,
                                                    FLAGS.num_classes, FLAGS.input_size)

    optimizer = adam(lr=FLAGS.learning_rate, decay=0.0005)
    objective = 'categorical_crossentropy'
    metrics = ['accuracy']
    model = model_fn(FLAGS, objective, optimizer, metrics)
    if FLAGS.restore_model_path != '' and file.exists(FLAGS.restore_model_path):
        if FLAGS.restore_model_path.startswith('s3://'):
            restore_model_name = FLAGS.restore_model_path.rsplit('/', 1)[1]
            file.copy(FLAGS.restore_model_path, '/cache/tmp/' + restore_model_name)
            model.load_weights('/cache/tmp/' + restore_model_name)
            os.remove('/cache/tmp/' + restore_model_name)
        else:
            model.load_weights(FLAGS.restore_model_path)
    if not os.path.exists(FLAGS.train_local):
        os.makedirs(FLAGS.train_local)
    tensorBoard = TensorBoard(log_dir=FLAGS.train_local)
    history = LossHistory(FLAGS)
    model.fit_generator(
        train_sequence,
        steps_per_epoch=len(train_sequence),
        epochs=FLAGS.max_epochs,
        verbose=1,
        callbacks=[history, tensorBoard],
        validation_data=validation_sequence,
        max_queue_size=10,
        workers=int(multiprocessing.cpu_count() * 0.7),
        use_multiprocessing=True,
        shuffle=True
    )

    print('training done!')

    if FLAGS.deploy_script_path != '':
        from save_model import save_pb_model
        save_pb_model(FLAGS, model)

    if FLAGS.test_data_url != '':
        print('test dataset predicting...')
        from eval import load_test_data
        img_names, test_data, test_labels = load_test_data(FLAGS)
        predictions = model.predict(test_data, verbose=0)

        right_count = 0
        for index, pred in enumerate(predictions):
            predict_label = np.argmax(pred, axis=0)
            test_label = test_labels[index]
            if predict_label == test_label:
                right_count += 1
        accuracy = right_count / len(img_names)
        print('accuracy: %0.4f' % accuracy)
        metric_file_name = os.path.join(FLAGS.train_local, 'metric.json')
        metric_file_content = '{"total_metric": {"total_metric_values": {"accuracy": %0.4f}}}' % accuracy
        with open(metric_file_name, "w") as f:
            f.write(metric_file_content + '\n')
    print('end')
