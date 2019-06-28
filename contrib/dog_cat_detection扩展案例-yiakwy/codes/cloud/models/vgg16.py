# Author: Lei Wang(lwang11@mtu.edu)
# Date: 24 JUN 2019

import os
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator # used for data augumentation
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint, LearningRateScheduler
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

K.set_image_data_format('channels_last')

# adpot from mask rcnn : https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py
class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)

def Conv2DBlock(inp, filters, kernel_size=(3, 3), dropouts=None, conv_repeated=2, padding='same', stage=None):
    name = 'conv_%s' % stage
    out = inp
    for i in range(conv_repeated):
        conv = KL.Conv2D(filters, kernel_size, padding=padding, name="{}_{}".format(name, i)) (out)
        bn = BatchNorm(name="bn_{}_{}".format(name, i))(conv, training=True)
        act = KL.Activation('relu', name="act_{}_{}".format(name, i))(bn)
        dropout = KL.Dropout(dropouts[i], name="dp_{}_{}".format(name, i))(act)
        out = act
        
    pooled = KL.MaxPooling2D(pool_size=(2,2), name="max_pooling2d_{}".format(stage))(out)
    out = pooled
    return out

# Loss Record
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

def lr_schedule_callback(epoch):
    lr = 1e-2
    if epoch > 75:
        lr = 5*1e-4
    elif epoch > 100:
        lr = 3*1e-4
    return lr


class VGG16(object):
    
    def __init__(self, mode, config, model_dir):
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.model = self.get_model(mode=self.mode)
    
    def get_model(self, mode):
        inp_image = KL.Input(shape=[self.config.HEIGHT, self.config.WIDTH, self.config.CHANNEL], name="input_image")
        self.inputs = (inp_image,)
        if mode == "training":
            # Feature Extractor Layers
            conv1 = Conv2DBlock(inp_image, 64, (3, 3), dropouts=(0.3, 0), stage=1)
            conv2 = Conv2DBlock(conv1, 128, (3, 3), dropouts=(0.4, 0), stage=2)
            conv3 = Conv2DBlock(conv2, 256, (3, 3), dropouts=(0.4, 0.4, 0), conv_repeated=3, stage=3)
            conv4 = Conv2DBlock(conv3, 512, (3, 3), dropouts=(0.4, 0.4, 0), conv_repeated=3, stage=4)
            conv5 = Conv2DBlock(conv4, 512, (3, 3), dropouts=(0.4, 0.4, 0), conv_repeated=3, stage=5)
            
            # Two stacked FC layers with the Classifier for trainning
            dropout = KL.Dropout(0.5, name="dp_fc_0")(conv5)
            flattened = KL.Flatten(name="flatten_fc_0")(dropout)
            fc1 = KL.Dense(512, activation='relu', name="fc_1")(flattened)
            bn = BatchNorm(name="bn_fc1")(fc1)
            fc2 = KL.Dense(512, activation='relu', name="fc_2")(bn)
            
            classifier = KL.Dense(self.config.NUM_CLASSES, activation='softmax')(fc2)
            self.outputs = (classifier,)
            
            # Model
            model = KM.Model(self.inputs, self.outputs, name='vgg16')
            return model
        
    def fit(self, train_data, train_label, optimizer_type='rmsprop', data_augumented=True):
        model = self.model
        # Initate optimizer
        if optimizer_type is 'rmsprop':
            optimizer = RMSprop(lr=1e-2, decay=1e-6)
        else:
            optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=False)
        # compile the program
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # early stopping strategy
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto') 
        history = LossHistory()
        
        filepath="{}/weights.best.checkpoint.hdf5".format(self.model_dir)
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        
        learning_rate_schedule = LearningRateScheduler(lr_schedule_callback)
 
        # define tenor visualization board
        tensor_board = TensorBoard(log_dir=self.model_dir)

        # data augumentation
        # adapted from https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.25)
        
        # begin to train
        if not data_augumented:
            model.fit(
                train_data,
                train_label,
                batch_size=self.config.BATCH_SIZE,
                epochs=self.config.EPOCHS,
                validation_split=0.25, # split 25% of training data for validation
                verbose=2,
                shuffle=True,
                callbacks=[history, early_stopping, checkpoint, learning_rate_schedule, tensor_board])
        else:
            # data enhancement
            datagen.fit(train_data)
            # trainning
            train_reader = datagen.flow(train_data, train_label, batch_size=self.config.BATCH_SIZE, subset='training')
            val_reader = datagen.flow(train_data, train_label, batch_size=self.config.BATCH_SIZE, subset='validation')
            
            model.fit_generator(train_reader,
                                validation_data=val_reader,
                                # steps_per_epoch=len(train_data)/self.config.BATCH_SIZE, # https://github.com/keras-team/keras/issues/10855
                                epochs=self.config.EPOCHS,
                                workers=8,
                                verbose=2,
                                callbacks=[history, checkpoint, tensor_board])
            
        return history
        
    def load_weights(self, weights=None):
        if weights is None:
            filepath="{}/weights.best.checkpoint.hdf5".format(self.model_dir)
        else:
            filepath=weights
            
        if os.path.isfile(filepath):
            model = self.model
            model.load_weights(filepath)
        else:
            print("{} does not exit!".format(filepath))
    
    def infer(self, imgs, verbose=0):
        model = self.model
        results = model.predict(imgs, verbose=verbose)
        return results
    
    def summary(self):
        self.model.summary()
    
    # @todo : TODO
    def get_layers(self):
        pass


