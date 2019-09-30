"""Xception V1 model for Keras.
On ImageNet, this model gets to a top-1 validation accuracy of 0.790
and a top-5 validation accuracy of 0.945.
Do note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function
is also different (same as Inception V3).
# Reference
- [Xception: Deep Learning with Depthwise Separable Convolutions](
    https://arxiv.org/abs/1610.02357)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, add, Dense, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D, TimeDistributed

from keras import backend as K

from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization


def get_weight_path():
    if K.image_dim_ordering() == 'th':
        return 'xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
    else:
        return 'xception_weights_tf_dim_ordering_tf_kernels.h5'


def get_img_output_length(width, height):
    def get_output_length(input_length):
        filter_sizes = [3, 3, 1, 1, 1, 1]
        strides = [2, 1, 2, 2, 2, 2]

        assert len(filter_sizes) == len(strides)

        for i in range(len(filter_sizes)):
            input_length = (input_length - filter_sizes[i]) // strides[i] + 1

        return input_length

    return get_output_length(width), get_output_length(height)


def nn_base(input_tensor=None, trainable=False):

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(32, (3, 3),
               strides=(2, 2),
               use_bias=False,
               name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1),
                      strides=(2, 2),
                      padding='same',
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3),
                     strides=(2, 2),
                     padding='same',
                     name='block2_pool')(x)
    x = add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2),
                     padding='same',
                     name='block3_pool')(x)
    x = add([x, residual])

    residual = Conv2D(728, (1, 1),
                      strides=(2, 2),
                      padding='same',
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2),
                     padding='same',
                     name='block4_pool')(x)
    x = add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3),
                            padding='same',
                            use_bias=False,
                            name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3),
                            padding='same',
                            use_bias=False,
                            name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3),
                            padding='same',
                            use_bias=False,
                            name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3),
                        padding='same',
                        use_bias=False,
                        name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3),
                     strides=(2, 2),
                     padding='same',
                     name='block13_pool')(x)
    x = add([x, residual])

    return x


def classifier_layers(x, input_shape, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
    # (hence a smaller stride in the region that follows the ROI pool)
    x = TimeDistributed(SeparableConv2D(1536, (3, 3),
                                        padding='same',
                                        use_bias=False),
                        name='block14_sepconv1')(x)
    x = TimeDistributed(BatchNormalization(), name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = TimeDistributed(SeparableConv2D(2048, (3, 3),
                                        padding='same',
                                        use_bias=False),
                        name='block14_sepconv2')(x)
    x = TimeDistributed(BatchNormalization(), name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    TimeDistributed(GlobalAveragePooling2D(), name='avg_pool')(x)

    return x


def rpn(base_layers, num_anchors):

    x = Conv2D(1024, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois, 14, 14, 1024)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois, 1024, 7, 7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)

    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]

