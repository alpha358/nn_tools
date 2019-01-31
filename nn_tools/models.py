# Credits: Julius R.


# KERNEL_REG = 1.e-4
# DROP_RATE = 0.25

# def bn_relu(x):
#     y = BatchNormalization(axis=-1)(x)
#     y = Activation('relu')(y)
#     return y

# def bn_relu_conv(filters):
#     def block(x):
#         y = bn_relu(x)
#         y = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(KERNEL_REG))(y)
#         return y

#     return block

# def identity_block(filters):
#     def block(x):
#         y = bn_relu_conv(filters)(x)
# #         y = Dropout(DROP_RATE)(y)
#         y = bn_relu_conv(filters)(y)
#         y = Add()([y, x])
#         return y

#     return block


# def conv_block(filters, stage):
#     def block(x):
#         if stage == 0:
#             strides = (1, 1)
#             y = x
#         else:
#             strides = (2, 2)
#             y = bn_relu(x)

#         y = Conv2D(filters, (3, 3), strides=strides, padding='same', kernel_regularizer=l2(KERNEL_REG))(y)
# #         y = Dropout(DROP_RATE)(y)
#         y = bn_relu_conv(filters)(y)

#         shortcut = Conv2D(filters, (1, 1), strides=strides, padding='valid',
#                           kernel_regularizer=l2(KERNEL_REG))(x)
#         y = Add()([y, shortcut])
#         return y

#     return block

# def build_resnet(repetitions, input_shape, classes):
#     img_input = Input(shape=input_shape)

#     init_filters = 64

#     # resnet bottom
#     x = BatchNormalization(axis=-1)(img_input)
#     x = Conv2D(init_filters, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(KERNEL_REG))(x)
#     x = bn_relu(x)
#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

#     # resnet body
#     for stage, rep in enumerate(repetitions):
#         filters = init_filters * (2**stage)
#         for block in range(rep):
#             if block == 0:
#                 x = conv_block(filters, stage)(x)
#             else:
#                 x = identity_block(filters)(x)

#     x = bn_relu(x)

#     # resnet top
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(classes)(x)
#     x = Activation('sigmoid')(x)

#     model = Model(img_input, x)

#     return model

# def ResNet18(input_shape, classes):
#     model = build_resnet((2, 2, 2, 2), input_shape, classes)
#     return model

# K.clear_session()
# tf.reset_default_graph()

# NUM_CLASSES = 28
# model = ResNet18(RESIZED_SHAPE, NUM_CLASSES)


# ======================== UNET =================
# credits: https://github.com/zhixuhao/unet/blob/master/model.py
#          https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
#          https://github.com/zhixuhao/unet

import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
