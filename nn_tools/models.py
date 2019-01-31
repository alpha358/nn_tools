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
