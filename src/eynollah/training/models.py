from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Concatenate, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Input, Layer
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import tensorflow as tf

resnet50_Weights_path='./pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def identity_block(input_tensor, kernel_size, filters, stage, block, data_format='channels_last'):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    
    bn_axis = 3 if data_format == 'channels_last' else 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1) , data_format=data_format , name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size , data_format=data_format ,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3 , (1, 1), data_format=data_format , name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), data_format='channels_last'):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    
    bn_axis = 3 if data_format == 'channels_last' else 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1) , data_format=data_format, strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size , data_format=data_format, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), data_format=data_format, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), data_format=data_format, strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


class PadMultiple(Layer):
    def __init__(self, mods, data_format='channels_last'):
        super().__init__()
        self.mods = mods
        self.data_format = data_format
   
    def call(self, x):
        h, w = self.mods
        padding = (
            [(0,0), (0, -tf.shape(x)[1] % h), (0, -tf.shape(x)[2] % w), (0,0)] if self.data_format == 'channels_last'
            else [(0,0), (0,0), (0, -tf.shape(x)[1] % h), (0, -tf.shape(x)[2] % w)])
        return tf.pad(x, padding)


class CutTo(Layer):
    def __init__(self, data_format='channels_last'):
        super().__init__()
        self.data_format = data_format
   
    def call(self, inputs):
        h, w = (1, 2) if self.data_format == 'channels_last' else (2,4)
        h, w = tf.shape(inputs[1])[h], tf.shape(inputs[1])[w]
        return inputs[0][:, :h, :w] if self.data_format == 'channels_last' else inputs[0][:, :, :h, :w]


def resnet50_unet(n_classes, input_height=None, input_width=None, weight_decay=1e-6, pretraining=False, last_activation='softmax', skip_last_batchnorm=False, light_version=False, data_format='channels_last'):
    """ Returns a U-NET model using the keras functional API. """
    img_input = Input(shape=(input_height, input_width, 3 ))
    padded_to_multiple = PadMultiple((32,32))(img_input)

    bn_axis = 3 if data_format == 'channels_last' else 1
    merge_axis = 3 if data_format == 'channels_last' else 1

    x = ZeroPadding2D((3, 3), data_format=data_format)(padded_to_multiple)
    x = Conv2D(64, (7, 7), data_format=data_format, strides=(2, 2), kernel_regularizer=l2(weight_decay), name='conv1')(x)
    f1 = x

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3) , data_format=data_format , strides=(2, 2))(x)
    

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), data_format=data_format)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', data_format=data_format)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', data_format=data_format)
    f2 = ZeroPadding2D(((1,0), (1,0)), data_format=data_format)(x)


    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', data_format=data_format)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', data_format=data_format)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', data_format=data_format)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', data_format=data_format)
    f3 = x 

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', data_format=data_format)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', data_format=data_format)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', data_format=data_format)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', data_format=data_format)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', data_format=data_format)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', data_format=data_format)
    f4 = x 

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', data_format=data_format)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', data_format=data_format)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', data_format=data_format)
    f5 = x 

    if pretraining:
        Model(img_input, x).load_weights(resnet50_Weights_path)

    if light_version:
        v512_2048 = Conv2D(512, (1, 1), padding='same', data_format=data_format, kernel_regularizer=l2(weight_decay))(f5)
        v512_2048 = BatchNormalization(axis=bn_axis)(v512_2048)
        v512_2048 = Activation('relu')(v512_2048)

        v512_1024 = Conv2D(512, (1, 1), padding='same', data_format=data_format, kernel_regularizer=l2(weight_decay))(f4)
        v512_1024 = BatchNormalization(axis=bn_axis)(v512_1024)
        v512_1024 = Activation('relu')(v512_1024)
        x, c = v512_2048, v512_1024 # continuation and concatenation layers
    else:
        v1024_2048 = Conv2D(1024, (1, 1), padding='same', data_format=data_format, kernel_regularizer=l2(weight_decay))(f5)
        v1024_2048 = BatchNormalization(axis=bn_axis)(v1024_2048)
        v1024_2048 = Activation('relu')(v1024_2048)
        x, c = v1024_2048, f4 # continuation and concatenation layers
    
    o = UpSampling2D((2,2), data_format=data_format)(x)
    o = Concatenate(axis=merge_axis)([o ,c])
    o = ZeroPadding2D( (1,1), data_format=data_format)(o)
    o = Conv2D(512, (3, 3), padding='valid', data_format=data_format, kernel_regularizer=l2(weight_decay))(o)
    o = BatchNormalization(axis=bn_axis)(o)
    o = Activation('relu')(o)

    o = UpSampling2D( (2,2), data_format=data_format)(o)
    o = Concatenate(axis=merge_axis)([ o ,f3])
    o = ZeroPadding2D( (1,1), data_format=data_format)(o)
    o = Conv2D( 256, (3, 3), padding='valid', data_format=data_format, kernel_regularizer=l2(weight_decay))(o)
    o = BatchNormalization(axis=bn_axis)(o)
    o = Activation('relu')(o)

    o = UpSampling2D( (2,2), data_format=data_format)(o)
    o = Concatenate(axis=merge_axis)([o,f2])
    o = ZeroPadding2D((1,1) , data_format=data_format)(o)
    o = Conv2D( 128 , (3, 3), padding='valid', data_format=data_format, kernel_regularizer=l2(weight_decay))(o)
    o = BatchNormalization(axis=bn_axis)(o)
    o = Activation('relu')(o)

    o = UpSampling2D( (2,2), data_format=data_format)(o)
    o = Concatenate(axis=merge_axis)([o,f1])
    o = ZeroPadding2D((1,1)  , data_format=data_format)(o)
    o = Conv2D( 64 , (3, 3), padding='valid', data_format=data_format, kernel_regularizer=l2(weight_decay))(o)
    o = BatchNormalization(axis=bn_axis)(o)
    o = Activation('relu')(o)

    o = UpSampling2D( (2,2), data_format=data_format)(o)
    o = Concatenate(axis=merge_axis)([o, padded_to_multiple])
    o = ZeroPadding2D((1,1)  , data_format=data_format)(o)
    o = Conv2D(32, (3, 3), padding='valid', data_format=data_format, kernel_regularizer=l2(weight_decay))(o)
    o = BatchNormalization(axis=bn_axis)(o)
    o = Activation('relu')(o)
    
    o =  Conv2D(n_classes, (1, 1), padding='same', data_format=data_format, kernel_regularizer=l2(weight_decay))(o)
    if not skip_last_batchnorm:
        o = BatchNormalization(axis=bn_axis)(o)
    
    o = Activation(last_activation)(o)
    o = CutTo()([o, img_input])
    
    return Model(img_input , o)
