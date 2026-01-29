from tensorflow import keras
from keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    Lambda,
    Layer,
    LayerNormalization,
    MaxPooling2D,
    MultiHeadAttention,
    UpSampling2D,
    ZeroPadding2D,
    add,
    concatenate
)
from keras.models import Model
import tensorflow as tf
# from keras import layers, models
from keras.regularizers import l2

from eynollah.patch_encoder import Patches, PatchEncoder

##mlp_head_units = [512, 256]#[2048, 1024]
###projection_dim = 64
##transformer_layers = 2#8
##num_heads = 1#4
resnet50_Weights_path = './pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1


class CTCLayer(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred
    
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def one_side_pad(x):
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    if IMAGE_ORDERING == 'channels_first':
        x = Lambda(lambda x: x[:, :, :-1, :-1])(x)
    elif IMAGE_ORDERING == 'channels_last':
        x = Lambda(lambda x: x[:, :-1, :-1, :])(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
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

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
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

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), data_format=IMAGE_ORDERING, strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, data_format=IMAGE_ORDERING, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), data_format=IMAGE_ORDERING, strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50_unet_light(n_classes, input_height=224, input_width=224, task="segmentation", weight_decay=1e-6, pretraining=False):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING, strides=(2, 2), kernel_regularizer=l2(weight_decay),
               name='conv1')(x)
    f1 = x

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = one_side_pad(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    f4 = x

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x

    if pretraining:
        model = Model(img_input, x).load_weights(resnet50_Weights_path)

    v512_2048 = Conv2D(512, (1, 1), padding='same', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay))(f5)
    v512_2048 = (BatchNormalization(axis=bn_axis))(v512_2048)
    v512_2048 = Activation('relu')(v512_2048)

    v512_1024 = Conv2D(512, (1, 1), padding='same', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay))(f4)
    v512_1024 = (BatchNormalization(axis=bn_axis))(v512_1024)
    v512_1024 = Activation('relu')(v512_1024)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(v512_2048)
    o = (concatenate([o, v512_1024], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f1], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, img_input], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(32, (3, 3), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)

    o = Conv2D(n_classes, (1, 1), padding='same', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay))(o)
    if task == "segmentation":
        o = (BatchNormalization(axis=bn_axis))(o)
        o = (Activation('softmax'))(o)
    else:
        o = (Activation('sigmoid'))(o)

    model = Model(img_input, o)
    return model


def resnet50_unet(n_classes, input_height=224, input_width=224, task="segmentation", weight_decay=1e-6, pretraining=False):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING, strides=(2, 2), kernel_regularizer=l2(weight_decay),
               name='conv1')(x)
    f1 = x

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = one_side_pad(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    f4 = x

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x

    if pretraining:
        Model(img_input, x).load_weights(resnet50_Weights_path)

    v1024_2048 = Conv2D(1024, (1, 1), padding='same', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay))(
        f5)
    v1024_2048 = (BatchNormalization(axis=bn_axis))(v1024_2048)
    v1024_2048 = Activation('relu')(v1024_2048)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(v1024_2048)
    o = (concatenate([o, f4], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f1], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, img_input], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(32, (3, 3), padding='valid', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)

    o = Conv2D(n_classes, (1, 1), padding='same', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay))(o)
    if task == "segmentation":
        o = (BatchNormalization(axis=bn_axis))(o)
        o = (Activation('softmax'))(o)
    else:
        o = (Activation('sigmoid'))(o)

    model = Model(img_input, o)

    return model


def vit_resnet50_unet(n_classes, patch_size_x, patch_size_y, num_patches, mlp_head_units=None, transformer_layers=8, num_heads =4, projection_dim = 64, input_height=224, input_width=224, task="segmentation", weight_decay=1e-6, pretraining=False):
    if mlp_head_units is None:
        mlp_head_units = [128, 64]
    inputs = Input(shape=(input_height, input_width, 3))
    
    #transformer_units = [
        #projection_dim * 2,
        #projection_dim,
    #]  # Size of the transformer layers
    IMAGE_ORDERING = 'channels_last'
    bn_axis=3

    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(inputs)
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING, strides=(2, 2),kernel_regularizer=l2(weight_decay), name='conv1')(x)
    f1 = x

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2))(x)
    
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = one_side_pad(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x 

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    f4 = x 

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x 
    
    if pretraining:
        model = Model(inputs, x).load_weights(resnet50_Weights_path)

    #num_patches = x.shape[1]*x.shape[2]
    
    #patch_size_y = input_height / x.shape[1]
    #patch_size_x = input_width / x.shape[2]
    #patch_size = patch_size_x * patch_size_y
    patches = Patches(patch_size_x, patch_size_y)(x)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=mlp_head_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])
    
    assert isinstance(x, Layer)
    encoded_patches = tf.reshape(encoded_patches, [-1, x.shape[1], x.shape[2] , int( projection_dim / (patch_size_x * patch_size_y) )])

    v1024_2048 = Conv2D( 1024 , (1, 1), padding='same', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay))(encoded_patches)
    v1024_2048 = (BatchNormalization(axis=bn_axis))(v1024_2048)
    v1024_2048 = Activation('relu')(v1024_2048)
    
    o = (UpSampling2D( (2, 2), data_format=IMAGE_ORDERING))(v1024_2048)
    o = (concatenate([o, f4],axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o ,f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f1], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, inputs],axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(32, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o = Conv2D(n_classes, (1, 1), padding='same', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay))(o)
    if task == "segmentation":
        o = (BatchNormalization(axis=bn_axis))(o)
        o = (Activation('softmax'))(o)
    else:
        o = (Activation('sigmoid'))(o)

    model = Model(inputs=inputs, outputs=o)
    
    return model

def vit_resnet50_unet_transformer_before_cnn(n_classes, patch_size_x, patch_size_y, num_patches, mlp_head_units=None, transformer_layers=8, num_heads =4, projection_dim = 64, input_height=224, input_width=224, task="segmentation", weight_decay=1e-6, pretraining=False):
    if mlp_head_units is None:
        mlp_head_units = [128, 64]
    inputs = Input(shape=(input_height, input_width, 3))
    
    ##transformer_units = [
        ##projection_dim * 2,
        ##projection_dim,
    ##]  # Size of the transformer layers
    IMAGE_ORDERING = 'channels_last'
    bn_axis=3
    
    patches = Patches(patch_size_x, patch_size_y)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=mlp_head_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])
    
    encoded_patches = tf.reshape(encoded_patches, [-1, input_height, input_width , int( projection_dim / (patch_size_x * patch_size_y) )])
    
    encoded_patches = Conv2D(3, (1, 1), padding='same', data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay), name='convinput')(encoded_patches)
    
    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(encoded_patches)
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING, strides=(2, 2),kernel_regularizer=l2(weight_decay), name='conv1')(x)
    f1 = x

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), data_format=IMAGE_ORDERING, strides=(2, 2))(x)
    
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = one_side_pad(x)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x 

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    f4 = x 

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x 
    
    if pretraining:
        model = Model(encoded_patches, x).load_weights(resnet50_Weights_path)

    v1024_2048 = Conv2D( 1024 , (1, 1), padding='same', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay))(x)
    v1024_2048 = (BatchNormalization(axis=bn_axis))(v1024_2048)
    v1024_2048 = Activation('relu')(v1024_2048)
    
    o = (UpSampling2D( (2, 2), data_format=IMAGE_ORDERING))(v1024_2048)
    o = (concatenate([o, f4],axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o ,f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f1], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, inputs],axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(32, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = (BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    o = Conv2D(n_classes, (1, 1), padding='same', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay))(o)
    if task == "segmentation":
        o = (BatchNormalization(axis=bn_axis))(o)
        o = (Activation('softmax'))(o)
    else:
        o = (Activation('sigmoid'))(o)

    model = Model(inputs=inputs, outputs=o)
    
    return model

def resnet50_classifier(n_classes,input_height=224,input_width=224,weight_decay=1e-6,pretraining=False):
    include_top=True
    assert input_height%32 == 0
    assert input_width%32 == 0

    
    img_input = Input(shape=(input_height,input_width , 3 ))

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING, strides=(2, 2),kernel_regularizer=l2(weight_decay), name='conv1')(x)
    f1 = x

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3) , data_format=IMAGE_ORDERING , strides=(2, 2))(x)
    

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = one_side_pad(x )


    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x 

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    f4 = x 

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    f5 = x 

    if pretraining:
        Model(img_input, x).load_weights(resnet50_Weights_path)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    
    ##
    x = Dense(256, activation='relu', name='fc512')(x)
    x=Dropout(0.2)(x)
    ##
    x = Dense(n_classes, activation='softmax', name='fc1000')(x)
    model = Model(img_input, x)
    
    


    return model

def machine_based_reading_order_model(n_classes,input_height=224,input_width=224,weight_decay=1e-6,pretraining=False):
    assert input_height%32 == 0
    assert input_width%32 == 0

    img_input = Input(shape=(input_height,input_width , 3 ))

    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x1 = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    x1 = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING, strides=(2, 2),kernel_regularizer=l2(weight_decay), name='conv1')(x1)

    x1 = BatchNormalization(axis=bn_axis, name='bn_conv1')(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((3, 3) , data_format=IMAGE_ORDERING , strides=(2, 2))(x1)
    
    x1 = conv_block(x1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x1 = identity_block(x1, 3, [64, 64, 256], stage=2, block='b')
    x1 = identity_block(x1, 3, [64, 64, 256], stage=2, block='c')

    x1 = conv_block(x1, 3, [128, 128, 512], stage=3, block='a')
    x1 = identity_block(x1, 3, [128, 128, 512], stage=3, block='b')
    x1 = identity_block(x1, 3, [128, 128, 512], stage=3, block='c')
    x1 = identity_block(x1, 3, [128, 128, 512], stage=3, block='d')

    x1 = conv_block(x1, 3, [256, 256, 1024], stage=4, block='a')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='b')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='c')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='d')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='e')
    x1 = identity_block(x1, 3, [256, 256, 1024], stage=4, block='f')

    x1 = conv_block(x1, 3, [512, 512, 2048], stage=5, block='a')
    x1 = identity_block(x1, 3, [512, 512, 2048], stage=5, block='b')
    x1 = identity_block(x1, 3, [512, 512, 2048], stage=5, block='c')
    
    if pretraining:
        Model(img_input , x1).load_weights(resnet50_Weights_path)
    
    x1 = AveragePooling2D((7, 7), name='avg_pool1')(x1)
    flattened = Flatten()(x1)
    
    o = Dense(256, activation='relu', name='fc512')(flattened)
    o=Dropout(0.2)(o)
    
    o = Dense(256, activation='relu', name='fc512a')(o)
    o=Dropout(0.2)(o)

    o = Dense(n_classes, activation='sigmoid', name='fc1000')(o)
    model = Model(img_input , o)

    return model

def cnn_rnn_ocr_model(image_height=None, image_width=None, n_classes=None, max_seq=None):
    input_img = tf.keras.Input(shape=(image_height, image_width, 3), name="image")
    labels = tf.keras.layers.Input(name="label", shape=(None,))

    x = tf.keras.layers.Conv2D(64,kernel_size=(3,3),padding="same")(input_img)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Activation("relu", name="relu1")(x)
    x = tf.keras.layers.Conv2D(64,kernel_size=(3,3),padding="same")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.Activation("relu", name="relu2")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1,2),strides=(1,2))(x)

    x = tf.keras.layers.Conv2D(128,kernel_size=(3,3),padding="same")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.Activation("relu", name="relu3")(x)
    x = tf.keras.layers.Conv2D(128,kernel_size=(3,3),padding="same")(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.Activation("relu", name="relu4")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1,2),strides=(1,2))(x)

    x = tf.keras.layers.Conv2D(256,kernel_size=(3,3),padding="same")(x)
    x = tf.keras.layers.BatchNormalization(name="bn5")(x)
    x = tf.keras.layers.Activation("relu", name="relu5")(x)
    x = tf.keras.layers.Conv2D(256,kernel_size=(3,3),padding="same")(x)
    x = tf.keras.layers.BatchNormalization(name="bn6")(x)
    x = tf.keras.layers.Activation("relu", name="relu6")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)

    x = tf.keras.layers.Conv2D(image_width,kernel_size=(3,3),padding="same")(x)
    x = tf.keras.layers.BatchNormalization(name="bn7")(x)
    x = tf.keras.layers.Activation("relu", name="relu7")(x)
    x = tf.keras.layers.Conv2D(image_width,kernel_size=(16,1))(x)
    x = tf.keras.layers.BatchNormalization(name="bn8")(x)
    x = tf.keras.layers.Activation("relu", name="relu8")(x)
    x2d = tf.keras.layers.MaxPool2D(pool_size=(1,2),strides=(1,2))(x)
    x4d = tf.keras.layers.MaxPool2D(pool_size=(1,2),strides=(1,2))(x2d)
    

    new_shape = (x.shape[1]*x.shape[2], x.shape[3])
    new_shape2 = (x2d.shape[1]*x2d.shape[2], x2d.shape[3])
    new_shape4 = (x4d.shape[1]*x4d.shape[2], x4d.shape[3])
    
    x = tf.keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x2d = tf.keras.layers.Reshape(target_shape=new_shape2, name="reshape2")(x2d)
    x4d = tf.keras.layers.Reshape(target_shape=new_shape4, name="reshape4")(x4d)
    

    xrnnorg = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(image_width, return_sequences=True, dropout=0.25))(x)
    xrnn2d = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(image_width, return_sequences=True, dropout=0.25))(x2d)
    xrnn4d = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(image_width, return_sequences=True, dropout=0.25))(x4d)
    
    xrnn2d = tf.keras.layers.Reshape(target_shape=(1, xrnn2d.shape[1], xrnn2d.shape[2]), name="reshape6")(xrnn2d)
    xrnn4d = tf.keras.layers.Reshape(target_shape=(1, xrnn4d.shape[1], xrnn4d.shape[2]), name="reshape8")(xrnn4d)
    

    xrnn2dup = tf.keras.layers.UpSampling2D(size=(1, 2), interpolation="nearest")(xrnn2d)
    xrnn4dup = tf.keras.layers.UpSampling2D(size=(1, 4), interpolation="nearest")(xrnn4d)
    
    xrnn2dup = tf.keras.layers.Reshape(target_shape=(xrnn2dup.shape[2], xrnn2dup.shape[3]), name="reshape10")(xrnn2dup)
    xrnn4dup = tf.keras.layers.Reshape(target_shape=(xrnn4dup.shape[2], xrnn4dup.shape[3]), name="reshape12")(xrnn4dup)

    addition = tf.keras.layers.Add()([xrnnorg, xrnn2dup, xrnn4dup])
    
    addition_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(image_width, return_sequences=True, dropout=0.25))(addition)
    
    out = tf.keras.layers.Conv1D(max_seq, 1, data_format="channels_first")(addition_rnn)
    out = tf.keras.layers.BatchNormalization(name="bn9")(out)
    out = tf.keras.layers.Activation("relu", name="relu9")(out)
    #out = tf.keras.layers.Conv1D(n_classes, 1, activation='relu', data_format="channels_last")(out)

    out = tf.keras.layers.Dense(
        n_classes, activation="softmax", name="dense2"
    )(out)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLayer(name="ctc_loss")(labels, out)
    
    model = tf.keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")

    return model
    
