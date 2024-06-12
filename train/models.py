import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

mlp_head_units = [2048, 1024]
#projection_dim = 64
transformer_layers = 8
num_heads = 4
resnet50_Weights_path = './pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size_x, patch_size_y):#__init__(self, **kwargs):#:__init__(self, patch_size):#__init__(self, **kwargs):
        super(Patches, self).__init__()
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y

    def call(self, images):
        #print(tf.shape(images)[1],'images')
        #print(self.patch_size,'self.patch_size')
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size_y, self.patch_size_x, 1],
            strides=[1, self.patch_size_y, self.patch_size_x, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'patch_size_x': self.patch_size_x,
            'patch_size_y': self.patch_size_y,
        })
        return config

class Patches_old(layers.Layer):
    def __init__(self, patch_size):#__init__(self, **kwargs):#:__init__(self, patch_size):#__init__(self, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        #print(tf.shape(images)[1],'images')
        #print(self.patch_size,'self.patch_size')
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        #print(patches.shape,patch_dims,'patch_dims')
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config

    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config
    
    
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

    x = layers.add([x, input_tensor])
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

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50_unet_light(n_classes, input_height=224, input_width=224, taks="segmentation", weight_decay=1e-6, pretraining=False):
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


def vit_resnet50_unet(n_classes, patch_size_x, patch_size_y, num_patches, projection_dim = 64, input_height=224, input_width=224, task="segmentation", weight_decay=1e-6, pretraining=False):
    inputs = layers.Input(shape=(input_height, input_width, 3))
    
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
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
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
    
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
