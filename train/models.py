from keras.models import *
from keras.layers import *
from keras import layers
from keras.regularizers import l2

resnet50_Weights_path='./pretrained_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
IMAGE_ORDERING ='channels_last'
MERGE_AXIS=-1


def one_side_pad( x ):
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    if IMAGE_ORDERING == 'channels_first':
        x = Lambda(lambda x : x[: , : , :-1 , :-1 ] )(x)
    elif IMAGE_ORDERING == 'channels_last':
        x = Lambda(lambda x : x[: , :-1 , :-1 , :  ] )(x)
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

    x = Conv2D(filters1, (1, 1) , data_format=IMAGE_ORDERING , name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size , data_format=IMAGE_ORDERING ,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3 , (1, 1), data_format=IMAGE_ORDERING , name=conv_name_base + '2c')(x)
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

    x = Conv2D(filters1, (1, 1) , data_format=IMAGE_ORDERING  , strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size , data_format=IMAGE_ORDERING  , padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1) , data_format=IMAGE_ORDERING  , name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1) , data_format=IMAGE_ORDERING  , strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50_unet_light(n_classes,input_height=224,input_width=224,weight_decay=1e-6,pretraining=False):
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
        model=Model( img_input , x ).load_weights(resnet50_Weights_path)


    v512_2048 =  Conv2D( 512 , (1, 1) , padding='same', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) )( f5 )
    v512_2048 = ( BatchNormalization(axis=bn_axis))(v512_2048)
    v512_2048 = Activation('relu')(v512_2048)
    
    

    v512_1024=Conv2D( 512 , (1, 1) , padding='same', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) )( f4 )
    v512_1024 = ( BatchNormalization(axis=bn_axis))(v512_1024)
    v512_1024 = Activation('relu')(v512_1024)
    
    
    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(v512_2048)
    o = ( concatenate([ o ,v512_1024],axis=MERGE_AXIS )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)


    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([ o ,f3],axis=MERGE_AXIS )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)


    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([o,f2],axis=MERGE_AXIS ) )
    o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay) ) )(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)



    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([o,f1],axis=MERGE_AXIS ) )
    o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) ))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)


    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([o,img_input],axis=MERGE_AXIS ) )
    o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 32 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) ))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)

    
    
    o =  Conv2D( n_classes , (1, 1) , padding='same', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) )( o )
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = (Activation('softmax'))(o)

    
    model = Model( img_input , o )
    return model

def resnet50_unet(n_classes,input_height=224,input_width=224,weight_decay=1e-6,pretraining=False):
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
        Model( img_input , x ).load_weights(resnet50_Weights_path)

    v1024_2048 =  Conv2D( 1024 , (1, 1) , padding='same', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) )( f5 )
    v1024_2048 = ( BatchNormalization(axis=bn_axis))(v1024_2048)
    v1024_2048 = Activation('relu')(v1024_2048)
    
    
    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(v1024_2048)
    o = ( concatenate([ o ,f4],axis=MERGE_AXIS )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)


    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([ o ,f3],axis=MERGE_AXIS )  )
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay)))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)


    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([o,f2],axis=MERGE_AXIS ) )
    o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING, kernel_regularizer=l2(weight_decay) ) )(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)


    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([o,f1],axis=MERGE_AXIS ) )
    o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) ))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)


    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( concatenate([o,img_input],axis=MERGE_AXIS ) )
    o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 32 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) ))(o)
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = Activation('relu')(o)
    
    
    o =  Conv2D( n_classes , (1, 1) , padding='same', data_format=IMAGE_ORDERING,kernel_regularizer=l2(weight_decay) )( o )
    o = ( BatchNormalization(axis=bn_axis))(o)
    o = (Activation('softmax'))(o)
    
    model = Model( img_input , o )
    

 

    return model
