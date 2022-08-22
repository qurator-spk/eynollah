import os
import sys
import tensorflow as tf
import keras , warnings
from keras.optimizers import *
from sacred import Experiment
from models import *
from utils import *
from metrics import *


    
    
def configuration():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


if __name__=='__main__':
    n_classes = 2
    input_height = 224
    input_width = 448
    weight_decay = 1e-6
    pretraining = False
    dir_of_weights = 'model_bin_sbb_ens.h5'
    
    #configuration()
    
    model = resnet50_unet(n_classes,  input_height, input_width,weight_decay,pretraining)
    model.load_weights(dir_of_weights)
    model.save('./name_in_another_python_version.h5')
    

