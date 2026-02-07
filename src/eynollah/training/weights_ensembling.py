import sys
from glob import glob
from os import environ, devnull
from os.path import join
from warnings import catch_warnings, simplefilter
import os

import numpy as np
from PIL import Image
import cv2
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(devnull, 'w')
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras import backend as tensorflow_backend
sys.stderr = stderr
from tensorflow.keras import layers
import tensorflow.keras.losses
from tensorflow.keras.layers import *
import click
import logging


class Patches(layers.Layer):
    def __init__(self, patch_size_x, patch_size_y):
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
        #patch_dims = patches.shape[-1]
        patch_dims = tf.shape(patches)[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'patch_size_x': self.patch_size_x,
            'patch_size_y': self.patch_size_y,
        })
        return config



class PatchEncoder(layers.Layer):
    def __init__(self, **kwargs):
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
    
    
def start_new_session():
    ###config = tf.compat.v1.ConfigProto()
    ###config.gpu_options.allow_growth = True

    ###self.session = tf.compat.v1.Session(config=config)  # tf.InteractiveSession()
    ###tensorflow_backend.set_session(self.session)
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    session = tf.compat.v1.Session(config=config)  # tf.InteractiveSession()
    tensorflow_backend.set_session(session)
    return session

def run_ensembling(dir_models, out):
    ls_models = os.listdir(dir_models)


    weights=[]

    for model_name in ls_models:
        model =  load_model(os.path.join(dir_models,model_name) , compile=False, custom_objects={'PatchEncoder':PatchEncoder, 'Patches': Patches})
        weights.append(model.get_weights())
        
    new_weights = list()

    for weights_list_tuple in zip(*weights):
        new_weights.append(
            [np.array(weights_).mean(axis=0)\
                for weights_ in zip(*weights_list_tuple)])
        


    new_weights = [np.array(x) for x in new_weights]
        
    model.set_weights(new_weights)
    model.save(out)
    os.system('cp '+os.path.join(os.path.join(dir_models,model_name) , "config.json ")+out)

@click.command()
@click.option(
    "--dir_models",
    "-dm",
    help="directory of models",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--out",
    "-o",
    help="output directory where ensembled model will be written.",
    type=click.Path(exists=False, file_okay=False),
)

def main(dir_models, out):
    run_ensembling(dir_models, out)
    
