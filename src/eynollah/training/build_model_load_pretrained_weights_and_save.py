import click
import tensorflow as tf

from .models import resnet50_unet


def configuration():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

@click.command()
def build_model_load_pretrained_weights_and_save():
    n_classes = 2
    input_height = 224
    input_width = 448
    weight_decay = 1e-6
    pretraining = False
    dir_of_weights = 'model_bin_sbb_ens.h5'

    # configuration()

    model = resnet50_unet(n_classes, input_height, input_width, weight_decay, pretraining)
    model.load_weights(dir_of_weights)
    model.save('./name_in_another_python_version.h5')
