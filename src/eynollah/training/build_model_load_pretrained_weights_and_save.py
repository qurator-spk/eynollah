import sys
import click
import tensorflow as tf

from .models import resnet50_unet


def configuration():
    try:
        for device in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("no GPU device available", file=sys.stderr)

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
