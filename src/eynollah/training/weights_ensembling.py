import os
from warnings import catch_warnings, simplefilter

import click
import numpy as np

os.environ['TF_USE_LEGACY_KERAS'] = '1' # avoid Keras 3 after TF 2.15
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ocrd_utils import tf_disable_interactive_logs
tf_disable_interactive_logs()
import tensorflow as tf
from tensorflow.keras.models import load_model

from ..patch_encoder import (
    PatchEncoder,
    Patches,
)
    
def run_ensembling(model_dirs, out_dir):
    all_weights = []

    for model_dir in model_dirs:
        assert os.path.isdir(model_dir), model_dir
        model = load_model(model_dir, compile=False,
                           custom_objects=dict(PatchEncoder=PatchEncoder,
                                               Patches=Patches))
        all_weights.append(model.get_weights())
        
    new_weights = []
    for layer_weights in zip(*all_weights):
        layer_weights = np.array([np.array(weights).mean(axis=0)
                                  for weights in zip(*layer_weights)])
        new_weights.append(layer_weights)

    #model = tf.keras.models.clone_model(model)
    model.set_weights(new_weights)

    model.save(out_dir)
    os.system('cp ' + os.path.join(model_dirs[0], "config.json ") + out_dir + "/")

@click.command()
@click.option(
    "--in",
    "-i",
    help="input directory of checkpoint models to be read",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--out",
    "-o",
    help="output directory where ensembled model will be written.",
    required=True,
    type=click.Path(exists=False, file_okay=False),
)
def ensemble_cli(in_, out):
    """
    mix multiple model weights

    Load a sequence of models and mix them into a single ensemble model
    by averaging their weights. Write the resulting model.
    """
    run_ensembling(in_, out)

