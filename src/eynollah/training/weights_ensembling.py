import os
from typing import Optional
from warnings import catch_warnings, simplefilter

import click
import numpy as np

os.environ['TF_USE_LEGACY_KERAS'] = '1' # avoid Keras 3 after TF 2.15
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ocrd_utils import tf_disable_interactive_logs
tf_disable_interactive_logs()
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from transformers import VisionEncoderDecoderModel

from ..patch_encoder import (
    PatchEncoder,
    Patches,
)
    
def run_ensembling(dir_models, out, framework):
    ls_models = os.listdir(dir_models)
    # model: Optional[VisionEncoderDecoderModel] = None
    # model_name: Optional[str] = None
    if framework=="torch":
        models = []
        sd_models = []
        
        for model_name in ls_models:
            model = VisionEncoderDecoderModel.from_pretrained(os.path.join(dir_models, model_name))
            models.append(model)
            sd_models.append(model.state_dict())
        for key in sd_models[0]:
            sd_models[0][key] = sum(sd[key] for sd in sd_models) / len(sd_models)
            
        model.load_state_dict(sd_models[0])
        os.system("mkdir "+out)
        torch.save(model.state_dict(), os.path.join(out, "pytorch_model.bin"))
        os.system('cp ' + os.path.join(os.path.join(dir_models, model_name), "config.json") + " " + out)
        
    else:
        weights=[]

        for model_name in ls_models:
            model =  load_model(os.path.join(dir_models, model_name), compile=False, custom_objects={'PatchEncoder':PatchEncoder, 'Patches': Patches})
            weights.append(model.get_weights())
            
        new_weights = list()

        for weights_list_tuple in zip(*weights):
            new_weights.append(
                [np.array(weights_).mean(axis=0)\
                    for weights_ in zip(*weights_list_tuple)])
            


        new_weights = [np.array(x) for x in new_weights]
            
        model.set_weights(new_weights)
        model.save(out)
        os.system('cp '+os.path.join(os.path.join(dir_models, model_name), "config.json") + " " + out)
        os.system('cp '+os.path.join(os.path.join(dir_models, model_name), "characters_org.txt") + " " + out)

@click.command()
@click.option(
    "--in",
    "-i",
    "in_",
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
@click.option(
    "--framework",
    "-fw",
    help="this parameter gets tensorflow or torch as model framework",
    type=click.Choice(['torch', 'tensorflow']),
    default="tensorflow"
)

def ensemble_cli(in_, out, framework):
    """
    mix multiple model weights

    Load a sequence of models and mix them into a single ensemble model
    by averaging their weights. Write the resulting model.
    """
    run_ensembling(in_, out, framework)
