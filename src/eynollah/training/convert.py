import os
from pathlib import Path
from shutil import copy2
import logging

import click

@click.command(context_settings=dict(
    help_option_names=['-h', '--help'],
    show_default=True))
@click.option(
    "--rebuild",
    "-r",
    help="build new model from code and then load existing weights (requires input in SavedModel directory format with config.json present)",
    is_flag=True
)
@click.option(
    "--format",
    "-f",
    "format_",
    help="data format to convert to",
    type=click.Choice(["hdf5", "keras", "tf", "tf-serving", "onnx"]),
    default="tf"
)
@click.option(
    "--in",
    "-i",
    "in_",
    help="path to input model (file in hdf5 / keras format, or directory in tf format)",
    required=True,
    type=click.Path(exists=True, dir_okay=True)
)
@click.option(
    "--out",
    "-o",
    help="path to output model  (file in hdf5 / keras / onnx format, or directory in tf / tf-serving format)",
    required=True,
    type=click.Path(exists=False, dir_okay=True)
)
def convert_cli(rebuild, format_, in_, out):
    """
    convert models for inference

    Load model from path, optionally by rebuilding, convert to output format and write model to path.
    """
    os.environ['TF_USE_LEGACY_KERAS'] = '1' # avoid Keras 3 after TF 2.15
    from ocrd_utils import tf_disable_interactive_logs
    tf_disable_interactive_logs()

    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.models import Model as KerasModel
    
    model_path = Path(in_)
    config_path = model_path / "config.json"
    if model_path.is_dir():
        assert (model_path / "keras_metadata.pb").exists(), (
            "input directory must be Keras model in SavedModel format")
    if rebuild:
        from .train import ex
        from .models import get_model

        assert config_path.exists(), (
            "rebuilding requires input model in SavedModel format with config.json")

        # merge defaults with existing config file
        ex.add_config(str(config_path))
        # some models deviate between training and inference
        ex.add_config(inference=True)
        # just retrieve final config (via pseudo-run)
        ex.main(lambda: 0)
        config = ex.run(options={'--loglevel': 'ERROR'}).config
        # use the config to capture the model builder
        model = get_model(config, logging.root)
        model.load_weights(model_path).assert_existing_objects_matched().expect_partial()
    else:
        model = load_model(model_path, compile=False)

        if isinstance(model, KerasModel):
            # cnn-rnn-ocr task deviates between training and inference
            try:
                model.get_layer(name='ctc_loss')
            except ValueError:
                pass
            else:
                model = KerasModel(
                    model.get_layer(name='image').input,
                    model.get_layer(name='dense2').output)

    if format_ in ["hdf5", "keras", "tf"]:
        kwargs = {"save_format": {"hdf5": "h5"}.get(format_, format_)}
        if format_ != "keras":
            kwargs["include_optimizer"] = False
        model.save(out, **kwargs)
    elif format_ == "tf-serving":
        model.export(out)
    elif format_ == "onnx":
        import tf2onnx
        tf2onnx.convert.from_keras(model, opset=18, output_path=out)
    else:
        raise ValueError("unknown output format '%s'" % format_)

    # copy config.json if possible
    if config_path.exists() and format_ in ['tf', 'tf-serving']:
        copy2(config_path, Path(out) / config_path.name)

        
