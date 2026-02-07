import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import click
import sys

from .build_model_load_pretrained_weights_and_save import build_model_load_pretrained_weights_and_save
from .generate_gt_for_training import main as generate_gt_cli
from .inference import main as inference_cli
from .train import ex
from .extract_line_gt import linegt_cli
from .weights_ensembling import main as ensemble_cli

@click.command(context_settings=dict(
        ignore_unknown_options=True,
))
@click.argument('SACRED_ARGS', nargs=-1, type=click.UNPROCESSED)
def train_cli(sacred_args):
    ex.run_commandline([sys.argv[0]] + list(sacred_args))

@click.group('training')
def main():
    pass

main.add_command(build_model_load_pretrained_weights_and_save)
main.add_command(generate_gt_cli, 'generate-gt')
main.add_command(inference_cli, 'inference')
main.add_command(train_cli, 'train')
main.add_command(linegt_cli, 'export_textline_images_and_text')
main.add_command(ensemble_cli, 'ensembling')
