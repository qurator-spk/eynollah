from dataclasses import dataclass
import logging
import sys
import os
from typing import Union

import click

from ..model_zoo import EynollahModelZoo
from .cli_models import models_cli

@dataclass()
class EynollahCliCtx:
    """
    Holds options relevant for all eynollah subcommands
    """
    model_zoo: EynollahModelZoo
    log_level : Union[str, None] = 'INFO'


@click.group()
@click.option(
    "--model-basedir",
    "-m",
    help="directory of models",
    # NOTE: not mandatory to exist so --help for subcommands works but will log a warning
    #       and raise exception when trying to load models in the CLI
    # type=click.Path(exists=True),
    default=f'{os.getcwd()}/models_eynollah',
)
@click.option(
    "--model-overrides",
    "-mv",
    help="override default versions of model categories, syntax is 'CATEGORY VARIANT PATH', e.g 'region light /path/to/model'. See eynollah list-models for the full list",
    type=(str, str, str),
    multiple=True,
)
@click.option(
    "--log_level",
    "-l",
    type=click.Choice(['OFF', 'DEBUG', 'INFO', 'WARN', 'ERROR']),
    help="Override log level globally to this",
)
@click.pass_context
def main(ctx, model_basedir, model_overrides, log_level):
    """
    eynollah - Document Layout Analysis, Image Enhancement, OCR
    """
    # Initialize logging
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.NOTSET)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(formatter)
    logging.getLogger('eynollah').addHandler(console_handler)
    logging.getLogger('eynollah').setLevel(log_level or logging.INFO)
    # Initialize model zoo
    model_zoo = EynollahModelZoo(basedir=model_basedir, model_overrides=model_overrides)
    # Initialize CLI context
    ctx.obj = EynollahCliCtx(
        model_zoo=model_zoo,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
