from dataclasses import dataclass
from typing import List, Tuple
import click
from .model_zoo import EynollahModelZoo

@dataclass()
class EynollahCliCtx():
    model_basedir: str
    model_overrides: List[Tuple[str, str, str]]


@click.group()
def models_cli():
    """
    Organize models for the various runners in eynollah.
    """

@models_cli.command('list')
@click.option(
    "--model",
    "-m",
    'model_basedir',
    help="directory of models",
    type=click.Path(exists=True, file_okay=False),
    # default=f"{os.environ['HOME']}/.local/share/ocrd-resources/ocrd-eynollah-segment",
    required=True,
)
@click.option(
    "--model-overrides",
    "-mv",
    help="override default versions of model categories, syntax is 'CATEGORY VARIANT PATH', e.g 'region light /path/to/model'. See eynollah list-models for the full list",
    type=(str, str, str),
    multiple=True,
)
@click.pass_context
def list_models(
    ctx,
    model_basedir: str,
    model_overrides: List[Tuple[str, str, str]],
):
    """
        List all the models in the zoo
    """
    ctx.obj = EynollahCliCtx(
        model_basedir=model_basedir,
        model_overrides=model_overrides
    )
    print(EynollahModelZoo(basedir=ctx.obj.model_basedir, model_overrides=ctx.obj.model_overrides))

