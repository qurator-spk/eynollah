from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple
import click

from eynollah.model_zoo.default_specs import MODELS_VERSION
from .model_zoo import EynollahModelZoo


@dataclass()
class EynollahCliCtx:
    model_zoo: EynollahModelZoo


@click.group()
@click.pass_context
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
def models_cli(
    ctx,
    model_basedir: str,
    model_overrides: List[Tuple[str, str, str]],
):
    """
    Organize models for the various runners in eynollah.
    """
    ctx.obj = EynollahCliCtx(model_zoo=EynollahModelZoo(basedir=model_basedir, model_overrides=model_overrides))


@models_cli.command('list')
@click.pass_context
def list_models(
    ctx,
):
    """
    List all the models in the zoo
    """
    print(ctx.obj.model_zoo)


@models_cli.command('package')
@click.option(
    '--set-version', '-V', 'version', help="Version to use for packaging", default=MODELS_VERSION, show_default=True
)
@click.argument('output_dir')
@click.pass_context
def package(
    ctx,
    version,
    output_dir,
):
    """
    Generate shell code to copy all the models in the zoo into properly named folders in OUTPUT_DIR for distribution.

    eynollah models -m SRC package OUTPUT_DIR

    SRC should contain a directory "models_eynollah" containing all the models.
    """
    mkdirs: Set[Path] = set([])
    copies: Set[Tuple[Path, Path]] = set([])
    for spec in ctx.obj.model_zoo.specs.specs:
        # skip these as they are dependent on the ocr model
        if spec.category in ('num_to_char', 'characters'):
            continue
        src: Path = ctx.obj.model_zoo.model_path(spec.category, spec.variant)
        # Only copy the top-most directory relative to models_eynollah
        while src.parent.name != 'models_eynollah':
            src = src.parent
        for dist in spec.dists:
            dist_dir = Path(f"{output_dir}/models_{dist}_{version}/models_eynollah")
            copies.add((src, dist_dir))
            mkdirs.add(dist_dir)
    for dir in mkdirs:
        print(f"mkdir -p {dir}")
    for (src, dst) in copies:
        print(f"cp -r {src} {dst}")
    for dir in mkdirs:
        zip_path = Path(f'../{dir.parent.name}.zip')
        print(f"(cd {dir}/..; zip -r {zip_path} models_eynollah)")
