from pathlib import Path
from typing import Set, Tuple
import click

from eynollah.model_zoo.default_specs import MODELS_VERSION

@click.group()
@click.pass_context
def models_cli(
    ctx,
):
    """
    Organize models for the various runners in eynollah.
    """
    assert ctx.obj.model_zoo


@models_cli.command('list')
@click.pass_context
def list_models(
    ctx,
):
    """
    List all the models in the zoo
    """
    print(f"Model basedir: {ctx.obj.model_zoo.model_basedir}")
    print(f"Model overrides: {ctx.obj.model_zoo.model_overrides}")
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
        print(f"mkdir -vp {dir}")
    for (src, dst) in copies:
        print(f"cp -vr {src} {dst}")
    for dir in mkdirs:
        zip_path = Path(f'../{dir.parent.name}.zip')
        print(f"(cd {dir}/..; zip -vr {zip_path} models_eynollah)")
