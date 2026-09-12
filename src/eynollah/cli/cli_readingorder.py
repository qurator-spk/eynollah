import click

@click.command(context_settings=dict(
    help_option_names=['-h', '--help'],
    show_default=True))
@click.option(
    "--model_based",
    "-mb",
    help="use machine-learning model instead of heuristic rules",
    is_flag=True,
)
@click.option(
    "--input",
    "-i",
    help="PAGE-XML input filename",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--dir_in",
    "-di",
    help="directory of PAGE-XML input files (instead of --input)",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--dir_imgs",
    "-dim",
    help="directory of image input files (in addition to --dir_in or --input; filename stems must match the XML files, with image file format suffixes). Not needed for --model_based.",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--out",
    "-o",
    help="directory for output images",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--overwrite",
    "-O",
    help="overwrite (instead of skipping) if output xml exists",
    is_flag=True,
)
@click.pass_context
def readingorder_cli(ctx, model_based, input, dir_in, dir_imgs, out, overwrite):
    """
    Generate ReadingOrder for existing segmentation from ML model or from heuristic rules
    """
    from ..reorder import Reorder
    assert bool(input) != bool(dir_in), "Either -i (single input) or -di (directory) must be provided, but not both."
    assert bool(model_based) or bool(dir_imgs), "For heuristic reading order, -dim must be provided, too."
    orderer = Reorder(model_zoo=ctx.obj.model_zoo,
                      device=ctx.obj.device,
                      model_based=model_based)
    orderer.run(overwrite=overwrite,
                xml_filename=input,
                dir_in=dir_in,
                dir_imgs=dir_imgs,
                dir_out=out,
    )

