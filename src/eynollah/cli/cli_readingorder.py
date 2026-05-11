import click

@click.command()
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
    "--out",
    "-o",
    help="directory for output images",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--device",
    "-D",
    help="placement of computations in predictors for each model type; if none (by default), will try to use first available GPU or fall back to CPU; set string to force using a device (e.g. 'GPU0', 'GPU1' or 'CPU'). Can also be a comma-separated list of model category to device mappings (e.g. 'col_classifier:CPU,page:GPU0,*:GPU1')",
)
@click.pass_context
def readingorder_cli(ctx, input, dir_in, out, device):
    """
    Generate ReadingOrder with a ML model
    """
    from ..mb_ro_on_layout import Reorder
    assert bool(input) != bool(dir_in), "Either -i (single input) or -di (directory) must be provided, but not both."
    orderer = Reorder(model_zoo=ctx.obj.model_zoo, device=device)
    orderer.run(xml_filename=input,
                dir_in=dir_in,
                dir_out=out,
    )

