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
@click.pass_context
def readingorder_cli(ctx, input, dir_in, out):
    """
    Generate ReadingOrder with a ML model
    """
    from ..mb_ro_on_layout import machine_based_reading_order_on_layout
    assert bool(input) != bool(dir_in), "Either -i (single input) or -di (directory) must be provided, but not both."
    orderer = machine_based_reading_order_on_layout(model_zoo=ctx.obj.model_zoo)
    orderer.run(xml_filename=input,
                dir_in=dir_in,
                dir_out=out,
    )

