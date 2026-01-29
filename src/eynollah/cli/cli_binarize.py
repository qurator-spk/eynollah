import click

@click.command()
@click.option('--patches/--no-patches', default=True, help='by enabling this parameter you let the model to see the image in patches.')
@click.option(
    "--input-image", "--image",
    "-i",
    help="input image filename",
    type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "--dir_in",
    "-di",
    help="directory of input images (instead of --image)",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--output",
    "-o",
    help="output image (if using -i) or output image directory (if using -di)",
    type=click.Path(file_okay=True, dir_okay=True),
    required=True,
)
@click.pass_context
def binarize_cli(
    ctx,
    patches,
    input_image,
    dir_in,
    output,
):
    """
    Binarize images with a ML model
    """
    from ..sbb_binarize import SbbBinarizer
    assert bool(input_image) != bool(dir_in), "Either -i (single input) or -di (directory) must be provided, but not both."
    binarizer = SbbBinarizer(model_zoo=ctx.obj.model_zoo)
    binarizer.run(
        image_path=input_image,
        use_patches=patches,
        output=output,
        dir_in=dir_in
    )

