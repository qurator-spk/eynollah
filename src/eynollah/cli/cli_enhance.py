import click

@click.command(context_settings=dict(
    help_option_names=['-h', '--help'],
    show_default=True))
@click.option(
    "--image",
    "-i",
    help="input image filename",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--out",
    "-o",
    help="directory for output image files",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--overwrite",
    "-O",
    help="overwrite (instead of skipping) if output image exists",
    is_flag=True,
)
@click.option(
    "--dir_in",
    "-di",
    help="directory of input images (instead of --image)",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--num_col_upper",
    "-ncu",
    default=0,
    type=click.IntRange(min=0),
    help="lower limit of columns in document image",
)
@click.option(
    "--num_col_lower",
    "-ncl",
    default=0,
    type=click.IntRange(min=0),
    help="upper limit of columns in document image",
)
@click.option(
    "--save_org_scale",
    "-sos",
    is_flag=True,
    help="save the enhanced image in original image size",
)
@click.pass_context
def enhance_cli(ctx, image, out, overwrite, dir_in, num_col_upper, num_col_lower, save_org_scale):
    """
    Enhance image
    """
    assert bool(image) != bool(dir_in), "Either -i (single input) or -di (directory) must be provided, but not both."
    from ..image_enhancer import Enhancer
    enhancer = Enhancer(
        model_zoo=ctx.obj.model_zoo,
        device=ctx.obj.device,
        num_col_upper=num_col_upper,
        num_col_lower=num_col_lower,
        save_org_scale=save_org_scale,
    )
    enhancer.run(overwrite=overwrite,
                 dir_in=dir_in,
                 image_filename=image,
                 dir_out=out,
    )

