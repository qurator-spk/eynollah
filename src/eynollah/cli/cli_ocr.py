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
    "--dir_in",
    "-di",
    help="directory of input images (instead of --image)",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--dir_in_bin",
    "-dib",
    help=("directory of binarized images (in addition to --dir_in for RGB images; filename stems must match the RGB image files, with '.png'. \n                                                                          Perform prediction using both RGB and binary images. (This may improve results for certain document images.)"),
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--dir_xmls",
    "-dx",
    help="directory of input PAGE-XML files (in addition to --dir_in; filename stems must match the image files, with '.xml' suffix).",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--out",
    "-o",
    help="directory for output PAGE-XML files",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--dir_out_image_text",
    "-doit",
    help="directory for output images, newly rendered with predicted text",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--overwrite",
    "-O",
    help="overwrite (instead of skipping) if output xml exists",
    is_flag=True,
)
@click.option(
    "--tr_ocr",
    "-trocr",
    is_flag=True,
    help="use transformer OCR (instead of classic CNN-RNN) model",
)
@click.option(
    "--do_not_mask_with_textline_contour",
    "-nmtc",
    is_flag=True,
    help="skip masking each cropped textline image with its corresponding textline contour",
)
@click.option(
    "--batch_size",
    "-bs",
    default=0,
    type=click.IntRange(min=0),
    help="number of inference batch size. Default b_s for trocr and cnn_rnn models are 2 and 8 respectively",
)
@click.option(
    "--min_conf_value_of_textline_text",
    "-min_conf",
    default=0.3,
    type=click.FloatRange(min=0.0, max=1.0),
    help="minimum OCR confidence threshold. Text lines with a lower confidence value will not be included in the output XML file.",
)
@click.pass_context
def ocr_cli(
    ctx,
    image,
    dir_in,
    dir_in_bin,
    dir_xmls,
    out,
    dir_out_image_text,
    overwrite,
    tr_ocr,
    do_not_mask_with_textline_contour,
    batch_size,
    min_conf_value_of_textline_text,
):
    """
    Recognize text with a CNN/RNN or transformer ML model.
    """
    assert bool(image) != bool(dir_in), "Either -i (single image) or -di (directory) must be provided, but not both."
    from ..eynollah_ocr import Eynollah_ocr
    eynollah_ocr = Eynollah_ocr(
        model_zoo=ctx.obj.model_zoo,
        device=ctx.obj.device,
        tr_ocr=tr_ocr,
        do_not_mask_with_textline_contour=do_not_mask_with_textline_contour,
        batch_size=batch_size,
        min_conf_value_of_textline_text=min_conf_value_of_textline_text,
    )
    eynollah_ocr.run(overwrite=overwrite,
                     dir_in=dir_in,
                     dir_in_bin=dir_in_bin,
                     image_filename=image,
                     dir_xmls=dir_xmls,
                     dir_out_image_text=dir_out_image_text,
                     dir_out=out,
    )
