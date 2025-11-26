import click

@click.command()
@click.option(
    "--image",
    "-i",
    help="input image filename",
    type=click.Path(exists=True, dir_okay=False),
)

@click.option(
    "--out",
    "-o",
    help="directory for output PAGE-XML files",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--overwrite",
    "-O",
    help="overwrite (instead of skipping) if output xml exists",
    is_flag=True,
)
@click.option(
    "--dir_in",
    "-di",
    help="directory of input images (instead of --image)",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--save_images",
    "-si",
    help="if a directory is given, images in documents will be cropped and saved there",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--save_layout",
    "-sl",
    help="if a directory is given, plot of layout will be saved there",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--save_deskewed",
    "-sd",
    help="if a directory is given, deskewed image will be saved there",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--save_all",
    "-sa",
    help="if a directory is given, all plots needed for documentation will be saved there",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--save_page",
    "-sp",
    help="if a directory is given, page crop of image will be saved there",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--enable-plotting/--disable-plotting",
    "-ep/-noep",
    is_flag=True,
    help="If set, will plot intermediary files and images",
)
@click.option(
    "--input_binary/--input-RGB",
    "-ib/-irgb",
    is_flag=True,
    help="In general, eynollah uses RGB as input but if the input document is very dark, very bright or for any other reason you can turn on input binarization. When this flag is set, eynollah will binarize the RGB input document, you should always provide RGB images to eynollah.",
)
@click.option(
    "--ignore_page_extraction/--extract_page_included",
    "-ipe/-epi",
    is_flag=True,
    help="if this parameter set to true, this tool would ignore page extraction",
)
@click.option(
    "--reading_order_machine_based/--heuristic_reading_order",
    "-romb/-hro",
    is_flag=True,
    help="if this parameter set to true, this tool would apply machine based reading order detection",
)
@click.option(
    "--num_col_upper",
    "-ncu",
    help="lower limit of columns in document image",
)
@click.option(
    "--num_col_lower",
    "-ncl",
    help="upper limit of columns in document image",
)
@click.option(
    "--threshold_art_class_layout",
    "-tharl",
    help="threshold of artifical class in the case of layout detection. The default value is 0.1",
)
@click.option(
    "--threshold_art_class_textline",
    "-thart",
    help="threshold of artifical class in the case of textline detection. The default value is 0.1",
)
@click.option(
    "--skip_layout_and_reading_order",
    "-slro/-noslro",
    is_flag=True,
    help="if this parameter set to true, this tool will ignore layout detection and reading order. It means that textline detection will be done within printspace and contours of textline will be written in xml output file.",
)
@click.pass_context
def extract_images_cli(
    ctx,
    image,
    out,
    overwrite,
    dir_in,
    save_images,
    save_layout,
    save_deskewed,
    save_all,
    save_page,
    enable_plotting,
    input_binary,
    reading_order_machine_based,
    num_col_upper,
    num_col_lower,
    threshold_art_class_textline,
    threshold_art_class_layout,
    skip_layout_and_reading_order,
    ignore_page_extraction,
):
    """
    Detect Layout (with optional image enhancement and reading order detection)
    """
    assert enable_plotting or not save_layout, "Plotting with -sl also requires -ep"
    assert enable_plotting or not save_deskewed, "Plotting with -sd also requires -ep"
    assert enable_plotting or not save_all, "Plotting with -sa also requires -ep"
    assert enable_plotting or not save_page, "Plotting with -sp also requires -ep"
    assert enable_plotting or not save_images, "Plotting with -si also requires -ep"
    assert not enable_plotting or save_layout or save_deskewed or save_all or save_page or save_images, \
        "Plotting with -ep also requires -sl, -sd, -sa, -sp, -si or -ae"
    assert bool(image) != bool(dir_in), "Either -i (single input) or -di (directory) must be provided, but not both."

    from ..extract_images import EynollahImageExtractor
    extractor = EynollahImageExtractor(
        model_zoo=ctx.obj.model_zoo,
        enable_plotting=enable_plotting,
        input_binary=input_binary,
        ignore_page_extraction=ignore_page_extraction,
        reading_order_machine_based=reading_order_machine_based,
        num_col_upper=num_col_upper,
        num_col_lower=num_col_lower,
        skip_layout_and_reading_order=skip_layout_and_reading_order,
        threshold_art_class_textline=threshold_art_class_textline,
        threshold_art_class_layout=threshold_art_class_layout,
    )
    extractor.run(overwrite=overwrite,
                 image_filename=image,
                 dir_in=dir_in,
                 dir_out=out,
                 dir_of_cropped_images=save_images,
                 dir_of_layout=save_layout,
                 dir_of_deskewed=save_deskewed,
                 dir_of_all=save_all,
                 dir_save_page=save_page,
    )

