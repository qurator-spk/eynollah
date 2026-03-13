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
    help="if a directory is given, cropped images of pages will be saved there",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--save_layout",
    "-sl",
    help="if a directory is given, plots of layout detection will be saved there",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--save_deskewed",
    "-sd",
    help="if a directory is given, plots of page deskewing will be saved there",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--save_all",
    "-sa",
    help="if a directory is given, all plots needed will be saved there",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--save_page",
    "-sp",
    help="if a directory is given, plots of page cropping will be saved there",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--enable-plotting",
    "-ep",
    is_flag=True,
    help="plot intermediary diagnostic images to files",
)
@click.option(
    "--allow-enhancement",
    "-ae",
    is_flag=True,
    help="check whether input image need resizing and enhancement. If so, output of resized and enhanced image and corresponding layout data will be written in out directory",
)
@click.option(
    "--curved-line",
    "-cl",
    is_flag=True,
    help="try to return most precise textline contours by deskewing and detecting textlines for all text regions individually. Requires much more computation.",
)
@click.option(
    "--full-layout",
    "-fl",
    is_flag=True,
    help="return all elements of layout, including headings and drop-capitals",
)
@click.option(
    "--tables",
    "-tab",
    is_flag=True,
    help="try to detect table regions",
)
@click.option(
    "--right2left",
    "-r2l",
    is_flag=True,
    help="extract right-to-left reading order (instead of left-to-right)",
)
@click.option(
    "--input_binary",
    "-ib",
    is_flag=True,
    help="In general, eynollah uses RGB as input, but if the input document is very dark, very bright or for any other reason you can turn on internal binarization here. When set, eynollah will binarize the RGB input document first.",
)
@click.option(
    "--allow_scaling",
    "-as",
    is_flag=True,
    help="check the scale and if needed it will scale it to perform better layout detection",
)
@click.option(
    "--headers_off",
    "-ho",
    is_flag=True,
    help="ignore headers role in reading order",
)
@click.option(
    "--ignore_page_extraction",
    "-ipe",
    is_flag=True,
    help="ignore page extraction (cropping via page frame detection model)",
)
@click.option(
    "--reading_order_machine_based",
    "-romb",
    is_flag=True,
    help="apply model based reading order detection",
)
@click.option(
    "--num_col_upper",
    "-ncu",
    default=0,
    type=click.IntRange(min=0),
    help="lower limit of columns in document image; 0 means autodetected from model",
)
@click.option(
    "--num_col_lower",
    "-ncl",
    default=0,
    type=click.IntRange(min=0),
    help="upper limit of columns in document image; 0 means autodetected from model",
)
@click.option(
    "--threshold_art_class_layout",
    "-tharl",
    default=0.1,
    type=click.FloatRange(min=0.0, max=1.0),
    help="confidence threshold of artifical boundary class during region detection",
)
@click.option(
    "--threshold_art_class_textline",
    "-thart",
    default=0.1,
    type=click.FloatRange(min=0.0, max=1.0),
    help="confidence threshold of artifical boundary class during textline detection",
)
@click.option(
    "--skip_layout_and_reading_order",
    "-slro",
    is_flag=True,
    help="ignore layout detection and reading order, i.e. textline detection will be done within entire printspace, and textline contours will be written into a single overall text region.",
)
@click.option(
    "--num-jobs",
    "-j",
    default=0,
    type=click.IntRange(min=0),
    help="number of parallel images to process (also helps better utilise GPU if available); 0 means based on autodetected number of processor cores",
)
@click.pass_context
def layout_cli(
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
    allow_enhancement,
    curved_line,
    full_layout,
    tables,
    right2left,
    input_binary,
    allow_scaling,
    headers_off,
    reading_order_machine_based,
    num_col_upper,
    num_col_lower,
    threshold_art_class_textline,
    threshold_art_class_layout,
    skip_layout_and_reading_order,
    ignore_page_extraction,
    num_jobs,
):
    """
    Detect Layout (with optional image enhancement and reading order detection)
    """
    from ..eynollah import Eynollah
    assert enable_plotting or not save_layout, "Plotting with -sl also requires -ep"
    assert enable_plotting or not save_deskewed, "Plotting with -sd also requires -ep"
    assert enable_plotting or not save_all, "Plotting with -sa also requires -ep"
    assert enable_plotting or not save_page, "Plotting with -sp also requires -ep"
    assert enable_plotting or not save_images, "Plotting with -si also requires -ep"
    assert not enable_plotting or save_layout or save_deskewed or save_all or save_page or save_images or allow_enhancement, \
        "Plotting with -ep also requires -sl, -sd, -sa, -sp, -si or -ae"
    assert bool(image) != bool(dir_in), "Either -i (single input) or -di (directory) must be provided, but not both."
    eynollah = Eynollah(
        model_zoo=ctx.obj.model_zoo,
        enable_plotting=enable_plotting,
        allow_enhancement=allow_enhancement,
        curved_line=curved_line,
        full_layout=full_layout,
        tables=tables,
        right2left=right2left,
        input_binary=input_binary,
        allow_scaling=allow_scaling,
        headers_off=headers_off,
        ignore_page_extraction=ignore_page_extraction,
        reading_order_machine_based=reading_order_machine_based,
        num_col_upper=num_col_upper,
        num_col_lower=num_col_lower,
        skip_layout_and_reading_order=skip_layout_and_reading_order,
        threshold_art_class_textline=threshold_art_class_textline,
        threshold_art_class_layout=threshold_art_class_layout,
    )
    eynollah.run(overwrite=overwrite,
                 image_filename=image,
                 dir_in=dir_in,
                 dir_out=out,
                 dir_of_cropped_images=save_images,
                 dir_of_layout=save_layout,
                 dir_of_deskewed=save_deskewed,
                 dir_of_all=save_all,
                 dir_save_page=save_page,
                 num_jobs=num_jobs,
    )

