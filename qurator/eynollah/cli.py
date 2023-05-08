import sys
import click
from ocrd_utils import initLogging, setOverrideLogLevel
from qurator.eynollah.eynollah import Eynollah


@click.command()
@click.option(
    "--image",
    "-i",
    help="image filename",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--out",
    "-o",
    help="directory to write output xml data",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--dir_in",
    "-di",
    help="directory of images",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--model",
    "-m",
    help="directory of models",
    type=click.Path(exists=True, file_okay=False),
    required=True,
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
    "--allow-enhancement/--no-allow-enhancement",
    "-ae/-noae",
    is_flag=True,
    help="if this parameter set to true, this tool would check that input image need resizing and enhancement or not. If so output of resized and enhanced image and corresponding layout data will be written in out directory",
)
@click.option(
    "--curved-line/--no-curvedline",
    "-cl/-nocl",
    is_flag=True,
    help="if this parameter set to true, this tool will try to return contoure of textlines instead of rectangle bounding box of textline. This should be taken into account that with this option the tool need more time to do process.",
)
@click.option(
    "--textline_light/--no-textline_light",
    "-tll/-notll",
    is_flag=True,
    help="if this parameter set to true, this tool will try to return contoure of textlines instead of rectangle bounding box of textline with a faster method.",
)
@click.option(
    "--full-layout/--no-full-layout",
    "-fl/-nofl",
    is_flag=True,
    help="if this parameter set to true, this tool will try to return all elements of layout.",
)
@click.option(
    "--tables/--no-tables",
    "-tab/-notab",
    is_flag=True,
    help="if this parameter set to true, this tool will try to detect tables.",
)
@click.option(
    "--input_binary/--input-RGB",
    "-ib/-irgb",
    is_flag=True,
    help="in general, eynollah uses RGB as input but if the input document is strongly dark, bright or for any other reason you can turn binarized input on. This option does not mean that you have to provide a binary image, otherwise this means that the tool itself will binarized the RGB input document.",
)
@click.option(
    "--allow_scaling/--no-allow-scaling",
    "-as/-noas",
    is_flag=True,
    help="if this parameter set to true, this tool would check the scale and if needed it will scale it to perform better layout detection",
)
@click.option(
    "--headers_off/--headers-on",
    "-ho/-noho",
    is_flag=True,
    help="if this parameter set to true, this tool would ignore headers role in reading order",
)
@click.option(
    "--light_version/--original",
    "-light/-org",
    is_flag=True,
    help="if this parameter set to true, this tool would use lighter version",
)
@click.option(
    "--ignore_page_extraction/--extract_page_included",
    "-ipe/-epi",
    is_flag=True,
    help="if this parameter set to true, this tool would ignore page extraction",
)
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(['OFF', 'DEBUG', 'INFO', 'WARN', 'ERROR']),
    help="Override log level globally to this",
)
def main(
    image,
    out,
    dir_in,
    model,
    save_images,
    save_layout,
    save_deskewed,
    save_all,
    save_page,
    enable_plotting,
    allow_enhancement,
    curved_line,
    textline_light,
    full_layout,
    tables,
    input_binary,
    allow_scaling,
    headers_off,
    light_version,
    ignore_page_extraction,
    log_level
):
    if log_level:
        setOverrideLogLevel(log_level)
    initLogging()
    if not enable_plotting and (save_layout or save_deskewed or save_all or save_page or save_images or allow_enhancement):
        print("Error: You used one of -sl, -sd, -sa, -sp, -si or -ae but did not enable plotting with -ep")
        sys.exit(1)
    elif enable_plotting and not (save_layout or save_deskewed or save_all or save_page or save_images or allow_enhancement):
        print("Error: You used -ep to enable plotting but set none of -sl, -sd, -sa, -sp, -si or -ae")
        sys.exit(1)
    if textline_light and not light_version:
        print('Error: You used -tll to enable light textline detection but -light is not enabled')
        sys.exit(1)
    eynollah = Eynollah(
        image_filename=image,
        dir_out=out,
        dir_in=dir_in,
        dir_models=model,
        dir_of_cropped_images=save_images,
        dir_of_layout=save_layout,
        dir_of_deskewed=save_deskewed,
        dir_of_all=save_all,
        dir_save_page=save_page,
        enable_plotting=enable_plotting,
        allow_enhancement=allow_enhancement,
        curved_line=curved_line,
        textline_light=textline_light,
        full_layout=full_layout,
        tables=tables,
        input_binary=input_binary,
        allow_scaling=allow_scaling,
        headers_off=headers_off,
        light_version=light_version,
        ignore_page_extraction=ignore_page_extraction,
    )
    eynollah.run()
    #pcgts = eynollah.run()
    ##eynollah.writer.write_pagexml(pcgts)

if __name__ == "__main__":
    main()
