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
    required=True,
)
@click.option(
    "--out",
    "-o",
    help="directory to write output xml data",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--model",
    "-m",
    help="directory of models",
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
    help="if this parameter set to true, this tool will try to return contoure of textlines instead of rectabgle bounding box of textline. This should be taken into account that with this option the tool need more time to do process.",
)
@click.option(
    "--full-layout/--no-full-layout",
    "-fl/-nofl",
    is_flag=True,
    help="if this parameter set to true, this tool will try to return all elements of layout.",
)
@click.option(
    "--allow_scaling/--no-allow-scaling",
    "-as/-noas",
    is_flag=True,
    help="if this parameter set to true, this tool would check the scale and if needed it will scale it to perform better layout detection",
)
@click.option(
    "--headers-off/--headers-on",
    "-ho/-noho",
    is_flag=True,
    help="if this parameter set to true, this tool would ignore headers role in reading order",
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
    model,
    save_images,
    save_layout,
    save_deskewed,
    save_all,
    enable_plotting,
    allow_enhancement,
    curved_line,
    full_layout,
    allow_scaling,
    headers_off,
    log_level
):
    if log_level:
        setOverrideLogLevel(log_level)
    initLogging()
    if not enable_plotting and (save_layout or save_deskewed or save_all or save_images):
        print("Error: You used one of -sl, -sd, -sa or -si but did not enable plotting with -ep")
        sys.exit(1)
    elif enable_plotting and not (save_layout or save_deskewed or save_all or save_images):
        print("Error: You used -ep to enable plotting but set none of -sl, -sd, -sa or -si")
        sys.exit(1)
    eynollah = Eynollah(
        image_filename=image,
        dir_out=out,
        dir_models=model,
        dir_of_cropped_images=save_images,
        dir_of_layout=save_layout,
        dir_of_deskewed=save_deskewed,
        dir_of_all=save_all,
        enable_plotting=enable_plotting,
        allow_enhancement=allow_enhancement,
        curved_line=curved_line,
        full_layout=full_layout,
        allow_scaling=allow_scaling,
        headers_off=headers_off,
    )
    pcgts = eynollah.run()
    eynollah.writer.write_pagexml(pcgts)

if __name__ == "__main__":
    main()
