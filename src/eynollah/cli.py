import sys
import click
from ocrd_utils import initLogging, getLevelName, getLogger
from eynollah.eynollah import Eynollah, Eynollah_ocr
from eynollah.sbb_binarize import SbbBinarizer

@click.group()
def main():
    pass

@main.command()
@click.option(
    "--dir_xml",
    "-dx",
    help="directory of GT page-xml files",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--dir_out_modal_image",
    "-domi",
    help="directory where ground truth images would be written",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--dir_out_classes",
    "-docl",
    help="directory where ground truth classes would be written",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--input_height",
    "-ih",
    help="input height",
)
@click.option(
    "--input_width",
    "-iw",
    help="input width",
)
@click.option(
    "--min_area_size",
    "-min",
    help="min area size of regions considered for reading order training.",
)
def machine_based_reading_order(dir_xml, dir_out_modal_image, dir_out_classes, input_height, input_width, min_area_size):
    xml_files_ind = os.listdir(dir_xml)

@main.command()
@click.option('--patches/--no-patches', default=True, help='by enabling this parameter you let the model to see the image in patches.')
@click.option('--model_dir', '-m', type=click.Path(exists=True, file_okay=False), required=True, help='directory containing models for prediction')
@click.argument('input_image')
@click.argument('output_image')
@click.option(
    "--dir_in",
    "-di",
    help="directory of images",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--dir_out",
    "-do",
    help="directory where the binarized images will be written",
    type=click.Path(exists=True, file_okay=False),
)
def binarization(patches, model_dir, input_image, output_image, dir_in, dir_out):
    if not dir_out and (dir_in):
        print("Error: You used -di but did not set -do")
        sys.exit(1)
    elif dir_out and not (dir_in):
        print("Error: You used -do to write out binarized images but have not set -di")
        sys.exit(1)
    SbbBinarizer(model_dir).run(image_path=input_image, use_patches=patches, save=output_image, dir_in=dir_in, dir_out=dir_out)




@main.command()
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
    "--overwrite",
    "-O",
    help="overwrite (instead of skipping) if output xml exists",
    is_flag=True,
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
    "--extract_only_images/--disable-extracting_only_images",
    "-eoi/-noeoi",
    is_flag=True,
    help="If a directory is given, only images in documents will be cropped and saved there and the other processing will not be done",
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
    "--right2left/--left2right",
    "-r2l/-l2r",
    is_flag=True,
    help="if this parameter set to true, this tool will extract right-to-left reading order.",
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
    "--reading_order_machine_based/--heuristic_reading_order",
    "-romb/-hro",
    is_flag=True,
    help="if this parameter set to true, this tool would apply machine based reading order detection",
)
@click.option(
    "--do_ocr",
    "-ocr/-noocr",
    is_flag=True,
    help="if this parameter set to true, this tool will try to do ocr",
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
    "--skip_layout_and_reading_order",
    "-slro/-noslro",
    is_flag=True,
    help="if this parameter set to true, this tool will ignore layout detection and reading order. It means that textline detection will be done within printspace and contours of textline will be written in xml output file.",
)
@click.option(
    "--log_level",
    "-l",
    type=click.Choice(['OFF', 'DEBUG', 'INFO', 'WARN', 'ERROR']),
    help="Override log level globally to this",
)

def layout(image, out, overwrite, dir_in, model, save_images, save_layout, save_deskewed, save_all, extract_only_images, save_page, enable_plotting, allow_enhancement, curved_line, textline_light, full_layout, tables, right2left, input_binary, allow_scaling, headers_off, light_version, reading_order_machine_based, do_ocr, num_col_upper, num_col_lower, skip_layout_and_reading_order, ignore_page_extraction, log_level):
    initLogging()
    if log_level:
        getLogger('eynollah').setLevel(getLevelName(log_level))
    if not enable_plotting and (save_layout or save_deskewed or save_all or save_page or save_images or allow_enhancement):
        raise ValueError("Plotting with -sl, -sd, -sa, -sp, -si or -ae also requires -ep")
    elif enable_plotting and not (save_layout or save_deskewed or save_all or save_page or save_images or allow_enhancement):
        raise ValueError("Plotting with -ep also requires -sl, -sd, -sa, -sp, -si or -ae")
    if textline_light and not light_version:
        raise ValueError("Light textline detection with -tll also requires -light")
    if light_version and not textline_light:
        raise ValueError("Light version with -light also requires light textline detection -tll")
    if extract_only_images and allow_enhancement:
        raise ValueError("Image extraction with -eoi can not be enabled alongside allow_enhancement -ae")
    if extract_only_images and allow_scaling:
        raise ValueError("Image extraction with -eoi can not be enabled alongside allow_scaling -as")
    if extract_only_images and light_version:
        raise ValueError("Image extraction with -eoi can not be enabled alongside light_version -light")
    if extract_only_images and curved_line:
        raise ValueError("Image extraction with -eoi can not be enabled alongside curved_line -cl")
    if extract_only_images and textline_light:
        raise ValueError("Image extraction with -eoi can not be enabled alongside textline_light -tll")
    if extract_only_images and full_layout:
        raise ValueError("Image extraction with -eoi can not be enabled alongside full_layout -fl")
    if extract_only_images and tables:
        raise ValueError("Image extraction with -eoi can not be enabled alongside tables -tab")
    if extract_only_images and right2left:
        raise ValueError("Image extraction with -eoi can not be enabled alongside right2left -r2l")
    if extract_only_images and headers_off:
        raise ValueError("Image extraction with -eoi can not be enabled alongside headers_off -ho")
    if image is None and dir_in is None:
        raise ValueError("Either a single image -i or a dir_in -di is required")
    eynollah = Eynollah(
        model,
        logger=getLogger('eynollah'),
        dir_out=out,
        dir_of_cropped_images=save_images,
        extract_only_images=extract_only_images,
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
        right2left=right2left,
        input_binary=input_binary,
        allow_scaling=allow_scaling,
        headers_off=headers_off,
        light_version=light_version,
        ignore_page_extraction=ignore_page_extraction,
        reading_order_machine_based=reading_order_machine_based,
        do_ocr=do_ocr,
        num_col_upper=num_col_upper,
        num_col_lower=num_col_lower,
        skip_layout_and_reading_order=skip_layout_and_reading_order,
    )
    if dir_in:
        eynollah.run(dir_in=dir_in, overwrite=overwrite)
    else:
        eynollah.run(image_filename=image, overwrite=overwrite)


@main.command()
@click.option(
    "--dir_in",
    "-di",
    help="directory of images",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--out",
    "-o",
    help="directory to write output xml data",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--dir_xmls",
    "-dx",
    help="directory of xmls",
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
    "--tr_ocr",
    "-trocr/-notrocr",
    is_flag=True,
    help="if this parameter set to true, transformer ocr will be applied, otherwise cnn_rnn model.",
)
@click.option(
    "--export_textline_images_and_text",
    "-etit/-noetit",
    is_flag=True,
    help="if this parameter set to true, images and text in xml will be exported into output dir. This files can be used for training a OCR engine.",
)
@click.option(
    "--do_not_mask_with_textline_contour",
    "-nmtc/-mtc",
    is_flag=True,
    help="if this parameter set to true, cropped textline images will not be masked with textline contour.",
)
@click.option(
    "--log_level",
    "-l",
    type=click.Choice(['OFF', 'DEBUG', 'INFO', 'WARN', 'ERROR']),
    help="Override log level globally to this",
)

def ocr(dir_in, out, dir_xmls, model, tr_ocr, export_textline_images_and_text, do_not_mask_with_textline_contour, log_level):
    initLogging()
    if log_level:
        getLogger('eynollah').setLevel(getLevelName(log_level))
    eynollah_ocr = Eynollah_ocr(
        dir_xmls=dir_xmls,
        dir_in=dir_in,
        dir_out=out,
        dir_models=model,
        tr_ocr=tr_ocr,
        export_textline_images_and_text=export_textline_images_and_text,
        do_not_mask_with_textline_contour=do_not_mask_with_textline_contour,
    )
    eynollah_ocr.run()

if __name__ == "__main__":
    main()
