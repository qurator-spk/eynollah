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
@click.argument('input_image', required=False)
@click.argument('output_image', required=False)
@click.option(
    "--dir_in",
    "-di",
    help="directory of input images",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--dir_out",
    "-do",
    help="directory for output images",
    type=click.Path(exists=True, file_okay=False),
)
def binarization(patches, model_dir, input_image, output_image, dir_in, dir_out):
    assert (dir_out is None) == (dir_in is None), "Options -di and -do are mutually dependent"
    assert (input_image is None) == (output_image is None), "INPUT_IMAGE and OUTPUT_IMAGE are mutually dependent"
    assert (dir_in is None) != (input_image is None), "Specify either -di and -do options, or INPUT_IMAGE and OUTPUT_IMAGE"
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
    assert enable_plotting or not save_layout, "Plotting with -sl also requires -ep"
    assert enable_plotting or not save_deskewed, "Plotting with -sd also requires -ep"
    assert enable_plotting or not save_all, "Plotting with -sa also requires -ep"
    assert enable_plotting or not save_page, "Plotting with -sp also requires -ep"
    assert enable_plotting or not save_images, "Plotting with -si also requires -ep"
    assert enable_plotting or not allow_enhancement, "Plotting with -ae also requires -ep"
    assert not enable_plotting or save_layout or save_deskewed or save_all or save_page or save_images or allow_enhancement, \
        "Plotting with -ep also requires -sl, -sd, -sa, -sp, -si or -ae"
    assert textline_light == light_version, "Both light textline detection -tll and light version -light must be set or unset equally"
    assert not extract_only_images or not allow_enhancement, "Image extraction -eoi can not be set alongside allow_enhancement -ae"
    assert not extract_only_images or not allow_scaling, "Image extraction -eoi can not be set alongside allow_scaling -as"
    assert not extract_only_images or not light_version, "Image extraction -eoi can not be set alongside light_version -light"
    assert not extract_only_images or not curved_line, "Image extraction -eoi can not be set alongside curved_line -cl"
    assert not extract_only_images or not textline_light, "Image extraction -eoi can not be set alongside textline_light -tll"
    assert not extract_only_images or not full_layout, "Image extraction -eoi can not be set alongside full_layout -fl"
    assert not extract_only_images or not tables, "Image extraction -eoi can not be set alongside tables -tab"
    assert not extract_only_images or not right2left, "Image extraction -eoi can not be set alongside right2left -r2l"
    assert not extract_only_images or not headers_off, "Image extraction -eoi can not be set alongside headers_off -ho"
    assert image or dir_in, "Either a single image -i or a dir_in -di is required"
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
    "--dir_in_bin",
    "-dib",
    help="directory of binarized images. This should be given if you want to do prediction based on both rgb and bin images. And all bin images are png files",
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
    "--dir_out_image_text",
    "-doit",
    help="directory of images with predicted text",
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
    "--draw_texts_on_image",
    "-dtoi/-ndtoi",
    is_flag=True,
    help="if this parameter set to true, the predicted texts will be displayed on an image.",
)
@click.option(
    "--prediction_with_both_of_rgb_and_bin",
    "-brb/-nbrb",
    is_flag=True,
    help="If this parameter is set to True, the prediction will be performed using both RGB and binary images. However, this does not necessarily improve results; it may be beneficial for certain document images.",
)
@click.option(
    "--log_level",
    "-l",
    type=click.Choice(['OFF', 'DEBUG', 'INFO', 'WARN', 'ERROR']),
    help="Override log level globally to this",
)

def ocr(dir_in, dir_in_bin, out, dir_xmls, dir_out_image_text, model, tr_ocr, export_textline_images_and_text, do_not_mask_with_textline_contour, draw_texts_on_image, prediction_with_both_of_rgb_and_bin, log_level):
    initLogging()
    if log_level:
        getLogger('eynollah').setLevel(getLevelName(log_level))
    eynollah_ocr = Eynollah_ocr(
        dir_xmls=dir_xmls,
        dir_out_image_text=dir_out_image_text,
        dir_in=dir_in,
        dir_in_bin=dir_in_bin,
        dir_out=out,
        dir_models=model,
        tr_ocr=tr_ocr,
        export_textline_images_and_text=export_textline_images_and_text,
        do_not_mask_with_textline_contour=do_not_mask_with_textline_contour,
        draw_texts_on_image=draw_texts_on_image,
        prediction_with_both_of_rgb_and_bin=prediction_with_both_of_rgb_and_bin,
    )
    eynollah_ocr.run()

if __name__ == "__main__":
    main()
