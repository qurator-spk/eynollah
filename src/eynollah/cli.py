import sys
import click
import logging
from ocrd_utils import initLogging, getLevelName, getLogger
from eynollah.eynollah import Eynollah, Eynollah_ocr
from eynollah.sbb_binarize import SbbBinarizer
from eynollah.image_enhancer import Enhancer
from eynollah.mb_ro_on_layout import machine_based_reading_order_on_layout

@click.group()
def main():
    pass

@main.command()
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
@click.option(
    "--model",
    "-m",
    help="directory of models",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--log_level",
    "-l",
    type=click.Choice(['OFF', 'DEBUG', 'INFO', 'WARN', 'ERROR']),
    help="Override log level globally to this",
)

def machine_based_reading_order(input, dir_in, out, model, log_level):
    assert bool(input) != bool(dir_in), "Either -i (single input) or -di (directory) must be provided, but not both."
    orderer = machine_based_reading_order_on_layout(model)
    if log_level:
        orderer.logger.setLevel(getLevelName(log_level))

    orderer.run(xml_filename=input,
                dir_in=dir_in,
                dir_out=out,
    )
    

@main.command()
@click.option('--patches/--no-patches', default=True, help='by enabling this parameter you let the model to see the image in patches.')
@click.option('--model_dir', '-m', type=click.Path(exists=True, file_okay=False), required=True, help='directory containing models for prediction')
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
@click.option(
    "--log_level",
    "-l",
    type=click.Choice(['OFF', 'DEBUG', 'INFO', 'WARN', 'ERROR']),
    help="Override log level globally to this",
)
def binarization(patches, model_dir, input_image, dir_in, output, log_level):
    assert bool(input_image) != bool(dir_in), "Either -i (single input) or -di (directory) must be provided, but not both."
    binarizer = SbbBinarizer(model_dir)
    if log_level:
        binarizer.log.setLevel(getLevelName(log_level))
    binarizer.run(image_path=input_image, use_patches=patches, output=output, dir_in=dir_in)


@main.command()
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
    "--model",
    "-m",
    help="directory of models",
    type=click.Path(exists=True, file_okay=False),
    required=True,
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
    "--save_org_scale/--no_save_org_scale",
    "-sos/-nosos",
    is_flag=True,
    help="if this parameter set to true, this tool will save the enhanced image in org scale.",
)
@click.option(
    "--log_level",
    "-l",
    type=click.Choice(['OFF', 'DEBUG', 'INFO', 'WARN', 'ERROR']),
    help="Override log level globally to this",
)

def enhancement(image, out, overwrite, dir_in, model, num_col_upper, num_col_lower, save_org_scale,  log_level):
    assert bool(image) != bool(dir_in), "Either -i (single input) or -di (directory) must be provided, but not both."
    initLogging()
    enhancer = Enhancer(
        model,
        num_col_upper=num_col_upper,
        num_col_lower=num_col_lower,
        save_org_scale=save_org_scale,
    )
    if log_level:
        enhancer.logger.setLevel(getLevelName(log_level))
    enhancer.run(overwrite=overwrite,
                 dir_in=dir_in,
                 image_filename=image,
                 dir_out=out,
    )

@main.command()
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
    "--model",
    "-m",
    help="directory of models",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--model_version",
    "-mv",
    help="override default versions of model categories",
    type=(str, str),
    multiple=True,
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
    "--transformer_ocr",
    "-tr/-notr",
    is_flag=True,
    help="if this parameter set to true, this tool will apply transformer ocr",
)
@click.option(
    "--batch_size_ocr",
    "-bs_ocr",
    help="number of inference batch size of ocr model. Default b_s for trocr and cnn_rnn models are 2 and 8 respectively",
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
# TODO move to top-level CLI context
@click.option(
    "--log_level",
    "-l",
    type=click.Choice(['OFF', 'DEBUG', 'INFO', 'WARN', 'ERROR']),
    help="Override 'eynollah' log level globally to this",
)
# 
@click.option(
    "--setup-logging",
    is_flag=True,
    help="Setup a basic console logger",
)

def layout(image, out, overwrite, dir_in, model, model_version, save_images, save_layout, save_deskewed, save_all, extract_only_images, save_page, enable_plotting, allow_enhancement, curved_line, textline_light, full_layout, tables, right2left, input_binary, allow_scaling, headers_off, light_version, reading_order_machine_based, do_ocr, transformer_ocr, batch_size_ocr, num_col_upper, num_col_lower, threshold_art_class_textline, threshold_art_class_layout, skip_layout_and_reading_order, ignore_page_extraction, log_level, setup_logging):
    if setup_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        getLogger('eynollah').addHandler(console_handler)
        getLogger('eynollah').setLevel(logging.INFO)
    else:
        initLogging()
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
    assert bool(image) != bool(dir_in), "Either -i (single input) or -di (directory) must be provided, but not both."
    eynollah = Eynollah(
        model,
        model_versions=model_version,
        extract_only_images=extract_only_images,
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
        transformer_ocr=transformer_ocr,
        batch_size_ocr=batch_size_ocr,
        num_col_upper=num_col_upper,
        num_col_lower=num_col_lower,
        skip_layout_and_reading_order=skip_layout_and_reading_order,
        threshold_art_class_textline=threshold_art_class_textline,
        threshold_art_class_layout=threshold_art_class_layout,
    )
    if log_level:
        eynollah.logger.setLevel(getLevelName(log_level))
    eynollah.run(overwrite=overwrite,
                 image_filename=image,
                 dir_in=dir_in,
                 dir_out=out,
                 dir_of_cropped_images=save_images,
                 dir_of_layout=save_layout,
                 dir_of_deskewed=save_deskewed,
                 dir_of_all=save_all,
                 dir_save_page=save_page,
    )

@main.command()
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
    help="directory of binarized images (in addition to --dir_in for RGB images; filename stems must match the RGB image files, with '.png' suffix).\nPerform prediction using both RGB and binary images. (This does not necessarily improve results, however it may be beneficial for certain document images.)",
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
    "--model",
    "-m",
    help="directory of models",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--model_name",
    help="Specific model file path to use for OCR",
    type=click.Path(exists=True, file_okay=False),
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
    "--batch_size",
    "-bs",
    help="number of inference batch size. Default b_s for trocr and cnn_rnn models are 2 and 8 respectively",
)
@click.option(
    "--dataset_abbrevation",
    "-ds_pref",
    help="in the case of extracting textline and text from a xml GT file user can add an abbrevation of dataset name to generated dataset",
)
@click.option(
    "--min_conf_value_of_textline_text",
    "-min_conf",
    help="minimum OCR confidence value. Text lines with a confidence value lower than this threshold will not be included in the output XML file.",
)
@click.option(
    "--log_level",
    "-l",
    type=click.Choice(['OFF', 'DEBUG', 'INFO', 'WARN', 'ERROR']),
    help="Override log level globally to this",
)

def ocr(image, dir_in, dir_in_bin, dir_xmls, out, dir_out_image_text, overwrite, model, model_name, tr_ocr, export_textline_images_and_text, do_not_mask_with_textline_contour, batch_size, dataset_abbrevation, min_conf_value_of_textline_text, log_level):
    initLogging()
        
    assert bool(model) != bool(model_name), "Either -m (model directory) or --model_name (specific model name) must be provided."
    assert not export_textline_images_and_text or not tr_ocr, "Exporting textline and text  -etit can not be set alongside transformer ocr -tr_ocr"
    assert not export_textline_images_and_text or not model, "Exporting textline and text  -etit can not be set alongside model -m"
    assert not export_textline_images_and_text or not batch_size, "Exporting textline and text  -etit can not be set alongside batch size -bs"
    assert not export_textline_images_and_text or not dir_in_bin, "Exporting textline and text  -etit can not be set alongside directory of bin images -dib"
    assert not export_textline_images_and_text or not dir_out_image_text, "Exporting textline and text  -etit can not be set alongside directory of images with predicted text -doit"
    assert bool(image) != bool(dir_in), "Either -i (single image) or -di (directory) must be provided, but not both."
    eynollah_ocr = Eynollah_ocr(
        dir_models=model,
        model_name=model_name,
        tr_ocr=tr_ocr,
        export_textline_images_and_text=export_textline_images_and_text,
        do_not_mask_with_textline_contour=do_not_mask_with_textline_contour,
        batch_size=batch_size,
        pref_of_dataset=dataset_abbrevation,
        min_conf_value_of_textline_text=min_conf_value_of_textline_text,
    )
    if log_level:
        eynollah_ocr.logger.setLevel(getLevelName(log_level))
    eynollah_ocr.run(overwrite=overwrite,
                     dir_in=dir_in,
                     dir_in_bin=dir_in_bin,
                     image_filename=image,
                     dir_xmls=dir_xmls,
                     dir_out_image_text=dir_out_image_text,
                     dir_out=out,
    )

if __name__ == "__main__":
    main()
