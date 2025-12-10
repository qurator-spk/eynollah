from logging import Logger, getLogger
from typing import Optional
from pathlib import Path
import os

import click
import cv2
import xml.etree.ElementTree as ET
import numpy as np

from ..utils import is_image_filename

@click.command()
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
    "--dir_xmls",
    "-dx",
    help="directory of input PAGE-XML files (in addition to --dir_in; filename stems must match the image files, with '.xml' suffix).",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--out",
    "-o",
    'dir_out',
    help="directory for output PAGE-XML files",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--dataset_abbrevation",
    "-ds_pref",
    'pref_of_dataset',
    help="in the case of extracting textline and text from a xml GT file user can add an abbrevation of dataset name to generated dataset",
)
@click.option(
    "--do_not_mask_with_textline_contour",
    "-nmtc/-mtc",
    is_flag=True,
    help="if this parameter set to true, cropped textline images will not be masked with textline contour.",
)
def linegt_cli(
    image,
    dir_in,
    dir_xmls,
    dir_out,
    pref_of_dataset,
    do_not_mask_with_textline_contour,
):
    assert bool(dir_in) ^ bool(image), "Set --dir-in or --image-filename, not both"
    if dir_in:
        ls_imgs = [
            os.path.join(dir_in, image) for image in filter(is_image_filename, os.listdir(dir_in))
        ]
    else:
        assert image
        ls_imgs = [image]

    for dir_img in ls_imgs:
        file_name = Path(dir_img).stem
        dir_xml = os.path.join(dir_xmls, file_name + '.xml')

        img = cv2.imread(dir_img)

        total_bb_coordinates = []

        tree1 = ET.parse(dir_xml, parser=ET.XMLParser(encoding="utf-8"))
        root1 = tree1.getroot()
        alltags = [elem.tag for elem in root1.iter()]

        name_space = alltags[0].split('}')[0]
        name_space = name_space.split('{')[1]

        region_tags = np.unique([x for x in alltags if x.endswith('TextRegion')])

        cropped_lines_region_indexer = []

        indexer_text_region = 0
        indexer_textlines = 0
        # FIXME: non recursive, use OCR-D PAGE generateDS API. Or use an existing tool for this purpose altogether
        for nn in root1.iter(region_tags):
            for child_textregion in nn:
                if child_textregion.tag.endswith("TextLine"):
                    for child_textlines in child_textregion:
                        if child_textlines.tag.endswith("Coords"):
                            cropped_lines_region_indexer.append(indexer_text_region)
                            p_h = child_textlines.attrib['points'].split(' ')
                            textline_coords = np.array([[int(x.split(',')[0]), int(x.split(',')[1])] for x in p_h])

                            x, y, w, h = cv2.boundingRect(textline_coords)

                            total_bb_coordinates.append([x, y, w, h])

                            img_poly_on_img = np.copy(img)

                            mask_poly = np.zeros(img.shape)
                            mask_poly = cv2.fillPoly(mask_poly, pts=[textline_coords], color=(1, 1, 1))

                            mask_poly = mask_poly[y : y + h, x : x + w, :]
                            img_crop = img_poly_on_img[y : y + h, x : x + w, :]

                            if not do_not_mask_with_textline_contour:
                                img_crop[mask_poly == 0] = 255

                            if img_crop.shape[0] == 0 or img_crop.shape[1] == 0:
                                continue
                        if child_textlines.tag.endswith("TextEquiv"):
                            for cheild_text in child_textlines:
                                if cheild_text.tag.endswith("Unicode"):
                                    textline_text = cheild_text.text
                                    if textline_text:
                                        base_name = os.path.join(
                                            dir_out, file_name + '_line_' + str(indexer_textlines)
                                        )
                                        if pref_of_dataset:
                                            base_name += '_' + pref_of_dataset
                                        if not do_not_mask_with_textline_contour:
                                            base_name += '_masked'

                                        with open(base_name + '.txt', 'w') as text_file:
                                            text_file.write(textline_text)
                                        cv2.imwrite(base_name + '.png', img_crop)
                                    indexer_textlines += 1
