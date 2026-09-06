"""
Machine learning based reading order detection
"""

# pyright: reportCallIssue=false
# pyright: reportUnboundVariable=false
# pyright: reportArgumentType=false

from __future__ import annotations
import logging 
import os
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import statistics

from ocrd_utils import (
    polygon_from_points,
    xywh_from_points,
)

from .eynollah import Eynollah
from .model_zoo import EynollahModelZoo
from .utils.resize import resize_image
from .utils import is_xml_filename

DPI_THRESHOLD = 298
KERNEL = np.ones((5, 5), np.uint8)


class Reorder(Eynollah):
    def __init__(
            self,
            *,
            model_zoo: EynollahModelZoo,
            logger : logging.Logger | None = None,
            device: str = '',
            model_based: bool = True,
            # also expose these on CLI?
            ignore_page_extraction: bool = False,
            right2left : bool = False,
            input_binary: bool = False,
            num_col_upper: int = 0,
            num_col_lower: int = 0,
    ):
        self.logger = logger or logging.getLogger('eynollah.reorder')
        self.model_based = model_based
        self.ignore_page_extraction = ignore_page_extraction
        self.tables = True # for find_num_col
        self.right2left = right2left
        self.input_binary = input_binary
        self.num_col_upper = num_col_upper
        self.num_col_lower = num_col_lower
        self.model_zoo = model_zoo
        self.setup_models(device=device)

    def setup_models(self, device=''):
        if self.model_based:
            loadable = ['reading_order']
        else:
            loadable = ['page', 'col_classifier']
            if self.input_binary:
                loadable.append('binarization')
        self.model_zoo.load_models(*loadable, device=device)
        for model in loadable:
            self.logger.debug("model %s has input shape %s", model,
                              self.model_zoo.get(model).input_shape)

    def read_xml(self,
                 xml_file: str,
                 label_text=1,
                 label_head=2,
                 label_imgs=5,
                 label_seps=6,
                 label_marg=8,
                 label_drop=4,
                 label_tabs=10,
    ):
        tree1 = ET.parse(xml_file, parser=ET.XMLParser(encoding='utf-8'))
        root1 = tree1.getroot()
        alltags=[elem.tag for elem in root1.iter()]
        link=alltags[0].split('}')[0]+'}'

        index_tot_regions = []
        tot_region_ref = []

        page = root1.find(link+'Page')
        height = int(page.get('imageHeight', 0))
        width = int(page.get('imageWidth', 0))
        skew = -float(page.get('orientation', 0))
        img_filename = page.get('imageFilename', '')

        for jj in root1.iter(link+'RegionRefIndexed'):
            index_tot_regions.append(jj.attrib['index'])
            tot_region_ref.append(jj.attrib['regionRef'])
            
        bb_coord_printspace = None
        if (link+'PrintSpace' in alltags or
            link+'Border' in alltags):
            tag_printspace = next(x for x in alltags
                                  if x.endswith(('Border', 'PrintSpace')))
            nn = page.find(tag_printspace)
            coords = nn.find(link + 'Coords')
            if points := coords.attrib.get('points'):
                xywh = xywh_from_points(points)
                bb_coord_printspace = [xywh['x'],
                                       xywh['y'],
                                       xywh['w'],
                                       xywh['h']]

        seps_cont = []
        imgs_cont = []
        tabs_cont = []
        text_para_cont = []
        text_para_ids = []
        text_drop_cont = []
        text_drop_ids = []
        text_head_cont = []
        text_head_ids = []
        text_marg_cont = []
        text_marg_ids = []

        for nn in root1.iter():
            if not nn.tag.endswith('Region'):
                continue
            if (coords := nn.find(link + 'Coords')) is None:
                continue
            if (points := coords.attrib.get('points')) is None:
                continue
            if (id_ := nn.get('id')) is None:
                continue
            poly = polygon_from_points(points)
            cont = np.array(poly, dtype=int)[:, np.newaxis]
            if nn.tag.endswith('}TextRegion'):
                type_ = nn.get('type', '')
                if type_ == 'drop-capital':
                    text_drop_cont.append(cont)
                    text_drop_ids.append(id_)
                elif type_ == 'heading':
                    text_head_cont.append(cont)
                    text_head_ids.append(id_)
                elif type_ == 'header':
                    # FIXME: do not keep that mapping
                    text_head_cont.append(cont)
                    text_head_ids.append(id_)
                elif type_ == 'marginalia':
                    text_marg_cont.append(cont)
                    text_marg_ids.append(id_)
                else:
                    text_para_cont.append(cont)
                    text_para_ids.append(id_)

            elif nn.tag.endswith('}TableRegion'):
                tabs_cont.append(cont)

            elif nn.tag.endswith('}GraphicRegion') or nn.tag.endswith('}ImageRegion'):
                imgs_cont.append(cont)

            elif nn.tag.endswith('}SeparatorRegion'):
                seps_cont.append(cont)

        img = np.zeros((height, width), dtype=np.uint8)
        img = cv2.fillPoly(img, pts=text_para_cont, color=label_text)
        img = cv2.fillPoly(img, pts=text_head_cont, color=label_head)
        img = cv2.fillPoly(img, pts=text_marg_cont, color=label_marg)
        img = cv2.fillPoly(img, pts=text_drop_cont, color=label_drop)
        img = cv2.fillPoly(img, pts=tabs_cont, color=label_tabs)
        img = cv2.fillPoly(img, pts=imgs_cont, color=label_imgs)
        img = cv2.fillPoly(img, pts=seps_cont, color=label_seps)

        return (tree1, root1,
                bb_coord_printspace,
                text_para_ids, text_head_ids, text_drop_ids,
                text_para_cont, text_head_cont, text_drop_cont,
                tot_region_ref,
                width, height, skew, img_filename,
                index_tot_regions,
                img)

    def run(self,
            overwrite: bool = False,
            xml_filename: str | None = None,
            dir_in: str | None = None,
            dir_imgs: str | None = None,
            dir_out: str | None = None,
    ):
        """
        Get image and scales, then extract the page of scanned image
        """
        self.logger.debug("enter run")

        if dir_in:
            t0_tot = time.time()
            ls_xmls  = [os.path.join(dir_in, xml_filename)
                        for xml_filename in filter(is_xml_filename,
                                                   os.listdir(dir_in))]
        elif xml_filename:
            ls_xmls = [xml_filename]
        else:
            raise ValueError("run requires either a single image filename or a directory")

        for filename in ls_xmls:
            self.run_single(filename,
                            dir_out=dir_out,
                            dir_imgs=dir_imgs,
                            overwrite=overwrite)

        if dir_in:
            self.logger.info("All jobs done in %.1fs", time.time() - t0_tot)

    def run_single(self,
                   xml_filename: str,
                   dir_imgs: str | None = None,
                   dir_out: str | None = None,
                   overwrite: bool = False,
                   label_imgs=5,
                   label_seps=6,
    ) -> None:
        self.logger.info(xml_filename)
        t0 = time.time()

        file_name = Path(xml_filename).stem
        (tree_xml, root_xml,
         _, # FIXME: crop img_poly and contours (bb_coord_printspace)
         para_ids, head_ids, drop_ids,
         para_cont, head_cont, drop_cont,
         _, # FIXME: do not ignore existing RO (tot_region_ref)
         width, height, skew, img_filename,
         _, # FIXME: do not ignore existing RO (index_tot_regions)
         region_labels) = self.read_xml(xml_filename)

        all_text_ids = np.array(para_ids + head_ids + drop_ids)

        self.logger.debug("ordering %d paragraphs, %d headings and %d drop-capitals",
                          len(para_ids), len(head_ids), len(drop_ids))
        if self.model_based:
            order_text = self.run_order_of_regions_with_model(
                para_cont,
                head_cont,
                drop_cont,
                region_labels)
        else:
            if img_filename and (
                    os.path.exists(img_path := img_filename) or
                    os.path.exists(img_path := os.path.join('..', img_filename)) or
                    dir_imgs and
                    os.path.exists(img_path := os.path.join(dir_imgs, img_filename)) or
                    dir_imgs and
                    os.path.exists(img_path := os.path.join(dir_imgs, os.path.basename(img_filename)))):
                img_filename = img_path
            else:
                xml_basename = Path(xml_filename).with_suffix('')
                def try_suffixes(basename, suffixes):
                    for suf in suffixes:
                        # ruff: ignore[SIM114]
                        if (filename := basename.with_suffix(suf)).exists():
                            yield filename
                        elif dir_imgs and (filename := (dir_imgs / basename).with_suffix(suf)).exists():
                            yield filename
                img_filename = next(try_suffixes(xml_basename,
                                                 ['.tif', '.TIF',
                                                  '.jpg', '.JPG',
                                                  '.png', '.PNG',
                                                  '.jpeg', '.gif']))
            # load and analyse image
            image = self.cache_images(image_filename=img_filename)
            _, num_col, _ = self.resize_and_enhance_image_with_column_classifier(image)
            if image['img_res'].shape[:2] != region_labels.shape:
                # image was resized by col-classifier
                # so bring label map to same size
                # (order rules have similar expectations)
                region_labels = resize_image(region_labels, *image['img_res'].shape[:2])
                scale_factor = np.array([[image['scale_x'], image['scale_y']]])
                para_cont = [(cont * scale_factor).astype(int) for cont in para_cont]
                head_cont = [(cont * scale_factor).astype(int) for cont in head_cont]
                drop_cont = [(cont * scale_factor).astype(int) for cont in drop_cont]

            # in Eynollah: regions_without_separators
            nonsep_labels = np.copy(region_labels)
            nonsep_labels[label_seps] = 0
            nonsep_labels[label_imgs] = 0

            # deskew
            if np.abs(skew) >= 0.13: # in Eynollah: SLOPE_THRESHOLD
                _, region_labels, nonsep_labels = self.get_deskewed_masks(
                    skew, np.zeros((1, 1)), region_labels, nonsep_labels)
                # also deskew contours
                # (directly instead of match_deskewed_contours)
                # rotate_image() does not enlarge canvas,
                # so our calculation must compensate
                h_o, w_o = image['img_res'].shape[:2]
                M = cv2.getRotationMatrix2D((0.5 * w_o, 0.5 * h_o), -skew, 1.0)[:2, :2]
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                off = np.array([[0.5 * (w_o * cos + h_o * sin - w_o),
                                 0.5 * (w_o * sin + h_o * cos - h_o)]],
                               dtype=int)
                if skew > 0:
                    off[0, 1] = -off[0, 1]
                else:
                    off[0, 0] = -off[0, 0]
                para_cont = [np.dot(cont, M).astype(int) - off for cont in para_cont]
                head_cont = [np.dot(cont, M).astype(int) - off for cont in head_cont]
                drop_cont = [np.dot(cont, M).astype(int) - off for cont in drop_cont]

            order_text = self.run_order_of_regions_heuristic(
                para_cont,
                head_cont,
                drop_cont,
                region_labels,
                nonsep_labels,
                num_col,
                False) # in Eynollah: erosion_hurts

        all_text_ids = all_text_ids[order_text]

        alltags=[elem.tag for elem in root_xml.iter()]

        link=alltags[0].split('}')[0]+'}'
        ET.register_namespace("", link[1:-1])

        page = root_xml.find(link+'Page')
        ro_old = page.find(link+'ReadingOrder')
        if ro_old is not None:
            page.remove(ro_old)

        ro_new = ET.Element('ReadingOrder')
        ro_group = ET.SubElement(ro_new, 'OrderedGroup')
        ro_group.set('id', "ro357564684568544579089")

        for index, id_text in enumerate(all_text_ids):
            ro_ref = ET.SubElement(ro_group, 'RegionRefIndexed')
            ro_ref.set('regionRef', id_text)
            ro_ref.set('index', str(index))

        pos = len(page.findall(link+'AlternativeImage') +
                  page.findall(link+'Border') +
                  page.findall(link+'PrintSpace'))
        page.insert(pos, ro_new)

        output_filename = os.path.join(dir_out or "", file_name + '.xml')
        self.logger.info("output filename: '%s'", output_filename)
        tree_xml.write(output_filename,
                       xml_declaration=True,
                       method='xml',
                       encoding="utf-8",
                       default_namespace=None)
        self.logger.info("Job done in %.1fs", time.time() - t0)
            
