# FIXME: fix all of those...
# pyright: reportOptionalSubscript=false

from __future__ import annotations
import logging
import logging.handlers
from pathlib import Path
from itertools import groupby
import os
import gc
import math
import time
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
from cv2.typing import MatLike
from xml.etree import ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np
from ocrd_utils import polygon_from_points, xywh_from_polygon


from .eynollah import Eynollah
from .model_zoo import EynollahModelZoo
from .utils import (
    is_image_filename,
    batched,
    pairwise,
)
from .utils.font import get_font
from .utils.xml import etree_namespace_for_element_tag
from .utils.resize import resize_image
from .utils.tiling import do_prediction
from .utils.utils_ocr import (
    break_curved_line_into_small_pieces_and_then_merge,
    fit_text_single_line,
    get_contours_and_bounding_boxes,
    get_orientation_moments,
    preprocess_and_resize_image_for_ocrcnn_model,
    return_textlines_split_if_needed,
    rotate_image_with_padding,
)


_instance = None
def _set_instance(instance):
    global _instance
    _instance = instance
def _run_single(*args, **kwargs):
    logq = kwargs.pop('logq')
    # replace all inherited handlers with queue handler
    logging.root.handlers.clear()
    _instance.logger.parent.handlers.clear()
    handler = logging.handlers.QueueHandler(logq)
    logging.root.addHandler(handler)
    return _instance.run_single(*args, **kwargs)


# TODO: refine typing
@dataclass
class EynollahOcrResult:
    extracted_texts_merged: list[str]
    extracted_confs_merged: list[float]
    cropped_lines_region_indexer: list[int]
    total_bb_coordinates: list[tuple[int, int, int, int]]

class Eynollah_ocr(Eynollah):
    def __init__(
        self,
        *,
        model_zoo: EynollahModelZoo,
        tr_ocr: bool = False,
        batch_size: int = 0,
        do_not_mask_with_textline_contour: bool = False,
        min_conf_value_of_textline_text: float = 0.3,
        logger: logging.Logger | None = None,
        device: str = '',
    ):
        self.tr_ocr = tr_ocr
        # masking for OCR and GT generation, relevant for skewed lines and bounding boxes
        self.do_not_mask_with_textline_contour = do_not_mask_with_textline_contour
        self.logger = logger if logger else logging.getLogger('eynollah.ocr')
        
        self.min_conf_value_of_textline_text = min_conf_value_of_textline_text
        self.b_s = batch_size or (2 if tr_ocr else 64)

        self.model_zoo = model_zoo
        self.setup_models(device=device)

    def setup_models(self, device=''):
        if self.tr_ocr:
            self.model_zoo.load_models(('ocr', 'tr'),
                                       device=device)
        else:
            self.model_zoo.load_models('ocr',
                                       'binarization',
                                       device=device)

    @property
    def device(self):
        return self.model_zoo.get('ocr').device

    def run_trocr(
        self,
        *,
        img: MatLike,
        page_tree: ET.ElementTree,
        page_ns,
    ) -> EynollahOcrResult:

        total_bb_coordinates = []
        cropped_lines = []
        cropped_lines_region_indexer = []
        cropped_lines_meging_indexing = []

        for n_region, region in enumerate(page_tree.getroot().iter('{%s}TextRegion' % page_ns)):
            for n_line, line in enumerate(region.iter('{%s}TextLine' % page_ns)):
                cropped_lines_region_indexer.append(n_region)

                coords = line.find('{%s}Coords' % page_ns)
                if coords is None:
                    self.logger.warning("region '%s' line '%s' has no Coords", region.attrib['id'], line.attrib['id'])
                    continue
                poly = np.array(polygon_from_points(coords.attrib['points'])).astype(int)
                cont = poly[:, np.newaxis]
                xywh = xywh_from_polygon(poly)
                x, y, w, h = xywh['x'], xywh['y'], xywh['w'], xywh['h']

                total_bb_coordinates.append((x, y, w, h))

                img_crop = img[y: y + h, x: x + w]
                if not self.do_not_mask_with_textline_contour:
                    mask_poly = np.zeros(img_crop.shape[:2], dtype=np.uint8)
                    mask_poly = cv2.fillPoly(mask_poly, pts=[cont - [x, y]], color=1)
                    img_crop[mask_poly == 0] = 255 # FIXME: or median color?

                if h > 0.1 * w:
                    cropped_lines.append(img_crop)
                    cropped_lines_meging_indexing.append(0)
                else:
                    splited_images, _ = return_textlines_split_if_needed(img_crop, None)
                    if splited_images:
                        cropped_lines.append(splited_images[0])
                        cropped_lines.append(splited_images[1])
                        cropped_lines_meging_indexing.append(1)
                        cropped_lines_meging_indexing.append(-1)
                    else:
                        cropped_lines.append(img_crop)
                        cropped_lines_meging_indexing.append(0)

        extracted_texts = []
        extracted_confs = []
        self.logger.debug("processing %d lines for %d regions",
                          len(cropped_lines), len(set(cropped_lines_region_indexer)))
        for imgs in batched(cropped_lines, self.b_s):
            text, conf = self.model_zoo.get('ocr').predict(imgs)
            extracted_confs.extend(conf)
            extracted_texts.extend(text)
        del cropped_lines
        gc.collect()

        extracted_texts_merged = [extracted_texts[ind]
                                  if cropped_lines_meging_indexing[ind] == 0
                                  else extracted_texts[ind] + " " + extracted_texts[ind + 1]
                                  for ind in range(len(cropped_lines_meging_indexing))
                                  if cropped_lines_meging_indexing[ind] >= 0]
        extracted_confs_merged = [extracted_confs[ind]
                                  if cropped_lines_meging_indexing[ind] == 0
                                  else 0.5 * (extracted_confs[ind] + extracted_confs[ind + 1])
                                  for ind in range(len(cropped_lines_meging_indexing))
                                  if cropped_lines_meging_indexing[ind] >= 0]

        return EynollahOcrResult(
            extracted_texts_merged=extracted_texts_merged,
            extracted_confs_merged=extracted_confs_merged,
            cropped_lines_region_indexer=cropped_lines_region_indexer,
            total_bb_coordinates=total_bb_coordinates,
        )
        
    def run_cnn(
        self,
        *,
        img: MatLike,
        img_bin: MatLike | None,
        page_tree: ET.ElementTree,
        page_ns,
    ) -> EynollahOcrResult:
        input_shape, _ = self.model_zoo.get('ocr').input_shape
        _, image_height, image_width, _ = input_shape

        total_bb_coordinates = []
        cropped_lines_rgb = []
        cropped_lines_bin = []
        cropped_lines_ver_index = []
        cropped_lines_region_indexer = []
        cropped_lines_meging_indexing = []

        img_rgb = img # cosmetic
        if img_bin is None:
            # run ad-hoc binarization
            self.logger.info("running binarization for ensemble input")
            img_bin = do_prediction(img, self.model_zoo.get("binarization"),
                                    patches=True,
                                    logger=self.logger,
                                    n_batch_inference=5)
            img_bin = np.repeat(img_bin[:, :, np.newaxis], 3, axis=2)
            img_bin = 255 * (img_bin == 0).astype(np.uint8)

        for n_region, region in enumerate(page_tree.getroot().iter('{%s}TextRegion' % page_ns)):
            type_textregion = region.attrib.get('type', 'paragraph')
            for n_line, line in enumerate(region.iter('{%s}TextLine' % page_ns)):
                cropped_lines_region_indexer.append(n_region)

                coords = line.find('{%s}Coords' % page_ns)
                if coords is None:
                    self.logger.warning("region '%s' line '%s' has no Coords", region.attrib['id'], line.attrib['id'])
                    continue
                poly = np.array(polygon_from_points(coords.attrib['points'])).astype(int)
                cont = poly[:, np.newaxis]
                xywh = xywh_from_polygon(poly)
                x, y, w, h = xywh['x'], xywh['y'], xywh['w'], xywh['h']
                            
                angle_radians = math.atan2(h, w)
                angle_degrees = math.degrees(angle_radians)
                if type_textregion=='drop-capital':
                    angle_degrees = 0

                total_bb_coordinates.append((x, y, w, h))
                            
                w_scaled = w * image_height / float(h)

                img_crop_rgb = img_rgb[y: y + h, x: x + w]
                img_crop_bin = img_bin[y: y + h, x: x + w]

                mask_poly = np.zeros(img_crop_rgb.shape[:2], dtype=np.uint8)
                mask_poly = cv2.fillPoly(mask_poly, pts=[cont - [x, y]], color=1)
                            
                if angle_degrees > 3:
                    better_des_slope = get_orientation_moments(cont)
                    img_crop_rgb = rotate_image_with_padding(img_crop_rgb, better_des_slope)
                    img_crop_bin = rotate_image_with_padding(img_crop_bin, better_des_slope)
                    mask_poly = rotate_image_with_padding(mask_poly, better_des_slope)
                    # get new bounding box
                    x_n, y_n, w_n, h_n = get_contours_and_bounding_boxes(mask_poly)
                    img_crop_rgb = img_crop_rgb[y_n: y_n + h_n, x_n: x_n + w_n]
                    img_crop_bin = img_crop_bin[y_n: y_n + h_n, x_n: x_n + w_n]
                    mask_poly = mask_poly[y_n: y_n + h_n, x_n: x_n + w_n]
                else:
                    better_des_slope = 0

                if not self.do_not_mask_with_textline_contour:
                    img_crop_rgb[mask_poly == 0] = 255 # FIXME: or median color?
                    img_crop_bin[mask_poly == 0] = 255

                if (type_textregion !='drop-capital' and
                    mask_poly.sum() < 0.50 * mask_poly.size and
                    w_scaled > 90):

                    img_crop_rgb, img_crop_bin = \
                        break_curved_line_into_small_pieces_and_then_merge(
                            img_crop_rgb, img_crop_bin, mask_poly)

                if w_scaled < 750:#1.5*image_width:
                    img_crop_split_rgb = img_crop_split_bin = None
                else:
                    img_crop_split_rgb, img_crop_split_bin = return_textlines_split_if_needed(
                        img_crop_rgb, img_crop_bin)
                if img_crop_split_rgb:
                    cropped_lines_rgb.extend(img_crop_split_rgb)
                    cropped_lines_bin.extend(img_crop_split_bin)
                    if abs(better_des_slope) > 45:
                        cropped_lines_ver_index.append(1)
                        cropped_lines_ver_index.append(1)
                    else:
                        cropped_lines_ver_index.append(0)
                        cropped_lines_ver_index.append(0)
                    cropped_lines_meging_indexing.append(1)
                    cropped_lines_meging_indexing.append(-1)
                else:
                    cropped_lines_rgb.append(img_crop_rgb)
                    cropped_lines_bin.append(img_crop_bin)
                    if abs(better_des_slope) > 45:
                        cropped_lines_ver_index.append(1)
                    else:
                        cropped_lines_ver_index.append(0)
                    cropped_lines_meging_indexing.append(0)

        cropped_lines_rgb = [preprocess_and_resize_image_for_ocrcnn_model(img, image_height, image_width)
                             for img in cropped_lines_rgb]
        cropped_lines_bin = [preprocess_and_resize_image_for_ocrcnn_model(img, image_height, image_width)
                             for img in cropped_lines_bin]

        extracted_texts = []
        extracted_confs = []
        self.logger.debug("processing %d lines for %d regions",
                          len(cropped_lines_rgb), len(set(cropped_lines_region_indexer)))
        cropped_lines = zip(cropped_lines_rgb, cropped_lines_bin, cropped_lines_ver_index)
        for batch in batched(cropped_lines, self.b_s):
            imgs_rgb, imgs_bin, ver_index = zip(*batch)
            ver_index = np.array(ver_index)
            imgs_rgb = np.stack(imgs_rgb)
            imgs_bin = np.stack(imgs_bin)
            if ver_index.any():
                imgs_rgb = np.append(imgs_rgb, imgs_rgb[ver_index > 0, ::-1, ::-1], axis=0)
                imgs_bin = np.append(imgs_bin, imgs_bin[ver_index > 0, ::-1, ::-1], axis=0)

            # inference model now yields (char-bytes, line-prob) instead of vocidx-softmax
            # (so ctc_decode and inverse StringLookup are included)
            # also, the model now expects a secondary binary input image
            preds, probs = self.model_zoo.get('ocr').predict((imgs_rgb, imgs_bin), verbose=0)
            
            if ver_index.any():
                preds, preds_ver = np.split(preds, [-np.count_nonzero(ver_index)], axis=0)
                probs, probs_ver = np.split(probs, [-np.count_nonzero(ver_index)], axis=0)
                flipped_ver_is_better = np.flatnonzero(probs_ver > probs[ver_index > 0])
                if len(flipped_ver_is_better):
                    self.logger.info("%d skewed lines perform better when flipped", len(flipped_ver_is_better))
                    preds[ver_index > 0][flipped_ver_is_better] = preds_ver[flipped_ver_is_better]
                    probs[ver_index > 0][flipped_ver_is_better] = probs_ver[flipped_ver_is_better]

            def nooov(x):
                if x == b'[UNK]':
                    return b''
                return x
            for pred, prob in zip(preds, probs):
                text = b''.join(map(nooov, pred.tolist())).decode('utf-8')
                extracted_texts.append(text)
                extracted_confs.append(prob)
        del cropped_lines_rgb
        del cropped_lines_bin
        gc.collect()
        
        extracted_texts_merged = [extracted_texts[ind]
                                  if cropped_lines_meging_indexing[ind] == 0
                                  else extracted_texts[ind] + " " + extracted_texts[ind + 1]
                                  for ind in range(len(cropped_lines_meging_indexing))
                                  if cropped_lines_meging_indexing[ind] >= 0]
        extracted_confs_merged = [extracted_confs[ind]
                                  if cropped_lines_meging_indexing[ind] == 0
                                  else 0.5 * (extracted_confs[ind] + extracted_confs[ind + 1])
                                  for ind in range(len(cropped_lines_meging_indexing))
                                  if cropped_lines_meging_indexing[ind] >= 0]

        return EynollahOcrResult(
            extracted_texts_merged=extracted_texts_merged,
            extracted_confs_merged=extracted_confs_merged,
            cropped_lines_region_indexer=cropped_lines_region_indexer,
            total_bb_coordinates=total_bb_coordinates,
        )
        
    def write_ocr(
        self,
        *,
        result: EynollahOcrResult,
        page_tree: ET.ElementTree,
        out_file_ocr,
        page_ns,
        img,
        out_image_with_text,
    ):
        cropped_lines_region_indexer = result.cropped_lines_region_indexer
        total_bb_coordinates = result.total_bb_coordinates
        extracted_texts_merged = result.extracted_texts_merged
        extracted_confs_merged = result.extracted_confs_merged

        if out_image_with_text:
            image_text = Image.new("RGB", (img.shape[1], img.shape[0]), "white")
            draw = ImageDraw.Draw(image_text)
            font = get_font(font_size=40)
            
            for indexer_text, (x_bb, y_bb, w_bb, h_bb) in enumerate(total_bb_coordinates):
                font = fit_text_single_line(draw, extracted_texts_merged[indexer_text],
                                            font.path, w_bb, int(h_bb*0.4) )
                
                ##draw.rectangle([x_bb, y_bb, x_bb + w_bb, y_bb + h_bb], outline="red", width=2)
                
                text_bbox = draw.textbbox((0, 0), extracted_texts_merged[indexer_text], font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                text_x = x_bb + (w_bb - text_width) // 2  # Center horizontally
                text_y = y_bb + (h_bb - text_height) // 2  # Center vertically

                # Draw the text
                draw.text((text_x, text_y), extracted_texts_merged[indexer_text], fill="black", font=font)
            image_text.save(out_image_with_text)

        cropped_lines_region_indexer = np.array(cropped_lines_region_indexer)
        for n_region, region in enumerate(page_tree.getroot().iter('{%s}TextRegion' % page_ns)):
            lines_indexer = np.flatnonzero(cropped_lines_region_indexer == n_region)
            if not len(lines_indexer):
                continue

            text_region = ""
            next_glue = ""
            for line_idx in lines_indexer:
                if extracted_confs_merged[line_idx] < self.min_conf_value_of_textline_text:
                    continue
                text_line = extracted_texts_merged[line_idx]
                if (text_line.endswith(('⸗', '-', '¬')) and
                    # last line of a region can still be wrapped
                    # around columns or pages
                    line_idx < len(lines_indexer) - 1):
                    text_region += next_glue + text_line[:-1]
                    next_glue = ""
                else:
                    text_region += next_glue + text_line
                    next_glue = " "

            region_textequiv = region.find('{%s}TextEquiv' % page_ns)
            if region_textequiv is None:
                region_textequiv = ET.SubElement(region, 'TextEquiv')
            region_teunicode = region_textequiv.find('{%s}Unicode' % page_ns)
            if region_teunicode is None:
                region_teunicode = ET.SubElement(region_textequiv, 'Unicode')
            region_teunicode.text = text_region

            for n_line, line in enumerate(region.iter('{%s}TextLine' % page_ns)):
                line_textequiv = line.find('{%s}TextEquiv' % page_ns)
                if line_textequiv is None:
                    line_textequiv = ET.SubElement(line, 'TextEquiv')
                line_teunicode = line_textequiv.find('{%s}Unicode' % page_ns)
                if line_teunicode is None:
                    line_teunicode = ET.SubElement(line_textequiv, 'Unicode')

                line_idx = lines_indexer[n_line]
                if extracted_confs_merged[line_idx] < self.min_conf_value_of_textline_text:
                    line.remove(line_textequiv)
                else:
                    line_textequiv.set('conf', str(round(extracted_confs_merged[line_idx], 2)))
                    line_teunicode.text = extracted_texts_merged[line_idx]
                
        ET.register_namespace("",page_ns)
        self.logger.info("output filename: '%s'", out_file_ocr)
        page_tree.write(out_file_ocr, xml_declaration=True, method='xml', encoding="utf-8", default_namespace=None)

    def run(self,
            *,
            overwrite: bool = False,
            dir_in: str = "",
            dir_in_bin: str = "",
            image_filename: str = "",
            dir_xmls: str,
            dir_out_image_text: str = "",
            dir_out: str,
            num_jobs: int = 0,
            halt_fail: float = 0,
    ):
        """
        Run OCR.

        Args:

            dir_in_bin (str): Prediction with RGB and binarized images for selected pages, should not be the default
        """
        if dir_in:
            t0_tot = time.time()
            ls_imgs = [os.path.join(dir_in, image_filename)
                       for image_filename in filter(is_image_filename,
                                                    os.listdir(dir_in))]
            if dir_in_bin and dir_in_bin == dir_in:
                # try filtering PNGs from rest
                def pathstem(filename):
                    return os.path.splitext(filename)[0]
                def notpng(filenames):
                    for filename in filenames:
                        if not filename.lower().endswith(".png"):
                            return filename
                    return filenames[0]
                ls_imgs = [notpng(files)
                           for _, files in groupby(sorted(ls_imgs),
                                                   key=pathstem)]
            with ProcessPoolExecutor(max_workers=num_jobs or None,
                                     mp_context=mp.get_context('fork'),
                                     initializer=_set_instance,
                                     initargs=(self,)
            ) as exe:
                jobs = {}
                mngr = mp.get_context('fork').Manager()
                n_success = n_fail = 0
                for img_filename in ls_imgs:
                    logq = mngr.Queue()
                    jobs[exe.submit(_run_single, img_filename,
                                    dir_out=dir_out,
                                    dir_xmls=dir_xmls,
                                    dir_in_bin=dir_in_bin,
                                    dir_out_image_text=dir_out_image_text,
                                    overwrite=overwrite,
                                    logq=logq)] = img_filename, logq
                for job in as_completed(list(jobs)):
                    img_filename, logq = jobs[job]
                    loglistener = logging.handlers.QueueListener(
                        logq, *self.logger.handlers,
                        *self.logger.parent.handlers,
                        respect_handler_level=False)
                    try:
                        loglistener.start()
                        job.result()
                        n_success += 1
                    except:
                        self.logger.exception("Job %s failed", img_filename)
                        n_fail += 1
                        if (halt_fail and
                            n_fail >= halt_fail * (len(jobs) if halt_fail < 1 else 1)):
                            self.logger.fatal("terminating after %d failures", n_fail)
                            for job in jobs:
                                job.cancel()
                            break
                    finally:
                        loglistener.stop()
            self.logger.info("%d of %d jobs successful", n_success, len(jobs))
            self.logger.info("All jobs done in %.1fs", time.time() - t0_tot)
        else:
            assert image_filename
            self.run_single(image_filename,
                            dir_xmls=dir_xmls,
                            dir_out=dir_out,
                            dir_in_bin=dir_in_bin,
                            dir_out_image_text=dir_out_image_text,
                            overwrite=overwrite)

    def run_single(self, 
                   img_filename: str,
                   dir_xmls: str,
                   dir_out: str = "",
                   dir_in_bin: str = "",
                   dir_out_image_text: str = "",
                   overwrite: bool = False,
    ):
        file_stem = Path(img_filename).stem
        page_file_in = os.path.join(dir_xmls, file_stem + '.xml')
        out_file_ocr = os.path.join(dir_out, file_stem + '.xml')

        if os.path.exists(out_file_ocr):
            if overwrite:
                self.logger.warning("will overwrite existing output file '%s'", out_file_ocr)
            else:
                self.logger.warning("will skip input for existing output file '%s'", out_file_ocr)
                return
        if not os.path.exists(page_file_in):
            self.logger.error("will skip missing input file '%s'", page_file_in)
            return

        t0 = time.time()

        img = cv2.imread(img_filename)
        self.logger.info(img_filename)
        page_tree = ET.parse(page_file_in, parser = ET.XMLParser(encoding="utf-8"))
        page_ns = etree_namespace_for_element_tag(page_tree.getroot().tag)

        out_image_with_text = None
        if dir_out_image_text:
            out_image_with_text = os.path.join(dir_out_image_text, file_stem + '.png')

        img_bin = None
        if dir_in_bin:
            img_bin = cv2.imread(os.path.join(dir_in_bin, file_stem+'.png'))


        if self.tr_ocr:
            result = self.run_trocr(
                img=img,
                page_tree=page_tree,
                page_ns=page_ns,
            )
        else:
            result = self.run_cnn( 
                img=img,
                page_tree=page_tree,
                page_ns=page_ns,
                img_bin=img_bin,
            )

        self.write_ocr(
            result=result,
            img=img,
            page_tree=page_tree,
            page_ns=page_ns,
            out_file_ocr=out_file_ocr,
            out_image_with_text=out_image_with_text,
        )
        self.logger.info("Job done in %.1fs", time.time() - t0)
