"""
document layout analysis (segmentation) with output in PAGE-XML
"""
# pylint: disable=no-member,invalid-name,line-too-long,missing-function-docstring,missing-class-docstring,too-many-branches
# pylint: disable=too-many-locals,wrong-import-position,too-many-lines,too-many-statements,chained-comparison,fixme,broad-except,c-extension-no-member
# pylint: disable=too-many-public-methods,too-many-arguments,too-many-instance-attributes,too-many-public-methods,
# pylint: disable=consider-using-enumerate
# FIXME: fix all of those...
# pyright: reportUnnecessaryTypeIgnoreComment=true
# pyright: reportPossiblyUnboundVariable=false
# pyright: reportOperatorIssue=false
# pyright: reportUnboundVariable=false
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalSubscript=false

import logging
import logging.handlers
import sys

from difflib import SequenceMatcher as sq
import math
import os
import time
from typing import Optional
from functools import partial
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from .model_zoo import EynollahModelZoo
from .utils.contour import (
    filter_contours_area_of_image,
    filter_contours_area_of_image_tables,
    find_center_of_contours,
    find_new_features_of_contours,
    find_features_of_contours,
    get_text_region_boxes_by_given_contours,
    get_region_confidences,
    return_contours_of_image,
    return_contours_of_interested_region,
    return_parent_contours,
    dilate_textregion_contours,
    dilate_textline_contours,
    match_deskewed_contours,
    polygon2contour,
    contour2polygon,
    join_polygons,
    make_intersection,
)
from .utils.rotate import rotate_image
from .utils.separate_lines import (
    return_deskew_slop,
    do_work_of_slopes_new_curved,
)
from .utils.marginals import get_marginals
from .utils.resize import resize_image
from .utils.shm import share_ndarray
from .utils import (
    ensure_array,
    is_image_filename,
    isNaN,
    crop_image_inside_box,
    box2rect,
    find_num_col,
    otsu_copy_binary,
    seg_mask_label,
    fill_bb_of_drop_capitals,
    split_textregion_main_vs_head,
    small_textlines_to_parent_adherence2,
    order_of_regions,
    find_number_of_columns_in_document,
    return_boxes_of_images_by_order_of_reading_new
)
from .utils.pil_cv2 import pil2cv
from .plot import EynollahPlotter
from .writer import EynollahXmlWriter

MIN_AREA_REGION = 0.000001
SLOPE_THRESHOLD = 0.13
RATIO_OF_TWO_MODEL_THRESHOLD = 95.50 #98.45:
DPI_THRESHOLD = 298
MAX_SLOPE = 999
KERNEL = np.ones((5, 5), np.uint8)


_instance = None
def _set_instance(instance):
    global _instance
    _instance = instance
def _run_single(*args, **kwargs):
    logq = kwargs.pop('logq')
    # replace all inherited handlers with queue handler
    logging.root.handlers.clear()
    _instance.logger.handlers.clear()
    handler = logging.handlers.QueueHandler(logq)
    logging.root.addHandler(handler)
    return _instance.run_single(*args, **kwargs)

class Eynollah:
    def __init__(
        self,
        *,
        model_zoo: EynollahModelZoo,
        device: str = '',
        enable_plotting : bool = False,
        allow_enhancement : bool = False,
        curved_line : bool = False,
        full_layout : bool = False,
        tables : bool = False,
        right2left : bool = False,
        input_binary : bool = False,
        allow_scaling : bool = False,
        headers_off : bool = False,
        ignore_page_extraction : bool = False,
        reading_order_machine_based : bool = False,
        num_col_upper : int = 0,
        num_col_lower : int = 0,
        threshold_art_class_layout: float = 0.1,
        threshold_art_class_textline: float = 0.1,
        skip_layout_and_reading_order : bool = False,
        num_jobs : int = 0,
        logger : Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger('eynollah')
        self.model_zoo = model_zoo
        self.plotter = None

        self.reading_order_machine_based = reading_order_machine_based
        self.enable_plotting = enable_plotting
        self.allow_enhancement = allow_enhancement
        self.curved_line = curved_line
        self.full_layout = full_layout
        self.tables = tables
        self.right2left = right2left
        # --input-binary sensible if image is very dark, if layout is not working.
        self.input_binary = input_binary
        self.allow_scaling = allow_scaling
        self.headers_off = headers_off
        self.ignore_page_extraction = ignore_page_extraction
        self.skip_layout_and_reading_order = skip_layout_and_reading_order
        self.num_col_upper = int(num_col_upper)
        self.num_col_lower = int(num_col_lower)
        self.threshold_art_class_layout = float(threshold_art_class_layout)
        self.threshold_art_class_textline = float(threshold_art_class_textline)

        t_start = time.time()

        self.logger.info("Loading models...")
        self.setup_models(device=device)
        self.logger.info(f"Model initialization complete ({time.time() - t_start:.1f}s)")

    def setup_models(self, device=''):

        # load models, depending on modes
        # (note: loading too many models can cause OOM on GPU/CUDA,
        #  thus, we try set up the minimal configuration for the current mode)
        # autosized variants: _resized or _patched (which one may depend on num_cols)
        # (but _resized for full page images is too slow - better resize on CPU in numpy)
        loadable = [
            "col_classifier",
            #"enhancement", # todo: enhancement_patched
            "page",
            #"region"
        ]
        if self.input_binary:
            loadable.append("binarization") # todo: binarization_patched
        loadable.append("textline") # textline_patched
        loadable.append("region_1_2")
        #loadable.append("region_1_2_patched")
        if self.full_layout:
            loadable.append("region_fl_np")
            #loadable.append("region_fl_patched")
        if self.reading_order_machine_based:
            loadable.append("reading_order") # todo: reading_order_patched
        if self.tables:
            loadable.append("table")

        self.model_zoo.load_models(*loadable, device=device)
        for model in loadable:
            # retrieve and cache output shapes
            if model.endswith(('_resized', '_patched')):
                # autosized models do not have a predefined input_shape
                # (and don't need one)
                continue
            self.logger.debug("model %s has input shape %s", model,
                              self.model_zoo.get(model).input_shape)

    def __del__(self):
        if model_zoo := getattr(self, 'model_zoo', None):
            if shutdown := getattr(model_zoo, 'shutdown', None):
                shutdown()
        del self.model_zoo

    def cache_images(self, image_filename=None, image_pil=None, dpi=None):
        ret = {}
        if image_pil:
            ret['img'] = pil2cv(image_pil)
        elif image_filename:
            ret['img'] = cv2.imread(image_filename)
        if image_filename:
            ret['name'] = Path(image_filename).stem
        else:
            ret['name'] = "image"
        ret['dpi'] = dpi or 100
        ret['img_grayscale'] = cv2.cvtColor(ret['img'], cv2.COLOR_BGR2GRAY)
        for prefix in ('',  '_grayscale'):
            ret[f'img{prefix}_uint8'] = ret[f'img{prefix}'].astype(np.uint8)
        return ret

    def imread(self, image: dict, grayscale=False, binary=False, uint8=True):
        key = 'img'
        if grayscale:
            key += '_grayscale'
        elif binary:
            key += '_bin'
        if uint8:
            key += '_uint8'
        return image[key].copy()

    def calculate_width_height_by_columns(self, img, num_col, conf_col, width_early):
        self.logger.debug("enter calculate_width_height_by_columns")
        if num_col == 1 and width_early < 1100:
            img_w_new = 2000
        elif num_col == 1 and width_early >= 2500:
            img_w_new = 2000
        elif num_col == 1:
            img_w_new = width_early
        elif num_col == 2 and width_early < 2000:
            img_w_new = 2400
        elif num_col == 2 and width_early >= 3500:
            img_w_new = 2400
        elif num_col == 2:
            img_w_new = width_early
        elif num_col == 3 and width_early < 2000:
            img_w_new = 3000
        elif num_col == 3 and width_early >= 4000:
            img_w_new = 3000
        elif num_col == 3:
            img_w_new = width_early
        elif num_col == 4 and width_early < 2500:
            img_w_new = 4000
        elif num_col == 4 and width_early >= 5000:
            img_w_new = 4000
        elif num_col == 4:
            img_w_new = width_early
        elif num_col == 5 and width_early < 3700:
            img_w_new = 5000
        elif num_col == 5 and width_early >= 7000:
            img_w_new = 5000
        elif num_col == 5:
            img_w_new = width_early
        elif num_col == 6 and width_early < 4500:
            img_w_new = 6500  # 5400
        else:
            img_w_new = width_early
        img_h_new = img_w_new * img.shape[0] // img.shape[1]

        if conf_col < 0.9 and img_w_new < width_early:
            # don't downsample if unconfident
            img_new = np.copy(img)
            img_is_resized = False
        #elif conf_col < 0.8 and img_h_new >= 8000:
        elif img_h_new >= 8000:
            # don't upsample if too large
            img_new = np.copy(img)
            img_is_resized = False
        else:
            img_new = resize_image(img, img_h_new, img_w_new)
            img_is_resized = True

        return img_new, img_is_resized

    def calculate_width_height_by_columns_1_2(self, img, num_col, conf_col, width_early):
        self.logger.debug("enter calculate_width_height_by_columns")
        if num_col == 1:
            img_w_new = 1000
        else:
            img_w_new = 1300
        img_h_new = img_w_new * img.shape[0] // img.shape[1]

        if conf_col < 0.9 and img_w_new < width_early:
            # don't downsample if unconfident
            img_new = np.copy(img)
            img_is_resized = False
        #elif conf_col < 0.8 and img_h_new >= 8000:
        elif img_h_new >= 8000:
            # don't upsample if too large
            img_new = np.copy(img)
            img_is_resized = False
        else:
            img_new = resize_image(img, img_h_new, img_w_new)
            img_is_resized = True

        return img_new, img_is_resized

    def resize_image_with_column_classifier(self, image):
        self.logger.debug("enter resize_image_with_column_classifier")
        img = self.imread(image, binary=self.input_binary)

        width_early = img.shape[1]
        _, page_coord = self.early_page_for_num_of_column_classification(image)

        if self.input_binary:
            img_in = img
        else:
            img_1ch = self.imread(image, grayscale=True, uint8=False)
            img_1ch = img_1ch[page_coord[0]: page_coord[1],
                              page_coord[2]: page_coord[3]]
            img_in = np.repeat(img_1ch[:, :, np.newaxis], 3, axis=2)
        img_in = img_in / 255.0
        img_in = cv2.resize(img_in, (448, 448), interpolation=cv2.INTER_NEAREST).astype(np.float16)

        label_p_pred = self.model_zoo.get("col_classifier").predict(img_in[np.newaxis], verbose=0)[0]
        num_col = np.argmax(label_p_pred) + 1
        conf_col = np.max(label_p_pred)

        self.logger.info("Found %s columns (%s)", num_col, np.around(label_p_pred, decimals=5))
        if num_col in (1, 2):
            fun = self.calculate_width_height_by_columns_1_2
        else:
            self.calculate_width_height_by_columns
        img_new, _ = fun(img, num_col, conf_col, width_early)

        if img_new.shape[1] > img.shape[1]:
            img_new = self.do_prediction(True, img_new, self.model_zoo.get("enhancement"),
                                         marginal_of_patch_percent=0,
                                         n_batch_inference=3,
                                         is_enhancement=True)
            self.logger.info("Enhancement applied")

        image['img_res'] = img_new
        image['scale_y'] = 1.0 * img_new.shape[0] / img.shape[0]
        image['scale_x'] = 1.0 * img_new.shape[1] / img.shape[1]
        return

    def resize_and_enhance_image_with_column_classifier(self, image):
        self.logger.debug("enter resize_and_enhance_image_with_column_classifier")
        dpi = image['dpi']
        img = self.imread(image)
        self.logger.info("Detected %s DPI", dpi)
        if self.input_binary:
            prediction_bin = self.do_prediction(True, img, self.model_zoo.get("binarization"), n_batch_inference=5)
            prediction_bin = 255 * (prediction_bin == 0)
            prediction_bin = np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            image['img_bin_uint8'] = prediction_bin
            img = np.copy(prediction_bin)
        else:
            image['img_bin_uint8'] = None

        width_early = img.shape[1]
        t1 = time.time()
        _, page_coord = self.early_page_for_num_of_column_classification(image)

        label_p_pred = np.ones(6)
        conf_col = 1.0
        if self.num_col_upper and not self.num_col_lower:
            num_col = self.num_col_upper
        elif self.num_col_lower and not self.num_col_upper:
            num_col = self.num_col_lower
        elif (not self.num_col_upper and not self.num_col_lower or
              self.num_col_upper != self.num_col_lower):
            if self.input_binary:
                img_in = img
            else:
                img_1ch = self.imread(image, grayscale=True)
                img_1ch = img_1ch[page_coord[0]: page_coord[1],
                                  page_coord[2]: page_coord[3]]
                img_in = np.repeat(img_1ch[:, :, np.newaxis], 3, axis=2)
            img_in = img_in / 255.0
            img_in = cv2.resize(img_in, (448, 448), interpolation=cv2.INTER_NEAREST).astype(np.float16)

            label_p_pred = self.model_zoo.get("col_classifier").predict(img_in[np.newaxis], verbose=0)[0]
            num_col = np.argmax(label_p_pred) + 1
            conf_col = np.max(label_p_pred)

            if self.num_col_upper and self.num_col_upper < num_col:
                num_col = self.num_col_upper
                conf_col = 1.0
            if self.num_col_lower and self.num_col_lower > num_col:
                num_col = self.num_col_lower
                conf_col = 1.0
        else:
            num_col = self.num_col_upper
            conf_col = 1.0

        self.logger.info("Found %d columns (%s)", num_col, np.around(label_p_pred, decimals=5))
        if num_col in (1,2):
            img_res, is_image_resized = self.calculate_width_height_by_columns_1_2(
                img, num_col, conf_col, width_early)
            is_image_enhanced = True
        elif dpi < DPI_THRESHOLD:
            img_res, is_image_resized = self.calculate_width_height_by_columns(
                img, num_col, conf_col, width_early)
            is_image_enhanced = True
        else:
            img_res = np.copy(img)
            is_image_resized = True # FIXME: not true actually, but branch is dead anyway
            is_image_enhanced = False

        self.logger.debug("exit resize_and_enhance_image_with_column_classifier")
        image['img_res'] = img_res
        image['scale_y'] = 1.0 * img_res.shape[0] / img.shape[0]
        image['scale_x'] = 1.0 * img_res.shape[1] / img.shape[1]
        return is_image_enhanced, num_col, is_image_resized

    def do_prediction(
            self, patches, img, model,
            n_batch_inference=1,
            marginal_of_patch_percent=0.1,
            thresholding_for_some_classes=False,
            thresholding_for_heading=False,
            heading_class=2,
            thresholding_for_artificial_class=False,
            threshold_art_class=0.1,
            artificial_class=2,
            is_enhancement=False,
    ):

        self.logger.debug("enter do_prediction (patches=%d)", patches)
        _, img_height_model, img_width_model, _ = model.input_shape
        img_h_page = img.shape[0]
        img_w_page = img.shape[1]

        img = img / 255.
        img = img.astype(np.float16)

        if not patches:
            img = resize_image(img, img_height_model, img_width_model)

            label_p_pred = model.predict(img[np.newaxis], verbose=0)[0]
            if is_enhancement:
                seg = (label_p_pred * 255).astype(np.uint8)
            else:
                seg = np.argmax(label_p_pred, axis=2)

            if thresholding_for_artificial_class:
                seg_mask_label(
                    seg, label_p_pred[:, :, artificial_class] >= threshold_art_class,
                    label=artificial_class,
                    skeletonize=True)

            if thresholding_for_heading:
                seg_mask_label(
                    seg, label_p_pred[:, :, heading_class] >= 0.2,
                    label=heading_class)

            return resize_image(seg, img_h_page, img_w_page).astype(np.uint8)

        if img_h_page < img_height_model:
            img = resize_image(img, img_height_model, img.shape[1])
        if img_w_page < img_width_model:
            img = resize_image(img, img.shape[0], img_width_model)

        self.logger.debug("Patch size: %sx%s", img_height_model, img_width_model)
        margin = int(marginal_of_patch_percent * img_height_model)
        width_mid = img_width_model - 2 * margin
        height_mid = img_height_model - 2 * margin
        img_h = img.shape[0]
        img_w = img.shape[1]
        if is_enhancement:
            prediction = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        else:
            prediction = np.zeros((img_h, img_w), dtype=np.uint8)
        if thresholding_for_artificial_class:
            mask_artificial_class = np.zeros((img_h, img_w), dtype=bool)
        nxf = math.ceil(img_w / float(width_mid))
        nyf = math.ceil(img_h / float(height_mid))

        list_i_s = []
        list_j_s = []
        list_x_u = []
        list_x_d = []
        list_y_u = []
        list_y_d = []

        batch_indexer = 0
        img_patch = np.zeros((n_batch_inference, img_height_model, img_width_model, 3), dtype=np.float16)
        for i in range(nxf):
            for j in range(nyf):
                index_x_d = i * width_mid
                index_x_u = index_x_d + img_width_model
                index_y_d = j * height_mid
                index_y_u = index_y_d + img_height_model
                if index_x_u > img_w:
                    index_x_u = img_w
                    index_x_d = img_w - img_width_model
                if index_y_u > img_h:
                    index_y_u = img_h
                    index_y_d = img_h - img_height_model

                list_i_s.append(i)
                list_j_s.append(j)
                list_x_u.append(index_x_u)
                list_x_d.append(index_x_d)
                list_y_d.append(index_y_d)
                list_y_u.append(index_y_u)

                img_patch[batch_indexer] = img[index_y_d:index_y_u,
                                               index_x_d:index_x_u]
                batch_indexer += 1

                if (batch_indexer == n_batch_inference or
                    # last batch
                    i == nxf - 1 and j == nyf - 1):
                    self.logger.debug("predicting patches on %s", str(img_patch.shape))
                    label_p_pred = model.predict(img_patch, verbose=0)
                    if is_enhancement:
                        seg = (label_p_pred * 255).astype(np.uint8)
                    else:
                        seg = np.argmax(label_p_pred, axis=3)

                    if thresholding_for_some_classes:
                        seg_mask_label(
                            seg, label_p_pred[:,:,:,4] > 0.03,
                            label=4) # 
                        seg_mask_label(
                            seg, label_p_pred[:,:,:,0] > 0.25,
                            label=0) # bg
                        seg_mask_label(
                            seg, label_p_pred[:,:,:,3] > 0.10 & seg == 0,
                            label=3) # line

                    if thresholding_for_artificial_class:
                        seg_art = label_p_pred[:, :, :, artificial_class] >= threshold_art_class

                    indexer_inside_batch = 0
                    for i_batch, j_batch in zip(list_i_s, list_j_s):
                        seg_in = seg[indexer_inside_batch]

                        if thresholding_for_artificial_class:
                            seg_in_art = seg_art[indexer_inside_batch]

                        index_y_u_in = list_y_u[indexer_inside_batch]
                        index_y_d_in = list_y_d[indexer_inside_batch]

                        index_x_u_in = list_x_u[indexer_inside_batch]
                        index_x_d_in = list_x_d[indexer_inside_batch]

                        where = np.index_exp[index_y_d_in:index_y_u_in,
                                             index_x_d_in:index_x_u_in]
                        if (i_batch == 0 and
                            j_batch == 0):
                            inbox = np.index_exp[0:-margin or None,
                                                 0:-margin or None]
                        elif (i_batch == nxf - 1 and
                              j_batch == nyf - 1):
                            inbox = np.index_exp[margin:,
                                                 margin:]
                        elif (i_batch == 0 and
                              j_batch == nyf - 1):
                            inbox = np.index_exp[margin:,
                                                 0:-margin or None]
                        elif (i_batch == nxf - 1 and
                              j_batch == 0):
                            inbox = np.index_exp[0:-margin or None,
                                                 margin:]
                        elif (i_batch == 0 and
                              j_batch != 0 and
                              j_batch != nyf - 1):
                            inbox = np.index_exp[margin:-margin or None,
                                                 0:-margin or None]
                        elif (i_batch == nxf - 1 and
                              j_batch != 0 and
                              j_batch != nyf - 1):
                            inbox = np.index_exp[margin:-margin or None,
                                                 margin:]
                        elif (i_batch != 0 and
                              i_batch != nxf - 1 and
                              j_batch == 0):
                            inbox = np.index_exp[0:-margin or None,
                                                 margin:-margin or None]
                        elif (i_batch != 0 and
                              i_batch != nxf - 1 and
                              j_batch == nyf - 1):
                            inbox = np.index_exp[margin:,
                                                 margin:-margin or None]
                        else:
                            inbox = np.index_exp[margin:-margin or None,
                                                 margin:-margin or None]
                        prediction[where][inbox] = seg_in[inbox]
                        if thresholding_for_artificial_class:
                            mask_artificial_class[where][inbox] = seg_in_art[inbox]

                        indexer_inside_batch += 1


                    list_i_s = []
                    list_j_s = []
                    list_x_u = []
                    list_x_d = []
                    list_y_u = []
                    list_y_d = []

                    batch_indexer = 0
                    img_patch[:] = 0

        if thresholding_for_artificial_class:
            seg_mask_label(prediction, mask_artificial_class,
                           label=artificial_class,
                           only=True,
                           skeletonize=True,
                           dilate=3)

        if img_h != img_h_page or img_w != img_w_page:
            prediction = resize_image(prediction, img_h_page, img_w_page)

        gc.collect()
        return prediction

    def do_prediction_new_concept(
            self, patches, img, model,
            n_batch_inference=1,
            marginal_of_patch_percent=0.1,
            thresholding_for_heading=False,
            heading_class=2,
            thresholding_for_artificial_class=False,
            threshold_art_class=0.1,
            artificial_class=4,
            separator_class=0,
    ):

        self.logger.debug("enter do_prediction_new_concept (patches=%d)", patches)
        _, img_height_model, img_width_model, _ = model.input_shape

        img = img / 255.0
        img = img.astype(np.float16)

        if not patches:
            img_h_page = img.shape[0]
            img_w_page = img.shape[1]
            img = resize_image(img, img_height_model, img_width_model)

            label_p_pred = model.predict(img[np.newaxis], verbose=0)[0]
            seg = np.argmax(label_p_pred, axis=2)

            prediction = resize_image(seg, img_h_page, img_w_page).astype(np.uint8)

            if thresholding_for_artificial_class:
                mask = resize_image(label_p_pred[:, :, artificial_class],
                                    img_h_page, img_w_page) >= threshold_art_class
                seg_mask_label(prediction, mask,
                               label=artificial_class,
                               only=True,
                               skeletonize=True,
                               dilate=3,
                               keep=separator_class)
            if thresholding_for_heading:
                mask = resize_image(label_p_pred[:, :, heading_class],
                                    img_h_page, img_w_page) >= 0.2
                seg_mask_label(prediction, mask,
                               label=heading_class)

            conf = label_p_pred[tuple(np.indices(seg.shape)) + (seg,)]
            conf = resize_image(conf, img_h_page, img_w_page)
            return prediction, conf

        if img.shape[0] < img_height_model:
            img = resize_image(img, img_height_model, img.shape[1])
        if img.shape[1] < img_width_model:
            img = resize_image(img, img.shape[0], img_width_model)

        self.logger.debug("Patch size: %sx%s", img_height_model, img_width_model)
        margin = int(marginal_of_patch_percent * img_height_model)
        width_mid = img_width_model - 2 * margin
        height_mid = img_height_model - 2 * margin
        img_h = img.shape[0]
        img_w = img.shape[1]
        prediction = np.zeros((img_h, img_w), dtype=np.uint8)
        confidence = np.zeros((img_h, img_w))
        if thresholding_for_artificial_class:
            mask_artificial_class = np.zeros((img_h, img_w), dtype=bool)
        nxf = math.ceil(img_w / float(width_mid))
        nyf = math.ceil(img_h / float(height_mid))

        list_i_s = []
        list_j_s = []
        list_x_u = []
        list_x_d = []
        list_y_u = []
        list_y_d = []

        batch_indexer = 0
        img_patch = np.zeros((n_batch_inference, img_height_model, img_width_model, 3), dtype=np.float16)
        for i in range(nxf):
            for j in range(nyf):
                index_x_d = i * width_mid
                index_x_u = index_x_d + img_width_model
                index_y_d = j * height_mid
                index_y_u = index_y_d + img_height_model
                if index_x_u > img_w:
                    index_x_u = img_w
                    index_x_d = img_w - img_width_model
                if index_y_u > img_h:
                    index_y_u = img_h
                    index_y_d = img_h - img_height_model

                list_i_s.append(i)
                list_j_s.append(j)
                list_x_u.append(index_x_u)
                list_x_d.append(index_x_d)
                list_y_d.append(index_y_d)
                list_y_u.append(index_y_u)

                img_patch[batch_indexer] = img[index_y_d:index_y_u,
                                               index_x_d:index_x_u]
                batch_indexer += 1

                if (batch_indexer == n_batch_inference or
                    # last batch
                    i == nxf - 1 and j == nyf - 1):
                    self.logger.debug("predicting patches on %s", str(img_patch.shape))
                    label_p_pred = model.predict(img_patch,verbose=0)
                    seg = np.argmax(label_p_pred, axis=3)
                    conf = label_p_pred[tuple(np.indices(seg.shape)) + (seg,)]

                    if thresholding_for_artificial_class:
                        seg_art = label_p_pred[:, :, :, artificial_class] >= threshold_art_class

                    indexer_inside_batch = 0
                    for i_batch, j_batch in zip(list_i_s, list_j_s):
                        seg_in = seg[indexer_inside_batch]
                        conf_in = conf[indexer_inside_batch]

                        if thresholding_for_artificial_class:
                            seg_in_art = seg_art[indexer_inside_batch]

                        index_y_u_in = list_y_u[indexer_inside_batch]
                        index_y_d_in = list_y_d[indexer_inside_batch]

                        index_x_u_in = list_x_u[indexer_inside_batch]
                        index_x_d_in = list_x_d[indexer_inside_batch]

                        where = np.index_exp[index_y_d_in:index_y_u_in,
                                             index_x_d_in:index_x_u_in]
                        if (i_batch == 0 and
                            j_batch == 0):
                            inbox = np.index_exp[0:-margin or None,
                                                 0:-margin or None]
                        elif (i_batch == nxf - 1 and
                              j_batch == nyf - 1):
                            inbox = np.index_exp[margin:,
                                                 margin:]
                        elif (i_batch == 0 and
                              j_batch == nyf - 1):
                            inbox = np.index_exp[margin:,
                                                 0:-margin or None]
                        elif (i_batch == nxf - 1 and
                              j_batch == 0):
                            inbox = np.index_exp[0:-margin or None,
                                                 margin:]
                        elif (i_batch == 0 and
                              j_batch != 0 and
                              j_batch != nyf - 1):
                            inbox = np.index_exp[margin:-margin or None,
                                                 0:-margin or None]
                        elif (i_batch == nxf - 1 and
                              j_batch != 0 and
                              j_batch != nyf - 1):
                            inbox = np.index_exp[margin:-margin or None,
                                                 margin:]
                        elif (i_batch != 0 and
                              i_batch != nxf - 1 and
                              j_batch == 0):
                            inbox = np.index_exp[0:-margin or None,
                                                 margin:-margin or None]
                        elif (i_batch != 0 and
                              i_batch != nxf - 1 and
                              j_batch == nyf - 1):
                            inbox = np.index_exp[margin:,
                                                 margin:-margin or None]
                        else:
                            inbox = np.index_exp[margin:-margin or None,
                                                 margin:-margin or None]
                        prediction[where][inbox] = seg_in[inbox]
                        confidence[where][inbox] = conf_in[inbox]
                        if thresholding_for_artificial_class:
                            mask_artificial_class[where][inbox] = seg_in_art[inbox]

                        indexer_inside_batch += 1

                    list_i_s = []
                    list_j_s = []
                    list_x_u = []
                    list_x_d = []
                    list_y_u = []
                    list_y_d = []

                    batch_indexer = 0
                    img_patch[:] = 0

        if thresholding_for_artificial_class:
            seg_mask_label(prediction, mask_artificial_class,
                           label=artificial_class,
                           only=True,
                           skeletonize=True,
                           dilate=3,
                           keep=separator_class)
        gc.collect()
        return prediction, confidence

    # variant of do_prediction_new_concept with no need
    # for resizing or tiling into patches - done on model
    # (Tensorflow/CUDA) side
    # (after loading wrapped resized or patched model)
    def do_prediction_new_concept_autosize(
            self, img, model,
            n_batch_inference=None,
            thresholding_for_heading=False,
            thresholding_for_artificial_class=False,
            threshold_art_class=0.1,
            artificial_class=4,
    ):
        self.logger.debug("enter do_prediction_new_concept (%s)", model.name)
        img = img / 255.0
        img = img.astype(np.float16)

        prediction = model.predict(img[np.newaxis])[0]
        confidence = prediction[:, :, 1]
        segmentation = np.argmax(prediction, axis=2).astype(np.uint8)

        if thresholding_for_artificial_class:
            seg_mask_label(segmentation,
                           prediction[:, :, artificial_class] >= threshold_art_class,
                           label=artificial_class,
                           only=True,
                           skeletonize=True,
                           dilate=3)
        if thresholding_for_heading:
            seg_mask_label(segmentation,
                           prediction[:, :, 2] >= 0.2,
                           label=2)
        gc.collect()
        return segmentation, confidence

    def extract_page(self, image):
        self.logger.debug("enter extract_page")
        cropped_page = img = image['img_res']
        h, w = img.shape[:2]
        page_coord = [0, h, 0, w]
        cont_page = [np.array([[[0, 0]],
                               [[w, 0]],
                               [[w, h]],
                               [[0, h]]])]
        if not self.ignore_page_extraction:
            #cv2.GaussianBlur(img, (5, 5), 0)
            prediction = self.do_prediction(False, img, self.model_zoo.get("page"))
            contours, _ = cv2.findContours(prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours)>0:
                cnt_size = np.array([cv2.contourArea(contours[j])
                                     for j in range(len(contours))])
                cnt = contours[np.argmax(cnt_size)]
                cont_page = [cnt]
                x, y, w, h = cv2.boundingRect(cnt)
                #if x <= 30:
                    #w += x
                    #x = 0
                #if (self.image.shape[1] - (x + w)) <= 30:
                    #w = w + (self.image.shape[1] - (x + w))
                #if y <= 30:
                    #h = h + y
                    #y = 0
                #if (self.image.shape[0] - (y + h)) <= 30:
                    #h = h + (self.image.shape[0] - (y + h))
                box = [x, y, w, h]
                cropped_page, page_coord = crop_image_inside_box(box, img)
            self.logger.debug("exit extract_page")
        return cropped_page, page_coord, cont_page

    def early_page_for_num_of_column_classification(self, image):
        img = self.imread(image, binary=self.input_binary)
        if not self.ignore_page_extraction:
            self.logger.debug("enter early_page_for_num_of_column_classification")
            img = cv2.GaussianBlur(img, (5, 5), 0)
            prediction = self.do_prediction(False, img, self.model_zoo.get("page"))
            prediction = cv2.dilate(prediction, KERNEL, iterations=3)
            contours, _ = cv2.findContours(prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)>0:
                cnt_size = np.array([cv2.contourArea(contours[j])
                                     for j in range(len(contours))])
                cnt = contours[np.argmax(cnt_size)]
                box = cv2.boundingRect(cnt)
            else:
                box = [0, 0, img.shape[1], img.shape[0]]
            self.logger.debug("exit early_page_for_num_of_column_classification")
        else:
            box = [0, 0, img.shape[1], img.shape[0]]
        cropped_page, page_coord = crop_image_inside_box(box, img)
        return cropped_page, page_coord

    def extract_text_regions_new(self, img, patches, cols):
        self.logger.debug("enter extract_text_regions_new")
        img_height_h = img.shape[0]
        img_width_h = img.shape[1]

        prediction_regions, confidence_regions = self.do_prediction_new_concept(
            patches, img, self.model_zoo.get("region_fl" if patches else "region_fl_np"),
            n_batch_inference=1,
            thresholding_for_heading=not patches)

        self.logger.debug("exit extract_text_regions_new")
        return prediction_regions, confidence_regions

    def extract_text_regions(self, img, patches, cols):
        self.logger.debug("enter extract_text_regions")
        img_height_h = img.shape[0]
        img_width_h = img.shape[1]
        model_region = self.model_zoo.get("region_fl" if patches else "region_fl_np")

        prediction_regions = self.do_prediction(patches, img, model_region,
                                                marginal_of_patch_percent=0.1)
        prediction_regions = resize_image(prediction_regions, img_height_h, img_width_h)
        self.logger.debug("exit extract_text_regions")
        return prediction_regions

    def get_textlines_of_a_textregion_sorted(self, textlines_textregion, cx_textline, cy_textline, w_h_textline):
        N = len(cy_textline)
        if N==0:
            return []

        diff_cy = np.abs( np.diff(sorted(cy_textline)) )
        diff_cx = np.abs(np.diff(sorted(cx_textline)) )


        if len(diff_cy)>0:
            mean_y_diff = np.mean(diff_cy)
            mean_x_diff = np.mean(diff_cx)
            count_hor = np.count_nonzero(np.array(w_h_textline) > 1)
            count_ver = len(w_h_textline) - count_hor

        else:
            mean_y_diff = 0
            mean_x_diff = 0
            count_hor = 1
            count_ver = 0


        if count_hor >= count_ver:
            row_threshold = mean_y_diff / 1.5  if mean_y_diff > 0 else 10

            indices_sorted_by_y = sorted(range(N), key=lambda i: cy_textline[i])

            rows = []
            current_row = [indices_sorted_by_y[0]]
            for i in range(1, N):
                current_idx = indices_sorted_by_y[i]
                prev_idx = current_row[0]
                if abs(cy_textline[current_idx] - cy_textline[prev_idx]) <= row_threshold:
                    current_row.append(current_idx)
                else:
                    rows.append(current_row)
                    current_row = [current_idx]
            rows.append(current_row)

            sorted_textlines = []
            for row in rows:
                row_sorted = sorted(row, key=lambda i: cx_textline[i])
                for idx in row_sorted:
                    sorted_textlines.append(textlines_textregion[idx])

        else:
            row_threshold = mean_x_diff / 1.5 if mean_x_diff > 0 else 10
            indices_sorted_by_x = sorted(range(N), key=lambda i: cx_textline[i])

            rows = []
            current_row = [indices_sorted_by_x[0]]

            for i in range(1, N):
                current_idy = indices_sorted_by_x[i]
                prev_idy = current_row[0]
                if abs(cx_textline[current_idy] - cx_textline[prev_idy] ) <= row_threshold:
                    current_row.append(current_idy)
                else:
                    rows.append(current_row)
                    current_row = [current_idy]
            rows.append(current_row)

            sorted_textlines = []
            for row in rows:
                row_sorted = sorted(row , key=lambda i: cy_textline[i])
                for idy in row_sorted:
                    sorted_textlines.append(textlines_textregion[idy])

        return sorted_textlines

    def get_slopes_and_deskew_new_light2(self, contours_par, textline_mask_tot, boxes, slope_deskew):

        polygons_of_textlines = return_contours_of_interested_region(textline_mask_tot,1,0.00001)
        cx_main_tot, cy_main_tot = find_center_of_contours(polygons_of_textlines)
        w_h_textlines = [cv2.boundingRect(polygon)[2:] for polygon in polygons_of_textlines]

        args_textlines = np.arange(len(polygons_of_textlines))
        all_found_textline_polygons = []
        slopes = []
        all_box_coord =[]

        for index, con_region_ind in enumerate(contours_par):
            results = [cv2.pointPolygonTest(con_region_ind, (cx_main_tot[ind], cy_main_tot[ind]), False)
                       for ind in args_textlines ]
            results = np.array(results)
            indexes_in = args_textlines[results==1]
            textlines_ins = [polygons_of_textlines[ind] for ind in indexes_in]
            cx_textline_in = [cx_main_tot[ind] for ind in indexes_in]
            cy_textline_in = [cy_main_tot[ind] for ind in indexes_in]
            w_h_textlines_in = [w_h_textlines[ind][0] / float(w_h_textlines[ind][1])  for ind in indexes_in]

            textlines_ins = self.get_textlines_of_a_textregion_sorted(textlines_ins,
                                                                      cx_textline_in,
                                                                      cy_textline_in,
                                                                      w_h_textlines_in)

            all_found_textline_polygons.append(textlines_ins)#[::-1])
            slopes.append(slope_deskew)

            crop_coor = box2rect(boxes[index])
            all_box_coord.append(crop_coor)

        return (all_found_textline_polygons,
                all_box_coord,
                slopes)

    def get_slopes_and_deskew_new_curved(self, contours_par, textline_mask_tot, boxes,
                                         num_col, scale_par, slope_deskew, name):
        if not len(contours_par):
            return [], [], []
        self.logger.debug("enter get_slopes_and_deskew_new_curved")
        results = map(partial(do_work_of_slopes_new_curved,
                              textline_mask_tot_ea=textline_mask_tot,
                              num_col=num_col,
                              scale_par=scale_par,
                              slope_deskew=slope_deskew,
                              MAX_SLOPE=MAX_SLOPE,
                              KERNEL=KERNEL,
                              logger=self.logger,
                              plotter=self.plotter,
                              name=name),
                      boxes, contours_par)
        results = list(results) # exhaust prior to release
        #textline_polygons, box_coord, slopes = zip(*results)
        self.logger.debug("exit get_slopes_and_deskew_new_curved")
        return tuple(zip(*results))

    def textline_contours(self, img, use_patches):
        self.logger.debug('enter textline_contours')

        if (self.tables or
            self.reading_order_machine_based or
            self.input_binary):
             # avoid OOM
            n_batch = 1
        else:
            n_batch = 3
        prediction_textline, conf_textline = self.do_prediction_new_concept(
            use_patches, img, self.model_zoo.get("textline"),
            artificial_class=2,
            n_batch_inference=n_batch,
            thresholding_for_artificial_class=True,
            threshold_art_class=self.threshold_art_class_textline)

        #prediction_textline_longshot = self.do_prediction(False, img, self.model_zoo.get("textline"))

        self.logger.debug('exit textline_contours')
        # suppress artificial boundary label
        result = (prediction_textline == 1).astype(np.uint8)
        #, (prediction_textline_longshot==1).astype(np.uint8)
        return result, conf_textline

    def get_early_layout(
            self, image,
            num_col_classifier,
            label_text=1,
            label_imgs=2,
            label_seps=3,
    ):
        self.logger.debug("enter get_early_layout")
        t_in = time.time()
        erosion_hurts = False
        img = image['img_res']
        img_height_h = img.shape[0]
        img_width_h = img.shape[1]
        img_org = image['img']
        img_height_org = img_org.shape[0]
        img_width_org = img_org.shape[1]

        if num_col_classifier == 1:
            img_w_new = 1000
        elif num_col_classifier == 2:
            img_w_new = 1500#1500
        elif num_col_classifier == 3:
            img_w_new = 2000
        elif num_col_classifier == 4:
            img_w_new = 2500
        elif num_col_classifier == 5:
            img_w_new = 3000
        else:
            img_w_new = 4000
        img_h_new = img_w_new * img.shape[0] // img.shape[1]
        img_resized = resize_image(img, img_h_new, img_w_new)
        self.logger.debug("detecting textlines on %s with %d colors",
                          str(img_resized.shape), len(np.unique(img_resized)))

        textline_mask_tot_ea, confidence_textline = self.run_textline(img_resized)
        textline_mask_tot_ea = resize_image(textline_mask_tot_ea, img_height_h, img_width_h)
        confidence_textline = resize_image(confidence_textline, img_height_h, img_width_h)
        if self.plotter:
            self.plotter.save_plot_of_textlines(textline_mask_tot_ea, img_resized, image['name'])

        if self.skip_layout_and_reading_order:
            self.logger.debug("exit get_early_layout")
            return None, erosion_hurts, None, None, textline_mask_tot_ea, None, None

        #print("inside 2 ", time.time()-t_in)
        if num_col_classifier == 1 or num_col_classifier == 2:
            if img_height_h / img_width_h > 2.5:
                patches = True
            else:
                patches = False
            self.logger.debug("resized to %dx%d for %d cols",
                              img_resized.shape[1], img_resized.shape[0],
                              num_col_classifier)
        else:
            new_w = (900+ (num_col_classifier-3)*100)
            new_h = new_w * img.shape[0] // img.shape[1]
            img_resized = resize_image(img_resized, new_h, new_w)
            self.logger.debug("resized to %dx%d (new_w=%d) for %d cols",
                              img_resized.shape[1], img_resized.shape[0],
                              new_w, num_col_classifier)
            patches = True

        prediction_regions, confidence_regions = \
            self.do_prediction_new_concept(
                patches, img_resized, self.model_zoo.get("region_1_2"),
                n_batch_inference=1,
                thresholding_for_artificial_class=True,
                threshold_art_class=self.threshold_art_class_layout,
                separator_class=label_seps)

        prediction_regions = resize_image(prediction_regions, img_height_h, img_width_h)
        confidence_regions = resize_image(confidence_regions, img_height_h, img_width_h)

        mask_texts_only = (prediction_regions == label_text).astype('uint8')
        mask_images_only = (prediction_regions == label_imgs).astype('uint8')
        mask_seps_only = (prediction_regions == label_seps).astype('uint8')

        ##if num_col_classifier == 1 or num_col_classifier == 2:
            ###mask_texts_only = cv2.erode(mask_texts_only, KERNEL, iterations=1)
            ##mask_texts_only = cv2.dilate(mask_texts_only, KERNEL, iterations=1)
        mask_texts_only = cv2.dilate(mask_texts_only, kernel=np.ones((2,2), np.uint8), iterations=1)

        polygons_seplines, hir_seplines = return_contours_of_image(mask_seps_only)
        polygons_seplines = filter_contours_area_of_image(
            mask_seps_only, polygons_seplines, hir_seplines, max_area=1, min_area=0.00001, dilate=1)

        polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only,1,0.00001)
        ##polygons_of_only_texts = dilate_textregion_contours(polygons_of_only_texts)
        polygons_of_only_seps = return_contours_of_interested_region(mask_seps_only,1,0.00001)

        text_regions_p = np.zeros_like(prediction_regions)
        text_regions_p = cv2.fillPoly(text_regions_p, pts=polygons_of_only_seps, color=label_seps)
        text_regions_p[mask_images_only == 1] = label_imgs
        text_regions_p = cv2.fillPoly(text_regions_p, pts=polygons_of_only_texts, color=label_text)

        textline_mask_tot_ea[text_regions_p == 0] = 0
        #plt.imshow(textline_mask_tot_ea)
        #plt.show()
        #print("inside 4 ", time.time()-t_in)
        self.logger.debug("exit get_early_layout")
        return (text_regions_p,
                erosion_hurts,
                polygons_seplines,
                polygons_of_only_texts,
                textline_mask_tot_ea,
                confidence_regions,
                confidence_textline)

    def do_order_of_regions(
            self,
            contours_only_text_parent,
            contours_only_text_parent_h,
            polygons_of_drop_capitals,
            boxes,
            textline_mask_tot
    ):

        self.logger.debug("enter do_order_of_regions")
        contours_only_text_parent = ensure_array(contours_only_text_parent)
        contours_only_text_parent_h = ensure_array(contours_only_text_parent_h)
        polygons_of_drop_capitals = ensure_array(polygons_of_drop_capitals)
        boxes = np.array(boxes, dtype=int) # to be on the safe side
        c_boxes = np.stack((0.5 * boxes[:, 2:4].sum(axis=1),
                            0.5 * boxes[:, 0:2].sum(axis=1)))

        def match_boxes(contours, only_centers: bool, kind: str):
            cx, cy, mx, Mx, my, My, mxy = find_new_features_of_contours(contours)
            cx = np.array(cx, dtype=int)
            cy = np.array(cy, dtype=int)
            arg_text_con = np.zeros(len(contours), dtype=int)
            for ii in range(len(contours)):
                box_found = False
                for jj, box in enumerate(boxes):
                    if ((cx[ii] >= box[0] and
                         cx[ii] < box[1] and
                         cy[ii] >= box[2] and
                         cy[ii] < box[3]) if only_centers else
                        (mx[ii] >= box[0] and
                         Mx[ii] < box[1] and
                         my[ii] >= box[2] and
                         My[ii] < box[3])):
                        arg_text_con[ii] = jj
                        box_found = True
                        # print(kind, "/matched ", ii, "\t", (mx[ii], Mx[ii], my[ii], My[ii]), "\tin", jj, box, only_centers)
                        break
                if not box_found:
                    dists_tr_from_box = np.linalg.norm(c_boxes - np.array([[cy[ii]], [cx[ii]]]), axis=0)
                    pcontained_in_box = ((boxes[:, 2] <= cy[ii]) & (cy[ii] < boxes[:, 3]) &
                                         (boxes[:, 0] <= cx[ii]) & (cx[ii] < boxes[:, 1]))
                    assert pcontained_in_box.any(), (ii, cx[ii], cy[ii])
                    ind_min = np.argmin(np.ma.masked_array(dists_tr_from_box, ~pcontained_in_box))
                    arg_text_con[ii] = ind_min
                    # print(kind, "/fallback ", ii, "\t", (mx[ii], Mx[ii], my[ii], My[ii]), "\tin", ind_min, boxes[ind_min], only_centers)
            return arg_text_con

        def order_from_boxes(only_centers: bool):
            arg_text_con_main = match_boxes(contours_only_text_parent, only_centers, "main")
            arg_text_con_head = match_boxes(contours_only_text_parent_h, only_centers, "head")
            arg_text_con_drop = match_boxes(polygons_of_drop_capitals, only_centers, "drop")
            args_contours_main = np.arange(len(contours_only_text_parent))
            args_contours_head = np.arange(len(contours_only_text_parent_h))
            args_contours_drop = np.arange(len(polygons_of_drop_capitals))
            order_by_con_main = np.zeros_like(arg_text_con_main)
            order_by_con_head = np.zeros_like(arg_text_con_head)
            order_by_con_drop = np.zeros_like(arg_text_con_drop)
            idx = 0
            for iij, box in enumerate(boxes):
                ys = slice(*box[2:4])
                xs = slice(*box[0:2])
                args_contours_box_main = args_contours_main[arg_text_con_main == iij]
                args_contours_box_head = args_contours_head[arg_text_con_head == iij]
                args_contours_box_drop = args_contours_drop[arg_text_con_drop == iij]

                _, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(
                    textline_mask_tot[ys, xs],
                    contours_only_text_parent[args_contours_box_main],
                    contours_only_text_parent_h[args_contours_box_head],
                    polygons_of_drop_capitals[args_contours_box_drop],
                    box[2], box[0])

                for tidx, kind in zip(index_by_kind_sorted, kind_of_texts_sorted):
                    if kind == 1:
                        # print(iij, "main", args_contours_box_main[tidx], "becomes", idx)
                        order_by_con_main[args_contours_box_main[tidx]] = idx
                    elif kind == 2:
                        # print(iij, "head", args_contours_box_head[tidx], "becomes", idx)
                        order_by_con_head[args_contours_box_head[tidx]] = idx
                    else:
                        # print(iij, "drop", args_contours_box_drop[tidx], "becomes", idx)
                        order_by_con_drop[args_contours_box_drop[tidx]] = idx
                    idx += 1

            # xml writer will create region ids in order of
            # - contours_only_text_parent (main text), followed by
            # - contours_only_text_parent_h (headings), and then
            # - polygons_of_drop_capitals,
            # and then create regionrefs into these ordered by order_text_new
            order_text_new = np.argsort(np.concatenate((order_by_con_main,
                                                        order_by_con_head,
                                                        order_by_con_drop)))
            return order_text_new

        try:
            results = order_from_boxes(False)
        except Exception as why:
            self.logger.exception(why)
            results = order_from_boxes(True)

        self.logger.debug("exit do_order_of_regions")
        return results

    def check_iou_of_bounding_box_and_contour_for_tables(
            self, layout, table_prediction_early, pixel_table, num_col_classifier):

        layout_org  = np.copy(layout)
        layout_org[layout_org == pixel_table] = 0
        layout = (layout == pixel_table).astype(np.uint8) * 1
        _, thresh = cv2.threshold(layout, 0, 255, 0)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_size = np.array([cv2.contourArea(cnt) for cnt in contours])

        contours_new = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            iou = cnt_size[i] /float(w*h) *100
            if iou<80:
                layout_contour = np.zeros(layout_org.shape[:2])
                layout_contour = cv2.fillPoly(layout_contour, pts=[contour] ,color=1)

                layout_contour_sum = layout_contour.sum(axis=0)
                layout_contour_sum_diff = np.diff(layout_contour_sum)
                layout_contour_sum_diff= np.abs(layout_contour_sum_diff)
                layout_contour_sum_diff_smoothed= gaussian_filter1d(layout_contour_sum_diff, 10)

                peaks, _ = find_peaks(layout_contour_sum_diff_smoothed, height=0)
                peaks= peaks[layout_contour_sum_diff_smoothed[peaks]>4]

                for j in range(len(peaks)):
                    layout_contour[:,peaks[j]-3+1:peaks[j]+1+3] = 0

                layout_contour=cv2.erode(layout_contour[:,:], KERNEL, iterations=5)
                layout_contour=cv2.dilate(layout_contour[:,:], KERNEL, iterations=5)

                layout_contour = layout_contour.astype(np.uint8)
                _, thresh = cv2.threshold(layout_contour, 0, 255, 0)

                contours_sep, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for ji in range(len(contours_sep) ):
                    contours_new.append(contours_sep[ji])
                    if num_col_classifier>=2:
                        only_recent_contour_image = np.zeros(layout.shape[:2])
                        only_recent_contour_image = cv2.fillPoly(only_recent_contour_image,
                                                                 pts=[contours_sep[ji]], color=1)
                        table_pixels_masked_from_early_pre = only_recent_contour_image * table_prediction_early
                        iou_in = 100. * table_pixels_masked_from_early_pre.sum() / only_recent_contour_image.sum()
                        #print(iou_in,'iou_in_in1')

                        if iou_in>30:
                            layout_org = cv2.fillPoly(layout_org, pts=[contours_sep[ji]], color=pixel_table)
                        else:
                            pass
                    else:
                        layout_org= cv2.fillPoly(layout_org, pts=[contours_sep[ji]], color=pixel_table)
            else:
                contours_new.append(contour)
                if num_col_classifier>=2:
                    only_recent_contour_image = np.zeros(layout.shape[:2])
                    only_recent_contour_image = cv2.fillPoly(only_recent_contour_image, pts=[contour],color=1)

                    table_pixels_masked_from_early_pre = only_recent_contour_image * table_prediction_early
                    iou_in = 100. * table_pixels_masked_from_early_pre.sum() / only_recent_contour_image.sum()
                    #print(iou_in,'iou_in')
                    if iou_in>30:
                        layout_org = cv2.fillPoly(layout_org, pts=[contour], color=pixel_table)
                    else:
                        pass
                else:
                    layout_org = cv2.fillPoly(layout_org, pts=[contour], color=pixel_table)

        return layout_org, contours_new

    def delete_separator_around(self, splitter_y, peaks_neg, image_by_region, label_seps, label_table):
        # format of subboxes: box=[x1, x2 , y1, y2]
        pix_del = 100
        for i in range(len(splitter_y)-1):
            for j in range(1,len(peaks_neg[i])-1):
                where = np.index_exp[splitter_y[i]:
                                     splitter_y[i+1],
                                     peaks_neg[i][j] - pix_del:
                                     peaks_neg[i][j] + pix_del,
                                     :]
                if image_by_region.ndim < 3:
                    where = where[:2]
                else:
                    print("image_by_region ndim is 3!") # rs
                image_by_region[where][image_by_region[where] == label_seps] = 0
                image_by_region[where][image_by_region[where] == label_table] = 0
        return image_by_region

    def add_tables_heuristic_to_layout(
            self, image_regions_eraly_p, boxes,
            slope_mean_hor, splitter_y, peaks_neg_tot, image_revised,
            num_col_classifier, min_area, label_seps):

        label_table =10
        image_revised_1 = self.delete_separator_around(splitter_y, peaks_neg_tot, image_revised, label_seps, label_table)

        try:
            image_revised_1[:,:30][image_revised_1[:,:30]==label_seps] = 0
            image_revised_1[:,-30:][image_revised_1[:,-30:]==label_seps] = 0
        except:
            pass
        boxes = np.array(boxes, dtype=int) # to be on the safe side

        img_comm = np.zeros(image_revised_1.shape, dtype=np.uint8)
        for indiv in np.unique(image_revised_1):
            image_col = (image_revised_1 == indiv).astype(np.uint8) * 255
            _, thresh = cv2.threshold(image_col, 0, 255, 0)
            contours,hirarchy=cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            if indiv==label_table:
                main_contours = filter_contours_area_of_image_tables(thresh, contours, hirarchy,
                                                                     max_area=1, min_area=0.001)
            else:
                main_contours = filter_contours_area_of_image_tables(thresh, contours, hirarchy,
                                                                     max_area=1, min_area=min_area)

            img_comm = cv2.fillPoly(img_comm, pts=main_contours, color=indiv)

        if not isNaN(slope_mean_hor):
            image_revised_last = np.zeros(image_regions_eraly_p.shape[:2])
            for i in range(len(boxes)):
                box_ys = slice(*boxes[i][2:4])
                box_xs = slice(*boxes[i][0:2])
                image_box = img_comm[box_ys, box_xs]
                try:
                    image_box_tabels_1 = (image_box == label_table) * 1
                    contours_tab,_=return_contours_of_image(image_box_tabels_1)
                    contours_tab=filter_contours_area_of_image_tables(image_box_tabels_1,contours_tab,_,1,0.003)
                    image_box_tabels_1 = (image_box == label_seps).astype(np.uint8) * 1
                    image_box_tabels_and_m_text = ( (image_box == label_table) |
                                                    (image_box == 1) ).astype(np.uint8) * 1

                    image_box_tabels_1 = cv2.dilate(image_box_tabels_1, KERNEL, iterations=5)

                    contours_table_m_text, _ = return_contours_of_image(image_box_tabels_and_m_text)
                    _, thresh = cv2.threshold(image_box_tabels_1, 0, 255, 0)
                    contours_line, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    y_min_main_line ,y_max_main_line=find_features_of_contours(contours_line)
                    y_min_main_tab ,y_max_main_tab=find_features_of_contours(contours_tab)

                    (cx_tab_m_text, cy_tab_m_text,
                     x_min_tab_m_text, x_max_tab_m_text,
                     y_min_tab_m_text, y_max_tab_m_text,
                     _) = find_new_features_of_contours(contours_table_m_text)
                    (cx_tabl, cy_tabl,
                     x_min_tabl, x_max_tabl,
                     y_min_tabl, y_max_tabl,
                     _) = find_new_features_of_contours(contours_tab)

                    if len(y_min_main_tab )>0:
                        y_down_tabs=[]
                        y_up_tabs=[]

                        for i_t in range(len(y_min_main_tab )):
                            y_down_tab=[]
                            y_up_tab=[]
                            for i_l in range(len(y_min_main_line)):
                                if (y_min_main_tab[i_t] > y_min_main_line[i_l] and
                                    y_max_main_tab[i_t] > y_min_main_line[i_l] and
                                    y_min_main_tab[i_t] > y_max_main_line[i_l] and
                                    y_max_main_tab[i_t] > y_min_main_line[i_l]):
                                    pass
                                elif (y_min_main_tab[i_t] < y_max_main_line[i_l] and
                                      y_max_main_tab[i_t] < y_max_main_line[i_l] and
                                      y_max_main_tab[i_t] < y_min_main_line[i_l] and
                                      y_min_main_tab[i_t] < y_min_main_line[i_l]):
                                    pass
                                elif abs(y_max_main_line[i_l] - y_min_main_line[i_l]) < 100:
                                    pass
                                else:
                                    y_up_tab.append(min([y_min_main_line[i_l],
                                                         y_min_main_tab[i_t]]))
                                    y_down_tab.append(max([y_max_main_line[i_l],
                                                           y_max_main_tab[i_t]]))

                            if len(y_up_tab)==0:
                                y_up_tabs.append(y_min_main_tab[i_t])
                                y_down_tabs.append(y_max_main_tab[i_t])
                            else:
                                y_up_tabs.append(min(y_up_tab))
                                y_down_tabs.append(max(y_down_tab))
                    else:
                        y_down_tabs=[]
                        y_up_tabs=[]
                        pass
                except:
                    y_down_tabs=[]
                    y_up_tabs=[]

                for ii in range(len(y_up_tabs)):
                    image_box[y_up_tabs[ii]:y_down_tabs[ii]] = label_table

                image_revised_last[box_ys, box_xs] = image_box
        else:
            for i in range(len(boxes)):
                box_ys = slice(*boxes[i][2:4])
                box_xs = slice(*boxes[i][0:2])
                image_box = img_comm[box_ys, box_xs]
                image_revised_last[box_ys, box_xs] = image_box

        if num_col_classifier==1:
            img_tables_col_1 = (image_revised_last == label_table).astype(np.uint8)
            contours_table_col1, _ = return_contours_of_image(img_tables_col_1)

            _,_ ,_ , _, y_min_tab_col1 ,y_max_tab_col1, _= find_new_features_of_contours(contours_table_col1)

            if len(y_min_tab_col1)>0:
                for ijv in range(len(y_min_tab_col1)):
                    image_revised_last[int(y_min_tab_col1[ijv]):int(y_max_tab_col1[ijv])] = label_table
        return image_revised_last

    def get_tables_from_model(self, img):
        table_prediction, table_confidence = self.do_prediction_new_concept(
            False, img,
            self.model_zoo.get("table"),
            thresholding_for_artificial_class=True,
            threshold_art_class=0.05,
            artificial_class=2)
        table_prediction = table_prediction.astype(np.uint8)
        return table_prediction, table_confidence

    def run_graphics_and_columns(
            self, text_regions_p_1, textline_mask_tot_ea,
            regions_confidence, textline_confidence,
            num_col_classifier, num_column_is_classified,
            erosion_hurts, image,
            label_imgs=2,
            label_seps=3,
    ):

        #print(text_regions_p_1.shape, 'text_regions_p_1 shape run graphics')
        #print(erosion_hurts, 'erosion_hurts')
        t_in_gr = time.time()

        image_page, page_coord, cont_page = self.extract_page(image)
        #print("inside graphics 1 ", time.time() - t_in_gr)
        if self.tables:
            table_prediction, table_confidence = self.get_tables_from_model(image_page)
        else:
            table_prediction = np.zeros(image_page.shape[:2], dtype=np.uint8)
            table_confidence = np.zeros(image_page.shape[:2], dtype=bool)

        if self.plotter:
            self.plotter.save_page_image(image_page, image['name'])

        if not self.ignore_page_extraction:
            mask_page = np.zeros_like(text_regions_p_1)
            mask_page = cv2.fillPoly(mask_page, pts=[cont_page[0]], color=1)
            mask_page = mask_page == 0

            text_regions_p_1[mask_page] = 0
            textline_mask_tot_ea[mask_page] = 0
            regions_confidence[mask_page] = 0
            textline_confidence[mask_page] = 0

        box = slice(*page_coord[0:2]), slice(*page_coord[2:4])
        text_regions_p_1 = text_regions_p_1[box]
        textline_mask_tot_ea = textline_mask_tot_ea[box]
        regions_confidence = regions_confidence[box]
        textline_confidence = textline_confidence[box]

        mask_images = (text_regions_p_1 == label_imgs).astype(np.uint8)
        mask_images = cv2.erode(mask_images, KERNEL, iterations=10)
        textline_mask_tot_ea[mask_images == 1] = 0
        textline_confidence[mask_images == 1] = 0

        img_only_regions_with_sep = ((text_regions_p_1 != label_seps) &
                                     (text_regions_p_1 != 0)).astype(np.uint8)

        #print("inside graphics 2 ", time.time() - t_in_gr)
        if erosion_hurts:
            img_only_regions = img_only_regions_with_sep
        else:
            img_only_regions = cv2.erode(img_only_regions_with_sep, KERNEL, iterations=6)

        ##print(img_only_regions.shape,'img_only_regions')
        ##plt.imshow(img_only_regions[:,:])
        ##plt.show()
        ##num_col, _ = find_num_col(img_only_regions, num_col_classifier, self.tables, multiplier=6.0)
        try:
            num_col, _ = find_num_col(img_only_regions, num_col_classifier, self.tables, multiplier=6.0)
            num_col = num_col + 1
            if not num_column_is_classified:
                num_col_classifier = num_col
            num_col_classifier = min(self.num_col_upper or num_col_classifier,
                                     max(self.num_col_lower or num_col_classifier,
                                         num_col_classifier))
        except Exception as why:
            self.logger.exception(why)
            num_col = None
        return (num_col, num_col_classifier,
                page_coord, image_page, cont_page,
                text_regions_p_1,
                table_prediction,
                textline_mask_tot_ea,
                regions_confidence,
                table_confidence,
                textline_confidence,
        )

    def run_graphics_and_columns_without_layout(self, textline_mask_tot_ea, image):
        image_page, page_coord, cont_page = self.extract_page(image)

        mask_page = np.zeros_like(textline_mask_tot_ea)
        mask_page = cv2.fillPoly(mask_page, pts=[cont_page[0]], color=1)
        mask_page = mask_page == 0

        textline_mask_tot_ea[mask_page] = 0
        box = slice(*page_coord[0:2]), slice(*page_coord[2:4])
        textline_mask_tot_ea = textline_mask_tot_ea[box]

        return page_coord, image_page, textline_mask_tot_ea, cont_page


    def run_enhancement(self, image):
        t_in = time.time()
        self.logger.info("Resizing and enhancing image...")
        is_image_enhanced, num_col_classifier, num_column_is_classified = \
            self.resize_and_enhance_image_with_column_classifier(image)
        self.logger.info("Image was %senhanced.", '' if is_image_enhanced else 'not ')
        if is_image_enhanced:
            if self.allow_enhancement:
                if self.plotter:
                    self.plotter.save_enhanced_image(image['img_res'], image['name'])
        else:
            # rs FIXME: dead branch (i.e. no actual enhancement/scaling done)
            #           also, unclear why col classifier should run again on same input
            #           (why not predict enhancement iff size(img_res) > size(img_org) ?)
            if self.allow_scaling:
                self.resize_image_with_column_classifier(image)

        #print("enhancement in ", time.time()-t_in)
        return num_col_classifier, num_column_is_classified

    def run_textline(self, image_page):
        textline_mask_tot_ea, textline_conf = self.textline_contours(image_page, True)
        #textline_mask_tot_ea = textline_mask_tot_ea.astype(np.int16)
        return textline_mask_tot_ea, textline_conf

    def run_deskew(self, textline_mask_tot_ea):
        if not np.any(textline_mask_tot_ea):
            self.logger.info("slope_deskew: empty page")
            return 0

        #print(textline_mask_tot_ea.shape, 'textline_mask_tot_ea deskew')
        slope_deskew = return_deskew_slop(cv2.erode(textline_mask_tot_ea, KERNEL, iterations=2), 2, 30, True,
                                          logger=self.logger, plotter=self.plotter)
        self.logger.info("slope_deskew: %.2f°", slope_deskew)
        return slope_deskew

    def run_marginals(
            self, num_col_classifier, slope_deskew, text_regions_p_1, table_prediction):

        text_regions_p = np.array(text_regions_p_1)
        if num_col_classifier in (1, 2):
            try:
                regions_without_separators = (text_regions_p == 1) * 1
                if self.tables:
                    regions_without_separators[table_prediction == 1] = 1
                regions_without_separators = regions_without_separators.astype(np.uint8)
                text_regions_p = get_marginals(
                    rotate_image(regions_without_separators, slope_deskew), text_regions_p,
                    num_col_classifier, slope_deskew, kernel=KERNEL)
            except Exception as e:
                self.logger.error("exception %s", e)

        return text_regions_p

    def get_full_layout(
            self, image_page,
            textline_mask_tot, text_regions_p,
            num_col_classifier,
            table_prediction,
            label_text=1,
            label_imgs=2,
            label_imgs_fl=5,
            label_imgs_fl_model=4,
            label_seps=3,
            label_seps_fl=6,
            label_seps_fl_model=5,
            label_marg=4,
            label_marg_fl=8,
            label_drop_fl=4,
            label_drop_fl_model=3,
            label_tabs=10,
    ):
        self.logger.debug('enter get_full_layout')
        t_full0 = time.time()

        # segment labels used by models/arrays:
        # class | early | old full (and decoded here) | new full (just predicted) | comment
        # ---
        # para | 1 |  1 | 1 |
        # head | - |  2 | 2 | used in split_textregion_main_vs_head()
        # drop | - |  4 | 3 | assigned from full model below
        # img  | 2 |  5 | 4 | mapped below
        # sep  | 3 |  6 | 5 | mapped + assigned from full model below
        # marg | 4 |  8 | - | rule-based in run_marginals() from early text
        # tab  | - | 10 | - | dedicated model, optional
        text_regions_p[text_regions_p == label_imgs] = label_imgs_fl
        text_regions_p[text_regions_p == label_seps] = label_seps_fl
        text_regions_p[text_regions_p == label_marg] = label_marg_fl

        regions_without_separators = (text_regions_p == label_text).astype(np.uint8)
        # regions_without_separators = ( text_regions_p == 1 | text_regions_p == 2 ) * 1

        image_page = image_page.astype(np.uint8)
        if self.full_layout:
            regions_fully, regionsfl_confidence = self.extract_text_regions_new(
                image_page,
                False, cols=num_col_classifier)

            # the separators in full layout will not be written on layout
            if not self.reading_order_machine_based:
                text_regions_p[regions_fully == label_seps_fl_model] = label_seps_fl

            drops = regions_fully == label_drop_fl_model
            regions_fully[drops] = label_text
            # rs: why erode to text here, when fill_bb... will mask out text (only allowing img/drop/bg)?
            drops = cv2.erode(drops.astype(np.uint8), KERNEL, iterations=1) == 1
            regions_fully[drops] = label_drop_fl_model
            drops = fill_bb_of_drop_capitals(regions_fully, text_regions_p)
            text_regions_p[drops] = label_drop_fl

            regions_without_separators[drops] = 1 # also cover in reading-order
        else:
            regions_fully = None,
            regionsfl_confidence = None

        if self.tables:
            text_regions_p[table_prediction == 1] = label_tabs
            regions_without_separators[table_prediction == 1] = 1

        # no need to return text_regions_p (inplace editing)
        self.logger.debug('exit get_full_layout')
        return (regions_fully, regionsfl_confidence,
                regions_without_separators)

    def get_deskewed_masks(
            self,
            slope_deskew,
            textline_mask_tot,
            text_regions_p,
            regions_without_separators,
    ):
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            textline_mask_tot_d = rotate_image(textline_mask_tot, slope_deskew)
            text_regions_p_d = rotate_image(text_regions_p, slope_deskew)
            regions_without_separators_d = rotate_image(regions_without_separators, slope_deskew)
        else:
            textline_mask_tot_d = None
            text_regions_p_d = None
            regions_without_separators_d = None
        return (
            textline_mask_tot_d,
            text_regions_p_d,
            regions_without_separators_d,
        )

    def run_boxes_order(
            self,
            text_regions_p,
            num_col_classifier,
            erosion_hurts,
            regions_without_separators,
            contours_h=None,
            label_seps_fl=6,
    ):
        _, _, matrix_of_seps_ch, splitter_y_new, _ = find_number_of_columns_in_document(
            text_regions_p, num_col_classifier, self.tables, label_seps_fl, contours_h=contours_h)

        if not erosion_hurts:
            regions_without_separators = regions_without_separators.astype(np.uint8)
            regions_without_separators = cv2.erode(regions_without_separators, KERNEL, iterations=6)

        boxes, _ = return_boxes_of_images_by_order_of_reading_new(
            splitter_y_new, regions_without_separators,
            text_regions_p == label_seps_fl, matrix_of_seps_ch,
            num_col_classifier, erosion_hurts, self.tables, self.right2left,
            logger=self.logger)
        return boxes

    def do_order_of_regions_with_model(
            self,
            contours_only_text_parent,
            contours_only_text_parent_h,
            # not trained on drops directly, but it does work:
            polygons_of_drop_capitals,
            text_regions_p,
            n_batch_inference=1, # 3 (causes OOM on 8 GB GPUs)
            # input labels as in run_boxes_full_layout
            # output labels as in RO model's read_xml
            label_text=1,
            label_head=2,
            label_imgs=5,
            label_imgs_ro=4,
            label_seps=6,
            label_seps_ro=5,
            label_marg=8,
            label_marg_ro=3,
            label_drop=4,
            # no drop-capital in RO model, yet
            label_drop_ro=4,
    ):
        model = self.model_zoo.get("reading_order")
        _, height_model, width_model, _ = model.input_shape

        ver_kernel = np.ones((5, 1), dtype=np.uint8)
        hor_kernel = np.ones((1, 5), dtype=np.uint8)
        min_cont_size_to_be_dilated = 10
        if len(contours_only_text_parent) > min_cont_size_to_be_dilated:
            (cx_conts, cy_conts,
             x_min_conts, x_max_conts,
             y_min_conts, y_max_conts,
             _) = find_new_features_of_contours(contours_only_text_parent)
            cx_conts = ensure_array(cx_conts)
            cy_conts = ensure_array(cy_conts)
            contours_only_text_parent = ensure_array(contours_only_text_parent)
            args_cont = np.arange(len(contours_only_text_parent))

            diff_x_conts = np.abs(x_max_conts[:]-x_min_conts)
            mean_x = np.mean(diff_x_conts)
            diff_x_ratio = diff_x_conts / mean_x

            args_cont_excluded = args_cont[diff_x_ratio >= 1.3]
            args_cont_included = args_cont[diff_x_ratio < 1.3]

            if len(args_cont_excluded):
                textregion_par = np.zeros_like(text_regions_p)
                textregion_par = cv2.fillPoly(textregion_par,
                                              pts=contours_only_text_parent[args_cont_included],
                                              color=1)
            else:
                textregion_par = (text_regions_p == 1).astype(np.uint8)

            textregion_par = cv2.erode(textregion_par, hor_kernel, iterations=2)
            textregion_par = cv2.dilate(textregion_par, ver_kernel, iterations=4)
            textregion_par = cv2.erode(textregion_par, hor_kernel, iterations=1)
            textregion_par = cv2.dilate(textregion_par, ver_kernel, iterations=5)
            textregion_par[text_regions_p > 1] = 0

            contours_only_dilated, hir_on_text_dilated = return_contours_of_image(textregion_par)
            contours_only_dilated = return_parent_contours(contours_only_dilated, hir_on_text_dilated)

            indexes_of_located_cont, _, cy_of_located = \
                self.return_indexes_of_contours_located_inside_another_list_of_contours(
                    contours_only_dilated,
                    cx_conts[args_cont_included],
                    cy_conts[args_cont_included],
                    args_cont_included)

            indexes_of_located_cont.extend(args_cont_excluded[:, np.newaxis])
            contours_only_dilated.extend(contours_only_text_parent[args_cont_excluded])

            missing_textregions = np.setdiff1d(args_cont, np.concatenate(indexes_of_located_cont))

            indexes_of_located_cont.extend(missing_textregions[:, np.newaxis])
            contours_only_dilated.extend(contours_only_text_parent[missing_textregions])

            args_cont_h = np.arange(len(contours_only_text_parent_h))
            indexes_of_located_cont.extend(args_cont_h[:, np.newaxis] +
                                           len(contours_only_text_parent))

            args_cont_drop = np.arange(len(polygons_of_drop_capitals))
            indexes_of_located_cont.extend(args_cont_drop[:, np.newaxis] +
                                           len(contours_only_text_parent) +
                                           len(contours_only_text_parent_h))

            co_text_all = contours_only_dilated
        else:
            co_text_all = list(contours_only_text_parent)

        img_poly = np.zeros_like(text_regions_p)
        img_poly[text_regions_p == label_text] = label_text
        img_poly[text_regions_p == label_head] = label_head
        img_poly[text_regions_p == 3] = label_imgs # rs: ??
        img_poly[text_regions_p == label_imgs] = label_imgs_ro
        img_poly[text_regions_p == label_marg] = label_marg_ro
        img_poly[text_regions_p == label_seps] = label_seps_ro

        img_header_and_sep = np.zeros_like(text_regions_p)
        for contour in contours_only_text_parent_h:
            # rs: why (max:max+12) instad of (min:max)?
            #     what about actual seps?
            img_header_and_sep[contour[:, 0, 1].max(): contour[:, 0, 1].max() + 12,
                               contour[:, 0, 0].min(): contour[:, 0, 0].max()] = 1
        co_text_all.extend(contours_only_text_parent_h)
        co_text_all.extend(polygons_of_drop_capitals)

        if not len(co_text_all):
            return []

        # fill polygons in lower resolution to be faster
        height, width = text_regions_p.shape
        labels_con = np.zeros((height // 6, width // 6, len(co_text_all)), dtype=bool)
        for i in range(len(co_text_all)):
            img = np.zeros(labels_con.shape[:2], dtype=np.uint8)
            cv2.fillPoly(img, pts=[co_text_all[i] // 6], color=1)
            labels_con[:, :, i] = img
        labels_con = resize_image(labels_con.astype(np.uint8), height_model, width_model).astype(bool)
        img_header_and_sep = resize_image(img_header_and_sep, height_model, width_model)
        img_poly = resize_image(img_poly, height_model, width_model)
        labels_con[img_poly == label_seps_ro] = 2
        labels_con[img_header_and_sep == 1] = 3
        labels_con = labels_con / 3.
        img_poly = img_poly / 5.

        input_1 = np.zeros((n_batch_inference, height_model, width_model, 3))
        ordered = [list(range(len(co_text_all)))]
        index_update = 0
        #print(labels_con.shape[2],"number of regions for reading order")
        while index_update>=0:
            ij_list = ordered.pop(index_update)
            i = ij_list.pop(0)

            ante_list = []
            post_list = []
            tot_counter = 0
            batch = []
            for j in ij_list:
                input_1[len(batch), :, :, 0] = labels_con[:, :, i]
                input_1[len(batch), :, :, 1] = img_poly
                input_1[len(batch), :, :, 2] = labels_con[:, :, j]

                tot_counter += 1
                batch.append(j)
                if tot_counter % n_batch_inference == 0 or tot_counter == len(ij_list):
                    y_pr = model.predict(input_1 , verbose=0)
                    for post_pr in y_pr:
                        if post_pr[0] >= 0.5:
                            post_list.append(j)
                        else:
                            ante_list.append(j)
                    batch = []

            if len(ante_list):
                ordered.insert(index_update, ante_list)
                index_update += 1
            ordered.insert(index_update, [i])
            if len(post_list):
                ordered.insert(index_update + 1, post_list)

            index_update = -1
            for index_next, ij_list in enumerate(ordered):
                if len(ij_list) > 1:
                    index_update = index_next
                    break

        ordered = [i[0] for i in ordered]

        if len(contours_only_text_parent) > min_cont_size_to_be_dilated:
            org_contours_indexes = []
            for i in ordered:
                if i < len(contours_only_dilated):
                    if i >= len(cy_of_located):
                        # excluded or missing dilated version of main region
                        org_contours_indexes.extend(indexes_of_located_cont[i])
                    else:
                        # reconstructed dilated version of main region
                        org_contours_indexes.extend(indexes_of_located_cont[i][
                            np.argsort(cy_of_located[i])])
                else:
                    # header or drop-capital region
                    org_contours_indexes.extend(indexes_of_located_cont[i])
            return org_contours_indexes
        else:
            return ordered

    def filter_contours_inside_a_bigger_one(self, contours, contours_d, shape,
                                            marginal_cnts=None, type_contour="textregion"):
        if type_contour == "textregion":
            areas = np.array(list(map(cv2.contourArea, contours)))
            areas = areas / float(np.prod(shape[:2]))
            cx_main, cy_main = find_center_of_contours(contours)

            contours = ensure_array(contours)
            indices_small = np.flatnonzero(areas < 1e-3)
            indices_large = np.flatnonzero(areas >= 1e-3)

            indices_drop = []
            for ind_small in indices_small:
                results = [cv2.pointPolygonTest(contours[ind_large],
                                                (cx_main[ind_small],
                                                 cy_main[ind_small]),
                                                False)
                           for ind_large in indices_large]
                results = np.array(results)
                if np.any(results == 1):
                    indices_drop.append(ind_small)
                elif marginal_cnts:
                    results = [cv2.pointPolygonTest(contour,
                                                    (cx_main[ind_small],
                                                     cy_main[ind_small]),
                                                    False)
                               for contour in marginal_cnts]
                    results = np.array(results)
                    if np.any(results == 1):
                        indices_drop.append(ind_small)

            contours = np.delete(contours, indices_drop, axis=0)
            if len(contours_d):
                contours_d = ensure_array(contours_d)
                contours_d = np.delete(contours_d, indices_drop, axis=0)

            return contours, contours_d

        else:
            contours_of_contours = []
            indexes_parent = []
            indexes_child = []
            for ind_region, textlines in enumerate(contours):
                contours_of_contours.extend(textlines)
                indexes_parent.extend([ind_region] * len(textlines))
                indexes_child.extend(list(range(len(textlines))))

            areas = np.array(list(map(cv2.contourArea, contours_of_contours)))
            cx, cy = find_center_of_contours(contours_of_contours)

            textline_in_textregion_index_to_del = {}
            for i in range(len(contours_of_contours)):
                args_other = np.setdiff1d(np.arange(len(contours_of_contours)), i)
                areas_other = areas[args_other]
                args_other_larger = args_other[areas_other > 1.5 * areas[i]]

                for ind in args_other_larger:
                    if cv2.pointPolygonTest(contours_of_contours[ind],
                                            (cx[i], cy[i]), False) == 1:
                        textline_in_textregion_index_to_del.setdefault(
                            indexes_parent[i], list()).append(
                                indexes_child[i])

            for where, which in textline_in_textregion_index_to_del.items():
                contours[where] = [line for idx, line in enumerate(contours[where])
                                   if idx not in which]

            return contours

    def return_indexes_of_contours_located_inside_another_list_of_contours(
            self, contours, centersx_loc, centersy_loc, indexes_loc):
        indexes = []
        centersx = []
        centersy = []
        for contour in contours:
            results = np.array([cv2.pointPolygonTest(contour, (px, py), False)
                                for px, py in zip(centersx_loc, centersy_loc)])
            indexes_in = (results == 0) | (results == 1)
            indexes.append(indexes_loc[indexes_in])
            centersx.append(centersx_loc[indexes_in])
            centersy.append(centersy_loc[indexes_in])

        return indexes, centersx, centersy

    def filter_contours_without_textline_inside(
            self, contours_par, contours_textline,
            contours_only_text_parent_d,
            conf_contours_textregions):

        assert len(contours_par) == len(contours_textline)
        indices = [ind for ind, lines in enumerate(contours_textline)
                   if len(lines)]
        def filterfun(lis):
            if len(lis) == 0:
                return []
            return [lis[ind] for ind in indices]

        return (filterfun(contours_par),
                filterfun(contours_textline),
                filterfun(contours_only_text_parent_d),
                filterfun(conf_contours_textregions),
        )

    def separate_marginals_to_left_and_right_and_order_from_top_to_down(
            self, polygons_of_marginals, all_found_textline_polygons_marginals, all_box_coord_marginals,
            slopes_marginals, conf_marginals, mid_point_of_page_width):
        cx_marg, cy_marg = find_center_of_contours(polygons_of_marginals)
        cx_marg = ensure_array(cx_marg)
        cy_marg = ensure_array(cy_marg)

        def split(lis):
            left, right = [], []
            for itm, prop in zip(lis, cx_marg < mid_point_of_page_width):
                (left if prop else right).append(itm)
            return left, right

        cy_marg_left, cy_marg_right = split(cy_marg)
        order_left = np.argsort(cy_marg_left)
        order_right = np.argsort(cy_marg_right)

        def splitsort(lis):
            left, right = split(lis)
            return [left[i] for i in order_left], [right[i] for i in order_right]

        return (*splitsort(polygons_of_marginals),
                *splitsort(all_found_textline_polygons_marginals),
                *splitsort(all_box_coord_marginals),
                *splitsort(slopes_marginals),
                *splitsort(conf_marginals))

    def run(self,
            overwrite: bool = False,
            image_filename: Optional[str] = None,
            dir_in: Optional[str] = None,
            dir_out: Optional[str] = None,
            dir_of_cropped_images: Optional[str] = None,
            dir_of_layout: Optional[str] = None,
            dir_of_deskewed: Optional[str] = None,
            dir_of_all: Optional[str] = None,
            dir_save_page: Optional[str] = None,
            num_jobs: int = 0,
            halt_fail: float = 0,
    ):
        """
        Get image and scales, then extract the page of scanned image
        """
        self.logger.debug("enter run")
        t0_tot = time.time()

        # Log enabled features directly
        enabled_modes = []
        if self.full_layout:
            enabled_modes.append("Full layout analysis")
        if self.tables:
            enabled_modes.append("Table detection")
        if enabled_modes:
            self.logger.info("Enabled modes: " + ", ".join(enabled_modes))
        if self.enable_plotting:
            self.logger.info("Saving debug plots")
            if dir_of_cropped_images:
                self.logger.info(f"Saving cropped images to: {dir_of_cropped_images}")
            if dir_of_layout:
                self.logger.info(f"Saving layout plots to: {dir_of_layout}")
            if dir_of_deskewed:
                self.logger.info(f"Saving deskewed images to: {dir_of_deskewed}")
            self.plotter = EynollahPlotter(
                dir_out=dir_out,
                dir_of_all=dir_of_all,
                dir_save_page=dir_save_page,
                dir_of_deskewed=dir_of_deskewed,
                dir_of_cropped_images=dir_of_cropped_images,
                dir_of_layout=dir_of_layout)
        else:
            self.plotter = None

        if dir_in:
            ls_imgs = [os.path.join(dir_in, image_filename)
                       for image_filename in filter(is_image_filename,
                                                    os.listdir(dir_in))]
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
                                    overwrite=overwrite,
                                    logq=logq)] = img_filename, logq
                for job in as_completed(list(jobs)):
                    img_filename, logq = jobs[job]
                    loglistener = logging.handlers.QueueListener(
                        logq, *self.logger.handlers, respect_handler_level=False)
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
            # for img_filename, result in zip(ls_imgs, results) ...
            self.logger.info("%d of %d jobs successful", n_success, len(jobs))
            self.logger.info("All jobs done in %.1fs", time.time() - t0_tot)
        elif image_filename:
            try:
                self.run_single(image_filename, dir_out=dir_out, overwrite=overwrite)
            except:
                self.logger.exception("Job failed")
        else:
            raise ValueError("run requires either a single image filename or a directory")

        if self.enable_plotting:
            del self.plotter

    def run_single(self,
                   img_filename: str,
                   dir_out: Optional[str] = None,
                   overwrite: bool = False,
                   img_pil=None,
                   pcgts=None,
    ) -> None:
        t0 = time.time()
        self.logger.info(img_filename)

        image = self.cache_images(image_filename=img_filename, image_pil=img_pil)
        writer = EynollahXmlWriter(
            dir_out=dir_out,
            image_filename=img_filename,
            image_width=image['img'].shape[1],
            image_height=image['img'].shape[0],
            curved_line=self.curved_line,
            pcgts=pcgts)

        if os.path.exists(writer.output_filename):
            if overwrite:
                self.logger.warning("will overwrite existing output file '%s'", writer.output_filename)
            else:
                self.logger.warning("will skip input for existing output file '%s'", writer.output_filename)
                return

        self.logger.info(f"Processing file: {writer.image_filename}")
        self.logger.info("Step 1/5: Image Enhancement")

        num_col_classifier, num_column_is_classified = \
            self.run_enhancement(image)
        writer.scale_x = image['scale_x']
        writer.scale_y = image['scale_y']

        self.logger.info(f"Image: {image['img_res'].shape[1]}x{image['img_res'].shape[0]}, "
                         f"scale {image['scale_x']:.1f}x{image['scale_y']:.1f}, "
                         f"{image['dpi']} DPI, {num_col_classifier} columns")
        self.logger.info(f"Enhancement complete ({time.time() - t0:.1f}s)")

        # Basic Processing Mode
        if self.skip_layout_and_reading_order:
            self.logger.info("Step 2/5: Basic Processing Mode")
            self.logger.info("Skipping layout analysis and reading order detection")

            _ ,_, _, _, textline_mask_tot_ea, _, _ = \
                self.get_early_layout(image, num_col_classifier)

            page_coord, image_page, textline_mask_tot_ea, cont_page = \
                self.run_graphics_and_columns_without_layout(textline_mask_tot_ea, image)

            ##all_found_textline_polygons =self.scale_contours_new(textline_mask_tot_ea)

            cnt_clean_rot_raw, hir_on_cnt_clean_rot = return_contours_of_image(textline_mask_tot_ea)
            all_found_textline_polygons = filter_contours_area_of_image(
                textline_mask_tot_ea, cnt_clean_rot_raw, hir_on_cnt_clean_rot, max_area=1, min_area=0.00001)

            cx_main_tot, cy_main_tot = find_center_of_contours(all_found_textline_polygons)
            w_h_textlines = [cv2.boundingRect(polygon)[2:]
                             for polygon in all_found_textline_polygons]
            w_h_textlines = [w / float(h) for w, h in w_h_textlines]

            all_found_textline_polygons = self.get_textlines_of_a_textregion_sorted(
                #all_found_textline_polygons[::-1]
                all_found_textline_polygons, cx_main_tot, cy_main_tot, w_h_textlines)
            all_found_textline_polygons = [ all_found_textline_polygons ]
            all_found_textline_polygons = dilate_textline_contours(all_found_textline_polygons)
            all_found_textline_polygons = self.filter_contours_inside_a_bigger_one(
                all_found_textline_polygons, None, None, type_contour="textline")

            order_text_new = [0]
            slopes =[0]
            conf_contours_textregions =[0]

            pcgts = writer.build_pagexml_no_full_layout(
                found_polygons_text_region=cont_page,
                page_coord=page_coord,
                order_of_texts=order_text_new,
                all_found_textline_polygons=all_found_textline_polygons,
                all_box_coord=page_coord,
                found_polygons_images=[],
                found_polygons_tables=[],
                found_polygons_marginals_left=[],
                found_polygons_marginals_right=[],
                all_found_textline_polygons_marginals_left=[],
                all_found_textline_polygons_marginals_right=[],
                all_box_coord_marginals_left=[],
                all_box_coord_marginals_right=[],
                slopes=slopes,
                slopes_marginals_left=[],
                slopes_marginals_right=[],
                cont_page=cont_page,
                polygons_seplines=[],
                skip_layout_reading_order=True
            )
            self.logger.info("Basic processing complete")
            writer.write_pagexml(pcgts)
            self.logger.info("Job done in %.1fs", time.time() - t0)
            return

        #print("text region early -1 in %.1fs", time.time() - t0)
        t1 = time.time()
        self.logger.info("Step 2/5: Layout Analysis")

        (text_regions_p_1,
         erosion_hurts,
         polygons_seplines,
         polygons_text_early,
         textline_mask_tot_ea,
         regions_confidence,
         textline_confidence) = self.get_early_layout(image, num_col_classifier)
        #print("text region early -2 in %.1fs", time.time() - t0)

        if num_col_classifier == 1 or num_col_classifier ==2:
            if num_col_classifier == 1:
                img_w_new = 1000
            else:
                img_w_new = 1300
            img_h_new = img_w_new * textline_mask_tot_ea.shape[0] // textline_mask_tot_ea.shape[1]

            textline_mask_tot_ea_deskew = resize_image(textline_mask_tot_ea,img_h_new, img_w_new )
            slope_deskew = self.run_deskew(textline_mask_tot_ea_deskew)
        else:
            slope_deskew = self.run_deskew(textline_mask_tot_ea)
        if self.plotter:
            self.plotter.save_deskewed_image(slope_deskew, image['img'], image['name'])
        #print("text region early -2,5 in %.1fs", time.time() - t0)
        #self.logger.info("Textregion detection took %.1fs ", time.time() - t1t)
        (num_col, num_col_classifier,
         page_coord, image_page, cont_page,
         text_regions_p_1, table_prediction, textline_mask_tot_ea,
         regions_confidence, table_confidence, textline_confidence) = \
                self.run_graphics_and_columns(text_regions_p_1, textline_mask_tot_ea,
                                              regions_confidence, textline_confidence,
                                              num_col_classifier, num_column_is_classified,
                                              erosion_hurts, image)
        #self.logger.info("run graphics %.1fs ", time.time() - t1t)
        #print("text region early -3 in %.1fs", time.time() - t0)
        textline_mask_tot_ea_org = np.copy(textline_mask_tot_ea)

        #plt.imshow(table_prediction)
        #plt.show()
        self.logger.info(f"Layout analysis complete ({time.time() - t1:.1f}s)")

        if not num_col and len(polygons_text_early) == 0:
            self.logger.info("No columns detected - generating empty PAGE-XML")

            pcgts = writer.build_pagexml_no_full_layout(
                found_polygons_text_region=[],
                page_coord=page_coord,
                order_of_texts=[],
                all_found_textline_polygons=[],
                all_box_coord=[],
                found_polygons_images=[],
                found_polygons_tables=[],
                found_polygons_marginals_left=[],
                found_polygons_marginals_right=[],
                all_found_textline_polygons_marginals_left=[],
                all_found_textline_polygons_marginals_right=[],
                all_box_coord_marginals_left=[],
                all_box_coord_marginals_right=[],
                slopes=[],
                slopes_marginals_left=[],
                slopes_marginals_right=[],
                cont_page=cont_page,
                polygons_seplines=[],
            )
            writer.write_pagexml(pcgts)
            self.logger.info("Job done in %.1fs", time.time() - t0)
            return

        #print("text region early in %.1fs", time.time() - t0)
        t1 = time.time()
        if num_col_classifier in (1,2):
            org_h_l_m = textline_mask_tot_ea.shape[0]
            org_w_l_m = textline_mask_tot_ea.shape[1]
            if num_col_classifier == 1:
                img_w_new = 2000
            else:
                img_w_new = 2400
            img_h_new = img_w_new * textline_mask_tot_ea.shape[0] // textline_mask_tot_ea.shape[1]

            image_page = resize_image(image_page, img_h_new, img_w_new)
            textline_mask_tot_ea = resize_image(textline_mask_tot_ea, img_h_new, img_w_new)
            text_regions_p_1 = resize_image(text_regions_p_1, img_h_new, img_w_new)
            table_prediction = resize_image(table_prediction, img_h_new, img_w_new)

        text_regions_p = \
            self.run_marginals(num_col_classifier, slope_deskew, text_regions_p_1, table_prediction)

        if self.plotter:
            self.plotter.save_plot_of_layout_main_all(text_regions_p, image_page, image['name'])
            self.plotter.save_plot_of_layout_main(text_regions_p, image_page, image['name'])

        label_text = 1
        label_imgs = 2
        label_imgs_fl = 5
        label_seps = 3
        label_seps_fl = 6
        label_marg = 4
        label_marg_fl = 8
        label_drop_fl = 4
        label_tabs = 10
        if image_page.size:
            # if ratio of text regions to page area is smaller that 30%,
            # then deskew angle will not be allowed to exceed 45
            if (abs(slope_deskew) > 45 and
                ((text_regions_p == label_text).sum() +
                 (text_regions_p == label_marg).sum()) <=
                0.3 * image_page.size):
                slope_deskew = 0

        # if there is no main text, then relabel marginalia as main
        if not np.any(text_regions_p == label_text):
            text_regions_p[text_regions_p == label_marg] = label_text

        self.logger.info("Step 3/5: Text Line Detection")

        if self.curved_line:
            self.logger.info("Mode: Curved line detection")

        if num_col_classifier in (1,2):
            image_page = resize_image(image_page, org_h_l_m, org_w_l_m)
            textline_mask_tot_ea = resize_image(textline_mask_tot_ea, org_h_l_m, org_w_l_m)
            text_regions_p = resize_image(text_regions_p, org_h_l_m, org_w_l_m)
            text_regions_p_1 = resize_image(text_regions_p_1, org_h_l_m, org_w_l_m)
            table_prediction = resize_image(table_prediction, org_h_l_m, org_w_l_m)

        self.logger.info(f"Detection of marginals took {time.time() - t1:.1f}s")
        t1 = time.time()
        regions_fully, regionsfl_confidence, regions_without_separators = \
            self.get_full_layout(image_page,
                                 textline_mask_tot_ea,
                                 text_regions_p,
                                 num_col_classifier,
                                 table_prediction)

        (text_regions_p_d,
         textline_mask_tot_ea_d,
         regions_without_separators_d) = self.get_deskewed_masks(
             slope_deskew,
             text_regions_p,
             textline_mask_tot_ea,
             regions_without_separators)

        min_area_mar = 0.00001
        marginal_mask = (text_regions_p == label_marg_fl).astype(np.uint8)
        marginal_mask = cv2.dilate(marginal_mask, KERNEL, iterations=2)
        polygons_of_marginals = return_contours_of_interested_region(marginal_mask, 1,
                                                                     min_area_mar)
        polygons_of_tables = return_contours_of_interested_region(text_regions_p, label_tabs,
                                                                  min_area_mar)
        polygons_of_images = return_contours_of_interested_region(text_regions_p, label_imgs_fl)
        conf_marginals = get_region_confidences(polygons_of_marginals, regions_confidence)
        conf_images = get_region_confidences(polygons_of_images, regions_confidence)
        conf_tables = get_region_confidences(polygons_of_tables, table_confidence)

        if self.full_layout:
            textline_mask_tot_ea_org[text_regions_p == label_drop_fl] = 0
            polygons_of_drop_capitals = return_contours_of_interested_region(text_regions_p,
                                                                             label_drop_fl,
                                                                             min_area=0.00003)
            conf_drops = get_region_confidences(polygons_of_drop_capitals, regionsfl_confidence)

        polygons_of_textregions = return_contours_of_interested_region(text_regions_p, label_text,
                                                                       min_area=MIN_AREA_REGION)
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            polygons_of_textregions_d = return_contours_of_interested_region(text_regions_p_d, label_text,
                                                                             min_area=MIN_AREA_REGION)
            if (len(polygons_of_textregions) and
                len(polygons_of_textregions_d)):
                polygons_of_textregions_d = \
                    match_deskewed_contours(
                        slope_deskew,
                        polygons_of_textregions,
                        polygons_of_textregions_d,
                        text_regions_p.shape,
                        text_regions_p_d.shape)
        else:
            polygons_of_textregions_d = []
        (polygons_of_textregions,
         polygons_of_textregions_d) = self.filter_contours_inside_a_bigger_one(
             polygons_of_textregions,
             polygons_of_textregions_d,
             text_regions_p.shape,
             marginal_cnts=polygons_of_marginals)
        polygons_of_textregions = dilate_textregion_contours(polygons_of_textregions)
        conf_textregions = get_region_confidences(polygons_of_textregions, regions_confidence)

        if not len(polygons_of_textregions):
            polygons_of_textregions = polygons_of_marginals
            polygons_of_marginals = []
            conf_textregions = conf_marginals
            conf_marginals = []

        #print("text region early 4 in %.1fs", time.time() - t0)
        boxes_text = get_text_region_boxes_by_given_contours(polygons_of_textregions)
        boxes_marginals = get_text_region_boxes_by_given_contours(polygons_of_marginals)
        #print("text region early 5 in %.1fs", time.time() - t0)
        ## birdan sora chock chakir
        if not self.curved_line:
            all_found_textline_polygons, \
                all_box_coord, slopes = self.get_slopes_and_deskew_new_light2(
                    polygons_of_textregions, textline_mask_tot_ea_org,
                    boxes_text, slope_deskew)
            all_found_textline_polygons_marginals, \
                all_box_coord_marginals, slopes_marginals = self.get_slopes_and_deskew_new_light2(
                    polygons_of_marginals, textline_mask_tot_ea_org,
                    boxes_marginals, slope_deskew)

            all_found_textline_polygons = dilate_textline_contours(
                all_found_textline_polygons)
            all_found_textline_polygons = self.filter_contours_inside_a_bigger_one(
                all_found_textline_polygons, None, None, type_contour="textline")
            all_found_textline_polygons_marginals = dilate_textline_contours(
                all_found_textline_polygons_marginals)
        else:
            scale_param = 1
            textline_mask_tot_ea_erode = cv2.erode(textline_mask_tot_ea, kernel=KERNEL, iterations=2)
            all_found_textline_polygons, \
                all_box_coord, slopes = self.get_slopes_and_deskew_new_curved(
                    polygons_of_textregions, textline_mask_tot_ea_erode,
                    boxes_text,
                    num_col_classifier, scale_param, slope_deskew, image['name'])
            all_found_textline_polygons = small_textlines_to_parent_adherence2(
                all_found_textline_polygons, textline_mask_tot_ea, num_col_classifier)
            all_found_textline_polygons_marginals, \
                all_box_coord_marginals, slopes_marginals = self.get_slopes_and_deskew_new_curved(
                    polygons_of_marginals, textline_mask_tot_ea_erode,
                    boxes_marginals,
                    num_col_classifier, scale_param, slope_deskew, image['name'])
            all_found_textline_polygons_marginals = small_textlines_to_parent_adherence2(
                all_found_textline_polygons_marginals, textline_mask_tot_ea, num_col_classifier)
        (polygons_of_textregions, all_found_textline_polygons,
         polygons_of_textregions_d, conf_textregions) = \
            self.filter_contours_without_textline_inside(
                polygons_of_textregions, all_found_textline_polygons,
                polygons_of_textregions_d, conf_textregions)

        (polygons_of_marginals_left,
         polygons_of_marginals_right,
         all_found_textline_polygons_marginals_left,
         all_found_textline_polygons_marginals_right,
         all_box_coord_marginals_left,
         all_box_coord_marginals_right,
         slopes_marginals_left,
         slopes_marginals_right,
         conf_marginals_left,
         conf_marginals_right) = \
             self.separate_marginals_to_left_and_right_and_order_from_top_to_down(
                 polygons_of_marginals,
                 all_found_textline_polygons_marginals,
                 all_box_coord_marginals,
                 slopes_marginals,
                 conf_marginals,
                 0.5 * text_regions_p.shape[1])
        # FIXME: get_region_confidences w/ textline_confidence on all types of textlines...

        #print(len(polygons_of_marginals), len(ordered_left_marginals), len(ordered_right_marginals), 'marginals ordred')

        if self.full_layout:
            (text_regions_p,
             polygons_of_textregions,
             polygons_of_textregions_h,
             polygons_of_textregions_d,
             polygons_of_textregions_h_d,
             all_box_coord,
             all_box_coord_h,
             all_found_textline_polygons,
             all_found_textline_polygons_h,
             slopes,
             slopes_h,
             conf_textregions,
             conf_textregions_h) = split_textregion_main_vs_head(
                 text_regions_p,
                 regions_fully,
                 polygons_of_textregions,
                 polygons_of_textregions_d,
                 all_box_coord,
                 all_found_textline_polygons,
                 slopes,
                 conf_textregions)

            if self.plotter:
                self.plotter.save_plot_of_layout(text_regions_p, image_page, image['name'])
                self.plotter.save_plot_of_layout_all(text_regions_p, image_page, image['name'])
            ##all_found_textline_polygons = adhere_drop_capital_region_into_corresponding_textline(
                ##text_regions_p, polygons_of_drop_capitals, polygons_of_textregions, polygons_of_textregions_h,
                ##all_box_coord, all_box_coord_h, all_found_textline_polygons, all_found_textline_polygons_h,
                ##kernel=KERNEL, curved_line=self.curved_line)
        else:
            polygons_of_drop_capitals = []
            polygons_of_textregions_h = []
            polygons_of_textregions_h_d = []

        if not self.reading_order_machine_based:
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                boxes = self.run_boxes_order(text_regions_p, num_col_classifier, erosion_hurts,
                                             regions_without_separators,
                                             contours_h=(None if self.headers_off or not self.full_layout
                                                         else polygons_of_textregions_h))
            else:
                boxes_d = self.run_boxes_order(text_regions_p_d, num_col_classifier, erosion_hurts,
                                               regions_without_separators_d,
                                               contours_h=(None if self.headers_off or not self.full_layout
                                                           else polygons_of_textregions_h_d))

        if self.plotter:
            self.plotter.write_images_into_directory(polygons_of_images, image_page,
                                                     image['scale_x'], image['scale_y'], image['name'])
        t_order = time.time()

        self.logger.info("Step 4/5: Reading Order Detection")
        if self.right2left:
            self.logger.info("Right-to-left mode enabled")
        if self.headers_off:
            self.logger.info("Headers ignored in reading order")

        if self.reading_order_machine_based:
            self.logger.info("Using machine-based detection")
            order_text_new = self.do_order_of_regions_with_model(
                polygons_of_textregions,
                polygons_of_textregions_h,
                polygons_of_drop_capitals,
                text_regions_p)
        else:
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                order_text_new = self.do_order_of_regions(
                    polygons_of_textregions,
                    polygons_of_textregions_h,
                    polygons_of_drop_capitals,
                    boxes, textline_mask_tot_ea)
            else:
                order_text_new = self.do_order_of_regions(
                    polygons_of_textregions_d,
                    polygons_of_textregions_h_d,
                    polygons_of_drop_capitals,
                    boxes_d, textline_mask_tot_ea_d)
        self.logger.info(f"Detection of reading order took {time.time() - t_order:.1f}s")

        self.logger.info("Step 5/5: Output Generation")

        if self.full_layout:
            pcgts = writer.build_pagexml_full_layout(
                found_polygons_text_region=polygons_of_textregions,
                found_polygons_text_region_h=polygons_of_textregions_h,
                page_coord=page_coord,
                order_of_texts=order_text_new,
                all_found_textline_polygons=all_found_textline_polygons,
                all_found_textline_polygons_h=all_found_textline_polygons_h,
                all_box_coord=all_box_coord,
                all_box_coord_h=all_box_coord_h,
                found_polygons_images=polygons_of_images,
                found_polygons_tables=polygons_of_tables,
                found_polygons_drop_capitals=polygons_of_drop_capitals,
                found_polygons_marginals_left=polygons_of_marginals_left,
                found_polygons_marginals_right=polygons_of_marginals_right,
                all_found_textline_polygons_marginals_left=all_found_textline_polygons_marginals_left,
                all_found_textline_polygons_marginals_right=all_found_textline_polygons_marginals_right,
                all_box_coord_marginals_left=all_box_coord_marginals_left,
                all_box_coord_marginals_right=all_box_coord_marginals_right,
                slopes=slopes,
                slopes_h=slopes_h,
                slopes_marginals_left=slopes_marginals_left,
                slopes_marginals_right=slopes_marginals_right,
                cont_page=cont_page,
                polygons_seplines=polygons_seplines,
                conf_textregions=conf_textregions,
                conf_textregions_h=conf_textregions_h,
                conf_marginals_left=conf_marginals_left,
                conf_marginals_right=conf_marginals_right,
                conf_images=conf_images,
                conf_tables=conf_tables,
                conf_drops=conf_drops,
            )
        else:
            pcgts = writer.build_pagexml_no_full_layout(
                found_polygons_text_region=polygons_of_textregions,
                page_coord=page_coord,
                order_of_texts=order_text_new,
                all_found_textline_polygons=all_found_textline_polygons,
                all_box_coord=all_box_coord,
                found_polygons_images=polygons_of_images,
                found_polygons_tables=polygons_of_tables,
                found_polygons_marginals_left=polygons_of_marginals_left,
                found_polygons_marginals_right=polygons_of_marginals_right,
                all_found_textline_polygons_marginals_left=all_found_textline_polygons_marginals_left,
                all_found_textline_polygons_marginals_right=all_found_textline_polygons_marginals_right,
                all_box_coord_marginals_left=all_box_coord_marginals_left,
                all_box_coord_marginals_right=all_box_coord_marginals_right,
                slopes=slopes,
                slopes_marginals_left=slopes_marginals_left,
                slopes_marginals_right=slopes_marginals_right,
                cont_page=cont_page,
                polygons_seplines=polygons_seplines,
                conf_textregions=conf_textregions,
                conf_marginals_left=conf_marginals_left,
                conf_marginals_right=conf_marginals_right,
                conf_images=conf_images,
                conf_tables=conf_tables,
            )

        writer.write_pagexml(pcgts)
        self.logger.info("Job done in %.1fs", time.time() - t0)
        return
