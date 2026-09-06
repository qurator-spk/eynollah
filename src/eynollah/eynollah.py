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

from __future__ import annotations
import logging
import logging.handlers
import sys

import os
import time
from itertools import compress
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None # type: ignore

from .model_zoo import EynollahModelZoo
from .utils.contour import (
    find_center_of_contours,
    find_new_features_of_contours,
    find_features_of_contours,
    get_region_confidences,
    return_contours_of_class,
    match_deskewed_contours,
    estimate_skew_contours,
)
from .utils.rotate import rotate_image
from .utils.separate_lines import (
    return_deskew_slop,
    do_work_of_slopes_new_curved,
)
from .utils.marginals import get_marginals
from .utils.resize import resize_image
from .utils.tiling import do_prediction, do_prediction_new_concept
from .utils import (
    Region,
    TextRegion,
    ensure_array,
    pairwise,
    itemgetter,
    is_image_filename,
    isNaN,
    crop_image_inside_box,
    box2slice,
    find_num_col,
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


_instance: Eynollah | None = None
def _set_instance(instance: Eynollah):
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
        allow_marginalia : str = "both",
        threshold_art_class_layout: float = 0.1,
        threshold_art_class_textline: float = 0.1,
        skip_layout_and_reading_order : bool = False,
        num_jobs : int = 0,
        logger : logging.Logger | None = None,
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
        self.allow_marginalia = allow_marginalia
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
        # ruff: disable[SIM114] (clearer as is)
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
        # ruff: enable[SIM114]
        img_h_new = img_w_new * img.shape[0] // img.shape[1]

        if conf_col < 0.9 and img_w_new < width_early:
            # don't downsample if unconfident
            img_new = np.copy(img)
            img_is_resized = False
        #elif conf_col < 0.8 and img_h_new >= 8000:
        elif conf_col < 0.9 and img_h_new >= 8000:
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
        elif conf_col < 0.9 and img_h_new >= 8000:
            # don't upsample if too large
            img_new = np.copy(img)
            img_is_resized = False
        else:
            img_new = resize_image(img, img_h_new, img_w_new)
            img_is_resized = True

        return img_new, img_is_resized

    # FIXME: actually may run enhancement model, should be renamed
    def resize_image_with_column_classifier(self, image):
        self.logger.debug("enter resize_image_with_column_classifier")
        img = self.imread(image, binary=self.input_binary)

        width_early = img.shape[1]
        page_img, page_coord = self.early_page_for_num_of_column_classification(img)

        label_p_pred = np.ones(6)
        conf_col = 1.0
        if self.num_col_upper and not self.num_col_lower:
            num_col = self.num_col_upper
        elif self.num_col_lower and not self.num_col_upper:
            num_col = self.num_col_lower
        elif (not self.num_col_upper and not self.num_col_lower or
              self.num_col_upper != self.num_col_lower):
            if self.input_binary:
                img_in = page_img
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

        self.logger.info("Found %s columns (%s)", num_col, np.around(label_p_pred, decimals=5))
        if num_col in (1, 2):
            fun = self.calculate_width_height_by_columns_1_2
        else:
            fun = self.calculate_width_height_by_columns
        img_new, _ = fun(img, num_col, conf_col, width_early)

        if img_new.shape[1] > img.shape[1]:
            img_new = do_prediction(img_new, self.model_zoo.get("enhancement"),
                                    patches=True,
                                    logger=self.logger,
                                    marginal_of_patch_percent=0,
                                    n_batch_inference=3,
                                    is_enhancement=True)
            self.logger.info("Enhancement applied")

        image['img_res'] = img_new
        image['scale_y'] = 1.0 * img_new.shape[0] / img.shape[0]
        image['scale_x'] = 1.0 * img_new.shape[1] / img.shape[1]

    # FIXME: does not actually run enhancement model, should be renamed
    def resize_and_enhance_image_with_column_classifier(self, image):
        self.logger.debug("enter resize_and_enhance_image_with_column_classifier")
        dpi = image['dpi']
        img = self.imread(image)
        self.logger.info("Detected %s DPI", dpi)
        if self.input_binary:
            prediction_bin = do_prediction(img, self.model_zoo.get("binarization"),
                                           patches=True,
                                           logger=self.logger,
                                           n_batch_inference=5)
            prediction_bin = 255 * (prediction_bin == 0)
            prediction_bin = np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            image['img_bin_uint8'] = prediction_bin
            img = np.copy(prediction_bin)
        else:
            image['img_bin_uint8'] = None

        width_early = img.shape[1]
        t1 = time.time()
        page_img, page_coord = self.early_page_for_num_of_column_classification(img)

        label_p_pred = np.ones(6)
        conf_col = 1.0
        if self.num_col_upper and not self.num_col_lower:
            num_col = self.num_col_upper
        elif self.num_col_lower and not self.num_col_upper:
            num_col = self.num_col_lower
        elif (not self.num_col_upper and not self.num_col_lower or
              self.num_col_upper != self.num_col_lower):
            if self.input_binary:
                img_in = page_img
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
        image['img_res'] = img_res.astype(np.uint8)
        image['scale_y'] = 1.0 * img_res.shape[0] / img.shape[0]
        image['scale_x'] = 1.0 * img_res.shape[1] / img.shape[1]
        return is_image_enhanced, num_col, is_image_resized

    def extract_page(self, image):
        page_cropped = img = image['img_res']
        h, w = img.shape[:2]
        page_cont = np.array([[[0, 0]],
                              [[w, 0]],
                              [[w, h]],
                              [[0, h]]])
        page_mask = np.ones((h, w), dtype=np.uint8)
        if not self.ignore_page_extraction:
            self.logger.debug("enter extract_page")
            #cv2.GaussianBlur(img, (5, 5), 0)
            prediction = do_prediction(img, self.model_zoo.get("page"),
                                       patches=False,
                                       logger=self.logger)
            contours, _ = cv2.findContours(prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours):
                areas = np.array(list(map(cv2.contourArea, contours)))
                page_cont = contours[np.argmax(areas)]
                box = (x, y, w, h) = cv2.boundingRect(page_cont)
                page_cropped = img[box2slice(box)]
                page_mask = np.zeros((h, w), dtype=np.uint8)
                page_mask = cv2.fillPoly(page_mask, pts=[page_cont - [x, y]], color=1)
            self.logger.debug("exit extract_page")
        return page_cont, page_cropped, page_mask

    def early_page_for_num_of_column_classification(self, img):
        if not self.ignore_page_extraction:
            self.logger.debug("enter early_page_for_num_of_column_classification")
            img2 = cv2.GaussianBlur(img, (5, 5), 0)
            prediction = do_prediction(img2, self.model_zoo.get("page"),
                                       patches=False,
                                       logger=self.logger)
            prediction = cv2.dilate(prediction, KERNEL, iterations=3)
            contours, _ = cv2.findContours(prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours):
                areas = np.array(list(map(cv2.contourArea, contours)))
                cnt = contours[np.argmax(areas)]
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

        prediction_regions, confidence_regions = do_prediction_new_concept(
            img, self.model_zoo.get("region_fl" if patches else "region_fl_np"),
            patches=patches,
            logger=self.logger,
            n_batch_inference=1,
            thresholding_for_heading=not patches)

        self.logger.debug("exit extract_text_regions_new")
        return prediction_regions, confidence_regions

    def extract_text_regions(self, img, patches, cols):
        self.logger.debug("enter extract_text_regions")
        img_height_h = img.shape[0]
        img_width_h = img.shape[1]
        model_region = self.model_zoo.get("region_fl" if patches else "region_fl_np")

        prediction_regions = do_prediction(img, model_region,
                                           patches=patches,
                                           logger=self.logger,
                                           marginal_of_patch_percent=0.1)
        prediction_regions = resize_image(prediction_regions, img_height_h, img_width_h)
        self.logger.debug("exit extract_text_regions")
        return prediction_regions

    def get_textlines_of_a_textregion_sorted(
            self,
            textlines_cont: list[np.ndarray],
            textlines_conf: list[float],
            textlines_cx: list[float],
            textlines_cy: list[float],
            textlines_w_h: list[tuple[int, int]],
    ):
        N = len(textlines_cont)
        if N <= 1:
            return textlines_cont, textlines_conf

        textlines_cx = np.array(textlines_cx)
        textlines_cy = np.array(textlines_cy)
        diff_cx = np.abs(np.diff(np.sort(textlines_cx)))
        diff_cy = np.abs(np.diff(np.sort(textlines_cy)))

        if N > 1:
            mean_y_diff = np.median(diff_cy)
            mean_x_diff = np.median(diff_cx)
            count_hor = np.count_nonzero(-np.diff(textlines_w_h, axis=1) > 0)
            count_ver = N - count_hor
        else:
            mean_y_diff = 0
            mean_x_diff = 0
            count_hor = 1
            count_ver = 0

        sorted_textlines_cont = []
        sorted_textlines_conf = []
        if count_hor >= count_ver:
            row_threshold = mean_y_diff / 1.5  if mean_y_diff > 0 else 10
            rows = []
            for prev_idx, curr_idx in pairwise(np.argsort(textlines_cy)):
                if not len(rows):
                    rows.append([prev_idx])
                if abs(textlines_cy[curr_idx] - textlines_cy[prev_idx]) <= row_threshold:
                    rows[-1].append(curr_idx)
                else:
                    rows.append([curr_idx])

            for row in rows:
                for idx in np.argsort(textlines_cx[row]):
                    sorted_textlines_cont.append(textlines_cont[row[idx]])
                    sorted_textlines_conf.append(textlines_conf[row[idx]])

        else:
            col_threshold = mean_x_diff / 1.5 if mean_x_diff > 0 else 10
            cols = []
            for prev_idx, curr_idx in pairwise(np.argsort(textlines_cx)):
                if not len(cols):
                    cols.append([prev_idx])
                if abs(textlines_cx[curr_idx] - textlines_cx[prev_idx]) <= col_threshold:
                    cols[-1].append(curr_idx)
                else:
                    cols.append([curr_idx])

            for col in cols:
                for idx in np.argsort(textlines_cy[col]):
                    sorted_textlines_cont.append(textlines_cont[col[idx]])
                    sorted_textlines_conf.append(textlines_conf[col[idx]])

        return sorted_textlines_cont, sorted_textlines_conf

    def get_slopes_and_deskew_new_light2(
            self, parents: list[TextRegion],
            textline_mask_tot: np.ndarray,
            textline_confidence: np.ndarray,
            slope_deskew: float
    ):
        textlines_cont = return_contours_of_class(textline_mask_tot, 1, 1e-4)
        textlines_conf = get_region_confidences(textlines_cont, textline_confidence)
        textlines_cx, textlines_cy = find_center_of_contours(textlines_cont)
        textlines_w_h = [cv2.boundingRect(polygon)[2:] for polygon in textlines_cont]
        textlines_args = np.arange(len(textlines_cont))

        # search for unbalanced shapes (center not in contour itself)
        for ind, cont in enumerate(textlines_cont):
            if (not cv2.isContourConvex(cont) and
                cv2.pointPolygonTest(cont,
                                     (textlines_cx[ind],
                                      textlines_cy[ind]),
                                     False) < 0):
                # find new representative point instead of center:
                # fit all points to straight line to pick from
                vx, vy, x, y = cv2.fitLine(cont, cv2.DIST_L2, 0, 0.01, 0.01)[:, 0]
                path = np.arange(textlines_w_h[ind][0] // 2)
                for dist in list(path) + list(-path):
                    new_cx = textlines_cx[ind] + dist
                    new_cy = int((new_cx - x) * vy / vx + y)
                    if cv2.pointPolygonTest(cont, (new_cx, new_cy), False) >= 0:
                        textlines_cx[ind] = new_cx
                        textlines_cy[ind] = new_cy
                        break

        for index, parent in enumerate(parents):
            results = [cv2.pointPolygonTest(parent.contour,
                                            (textlines_cx[ind],
                                             textlines_cy[ind]),
                                            False)
                       for ind in textlines_args]
            results = np.array(results)
            indexes_in = textlines_args[results >= 0]
            get_in = itemgetter(indexes_in)
            textlines_in_cont, textlines_in_conf = self.get_textlines_of_a_textregion_sorted(
                get_in(textlines_cont), get_in(textlines_conf),
                get_in(textlines_cx), get_in(textlines_cy),
                get_in(textlines_w_h))

            parent.lines = [Region(cont, conf=conf) #[::-1]
                            for cont, conf in zip(textlines_in_cont, textlines_in_conf)]

            try:
                parent.skew = estimate_skew_contours(textlines_in_cont)
            except ValueError:
                parent.skew = slope_deskew
            # plt.imshow(textline_mask_tot)
            # for i, contour in enumerate(textlines_in_cont):
            #     plt.plot(*contour[:, 0].T, linewidth=2, color='green')
            #     plt.text(*contour[:, 0].mean(axis=0), str(i + 1))
            # if not len(textlines_in_cont):
            #     plt.plot(*parent.contour[:, 0].T, linewidth=2, color='red')
            # plt.show()

    def get_slopes_and_deskew_new_curved(
            self, parents: list[TextRegion],
            textline_mask_tot,
            textline_confidence,
            num_col, slope_deskew, name
    ):
        if not len(parents):
            return
        self.logger.debug("enter get_slopes_and_deskew_new_curved")
        kwargs = dict(textline_mask_tot_ea=textline_mask_tot,
                      num_col=num_col,
                      slope_deskew=slope_deskew,
                      MAX_SLOPE=MAX_SLOPE,
                      KERNEL=KERNEL,
                      logger=self.logger,
                      plotter=self.plotter,
                      name=name
        )
        all_lines = []
        for parent in parents:
            textlines_cont, skew = do_work_of_slopes_new_curved(parent.contour, **kwargs)
            all_lines.extend(textlines_cont)
            parent.lines = [Region(cont) for cont in textlines_cont]
            parent.skew = skew
        # more efficient to run all in one:
        all_confs = get_region_confidences(all_lines, textline_confidence)
        get_confs = iter(all_confs)
        for parent in parents:
            for line, conf in zip(parent.lines, get_confs):
                line.conf = conf
        self.logger.debug("exit get_slopes_and_deskew_new_curved")

    def textline_contours(self, img, use_patches):
        self.logger.debug('enter textline_contours')

        if (self.tables or
            self.reading_order_machine_based or
            self.input_binary):
             # avoid OOM
            n_batch = 1
        else:
            n_batch = 3
        prediction_textline, conf_textline = do_prediction_new_concept(
            img, self.model_zoo.get("textline"),
            patches=use_patches,
            logger=self.logger,
            artificial_class=2,
            n_batch_inference=n_batch,
            thresholding_for_artificial_class=True,
            threshold_art_class=self.threshold_art_class_textline)

        #prediction_textline_longshot = do_prediction(img, self.model_zoo.get("textline"), patches=False)

        self.logger.debug('exit textline_contours')
        # suppress artificial boundary label
        result = (prediction_textline == 1).astype(np.uint8)
        #, (prediction_textline_longshot==1).astype(np.uint8)
        return result, conf_textline

    def get_early_layout(
            self, img,
            num_col_classifier,
            label_text=1,
            label_imgs=2,
            label_seps=3,
            label_art=4,
            label_tabs=10,
    ):
        self.logger.debug("enter get_early_layout")
        t_in = time.time()
        erosion_hurts = False
        # already cropped
        img_height_h, img_width_h = img.shape[:2]

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
        img_h_new = img_w_new * img_height_h // img_width_h
        img_resized = resize_image(img, img_h_new, img_w_new)
        self.logger.debug("detecting textlines on %s with %d colors",
                          str(img_resized.shape), len(np.unique(img_resized)))

        textline_mask_tot_ea, confidence_textline = self.run_textline(img_resized)
        textline_mask_tot_ea = resize_image(textline_mask_tot_ea, img_height_h, img_width_h)
        confidence_textline = resize_image(confidence_textline, img_height_h, img_width_h)

        if self.skip_layout_and_reading_order:
            self.logger.debug("exit get_early_layout")
            return erosion_hurts, None, None, None, None, textline_mask_tot_ea, None, None

        #print("inside 2 ", time.time()-t_in)
        if num_col_classifier < 3:
            if img_height_h / img_width_h > 2.5:
                patches = True
            else:
                patches = False
            self.logger.debug("resized to %dx%d for %d cols",
                              img_w_new, img_h_new, num_col_classifier)
        else:
            # training generate-gt pagexml2label's custom_config for region_1_2 model
            # seems to use columns_width (1000/1300)/1600/1900/2200/2500, but the
            # earliest implementation extrapolating region_1_2 model for more columns
            # used (1000/1500)/900/1000/1100/1200, which seems too small,
            # (and it does fracture small text regions), while the trained tile sizes
            # seem to cause problems with larger features (large headings and images),
            # hence the following compromise:
            new_w = 1200 + 200 * (num_col_classifier - 3)
            new_h = new_w * img_height_h // img_width_h
            img_resized = resize_image(img_resized, new_h, new_w)
            self.logger.debug("resized to %dx%d for %d cols",
                              new_w, new_h, num_col_classifier)
            patches = True

        prediction_regions, confidence_regions = do_prediction_new_concept(
            img_resized, self.model_zoo.get("region_1_2"),
            patches=patches,
            logger=self.logger,
            n_batch_inference=1,
            thresholding_for_artificial_class=True,
            threshold_art_class=self.threshold_art_class_layout,
            artificial_class=label_art,
            separator_class=label_seps)

        prediction_regions = resize_image(prediction_regions, img_height_h, img_width_h)
        confidence_regions = resize_image(confidence_regions, img_height_h, img_width_h)

        if self.tables:
            prediction_tables, confidence_tables = self.get_tables_from_model(img)
        else:
            prediction_tables = np.zeros(img.shape[:2], dtype=np.uint8)
            confidence_tables = np.zeros(img.shape[:2], dtype=bool)

        mask_texts_only = (prediction_regions == label_text).astype('uint8')
        mask_art_only = (prediction_regions == label_art)
        mask_images_only = (prediction_regions == label_imgs)
        mask_seps_only = (prediction_regions == label_seps).astype('uint8')
        mask_tabs_only = prediction_tables

        # if num_col_classifier == 1 or num_col_classifier == 2:
        #     mask_texts_only = cv2.morphologyEx(mask_texts_only, cv2.MORPH_OPEN, KERNEL, iterations=1)
        mask_texts_only = cv2.dilate(mask_texts_only, kernel=np.ones((2, 2), np.uint8), iterations=1)

        texts_only_cont = return_contours_of_class(mask_texts_only, 1, min_area=1e-4)
        seps_only_cont = return_contours_of_class(mask_seps_only, 1, min_area=1e-5, holes=True)
        tabs_only_cont = return_contours_of_class(mask_tabs_only, 1, min_area=1e-4)

        text_regions_p = np.zeros_like(prediction_regions)
        text_regions_p = cv2.fillPoly(text_regions_p, pts=seps_only_cont, color=label_seps)
        text_regions_p[mask_images_only] = label_imgs
        text_regions_p = cv2.fillPoly(text_regions_p, pts=texts_only_cont, color=label_text)
        text_regions_p = cv2.fillPoly(text_regions_p, pts=tabs_only_cont, color=label_tabs)
        text_regions_p[mask_art_only] = 0 # ensure instances are still separated

        # plt.figure("early layout regions")
        # plt.subplot(1, 2, 1, title="early regions")
        # plt.imshow(prediction_regions)
        # plt.subplot(1, 2, 2, title="(reconstructed)")
        # plt.imshow(text_regions_p)
        # plt.show()
        # plt.figure("early layout lines")
        # plt.subplot(1, 2, 1, title="line mask")
        # plt.imshow(textline_mask_tot_ea)

        textline_mask_tot_ea[text_regions_p != label_text] = 0
        confidence_textline[text_regions_p != label_text] = 0
        confidence_regions[text_regions_p == label_tabs] = \
            confidence_tables[text_regions_p == label_tabs]

        regions_without_separators = ((text_regions_p == label_text) |
                                      (text_regions_p == label_tabs)).astype(np.uint8)
        # plt.subplot(1, 2, 2, title="(following text regions)")
        # plt.imshow(textline_mask_tot_ea)
        # plt.show()
        #print("inside 4 ", time.time()-t_in)
        self.logger.debug("exit get_early_layout")
        return (erosion_hurts,
                seps_only_cont,
                texts_only_cont,
                regions_without_separators,
                text_regions_p,
                textline_mask_tot_ea,
                confidence_regions,
                confidence_textline)

    def get_order_of_regions(
            self,
            contours_only_text_parent,
            contours_only_text_parent_h,
            contours_drop_capitals,
            boxes,
    ):
        self.logger.debug("enter get_order_of_regions")
        contours_only_text_parent = ensure_array(contours_only_text_parent)
        contours_only_text_parent_h = ensure_array(contours_only_text_parent_h)
        contours_drop_capitals = ensure_array(contours_drop_capitals)
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
            arg_text_con_drop = match_boxes(contours_drop_capitals, only_centers, "drop")
            args_contours_main = np.arange(len(contours_only_text_parent))
            args_contours_head = np.arange(len(contours_only_text_parent_h))
            args_contours_drop = np.arange(len(contours_drop_capitals))
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
                    contours_only_text_parent[args_contours_box_main],
                    contours_only_text_parent_h[args_contours_box_head],
                    contours_drop_capitals[args_contours_box_drop],
                    r2l=self.right2left
                )

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
            # - contours_drop_capitals,
            # and then create regionrefs into these ordered by order_text_new
            order_text_new = np.argsort(np.concatenate((order_by_con_main,
                                                        order_by_con_head,
                                                        order_by_con_drop)))
            return order_text_new

        try:
            results = order_from_boxes(False)
        except Exception:
            self.logger.exception("cannot match region contours w/ reading order boxes")
            results = order_from_boxes(True)

        self.logger.debug("exit get_order_of_regions")
        return results

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

    def get_tables_from_model(self, img):
        table_prediction, table_confidence = do_prediction_new_concept(
            img, self.model_zoo.get("table"),
            patches=False,
            logger=self.logger,
            thresholding_for_artificial_class=True,
            threshold_art_class=0.05,
            artificial_class=2)
        table_prediction = table_prediction.astype(np.uint8)
        return table_prediction, table_confidence

    def run_columns(
            self, text_regions_p_1,
            slope_deskew,
            num_col_classifier,
            num_column_is_classified,
            erosion_hurts,
            label_imgs=2,
            label_seps=3,
    ):
        """post-process column classifier result"""
        t_in_gr = time.time()
        regions_without_separators = ((text_regions_p_1 != label_seps) &
                                      (text_regions_p_1 != 0)).astype(np.uint8)
        if not erosion_hurts:
            regions_without_separators = cv2.erode(regions_without_separators, KERNEL, iterations=6)
        # deskew for clearer signal
        regions_without_separators = rotate_image(regions_without_separators, slope_deskew)
        try:
            num_col, _ = find_num_col(regions_without_separators, num_col_classifier, self.tables, multiplier=6.0)
            num_col = num_col + 1
            if not num_column_is_classified:
                num_col_classifier = num_col
        except Exception:
            self.logger.exception("cannot determine overall number of columns analytically")
            num_col = None
        num_col_classifier = min(self.num_col_upper or num_col_classifier,
                                 max(self.num_col_lower or num_col_classifier,
                                     num_col_classifier))
        return num_col, num_col_classifier

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

    def run_deskew(self, textline_mask_tot_ea, axis=1):
        if not np.any(textline_mask_tot_ea):
            self.logger.info("slope_deskew: empty page")
            return 0

        #print(textline_mask_tot_ea.shape, 'textline_mask_tot_ea deskew')
        textline_mask_tot_ea = cv2.erode(textline_mask_tot_ea, KERNEL, iterations=2)
        slope_deskew = return_deskew_slop(textline_mask_tot_ea, 2,
                                          n_tot_angles=30,
                                          main_page=True,
                                          axis=axis,
                                          logger=self.logger, plotter=self.plotter)
        self.logger.info("slope_deskew: %.2f°", slope_deskew)
        return slope_deskew

    def run_marginals(self, *args):
        get_marginals(*args,
                      allow_l=self.allow_marginalia in ('left', 'both'),
                      allow_r=self.allow_marginalia in ('right', 'both'),
                      kernel=KERNEL)

    def get_full_layout(
            self, image_page,
            text_regions_p,
            num_col_classifier,
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
        # head | - |  2 | 2 | used in split_textregion_main_vs_head() afterwards
        # drop | - |  4 | 3 | assigned from full model below
        # img  | 2 |  5 | 4 | mapped below
        # sep  | 3 |  6 | 5 | mapped + assigned from full model below
        # marg | 4 |  8 | - | rule-based in run_marginals() from early text
        # tab  | - | 10 | - | dedicated model, optional
        text_regions_p[text_regions_p == label_imgs] = label_imgs_fl
        text_regions_p[text_regions_p == label_seps] = label_seps_fl
        text_regions_p[text_regions_p == label_marg] = label_marg_fl

        if self.full_layout:
            regions_fully, regionsfl_confidence = self.extract_text_regions_new(
                image_page,
                False, cols=num_col_classifier)

            drops = regions_fully == label_drop_fl_model
            regions_fully[drops] = label_text
            # rs: why erode to text here, when fill_bb... will mask out text (only allowing img/drop/bg)?
            drops = cv2.erode(drops.astype(np.uint8), KERNEL, iterations=1) == 1
            regions_fully[drops] = label_drop_fl_model
            drops = fill_bb_of_drop_capitals(regions_fully, text_regions_p)
            text_regions_p[drops] = label_drop_fl
        else:
            regions_fully = None,
            regionsfl_confidence = None

        # no need to return text_regions_p (inplace editing)
        self.logger.debug('exit get_full_layout')
        return regions_fully, regionsfl_confidence

    def get_deskewed_masks(
            self,
            slope_deskew,
            textline_mask_tot,
            text_regions_p,
            regions_without_separators,
    ):
        return (rotate_image(textline_mask_tot, slope_deskew),
                rotate_image(text_regions_p, slope_deskew),
                rotate_image(regions_without_separators, slope_deskew),
        )

    def get_boxes_order(
            self,
            text_regions_p,
            num_col_classifier,
            erosion_hurts,
            regions_without_separators,
            contours_h=[], # noqa: B006 (not modified)
            label_seps_fl=6,
    ):
        separator_mask = text_regions_p == label_seps_fl

        matrix_of_seps_ch, splitter_y_new = find_number_of_columns_in_document(
            regions_without_separators, separator_mask, num_col_classifier, self.tables,
            contours_h=contours_h)

        boxes, _ = return_boxes_of_images_by_order_of_reading_new(
            splitter_y_new, regions_without_separators,
            separator_mask, matrix_of_seps_ch,
            num_col_classifier, erosion_hurts, self.tables, self.right2left,
            logger=self.logger)
        return boxes

    def run_order_of_regions_with_model(
            self,
            contours_only_text_parent,
            contours_only_text_parent_h,
            # not trained on drops directly, but it does work:
            contours_drop_capitals,
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

            contours_only_dilated = return_contours_of_class(textregion_par, 1)

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

            args_cont_drop = np.arange(len(contours_drop_capitals))
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
        co_text_all.extend(contours_drop_capitals)

        if not len(co_text_all):
            return []

        # fill contours in lower resolution to be faster
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
            batch = []
            for tot_counter, j in enumerate(ij_list):
                input_1[len(batch), :, :, 0] = labels_con[:, :, i]
                input_1[len(batch), :, :, 1] = img_poly
                input_1[len(batch), :, :, 2] = labels_con[:, :, j]

                batch.append(j)
                if ((tot_counter + 1) % n_batch_inference == 0 # batch full
                    or tot_counter == len(ij_list) - 1): # last batch
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

    def run_order_of_regions_heuristic(
            self,
            textregions_cont,
            textregions_h_cont,
            drop_caps_cont,
            text_regions_p,
            regions_without_separators,
            num_col_classifier,
            erosion_hurts,
    ):
        if not erosion_hurts:
            regions_without_separators = cv2.erode(regions_without_separators, KERNEL, iterations=2)

        boxes = self.get_boxes_order(text_regions_p,
                                     num_col_classifier,
                                     erosion_hurts,
                                     regions_without_separators,
                                     contours_h=textregions_h_cont)
        order_text = self.get_order_of_regions(
            textregions_cont,
            textregions_h_cont,
            drop_caps_cont,
            boxes)
        return order_text

    def filter_small_regions(
            self,
            textregions: list[Region],
            textregions_d: list[Region],
            area_factor: float,
            marginals: list[Region],
    ) -> tuple[list[Region], list[Region]]:
        """
        Split list of contours (and optionally deskewed contours) into
        small (<0.1% area) and large (>=0.1%) candidates. Then identify
        those small contours whose center point is properly contained
        by some large contour (or optionally by some marginal contour).
        Remove the latter ones from the list of contours (and deskewed
        contours).
        """
        areas = np.array([textregion.area for textregion in textregions]) * area_factor
        indices_small = np.flatnonzero(areas < 1e-3)
        indices_large = np.flatnonzero(areas >= 1e-3)
        keep = [True] * len(areas)
        for ind_small in indices_small:
            results = [cv2.pointPolygonTest(textregions[ind_large].contour,
                                            (textregions[ind_small].cx,
                                             textregions[ind_small].cy),
                                            False)
                       for ind_large in indices_large]
            results = np.array(results)
            if np.any(results == 1):
                keep[ind_small] = False
            elif len(marginals):
                results = [cv2.pointPolygonTest(marginal.contour,
                                                (textregions[ind_small].cx,
                                                 textregions[ind_small].cy),
                                                False)
                           for marginal in marginals]
                results = np.array(results)
                if np.any(results == 1):
                    keep[ind_small] = False

        textregions = list(compress(textregions, keep))
        if len(textregions_d):
            textregions_d = list(compress(textregions_d, keep))

        return textregions, textregions_d

    def filter_small_textlines(
            self,
            textregions: list[TextRegion],
    ) -> list[TextRegion]:
        textlines = []
        indexes_parent = []
        indexes_child = []
        all_keep = []
        for ind_region, region in enumerate(textregions):
            textlines.extend(region.lines)
            indexes_parent.extend([ind_region] * len(region.lines))
            indexes_child.extend(list(range(len(region.lines))))
            all_keep.append([True] * len(region.lines))

        areas = np.array([textline.area for textline in textlines])
        for i, textline in enumerate(textlines):
            args_other = np.setdiff1d(np.arange(len(textlines)), i)
            areas_other = areas[args_other]
            for ind in args_other[areas_other > 1.5 * areas[i]]:
                if cv2.pointPolygonTest(textlines[ind].contour,
                                        (textline.cx,
                                         textline.cy),
                                        False) == 1:
                    all_keep[indexes_parent[i]][indexes_child[i]] = False

        for textregion, keep in zip(textregions, all_keep):
            textregion.lines = list(compress(textregion.lines, keep))
        return textregions

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

    def filter_textregions_without_textlines(self, textregions, textregions_d):
        keep = [len(textregion.lines) > 0 for textregion in textregions]
        return (list(compress(textregions, keep)),
                list(compress(textregions_d, keep)))

    def separate_marginals_and_order(self, marginals, mid_point_of_page_width):
        left = []
        right = []
        for marginal in marginals:
            (left, right)[marginal.cx  < mid_point_of_page_width].append(marginal)

        order_left = itemgetter(np.argsort([marginal.cy for marginal in left]))
        order_right = itemgetter(np.argsort([marginal.cy for marginal in right]))

        return order_left(left), order_right(right)

    def run(self,
            overwrite: bool = False,
            image_filename: str | None = None,
            dir_in: str | None = None,
            dir_out: str | None = None,
            dir_of_cropped_images: str | None = None,
            dir_of_layout: str | None = None,
            dir_of_deskewed: str | None = None,
            dir_of_all: str | None = None,
            dir_save_page: str | None = None,
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
                   dir_out: str | None = None,
                   overwrite: bool = False,
                   img_pil=None,
                   pcgts=None,
    ) -> None:
        label_text = 1
        label_imgs = 2
        label_imgs_fl = 5
        label_seps = 3
        label_seps_fl = 6
        label_marg = 4
        label_marg_fl = 8
        label_drop_fl = 4
        label_tabs = 10

        t0 = time.time()
        self.logger.info(img_filename)

        image = self.cache_images(image_filename=img_filename, image_pil=img_pil)
        writer = EynollahXmlWriter(
            dir_out=dir_out,
            image_filename=img_filename,
            image_width=image['img'].shape[1],
            image_height=image['img'].shape[0],
            pcgts=pcgts)

        if os.path.exists(writer.output_filename):
            if overwrite:
                self.logger.warning("will overwrite existing output file '%s'", writer.output_filename)
            else:
                self.logger.warning("will skip input for existing output file '%s'", writer.output_filename)
                return

        self.logger.info(f"Processing file: {writer.image_filename}")
        self.logger.info("Step 1/5: Image Enhancement")

        num_col_classifier, num_column_is_classified = self.run_enhancement(image)
        writer.scale_x = image['scale_x']
        writer.scale_y = image['scale_y']

        self.logger.info(f"Image: {image['img_res'].shape[1]}x{image['img_res'].shape[0]}, "
                         f"scale {image['scale_x']:.1f}x{image['scale_y']:.1f}, "
                         f"{image['dpi']} DPI, {num_col_classifier} columns")
        self.logger.info(f"Enhancement complete ({time.time() - t0:.1f}s)")

        t1 = time.time()
        page_cont, image_page, mask_page = self.extract_page(image)
        page = Region(page_cont)
        if not self.ignore_page_extraction:
            self.logger.debug("Cropped page is %dx%d", image_page.shape[1], image_page.shape[0])
            self.logger.info("Cropping took %.1fs", time.time() - t1)
        if self.plotter:
            self.plotter.save_page_image(image_page, image['name'])

        # Basic Processing Mode
        if self.skip_layout_and_reading_order:
            self.logger.info("Step 2/5: Basic Processing Mode")
            self.logger.info("Skipping layout analysis and reading order detection")

            _, _, _, _, _, textline_mask_tot_ea, _, textline_confidence = \
                self.get_early_layout(image_page, num_col_classifier)

            textline_mask_tot_ea *= mask_page
            textline_confidence *= mask_page
            textlines_cont = return_contours_of_class(textline_mask_tot_ea, 1, min_area=1e-5)
            textlines_conf = get_region_confidences(textlines_cont, textline_confidence)

            textlines_cx, textlines_cy = find_center_of_contours(textlines_cont)
            textlines_w_h = [cv2.boundingRect(cont)[2:]
                             for cont in textlines_cont]
            textlines_cont, textlines_conf = self.get_textlines_of_a_textregion_sorted(
                textlines_cont, textlines_conf,
                textlines_cx, textlines_cy, textlines_w_h)
            textregions = [
                TextRegion(page.contour, lines=[
                    Region(cont, conf=conf)
                    for cont, conf in zip(textlines_cont, textlines_conf)])
            ]
            textregions = self.filter_small_textlines(textregions)
            self.logger.info("Basic processing complete")

            pcgts = writer.build_pagexml(
                page=page,
                img_bin=self.imread(image, binary=True) if self.input_binary else None,
                num_col=num_col_classifier,
                order_of_texts=[0],
                textregions=textregions,
            )
            if writer.pcgts is None:
                writer.write_pagexml(pcgts)
            self.logger.info("Job done in %.1fs", time.time() - t0)
            return

        t1 = time.time()
        self.logger.info("Step 2/5: Layout Analysis")

        (erosion_hurts,
         seplines_cont,
         text_early_cont,
         regions_without_separators,
         text_regions_p,
         textline_mask_tot_ea,
         regions_confidence,
         textline_confidence) = self.get_early_layout(image['img_res'], num_col_classifier)
        t2 = time.time()
        self.logger.info("Early layout took %.1fs", t2 - t1)
        if self.plotter:
            self.plotter.save_plot_of_textlines(textline_mask_tot_ea, image['img_res'], image['name'])

        if num_col_classifier == 1:
            # variation of projection profile: from gaps between text lines
            axis = 1
        else:
            # variation of projection profile: from column gaps
            axis = 0
        if num_col_classifier < 3:
            if num_col_classifier == 1:
                img_w_new = 1000
            else:
                img_w_new = 1300
            img_h_new = img_w_new * textline_mask_tot_ea.shape[0] // textline_mask_tot_ea.shape[1]

            textline_mask_tot_ea_deskew = resize_image(textline_mask_tot_ea,img_h_new, img_w_new )
            slope_deskew = self.run_deskew(textline_mask_tot_ea_deskew, axis=axis)
        else:
            slope_deskew = self.run_deskew(textline_mask_tot_ea, axis=axis)
        # if ratio of text regions to page area is smaller that 30%,
        # then ignore skew angle above 45°
        if (abs(slope_deskew) > 45 and
            ((text_regions_p == label_text).sum()) <= 0.3 * image_page.size):
            slope_deskew = 0
        page.skew = slope_deskew
        if self.plotter:
            self.plotter.save_deskewed_image(slope_deskew, image['img'], image['name'])
        t3 = time.time()
        self.logger.info("Deskewing took %.1fs", t3 - t2)

        # FIXME: post-hoc cropping (remove when models support it, and replace image['img_res'] with image_page)
        page_box = cv2.boundingRect(page.contour)
        text_early_cont = [np.minimum([page_box[2:]],
                                      cont - [page_box[:2]])
                           for cont in text_early_cont]
        seplines_conf = get_region_confidences(seplines_cont, regions_confidence)
        seplines = [Region(np.minimum([page_box[2:]],
                                      cont - [page_box[:2]]),
                           conf=conf)
                    for cont, conf in zip(seplines_cont, seplines_conf)]
        page_box = box2slice(page_box)
        regions_without_separators = regions_without_separators[page_box] * mask_page
        text_regions_p = text_regions_p[page_box] * mask_page
        textline_mask_tot_ea = textline_mask_tot_ea[page_box] * mask_page
        regions_confidence = regions_confidence[page_box] * mask_page
        textline_confidence = textline_confidence[page_box] * mask_page

        num_col, num_col_classifier = \
            self.run_columns(text_regions_p,
                             slope_deskew,
                             num_col_classifier,
                             num_column_is_classified,
                             erosion_hurts)
        t4 = time.time()
        textline_mask_tot_ea_org = np.copy(textline_mask_tot_ea)

        if not num_col and len(text_early_cont) == 0 or not image_page.size:
            self.logger.info("No columns detected - generating empty PAGE-XML")

            pcgts = writer.build_pagexml(
                page=page,
                img_bin=self.imread(image, binary=True) if self.input_binary else None,
                num_col=0,
            )
            if writer.pcgts is None:
                writer.write_pagexml(pcgts)
            self.logger.info("Job done in %.1fs", time.time() - t0)
            return

        if num_col_classifier in (1,2) and self.allow_marginalia != 'off':
            self.run_marginals(num_col_classifier, slope_deskew,
                               text_regions_p, textline_mask_tot_ea_org)
            t5 = time.time()
            self.logger.info("Marginalia extraction took %.1fs", t5 - t4)
        else:
            t5 = time.time()

        if self.plotter:
            self.plotter.save_plot_of_layout_main_all(text_regions_p, image_page, image['name'])
            self.plotter.save_plot_of_layout_main(text_regions_p, image_page, image['name'])

        regions_fully, regionsfl_confidence = \
            self.get_full_layout(image_page, text_regions_p, num_col_classifier)

        if self.full_layout:
            drop_caps_cont = return_contours_of_class(text_regions_p, label_drop_fl, min_area=1e-4)
            drop_caps_conf = get_region_confidences(drop_caps_cont, regionsfl_confidence)
            drop_caps = [Region(cont, conf=conf)
                         for cont, conf in zip(drop_caps_cont, drop_caps_conf)]
            drop_caps_mask = np.zeros_like(regions_without_separators)
            drop_caps_mask = cv2.fillPoly(drop_caps_mask, pts=drop_caps_cont, color=1)
            regions_without_separators[drop_caps_mask] = 1 # also cover in reading-order
            textline_mask_tot_ea_org[drop_caps_mask] = 0 # skip for textlines
            textline_mask_tot_ea[drop_caps_mask] = 1 # needed for reading order
            t6 = time.time()
            self.logger.info("Full layout took %.1fs", t6 - t5)
        else:
            drop_caps = []
            t6 = time.time()
        self.logger.info("Step 3/5: Contour extraction")

        min_area_mar = 1e-5
        marginal_mask = (text_regions_p == label_marg_fl).astype(np.uint8)
        marginal_mask = cv2.dilate(marginal_mask, KERNEL, iterations=2)
        marginals_cont = return_contours_of_class(marginal_mask, 1, min_area_mar)
        marginals_conf = get_region_confidences(marginals_cont, regions_confidence)
        marginals = [Region(cont, conf=conf)
                     for cont, conf in zip(marginals_cont, marginals_conf)]
        tables_cont = return_contours_of_class(text_regions_p, label_tabs, MIN_AREA_REGION)
        tables_conf = get_region_confidences(tables_cont, regions_confidence)
        tables = [Region(cont, conf=conf)
                  for cont, conf in zip(tables_cont, tables_conf)]
        images_cont = return_contours_of_class(text_regions_p, label_imgs_fl, 2e-4)
        images_conf = get_region_confidences(images_cont, regions_confidence)
        images = [Region(cont, conf=conf)
                  for cont, conf in zip(images_cont, images_conf)]

        textregions_cont = return_contours_of_class(text_regions_p, label_text, MIN_AREA_REGION)
        textregions = [TextRegion(cont, lines=[]) for cont in textregions_cont]

        if np.abs(slope_deskew) >= SLOPE_THRESHOLD and not self.reading_order_machine_based:
            (text_regions_p_d,
             textline_mask_tot_ea_d,
             regions_without_separators_d) = self.get_deskewed_masks(
                 slope_deskew,
                 text_regions_p,
                 textline_mask_tot_ea,
                 regions_without_separators)

            textregions_cont_d = return_contours_of_class(text_regions_p_d, label_text, MIN_AREA_REGION)
            textregions_d = [TextRegion(cont, lines=[]) for cont in textregions_cont_d]
            if (len(textregions) and
                len(textregions_d)):
                textregions_cont_d = \
                    match_deskewed_contours(
                        slope_deskew,
                        textregions,
                        textregions_d,
                        text_regions_p.shape,
                        text_regions_p_d.shape)
                textregions_d = [TextRegion(cont, lines=[]) for cont in textregions_cont_d]
        else:
            textregions_d = []

        area_factor = np.reciprocal(np.prod(text_regions_p.shape).astype(float))
        textregions, textregions_d = self.filter_small_regions(
             textregions, textregions_d,
             area_factor,
             marginals)
        textregions_conf = get_region_confidences(textregions_cont, regions_confidence)
        for textregion, conf in zip(textregions, textregions_conf):
            textregion.conf = conf

        t7 = time.time()
        self.logger.info("Region contours took %.1fs", t7 - t6)

        if not self.curved_line:
            self.logger.info("Mode: Light line detection")
            args = (textline_mask_tot_ea_org,
                    textline_confidence,
                    slope_deskew)
            self.get_slopes_and_deskew_new_light2(textregions, *args)
            self.get_slopes_and_deskew_new_light2(marginals, *args)
            textregions = self.filter_small_textlines(textregions)
        else:
            self.logger.info("Mode: Curved line detection")

            textline_mask_tot_ea_erode = cv2.erode(textline_mask_tot_ea_org, kernel=KERNEL, iterations=2)
            args = (textline_mask_tot_ea_erode,
                    textline_confidence,
                    num_col_classifier,
                    slope_deskew,
                    image['name'])
            self.get_slopes_and_deskew_new_curved(textregions, *args)
            small_textlines_to_parent_adherence2(textregions, area_factor, num_col_classifier)
            self.get_slopes_and_deskew_new_curved(marginals, *args)
            small_textlines_to_parent_adherence2(marginals, area_factor, num_col_classifier)

        textregions, textregions_d = self.filter_textregions_without_textlines(
            textregions, textregions_d)
        t8 = time.time()
        self.logger.info("Line contours took %.1fs", t8 - t7)

        (marginals_left,
         marginals_right) = self.separate_marginals_and_order(
             marginals, 0.5 * text_regions_p.shape[1])

        if self.full_layout:
            (text_regions_p,
             textregions,
             textregions_h,
             textregions_d,
             textregions_h_d) = split_textregion_main_vs_head(
                 text_regions_p,
                 regions_fully,
                 textregions,
                 textregions_d)

            if self.plotter:
                self.plotter.save_plot_of_layout(text_regions_p, image_page, image['name'])
                self.plotter.save_plot_of_layout_all(text_regions_p, image_page, image['name'])
        else:
            textregions_h = []
            textregions_h_d = []

        def contours(regions):
            return [region.contour for region in regions]
        if self.plotter:
            self.plotter.write_images_into_directory(contours(images), image_page,
                                                     image['scale_x'], image['scale_y'], image['name'])

        t_order = time.time()
        self.logger.info("Step 4/5: Reading Order")
        if self.right2left:
            self.logger.info("Right-to-left mode enabled")
        if self.headers_off:
            self.logger.info("Headers ignored in reading order")

        if self.reading_order_machine_based:
            self.logger.info("Using machine-based detection")
            order_text = self.run_order_of_regions_with_model(
                contours(textregions),
                contours(textregions_h) if not self.headers_off else [],
                contours(drop_caps),
                text_regions_p)
        else:
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                order_text = self.run_order_of_regions_heuristic(
                    contours(textregions),
                    contours(textregions_h) if not self.headers_off else [],
                    contours(drop_caps),
                    text_regions_p,
                    regions_without_separators,
                    num_col_classifier,
                    erosion_hurts)
            else:
                order_text = self.run_order_of_regions_heuristic(
                    contours(textregions_d),
                    contours(textregions_h_d) if not self.headers_off else [],
                    contours(drop_caps),
                    text_regions_p_d,
                    regions_without_separators_d,
                    num_col_classifier,
                    erosion_hurts)
        self.logger.info(f"Detection of reading order took {time.time() - t_order:.1f}s")

        self.logger.info("Step 5/5: Output Generation")
        pcgts = writer.build_pagexml(
            page=page,
            img_bin=self.imread(image, binary=True) if self.input_binary else None,
            num_col=num_col_classifier,
            order_of_texts=order_text,
            textregions=textregions,
            textregions_h=textregions_h,
            images=images,
            tables=tables,
            drop_caps=drop_caps,
            marginals_left=marginals_left,
            marginals_right=marginals_right,
            seplines=seplines,
        )
        if writer.pcgts is None:
            writer.write_pagexml(pcgts)
        self.logger.info("Job done in %.1fs", time.time() - t0)
        return
