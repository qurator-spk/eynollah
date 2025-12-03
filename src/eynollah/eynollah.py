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
import sys

from difflib import SequenceMatcher as sq
import math
import os
import time
from typing import Optional
from functools import partial
from pathlib import Path
from multiprocessing import cpu_count
import gc

from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
import shapely.affinity
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize
from ocrd_utils import tf_disable_interactive_logs
import statistics

tf_disable_interactive_logs()

import tensorflow as tf
try:
    import torch
except ImportError:
    torch = None
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
    get_textregion_contours_in_org_image_light,
    return_contours_of_image,
    return_contours_of_interested_region,
    return_parent_contours,
    dilate_textregion_contours,
    dilate_textline_contours,
    polygon2contour,
    contour2polygon,
    join_polygons,
    make_intersection,
)
from .utils.rotate import (
    rotate_image,
    rotation_not_90_func,
    rotation_not_90_func_full_layout,
)
from .utils.separate_lines import (
    return_deskew_slop,
    do_work_of_slopes_new_curved,
)
from .utils.marginals import get_marginals
from .utils.resize import resize_image
from .utils.shm import share_ndarray
from .utils import (
    is_image_filename,
    isNaN,
    crop_image_inside_box,
    box2rect,
    find_num_col,
    otsu_copy_binary,
    putt_bb_of_drop_capitals_of_model_in_patches_in_layout,
    check_any_text_region_in_model_one_is_main_or_header_light,
    small_textlines_to_parent_adherence2,
    order_of_regions,
    find_number_of_columns_in_document,
    return_boxes_of_images_by_order_of_reading_new
)
from .utils.pil_cv2 import pil2cv
from .utils.xml import order_and_id_of_texts
from .plot import EynollahPlotter
from .writer import EynollahXmlWriter

MIN_AREA_REGION = 0.000001
SLOPE_THRESHOLD = 0.13
RATIO_OF_TWO_MODEL_THRESHOLD = 95.50 #98.45:
DPI_THRESHOLD = 298
MAX_SLOPE = 999
KERNEL = np.ones((5, 5), np.uint8)

projection_dim = 64
patch_size = 1
num_patches =21*21#14*14#28*28#14*14#28*28



class Eynollah:
    def __init__(
        self,
        *,
        model_zoo: EynollahModelZoo,
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
        num_col_upper : Optional[int] = None,
        num_col_lower : Optional[int] = None,
        threshold_art_class_layout: Optional[float] = None,
        threshold_art_class_textline: Optional[float] = None,
        skip_layout_and_reading_order : bool = False,
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
        if num_col_upper:
            self.num_col_upper = int(num_col_upper)
        else:
            self.num_col_upper = num_col_upper
        if num_col_lower:
            self.num_col_lower = int(num_col_lower)
        else:
            self.num_col_lower = num_col_lower

        # for parallelization of CPU-intensive tasks:
        self.executor = ProcessPoolExecutor(max_workers=cpu_count())
            
        if threshold_art_class_layout:
            self.threshold_art_class_layout = float(threshold_art_class_layout)
        else:
            self.threshold_art_class_layout = 0.1
            
        if threshold_art_class_textline:
            self.threshold_art_class_textline = float(threshold_art_class_textline)
        else:
            self.threshold_art_class_textline = 0.1

        t_start = time.time()

        # #gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        # #gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=7.7, allow_growth=True)
        # #session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.allow_growth = True
        # #session = tf.InteractiveSession()
        # session = tf.compat.v1.Session(config=config)
        # set_session(session)
        try:
            for device in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(device, True)
        except:
            self.logger.warning("no GPU device available")
            
        self.logger.info("Loading models...")
        self.setup_models()
        self.logger.info(f"Model initialization complete ({time.time() - t_start:.1f}s)")

    def setup_models(self):

        # load models, depending on modes
        # (note: loading too many models can cause OOM on GPU/CUDA,
        #  thus, we try set up the minimal configuration for the current mode)
        loadable = [
            "col_classifier",
            "binarization",
            "page",
            "region"
        ]
        loadable.append(("textline"))
        loadable.append("region_1_2")
        if self.full_layout:
            loadable.append("region_fl_np")
            #loadable.append("region_fl")
        if self.reading_order_machine_based:
            loadable.append("reading_order")
        if self.tables:
            loadable.append(("table"))

        self.model_zoo.load_models(*loadable)

    def __del__(self):
        if hasattr(self, 'executor') and getattr(self, 'executor'):
            assert self.executor
            self.executor.shutdown()
            self.executor = None
        self.model_zoo.shutdown()

    @property
    def device(self):
        # TODO why here and why only for tr?
        assert torch
        if torch.cuda.is_available():
            self.logger.info("Using GPU acceleration")
            return torch.device("cuda:0")
        self.logger.info("Using CPU processing")
        return torch.device("cpu")

    def cache_images(self, image_filename=None, image_pil=None, dpi=None):
        ret = {}
        t_c0 = time.time()
        if image_filename:
            ret['img'] = cv2.imread(image_filename)
            self.dpi = 100
        else:
            ret['img'] = pil2cv(image_pil)
            self.dpi = 100
        ret['img_grayscale'] = cv2.cvtColor(ret['img'], cv2.COLOR_BGR2GRAY)
        for prefix in ('',  '_grayscale'):
            ret[f'img{prefix}_uint8'] = ret[f'img{prefix}'].astype(np.uint8)
        self._imgs = ret
        if dpi is not None:
            self.dpi = dpi

    def reset_file_name_dir(self, image_filename, dir_out):
        t_c = time.time()
        self.cache_images(image_filename=image_filename)
        self.writer = EynollahXmlWriter(
            dir_out=dir_out,
            image_filename=image_filename,
            curved_line=self.curved_line)

    def imread(self, grayscale=False, uint8=True):
        key = 'img'
        if grayscale:
            key += '_grayscale'
        if uint8:
            key += '_uint8'
        return self._imgs[key].copy()

    def predict_enhancement(self, img):
        self.logger.debug("enter predict_enhancement")

        img_height_model = self.model_zoo.get("enhancement").layers[-1].output_shape[1]
        img_width_model = self.model_zoo.get("enhancement").layers[-1].output_shape[2]
        if img.shape[0] < img_height_model:
            img = cv2.resize(img, (img.shape[1], img_width_model), interpolation=cv2.INTER_NEAREST)
        if img.shape[1] < img_width_model:
            img = cv2.resize(img, (img_height_model, img.shape[0]), interpolation=cv2.INTER_NEAREST)
        margin = int(0 * img_width_model)
        width_mid = img_width_model - 2 * margin
        height_mid = img_height_model - 2 * margin
        img = img / 255.
        img_h = img.shape[0]
        img_w = img.shape[1]

        prediction_true = np.zeros((img_h, img_w, 3))
        nxf = img_w / float(width_mid)
        nyf = img_h / float(height_mid)
        nxf = int(nxf) + 1 if nxf > int(nxf) else int(nxf)
        nyf = int(nyf) + 1 if nyf > int(nyf) else int(nyf)

        for i in range(nxf):
            for j in range(nyf):
                if i == 0:
                    index_x_d = i * width_mid
                    index_x_u = index_x_d + img_width_model
                else:
                    index_x_d = i * width_mid
                    index_x_u = index_x_d + img_width_model
                if j == 0:
                    index_y_d = j * height_mid
                    index_y_u = index_y_d + img_height_model
                else:
                    index_y_d = j * height_mid
                    index_y_u = index_y_d + img_height_model

                if index_x_u > img_w:
                    index_x_u = img_w
                    index_x_d = img_w - img_width_model
                if index_y_u > img_h:
                    index_y_u = img_h
                    index_y_d = img_h - img_height_model

                img_patch = img[np.newaxis, index_y_d:index_y_u, index_x_d:index_x_u, :]
                label_p_pred = self.model_zoo.get("enhancement").predict(img_patch, verbose=0)
                seg = label_p_pred[0, :, :, :] * 255

                if i == 0 and j == 0:
                    prediction_true[index_y_d + 0:index_y_u - margin,
                                    index_x_d + 0:index_x_u - margin] = \
                                        seg[0:-margin or None,
                                            0:-margin or None]
                elif i == nxf - 1 and j == nyf - 1:
                    prediction_true[index_y_d + margin:index_y_u - 0,
                                    index_x_d + margin:index_x_u - 0] = \
                                        seg[margin:,
                                            margin:]
                elif i == 0 and j == nyf - 1:
                    prediction_true[index_y_d + margin:index_y_u - 0,
                                    index_x_d + 0:index_x_u - margin] = \
                                        seg[margin:,
                                            0:-margin or None]
                elif i == nxf - 1 and j == 0:
                    prediction_true[index_y_d + 0:index_y_u - margin,
                                    index_x_d + margin:index_x_u - 0] = \
                                        seg[0:-margin or None,
                                            margin:]
                elif i == 0 and j != 0 and j != nyf - 1:
                    prediction_true[index_y_d + margin:index_y_u - margin,
                                    index_x_d + 0:index_x_u - margin] = \
                                        seg[margin:-margin or None,
                                            0:-margin or None]
                elif i == nxf - 1 and j != 0 and j != nyf - 1:
                    prediction_true[index_y_d + margin:index_y_u - margin,
                                    index_x_d + margin:index_x_u - 0] = \
                                        seg[margin:-margin or None,
                                            margin:]
                elif i != 0 and i != nxf - 1 and j == 0:
                    prediction_true[index_y_d + 0:index_y_u - margin,
                                    index_x_d + margin:index_x_u - margin] = \
                                        seg[0:-margin or None,
                                            margin:-margin or None]
                elif i != 0 and i != nxf - 1 and j == nyf - 1:
                    prediction_true[index_y_d + margin:index_y_u - 0,
                                    index_x_d + margin:index_x_u - margin] = \
                                        seg[margin:,
                                            margin:-margin or None]
                else:
                    prediction_true[index_y_d + margin:index_y_u - margin,
                                    index_x_d + margin:index_x_u - margin] = \
                                        seg[margin:-margin or None,
                                            margin:-margin or None]

        prediction_true = prediction_true.astype(int)
        return prediction_true

    def calculate_width_height_by_columns(self, img, num_col, width_early, label_p_pred):
        self.logger.debug("enter calculate_width_height_by_columns")
        if num_col == 1 and width_early < 1100:
            img_w_new = 2000
        elif num_col == 1 and width_early >= 2500:
            img_w_new = 2000
        elif num_col == 1 and width_early >= 1100 and width_early < 2500:
            img_w_new = width_early
        elif num_col == 2 and width_early < 2000:
            img_w_new = 2400
        elif num_col == 2 and width_early >= 3500:
            img_w_new = 2400
        elif num_col == 2 and width_early >= 2000 and width_early < 3500:
            img_w_new = width_early
        elif num_col == 3 and width_early < 2000:
            img_w_new = 3000
        elif num_col == 3 and width_early >= 4000:
            img_w_new = 3000
        elif num_col == 3 and width_early >= 2000 and width_early < 4000:
            img_w_new = width_early
        elif num_col == 4 and width_early < 2500:
            img_w_new = 4000
        elif num_col == 4 and width_early >= 5000:
            img_w_new = 4000
        elif num_col == 4 and width_early >= 2500 and width_early < 5000:
            img_w_new = width_early
        elif num_col == 5 and width_early < 3700:
            img_w_new = 5000
        elif num_col == 5 and width_early >= 7000:
            img_w_new = 5000
        elif num_col == 5 and width_early >= 3700 and width_early < 7000:
            img_w_new = width_early
        elif num_col == 6 and width_early < 4500:
            img_w_new = 6500  # 5400
        else:
            img_w_new = width_early
        img_h_new = img_w_new * img.shape[0] // img.shape[1]

        if label_p_pred[0][int(num_col - 1)] < 0.9 and img_w_new < width_early:
            img_new = np.copy(img)
            num_column_is_classified = False
        #elif label_p_pred[0][int(num_col - 1)] < 0.8 and img_h_new >= 8000:
        elif img_h_new >= 8000:
            img_new = np.copy(img)
            num_column_is_classified = False
        else:
            img_new = resize_image(img, img_h_new, img_w_new)
            num_column_is_classified = True

        return img_new, num_column_is_classified

    def calculate_width_height_by_columns_1_2(self, img, num_col, width_early, label_p_pred):
        self.logger.debug("enter calculate_width_height_by_columns")
        if num_col == 1:
            img_w_new = 1000
        else:
            img_w_new = 1300
        img_h_new = img_w_new * img.shape[0] // img.shape[1]

        if label_p_pred[0][int(num_col - 1)] < 0.9 and img_w_new < width_early:
            img_new = np.copy(img)
            num_column_is_classified = False
        #elif label_p_pred[0][int(num_col - 1)] < 0.8 and img_h_new >= 8000:
        elif img_h_new >= 8000:
            img_new = np.copy(img)
            num_column_is_classified = False
        else:
            img_new = resize_image(img, img_h_new, img_w_new)
            num_column_is_classified = True

        return img_new, num_column_is_classified

    def resize_image_with_column_classifier(self, is_image_enhanced, img_bin):
        self.logger.debug("enter resize_image_with_column_classifier")
        if self.input_binary:
            img = np.copy(img_bin)
        else:
            img = self.imread()

        _, page_coord = self.early_page_for_num_of_column_classification(img)

        if self.input_binary:
            img_in = np.copy(img)
            img_in = img_in / 255.0
            width_early = img_in.shape[1]
            img_in = cv2.resize(img_in, (448, 448), interpolation=cv2.INTER_NEAREST)
            img_in = img_in.reshape(1, 448, 448, 3)
        else:
            img_1ch = self.imread(grayscale=True, uint8=False)
            width_early = img_1ch.shape[1]
            img_1ch = img_1ch[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]

            # plt.imshow(img_1ch)
            # plt.show()
            img_1ch = img_1ch / 255.0
            img_1ch = cv2.resize(img_1ch, (448, 448), interpolation=cv2.INTER_NEAREST)

            img_in = np.zeros((1, img_1ch.shape[0], img_1ch.shape[1], 3))
            img_in[0, :, :, 0] = img_1ch[:, :]
            img_in[0, :, :, 1] = img_1ch[:, :]
            img_in[0, :, :, 2] = img_1ch[:, :]

        label_p_pred = self.model_zoo.get("col_classifier").predict(img_in, verbose=0)
        num_col = np.argmax(label_p_pred[0]) + 1

        self.logger.info("Found %s columns (%s)", num_col, label_p_pred)
        img_new, _ = self.calculate_width_height_by_columns(img, num_col, width_early, label_p_pred)

        if img_new.shape[1] > img.shape[1]:
            img_new = self.predict_enhancement(img_new)
            is_image_enhanced = True

        return img, img_new, is_image_enhanced

    def resize_and_enhance_image_with_column_classifier(self):
        self.logger.debug("enter resize_and_enhance_image_with_column_classifier")
        dpi = self.dpi
        self.logger.info("Detected %s DPI", dpi)
        if self.input_binary:
            img = self.imread()
            prediction_bin = self.do_prediction(True, img, self.model_zoo.get("binarization"), n_batch_inference=5)
            prediction_bin = 255 * (prediction_bin[:,:,0] == 0)
            prediction_bin = np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            img= np.copy(prediction_bin)
            img_bin = prediction_bin
        else:
            img = self.imread()
            img_bin = None

        width_early = img.shape[1]
        t1 = time.time()
        _, page_coord = self.early_page_for_num_of_column_classification(img_bin)

        self.image_page_org_size = img[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3], :]
        self.page_coord = page_coord

        if self.num_col_upper and not self.num_col_lower:
            num_col = self.num_col_upper
            label_p_pred = [np.ones(6)]
        elif self.num_col_lower and not self.num_col_upper:
            num_col = self.num_col_lower
            label_p_pred = [np.ones(6)]
        elif not self.num_col_upper and not self.num_col_lower:
            if self.input_binary:
                img_in = np.copy(img)
                img_in = img_in / 255.0
                img_in = cv2.resize(img_in, (448, 448), interpolation=cv2.INTER_NEAREST)
                img_in = img_in.reshape(1, 448, 448, 3)
            else:
                img_1ch = self.imread(grayscale=True)
                width_early = img_1ch.shape[1]
                img_1ch = img_1ch[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]

                img_1ch = img_1ch / 255.0
                img_1ch = cv2.resize(img_1ch, (448, 448), interpolation=cv2.INTER_NEAREST)
                img_in = np.zeros((1, img_1ch.shape[0], img_1ch.shape[1], 3))
                img_in[0, :, :, 0] = img_1ch[:, :]
                img_in[0, :, :, 1] = img_1ch[:, :]
                img_in[0, :, :, 2] = img_1ch[:, :]

            label_p_pred = self.model_zoo.get("col_classifier").predict(img_in, verbose=0)
            num_col = np.argmax(label_p_pred[0]) + 1
            
        elif (self.num_col_upper and self.num_col_lower) and (self.num_col_upper!=self.num_col_lower):
            if self.input_binary:
                img_in = np.copy(img)
                img_in = img_in / 255.0
                img_in = cv2.resize(img_in, (448, 448), interpolation=cv2.INTER_NEAREST)
                img_in = img_in.reshape(1, 448, 448, 3)
            else:
                img_1ch = self.imread(grayscale=True)
                width_early = img_1ch.shape[1]
                img_1ch = img_1ch[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]

                img_1ch = img_1ch / 255.0
                img_1ch = cv2.resize(img_1ch, (448, 448), interpolation=cv2.INTER_NEAREST)
                img_in = np.zeros((1, img_1ch.shape[0], img_1ch.shape[1], 3))
                img_in[0, :, :, 0] = img_1ch[:, :]
                img_in[0, :, :, 1] = img_1ch[:, :]
                img_in[0, :, :, 2] = img_1ch[:, :]

            label_p_pred = self.model_zoo.get("col_classifier").predict(img_in, verbose=0)
            num_col = np.argmax(label_p_pred[0]) + 1

            if num_col > self.num_col_upper:
                num_col = self.num_col_upper
                label_p_pred = [np.ones(6)]
            if num_col < self.num_col_lower:
                num_col = self.num_col_lower
                label_p_pred = [np.ones(6)]
        else:
            num_col = self.num_col_upper
            label_p_pred = [np.ones(6)]

        self.logger.info("Found %d columns (%s)", num_col, np.around(label_p_pred, decimals=5))
        if dpi < DPI_THRESHOLD:
            if num_col in (1,2):
                img_new, num_column_is_classified = self.calculate_width_height_by_columns_1_2(
                    img, num_col, width_early, label_p_pred)
            else:
                img_new, num_column_is_classified = self.calculate_width_height_by_columns(
                    img, num_col, width_early, label_p_pred)
            image_res = np.copy(img_new)
            is_image_enhanced = True
        else:
            if num_col in (1,2):
                img_new, num_column_is_classified = self.calculate_width_height_by_columns_1_2(
                    img, num_col, width_early, label_p_pred)
                image_res = np.copy(img_new)
                is_image_enhanced = True
            else:
                num_column_is_classified = True
                image_res = np.copy(img)
                is_image_enhanced = False

        self.logger.debug("exit resize_and_enhance_image_with_column_classifier")
        return is_image_enhanced, img, image_res, num_col, num_column_is_classified, img_bin

    # pylint: disable=attribute-defined-outside-init
    def get_image_and_scales(self, img_org, img_res, scale):
        self.logger.debug("enter get_image_and_scales")
        self.image = np.copy(img_res)
        self.image_org = np.copy(img_org)
        self.height_org = self.image.shape[0]
        self.width_org = self.image.shape[1]

        self.img_hight_int = int(self.image.shape[0] * scale)
        self.img_width_int = int(self.image.shape[1] * scale)
        self.scale_y: float = self.img_hight_int / float(self.image.shape[0])
        self.scale_x: float = self.img_width_int / float(self.image.shape[1])

        self.image = resize_image(self.image, self.img_hight_int, self.img_width_int)

        # Also set for the plotter
        if self.plotter:
            self.plotter.image_org = self.image_org
            self.plotter.scale_y = self.scale_y
            self.plotter.scale_x = self.scale_x
        # Also set for the writer
        self.writer.image_org = self.image_org
        self.writer.scale_y = self.scale_y
        self.writer.scale_x = self.scale_x
        self.writer.height_org = self.height_org
        self.writer.width_org = self.width_org

    def get_image_and_scales_after_enhancing(self, img_org, img_res):
        self.logger.debug("enter get_image_and_scales_after_enhancing")
        self.image = np.copy(img_res)
        self.image = self.image.astype(np.uint8)
        self.image_org = np.copy(img_org)
        self.height_org = self.image_org.shape[0]
        self.width_org = self.image_org.shape[1]

        self.scale_y = img_res.shape[0] / float(self.image_org.shape[0])
        self.scale_x = img_res.shape[1] / float(self.image_org.shape[1])

        # Also set for the plotter
        if self.plotter:
            self.plotter.image_org = self.image_org
            self.plotter.scale_y = self.scale_y
            self.plotter.scale_x = self.scale_x
        # Also set for the writer
        self.writer.image_org = self.image_org
        self.writer.scale_y = self.scale_y
        self.writer.scale_x = self.scale_x
        self.writer.height_org = self.height_org
        self.writer.width_org = self.width_org

    def do_prediction(
            self, patches, img, model,
            n_batch_inference=1, marginal_of_patch_percent=0.1,
            thresholding_for_some_classes_in_light_version=False,
            thresholding_for_artificial_class_in_light_version=False,
            thresholding_for_fl_light_version=False,
            threshold_art_class_textline=0.1):

        self.logger.debug("enter do_prediction (patches=%d)", patches)
        img_height_model = model.layers[-1].output_shape[1]
        img_width_model = model.layers[-1].output_shape[2]

        if not patches:
            img_h_page = img.shape[0]
            img_w_page = img.shape[1]
            img = img / float(255.0)
            img = resize_image(img, img_height_model, img_width_model)

            label_p_pred = model.predict(img[np.newaxis], verbose=0)
            seg = np.argmax(label_p_pred, axis=3)[0]

            if thresholding_for_artificial_class_in_light_version:
                seg_art = label_p_pred[0,:,:,2]

                seg_art[seg_art<threshold_art_class_textline] = 0
                seg_art[seg_art>0] =1
                
                skeleton_art = skeletonize(seg_art)
                skeleton_art = skeleton_art*1

                seg[skeleton_art==1]=2
                
            if thresholding_for_fl_light_version:
                seg_header = label_p_pred[0,:,:,2]

                seg_header[seg_header<0.2] = 0
                seg_header[seg_header>0] =1

                seg[seg_header==1]=2
                
            seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = resize_image(seg_color, img_h_page, img_w_page).astype(np.uint8)
            return prediction_true

        if img.shape[0] < img_height_model:
            img = resize_image(img, img_height_model, img.shape[1])
        if img.shape[1] < img_width_model:
            img = resize_image(img, img.shape[0], img_width_model)

        self.logger.debug("Patch size: %sx%s", img_height_model, img_width_model)
        margin = int(marginal_of_patch_percent * img_height_model)
        width_mid = img_width_model - 2 * margin
        height_mid = img_height_model - 2 * margin
        img = img / 255.
        #img = img.astype(np.float16)
        img_h = img.shape[0]
        img_w = img.shape[1]
        prediction_true = np.zeros((img_h, img_w, 3))
        mask_true = np.zeros((img_h, img_w))
        nxf = math.ceil(img_w / float(width_mid))
        nyf = math.ceil(img_h / float(height_mid))

        list_i_s = []
        list_j_s = []
        list_x_u = []
        list_x_d = []
        list_y_u = []
        list_y_d = []

        batch_indexer = 0
        img_patch = np.zeros((n_batch_inference, img_height_model, img_width_model, 3))
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

                img_patch[batch_indexer,:,:,:] = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
                batch_indexer += 1

                if (batch_indexer == n_batch_inference or
                    # last batch
                    i == nxf - 1 and j == nyf - 1):
                    self.logger.debug("predicting patches on %s", str(img_patch.shape))
                    label_p_pred = model.predict(img_patch, verbose=0)
                    seg = np.argmax(label_p_pred, axis=3)

                    if thresholding_for_some_classes_in_light_version:
                        seg_not_base = label_p_pred[:,:,:,4]
                        seg_not_base[seg_not_base>0.03] =1
                        seg_not_base[seg_not_base<1] =0

                        seg_line = label_p_pred[:,:,:,3]
                        seg_line[seg_line>0.1] =1
                        seg_line[seg_line<1] =0

                        seg_background = label_p_pred[:,:,:,0]
                        seg_background[seg_background>0.25] =1
                        seg_background[seg_background<1] =0

                        seg[seg_not_base==1]=4
                        seg[seg_background==1]=0
                        seg[(seg_line==1) & (seg==0)]=3
                    if thresholding_for_artificial_class_in_light_version:
                        seg_art = label_p_pred[:,:,:,2]

                        seg_art[seg_art<threshold_art_class_textline] = 0
                        seg_art[seg_art>0] =1

                        ##seg[seg_art==1]=2

                    indexer_inside_batch = 0
                    for i_batch, j_batch in zip(list_i_s, list_j_s):
                        seg_in = seg[indexer_inside_batch]
                        
                        if thresholding_for_artificial_class_in_light_version:
                            seg_in_art = seg_art[indexer_inside_batch]

                        index_y_u_in = list_y_u[indexer_inside_batch]
                        index_y_d_in = list_y_d[indexer_inside_batch]

                        index_x_u_in = list_x_u[indexer_inside_batch]
                        index_x_d_in = list_x_d[indexer_inside_batch]

                        if i_batch == 0 and j_batch == 0:
                            prediction_true[index_y_d_in + 0:index_y_u_in - margin,
                                            index_x_d_in + 0:index_x_u_in - margin] = \
                                                seg_in[0:-margin or None,
                                                       0:-margin or None,
                                                       np.newaxis]
                            if thresholding_for_artificial_class_in_light_version:
                                prediction_true[index_y_d_in + 0:index_y_u_in - margin,
                                                index_x_d_in + 0:index_x_u_in - margin, 1] = \
                                                    seg_in_art[0:-margin or None,
                                                        0:-margin or None]
                                
                        elif i_batch == nxf - 1 and j_batch == nyf - 1:
                            prediction_true[index_y_d_in + margin:index_y_u_in - 0,
                                            index_x_d_in + margin:index_x_u_in - 0] = \
                                                seg_in[margin:,
                                                       margin:,
                                                       np.newaxis]
                            if thresholding_for_artificial_class_in_light_version:
                                prediction_true[index_y_d_in + margin:index_y_u_in - 0,
                                                index_x_d_in + margin:index_x_u_in - 0, 1] = \
                                                    seg_in_art[margin:,
                                                        margin:]
                                
                        elif i_batch == 0 and j_batch == nyf - 1:
                            prediction_true[index_y_d_in + margin:index_y_u_in - 0,
                                            index_x_d_in + 0:index_x_u_in - margin] = \
                                                seg_in[margin:,
                                                       0:-margin or None,
                                                       np.newaxis]
                            if thresholding_for_artificial_class_in_light_version:
                                prediction_true[index_y_d_in + margin:index_y_u_in - 0,
                                                index_x_d_in + 0:index_x_u_in - margin, 1] = \
                                                    seg_in_art[margin:,
                                                        0:-margin or None]
                                
                        elif i_batch == nxf - 1 and j_batch == 0:
                            prediction_true[index_y_d_in + 0:index_y_u_in - margin,
                                            index_x_d_in + margin:index_x_u_in - 0] = \
                                                seg_in[0:-margin or None,
                                                       margin:,
                                                       np.newaxis]
                            if thresholding_for_artificial_class_in_light_version:
                                prediction_true[index_y_d_in + 0:index_y_u_in - margin,
                                                index_x_d_in + margin:index_x_u_in - 0, 1] = \
                                                    seg_in_art[0:-margin or None,
                                                        margin:]
                                
                        elif i_batch == 0 and j_batch != 0 and j_batch != nyf - 1:
                            prediction_true[index_y_d_in + margin:index_y_u_in - margin,
                                            index_x_d_in + 0:index_x_u_in - margin] = \
                                                seg_in[margin:-margin or None,
                                                       0:-margin or None,
                                                       np.newaxis]
                            if thresholding_for_artificial_class_in_light_version:
                                prediction_true[index_y_d_in + margin:index_y_u_in - margin,
                                                index_x_d_in + 0:index_x_u_in - margin, 1] = \
                                                    seg_in_art[margin:-margin or None,
                                                        0:-margin or None]
                                
                        elif i_batch == nxf - 1 and j_batch != 0 and j_batch != nyf - 1:
                            prediction_true[index_y_d_in + margin:index_y_u_in - margin,
                                            index_x_d_in + margin:index_x_u_in - 0] = \
                                                seg_in[margin:-margin or None,
                                                       margin:,
                                                       np.newaxis]
                            if thresholding_for_artificial_class_in_light_version:
                                prediction_true[index_y_d_in + margin:index_y_u_in - margin,
                                                index_x_d_in + margin:index_x_u_in - 0, 1] = \
                                                    seg_in_art[margin:-margin or None,
                                                        margin:]
                                
                        elif i_batch != 0 and i_batch != nxf - 1 and j_batch == 0:
                            prediction_true[index_y_d_in + 0:index_y_u_in - margin,
                                            index_x_d_in + margin:index_x_u_in - margin] = \
                                                seg_in[0:-margin or None,
                                                       margin:-margin or None,
                                                       np.newaxis]
                            if thresholding_for_artificial_class_in_light_version:
                                prediction_true[index_y_d_in + 0:index_y_u_in - margin,
                                                index_x_d_in + margin:index_x_u_in - margin, 1] = \
                                                    seg_in_art[0:-margin or None,
                                                        margin:-margin or None]
                                
                        elif i_batch != 0 and i_batch != nxf - 1 and j_batch == nyf - 1:
                            prediction_true[index_y_d_in + margin:index_y_u_in - 0,
                                            index_x_d_in + margin:index_x_u_in - margin] = \
                                                seg_in[margin:,
                                                       margin:-margin or None,
                                                       np.newaxis]
                            if thresholding_for_artificial_class_in_light_version:
                                prediction_true[index_y_d_in + margin:index_y_u_in - 0,
                                                index_x_d_in + margin:index_x_u_in - margin, 1] = \
                                                    seg_in_art[margin:,
                                                        margin:-margin or None]
                                
                        else:
                            prediction_true[index_y_d_in + margin:index_y_u_in - margin,
                                            index_x_d_in + margin:index_x_u_in - margin] = \
                                                seg_in[margin:-margin or None,
                                                       margin:-margin or None,
                                                       np.newaxis]
                            if thresholding_for_artificial_class_in_light_version:
                                prediction_true[index_y_d_in + margin:index_y_u_in - margin,
                                                index_x_d_in + margin:index_x_u_in - margin, 1] = \
                                                    seg_in_art[margin:-margin or None,
                                                        margin:-margin or None]
                        indexer_inside_batch += 1


                    list_i_s = []
                    list_j_s = []
                    list_x_u = []
                    list_x_d = []
                    list_y_u = []
                    list_y_d = []

                    batch_indexer = 0
                    img_patch[:] = 0

        prediction_true = prediction_true.astype(np.uint8)
        
        if thresholding_for_artificial_class_in_light_version:
            kernel_min = np.ones((3, 3), np.uint8)
            prediction_true[:,:,0][prediction_true[:,:,0]==2] = 0
            
            skeleton_art = skeletonize(prediction_true[:,:,1])
            skeleton_art = skeleton_art*1
            
            skeleton_art = skeleton_art.astype('uint8')
            
            skeleton_art = cv2.dilate(skeleton_art, kernel_min, iterations=1)

            prediction_true[:,:,0][skeleton_art==1]=2
        #del model
        gc.collect()
        return prediction_true

    def do_prediction_new_concept(
            self, patches, img, model,
            n_batch_inference=1, marginal_of_patch_percent=0.1,
            thresholding_for_some_classes_in_light_version=False,
            thresholding_for_artificial_class_in_light_version=False,
            threshold_art_class_textline=0.1,
            threshold_art_class_layout=0.1):

        self.logger.debug("enter do_prediction_new_concept")
        img_height_model = model.layers[-1].output_shape[1]
        img_width_model = model.layers[-1].output_shape[2]

        if not patches:
            img_h_page = img.shape[0]
            img_w_page = img.shape[1]
            img = img / 255.0
            img = resize_image(img, img_height_model, img_width_model)

            label_p_pred = model.predict(img[np.newaxis], verbose=0)
            seg = np.argmax(label_p_pred, axis=3)[0]

            seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = resize_image(seg_color, img_h_page, img_w_page).astype(np.uint8)
            
            if thresholding_for_artificial_class_in_light_version:
                kernel_min = np.ones((3, 3), np.uint8)
                seg_art = label_p_pred[0,:,:,4]
                seg_art[seg_art<threshold_art_class_layout] =0
                seg_art[seg_art>0] =1
                #seg[seg_art==1]=4
                seg_art = resize_image(seg_art, img_h_page, img_w_page).astype(np.uint8)
                
                prediction_true[:,:,0][prediction_true[:,:,0]==4] = 0
                
                skeleton_art = skeletonize(seg_art)
                skeleton_art = skeleton_art*1
                
                skeleton_art = skeleton_art.astype('uint8')
                
                skeleton_art = cv2.dilate(skeleton_art, kernel_min, iterations=1)
                
                prediction_true[:,:,0][skeleton_art==1] = 4
                
            return prediction_true , resize_image(label_p_pred[0, :, :, 1] , img_h_page, img_w_page)

        if img.shape[0] < img_height_model:
            img = resize_image(img, img_height_model, img.shape[1])
        if img.shape[1] < img_width_model:
            img = resize_image(img, img.shape[0], img_width_model)

        self.logger.debug("Patch size: %sx%s", img_height_model, img_width_model)
        margin = int(marginal_of_patch_percent * img_height_model)
        width_mid = img_width_model - 2 * margin
        height_mid = img_height_model - 2 * margin
        img = img / 255.0
        img = img.astype(np.float16)
        img_h = img.shape[0]
        img_w = img.shape[1]
        prediction_true = np.zeros((img_h, img_w, 3))
        confidence_matrix = np.zeros((img_h, img_w))
        mask_true = np.zeros((img_h, img_w))
        nxf = img_w / float(width_mid)
        nyf = img_h / float(height_mid)
        nxf = int(nxf) + 1 if nxf > int(nxf) else int(nxf)
        nyf = int(nyf) + 1 if nyf > int(nyf) else int(nyf)

        list_i_s = []
        list_j_s = []
        list_x_u = []
        list_x_d = []
        list_y_u = []
        list_y_d = []

        batch_indexer = 0
        img_patch = np.zeros((n_batch_inference, img_height_model, img_width_model, 3))
        for i in range(nxf):
            for j in range(nyf):
                if i == 0:
                    index_x_d = i * width_mid
                    index_x_u = index_x_d + img_width_model
                else:
                    index_x_d = i * width_mid
                    index_x_u = index_x_d + img_width_model
                if j == 0:
                    index_y_d = j * height_mid
                    index_y_u = index_y_d + img_height_model
                else:
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

                img_patch[batch_indexer] = img[index_y_d:index_y_u, index_x_d:index_x_u]
                batch_indexer += 1

                if (batch_indexer == n_batch_inference or
                    # last batch
                    i == nxf - 1 and j == nyf - 1):
                    self.logger.debug("predicting patches on %s", str(img_patch.shape))
                    label_p_pred = model.predict(img_patch,verbose=0)
                    seg = np.argmax(label_p_pred, axis=3)

                    if thresholding_for_some_classes_in_light_version:
                        seg_art = label_p_pred[:,:,:,4]
                        seg_art[seg_art<threshold_art_class_layout] =0
                        seg_art[seg_art>0] =1

                        seg_line = label_p_pred[:,:,:,3]
                        seg_line[seg_line>0.4] =1#seg_line[seg_line>0.5] =1#seg_line[seg_line>0.1] =1
                        seg_line[seg_line<1] =0

                        ##seg[seg_art==1]=4
                        #seg[(seg_line==1) & (seg==0)]=3
                    if thresholding_for_artificial_class_in_light_version:
                        seg_art = label_p_pred[:,:,:,2]

                        seg_art[seg_art<threshold_art_class_textline] = 0
                        seg_art[seg_art>0] =1

                        ##seg[seg_art==1]=2

                    indexer_inside_batch = 0
                    for i_batch, j_batch in zip(list_i_s, list_j_s):
                        seg_in = seg[indexer_inside_batch]
                        
                        if (thresholding_for_artificial_class_in_light_version or
                            thresholding_for_some_classes_in_light_version):
                            seg_in_art = seg_art[indexer_inside_batch]

                        index_y_u_in = list_y_u[indexer_inside_batch]
                        index_y_d_in = list_y_d[indexer_inside_batch]

                        index_x_u_in = list_x_u[indexer_inside_batch]
                        index_x_d_in = list_x_d[indexer_inside_batch]

                        if i_batch == 0 and j_batch == 0:
                            prediction_true[index_y_d_in + 0:index_y_u_in - margin,
                                            index_x_d_in + 0:index_x_u_in - margin] = \
                                                seg_in[0:-margin or None,
                                                       0:-margin or None,
                                                       np.newaxis]
                            confidence_matrix[index_y_d_in + 0:index_y_u_in - margin,
                                            index_x_d_in + 0:index_x_u_in - margin] = \
                                                label_p_pred[0, 0:-margin or None,
                                                       0:-margin or None,
                                                       1]
                            if (thresholding_for_artificial_class_in_light_version or
                                thresholding_for_some_classes_in_light_version):
                                prediction_true[index_y_d_in + 0:index_y_u_in - margin,
                                                index_x_d_in + 0:index_x_u_in - margin, 1] = \
                                                    seg_in_art[0:-margin or None,
                                                        0:-margin or None]
                            
                        elif i_batch == nxf - 1 and j_batch == nyf - 1:
                            prediction_true[index_y_d_in + margin:index_y_u_in - 0,
                                            index_x_d_in + margin:index_x_u_in - 0] = \
                                                seg_in[margin:,
                                                       margin:,
                                                       np.newaxis]
                            confidence_matrix[index_y_d_in + margin:index_y_u_in - 0,
                                            index_x_d_in + margin:index_x_u_in - 0] = \
                                                label_p_pred[0, margin:,
                                                       margin:,
                                                       1]
                            if (thresholding_for_artificial_class_in_light_version or
                                thresholding_for_some_classes_in_light_version):
                                prediction_true[index_y_d_in + margin:index_y_u_in - 0,
                                                index_x_d_in + margin:index_x_u_in - 0, 1] = \
                                                    seg_in_art[margin:,
                                                        margin:]
                            
                        elif i_batch == 0 and j_batch == nyf - 1:
                            prediction_true[index_y_d_in + margin:index_y_u_in - 0,
                                            index_x_d_in + 0:index_x_u_in - margin] = \
                                                seg_in[margin:,
                                                       0:-margin or None,
                                                       np.newaxis]
                            confidence_matrix[index_y_d_in + margin:index_y_u_in - 0,
                                            index_x_d_in + 0:index_x_u_in - margin] = \
                                                label_p_pred[0, margin:,
                                                       0:-margin or None,
                                                       1]
                                            
                            if (thresholding_for_artificial_class_in_light_version or
                                thresholding_for_some_classes_in_light_version):
                                prediction_true[index_y_d_in + margin:index_y_u_in - 0,
                                                index_x_d_in + 0:index_x_u_in - margin, 1] = \
                                                    seg_in_art[margin:,
                                                        0:-margin or None]
                            
                        elif i_batch == nxf - 1 and j_batch == 0:
                            prediction_true[index_y_d_in + 0:index_y_u_in - margin,
                                            index_x_d_in + margin:index_x_u_in - 0] = \
                                                seg_in[0:-margin or None,
                                                       margin:,
                                                       np.newaxis]
                            confidence_matrix[index_y_d_in + 0:index_y_u_in - margin,
                                            index_x_d_in + margin:index_x_u_in - 0] = \
                                                label_p_pred[0, 0:-margin or None,
                                                       margin:,
                                                       1]
                            if (thresholding_for_artificial_class_in_light_version or
                                thresholding_for_some_classes_in_light_version):
                                prediction_true[index_y_d_in + 0:index_y_u_in - margin,
                                                index_x_d_in + margin:index_x_u_in - 0, 1] = \
                                                    seg_in_art[0:-margin or None,
                                                        margin:]
                            
                        elif i_batch == 0 and j_batch != 0 and j_batch != nyf - 1:
                            prediction_true[index_y_d_in + margin:index_y_u_in - margin,
                                            index_x_d_in + 0:index_x_u_in - margin] = \
                                                seg_in[margin:-margin or None,
                                                       0:-margin or None,
                                                       np.newaxis]
                            confidence_matrix[index_y_d_in + margin:index_y_u_in - margin,
                                            index_x_d_in + 0:index_x_u_in - margin] = \
                                                label_p_pred[0, margin:-margin or None,
                                                       0:-margin or None,
                                                       1]
                            if (thresholding_for_artificial_class_in_light_version or
                                thresholding_for_some_classes_in_light_version):
                                prediction_true[index_y_d_in + margin:index_y_u_in - margin,
                                                index_x_d_in + 0:index_x_u_in - margin, 1] = \
                                                    seg_in_art[margin:-margin or None,
                                                        0:-margin or None]
                        elif i_batch == nxf - 1 and j_batch != 0 and j_batch != nyf - 1:
                            prediction_true[index_y_d_in + margin:index_y_u_in - margin,
                                            index_x_d_in + margin:index_x_u_in - 0] = \
                                                seg_in[margin:-margin or None,
                                                       margin:,
                                                       np.newaxis]
                            confidence_matrix[index_y_d_in + margin:index_y_u_in - margin,
                                            index_x_d_in + margin:index_x_u_in - 0] = \
                                                label_p_pred[0, margin:-margin or None,
                                                       margin:,
                                                       1]
                            if (thresholding_for_artificial_class_in_light_version or
                                thresholding_for_some_classes_in_light_version):
                                prediction_true[index_y_d_in + margin:index_y_u_in - margin,
                                                index_x_d_in + margin:index_x_u_in - 0, 1] = \
                                                    seg_in_art[margin:-margin or None,
                                                        margin:]
                        elif i_batch != 0 and i_batch != nxf - 1 and j_batch == 0:
                            prediction_true[index_y_d_in + 0:index_y_u_in - margin,
                                            index_x_d_in + margin:index_x_u_in - margin] = \
                                                seg_in[0:-margin or None,
                                                       margin:-margin or None,
                                                       np.newaxis]
                            confidence_matrix[index_y_d_in + 0:index_y_u_in - margin,
                                            index_x_d_in + margin:index_x_u_in - margin] = \
                                                label_p_pred[0, 0:-margin or None,
                                                       margin:-margin or None,
                                                       1]
                            if (thresholding_for_artificial_class_in_light_version or
                                thresholding_for_some_classes_in_light_version):
                                prediction_true[index_y_d_in + 0:index_y_u_in - margin,
                                                index_x_d_in + margin:index_x_u_in - margin, 1] = \
                                                    seg_in_art[0:-margin or None,
                                                        margin:-margin or None]
                        elif i_batch != 0 and i_batch != nxf - 1 and j_batch == nyf - 1:
                            prediction_true[index_y_d_in + margin:index_y_u_in - 0,
                                            index_x_d_in + margin:index_x_u_in - margin] = \
                                                seg_in[margin:,
                                                       margin:-margin or None,
                                                       np.newaxis]
                            confidence_matrix[index_y_d_in + margin:index_y_u_in - 0,
                                            index_x_d_in + margin:index_x_u_in - margin] = \
                                                label_p_pred[0, margin:,
                                                       margin:-margin or None,
                                                       1]
                            if (thresholding_for_artificial_class_in_light_version or
                                thresholding_for_some_classes_in_light_version):
                                prediction_true[index_y_d_in + margin:index_y_u_in - 0,
                                                index_x_d_in + margin:index_x_u_in - margin, 1] = \
                                                    seg_in_art[margin:,
                                                        margin:-margin or None]
                        else:
                            prediction_true[index_y_d_in + margin:index_y_u_in - margin,
                                            index_x_d_in + margin:index_x_u_in - margin] = \
                                                seg_in[margin:-margin or None,
                                                       margin:-margin or None,
                                                       np.newaxis]
                            confidence_matrix[index_y_d_in + margin:index_y_u_in - margin,
                                            index_x_d_in + margin:index_x_u_in - margin] = \
                                                label_p_pred[0, margin:-margin or None,
                                                       margin:-margin or None,
                                                       1]
                            if (thresholding_for_artificial_class_in_light_version or
                                thresholding_for_some_classes_in_light_version):
                                prediction_true[index_y_d_in + margin:index_y_u_in - margin,
                                                index_x_d_in + margin:index_x_u_in - margin, 1] = \
                                                    seg_in_art[margin:-margin or None,
                                                        margin:-margin or None]
                        indexer_inside_batch += 1

                    list_i_s = []
                    list_j_s = []
                    list_x_u = []
                    list_x_d = []
                    list_y_u = []
                    list_y_d = []

                    batch_indexer = 0
                    img_patch[:] = 0

        prediction_true = prediction_true.astype(np.uint8)
        
        if thresholding_for_artificial_class_in_light_version:
            kernel_min = np.ones((3, 3), np.uint8)
            prediction_true[:,:,0][prediction_true[:,:,0]==2] = 0
            
            skeleton_art = skeletonize(prediction_true[:,:,1])
            skeleton_art = skeleton_art*1
            
            skeleton_art = skeleton_art.astype('uint8')
            
            skeleton_art = cv2.dilate(skeleton_art, kernel_min, iterations=1)

            prediction_true[:,:,0][skeleton_art==1]=2
            
        if thresholding_for_some_classes_in_light_version:
            kernel_min = np.ones((3, 3), np.uint8)
            prediction_true[:,:,0][prediction_true[:,:,0]==4] = 0
            
            skeleton_art = skeletonize(prediction_true[:,:,1])
            skeleton_art = skeleton_art*1
            
            skeleton_art = skeleton_art.astype('uint8')
            
            skeleton_art = cv2.dilate(skeleton_art, kernel_min, iterations=1)

            prediction_true[:,:,0][skeleton_art==1]=4
        gc.collect()
        return prediction_true, confidence_matrix

    def extract_page(self):
        self.logger.debug("enter extract_page")
        cont_page = []
        if not self.ignore_page_extraction:
            img = np.copy(self.image)#cv2.GaussianBlur(self.image, (5, 5), 0)
            img_page_prediction = self.do_prediction(False, img, self.model_zoo.get("page"))
            imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgray, 0, 255, 0)
            ##thresh = cv2.dilate(thresh, KERNEL, iterations=3)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours)>0:
                cnt_size = np.array([cv2.contourArea(contours[j])
                                     for j in range(len(contours))])
                cnt = contours[np.argmax(cnt_size)]
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
            else:
                box = [0, 0, img.shape[1], img.shape[0]]
            cropped_page, page_coord = crop_image_inside_box(box, self.image)
            cont_page = [cnt]
            #cont_page.append(np.array([[page_coord[2], page_coord[0]],
                                       #[page_coord[3], page_coord[0]],
                                       #[page_coord[3], page_coord[1]],
                                       #[page_coord[2], page_coord[1]]]))
            self.logger.debug("exit extract_page")
        else:
            box = [0, 0, self.image.shape[1], self.image.shape[0]]
            cropped_page, page_coord = crop_image_inside_box(box, self.image)
            cont_page.append(np.array([[page_coord[2], page_coord[0]],
                                       [page_coord[3], page_coord[0]],
                                       [page_coord[3], page_coord[1]],
                                       [page_coord[2], page_coord[1]]]))
        return cropped_page, page_coord, cont_page

    def early_page_for_num_of_column_classification(self,img_bin):
        if not self.ignore_page_extraction:
            self.logger.debug("enter early_page_for_num_of_column_classification")
            if self.input_binary:
                img = np.copy(img_bin).astype(np.uint8)
            else:
                img = self.imread()
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img_page_prediction = self.do_prediction(False, img, self.model_zoo.get("page"))

            imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgray, 0, 255, 0)
            thresh = cv2.dilate(thresh, KERNEL, iterations=3)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)>0:
                cnt_size = np.array([cv2.contourArea(contours[j])
                                     for j in range(len(contours))])
                cnt = contours[np.argmax(cnt_size)]
                box = cv2.boundingRect(cnt)
            else:
                box = [0, 0, img.shape[1], img.shape[0]]
            cropped_page, page_coord = crop_image_inside_box(box, img)

            self.logger.debug("exit early_page_for_num_of_column_classification")
        else:
            img = self.imread()
            box = [0, 0, img.shape[1], img.shape[0]]
            cropped_page, page_coord = crop_image_inside_box(box, img)
        return cropped_page, page_coord

    def extract_text_regions_new(self, img, patches, cols):
        self.logger.debug("enter extract_text_regions")
        img_height_h = img.shape[0]
        img_width_h = img.shape[1]
        model_region = self.model_zoo.get("region_fl") if patches else self.model_zoo.get("region_fl_np")

        thresholding_for_fl_light_version = True
        if not patches:
            img = otsu_copy_binary(img).astype(np.uint8)
            prediction_regions = None
            thresholding_for_fl_light_version = False
        elif cols:
            img = otsu_copy_binary(img).astype(np.uint8)
            if cols == 1:
                img = resize_image(img, int(img_height_h * 1000 / float(img_width_h)), 1000).astype(np.uint8)
            elif cols == 2:
                img = resize_image(img, int(img_height_h * 1300 / float(img_width_h)), 1300).astype(np.uint8)
            elif cols == 3:
                img = resize_image(img, int(img_height_h * 1600 / float(img_width_h)), 1600).astype(np.uint8)
            elif cols == 4:
                img = resize_image(img, int(img_height_h * 1900 / float(img_width_h)), 1900).astype(np.uint8)
            elif cols == 5:
                img = resize_image(img, int(img_height_h * 2200 / float(img_width_h)), 2200).astype(np.uint8)
            else:
                img = resize_image(img, int(img_height_h * 2500 / float(img_width_h)), 2500).astype(np.uint8)

        prediction_regions = self.do_prediction(patches, img, model_region,
                                                marginal_of_patch_percent=0.1,
                                                n_batch_inference=3,
                                                thresholding_for_fl_light_version=thresholding_for_fl_light_version)
        prediction_regions = resize_image(prediction_regions, img_height_h, img_width_h)
        self.logger.debug("exit extract_text_regions")
        return prediction_regions, prediction_regions

    def extract_text_regions(self, img, patches, cols):
        self.logger.debug("enter extract_text_regions")
        img_height_h = img.shape[0]
        img_width_h = img.shape[1]
        model_region = self.model_zoo.get("region_fl") if patches else self.model_zoo.get("region_fl_np")

        prediction_regions = self.do_prediction(patches, img, model_region, marginal_of_patch_percent=0.1)
        prediction_regions = resize_image(prediction_regions, img_height_h, img_width_h)
        self.logger.debug("exit extract_text_regions")
        return prediction_regions, None
        
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
                                         mask_texts_only, num_col, scale_par, slope_deskew):
        if not len(contours_par):
            return [], [], []
        self.logger.debug("enter get_slopes_and_deskew_new_curved")
        with share_ndarray(textline_mask_tot) as textline_mask_tot_shared:
            with share_ndarray(mask_texts_only) as mask_texts_only_shared:
                assert self.executor
                results = self.executor.map(partial(do_work_of_slopes_new_curved,
                                            textline_mask_tot_ea=textline_mask_tot_shared,
                                            mask_texts_only=mask_texts_only_shared,
                                            num_col=num_col,
                                            scale_par=scale_par,
                                            slope_deskew=slope_deskew,
                                            MAX_SLOPE=MAX_SLOPE,
                                            KERNEL=KERNEL,
                                            logger=self.logger,
                                            plotter=self.plotter,),
                                    boxes, contours_par)
                results = list(results) # exhaust prior to release
        #textline_polygons, box_coord, slopes = zip(*results)
        self.logger.debug("exit get_slopes_and_deskew_new_curved")
        return tuple(zip(*results))

    def textline_contours(self, img, use_patches, scaler_h, scaler_w, num_col_classifier=None):
        self.logger.debug('enter textline_contours')

        #img = img.astype(np.uint8)
        img_org = np.copy(img)
        img_h = img_org.shape[0]
        img_w = img_org.shape[1]
        img = resize_image(img_org, int(img_org.shape[0] * scaler_h), int(img_org.shape[1] * scaler_w))

        prediction_textline = self.do_prediction(use_patches, img, self.model_zoo.get("textline"),
                                                 marginal_of_patch_percent=0.15,
                                                 n_batch_inference=3,
                                                 threshold_art_class_textline=self.threshold_art_class_textline)

        prediction_textline = resize_image(prediction_textline, img_h, img_w)
        textline_mask_tot_ea_art = (prediction_textline[:,:]==2)*1

        old_art = np.copy(textline_mask_tot_ea_art)
        
        textline_mask_tot_ea_lines = (prediction_textline[:,:]==1)*1
        textline_mask_tot_ea_lines = textline_mask_tot_ea_lines.astype('uint8')

        prediction_textline[:,:][textline_mask_tot_ea_lines[:,:]==1]=1
            
        #cv2.imwrite('prediction_textline2.png', prediction_textline[:,:,0])

        prediction_textline_longshot = self.do_prediction(False, img, self.model_zoo.get("textline"))
        prediction_textline_longshot_true_size = resize_image(prediction_textline_longshot, img_h, img_w)
        
        
        #cv2.imwrite('prediction_textline.png', prediction_textline[:,:,0])
        #sys.exit()
        self.logger.debug('exit textline_contours')
        return ((prediction_textline[:, :, 0]==1).astype(np.uint8),
                (prediction_textline_longshot_true_size[:, :, 0]==1).astype(np.uint8))


    

    def get_regions_light_v(self,img,is_image_enhanced, num_col_classifier):
        self.logger.debug("enter get_regions_light_v")
        t_in = time.time()
        erosion_hurts = False
        img_org = np.copy(img)
        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]

        #print(num_col_classifier,'num_col_classifier')

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
        img_h_new = img_w_new * img_org.shape[0] // img_org.shape[1]
        img_resized = resize_image(img,img_h_new, img_w_new )

        t_bin = time.time()
        #if (not self.input_binary) or self.full_layout:
        #if self.input_binary:
            #img_bin = np.copy(img_resized)
        ###if (not self.input_binary and self.full_layout) or (not self.input_binary and num_col_classifier >= 30):
            ###prediction_bin = self.do_prediction(True, img_resized, self.model_zoo.get_model("binarization"), n_batch_inference=5)

            ####print("inside bin ", time.time()-t_bin)
            ###prediction_bin=prediction_bin[:,:,0]
            ###prediction_bin = (prediction_bin[:,:]==0)*1
            ###prediction_bin = prediction_bin*255

            ###prediction_bin =np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2)

            ###prediction_bin = prediction_bin.astype(np.uint16)
            ####img= np.copy(prediction_bin)
            ###img_bin = np.copy(prediction_bin)
        ###else:
            ###img_bin = np.copy(img_resized)
        img_bin = np.copy(img_resized)
        #print("inside 1 ", time.time()-t_in)

        ###textline_mask_tot_ea = self.run_textline(img_bin)
        self.logger.debug("detecting textlines on %s with %d colors",
                          str(img_resized.shape), len(np.unique(img_resized)))
        textline_mask_tot_ea = self.run_textline(img_resized, num_col_classifier)
        textline_mask_tot_ea = resize_image(textline_mask_tot_ea,img_height_h, img_width_h )

        #print(self.image_org.shape)
        #cv2.imwrite('textline.png', textline_mask_tot_ea)

        #plt.imshwo(self.image_page_org_size)
        #plt.show()
        if self.skip_layout_and_reading_order:
            img_bin = resize_image(img_bin,img_height_h, img_width_h )
            self.logger.debug("exit get_regions_light_v")
            return None, erosion_hurts, None, None, textline_mask_tot_ea, img_bin, None

        #print("inside 2 ", time.time()-t_in)
        if num_col_classifier == 1 or num_col_classifier == 2:
            if self.image_org.shape[0]/self.image_org.shape[1] > 2.5:
                self.logger.debug("resized to %dx%d for %d cols",
                                  img_resized.shape[1], img_resized.shape[0], num_col_classifier)
                prediction_regions_org, confidence_matrix = self.do_prediction_new_concept(
                    True, img_resized, self.model_zoo.get("region_1_2"), n_batch_inference=1,
                    thresholding_for_some_classes_in_light_version=True,
                    threshold_art_class_layout=self.threshold_art_class_layout)
            else:
                prediction_regions_org = np.zeros((self.image_org.shape[0], self.image_org.shape[1], 3))
                confidence_matrix = np.zeros((self.image_org.shape[0], self.image_org.shape[1]))
                prediction_regions_page, confidence_matrix_page = self.do_prediction_new_concept(
                    False, self.image_page_org_size, self.model_zoo.get("region_1_2"), n_batch_inference=1,
                    thresholding_for_artificial_class_in_light_version=True,
                    threshold_art_class_layout=self.threshold_art_class_layout)
                ys = slice(*self.page_coord[0:2])
                xs = slice(*self.page_coord[2:4])
                prediction_regions_org[ys, xs] = prediction_regions_page
                confidence_matrix[ys, xs] = confidence_matrix_page

        else:
            new_h = (900+ (num_col_classifier-3)*100)
            img_resized = resize_image(img_bin, int(new_h * img_bin.shape[0] /img_bin.shape[1]), new_h)
            self.logger.debug("resized to %dx%d (new_h=%d) for %d cols",
                              img_resized.shape[1], img_resized.shape[0], new_h, num_col_classifier)
            prediction_regions_org, confidence_matrix = self.do_prediction_new_concept(
                True, img_resized, self.model_zoo.get("region_1_2"), n_batch_inference=2,
                thresholding_for_some_classes_in_light_version=True,
                threshold_art_class_layout=self.threshold_art_class_layout)
        ###prediction_regions_org = self.do_prediction(True, img_bin, self.model_zoo.get_model("region"),
        ###n_batch_inference=3,
        ###thresholding_for_some_classes_in_light_version=True)
        #print("inside 3 ", time.time()-t_in)
        #plt.imshow(prediction_regions_org[:,:,0])
        #plt.show()

        prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h )
        confidence_matrix = resize_image(confidence_matrix, img_height_h, img_width_h )
        img_bin = resize_image(img_bin, img_height_h, img_width_h )
        prediction_regions_org=prediction_regions_org[:,:,0]

        mask_lines_only = (prediction_regions_org[:,:] ==3)*1
        mask_texts_only = (prediction_regions_org[:,:] ==1)*1
        mask_texts_only = mask_texts_only.astype('uint8')

        ##if num_col_classifier == 1 or num_col_classifier == 2:
            ###mask_texts_only = cv2.erode(mask_texts_only, KERNEL, iterations=1)
            ##mask_texts_only = cv2.dilate(mask_texts_only, KERNEL, iterations=1)

        mask_texts_only = cv2.dilate(mask_texts_only, kernel=np.ones((2,2), np.uint8), iterations=1)
        mask_images_only=(prediction_regions_org[:,:] ==2)*1

        polygons_seplines, hir_seplines = return_contours_of_image(mask_lines_only)
        test_khat = np.zeros(prediction_regions_org.shape)
        test_khat = cv2.fillPoly(test_khat, pts=polygons_seplines, color=(1,1,1))

        #plt.imshow(test_khat[:,:])
        #plt.show()
        #for jv in range(1):
            #print(jv, hir_seplines[0][232][3])
            #test_khat = np.zeros(prediction_regions_org.shape)
            #test_khat = cv2.fillPoly(test_khat, pts = [polygons_seplines[232]], color=(1,1,1))
            #plt.imshow(test_khat[:,:])
            #plt.show()

        polygons_seplines = filter_contours_area_of_image(
            mask_lines_only, polygons_seplines, hir_seplines, max_area=1, min_area=0.00001, dilate=1)

        test_khat = np.zeros(prediction_regions_org.shape)
        test_khat = cv2.fillPoly(test_khat, pts = polygons_seplines, color=(1,1,1))

        #plt.imshow(test_khat[:,:])
        #plt.show()
        #sys.exit()

        polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only,1,0.00001)
        ##polygons_of_only_texts = dilate_textregion_contours(polygons_of_only_texts)
        polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only,1,0.00001)

        text_regions_p_true = np.zeros(prediction_regions_org.shape)
        text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts=polygons_of_only_lines, color=(3,3,3))

        text_regions_p_true[:,:][mask_images_only[:,:] == 1] = 2
        text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_texts, color=(1,1,1))

        textline_mask_tot_ea[(text_regions_p_true==0) | (text_regions_p_true==4) ] = 0
        #plt.imshow(textline_mask_tot_ea)
        #plt.show()
        #print("inside 4 ", time.time()-t_in)
        self.logger.debug("exit get_regions_light_v")
        return (text_regions_p_true,
                erosion_hurts,
                polygons_seplines,
                polygons_of_only_texts,
                textline_mask_tot_ea,
                img_bin,
                confidence_matrix)

    def do_order_of_regions(
            self, contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot):

        self.logger.debug("enter do_order_of_regions")
        contours_only_text_parent = np.array(contours_only_text_parent)
        contours_only_text_parent_h = np.array(contours_only_text_parent_h)
        boxes = np.array(boxes, dtype=int) # to be on the safe side
        c_boxes = np.stack((0.5 * boxes[:, 2:4].sum(axis=1),
                            0.5 * boxes[:, 0:2].sum(axis=1)))
        cx_main, cy_main, mx_main, Mx_main, my_main, My_main, mxy_main = find_new_features_of_contours(
            contours_only_text_parent)
        cx_head, cy_head, mx_head, Mx_head, my_head, My_head, mxy_head = find_new_features_of_contours(
            contours_only_text_parent_h)

        def match_boxes(only_centers: bool):
            arg_text_con_main = np.zeros(len(contours_only_text_parent), dtype=int)
            for ii in range(len(contours_only_text_parent)):
                check_if_textregion_located_in_a_box = False
                for jj, box in enumerate(boxes):
                    if ((cx_main[ii] >= box[0] and
                         cx_main[ii] < box[1] and
                         cy_main[ii] >= box[2] and
                         cy_main[ii] < box[3]) if only_centers else
                        (mx_main[ii] >= box[0] and
                         Mx_main[ii] < box[1] and
                         my_main[ii] >= box[2] and
                         My_main[ii] < box[3])):
                        arg_text_con_main[ii] = jj
                        check_if_textregion_located_in_a_box = True
                        break
                if not check_if_textregion_located_in_a_box:
                    dists_tr_from_box = np.linalg.norm(c_boxes - np.array([[cy_main[ii]], [cx_main[ii]]]), axis=0)
                    pcontained_in_box = ((boxes[:, 2] <= cy_main[ii]) & (cy_main[ii] < boxes[:, 3]) &
                                         (boxes[:, 0] <= cx_main[ii]) & (cx_main[ii] < boxes[:, 1]))
                    ind_min = np.argmin(np.ma.masked_array(dists_tr_from_box, ~pcontained_in_box))
                    arg_text_con_main[ii] = ind_min
            args_contours_main = np.arange(len(contours_only_text_parent))
            order_by_con_main = np.zeros_like(arg_text_con_main)

            arg_text_con_head = np.zeros(len(contours_only_text_parent_h), dtype=int)
            for ii in range(len(contours_only_text_parent_h)):
                check_if_textregion_located_in_a_box = False
                for jj, box in enumerate(boxes):
                    if ((cx_head[ii] >= box[0] and
                         cx_head[ii] < box[1] and
                         cy_head[ii] >= box[2] and
                         cy_head[ii] < box[3]) if only_centers else
                        (mx_head[ii] >= box[0] and
                         Mx_head[ii] < box[1] and
                         my_head[ii] >= box[2] and
                         My_head[ii] < box[3])):
                        arg_text_con_head[ii] = jj
                        check_if_textregion_located_in_a_box = True
                        break
                if not check_if_textregion_located_in_a_box:
                    dists_tr_from_box = np.linalg.norm(c_boxes - np.array([[cy_head[ii]], [cx_head[ii]]]), axis=0)
                    pcontained_in_box = ((boxes[:, 2] <= cy_head[ii]) & (cy_head[ii] < boxes[:, 3]) &
                                         (boxes[:, 0] <= cx_head[ii]) & (cx_head[ii] < boxes[:, 1]))
                    ind_min = np.argmin(np.ma.masked_array(dists_tr_from_box, ~pcontained_in_box))
                    arg_text_con_head[ii] = ind_min
            args_contours_head = np.arange(len(contours_only_text_parent_h))
            order_by_con_head = np.zeros_like(arg_text_con_head)

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij, box in enumerate(boxes):
                ys = slice(*box[2:4])
                xs = slice(*box[0:2])
                args_contours_box_main = args_contours_main[arg_text_con_main == iij]
                args_contours_box_head = args_contours_head[arg_text_con_head == iij]
                con_inter_box = contours_only_text_parent[args_contours_box_main]
                con_inter_box_h = contours_only_text_parent_h[args_contours_box_head]

                indexes_sorted, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(
                    textline_mask_tot[ys, xs], con_inter_box, con_inter_box_h, box[2])

                order_of_texts, id_of_texts = order_and_id_of_texts(
                    con_inter_box, con_inter_box_h,
                    indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = indexes_sorted[kind_of_texts_sorted == 1]
                indexes_by_type_main = index_by_kind_sorted[kind_of_texts_sorted == 1]
                indexes_sorted_head = indexes_sorted[kind_of_texts_sorted == 2]
                indexes_by_type_head = index_by_kind_sorted[kind_of_texts_sorted == 2]

                for zahler, _ in enumerate(args_contours_box_main):
                    arg_order_v = indexes_sorted_main[zahler]
                    order_by_con_main[args_contours_box_main[indexes_by_type_main[zahler]]] = \
                        np.flatnonzero(indexes_sorted == arg_order_v) + ref_point

                for zahler, _ in enumerate(args_contours_box_head):
                    arg_order_v = indexes_sorted_head[zahler]
                    order_by_con_head[args_contours_box_head[indexes_by_type_head[zahler]]] = \
                        np.flatnonzero(indexes_sorted == arg_order_v) + ref_point

                for jji in range(len(id_of_texts)):
                    order_of_texts_tot.append(order_of_texts[jji] + ref_point)
                    id_of_texts_tot.append(id_of_texts[jji])
                ref_point += len(id_of_texts)

            order_of_texts_tot = np.concatenate((order_by_con_main,
                                                 order_by_con_head))
            order_text_new = np.argsort(order_of_texts_tot)
            return order_text_new, id_of_texts_tot

        try:
            results = match_boxes(False)
        except Exception as why:
            self.logger.error(why)
            results = match_boxes(True)

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

    def delete_separator_around(self, spliter_y,peaks_neg,image_by_region, pixel_line, pixel_table):
        # format of subboxes: box=[x1, x2 , y1, y2]
        pix_del = 100
        if len(image_by_region.shape)==3:
            for i in range(len(spliter_y)-1):
                for j in range(1,len(peaks_neg[i])-1):
                    ys = slice(int(spliter_y[i]),
                               int(spliter_y[i+1]))
                    xs = slice(peaks_neg[i][j] - pix_del,
                               peaks_neg[i][j] + pix_del)
                    image_by_region[ys,xs,0][image_by_region[ys,xs,0]==pixel_line] = 0
                    image_by_region[ys,xs,0][image_by_region[ys,xs,1]==pixel_line] = 0
                    image_by_region[ys,xs,0][image_by_region[ys,xs,2]==pixel_line] = 0

                    image_by_region[ys,xs,0][image_by_region[ys,xs,0]==pixel_table] = 0
                    image_by_region[ys,xs,0][image_by_region[ys,xs,1]==pixel_table] = 0
                    image_by_region[ys,xs,0][image_by_region[ys,xs,2]==pixel_table] = 0
        else:
            for i in range(len(spliter_y)-1):
                for j in range(1,len(peaks_neg[i])-1):
                    ys = slice(int(spliter_y[i]),
                               int(spliter_y[i+1]))
                    xs = slice(peaks_neg[i][j] - pix_del,
                               peaks_neg[i][j] + pix_del)
                    image_by_region[ys,xs][image_by_region[ys,xs]==pixel_line] = 0
                    image_by_region[ys,xs][image_by_region[ys,xs]==pixel_table] = 0
        return image_by_region

    def add_tables_heuristic_to_layout(
            self, image_regions_eraly_p, boxes,
            slope_mean_hor, spliter_y, peaks_neg_tot, image_revised,
            num_col_classifier, min_area, pixel_line):

        pixel_table =10
        image_revised_1 = self.delete_separator_around(spliter_y, peaks_neg_tot, image_revised, pixel_line, pixel_table)

        try:
            image_revised_1[:,:30][image_revised_1[:,:30]==pixel_line] = 0
            image_revised_1[:,-30:][image_revised_1[:,-30:]==pixel_line] = 0
        except:
            pass
        boxes = np.array(boxes, dtype=int) # to be on the safe side

        img_comm = np.zeros(image_revised_1.shape, dtype=np.uint8)
        for indiv in np.unique(image_revised_1):
            image_col = (image_revised_1 == indiv).astype(np.uint8) * 255
            _, thresh = cv2.threshold(image_col, 0, 255, 0)
            contours,hirarchy=cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            if indiv==pixel_table:
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
                    image_box_tabels_1 = (image_box == pixel_table) * 1
                    contours_tab,_=return_contours_of_image(image_box_tabels_1)
                    contours_tab=filter_contours_area_of_image_tables(image_box_tabels_1,contours_tab,_,1,0.003)
                    image_box_tabels_1 = (image_box == pixel_line).astype(np.uint8) * 1
                    image_box_tabels_and_m_text = ( (image_box == pixel_table) |
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
                    image_box[y_up_tabs[ii]:y_down_tabs[ii]] = pixel_table

                image_revised_last[box_ys, box_xs] = image_box
        else:
            for i in range(len(boxes)):
                box_ys = slice(*boxes[i][2:4])
                box_xs = slice(*boxes[i][0:2])
                image_box = img_comm[box_ys, box_xs]
                image_revised_last[box_ys, box_xs] = image_box

        if num_col_classifier==1:
            img_tables_col_1 = (image_revised_last == pixel_table).astype(np.uint8)
            contours_table_col1, _ = return_contours_of_image(img_tables_col_1)

            _,_ ,_ , _, y_min_tab_col1 ,y_max_tab_col1, _= find_new_features_of_contours(contours_table_col1)

            if len(y_min_tab_col1)>0:
                for ijv in range(len(y_min_tab_col1)):
                    image_revised_last[int(y_min_tab_col1[ijv]):int(y_max_tab_col1[ijv])] = pixel_table
        return image_revised_last

    def get_tables_from_model(self, img, num_col_classifier):
        img_org = np.copy(img)
        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]
        patches = False
        prediction_table, _ = self.do_prediction_new_concept(patches, img, self.model_zoo.get("table"))
        prediction_table = prediction_table.astype(np.int16)
        return prediction_table[:,:,0]

    def run_graphics_and_columns_light(
            self, text_regions_p_1, textline_mask_tot_ea,
            num_col_classifier, num_column_is_classified, erosion_hurts, img_bin_light):

        #print(text_regions_p_1.shape, 'text_regions_p_1 shape run graphics')
        #print(erosion_hurts, 'erosion_hurts')
        t_in_gr = time.time()
        img_g = self.imread(grayscale=True, uint8=True)

        img_g3 = np.zeros((img_g.shape[0], img_g.shape[1], 3))
        img_g3 = img_g3.astype(np.uint8)
        img_g3[:, :, 0] = img_g[:, :]
        img_g3[:, :, 1] = img_g[:, :]
        img_g3[:, :, 2] = img_g[:, :]

        image_page, page_coord, cont_page = self.extract_page()
        #print("inside graphics 1 ", time.time() - t_in_gr)
        if self.tables:
            table_prediction = self.get_tables_from_model(image_page, num_col_classifier)
        else:
            table_prediction = np.zeros((image_page.shape[0], image_page.shape[1])).astype(np.int16)

        if self.plotter:
            self.plotter.save_page_image(image_page)
        
        if not self.ignore_page_extraction:
            mask_page = np.zeros((text_regions_p_1.shape[0], text_regions_p_1.shape[1])).astype(np.int8)
            mask_page = cv2.fillPoly(mask_page, pts=[cont_page[0]], color=(1,))
            
            text_regions_p_1[mask_page==0] = 0
            textline_mask_tot_ea[mask_page==0] = 0
        
        text_regions_p_1 = text_regions_p_1[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        textline_mask_tot_ea = textline_mask_tot_ea[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        img_bin_light = img_bin_light[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        
        ###text_regions_p_1 = text_regions_p_1[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        ###textline_mask_tot_ea = textline_mask_tot_ea[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        ###img_bin_light = img_bin_light[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]

        mask_images = (text_regions_p_1[:, :] == 2) * 1
        mask_images = mask_images.astype(np.uint8)
        mask_images = cv2.erode(mask_images[:, :], KERNEL, iterations=10)
        mask_lines = (text_regions_p_1[:, :] == 3) * 1
        mask_lines = mask_lines.astype(np.uint8)
        img_only_regions_with_sep = ((text_regions_p_1[:, :] != 3) & (text_regions_p_1[:, :] != 0)) * 1
        img_only_regions_with_sep = img_only_regions_with_sep.astype(np.uint8)

        #print("inside graphics 2 ", time.time() - t_in_gr)
        if erosion_hurts:
            img_only_regions = np.copy(img_only_regions_with_sep[:,:])
        else:
            img_only_regions = cv2.erode(img_only_regions_with_sep[:,:], KERNEL, iterations=6)

        ##print(img_only_regions.shape,'img_only_regions')
        ##plt.imshow(img_only_regions[:,:])
        ##plt.show()
        ##num_col, _ = find_num_col(img_only_regions, num_col_classifier, self.tables, multiplier=6.0)
        try:
            num_col, _ = find_num_col(img_only_regions, num_col_classifier, self.tables, multiplier=6.0)
            num_col = num_col + 1
            if not num_column_is_classified:
                num_col_classifier = num_col + 1
            num_col_classifier = min(self.num_col_upper or num_col_classifier,
                                     max(self.num_col_lower or num_col_classifier,
                                         num_col_classifier))
        except Exception as why:
            self.logger.error(why)
            num_col = None
        #print("inside graphics 3 ", time.time() - t_in_gr)
        return (num_col, num_col_classifier, img_only_regions, page_coord, image_page, mask_images, mask_lines,
                text_regions_p_1, cont_page, table_prediction, textline_mask_tot_ea, img_bin_light)

    def run_graphics_and_columns_without_layout(self, textline_mask_tot_ea, img_bin_light):
        #print(text_regions_p_1.shape, 'text_regions_p_1 shape run graphics')
        #print(erosion_hurts, 'erosion_hurts')
        t_in_gr = time.time()
        img_g = self.imread(grayscale=True, uint8=True)

        img_g3 = np.zeros((img_g.shape[0], img_g.shape[1], 3))
        img_g3 = img_g3.astype(np.uint8)
        img_g3[:, :, 0] = img_g[:, :]
        img_g3[:, :, 1] = img_g[:, :]
        img_g3[:, :, 2] = img_g[:, :]

        image_page, page_coord, cont_page = self.extract_page()
        #print("inside graphics 1 ", time.time() - t_in_gr)

        textline_mask_tot_ea = textline_mask_tot_ea[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        img_bin_light = img_bin_light[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]

        return  page_coord, image_page, textline_mask_tot_ea, img_bin_light, cont_page


    def run_enhancement(self):
        t_in = time.time()
        self.logger.info("Resizing and enhancing image...")
        is_image_enhanced, img_org, img_res, num_col_classifier, num_column_is_classified, img_bin = \
            self.resize_and_enhance_image_with_column_classifier()
        self.logger.info("Image was %senhanced.", '' if is_image_enhanced else 'not ')
        scale = 1
        if is_image_enhanced:
            if self.allow_enhancement:
                #img_res = img_res.astype(np.uint8)
                self.get_image_and_scales(img_org, img_res, scale)
                if self.plotter:
                    self.plotter.save_enhanced_image(img_res)
            else:
                self.get_image_and_scales_after_enhancing(img_org, img_res)
        else:
            self.get_image_and_scales(img_org, img_res, scale)
            if self.allow_scaling:
                img_org, img_res, is_image_enhanced = \
                    self.resize_image_with_column_classifier(is_image_enhanced, img_bin)
                self.get_image_and_scales_after_enhancing(img_org, img_res)
        #print("enhancement in ", time.time()-t_in)
        return img_res, is_image_enhanced, num_col_classifier, num_column_is_classified

    def run_textline(self, image_page, num_col_classifier=None):
        scaler_h_textline = 1#1.3  # 1.2#1.2
        scaler_w_textline = 1#1.3  # 0.9#1
        #print(image_page.shape)
        textline_mask_tot_ea, _ = self.textline_contours(image_page, True,
                                                         scaler_h_textline,
                                                         scaler_w_textline,
                                                         num_col_classifier)
        textline_mask_tot_ea = textline_mask_tot_ea.astype(np.int16)

        if self.plotter:
            self.plotter.save_plot_of_textlines(textline_mask_tot_ea, image_page)
        return textline_mask_tot_ea

    def run_deskew(self, textline_mask_tot_ea):
        #print(textline_mask_tot_ea.shape, 'textline_mask_tot_ea deskew')
        slope_deskew = return_deskew_slop(cv2.erode(textline_mask_tot_ea, KERNEL, iterations=2), 2, 30, True,
                                          map=self.executor.map, logger=self.logger, plotter=self.plotter)
        if self.plotter:
            self.plotter.save_deskewed_image(slope_deskew)
        self.logger.info("slope_deskew: %.2f°", slope_deskew)
        return slope_deskew

    def run_marginals(
            self, textline_mask_tot_ea, mask_images, mask_lines,
            num_col_classifier, slope_deskew, text_regions_p_1, table_prediction):

        textline_mask_tot = textline_mask_tot_ea[:, :]
        textline_mask_tot[mask_images[:, :] == 1] = 0

        text_regions_p_1[mask_lines[:, :] == 1] = 3
        text_regions_p = text_regions_p_1[:, :]
        text_regions_p = np.array(text_regions_p)
        if num_col_classifier in (1, 2):
            try:
                regions_without_separators = (text_regions_p[:, :] == 1) * 1
                if self.tables:
                    regions_without_separators[table_prediction==1] = 1
                regions_without_separators = regions_without_separators.astype(np.uint8)
                text_regions_p = get_marginals(
                    rotate_image(regions_without_separators, slope_deskew), text_regions_p,
                    num_col_classifier, slope_deskew, kernel=KERNEL)
            except Exception as e:
                self.logger.error("exception %s", e)

        return textline_mask_tot, text_regions_p

    def run_boxes_no_full_layout(
            self, image_page, textline_mask_tot, text_regions_p,
            slope_deskew, num_col_classifier, table_prediction, erosion_hurts):

        self.logger.debug('enter run_boxes_no_full_layout')
        t_0_box = time.time()
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            _, textline_mask_tot_d, text_regions_p_1_n, table_prediction_n = rotation_not_90_func(
                image_page, textline_mask_tot, text_regions_p, table_prediction, slope_deskew)
            text_regions_p_1_n = resize_image(text_regions_p_1_n, text_regions_p.shape[0], text_regions_p.shape[1])
            textline_mask_tot_d = resize_image(textline_mask_tot_d, text_regions_p.shape[0], text_regions_p.shape[1])
            table_prediction_n = resize_image(table_prediction_n, text_regions_p.shape[0], text_regions_p.shape[1])
            regions_without_separators_d = (text_regions_p_1_n[:, :] == 1) * 1
            if self.tables:
                regions_without_separators_d[table_prediction_n[:,:] == 1] = 1
        regions_without_separators = (text_regions_p[:, :] == 1) * 1
        # ( (text_regions_p[:,:]==1) | (text_regions_p[:,:]==2) )*1
        #self.return_regions_without_separators_new(text_regions_p[:,:,0],img_only_regions)
        #print(time.time()-t_0_box,'time box in 1')
        if self.tables:
            regions_without_separators[table_prediction ==1 ] = 1
        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
            text_regions_p_1_n = None
            textline_mask_tot_d = None
            regions_without_separators_d = None
        pixel_lines = 3
        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
            _, _, matrix_of_lines_ch, splitter_y_new, _ = find_number_of_columns_in_document(
                text_regions_p, num_col_classifier, self.tables, pixel_lines)

        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            _, _, matrix_of_lines_ch_d, splitter_y_new_d, _ = find_number_of_columns_in_document(
                text_regions_p_1_n, num_col_classifier, self.tables, pixel_lines)
        #print(time.time()-t_0_box,'time box in 2')
        self.logger.info("num_col_classifier: %s", num_col_classifier)

        if num_col_classifier >= 3:
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                regions_without_separators = regions_without_separators.astype(np.uint8)
                regions_without_separators = cv2.erode(regions_without_separators[:, :], KERNEL, iterations=6)
            else:
                regions_without_separators_d = regions_without_separators_d.astype(np.uint8)
                regions_without_separators_d = cv2.erode(regions_without_separators_d[:, :], KERNEL, iterations=6)
        #print(time.time()-t_0_box,'time box in 3')
        t1 = time.time()
        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
            boxes, peaks_neg_tot_tables = return_boxes_of_images_by_order_of_reading_new(
                splitter_y_new, regions_without_separators, matrix_of_lines_ch,
                num_col_classifier, erosion_hurts, self.tables, self.right2left)
            boxes_d = None
            self.logger.debug("len(boxes): %s", len(boxes))
            #print(time.time()-t_0_box,'time box in 3.1')

        else:
            boxes_d, peaks_neg_tot_tables_d = return_boxes_of_images_by_order_of_reading_new(
                splitter_y_new_d, regions_without_separators_d, matrix_of_lines_ch_d,
                num_col_classifier, erosion_hurts, self.tables, self.right2left)
            boxes = None
            self.logger.debug("len(boxes): %s", len(boxes_d))

        #print(time.time()-t_0_box,'time box in 4')
        self.logger.info("detecting boxes took %.1fs", time.time() - t1)

        if self.tables:
            text_regions_p[table_prediction == 1] = 10
            img_revised_tab = text_regions_p[:,:]
        else:
            img_revised_tab = text_regions_p[:,:]
        #img_revised_tab = text_regions_p[:, :]
        polygons_of_images = return_contours_of_interested_region(text_regions_p, 2)

        pixel_img = 4
        min_area_mar = 0.00001
        marginal_mask = (text_regions_p[:,:]==pixel_img)*1
        marginal_mask = marginal_mask.astype('uint8')
        marginal_mask = cv2.dilate(marginal_mask, KERNEL, iterations=2)

        polygons_of_marginals = return_contours_of_interested_region(marginal_mask, 1, min_area_mar)

        pixel_img = 10
        contours_tables = return_contours_of_interested_region(text_regions_p, pixel_img, min_area_mar)
        #print(time.time()-t_0_box,'time box in 5')
        self.logger.debug('exit run_boxes_no_full_layout')
        return (polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d,
                regions_without_separators_d, boxes, boxes_d,
                polygons_of_marginals, contours_tables)

    def run_boxes_full_layout(
            self, image_page, textline_mask_tot, text_regions_p,
            slope_deskew, num_col_classifier, img_only_regions,
            table_prediction, erosion_hurts, img_bin_light):

        self.logger.debug('enter run_boxes_full_layout')
        t_full0 = time.time()
        if self.tables:
            text_regions_p[:,:][table_prediction[:,:]==1] = 10
            img_revised_tab = text_regions_p[:,:]
            if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                _, textline_mask_tot_d, text_regions_p_1_n, table_prediction_n = \
                    rotation_not_90_func(image_page, textline_mask_tot, text_regions_p,
                                         table_prediction, slope_deskew)

                text_regions_p_1_n = resize_image(text_regions_p_1_n,
                                                  text_regions_p.shape[0],
                                                  text_regions_p.shape[1])
                textline_mask_tot_d = resize_image(textline_mask_tot_d,
                                                   text_regions_p.shape[0],
                                                   text_regions_p.shape[1])
                table_prediction_n = resize_image(table_prediction_n,
                                                  text_regions_p.shape[0],
                                                  text_regions_p.shape[1])

                regions_without_separators_d = (text_regions_p_1_n[:,:] == 1)*1
                regions_without_separators_d[table_prediction_n[:,:] == 1] = 1
            else:
                text_regions_p_1_n = None
                textline_mask_tot_d = None
                regions_without_separators_d = None
            # regions_without_separators = ( text_regions_p[:,:]==1 | text_regions_p[:,:]==2 )*1
            #self.return_regions_without_separators_new(text_regions_p[:,:,0],img_only_regions)
            regions_without_separators = (text_regions_p[:,:] == 1)*1
            regions_without_separators[table_prediction == 1] = 1


        pixel_img = 4
        min_area_mar = 0.00001

        marginal_mask = (text_regions_p[:,:]==pixel_img)*1
        marginal_mask = marginal_mask.astype('uint8')
        marginal_mask = cv2.dilate(marginal_mask, KERNEL, iterations=2)

        polygons_of_marginals = return_contours_of_interested_region(marginal_mask, 1, min_area_mar)

        pixel_img = 10
        contours_tables = return_contours_of_interested_region(text_regions_p, pixel_img, min_area_mar)

        # set first model with second model
        text_regions_p[:, :][text_regions_p[:, :] == 2] = 5
        text_regions_p[:, :][text_regions_p[:, :] == 3] = 6
        text_regions_p[:, :][text_regions_p[:, :] == 4] = 8

        image_page = image_page.astype(np.uint8)
        #print("full inside 1", time.time()- t_full0)
        regions_fully, regions_fully_only_drop = self.extract_text_regions_new(
            img_bin_light,
            False, cols=num_col_classifier)
        #print("full inside 2", time.time()- t_full0)
        # 6 is the separators lable in old full layout model
        # 4 is the drop capital class in old full layout model
        # in the new full layout drop capital is 3 and separators are 5
        
        # the separators in full layout will not be written on layout
        if not self.reading_order_machine_based:
            text_regions_p[:,:][regions_fully[:,:,0]==5]=6
        ###regions_fully[:, :, 0][regions_fully_only_drop[:, :, 0] == 3] = 4

        #text_regions_p[:,:][regions_fully[:,:,0]==6]=6
        ##regions_fully_only_drop = put_drop_out_from_only_drop_model(regions_fully_only_drop, text_regions_p)
        ##regions_fully[:, :, 0][regions_fully_only_drop[:, :] == 4] = 4
        drop_capital_label_in_full_layout_model = 3

        drops = (regions_fully[:,:,0]==drop_capital_label_in_full_layout_model)*1
        drops= drops.astype(np.uint8)

        regions_fully[:,:,0][regions_fully[:,:,0]==drop_capital_label_in_full_layout_model] = 1

        drops = cv2.erode(drops[:,:], KERNEL, iterations=1)
        regions_fully[:,:,0][drops[:,:]==1] = drop_capital_label_in_full_layout_model

        regions_fully = putt_bb_of_drop_capitals_of_model_in_patches_in_layout(
            regions_fully, drop_capital_label_in_full_layout_model, text_regions_p)
        ##regions_fully_np, _ = self.extract_text_regions(image_page, False, cols=num_col_classifier)
        ##if num_col_classifier > 2:
            ##regions_fully_np[:, :, 0][regions_fully_np[:, :, 0] == 4] = 0
        ##else:
            ##regions_fully_np = filter_small_drop_capitals_from_no_patch_layout(regions_fully_np, text_regions_p)

        ###regions_fully = boosting_headers_by_longshot_region_segmentation(regions_fully,
        ###    regions_fully_np, img_only_regions)
        # plt.imshow(regions_fully[:,:,0])
        # plt.show()
        text_regions_p[:, :][regions_fully[:, :, 0] == drop_capital_label_in_full_layout_model] = 4
        ####text_regions_p[:, :][regions_fully_np[:, :, 0] == 4] = 4
        #plt.imshow(text_regions_p)
        #plt.show()
        ####if not self.tables:
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            _, textline_mask_tot_d, text_regions_p_1_n, regions_fully_n = rotation_not_90_func_full_layout(
                image_page, textline_mask_tot, text_regions_p, regions_fully, slope_deskew)

            text_regions_p_1_n = resize_image(text_regions_p_1_n, text_regions_p.shape[0], text_regions_p.shape[1])
            textline_mask_tot_d = resize_image(textline_mask_tot_d, text_regions_p.shape[0], text_regions_p.shape[1])
            regions_fully_n = resize_image(regions_fully_n, text_regions_p.shape[0], text_regions_p.shape[1])
            if not self.tables:
                regions_without_separators_d = (text_regions_p_1_n[:, :] == 1) * 1
        else:
            text_regions_p_1_n = None
            textline_mask_tot_d = None
            regions_without_separators_d = None
        if not self.tables:
            regions_without_separators = (text_regions_p[:, :] == 1) * 1
        img_revised_tab = np.copy(text_regions_p[:, :])
        polygons_of_images = return_contours_of_interested_region(img_revised_tab, 5)

        self.logger.debug('exit run_boxes_full_layout')
        #print("full inside 3", time.time()- t_full0)
        return (polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d,
                regions_without_separators_d, regions_fully, regions_without_separators,
                polygons_of_marginals, contours_tables)

    def do_order_of_regions_with_model(self, contours_only_text_parent, contours_only_text_parent_h, text_regions_p):
        
        height1 =672#448
        width1 = 448#224

        height2 =672#448
        width2= 448#224

        height3 =672#448
        width3 = 448#224
        
        inference_bs = 3
        
        ver_kernel = np.ones((5, 1), dtype=np.uint8)
        hor_kernel = np.ones((1, 5), dtype=np.uint8)
        
        
        min_cont_size_to_be_dilated = 10
        if len(contours_only_text_parent)>min_cont_size_to_be_dilated:
            (cx_conts, cy_conts,
             x_min_conts, x_max_conts,
             y_min_conts, y_max_conts,
             _) = find_new_features_of_contours(contours_only_text_parent)
            args_cont_located = np.array(range(len(contours_only_text_parent)))
            
            diff_y_conts = np.abs(y_max_conts[:]-y_min_conts)
            diff_x_conts = np.abs(x_max_conts[:]-x_min_conts)
            
            mean_x = statistics.mean(diff_x_conts)
            median_x = statistics.median(diff_x_conts)
            
            
            diff_x_ratio= diff_x_conts/mean_x
            
            args_cont_located_excluded = args_cont_located[diff_x_ratio>=1.3]
            args_cont_located_included = args_cont_located[diff_x_ratio<1.3]
            
            contours_only_text_parent_excluded = [contours_only_text_parent[ind]
                                                  #contours_only_text_parent[diff_x_ratio>=1.3]
                                                  for ind in range(len(contours_only_text_parent))
                                                  if diff_x_ratio[ind]>=1.3]
            contours_only_text_parent_included = [contours_only_text_parent[ind]
                                                  #contours_only_text_parent[diff_x_ratio<1.3]
                                                  for ind in range(len(contours_only_text_parent))
                                                  if diff_x_ratio[ind]<1.3]
            
            cx_conts_excluded = [cx_conts[ind]
                                 #cx_conts[diff_x_ratio>=1.3]
                                 for ind in range(len(cx_conts))
                                 if diff_x_ratio[ind]>=1.3]
            cx_conts_included = [cx_conts[ind]
                                 #cx_conts[diff_x_ratio<1.3]
                                 for ind in range(len(cx_conts))
                                 if diff_x_ratio[ind]<1.3]
            cy_conts_excluded = [cy_conts[ind]
                                 #cy_conts[diff_x_ratio>=1.3]
                                 for ind in range(len(cy_conts))
                                 if diff_x_ratio[ind]>=1.3]
            cy_conts_included = [cy_conts[ind]
                                 #cy_conts[diff_x_ratio<1.3]
                                 for ind in range(len(cy_conts))
                                 if diff_x_ratio[ind]<1.3]
            
            #print(diff_x_ratio, 'ratio')
            text_regions_p = text_regions_p.astype('uint8')
            
            if len(contours_only_text_parent_excluded)>0:
                textregion_par = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1])).astype('uint8')
                textregion_par = cv2.fillPoly(textregion_par, pts=contours_only_text_parent_included, color=(1,1))
            else:
                textregion_par = (text_regions_p[:,:]==1)*1
                textregion_par = textregion_par.astype('uint8')
                
            text_regions_p_textregions_dilated = cv2.erode(textregion_par , hor_kernel, iterations=2)
            text_regions_p_textregions_dilated = cv2.dilate(text_regions_p_textregions_dilated , ver_kernel, iterations=4)
            text_regions_p_textregions_dilated = cv2.erode(text_regions_p_textregions_dilated , hor_kernel, iterations=1)
            text_regions_p_textregions_dilated = cv2.dilate(text_regions_p_textregions_dilated , ver_kernel, iterations=5)
            text_regions_p_textregions_dilated[text_regions_p[:,:]>1] = 0
            
            
            contours_only_dilated, hir_on_text_dilated = return_contours_of_image(text_regions_p_textregions_dilated)
            contours_only_dilated = return_parent_contours(contours_only_dilated, hir_on_text_dilated)
            
            indexes_of_located_cont, center_x_coordinates_of_located, center_y_coordinates_of_located = \
                self.return_indexes_of_contours_located_inside_another_list_of_contours(
                    contours_only_dilated, contours_only_text_parent_included,
                    cx_conts_included, cy_conts_included, args_cont_located_included)
            
            
            if len(args_cont_located_excluded)>0:
                for ind in args_cont_located_excluded:
                    indexes_of_located_cont.append(np.array([ind]))
                    contours_only_dilated.append(contours_only_text_parent[ind])
                    center_y_coordinates_of_located.append(0)
            
            array_list = [np.array([elem]) if isinstance(elem, int) else elem for elem in indexes_of_located_cont]
            flattened_array = np.concatenate([arr.ravel() for arr in array_list])
            #print(len( np.unique(flattened_array)), 'indexes_of_located_cont uniques')
            
            missing_textregions = list( set(range(len(contours_only_text_parent))) - set(flattened_array) )
            #print(missing_textregions, 'missing_textregions')

            for ind in missing_textregions:
                indexes_of_located_cont.append(np.array([ind]))
                contours_only_dilated.append(contours_only_text_parent[ind])
                center_y_coordinates_of_located.append(0)
                
                
            if contours_only_text_parent_h:
                for vi in range(len(contours_only_text_parent_h)):
                    indexes_of_located_cont.append(int(vi+len(contours_only_text_parent)))
                    
            array_list = [np.array([elem]) if isinstance(elem, int) else elem for elem in indexes_of_located_cont]
            flattened_array = np.concatenate([arr.ravel() for arr in array_list])
        
        y_len = text_regions_p.shape[0]
        x_len = text_regions_p.shape[1]

        img_poly = np.zeros((y_len,x_len), dtype='uint8')
        img_poly[text_regions_p[:,:]==1] = 1
        img_poly[text_regions_p[:,:]==2] = 2
        img_poly[text_regions_p[:,:]==3] = 4
        img_poly[text_regions_p[:,:]==6] = 5
        
        img_header_and_sep = np.zeros((y_len,x_len), dtype='uint8')
        if contours_only_text_parent_h:
            _, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, _ = find_new_features_of_contours(
                contours_only_text_parent_h)
            for j in range(len(cy_main)):
                img_header_and_sep[int(y_max_main[j]):int(y_max_main[j])+12,
                                   int(x_min_main[j]):int(x_max_main[j])] = 1
            co_text_all_org = contours_only_text_parent + contours_only_text_parent_h
            if len(contours_only_text_parent)>min_cont_size_to_be_dilated:
                co_text_all = contours_only_dilated + contours_only_text_parent_h
            else:
                co_text_all = contours_only_text_parent + contours_only_text_parent_h
        else:
            co_text_all_org = contours_only_text_parent
            if len(contours_only_text_parent)>min_cont_size_to_be_dilated:
                co_text_all = contours_only_dilated
            else:
                co_text_all = contours_only_text_parent

        if not len(co_text_all):
            return [], []

        labels_con = np.zeros((int(y_len /6.), int(x_len/6.), len(co_text_all)), dtype=bool)
        co_text_all = [(i/6).astype(int) for i in co_text_all]
        for i in range(len(co_text_all)):
            img = labels_con[:,:,i].astype(np.uint8)
            
            #img = cv2.resize(img, (int(img.shape[1]/6), int(img.shape[0]/6)), interpolation=cv2.INTER_NEAREST)
            
            cv2.fillPoly(img, pts=[co_text_all[i]], color=(1,))
            labels_con[:,:,i] = img


        labels_con = resize_image(labels_con.astype(np.uint8), height1, width1).astype(bool)
        img_header_and_sep = resize_image(img_header_and_sep, height1, width1)
        img_poly = resize_image(img_poly, height3, width3)
        

        
        input_1 = np.zeros((inference_bs, height1, width1, 3))
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
                img1 = labels_con[:,:,i].astype(float)
                img2 = labels_con[:,:,j].astype(float)
                img1[img_poly==5] = 2
                img2[img_poly==5] = 2
                img1[img_header_and_sep==1] = 3
                img2[img_header_and_sep==1] = 3

                input_1[len(batch), :, :, 0] = img1 / 3.
                input_1[len(batch), :, :, 2] = img2 / 3.
                input_1[len(batch), :, :, 1] = img_poly / 5.

                tot_counter += 1
                batch.append(j)
                if tot_counter % inference_bs == 0 or tot_counter == len(ij_list):
                    y_pr = self.model_zoo.get("reading_order").predict(input_1 , verbose=0)
                    for jb, j in enumerate(batch):
                        if y_pr[jb][0]>=0.5:
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
        
        if len(contours_only_text_parent)>min_cont_size_to_be_dilated:
            org_contours_indexes = []
            for ind in range(len(ordered)):
                region_with_curr_order = ordered[ind]
                if region_with_curr_order < len(contours_only_dilated):
                    if np.isscalar(indexes_of_located_cont[region_with_curr_order]):
                        org_contours_indexes.extend([indexes_of_located_cont[region_with_curr_order]])
                    else:
                        arg_sort_located_cont = np.argsort(center_y_coordinates_of_located[region_with_curr_order])
                        org_contours_indexes.extend(
                            np.array(indexes_of_located_cont[region_with_curr_order])[arg_sort_located_cont])
                else:
                    org_contours_indexes.extend([indexes_of_located_cont[region_with_curr_order]])
            
            region_ids = ['region_%04d' % i for i in range(len(co_text_all_org))]
            return org_contours_indexes, region_ids
        else:
            region_ids = ['region_%04d' % i for i in range(len(co_text_all_org))]
            return ordered, region_ids

    def filter_contours_inside_a_bigger_one(self, contours, contours_d_ordered, image,
                                            marginal_cnts=None, type_contour="textregion"):
        if type_contour == "textregion":
            areas = np.array(list(map(cv2.contourArea, contours)))
            area_tot = image.shape[0]*image.shape[1]
            areas_ratio = areas / area_tot
            cx_main, cy_main = find_center_of_contours(contours)

            contours_index_small = np.flatnonzero(areas_ratio < 1e-3)
            contours_index_large = np.flatnonzero(areas_ratio >= 1e-3)

            #contours_> = [contours[ind] for ind in contours_index_large]
            indexes_to_be_removed = []
            for ind_small in contours_index_small:
                results = [cv2.pointPolygonTest(contours[ind_large], (cx_main[ind_small],
                                                                      cy_main[ind_small]),
                                                False)
                           for ind_large in contours_index_large]
                results = np.array(results)
                if np.any(results==1):
                    indexes_to_be_removed.append(ind_small)
                elif marginal_cnts:
                    results_marginal = [cv2.pointPolygonTest(marginal_cnt,
                                                             (cx_main[ind_small],
                                                              cy_main[ind_small]),
                                                             False)
                                        for marginal_cnt in marginal_cnts]
                    results_marginal = np.array(results_marginal)
                    if np.any(results_marginal==1):
                        indexes_to_be_removed.append(ind_small)

            contours = np.delete(contours, indexes_to_be_removed, axis=0)
            if len(contours_d_ordered):
                contours_d_ordered = np.delete(contours_d_ordered, indexes_to_be_removed, axis=0)

            return contours, contours_d_ordered

        else:
            contours_txtline_of_all_textregions = []
            indexes_of_textline_tot = []
            index_textline_inside_textregion = []
            for ind_region, textlines in enumerate(contours):
                contours_txtline_of_all_textregions.extend(textlines)
                index_textline_inside_textregion.extend(list(range(len(textlines))))
                indexes_of_textline_tot.extend([ind_region] * len(textlines))

            areas_tot = np.array(list(map(cv2.contourArea, contours_txtline_of_all_textregions)))
            area_tot_tot = image.shape[0]*image.shape[1]
            cx_main_tot, cy_main_tot = find_center_of_contours(contours_txtline_of_all_textregions)

            textline_in_textregion_index_to_del = {}
            for ij in range(len(contours_txtline_of_all_textregions)):
                area_of_con_interest = areas_tot[ij]
                args_without = np.delete(np.arange(len(contours_txtline_of_all_textregions)), ij)
                areas_without = areas_tot[args_without]
                args_with_bigger_area = args_without[areas_without > 1.5*area_of_con_interest]

                if len(args_with_bigger_area)>0:
                    results = [cv2.pointPolygonTest(contours_txtline_of_all_textregions[ind],
                                                    (cx_main_tot[ij],
                                                     cy_main_tot[ij]),
                                                    False)
                               for ind in args_with_bigger_area ]
                    results = np.array(results)
                    if np.any(results==1):
                        #print(indexes_of_textline_tot[ij], index_textline_inside_textregion[ij])
                        textline_in_textregion_index_to_del.setdefault(
                            indexes_of_textline_tot[ij], list()).append(
                                index_textline_inside_textregion[ij])
                        #contours[indexes_of_textline_tot[ij]].pop(index_textline_inside_textregion[ij])

            for textregion_index_to_del in textline_in_textregion_index_to_del:
                contours[textregion_index_to_del] = list(np.delete(
                    contours[textregion_index_to_del],
                    textline_in_textregion_index_to_del[textregion_index_to_del],
                    # needed so numpy does not flatten the entire result when 0 left
                    axis=0))

            return contours
        
    def return_indexes_of_contours_located_inside_another_list_of_contours(
            self, contours, contours_loc, cx_main_loc, cy_main_loc, indexes_loc):
        indexes_of_located_cont = []
        center_x_coordinates_of_located = []
        center_y_coordinates_of_located = []
        #M_main_tot = [cv2.moments(contours_loc[j])
                        #for j in range(len(contours_loc))]
        #cx_main_loc = [(M_main_tot[j]["m10"] / (M_main_tot[j]["m00"] + 1e-32)) for j in range(len(M_main_tot))]
        #cy_main_loc = [(M_main_tot[j]["m01"] / (M_main_tot[j]["m00"] + 1e-32)) for j in range(len(M_main_tot))]
        
        for ij in range(len(contours)):
            results = [cv2.pointPolygonTest(contours[ij], (cx_main_loc[ind], cy_main_loc[ind]), False)
                        for ind in range(len(cy_main_loc)) ]
            results = np.array(results)
            indexes_in = np.where((results == 0) | (results == 1))
            # [(results == 0) | (results == 1)]#np.where((results == 0) | (results == 1))
            indexes = indexes_loc[indexes_in]

            indexes_of_located_cont.append(indexes)
            center_x_coordinates_of_located.append(np.array(cx_main_loc)[indexes_in] )
            center_y_coordinates_of_located.append(np.array(cy_main_loc)[indexes_in] )
            
        return indexes_of_located_cont, center_x_coordinates_of_located, center_y_coordinates_of_located
        

    def filter_contours_without_textline_inside(
            self, contours_par, contours_textline,
            contours_only_text_parent_d_ordered,
            conf_contours_textregions):

        assert len(contours_par) == len(contours_textline)
        indices = np.arange(len(contours_textline))
        indices = np.delete(indices, np.flatnonzero([len(lines) == 0 for lines in contours_textline]))
        def filterfun(lis):
            if len(lis) == 0:
                return []
            return list(np.array(lis)[indices])

        return (filterfun(contours_par),
                filterfun(contours_textline),
                filterfun(contours_only_text_parent_d_ordered),
                filterfun(conf_contours_textregions),
                # indices
        )

    def separate_marginals_to_left_and_right_and_order_from_top_to_down(
            self, polygons_of_marginals, all_found_textline_polygons_marginals, all_box_coord_marginals,
            slopes_marginals, mid_point_of_page_width):
        cx_marg, cy_marg = find_center_of_contours(polygons_of_marginals)
        cx_marg = np.array(cx_marg)
        cy_marg = np.array(cy_marg)

        def split(lis):
            array = np.array(lis)
            return (list(array[cx_marg < mid_point_of_page_width]),
                    list(array[cx_marg >= mid_point_of_page_width]))

        (poly_marg_left,
         poly_marg_right) = \
             split(polygons_of_marginals)

        (all_found_textline_polygons_marginals_left,
         all_found_textline_polygons_marginals_right) = \
            split(all_found_textline_polygons_marginals)
        
        (all_box_coord_marginals_left,
         all_box_coord_marginals_right) = \
             split(all_box_coord_marginals)
        
        (slopes_marg_left,
         slopes_marg_right) = \
             split(slopes_marginals)
        
        (cy_marg_left,
         cy_marg_right) = \
             split(cy_marg)

        order_left = np.argsort(cy_marg_left)
        order_right = np.argsort(cy_marg_right)
        def sort_left(lis):
            return list(np.array(lis)[order_left])
        def sort_right(lis):
            return list(np.array(lis)[order_right])
        
        ordered_left_marginals = sort_left(poly_marg_left)
        ordered_right_marginals = sort_right(poly_marg_right)
        
        ordered_left_marginals_textline = sort_left(all_found_textline_polygons_marginals_left)
        ordered_right_marginals_textline = sort_right(all_found_textline_polygons_marginals_right)
        
        ordered_left_marginals_bbox = sort_left(all_box_coord_marginals_left)
        ordered_right_marginals_bbox = sort_right(all_box_coord_marginals_right)
        
        ordered_left_slopes_marginals = sort_left(slopes_marg_left)
        ordered_right_slopes_marginals = sort_right(slopes_marg_right)
        
        return (ordered_left_marginals,
                ordered_right_marginals,
                ordered_left_marginals_textline,
                ordered_right_marginals_textline,
                ordered_left_marginals_bbox,
                ordered_right_marginals_bbox,
                ordered_left_slopes_marginals,
                ordered_right_slopes_marginals)

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

        if dir_in:
            ls_imgs = [os.path.join(dir_in, image_filename)
                       for image_filename in filter(is_image_filename,
                                                    os.listdir(dir_in))]
        elif image_filename:
            ls_imgs = [image_filename]
        else:
            raise ValueError("run requires either a single image filename or a directory")

        for img_filename in ls_imgs:
            self.logger.info(img_filename)
            t0 = time.time()

            self.reset_file_name_dir(img_filename, dir_out)
            if self.enable_plotting:
                self.plotter = EynollahPlotter(dir_out=dir_out,
                                               dir_of_all=dir_of_all,
                                               dir_save_page=dir_save_page,
                                               dir_of_deskewed=dir_of_deskewed,
                                               dir_of_cropped_images=dir_of_cropped_images,
                                               dir_of_layout=dir_of_layout,
                                               image_filename_stem=Path(image_filename).stem)
            #print("text region early -11 in %.1fs", time.time() - t0)
            if os.path.exists(self.writer.output_filename):
                if overwrite:
                    self.logger.warning("will overwrite existing output file '%s'", self.writer.output_filename)
                else:
                    self.logger.warning("will skip input for existing output file '%s'", self.writer.output_filename)
                    continue

            pcgts = self.run_single()
            self.logger.info("Job done in %.1fs", time.time() - t0)
            self.writer.write_pagexml(pcgts)

        if dir_in:
            self.logger.info("All jobs done in %.1fs", time.time() - t0_tot)

    def run_single(self):
        t0 = time.time()
    
        self.logger.info(f"Processing file: {self.writer.image_filename}")
        self.logger.info("Step 1/5: Image Enhancement")
        
        img_res, is_image_enhanced, num_col_classifier, num_column_is_classified = \
            self.run_enhancement()
        
        self.logger.info(f"Image: {self.image.shape[1]}x{self.image.shape[0]}, "
                         f"{self.dpi} DPI, {num_col_classifier} columns")
        if is_image_enhanced:
            self.logger.info("Enhancement applied")
        
        self.logger.info(f"Enhancement complete ({time.time() - t0:.1f}s)")
        

        # Basic Processing Mode
        if self.skip_layout_and_reading_order:
            self.logger.info("Step 2/5: Basic Processing Mode")
            self.logger.info("Skipping layout analysis and reading order detection")
    
            _ ,_, _, _, textline_mask_tot_ea, img_bin_light, _ = \
                self.get_regions_light_v(img_res, is_image_enhanced, num_col_classifier,)

            page_coord, image_page, textline_mask_tot_ea, img_bin_light, cont_page = \
                self.run_graphics_and_columns_without_layout(textline_mask_tot_ea, img_bin_light)

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
                all_found_textline_polygons, None, textline_mask_tot_ea, type_contour="textline")
            
            order_text_new = [0]
            slopes =[0]
            conf_contours_textregions =[0]
            
            pcgts = self.writer.build_pagexml_no_full_layout(
                found_polygons_text_region=cont_page,                   
                page_coord=page_coord,                  
                order_of_texts=order_text_new,              
                all_found_textline_polygons=all_found_textline_polygons,   
                all_box_coord=page_coord,                  
                found_polygons_text_region_img=[],                          
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
                found_polygons_tables=[],                          
            )
            self.logger.info("Basic processing complete")
            return pcgts

        #print("text region early -1 in %.1fs", time.time() - t0)
        t1 = time.time()
        self.logger.info("Step 2/5: Layout Analysis")
        
        self.logger.info("Using light version processing")
        text_regions_p_1 ,erosion_hurts, polygons_seplines, polygons_text_early, \
            textline_mask_tot_ea, img_bin_light, confidence_matrix = \
            self.get_regions_light_v(img_res, is_image_enhanced, num_col_classifier)
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
        #print("text region early -2,5 in %.1fs", time.time() - t0)
        #self.logger.info("Textregion detection took %.1fs ", time.time() - t1t)
        num_col, num_col_classifier, img_only_regions, page_coord, image_page, mask_images, mask_lines, \
            text_regions_p_1, cont_page, table_prediction, textline_mask_tot_ea, img_bin_light = \
                self.run_graphics_and_columns_light(text_regions_p_1, textline_mask_tot_ea,
                                                    num_col_classifier, num_column_is_classified,
                                                    erosion_hurts, img_bin_light)
        #self.logger.info("run graphics %.1fs ", time.time() - t1t)
        #print("text region early -3 in %.1fs", time.time() - t0)
        textline_mask_tot_ea_org = np.copy(textline_mask_tot_ea)

        #plt.imshow(table_prediction)
        #plt.show()
        self.logger.info(f"Layout analysis complete ({time.time() - t1:.1f}s)")

        if not num_col and len(polygons_text_early) == 0:
            self.logger.info("No columns detected - generating empty PAGE-XML")
    
            pcgts = self.writer.build_pagexml_no_full_layout(
                found_polygons_text_region=[],
                page_coord=page_coord,
                order_of_texts=[],
                all_found_textline_polygons=[],
                all_box_coord=[],
                found_polygons_text_region_img=[],
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
                found_polygons_tables=[],
            )
            return pcgts

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

            image_page = resize_image(image_page,img_h_new, img_w_new )
            textline_mask_tot_ea = resize_image(textline_mask_tot_ea,img_h_new, img_w_new )
            mask_images = resize_image(mask_images,img_h_new, img_w_new )
            mask_lines = resize_image(mask_lines,img_h_new, img_w_new )
            text_regions_p_1 = resize_image(text_regions_p_1,img_h_new, img_w_new )
            table_prediction = resize_image(table_prediction,img_h_new, img_w_new )

        textline_mask_tot, text_regions_p = \
            self.run_marginals(textline_mask_tot_ea, mask_images, mask_lines,
                               num_col_classifier, slope_deskew, text_regions_p_1, table_prediction)
        if self.plotter:
            self.plotter.save_plot_of_layout_main_all(text_regions_p, image_page)
            self.plotter.save_plot_of_layout_main(text_regions_p, image_page)

        if image_page.size:
            # if ratio of text regions to page area is smaller that 30%,
            # then deskew angle will not be allowed to exceed 45
            if (abs(slope_deskew) > 45 and
                ((text_regions_p == 1).sum() +
                 (text_regions_p == 4).sum()) / float(image_page.size) <= 0.3):
                slope_deskew = 0

        # if there is no main text, then relabel marginalia as main
        if (text_regions_p == 1).sum() == 0:
            text_regions_p[text_regions_p == 4] = 1

        self.logger.info("Step 3/5: Text Line Detection")
        
        if self.curved_line:
            self.logger.info("Mode: Curved line detection")

        if num_col_classifier in (1,2):
            image_page = resize_image(image_page,org_h_l_m, org_w_l_m )
            textline_mask_tot_ea = resize_image(textline_mask_tot_ea,org_h_l_m, org_w_l_m )
            text_regions_p = resize_image(text_regions_p,org_h_l_m, org_w_l_m )
            textline_mask_tot = resize_image(textline_mask_tot,org_h_l_m, org_w_l_m )
            text_regions_p_1 = resize_image(text_regions_p_1,org_h_l_m, org_w_l_m )
            table_prediction = resize_image(table_prediction,org_h_l_m, org_w_l_m )

        self.logger.info(f"Detection of marginals took {time.time() - t1:.1f}s")
        ## birdan sora chock chakir
        t1 = time.time()
        if not self.full_layout:
            polygons_of_images, img_revised_tab, text_regions_p_1_n, \
                textline_mask_tot_d, regions_without_separators_d, \
                boxes, boxes_d, polygons_of_marginals, contours_tables = \
                self.run_boxes_no_full_layout(image_page, textline_mask_tot, text_regions_p, slope_deskew,
                                              num_col_classifier, table_prediction, erosion_hurts)
            ###polygons_of_marginals = dilate_textregion_contours(polygons_of_marginals)
        else:
            polygons_of_images, img_revised_tab, text_regions_p_1_n, \
                textline_mask_tot_d, regions_without_separators_d, \
                regions_fully, regions_without_separators, polygons_of_marginals, contours_tables = \
                self.run_boxes_full_layout(image_page, textline_mask_tot, text_regions_p, slope_deskew,
                                           num_col_classifier, img_only_regions, table_prediction, erosion_hurts,
                                           img_bin_light)
            ###polygons_of_marginals = dilate_textregion_contours(polygons_of_marginals)
            drop_label_in_full_layout = 4
            textline_mask_tot_ea_org[img_revised_tab==drop_label_in_full_layout] = 0


        text_only = (img_revised_tab[:, :] == 1) * 1
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            text_only_d = (text_regions_p_1_n[:, :] == 1) * 1

        #print("text region early 2 in %.1fs", time.time() - t0)
        ###min_con_area = 0.000005
        contours_only_text, hir_on_text = return_contours_of_image(text_only)
        contours_only_text_parent = return_parent_contours(contours_only_text, hir_on_text)
        contours_only_text_parent_d_ordered = []
        contours_only_text_parent_d = []

        if len(contours_only_text_parent) > 0:
            areas_tot_text = np.prod(text_only.shape)
            areas_cnt_text = np.array([cv2.contourArea(c) for c in contours_only_text_parent])
            areas_cnt_text = areas_cnt_text / float(areas_tot_text)
            #self.logger.info('areas_cnt_text %s', areas_cnt_text)
            contours_only_text_parent = np.array(contours_only_text_parent)[areas_cnt_text > MIN_AREA_REGION]
            areas_cnt_text_parent = areas_cnt_text[areas_cnt_text > MIN_AREA_REGION]

            index_con_parents = np.argsort(areas_cnt_text_parent)
            contours_only_text_parent = contours_only_text_parent[index_con_parents]
            areas_cnt_text_parent = areas_cnt_text_parent[index_con_parents]

            centers = np.stack(find_center_of_contours(contours_only_text_parent)) # [2, N]

            center0 = centers[:, -1:] # [2, 1]

            if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                contours_only_text_d, hir_on_text_d = return_contours_of_image(text_only_d)
                contours_only_text_parent_d = return_parent_contours(contours_only_text_d, hir_on_text_d)

                areas_tot_text_d = np.prod(text_only_d.shape)
                areas_cnt_text_d = np.array([cv2.contourArea(c) for c in contours_only_text_parent_d])
                areas_cnt_text_d = areas_cnt_text_d / float(areas_tot_text_d)

                contours_only_text_parent_d = np.array(contours_only_text_parent_d)[areas_cnt_text_d > MIN_AREA_REGION]
                areas_cnt_text_d = areas_cnt_text_d[areas_cnt_text_d > MIN_AREA_REGION]

                if len(contours_only_text_parent_d):
                    index_con_parents_d = np.argsort(areas_cnt_text_d)
                    contours_only_text_parent_d = np.array(contours_only_text_parent_d)[index_con_parents_d]
                    areas_cnt_text_d = areas_cnt_text_d[index_con_parents_d]

                    centers_d = np.stack(find_center_of_contours(contours_only_text_parent_d)) # [2, N]

                    center0_d = centers_d[:, -1:].copy() # [2, 1]

                    # find the largest among the largest 5 deskewed contours
                    # that is also closest to the largest original contour
                    last5_centers_d = centers_d[:, -5:]
                    dists_d = np.linalg.norm(center0 - last5_centers_d, axis=0)
                    ind_largest = len(contours_only_text_parent_d) - last5_centers_d.shape[1] + np.argmin(dists_d)
                    center0_d[:, 0] = centers_d[:, ind_largest]

                    # order new contours the same way as the undeskewed contours
                    # (by calculating the offset of the largest contours, respectively,
                    #  of the new and undeskewed image; then for each contour,
                    #  finding the closest new contour, with proximity calculated
                    #  as distance of their centers modulo offset vector)
                    (h, w) = text_only.shape[:2]
                    center = (w // 2.0, h // 2.0)
                    M = cv2.getRotationMatrix2D(center, slope_deskew, 1.0)
                    M_22 = np.array(M)[:2, :2]
                    center0 = np.dot(M_22, center0) # [2, 1]
                    offset = center0 - center0_d # [2, 1]

                    centers = np.dot(M_22, centers) - offset # [2,N]
                    # add dimension for area (so only contours of similar size will be considered close)
                    centers = np.append(centers, areas_cnt_text_parent[np.newaxis], axis=0)
                    centers_d = np.append(centers_d, areas_cnt_text_d[np.newaxis], axis=0)

                    dists = np.zeros((len(contours_only_text_parent), len(contours_only_text_parent_d)))
                    for i in range(len(contours_only_text_parent)):
                        dists[i] = np.linalg.norm(centers[:, i:i + 1] - centers_d, axis=0)
                    corresp = np.zeros(dists.shape, dtype=bool)
                    # keep searching next-closest until at least one correspondence on each side
                    while not np.all(corresp.sum(axis=1)) and not np.all(corresp.sum(axis=0)):
                        idx = np.nanargmin(dists)
                        i, j = np.unravel_index(idx, dists.shape)
                        dists[i, j] = np.nan
                        corresp[i, j] = True
                    #print("original/deskewed adjacency", corresp.nonzero())
                    contours_only_text_parent_d_ordered = np.zeros_like(contours_only_text_parent)
                    contours_only_text_parent_d_ordered = contours_only_text_parent_d[np.argmax(corresp, axis=1)]
                    # img1 = np.zeros(text_only_d.shape[:2], dtype=np.uint8)
                    # for i in range(len(contours_only_text_parent)):
                    #     cv2.fillPoly(img1, pts=[contours_only_text_parent_d_ordered[i]], color=i + 1)
                    # plt.subplot(2, 2, 1, title="direct corresp contours")
                    # plt.imshow(img1)
                    # img2 = np.zeros(text_only_d.shape[:2], dtype=np.uint8)
                    # join deskewed regions mapping to single original ones
                    for i in range(len(contours_only_text_parent)):
                        if np.count_nonzero(corresp[i]) > 1:
                            indices = np.flatnonzero(corresp[i])
                            #print("joining", indices)
                            polygons_d = [contour2polygon(contour)
                                          for contour in contours_only_text_parent_d[indices]]
                            contour_d = polygon2contour(join_polygons(polygons_d))
                            contours_only_text_parent_d_ordered[i] = contour_d
                    #         cv2.fillPoly(img2, pts=[contour_d], color=i + 1)
                    # plt.subplot(2, 2, 3, title="joined contours")
                    # plt.imshow(img2)
                    # img3 = np.zeros(text_only_d.shape[:2], dtype=np.uint8)
                    # split deskewed regions mapping to multiple original ones
                    def deskew(polygon):
                        polygon = shapely.affinity.rotate(polygon, -slope_deskew, origin=center)
                        polygon = shapely.affinity.translate(polygon, *offset.squeeze())
                        return polygon
                    for j in range(len(contours_only_text_parent_d)):
                        if np.count_nonzero(corresp[:, j]) > 1:
                            indices = np.flatnonzero(corresp[:, j])
                            #print("splitting along", indices)
                            polygons = [deskew(contour2polygon(contour))
                                        for contour in contours_only_text_parent[indices]]
                            polygon_d = contour2polygon(contours_only_text_parent_d[j])
                            polygons_d = [make_intersection(polygon_d, polygon)
                                          for polygon in polygons]
                            # ignore where there is no actual overlap
                            indices = indices[np.flatnonzero(polygons_d)]
                            contours_d = [polygon2contour(polygon_d)
                                          for polygon_d in polygons_d
                                          if polygon_d]
                            contours_only_text_parent_d_ordered[indices] = contours_d
                    #         cv2.fillPoly(img3, pts=contours_d, color=j + 1)
                    # plt.subplot(2, 2, 4, title="split contours")
                    # plt.imshow(img3)
                    # img4 = np.zeros(text_only_d.shape[:2], dtype=np.uint8)
                    # for i in range(len(contours_only_text_parent)):
                    #     cv2.fillPoly(img4, pts=[contours_only_text_parent_d_ordered[i]], color=i + 1)
                    # plt.subplot(2, 2, 2, title="result contours")
                    # plt.imshow(img4)
                    # plt.show()

        if not len(contours_only_text_parent):
            # stop early
            empty_marginals = [[]] * len(polygons_of_marginals)
            if self.full_layout:
                pcgts = self.writer.build_pagexml_full_layout(
                    found_polygons_text_region=[],
                    found_polygons_text_region_h=[],
                    page_coord=page_coord,
                    order_of_texts=[],
                    all_found_textline_polygons=[],
                    all_found_textline_polygons_h=[],
                    all_box_coord=[],
                    all_box_coord_h=[],
                    found_polygons_text_region_img=polygons_of_images,
                    found_polygons_tables=contours_tables,
                    found_polygons_drop_capitals=[],
                    found_polygons_marginals_left=polygons_of_marginals,
                    found_polygons_marginals_right=polygons_of_marginals,
                    all_found_textline_polygons_marginals_left=empty_marginals,
                    all_found_textline_polygons_marginals_right=empty_marginals,
                    all_box_coord_marginals_left=empty_marginals,
                    all_box_coord_marginals_right=empty_marginals,
                    slopes=[],
                    slopes_h=[],
                    slopes_marginals_left=[],
                    slopes_marginals_right=[],
                    cont_page=cont_page,
                    polygons_seplines=polygons_seplines
                )
            else:
                pcgts = self.writer.build_pagexml_no_full_layout(
                    found_polygons_text_region=[],
                    page_coord=page_coord,
                    order_of_texts=[],
                    all_found_textline_polygons=[],
                    all_box_coord=[],
                    found_polygons_text_region_img=polygons_of_images,
                    found_polygons_marginals_left=polygons_of_marginals,
                    found_polygons_marginals_right=polygons_of_marginals,
                    all_found_textline_polygons_marginals_left=empty_marginals,
                    all_found_textline_polygons_marginals_right=empty_marginals,
                    all_box_coord_marginals_left=empty_marginals,
                    all_box_coord_marginals_right=empty_marginals,
                    slopes=[],
                    slopes_marginals_left=[],
                    slopes_marginals_right=[],
                    cont_page=cont_page,
                    polygons_seplines=polygons_seplines,
                    found_polygons_tables=contours_tables
                )
            return pcgts


        #print("text region early 3 in %.1fs", time.time() - t0)
        contours_only_text_parent = dilate_textregion_contours(contours_only_text_parent)
        contours_only_text_parent , contours_only_text_parent_d_ordered = self.filter_contours_inside_a_bigger_one(
            contours_only_text_parent, contours_only_text_parent_d_ordered, text_only,
            marginal_cnts=polygons_of_marginals)
        #print("text region early 3.5 in %.1fs", time.time() - t0)
        conf_contours_textregions = get_textregion_contours_in_org_image_light(
            contours_only_text_parent, self.image, confidence_matrix)
        #contours_only_text_parent = dilate_textregion_contours(contours_only_text_parent)
        #print("text region early 4 in %.1fs", time.time() - t0)
        boxes_text = get_text_region_boxes_by_given_contours(contours_only_text_parent)
        boxes_marginals = get_text_region_boxes_by_given_contours(polygons_of_marginals)
        #print("text region early 5 in %.1fs", time.time() - t0)
        ## birdan sora chock chakir
        if not self.curved_line:
            all_found_textline_polygons, \
                all_box_coord, slopes = self.get_slopes_and_deskew_new_light2(
                    contours_only_text_parent, textline_mask_tot_ea_org,
                    boxes_text, slope_deskew)
            all_found_textline_polygons_marginals, \
                all_box_coord_marginals, slopes_marginals = self.get_slopes_and_deskew_new_light2(
                    polygons_of_marginals, textline_mask_tot_ea_org,
                    boxes_marginals, slope_deskew)

            all_found_textline_polygons = dilate_textline_contours(
                all_found_textline_polygons)
            all_found_textline_polygons = self.filter_contours_inside_a_bigger_one(
                all_found_textline_polygons, None, textline_mask_tot_ea_org, type_contour="textline")
            all_found_textline_polygons_marginals = dilate_textline_contours(
                all_found_textline_polygons_marginals)
            contours_only_text_parent, all_found_textline_polygons, \
                contours_only_text_parent_d_ordered, conf_contours_textregions = \
                self.filter_contours_without_textline_inside(
                    contours_only_text_parent, all_found_textline_polygons,
                    contours_only_text_parent_d_ordered, conf_contours_textregions)
        else:
            scale_param = 1
            textline_mask_tot_ea_erode = cv2.erode(textline_mask_tot_ea, kernel=KERNEL, iterations=2)
            all_found_textline_polygons, \
                all_box_coord, slopes = self.get_slopes_and_deskew_new_curved(
                    contours_only_text_parent, textline_mask_tot_ea_erode,
                    boxes_text, text_only,
                    num_col_classifier, scale_param, slope_deskew)
            all_found_textline_polygons = small_textlines_to_parent_adherence2(
                all_found_textline_polygons, textline_mask_tot_ea, num_col_classifier)
            all_found_textline_polygons_marginals, \
                all_box_coord_marginals, slopes_marginals = self.get_slopes_and_deskew_new_curved(
                    polygons_of_marginals, textline_mask_tot_ea_erode,
                    boxes_marginals, text_only,
                    num_col_classifier, scale_param, slope_deskew)
            all_found_textline_polygons_marginals = small_textlines_to_parent_adherence2(
                all_found_textline_polygons_marginals, textline_mask_tot_ea, num_col_classifier)
        
        mid_point_of_page_width = text_regions_p.shape[1] / 2.
        (polygons_of_marginals_left, polygons_of_marginals_right,
         all_found_textline_polygons_marginals_left, all_found_textline_polygons_marginals_right,
         all_box_coord_marginals_left, all_box_coord_marginals_right,
         slopes_marginals_left, slopes_marginals_right) = \
             self.separate_marginals_to_left_and_right_and_order_from_top_to_down(
                 polygons_of_marginals, all_found_textline_polygons_marginals, all_box_coord_marginals,
                 slopes_marginals, mid_point_of_page_width)
        
        #print(len(polygons_of_marginals), len(ordered_left_marginals), len(ordered_right_marginals), 'marginals ordred')

        if self.full_layout:
            fun = check_any_text_region_in_model_one_is_main_or_header_light
            text_regions_p, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, \
                all_found_textline_polygons, all_found_textline_polygons_h, slopes, slopes_h, \
                contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered, \
                    conf_contours_textregions, conf_contours_textregions_h = fun(
                        text_regions_p, regions_fully, contours_only_text_parent,
                        all_box_coord, all_found_textline_polygons,
                        slopes, contours_only_text_parent_d_ordered, conf_contours_textregions)

            if self.plotter:
                self.plotter.save_plot_of_layout(text_regions_p, image_page)
                self.plotter.save_plot_of_layout_all(text_regions_p, image_page)

            label_img = 4
            polygons_of_drop_capitals = return_contours_of_interested_region(text_regions_p, label_img,
                                                                             min_area=0.00003)
            ##all_found_textline_polygons = adhere_drop_capital_region_into_corresponding_textline(
                ##text_regions_p, polygons_of_drop_capitals, contours_only_text_parent, contours_only_text_parent_h,
                ##all_box_coord, all_box_coord_h, all_found_textline_polygons, all_found_textline_polygons_h,
                ##kernel=KERNEL, curved_line=self.curved_line)

            if not self.reading_order_machine_based:
                label_seps = 6
                if not self.headers_off:
                    if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                        num_col, _, matrix_of_lines_ch, splitter_y_new, _ = find_number_of_columns_in_document(
                            text_regions_p, num_col_classifier, self.tables,  label_seps, contours_only_text_parent_h)
                    else:
                        _, _, matrix_of_lines_ch_d, splitter_y_new_d, _ = find_number_of_columns_in_document(
                            text_regions_p_1_n, num_col_classifier, self.tables, label_seps, contours_only_text_parent_h_d_ordered)
                elif self.headers_off:
                    if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                        num_col, _, matrix_of_lines_ch, splitter_y_new, _ = find_number_of_columns_in_document(
                            text_regions_p, num_col_classifier, self.tables,  label_seps)
                    else:
                        _, _, matrix_of_lines_ch_d, splitter_y_new_d, _ = find_number_of_columns_in_document(
                            text_regions_p_1_n, num_col_classifier, self.tables, label_seps)

                if num_col_classifier >= 3:
                    if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                        regions_without_separators = regions_without_separators.astype(np.uint8)
                        regions_without_separators = cv2.erode(regions_without_separators[:, :], KERNEL, iterations=6)
                    else:
                        regions_without_separators_d = regions_without_separators_d.astype(np.uint8)
                        regions_without_separators_d = cv2.erode(regions_without_separators_d[:, :], KERNEL, iterations=6)

                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                    boxes, peaks_neg_tot_tables = return_boxes_of_images_by_order_of_reading_new(
                        splitter_y_new, regions_without_separators, matrix_of_lines_ch,
                        num_col_classifier, erosion_hurts, self.tables, self.right2left,
                        logger=self.logger)
                else:
                    boxes_d, peaks_neg_tot_tables_d = return_boxes_of_images_by_order_of_reading_new(
                        splitter_y_new_d, regions_without_separators_d, matrix_of_lines_ch_d,
                        num_col_classifier, erosion_hurts, self.tables, self.right2left,
                        logger=self.logger)
        else:
            contours_only_text_parent_h = []
            contours_only_text_parent_h_d_ordered = []

        if self.plotter:
            self.plotter.write_images_into_directory(polygons_of_images, image_page)
        t_order = time.time()

        self.logger.info("Step 4/5: Reading Order Detection")

        if self.reading_order_machine_based:
            self.logger.info("Using machine-based detection")
        if self.right2left:
            self.logger.info("Right-to-left mode enabled")
        if self.headers_off:
            self.logger.info("Headers ignored in reading order")

        if self.reading_order_machine_based:
            order_text_new, id_of_texts_tot = self.do_order_of_regions_with_model(
                contours_only_text_parent, contours_only_text_parent_h, text_regions_p)
        else:
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                order_text_new, id_of_texts_tot = self.do_order_of_regions(
                    contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot)
            else:
                order_text_new, id_of_texts_tot = self.do_order_of_regions(
                    contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered,
                    boxes_d, textline_mask_tot_d)
        self.logger.info(f"Detection of reading order took {time.time() - t_order:.1f}s")

        self.logger.info("Step 5/5: Output Generation")

        if self.full_layout:
            pcgts = self.writer.build_pagexml_full_layout(
                found_polygons_text_region=contours_only_text_parent,
                found_polygons_text_region_h=contours_only_text_parent_h,
                page_coord=page_coord,
                order_of_texts=order_text_new,
                all_found_textline_polygons=all_found_textline_polygons,
                all_found_textline_polygons_h=all_found_textline_polygons_h,
                all_box_coord=all_box_coord,
                all_box_coord_h=all_box_coord_h,
                found_polygons_text_region_img=polygons_of_images,
                found_polygons_tables=contours_tables,
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
                conf_contours_textregions=conf_contours_textregions,
                conf_contours_textregions_h=conf_contours_textregions_h
            )
        else:
            pcgts = self.writer.build_pagexml_no_full_layout(
                found_polygons_text_region=contours_only_text_parent,
                page_coord=page_coord,
                order_of_texts=order_text_new,
                all_found_textline_polygons=all_found_textline_polygons,
                all_box_coord=all_box_coord,
                found_polygons_text_region_img=polygons_of_images,
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
                found_polygons_tables=contours_tables,
            )
            
        return pcgts
