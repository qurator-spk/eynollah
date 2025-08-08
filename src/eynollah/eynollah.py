# pylint: disable=no-member,invalid-name,line-too-long,missing-function-docstring,missing-class-docstring,too-many-branches
# pylint: disable=too-many-locals,wrong-import-position,too-many-lines,too-many-statements,chained-comparison,fixme,broad-except,c-extension-no-member
# pylint: disable=too-many-public-methods,too-many-arguments,too-many-instance-attributes,too-many-public-methods,
# pylint: disable=consider-using-enumerate
"""
document layout analysis (segmentation) with output in PAGE-XML
"""

from logging import Logger
from difflib import SequenceMatcher as sq
from PIL import Image, ImageDraw, ImageFont
import math
import os
import sys
import time
from typing import Optional
import atexit
import warnings
from functools import partial
from pathlib import Path
from multiprocessing import cpu_count
import gc
import copy
import json
from loky import ProcessPoolExecutor
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from numba import cuda
from skimage.morphology import skeletonize
from ocrd import OcrdPage
from ocrd_utils import getLogger, tf_disable_interactive_logs
import statistics

try:
    import torch
except ImportError:
    torch = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    TrOCRProcessor = VisionEncoderDecoderModel = None

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf_disable_interactive_logs()
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import load_model
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")
# use tf1 compatibility for keras backend
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras import layers
from tensorflow.keras.layers import StringLookup

from .utils.contour import (
    filter_contours_area_of_image,
    filter_contours_area_of_image_tables,
    find_contours_mean_y_diff,
    find_new_features_of_contours,
    find_features_of_contours,
    get_text_region_boxes_by_given_contours,
    get_textregion_contours_in_org_image,
    get_textregion_contours_in_org_image_light,
    return_contours_of_image,
    return_contours_of_interested_region,
    return_contours_of_interested_region_by_min_size,
    return_contours_of_interested_textline,
    return_parent_contours,
)
from .utils.rotate import (
    rotate_image,
    rotation_not_90_func,
    rotation_not_90_func_full_layout,
    rotation_image_new
)
from .utils.utils_ocr import (
    return_textline_contour_with_added_box_coordinate,
    preprocess_and_resize_image_for_ocrcnn_model,
    return_textlines_split_if_needed,
    decode_batch_predictions,
    return_rnn_cnn_ocr_of_given_textlines,
    fit_text_single_line,
    break_curved_line_into_small_pieces_and_then_merge,
    get_orientation_moments,
    rotate_image_with_padding,
    get_contours_and_bounding_boxes
)
from .utils.separate_lines import (
    textline_contours_postprocessing,
    separate_lines_new2,
    return_deskew_slop,
    return_deskew_slop_old_mp,
    do_work_of_slopes_new,
    do_work_of_slopes_new_curved,
    do_work_of_slopes_new_light,
)
from .utils.drop_capitals import (
    adhere_drop_capital_region_into_corresponding_textline,
    filter_small_drop_capitals_from_no_patch_layout
)
from .utils.marginals import get_marginals
from .utils.resize import resize_image
from .utils import (
    boosting_headers_by_longshot_region_segmentation,
    crop_image_inside_box,
    find_num_col,
    otsu_copy_binary,
    put_drop_out_from_only_drop_model,
    putt_bb_of_drop_capitals_of_model_in_patches_in_layout,
    check_any_text_region_in_model_one_is_main_or_header,
    check_any_text_region_in_model_one_is_main_or_header_light,
    small_textlines_to_parent_adherence2,
    order_of_regions,
    find_number_of_columns_in_document,
    return_boxes_of_images_by_order_of_reading_new
)
from .utils.pil_cv2 import check_dpi, pil2cv
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


class Patches(layers.Layer):
    def __init__(self, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config

class Eynollah:
    def __init__(
        self,
        dir_models : str,
        dir_out : Optional[str] = None,
        dir_of_cropped_images : Optional[str] = None,
        extract_only_images : bool =False,
        dir_of_layout : Optional[str] = None,
        dir_of_deskewed : Optional[str] = None,
        dir_of_all : Optional[str] = None,
        dir_save_page : Optional[str] = None,
        enable_plotting : bool = False,
        allow_enhancement : bool = False,
        curved_line : bool = False,
        textline_light : bool = False,
        full_layout : bool = False,
        tables : bool = False,
        right2left : bool = False,
        input_binary : bool = False,
        allow_scaling : bool = False,
        headers_off : bool = False,
        light_version : bool = False,
        ignore_page_extraction : bool = False,
        reading_order_machine_based : bool = False,
        do_ocr : bool = False,
        transformer_ocr: bool = False,
        batch_size_ocr: Optional[int] = None,
        num_col_upper : Optional[int] = None,
        num_col_lower : Optional[int] = None,
        threshold_art_class_layout: Optional[float] = None,
        threshold_art_class_textline: Optional[float] = None,
        skip_layout_and_reading_order : bool = False,
        logger : Optional[Logger] = None,
    ):
        if skip_layout_and_reading_order:
            textline_light = True
        self.light_version = light_version
        self.dir_out = dir_out
        self.dir_of_all = dir_of_all
        self.dir_save_page = dir_save_page
        self.reading_order_machine_based = reading_order_machine_based
        self.dir_of_deskewed = dir_of_deskewed
        self.dir_of_deskewed =  dir_of_deskewed
        self.dir_of_cropped_images=dir_of_cropped_images
        self.dir_of_layout=dir_of_layout
        self.enable_plotting = enable_plotting
        self.allow_enhancement = allow_enhancement
        self.curved_line = curved_line
        self.textline_light = textline_light
        self.full_layout = full_layout
        self.tables = tables
        self.right2left = right2left
        self.input_binary = input_binary
        self.allow_scaling = allow_scaling
        self.headers_off = headers_off
        self.light_version = light_version
        self.extract_only_images = extract_only_images
        self.ignore_page_extraction = ignore_page_extraction
        self.skip_layout_and_reading_order = skip_layout_and_reading_order
        self.ocr = do_ocr
        self.tr = transformer_ocr
        if num_col_upper:
            self.num_col_upper = int(num_col_upper)
        else:
            self.num_col_upper = num_col_upper
        if num_col_lower:
            self.num_col_lower = int(num_col_lower)
        else:
            self.num_col_lower = num_col_lower
            
        if threshold_art_class_layout:
            self.threshold_art_class_layout = float(threshold_art_class_layout)
        else:
            self.threshold_art_class_layout = 0.1
            
        if threshold_art_class_textline:
            self.threshold_art_class_textline = float(threshold_art_class_textline)
        else:
            self.threshold_art_class_textline = 0.1
            
        self.logger = logger if logger else getLogger('eynollah')
        # for parallelization of CPU-intensive tasks:
        self.executor = ProcessPoolExecutor(max_workers=cpu_count(), timeout=1200)
        atexit.register(self.executor.shutdown)
        self.dir_models = dir_models
        self.model_dir_of_enhancement = dir_models + "/eynollah-enhancement_20210425"
        self.model_dir_of_binarization = dir_models + "/eynollah-binarization_20210425"
        self.model_dir_of_col_classifier = dir_models + "/eynollah-column-classifier_20210425"
        self.model_region_dir_p = dir_models + "/eynollah-main-regions-aug-scaling_20210425"
        self.model_region_dir_p2 = dir_models + "/eynollah-main-regions-aug-rotation_20210425"
        #"/modelens_full_lay_1_3_031124"
        #"/modelens_full_lay_13__3_19_241024"
        #"/model_full_lay_13_241024"
        #"/modelens_full_lay_13_17_231024"
        #"/modelens_full_lay_1_2_221024"
        #"/eynollah-full-regions-1column_20210425"
        self.model_region_dir_fully_np = dir_models + "/modelens_full_lay_1__4_3_091124"
        #self.model_region_dir_fully = dir_models + "/eynollah-full-regions-3+column_20210425"
        self.model_page_dir = dir_models + "/eynollah-page-extraction_20210425"
        self.model_region_dir_p_ens = dir_models + "/eynollah-main-regions-ensembled_20210425"
        self.model_region_dir_p_ens_light = dir_models + "/eynollah-main-regions_20220314"
        self.model_region_dir_p_ens_light_only_images_extraction = dir_models + "/eynollah-main-regions_20231127_672_org_ens_11_13_16_17_18"
        self.model_reading_order_dir = dir_models + "/model_step_4800000_mb_ro"#"/model_ens_reading_order_machine_based"
        #"/modelens_12sp_elay_0_3_4__3_6_n"
        #"/modelens_earlylayout_12spaltige_2_3_5_6_7_8"
        #"/modelens_early12_sp_2_3_5_6_7_8_9_10_12_14_15_16_18"
        #"/modelens_1_2_4_5_early_lay_1_2_spaltige"
        #"/model_3_eraly_layout_no_patches_1_2_spaltige"
        self.model_region_dir_p_1_2_sp_np = dir_models + "/modelens_e_l_all_sp_0_1_2_3_4_171024"
        ##self.model_region_dir_fully_new = dir_models + "/model_2_full_layout_new_trans"
        #"/modelens_full_lay_1_3_031124"
        #"/modelens_full_lay_13__3_19_241024"
        #"/model_full_lay_13_241024"
        #"/modelens_full_lay_13_17_231024"
        #"/modelens_full_lay_1_2_221024"
        #"/modelens_full_layout_24_till_28"
        #"/model_2_full_layout_new_trans"
        self.model_region_dir_fully = dir_models + "/modelens_full_lay_1__4_3_091124"
        if self.textline_light:
            #"/modelens_textline_1_4_16092024"
            #"/model_textline_ens_3_4_5_6_artificial"
            #"/modelens_textline_1_3_4_20240915"
            #"/model_textline_ens_3_4_5_6_artificial"
            #"/modelens_textline_9_12_13_14_15"
            #"/eynollah-textline_light_20210425"
            self.model_textline_dir = dir_models + "/modelens_textline_0_1__2_4_16092024"
        else:
            #"/eynollah-textline_20210425"
            self.model_textline_dir = dir_models + "/modelens_textline_0_1__2_4_16092024"
        if self.ocr and self.tr:
            self.model_ocr_dir = dir_models + "/trocr_model_ens_of_3_checkpoints_201124"
        elif self.ocr and not self.tr:
            self.model_ocr_dir = dir_models + "/model_eynollah_ocr_cnnrnn_20250805"
        if self.tables:
            if self.light_version:
                self.model_table_dir = dir_models + "/modelens_table_0t4_201124"
            else:
                self.model_table_dir = dir_models + "/eynollah-tables_20210319"

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

        self.model_page = self.our_load_model(self.model_page_dir)
        self.model_classifier = self.our_load_model(self.model_dir_of_col_classifier)
        self.model_bin = self.our_load_model(self.model_dir_of_binarization)
        if self.extract_only_images:
            self.model_region = self.our_load_model(self.model_region_dir_p_ens_light_only_images_extraction)
        else:
            self.model_textline = self.our_load_model(self.model_textline_dir)
            if self.light_version:
                self.model_region = self.our_load_model(self.model_region_dir_p_ens_light)
                self.model_region_1_2 = self.our_load_model(self.model_region_dir_p_1_2_sp_np)
            else:
                self.model_region = self.our_load_model(self.model_region_dir_p_ens)
                self.model_region_p2 = self.our_load_model(self.model_region_dir_p2)
                self.model_enhancement = self.our_load_model(self.model_dir_of_enhancement)
            ###self.model_region_fl_new = self.our_load_model(self.model_region_dir_fully_new)
            self.model_region_fl_np = self.our_load_model(self.model_region_dir_fully_np)
            self.model_region_fl = self.our_load_model(self.model_region_dir_fully)
            if self.reading_order_machine_based:
                self.model_reading_order = self.our_load_model(self.model_reading_order_dir)
            if self.ocr and self.tr:
                self.model_ocr = VisionEncoderDecoderModel.from_pretrained(self.model_ocr_dir)
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                #("microsoft/trocr-base-printed")#("microsoft/trocr-base-handwritten")
                self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            elif self.ocr and not self.tr:
                model_ocr = load_model(self.model_ocr_dir , compile=False)
                
                self.prediction_model = tf.keras.models.Model(
                                model_ocr.get_layer(name = "image").input, 
                                model_ocr.get_layer(name = "dense2").output)
                if not batch_size_ocr:
                    self.b_s_ocr = 8
                else:
                    self.b_s_ocr = int(batch_size_ocr)

                    
                with open(os.path.join(self.model_ocr_dir, "characters_org.txt"),"r") as config_file:
                    characters = json.load(config_file)

                    
                AUTOTUNE = tf.data.AUTOTUNE

                # Mapping characters to integers.
                char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

                # Mapping integers back to original characters.
                self.num_to_char = StringLookup(
                    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
                )
                
            if self.tables:
                self.model_table = self.our_load_model(self.model_table_dir)

    def cache_images(self, image_filename=None, image_pil=None, dpi=None):
        ret = {}
        t_c0 = time.time()
        if image_filename:
            ret['img'] = cv2.imread(image_filename)
            if self.light_version:
                self.dpi = 100
            else:
                self.dpi = check_dpi(image_filename)
        else:
            ret['img'] = pil2cv(image_pil)
            if self.light_version:
                self.dpi = 100
            else:
                self.dpi = check_dpi(image_pil)
        ret['img_grayscale'] = cv2.cvtColor(ret['img'], cv2.COLOR_BGR2GRAY)
        for prefix in ('',  '_grayscale'):
            ret[f'img{prefix}_uint8'] = ret[f'img{prefix}'].astype(np.uint8)
        self._imgs = ret
        if dpi is not None:
            self.dpi = dpi

    def reset_file_name_dir(self, image_filename):
        t_c = time.time()
        self.cache_images(image_filename=image_filename)

        self.plotter = None if not self.enable_plotting else EynollahPlotter(
            dir_out=self.dir_out,
            dir_of_all=self.dir_of_all,
            dir_save_page=self.dir_save_page,
            dir_of_deskewed=self.dir_of_deskewed,
            dir_of_cropped_images=self.dir_of_cropped_images,
            dir_of_layout=self.dir_of_layout,
            image_filename_stem=Path(Path(image_filename).name).stem)

        self.writer = EynollahXmlWriter(
            dir_out=self.dir_out,
            image_filename=image_filename,
            curved_line=self.curved_line,
            textline_light = self.textline_light)

    def imread(self, grayscale=False, uint8=True):
        key = 'img'
        if grayscale:
            key += '_grayscale'
        if uint8:
            key += '_uint8'
        return self._imgs[key].copy()

    def isNaN(self, num):
        return num != num

    def predict_enhancement(self, img):
        self.logger.debug("enter predict_enhancement")

        img_height_model = self.model_enhancement.layers[-1].output_shape[1]
        img_width_model = self.model_enhancement.layers[-1].output_shape[2]
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
                label_p_pred = self.model_enhancement.predict(img_patch, verbose=0)
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

    def calculate_width_height_by_columns_extract_only_images(self, img, num_col, width_early, label_p_pred):
        self.logger.debug("enter calculate_width_height_by_columns")
        if num_col == 1:
            img_w_new = 700
        elif num_col == 2:
            img_w_new = 900
        elif num_col == 3:
            img_w_new = 1500
        elif num_col == 4:
            img_w_new = 1800
        elif num_col == 5:
            img_w_new = 2200
        elif num_col == 6:
            img_w_new = 2500
        img_h_new = img_w_new * img.shape[0] // img.shape[1]

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

        label_p_pred = self.model_classifier.predict(img_in, verbose=0)
        num_col = np.argmax(label_p_pred[0]) + 1

        self.logger.info("Found %s columns (%s)", num_col, label_p_pred)
        img_new, _ = self.calculate_width_height_by_columns(img, num_col, width_early, label_p_pred)

        if img_new.shape[1] > img.shape[1]:
            img_new = self.predict_enhancement(img_new)
            is_image_enhanced = True

        return img, img_new, is_image_enhanced

    def resize_and_enhance_image_with_column_classifier(self, light_version):
        self.logger.debug("enter resize_and_enhance_image_with_column_classifier")
        dpi = self.dpi
        self.logger.info("Detected %s DPI", dpi)
        if self.input_binary:
            img = self.imread()
            prediction_bin = self.do_prediction(True, img, self.model_bin, n_batch_inference=5)
            prediction_bin = 255 * (prediction_bin[:,:,0]==0)
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

            label_p_pred = self.model_classifier.predict(img_in, verbose=0)
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

            label_p_pred = self.model_classifier.predict(img_in, verbose=0)
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
        if not self.extract_only_images:
            if dpi < DPI_THRESHOLD:
                if light_version and num_col in (1,2):
                    img_new, num_column_is_classified = self.calculate_width_height_by_columns_1_2(
                        img, num_col, width_early, label_p_pred)
                else:
                    img_new, num_column_is_classified = self.calculate_width_height_by_columns(
                        img, num_col, width_early, label_p_pred)
                if light_version:
                    image_res = np.copy(img_new)
                else:
                    image_res = self.predict_enhancement(img_new)
                is_image_enhanced = True
            else:
                if light_version and num_col in (1,2):
                    img_new, num_column_is_classified = self.calculate_width_height_by_columns_1_2(
                        img, num_col, width_early, label_p_pred)
                    image_res = np.copy(img_new)
                    is_image_enhanced = True
                else:
                    num_column_is_classified = True
                    image_res = np.copy(img)
                    is_image_enhanced = False
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
        self.scale_y = self.img_hight_int / float(self.image.shape[0])
        self.scale_x = self.img_width_int / float(self.image.shape[1])

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
            thresholding_for_artificial_class_in_light_version=False, thresholding_for_fl_light_version=False, threshold_art_class_textline=0.1):

        self.logger.debug("enter do_prediction")
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

    def do_padding_with_scale(self, img, scale):
        h_n = int(img.shape[0]*scale)
        w_n = int(img.shape[1]*scale)

        channel0_avg = int( np.mean(img[:,:,0]) )
        channel1_avg = int( np.mean(img[:,:,1]) )
        channel2_avg = int( np.mean(img[:,:,2]) )

        h_diff = img.shape[0] - h_n
        w_diff = img.shape[1] - w_n

        h_start = int(0.5 * h_diff)
        w_start = int(0.5 * w_diff)

        img_res = resize_image(img, h_n, w_n)
        #label_res = resize_image(label, h_n, w_n)

        img_scaled_padded = np.copy(img)

        #label_scaled_padded = np.zeros(label.shape)

        img_scaled_padded[:,:,0] = channel0_avg
        img_scaled_padded[:,:,1] = channel1_avg
        img_scaled_padded[:,:,2] = channel2_avg

        img_scaled_padded[h_start:h_start+h_n, w_start:w_start+w_n,:] = img_res[:,:,:]
        #label_scaled_padded[h_start:h_start+h_n, w_start:w_start+w_n,:] = label_res[:,:,:]

        return img_scaled_padded#, label_scaled_padded

    def do_prediction_new_concept_scatter_nd(
            self, patches, img, model,
            n_batch_inference=1, marginal_of_patch_percent=0.1,
            thresholding_for_some_classes_in_light_version=False,
            thresholding_for_artificial_class_in_light_version=False):

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

            if thresholding_for_artificial_class_in_light_version:
                #seg_text = label_p_pred[0,:,:,1]
                #seg_text[seg_text<0.2] =0
                #seg_text[seg_text>0] =1
                #seg[seg_text==1]=1

                seg_art = label_p_pred[0,:,:,4]
                seg_art[seg_art<0.2] =0
                seg_art[seg_art>0] =1
                seg[seg_art==1]=4

            seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = resize_image(seg_color, img_h_page, img_w_page).astype(np.uint8)
            return prediction_true

        if img.shape[0] < img_height_model:
            img = resize_image(img, img_height_model, img.shape[1])
        if img.shape[1] < img_width_model:
            img = resize_image(img, img.shape[0], img_width_model)

        self.logger.debug("Patch size: %sx%s", img_height_model, img_width_model)
        ##margin = int(marginal_of_patch_percent * img_height_model)
        #width_mid = img_width_model - 2 * margin
        #height_mid = img_height_model - 2 * margin
        img = img / 255.0
        img = img.astype(np.float16)
        img_h = img.shape[0]
        img_w = img.shape[1]

        stride_x = img_width_model - 100
        stride_y = img_height_model - 100

        one_tensor = tf.ones_like(img)
        img_patches, one_patches = tf.image.extract_patches(
            images=[img, one_tensor],
            sizes=[1, img_height_model, img_width_model, 1],
            strides=[1, stride_y, stride_x, 1],
            rates=[1, 1, 1, 1],
            padding='SAME')
        img_patches = tf.squeeze(img_patches)
        one_patches = tf.squeeze(one_patches)
        img_patches_resh = tf.reshape(img_patches, shape=(img_patches.shape[0] * img_patches.shape[1],
                                                          img_height_model, img_width_model, 3))
        pred_patches = model.predict(img_patches_resh, batch_size=n_batch_inference)
        one_patches = tf.reshape(one_patches, shape=(img_patches.shape[0] * img_patches.shape[1],
                                                     img_height_model, img_width_model, 3))
        x = tf.range(img.shape[1])
        y = tf.range(img.shape[0])
        x, y = tf.meshgrid(x, y)
        indices = tf.stack([y, x], axis=-1)

        indices_patches = tf.image.extract_patches(
            images=tf.expand_dims(indices, axis=0),
            sizes=[1, img_height_model, img_width_model, 1],
            strides=[1, stride_y, stride_x, 1],
            rates=[1, 1, 1, 1],
            padding='SAME')
        indices_patches =  tf.squeeze(indices_patches)
        indices_patches = tf.reshape(indices_patches, shape=(img_patches.shape[0] * img_patches.shape[1],
                                                             img_height_model, img_width_model, 2))
        margin_y = int( 0.5 * (img_height_model - stride_y) )
        margin_x = int( 0.5 * (img_width_model - stride_x) )

        mask_margin = np.zeros((img_height_model, img_width_model))
        mask_margin[margin_y:img_height_model - margin_y,
                    margin_x:img_width_model - margin_x] = 1

        indices_patches_array = indices_patches.numpy()
        for i in range(indices_patches_array.shape[0]):
            indices_patches_array[i,:,:,0] = indices_patches_array[i,:,:,0]*mask_margin
            indices_patches_array[i,:,:,1] = indices_patches_array[i,:,:,1]*mask_margin

        reconstructed = tf.scatter_nd(
            indices=indices_patches_array,
            updates=pred_patches,
            shape=(img.shape[0], img.shape[1], pred_patches.shape[-1])).numpy()

        prediction_true = np.argmax(reconstructed, axis=2).astype(np.uint8)
        gc.collect()
        return np.repeat(prediction_true[:, :, np.newaxis], 3, axis=2)

    def do_prediction_new_concept(
            self, patches, img, model,
            n_batch_inference=1, marginal_of_patch_percent=0.1,
            thresholding_for_some_classes_in_light_version=False,
            thresholding_for_artificial_class_in_light_version=False, threshold_art_class_textline=0.1, threshold_art_class_layout=0.1):

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
                        
                        if thresholding_for_artificial_class_in_light_version or thresholding_for_some_classes_in_light_version:
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
                            if thresholding_for_artificial_class_in_light_version or thresholding_for_some_classes_in_light_version:
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
                            if thresholding_for_artificial_class_in_light_version or thresholding_for_some_classes_in_light_version:
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
                                            
                            if thresholding_for_artificial_class_in_light_version or thresholding_for_some_classes_in_light_version:
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
                            if thresholding_for_artificial_class_in_light_version or thresholding_for_some_classes_in_light_version:
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
                            if thresholding_for_artificial_class_in_light_version or thresholding_for_some_classes_in_light_version:
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
                            if thresholding_for_artificial_class_in_light_version or thresholding_for_some_classes_in_light_version:
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
                            if thresholding_for_artificial_class_in_light_version or thresholding_for_some_classes_in_light_version:
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
                            if thresholding_for_artificial_class_in_light_version or thresholding_for_some_classes_in_light_version:
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
                            if thresholding_for_artificial_class_in_light_version or thresholding_for_some_classes_in_light_version:
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
            img = cv2.GaussianBlur(self.image, (5, 5), 0)
            img_page_prediction = self.do_prediction(False, img, self.model_page)
            imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgray, 0, 255, 0)
            thresh = cv2.dilate(thresh, KERNEL, iterations=3)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours)>0:
                cnt_size = np.array([cv2.contourArea(contours[j])
                                     for j in range(len(contours))])
                cnt = contours[np.argmax(cnt_size)]
                x, y, w, h = cv2.boundingRect(cnt)
                if x <= 30:
                    w += x
                    x = 0
                if (self.image.shape[1] - (x + w)) <= 30:
                    w = w + (self.image.shape[1] - (x + w))
                if y <= 30:
                    h = h + y
                    y = 0
                if (self.image.shape[0] - (y + h)) <= 30:
                    h = h + (self.image.shape[0] - (y + h))
                box = [x, y, w, h]
            else:
                box = [0, 0, img.shape[1], img.shape[0]]
            cropped_page, page_coord = crop_image_inside_box(box, self.image)
            cont_page.append(np.array([[page_coord[2], page_coord[0]],
                                       [page_coord[3], page_coord[0]],
                                       [page_coord[3], page_coord[1]],
                                       [page_coord[2], page_coord[1]]]))
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
            img_page_prediction = self.do_prediction(False, img, self.model_page)

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
        model_region = self.model_region_fl if patches else self.model_region_fl_np

        if self.light_version:
            thresholding_for_fl_light_version = True
        elif not patches:
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

        prediction_regions = self.do_prediction(patches, img, model_region, marginal_of_patch_percent=0.1, n_batch_inference=3, thresholding_for_fl_light_version=thresholding_for_fl_light_version)
        prediction_regions = resize_image(prediction_regions, img_height_h, img_width_h)
        self.logger.debug("exit extract_text_regions")
        return prediction_regions, prediction_regions

    def extract_text_regions(self, img, patches, cols):
        self.logger.debug("enter extract_text_regions")
        img_height_h = img.shape[0]
        img_width_h = img.shape[1]
        model_region = self.model_region_fl if patches else self.model_region_fl_np

        if not patches:
            img = otsu_copy_binary(img)
            img = img.astype(np.uint8)
            prediction_regions2 = None
        elif cols:
            if cols == 1:
                img_height_new = int(img_height_h * 0.7)
                img_width_new = int(img_width_h * 0.7)
            elif cols == 2:
                img_height_new = int(img_height_h * 0.4)
                img_width_new = int(img_width_h * 0.4)
            else:
                img_height_new = int(img_height_h * 0.3)
                img_width_new = int(img_width_h * 0.3)
            img2 = otsu_copy_binary(img)
            img2 = img2.astype(np.uint8)
            img2 = resize_image(img2, img_height_new, img_width_new)
            prediction_regions2 = self.do_prediction(patches, img2, model_region, marginal_of_patch_percent=0.1)
            prediction_regions2 = resize_image(prediction_regions2, img_height_h, img_width_h)

            img = otsu_copy_binary(img).astype(np.uint8)
            if cols == 1:
                img = resize_image(img, int(img_height_h * 0.5), int(img_width_h * 0.5)).astype(np.uint8)
            elif cols == 2 and img_width_h >= 2000:
                img = resize_image(img, int(img_height_h * 0.9), int(img_width_h * 0.9)).astype(np.uint8)
            elif cols == 3 and ((self.scale_x == 1 and img_width_h > 3000) or
                                (self.scale_x != 1 and img_width_h > 2800)):
                img = resize_image(img, 2800 * img_height_h // img_width_h, 2800).astype(np.uint8)
            elif cols == 4 and ((self.scale_x == 1 and img_width_h > 4000) or
                                (self.scale_x != 1 and img_width_h > 3700)):
                img = resize_image(img, 3700 * img_height_h // img_width_h, 3700).astype(np.uint8)
            elif cols == 4:
                img = resize_image(img, int(img_height_h * 0.9), int(img_width_h * 0.9)).astype(np.uint8)
            elif cols == 5 and self.scale_x == 1 and img_width_h > 5000:
                img = resize_image(img, int(img_height_h * 0.7), int(img_width_h * 0.7)).astype(np.uint8)
            elif cols == 5:
                img = resize_image(img, int(img_height_h * 0.9), int(img_width_h * 0.9)).astype(np.uint8)
            elif img_width_h > 5600:
                img = resize_image(img, 5600 * img_height_h // img_width_h, 5600).astype(np.uint8)
            else:
                img = resize_image(img, int(img_height_h * 0.9), int(img_width_h * 0.9)).astype(np.uint8)

        prediction_regions = self.do_prediction(patches, img, model_region, marginal_of_patch_percent=0.1)
        prediction_regions = resize_image(prediction_regions, img_height_h, img_width_h)
        self.logger.debug("exit extract_text_regions")
        return prediction_regions, prediction_regions2

    def get_slopes_and_deskew_new_light2(self, contours, contours_par, textline_mask_tot, image_page_rotated, boxes, slope_deskew):

        polygons_of_textlines = return_contours_of_interested_region(textline_mask_tot,1,0.00001)
        M_main_tot = [cv2.moments(polygons_of_textlines[j])
                      for j in range(len(polygons_of_textlines))]
        cx_main_tot = [(M_main_tot[j]["m10"] / (M_main_tot[j]["m00"] + 1e-32)) for j in range(len(M_main_tot))]
        cy_main_tot = [(M_main_tot[j]["m01"] / (M_main_tot[j]["m00"] + 1e-32)) for j in range(len(M_main_tot))]

        args_textlines = np.array(range(len(polygons_of_textlines)))
        all_found_textline_polygons = []
        slopes = []
        all_box_coord =[]

        for index, con_region_ind in enumerate(contours_par):
            results = [cv2.pointPolygonTest(con_region_ind, (cx_main_tot[ind], cy_main_tot[ind]), False)
                       for ind in args_textlines ]
            results = np.array(results)
            indexes_in = args_textlines[results==1]
            textlines_ins = [polygons_of_textlines[ind] for ind in indexes_in]

            all_found_textline_polygons.append(textlines_ins[::-1])
            slopes.append(slope_deskew)

            _, crop_coor = crop_image_inside_box(boxes[index],image_page_rotated)
            all_box_coord.append(crop_coor)

        return all_found_textline_polygons, boxes, contours, contours_par, all_box_coord, np.array(range(len(contours_par))), slopes

    def get_slopes_and_deskew_new_light(self, contours, contours_par, textline_mask_tot, image_page_rotated, boxes, slope_deskew):
        if not len(contours):
            return [], [], [], [], [], [], []
        self.logger.debug("enter get_slopes_and_deskew_new_light")
        results = self.executor.map(partial(do_work_of_slopes_new_light,
                                            textline_mask_tot_ea=textline_mask_tot,
                                            image_page_rotated=image_page_rotated,
                                            slope_deskew=slope_deskew,textline_light=self.textline_light,
                                            logger=self.logger,),
                                    boxes, contours, contours_par, range(len(contours_par)))
        #textline_polygons, boxes, text_regions, text_regions_par, box_coord, index_text_con, slopes = zip(*results)
        self.logger.debug("exit get_slopes_and_deskew_new_light")
        return tuple(zip(*results))

    def get_slopes_and_deskew_new(self, contours, contours_par, textline_mask_tot, image_page_rotated, boxes, slope_deskew):
        if not len(contours):
            return [], [], [], [], [], [], []
        self.logger.debug("enter get_slopes_and_deskew_new")
        results = self.executor.map(partial(do_work_of_slopes_new,
                                            textline_mask_tot_ea=textline_mask_tot,
                                            image_page_rotated=image_page_rotated,
                                            slope_deskew=slope_deskew,
                                            MAX_SLOPE=MAX_SLOPE,
                                            KERNEL=KERNEL,
                                            logger=self.logger,
                                            plotter=self.plotter,),
                                    boxes, contours, contours_par, range(len(contours_par)))
        #textline_polygons, boxes, text_regions, text_regions_par, box_coord, index_text_con, slopes = zip(*results)
        self.logger.debug("exit get_slopes_and_deskew_new")
        return tuple(zip(*results))

    def get_slopes_and_deskew_new_curved(self, contours, contours_par, textline_mask_tot, image_page_rotated, boxes, mask_texts_only, num_col, scale_par, slope_deskew):
        if not len(contours):
            return [], [], [], [], [], [], []
        self.logger.debug("enter get_slopes_and_deskew_new_curved")
        results = self.executor.map(partial(do_work_of_slopes_new_curved,
                                            textline_mask_tot_ea=textline_mask_tot,
                                            image_page_rotated=image_page_rotated,
                                            mask_texts_only=mask_texts_only,
                                            num_col=num_col,
                                            scale_par=scale_par,
                                            slope_deskew=slope_deskew,
                                            MAX_SLOPE=MAX_SLOPE,
                                            KERNEL=KERNEL,
                                            logger=self.logger,
                                            plotter=self.plotter,),
                                    boxes, contours, contours_par, range(len(contours_par)))
        #textline_polygons, boxes, text_regions, text_regions_par, box_coord, index_text_con, slopes = zip(*results)
        self.logger.debug("exit get_slopes_and_deskew_new_curved")
        return tuple(zip(*results))

    def textline_contours(self, img, use_patches, scaler_h, scaler_w, num_col_classifier=None):
        self.logger.debug('enter textline_contours')

        #img = img.astype(np.uint8)
        img_org = np.copy(img)
        img_h = img_org.shape[0]
        img_w = img_org.shape[1]
        img = resize_image(img_org, int(img_org.shape[0] * scaler_h), int(img_org.shape[1] * scaler_w))

        prediction_textline = self.do_prediction(
            use_patches, img, self.model_textline,
            marginal_of_patch_percent=0.15, n_batch_inference=3,
            thresholding_for_artificial_class_in_light_version=self.textline_light, threshold_art_class_textline=self.threshold_art_class_textline)
        #if not self.textline_light:
            #if num_col_classifier==1:
                #prediction_textline_nopatch = self.do_prediction(False, img, self.model_textline)
                #prediction_textline[:,:][prediction_textline_nopatch[:,:]==0] = 0

        prediction_textline = resize_image(prediction_textline, img_h, img_w)
        textline_mask_tot_ea_art = (prediction_textline[:,:]==2)*1

        old_art = np.copy(textline_mask_tot_ea_art)
        if not self.textline_light:
            textline_mask_tot_ea_art = textline_mask_tot_ea_art.astype('uint8')
            #textline_mask_tot_ea_art = cv2.dilate(textline_mask_tot_ea_art, KERNEL, iterations=1)
            prediction_textline[:,:][textline_mask_tot_ea_art[:,:]==1]=2
        """
        else:
            textline_mask_tot_ea_art = textline_mask_tot_ea_art.astype('uint8')
            hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1))
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            ##cv2.imwrite('textline_mask_tot_ea_art.png', textline_mask_tot_ea_art)
            textline_mask_tot_ea_art = cv2.dilate(textline_mask_tot_ea_art, hor_kernel, iterations=1)
            
            ###cv2.imwrite('dil_textline_mask_tot_ea_art.png', dil_textline_mask_tot_ea_art)
            
            textline_mask_tot_ea_art = textline_mask_tot_ea_art.astype('uint8')
            
            #print(np.shape(dil_textline_mask_tot_ea_art), np.unique(dil_textline_mask_tot_ea_art), 'dil_textline_mask_tot_ea_art')
            tsk = time.time()
            skeleton_art_textline = skeletonize(textline_mask_tot_ea_art[:,:,0])
            
            skeleton_art_textline =  skeleton_art_textline*1
            
            skeleton_art_textline = skeleton_art_textline.astype('uint8')
            
            skeleton_art_textline = cv2.dilate(skeleton_art_textline, kernel, iterations=1)
            
            #print(np.unique(skeleton_art_textline), np.shape(skeleton_art_textline))
            
            #print(skeleton_art_textline, np.unique(skeleton_art_textline))
            
            #cv2.imwrite('skeleton_art_textline.png', skeleton_art_textline)

            
            prediction_textline[:,:,0][skeleton_art_textline[:,:]==1]=2
            
            #cv2.imwrite('prediction_textline1.png', prediction_textline[:,:,0])
            
            ##hor_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
            ##ver_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            ##textline_mask_tot_ea_main = (prediction_textline[:,:]==1)*1
            ##textline_mask_tot_ea_main = textline_mask_tot_ea_main.astype('uint8')
            
            ##dil_textline_mask_tot_ea_main = cv2.erode(textline_mask_tot_ea_main, ver_kernel2, iterations=1)
            
            ##dil_textline_mask_tot_ea_main = cv2.dilate(textline_mask_tot_ea_main, hor_kernel2, iterations=1)
            
            ##dil_textline_mask_tot_ea_main = cv2.dilate(textline_mask_tot_ea_main, ver_kernel2, iterations=1)
            
            ##prediction_textline[:,:][dil_textline_mask_tot_ea_main[:,:]==1]=1
            
        """
        
        textline_mask_tot_ea_lines = (prediction_textline[:,:]==1)*1
        textline_mask_tot_ea_lines = textline_mask_tot_ea_lines.astype('uint8')
        if not self.textline_light:
            textline_mask_tot_ea_lines = cv2.dilate(textline_mask_tot_ea_lines, KERNEL, iterations=1)

        prediction_textline[:,:][textline_mask_tot_ea_lines[:,:]==1]=1
        if not self.textline_light:
            prediction_textline[:,:][old_art[:,:]==1]=2
            
        #cv2.imwrite('prediction_textline2.png', prediction_textline[:,:,0])

        prediction_textline_longshot = self.do_prediction(False, img, self.model_textline)
        prediction_textline_longshot_true_size = resize_image(prediction_textline_longshot, img_h, img_w)
        
        
        #cv2.imwrite('prediction_textline.png', prediction_textline[:,:,0])
        #sys.exit()
        self.logger.debug('exit textline_contours')
        return ((prediction_textline[:, :, 0]==1).astype(np.uint8),
                (prediction_textline_longshot_true_size[:, :, 0]==1).astype(np.uint8))


    def do_work_of_slopes(self, q, poly, box_sub, boxes_per_process, textline_mask_tot, contours_per_process):
        self.logger.debug('enter do_work_of_slopes')
        slope_biggest = 0
        slopes_sub = []
        boxes_sub_new = []
        poly_sub = []
        for mv in range(len(boxes_per_process)):
            crop_img, _ = crop_image_inside_box(boxes_per_process[mv], np.repeat(textline_mask_tot[:, :, np.newaxis], 3, axis=2))
            crop_img = crop_img[:, :, 0]
            crop_img = cv2.erode(crop_img, KERNEL, iterations=2)
            try:
                textline_con, hierarchy = return_contours_of_image(crop_img)
                textline_con_fil = filter_contours_area_of_image(crop_img, textline_con, hierarchy, max_area=1, min_area=0.0008)
                y_diff_mean = find_contours_mean_y_diff(textline_con_fil)
                sigma_des = max(1, int(y_diff_mean * (4.0 / 40.0)))
                crop_img[crop_img > 0] = 1
                slope_corresponding_textregion = return_deskew_slop_old_mp(crop_img, sigma_des,
                                                                    logger=self.logger, plotter=self.plotter)
            except Exception as why:
                self.logger.error(why)
                slope_corresponding_textregion = MAX_SLOPE

            if slope_corresponding_textregion == MAX_SLOPE:
                slope_corresponding_textregion = slope_biggest
            slopes_sub.append(slope_corresponding_textregion)

            cnt_clean_rot = textline_contours_postprocessing(
                crop_img, slope_corresponding_textregion, contours_per_process[mv], boxes_per_process[mv])

            poly_sub.append(cnt_clean_rot)
            boxes_sub_new.append(boxes_per_process[mv])

        q.put(slopes_sub)
        poly.put(poly_sub)
        box_sub.put(boxes_sub_new)
        self.logger.debug('exit do_work_of_slopes')

    def get_regions_light_v_extract_only_images(self,img,is_image_enhanced, num_col_classifier):
        self.logger.debug("enter get_regions_extract_images_only")
        erosion_hurts = False
        img_org = np.copy(img)
        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]

        if num_col_classifier == 1:
            img_w_new = 700
        elif num_col_classifier == 2:
            img_w_new = 900
        elif num_col_classifier == 3:
            img_w_new = 1500
        elif num_col_classifier == 4:
            img_w_new = 1800
        elif num_col_classifier == 5:
            img_w_new = 2200
        elif num_col_classifier == 6:
            img_w_new = 2500
        img_h_new = int(img.shape[0] / float(img.shape[1]) * img_w_new)
        img_resized = resize_image(img,img_h_new, img_w_new )

        prediction_regions_org, _ = self.do_prediction_new_concept(True, img_resized, self.model_region)

        prediction_regions_org = resize_image(prediction_regions_org,img_height_h, img_width_h )
        image_page, page_coord, cont_page = self.extract_page()

        prediction_regions_org = prediction_regions_org[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        prediction_regions_org=prediction_regions_org[:,:,0]

        mask_lines_only = (prediction_regions_org[:,:] ==3)*1
        mask_texts_only = (prediction_regions_org[:,:] ==1)*1
        mask_images_only=(prediction_regions_org[:,:] ==2)*1

        polygons_lines_xml, hir_lines_xml = return_contours_of_image(mask_lines_only)
        polygons_lines_xml = textline_con_fil = filter_contours_area_of_image(
            mask_lines_only, polygons_lines_xml, hir_lines_xml, max_area=1, min_area=0.00001)

        polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only,1,0.00001)
        polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only,1,0.00001)

        text_regions_p_true = np.zeros(prediction_regions_org.shape)
        text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_lines, color=(3,3,3))

        text_regions_p_true[:,:][mask_images_only[:,:] == 1] = 2
        text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts=polygons_of_only_texts, color=(1,1,1))

        text_regions_p_true[text_regions_p_true.shape[0]-15:text_regions_p_true.shape[0], :] = 0
        text_regions_p_true[:, text_regions_p_true.shape[1]-15:text_regions_p_true.shape[1]] = 0

        ##polygons_of_images = return_contours_of_interested_region(text_regions_p_true, 2, 0.0001)
        polygons_of_images = return_contours_of_interested_region(text_regions_p_true, 2, 0.001)
        image_boundary_of_doc = np.zeros((text_regions_p_true.shape[0], text_regions_p_true.shape[1]))

        ###image_boundary_of_doc[:6, :] = 1
        ###image_boundary_of_doc[text_regions_p_true.shape[0]-6:text_regions_p_true.shape[0], :] = 1

        ###image_boundary_of_doc[:, :6] = 1
        ###image_boundary_of_doc[:, text_regions_p_true.shape[1]-6:text_regions_p_true.shape[1]] = 1

        polygons_of_images_fin = []
        for ploy_img_ind in polygons_of_images:
            """
            test_poly_image = np.zeros((text_regions_p_true.shape[0], text_regions_p_true.shape[1]))
            test_poly_image = cv2.fillPoly(test_poly_image, pts=[ploy_img_ind], color=(1,1,1))

            test_poly_image = test_poly_image + image_boundary_of_doc
            test_poly_image_intersected_area = ( test_poly_image[:,:]==2 )*1

            test_poly_image_intersected_area = test_poly_image_intersected_area.sum()

            if test_poly_image_intersected_area==0:
                ##polygons_of_images_fin.append(ploy_img_ind)

                box = cv2.boundingRect(ploy_img_ind)
                _, page_coord_img = crop_image_inside_box(box, text_regions_p_true)
                # cont_page.append(np.array([[page_coord[2], page_coord[0]],
                #                            [page_coord[3], page_coord[0]],
                #                            [page_coord[3], page_coord[1]],
                #                            [page_coord[2], page_coord[1]]]))
                polygons_of_images_fin.append(np.array([[page_coord_img[2], page_coord_img[0]],
                                                        [page_coord_img[3], page_coord_img[0]],
                                                        [page_coord_img[3], page_coord_img[1]],
                                                        [page_coord_img[2], page_coord_img[1]]]) )
            """
            box = x, y, w, h = cv2.boundingRect(ploy_img_ind)
            if h < 150 or w < 150:
                pass
            else:
                _, page_coord_img = crop_image_inside_box(box, text_regions_p_true)
                # cont_page.append(np.array([[page_coord[2], page_coord[0]],
                #                            [page_coord[3], page_coord[0]],
                #                            [page_coord[3], page_coord[1]],
                #                            [page_coord[2], page_coord[1]]]))
                polygons_of_images_fin.append(np.array([[page_coord_img[2], page_coord_img[0]],
                                                        [page_coord_img[3], page_coord_img[0]],
                                                        [page_coord_img[3], page_coord_img[1]],
                                                        [page_coord_img[2], page_coord_img[1]]]))

        self.logger.debug("exit get_regions_extract_images_only")
        return text_regions_p_true, erosion_hurts, polygons_lines_xml, polygons_of_images_fin, image_page, page_coord, cont_page

    def get_regions_light_v(self,img,is_image_enhanced, num_col_classifier, skip_layout_and_reading_order=False):
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
            ###prediction_bin = self.do_prediction(True, img_resized, self.model_bin, n_batch_inference=5)

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
        if (self.ocr and self.tr) and not self.input_binary:
            prediction_bin = self.do_prediction(True, img_resized, self.model_bin, n_batch_inference=5)
            prediction_bin = 255 * (prediction_bin[:,:,0] == 0)
            prediction_bin = np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2)
            prediction_bin = prediction_bin.astype(np.uint16)
            #img= np.copy(prediction_bin)
            img_bin = np.copy(prediction_bin)
        else:
            img_bin = np.copy(img_resized)
        #print("inside 1 ", time.time()-t_in)

        ###textline_mask_tot_ea = self.run_textline(img_bin)
        self.logger.debug("detecting textlines on %s with %d colors", str(img_resized.shape), len(np.unique(img_resized)))
        textline_mask_tot_ea = self.run_textline(img_resized, num_col_classifier)
        textline_mask_tot_ea = resize_image(textline_mask_tot_ea,img_height_h, img_width_h )

        #print(self.image_org.shape)
        #cv2.imwrite('textline.png', textline_mask_tot_ea)

        #plt.imshwo(self.image_page_org_size)
        #plt.show()
        if not skip_layout_and_reading_order:
            #print("inside 2 ", time.time()-t_in)
            if num_col_classifier == 1 or num_col_classifier == 2:
                if self.image_org.shape[0]/self.image_org.shape[1] > 2.5:
                    self.logger.debug("resized to %dx%d for %d cols",
                                      img_resized.shape[1], img_resized.shape[0], num_col_classifier)
                    prediction_regions_org, confidence_matrix = self.do_prediction_new_concept(
                        True, img_resized, self.model_region_1_2, n_batch_inference=1,
                        thresholding_for_some_classes_in_light_version=True, threshold_art_class_layout=self.threshold_art_class_layout)
                else:
                    prediction_regions_org = np.zeros((self.image_org.shape[0], self.image_org.shape[1], 3))
                    confidence_matrix = np.zeros((self.image_org.shape[0], self.image_org.shape[1]))
                    prediction_regions_page, confidence_matrix_page = self.do_prediction_new_concept(
                        False, self.image_page_org_size, self.model_region_1_2, n_batch_inference=1,
                        thresholding_for_artificial_class_in_light_version=True, threshold_art_class_layout=self.threshold_art_class_layout)
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
                    True, img_resized, self.model_region_1_2, n_batch_inference=2,
                    thresholding_for_some_classes_in_light_version=True, threshold_art_class_layout=self.threshold_art_class_layout)
            ###prediction_regions_org = self.do_prediction(True, img_bin, self.model_region, n_batch_inference=3, thresholding_for_some_classes_in_light_version=True)
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

            polygons_lines_xml, hir_lines_xml = return_contours_of_image(mask_lines_only)
            test_khat = np.zeros(prediction_regions_org.shape)
            test_khat = cv2.fillPoly(test_khat, pts=polygons_lines_xml, color=(1,1,1))

            #plt.imshow(test_khat[:,:])
            #plt.show()
            #for jv in range(1):
                #print(jv, hir_lines_xml[0][232][3])
                #test_khat = np.zeros(prediction_regions_org.shape)
                #test_khat = cv2.fillPoly(test_khat, pts = [polygons_lines_xml[232]], color=(1,1,1))
                #plt.imshow(test_khat[:,:])
                #plt.show()

            polygons_lines_xml = filter_contours_area_of_image(
                mask_lines_only, polygons_lines_xml, hir_lines_xml, max_area=1, min_area=0.00001)

            test_khat = np.zeros(prediction_regions_org.shape)
            test_khat = cv2.fillPoly(test_khat, pts = polygons_lines_xml, color=(1,1,1))

            #plt.imshow(test_khat[:,:])
            #plt.show()
            #sys.exit()

            polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only,1,0.00001)
            ##polygons_of_only_texts = self.dilate_textregions_contours(polygons_of_only_texts)
            polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only,1,0.00001)

            text_regions_p_true = np.zeros(prediction_regions_org.shape)
            text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts=polygons_of_only_lines, color=(3,3,3))

            text_regions_p_true[:,:][mask_images_only[:,:] == 1] = 2
            text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_texts, color=(1,1,1))

            #plt.imshow(textline_mask_tot_ea)
            #plt.show()

            textline_mask_tot_ea[(text_regions_p_true==0) | (text_regions_p_true==4) ] = 0

            #plt.imshow(textline_mask_tot_ea)
            #plt.show()
            #print("inside 4 ", time.time()-t_in)
            self.logger.debug("exit get_regions_light_v")
            return text_regions_p_true, erosion_hurts, polygons_lines_xml, textline_mask_tot_ea, img_bin, confidence_matrix
        else:
            img_bin = resize_image(img_bin,img_height_h, img_width_h )
            self.logger.debug("exit get_regions_light_v")
            return None, erosion_hurts, None, textline_mask_tot_ea, img_bin, None

    def get_regions_from_xy_2models(self,img,is_image_enhanced, num_col_classifier):
        self.logger.debug("enter get_regions_from_xy_2models")
        erosion_hurts = False
        img_org = np.copy(img)
        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]

        ratio_y=1.3
        ratio_x=1

        img = resize_image(img_org, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))
        prediction_regions_org_y = self.do_prediction(True, img, self.model_region)
        prediction_regions_org_y = resize_image(prediction_regions_org_y, img_height_h, img_width_h )

        #plt.imshow(prediction_regions_org_y[:,:,0])
        #plt.show()
        prediction_regions_org_y = prediction_regions_org_y[:,:,0]
        mask_zeros_y = (prediction_regions_org_y[:,:]==0)*1

        ##img_only_regions_with_sep = ( (prediction_regions_org_y[:,:] != 3) & (prediction_regions_org_y[:,:] != 0) )*1
        img_only_regions_with_sep = (prediction_regions_org_y == 1).astype(np.uint8)
        try:
            img_only_regions = cv2.erode(img_only_regions_with_sep[:,:], KERNEL, iterations=20)
            _, _ = find_num_col(img_only_regions, num_col_classifier, self.tables, multiplier=6.0)
            img = resize_image(img_org, int(img_org.shape[0]), int(img_org.shape[1]*(1.2 if is_image_enhanced else 1)))

            prediction_regions_org = self.do_prediction(True, img, self.model_region)
            prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h )

            prediction_regions_org=prediction_regions_org[:,:,0]
            prediction_regions_org[(prediction_regions_org[:,:]==1) & (mask_zeros_y[:,:]==1)]=0

            img = resize_image(img_org, int(img_org.shape[0]), int(img_org.shape[1]))

            prediction_regions_org2 = self.do_prediction(True, img, self.model_region_p2, marginal_of_patch_percent=0.2)
            prediction_regions_org2=resize_image(prediction_regions_org2, img_height_h, img_width_h )

            mask_zeros2 = (prediction_regions_org2[:,:,0] == 0)
            mask_lines2 = (prediction_regions_org2[:,:,0] == 3)
            text_sume_early = (prediction_regions_org[:,:] == 1).sum()
            prediction_regions_org_copy = np.copy(prediction_regions_org)
            prediction_regions_org_copy[(prediction_regions_org_copy[:,:]==1) & (mask_zeros2[:,:]==1)] = 0
            text_sume_second = ((prediction_regions_org_copy[:,:]==1)*1).sum()
            rate_two_models = 100. * text_sume_second / text_sume_early

            self.logger.info("ratio_of_two_models: %s", rate_two_models)
            if not(is_image_enhanced and rate_two_models < RATIO_OF_TWO_MODEL_THRESHOLD):
                prediction_regions_org = np.copy(prediction_regions_org_copy)

            prediction_regions_org[(mask_lines2[:,:]==1) & (prediction_regions_org[:,:]==0)]=3
            mask_lines_only=(prediction_regions_org[:,:]==3)*1
            prediction_regions_org = cv2.erode(prediction_regions_org[:,:], KERNEL, iterations=2)
            prediction_regions_org = cv2.dilate(prediction_regions_org[:,:], KERNEL, iterations=2)

            if rate_two_models<=40:
                if self.input_binary:
                    prediction_bin = np.copy(img_org)
                else:
                    prediction_bin = self.do_prediction(True, img_org, self.model_bin, n_batch_inference=5)
                    prediction_bin = resize_image(prediction_bin, img_height_h, img_width_h )
                    prediction_bin = 255 * (prediction_bin[:,:,0]==0)
                    prediction_bin = np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2)

                ratio_y=1
                ratio_x=1

                img = resize_image(prediction_bin, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))

                prediction_regions_org = self.do_prediction(True, img, self.model_region)
                prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h )
                prediction_regions_org=prediction_regions_org[:,:,0]

                mask_lines_only=(prediction_regions_org[:,:]==3)*1

            mask_texts_only=(prediction_regions_org[:,:]==1)*1
            mask_images_only=(prediction_regions_org[:,:]==2)*1

            polygons_lines_xml, hir_lines_xml = return_contours_of_image(mask_lines_only)
            polygons_lines_xml = filter_contours_area_of_image(
                mask_lines_only, polygons_lines_xml, hir_lines_xml, max_area=1, min_area=0.00001)

            polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only, 1, 0.00001)
            polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only, 1, 0.00001)

            text_regions_p_true = np.zeros(prediction_regions_org.shape)
            text_regions_p_true = cv2.fillPoly(text_regions_p_true,pts = polygons_of_only_lines, color=(3, 3, 3))
            text_regions_p_true[:,:][mask_images_only[:,:] == 1] = 2

            text_regions_p_true=cv2.fillPoly(text_regions_p_true,pts=polygons_of_only_texts, color=(1,1,1))

            self.logger.debug("exit get_regions_from_xy_2models")
            return text_regions_p_true, erosion_hurts, polygons_lines_xml
        except:
            if self.input_binary:
                prediction_bin = np.copy(img_org)
                prediction_bin = self.do_prediction(True, img_org, self.model_bin, n_batch_inference=5)
                prediction_bin = resize_image(prediction_bin, img_height_h, img_width_h )
                prediction_bin = 255 * (prediction_bin[:,:,0]==0)
                prediction_bin = np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2)
            else:
                prediction_bin = np.copy(img_org)
            ratio_y=1
            ratio_x=1


            img = resize_image(prediction_bin, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))
            prediction_regions_org = self.do_prediction(True, img, self.model_region)
            prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h )
            prediction_regions_org=prediction_regions_org[:,:,0]

            #mask_lines_only=(prediction_regions_org[:,:]==3)*1
            #img = resize_image(img_org, int(img_org.shape[0]*1), int(img_org.shape[1]*1))

            #prediction_regions_org = self.do_prediction(True, img, self.model_region)

            #prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h )

            #prediction_regions_org = prediction_regions_org[:,:,0]

            #prediction_regions_org[(prediction_regions_org[:,:] == 1) & (mask_zeros_y[:,:] == 1)]=0


            mask_lines_only = (prediction_regions_org == 3)*1
            mask_texts_only = (prediction_regions_org == 1)*1
            mask_images_only= (prediction_regions_org == 2)*1

            polygons_lines_xml, hir_lines_xml = return_contours_of_image(mask_lines_only)
            polygons_lines_xml = filter_contours_area_of_image(
                mask_lines_only, polygons_lines_xml, hir_lines_xml, max_area=1, min_area=0.00001)

            polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only,1,0.00001)
            polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only,1,0.00001)

            text_regions_p_true = np.zeros(prediction_regions_org.shape)
            text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_lines, color=(3,3,3))

            text_regions_p_true[:,:][mask_images_only[:,:] == 1] = 2
            text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_texts, color=(1,1,1))

            erosion_hurts = True
            self.logger.debug("exit get_regions_from_xy_2models")
            return text_regions_p_true, erosion_hurts, polygons_lines_xml

    def do_order_of_regions_full_layout(
            self, contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot):

        self.logger.debug("enter do_order_of_regions_full_layout")
        boxes = np.array(boxes, dtype=int) # to be on the safe side
        cx_text_only, cy_text_only, x_min_text_only, _, _, _, y_cor_x_min_main = find_new_features_of_contours(
            contours_only_text_parent)
        cx_text_only_h, cy_text_only_h, x_min_text_only_h, _, _, _, y_cor_x_min_main_h = find_new_features_of_contours(
            contours_only_text_parent_h)

        try:
            arg_text_con = []
            for ii in range(len(cx_text_only)):
                check_if_textregion_located_in_a_box = False
                for jj in range(len(boxes)):
                    if (x_min_text_only[ii] + 80 >= boxes[jj][0] and
                        x_min_text_only[ii] + 80 < boxes[jj][1] and
                        y_cor_x_min_main[ii] >= boxes[jj][2] and
                        y_cor_x_min_main[ii] < boxes[jj][3]):
                        arg_text_con.append(jj)
                        check_if_textregion_located_in_a_box = True
                        break
                if not check_if_textregion_located_in_a_box:
                    dists_tr_from_box = [math.sqrt((cx_text_only[ii] - boxes[jj][1]) ** 2 +
                                                   (cy_text_only[ii] - boxes[jj][2]) ** 2)
                                         for jj in range(len(boxes))]
                    ind_min = np.argmin(dists_tr_from_box)
                    arg_text_con.append(ind_min)
            args_contours = np.array(range(len(arg_text_con)))
            arg_text_con_h = []
            for ii in range(len(cx_text_only_h)):
                check_if_textregion_located_in_a_box = False
                for jj in range(len(boxes)):
                    if (x_min_text_only_h[ii] + 80 >= boxes[jj][0] and
                        x_min_text_only_h[ii] + 80 < boxes[jj][1] and
                        y_cor_x_min_main_h[ii] >= boxes[jj][2] and
                        y_cor_x_min_main_h[ii] < boxes[jj][3]):
                        arg_text_con_h.append(jj)
                        check_if_textregion_located_in_a_box = True
                        break
                if not check_if_textregion_located_in_a_box:
                    dists_tr_from_box = [math.sqrt((cx_text_only_h[ii] - boxes[jj][1]) ** 2 +
                                                   (cy_text_only_h[ii] - boxes[jj][2]) ** 2)
                                         for jj in range(len(boxes))]
                    ind_min = np.argmin(dists_tr_from_box)
                    arg_text_con_h.append(ind_min)
            args_contours_h = np.array(range(len(arg_text_con_h)))

            order_by_con_head = np.zeros(len(arg_text_con_h))
            order_by_con_main = np.zeros(len(arg_text_con))

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij in range(len(boxes)):
                ys = slice(*boxes[iij][2:4])
                xs = slice(*boxes[iij][0:2])
                args_contours_box = args_contours[np.array(arg_text_con) == iij]
                args_contours_box_h = args_contours_h[np.array(arg_text_con_h) == iij]
                con_inter_box = []
                con_inter_box_h = []

                for box in args_contours_box:
                    con_inter_box.append(contours_only_text_parent[box])

                for box in args_contours_box_h:
                    con_inter_box_h.append(contours_only_text_parent_h[box])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(
                    textline_mask_tot[ys, xs], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(
                    con_inter_box, con_inter_box_h,
                    matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_sorted_head = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 2]
                indexes_by_type_head = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 2]

                for zahler, _ in enumerate(args_contours_box):
                    arg_order_v = indexes_sorted_main[zahler]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = \
                        np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for zahler, _ in enumerate(args_contours_box_h):
                    arg_order_v = indexes_sorted_head[zahler]
                    order_by_con_head[args_contours_box_h[indexes_by_type_head[zahler]]] = \
                        np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for jji in range(len(id_of_texts)):
                    order_of_texts_tot.append(order_of_texts[jji] + ref_point)
                    id_of_texts_tot.append(id_of_texts[jji])
                ref_point += len(id_of_texts)

            order_of_texts_tot = []
            for tj1 in range(len(contours_only_text_parent)):
                order_of_texts_tot.append(int(order_by_con_main[tj1]))

            for tj1 in range(len(contours_only_text_parent_h)):
                order_of_texts_tot.append(int(order_by_con_head[tj1]))

            order_text_new = []
            for iii in range(len(order_of_texts_tot)):
                order_text_new.append(np.where(np.array(order_of_texts_tot) == iii)[0][0])

        except Exception as why:
            self.logger.error(why)
            arg_text_con = []
            for ii in range(len(cx_text_only)):
                check_if_textregion_located_in_a_box = False
                for jj in range(len(boxes)):
                    if (cx_text_only[ii] >= boxes[jj][0] and
                        cx_text_only[ii] < boxes[jj][1] and
                        cy_text_only[ii] >= boxes[jj][2] and
                        cy_text_only[ii] < boxes[jj][3]):
                        # this is valid if the center of region identify in which box it is located
                        arg_text_con.append(jj)
                        check_if_textregion_located_in_a_box = True
                        break

                if not check_if_textregion_located_in_a_box:
                    dists_tr_from_box = [math.sqrt((cx_text_only[ii] - boxes[jj][1]) ** 2 +
                                                   (cy_text_only[ii] - boxes[jj][2]) ** 2)
                                         for jj in range(len(boxes))]
                    ind_min = np.argmin(dists_tr_from_box)
                    arg_text_con.append(ind_min)
            args_contours = np.array(range(len(arg_text_con)))
            order_by_con_main = np.zeros(len(arg_text_con))

            ############################# head

            arg_text_con_h = []
            for ii in range(len(cx_text_only_h)):
                check_if_textregion_located_in_a_box = False
                for jj in range(len(boxes)):
                    if (cx_text_only_h[ii] >= boxes[jj][0] and
                        cx_text_only_h[ii] < boxes[jj][1] and
                        cy_text_only_h[ii] >= boxes[jj][2] and
                        cy_text_only_h[ii] < boxes[jj][3]):
                        # this is valid if the center of region identify in which box it is located
                        arg_text_con_h.append(jj)
                        check_if_textregion_located_in_a_box = True
                        break
                if not check_if_textregion_located_in_a_box:
                    dists_tr_from_box = [math.sqrt((cx_text_only_h[ii] - boxes[jj][1]) ** 2 +
                                                   (cy_text_only_h[ii] - boxes[jj][2]) ** 2)
                                         for jj in range(len(boxes))]
                    ind_min = np.argmin(dists_tr_from_box)
                    arg_text_con_h.append(ind_min)
            args_contours_h = np.array(range(len(arg_text_con_h)))
            order_by_con_head = np.zeros(len(arg_text_con_h))

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij, _ in enumerate(boxes):
                ys = slice(*boxes[iij][2:4])
                xs = slice(*boxes[iij][0:2])
                args_contours_box = args_contours[np.array(arg_text_con) == iij]
                args_contours_box_h = args_contours_h[np.array(arg_text_con_h) == iij]
                con_inter_box = []
                con_inter_box_h = []

                for box in args_contours_box:
                    con_inter_box.append(contours_only_text_parent[box])

                for box in args_contours_box_h:
                    con_inter_box_h.append(contours_only_text_parent_h[box])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(
                    textline_mask_tot[ys, xs], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(
                    con_inter_box, con_inter_box_h,
                    matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_sorted_head = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 2]
                indexes_by_type_head = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 2]

                for zahler, _ in enumerate(args_contours_box):
                    arg_order_v = indexes_sorted_main[zahler]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = \
                        np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for zahler, _ in enumerate(args_contours_box_h):
                    arg_order_v = indexes_sorted_head[zahler]
                    order_by_con_head[args_contours_box_h[indexes_by_type_head[zahler]]] = \
                        np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for jji, _ in enumerate(id_of_texts):
                    order_of_texts_tot.append(order_of_texts[jji] + ref_point)
                    id_of_texts_tot.append(id_of_texts[jji])
                ref_point += len(id_of_texts)

            order_of_texts_tot = []
            for tj1 in range(len(contours_only_text_parent)):
                order_of_texts_tot.append(int(order_by_con_main[tj1]))

            for tj1 in range(len(contours_only_text_parent_h)):
                order_of_texts_tot.append(int(order_by_con_head[tj1]))

            order_text_new = []
            for iii in range(len(order_of_texts_tot)):
                order_text_new.append(np.where(np.array(order_of_texts_tot) == iii)[0][0])

        self.logger.debug("exit do_order_of_regions_full_layout")
        return order_text_new, id_of_texts_tot

    def do_order_of_regions_no_full_layout(
            self, contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot):

        self.logger.debug("enter do_order_of_regions_no_full_layout")
        boxes = np.array(boxes, dtype=int) # to be on the safe side
        cx_text_only, cy_text_only, x_min_text_only, _, _, _, y_cor_x_min_main = find_new_features_of_contours(
            contours_only_text_parent)

        try:
            arg_text_con = []
            for ii in range(len(cx_text_only)):
                check_if_textregion_located_in_a_box = False
                for jj in range(len(boxes)):
                    if (x_min_text_only[ii] + 80 >= boxes[jj][0] and
                        x_min_text_only[ii] + 80 < boxes[jj][1] and
                        y_cor_x_min_main[ii] >= boxes[jj][2] and
                        y_cor_x_min_main[ii] < boxes[jj][3]):
                        arg_text_con.append(jj)
                        check_if_textregion_located_in_a_box = True
                        break
                if not check_if_textregion_located_in_a_box:
                    dists_tr_from_box = [math.sqrt((cx_text_only[ii] - boxes[jj][1]) ** 2 +
                                                   (cy_text_only[ii] - boxes[jj][2]) ** 2)
                                         for jj in range(len(boxes))]
                    ind_min = np.argmin(dists_tr_from_box)
                    arg_text_con.append(ind_min)
            args_contours = np.array(range(len(arg_text_con)))
            order_by_con_main = np.zeros(len(arg_text_con))

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij in range(len(boxes)):
                ys = slice(*boxes[iij][2:4])
                xs = slice(*boxes[iij][0:2])
                args_contours_box = args_contours[np.array(arg_text_con) == iij]
                con_inter_box = []
                con_inter_box_h = []
                for i in range(len(args_contours_box)):
                    con_inter_box.append(contours_only_text_parent[args_contours_box[i]])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(
                    textline_mask_tot[ys, xs], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(
                    con_inter_box, con_inter_box_h,
                    matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]

                for zahler, _ in enumerate(args_contours_box):
                    arg_order_v = indexes_sorted_main[zahler]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = \
                        np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for jji, _ in enumerate(id_of_texts):
                    order_of_texts_tot.append(order_of_texts[jji] + ref_point)
                    id_of_texts_tot.append(id_of_texts[jji])
                ref_point += len(id_of_texts)

            order_of_texts_tot = []
            for tj1 in range(len(contours_only_text_parent)):
                order_of_texts_tot.append(int(order_by_con_main[tj1]))

            order_text_new = []
            for iii in range(len(order_of_texts_tot)):
                order_text_new.append(np.where(np.array(order_of_texts_tot) == iii)[0][0])

        except Exception as why:
            self.logger.error(why)
            arg_text_con = []
            for ii in range(len(cx_text_only)):
                check_if_textregion_located_in_a_box = False
                for jj in range(len(boxes)):
                    if (cx_text_only[ii] >= boxes[jj][0] and
                        cx_text_only[ii] < boxes[jj][1] and
                        cy_text_only[ii] >= boxes[jj][2] and
                        cy_text_only[ii] < boxes[jj][3]):
                        # this is valid if the center of region identify in which box it is located
                        arg_text_con.append(jj)
                        check_if_textregion_located_in_a_box = True
                        break
                if not check_if_textregion_located_in_a_box:
                    dists_tr_from_box = [math.sqrt((cx_text_only[ii] - boxes[jj][1]) ** 2 +
                                                   (cy_text_only[ii] - boxes[jj][2]) ** 2)
                                         for jj in range(len(boxes))]
                    ind_min = np.argmin(dists_tr_from_box)
                    arg_text_con.append(ind_min)
            args_contours = np.array(range(len(arg_text_con)))
            order_by_con_main = np.zeros(len(arg_text_con))

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij in range(len(boxes)):
                ys = slice(*boxes[iij][2:4])
                xs = slice(*boxes[iij][0:2])
                args_contours_box = args_contours[np.array(arg_text_con) == iij]
                con_inter_box = []
                con_inter_box_h = []
                for i in range(len(args_contours_box)):
                    con_inter_box.append(contours_only_text_parent[args_contours_box[i]])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(
                    textline_mask_tot[ys, xs], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(
                    con_inter_box, con_inter_box_h,
                    matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]

                for zahler, _ in enumerate(args_contours_box):
                    arg_order_v = indexes_sorted_main[zahler]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = \
                        np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for jji, _ in enumerate(id_of_texts):
                    order_of_texts_tot.append(order_of_texts[jji] + ref_point)
                    id_of_texts_tot.append(id_of_texts[jji])
                ref_point += len(id_of_texts)

            order_of_texts_tot = []

            for tj1 in range(len(contours_only_text_parent)):
                order_of_texts_tot.append(int(order_by_con_main[tj1]))

            order_text_new = []
            for iii in range(len(order_of_texts_tot)):
                order_text_new.append(np.where(np.array(order_of_texts_tot) == iii)[0][0])

        self.logger.debug("exit do_order_of_regions_no_full_layout")
        return order_text_new, id_of_texts_tot

    def check_iou_of_bounding_box_and_contour_for_tables(
            self, layout, table_prediction_early, pixel_table, num_col_classifier):

        layout_org  = np.copy(layout)
        layout_org[:,:,0][layout_org[:,:,0]==pixel_table] = 0
        layout = (layout[:,:,0]==pixel_table)*1

        layout =np.repeat(layout[:, :, np.newaxis], 3, axis=2)
        layout = layout.astype(np.uint8)
        imgray = cv2.cvtColor(layout, cv2.COLOR_BGR2GRAY )
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_size = np.array([cv2.contourArea(contours[j])
                             for j in range(len(contours))])

        contours_new = []
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            iou = cnt_size[i] /float(w*h) *100
            if iou<80:
                layout_contour = np.zeros((layout_org.shape[0], layout_org.shape[1]))
                layout_contour= cv2.fillPoly(layout_contour,pts=[contours[i]] ,color=(1,1,1))

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

                layout_contour =np.repeat(layout_contour[:, :, np.newaxis], 3, axis=2)
                layout_contour = layout_contour.astype(np.uint8)

                imgray = cv2.cvtColor(layout_contour, cv2.COLOR_BGR2GRAY )
                _, thresh = cv2.threshold(imgray, 0, 255, 0)

                contours_sep, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for ji in range(len(contours_sep) ):
                    contours_new.append(contours_sep[ji])
                    if num_col_classifier>=2:
                        only_recent_contour_image = np.zeros((layout.shape[0],layout.shape[1]))
                        only_recent_contour_image= cv2.fillPoly(only_recent_contour_image, pts=[contours_sep[ji]], color=(1,1,1))
                        table_pixels_masked_from_early_pre = only_recent_contour_image * table_prediction_early
                        iou_in = 100. * table_pixels_masked_from_early_pre.sum() / only_recent_contour_image.sum()
                        #print(iou_in,'iou_in_in1')

                        if iou_in>30:
                            layout_org= cv2.fillPoly(layout_org, pts=[contours_sep[ji]], color=3 * (pixel_table,))
                        else:
                            pass
                    else:
                        layout_org= cv2.fillPoly(layout_org, pts=[contours_sep[ji]], color=3 * (pixel_table,))
            else:
                contours_new.append(contours[i])
                if num_col_classifier>=2:
                    only_recent_contour_image = np.zeros((layout.shape[0],layout.shape[1]))
                    only_recent_contour_image= cv2.fillPoly(only_recent_contour_image,pts=[contours[i]] ,color=(1,1,1))

                    table_pixels_masked_from_early_pre = only_recent_contour_image * table_prediction_early
                    iou_in = 100. * table_pixels_masked_from_early_pre.sum() / only_recent_contour_image.sum()
                    #print(iou_in,'iou_in')
                    if iou_in>30:
                        layout_org= cv2.fillPoly(layout_org, pts=[contours[i]], color=3 * (pixel_table,))
                    else:
                        pass
                else:
                    layout_org= cv2.fillPoly(layout_org, pts=[contours[i]], color=3 * (pixel_table,))

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

        img_comm_e = np.zeros(image_revised_1.shape)
        img_comm = np.repeat(img_comm_e[:, :, np.newaxis], 3, axis=2)

        for indiv in np.unique(image_revised_1):
            image_col=(image_revised_1==indiv)*255
            img_comm_in=np.repeat(image_col[:, :, np.newaxis], 3, axis=2)
            img_comm_in=img_comm_in.astype(np.uint8)

            imgray = cv2.cvtColor(img_comm_in, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 0, 255, 0)
            contours,hirarchy=cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            if indiv==pixel_table:
                main_contours = filter_contours_area_of_image_tables(thresh, contours, hirarchy, max_area = 1, min_area = 0.001)
            else:
                main_contours = filter_contours_area_of_image_tables(thresh, contours, hirarchy, max_area = 1, min_area = min_area)

            img_comm = cv2.fillPoly(img_comm, pts = main_contours, color = (indiv, indiv, indiv))
            img_comm = img_comm.astype(np.uint8)

        if not self.isNaN(slope_mean_hor):
            image_revised_last = np.zeros((image_regions_eraly_p.shape[0], image_regions_eraly_p.shape[1],3))
            for i in range(len(boxes)):
                box_ys = slice(*boxes[i][2:4])
                box_xs = slice(*boxes[i][0:2])
                image_box = img_comm[box_ys, box_xs]
                try:
                    image_box_tabels_1=(image_box[:,:,0]==pixel_table)*1
                    contours_tab,_=return_contours_of_image(image_box_tabels_1)
                    contours_tab=filter_contours_area_of_image_tables(image_box_tabels_1,contours_tab,_,1,0.003)
                    image_box_tabels_1=(image_box[:,:,0]==pixel_line)*1

                    image_box_tabels_and_m_text=( (image_box[:,:,0]==pixel_table) | (image_box[:,:,0]==1) )*1
                    image_box_tabels_and_m_text=image_box_tabels_and_m_text.astype(np.uint8)

                    image_box_tabels_1=image_box_tabels_1.astype(np.uint8)
                    image_box_tabels_1 = cv2.dilate(image_box_tabels_1,KERNEL,iterations = 5)

                    contours_table_m_text,_=return_contours_of_image(image_box_tabels_and_m_text)
                    image_box_tabels=np.repeat(image_box_tabels_1[:, :, np.newaxis], 3, axis=2)

                    image_box_tabels=image_box_tabels.astype(np.uint8)
                    imgray = cv2.cvtColor(image_box_tabels, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                    contours_line,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                    y_min_main_line ,y_max_main_line=find_features_of_contours(contours_line)
                    y_min_main_tab ,y_max_main_tab=find_features_of_contours(contours_tab)

                    cx_tab_m_text,cy_tab_m_text ,x_min_tab_m_text , x_max_tab_m_text, y_min_tab_m_text ,y_max_tab_m_text, _= find_new_features_of_contours(contours_table_m_text)
                    cx_tabl,cy_tabl ,x_min_tabl , x_max_tabl, y_min_tabl ,y_max_tabl,_= find_new_features_of_contours(contours_tab)

                    if len(y_min_main_tab )>0:
                        y_down_tabs=[]
                        y_up_tabs=[]

                        for i_t in range(len(y_min_main_tab )):
                            y_down_tab=[]
                            y_up_tab=[]
                            for i_l in range(len(y_min_main_line)):
                                if y_min_main_tab[i_t]>y_min_main_line[i_l] and  y_max_main_tab[i_t]>y_min_main_line[i_l] and y_min_main_tab[i_t]>y_max_main_line[i_l] and y_max_main_tab[i_t]>y_min_main_line[i_l]:
                                    pass
                                elif y_min_main_tab[i_t]<y_max_main_line[i_l] and y_max_main_tab[i_t]<y_max_main_line[i_l] and y_max_main_tab[i_t]<y_min_main_line[i_l] and y_min_main_tab[i_t]<y_min_main_line[i_l]:
                                    pass
                                elif np.abs(y_max_main_line[i_l]-y_min_main_line[i_l])<100:
                                    pass
                                else:
                                    y_up_tab.append(np.min([y_min_main_line[i_l], y_min_main_tab[i_t] ])  )
                                    y_down_tab.append( np.max([ y_max_main_line[i_l],y_max_main_tab[i_t] ]) )

                            if len(y_up_tab)==0:
                                y_up_tabs.append(y_min_main_tab[i_t])
                                y_down_tabs.append(y_max_main_tab[i_t])
                            else:
                                y_up_tabs.append(np.min(y_up_tab))
                                y_down_tabs.append(np.max(y_down_tab))
                    else:
                        y_down_tabs=[]
                        y_up_tabs=[]
                        pass
                except:
                    y_down_tabs=[]
                    y_up_tabs=[]

                for ii in range(len(y_up_tabs)):
                    image_box[y_up_tabs[ii]:y_down_tabs[ii],:,0]=pixel_table

                image_revised_last[box_ys, box_xs] = image_box
        else:
            for i in range(len(boxes)):
                box_ys = slice(*boxes[i][2:4])
                box_xs = slice(*boxes[i][0:2])
                image_box = img_comm[box_ys, box_xs]
                image_revised_last[box_ys, box_xs] = image_box

        if num_col_classifier==1:
            img_tables_col_1 = (image_revised_last[:,:,0] == pixel_table).astype(np.uint8)
            contours_table_col1, _ = return_contours_of_image(img_tables_col_1)

            _,_ ,_ , _, y_min_tab_col1 ,y_max_tab_col1, _= find_new_features_of_contours(contours_table_col1)

            if len(y_min_tab_col1)>0:
                for ijv in range(len(y_min_tab_col1)):
                    image_revised_last[int(y_min_tab_col1[ijv]):int(y_max_tab_col1[ijv]),:,:]=pixel_table
        return image_revised_last

    def do_order_of_regions(self, *args, **kwargs):
        if self.full_layout:
            return self.do_order_of_regions_full_layout(*args, **kwargs)
        return self.do_order_of_regions_no_full_layout(*args, **kwargs)

    def get_tables_from_model(self, img, num_col_classifier):
        img_org = np.copy(img)
        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]
        patches = False
        if self.light_version:
            prediction_table, _ = self.do_prediction_new_concept(patches, img, self.model_table)
            prediction_table = prediction_table.astype(np.int16)
            return prediction_table[:,:,0]
        else:
            if num_col_classifier < 4 and num_col_classifier > 2:
                prediction_table = self.do_prediction(patches, img, self.model_table)
                pre_updown = self.do_prediction(patches, cv2.flip(img[:,:,:], -1), self.model_table)
                pre_updown = cv2.flip(pre_updown, -1)

                prediction_table[:,:,0][pre_updown[:,:,0]==1]=1
                prediction_table = prediction_table.astype(np.int16)

            elif num_col_classifier ==2:
                height_ext = 0 # img.shape[0] // 4
                h_start = height_ext // 2
                width_ext = img.shape[1] // 8
                w_start = width_ext // 2

                img_new = np.zeros((img.shape[0] + height_ext,
                                    img.shape[1] + width_ext,
                                    img.shape[2])).astype(float)
                ys = slice(h_start, h_start + img.shape[0])
                xs = slice(w_start, w_start + img.shape[1])
                img_new[ys, xs] = img

                prediction_ext = self.do_prediction(patches, img_new, self.model_table)
                pre_updown = self.do_prediction(patches, cv2.flip(img_new[:,:,:], -1), self.model_table)
                pre_updown = cv2.flip(pre_updown, -1)

                prediction_table = prediction_ext[ys, xs]
                prediction_table_updown = pre_updown[ys, xs]

                prediction_table[:,:,0][prediction_table_updown[:,:,0]==1]=1
                prediction_table = prediction_table.astype(np.int16)
            elif num_col_classifier ==1:
                height_ext = 0 # img.shape[0] // 4
                h_start = height_ext // 2
                width_ext = img.shape[1] // 4
                w_start = width_ext // 2

                img_new =np.zeros((img.shape[0] + height_ext,
                                   img.shape[1] + width_ext,
                                   img.shape[2])).astype(float)
                ys = slice(h_start, h_start + img.shape[0])
                xs = slice(w_start, w_start + img.shape[1])
                img_new[ys, xs] = img

                prediction_ext = self.do_prediction(patches, img_new, self.model_table)
                pre_updown = self.do_prediction(patches, cv2.flip(img_new[:,:,:], -1), self.model_table)
                pre_updown = cv2.flip(pre_updown, -1)

                prediction_table = prediction_ext[ys, xs]
                prediction_table_updown = pre_updown[ys, xs]

                prediction_table[:,:,0][prediction_table_updown[:,:,0]==1]=1
                prediction_table = prediction_table.astype(np.int16)
            else:
                prediction_table = np.zeros(img.shape)
                img_w_half = img.shape[1] // 2

                pre1 = self.do_prediction(patches, img[:,0:img_w_half,:], self.model_table)
                pre2 = self.do_prediction(patches, img[:,img_w_half:,:], self.model_table)
                pre_full = self.do_prediction(patches, img[:,:,:], self.model_table)
                pre_updown = self.do_prediction(patches, cv2.flip(img[:,:,:], -1), self.model_table)
                pre_updown = cv2.flip(pre_updown, -1)

                prediction_table_full_erode = cv2.erode(pre_full[:,:,0], KERNEL, iterations=4)
                prediction_table_full_erode = cv2.dilate(prediction_table_full_erode, KERNEL, iterations=4)

                prediction_table_full_updown_erode = cv2.erode(pre_updown[:,:,0], KERNEL, iterations=4)
                prediction_table_full_updown_erode = cv2.dilate(prediction_table_full_updown_erode, KERNEL, iterations=4)

                prediction_table[:,0:img_w_half,:] = pre1[:,:,:]
                prediction_table[:,img_w_half:,:] = pre2[:,:,:]

                prediction_table[:,:,0][prediction_table_full_erode[:,:]==1]=1
                prediction_table[:,:,0][prediction_table_full_updown_erode[:,:]==1]=1
                prediction_table = prediction_table.astype(np.int16)

            #prediction_table_erode = cv2.erode(prediction_table[:,:,0], self.kernel, iterations=6)
            #prediction_table_erode = cv2.dilate(prediction_table_erode, self.kernel, iterations=6)

            prediction_table_erode = cv2.erode(prediction_table[:,:,0], KERNEL, iterations=20)
            prediction_table_erode = cv2.dilate(prediction_table_erode, KERNEL, iterations=20)
            return prediction_table_erode.astype(np.int16)

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

        text_regions_p_1 = text_regions_p_1[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        textline_mask_tot_ea = textline_mask_tot_ea[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        img_bin_light = img_bin_light[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]

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

    def run_graphics_and_columns(
            self, text_regions_p_1,
            num_col_classifier, num_column_is_classified, erosion_hurts):

        t_in_gr = time.time()
        img_g = self.imread(grayscale=True, uint8=True)

        img_g3 = np.zeros((img_g.shape[0], img_g.shape[1], 3))
        img_g3 = img_g3.astype(np.uint8)
        img_g3[:, :, 0] = img_g[:, :]
        img_g3[:, :, 1] = img_g[:, :]
        img_g3[:, :, 2] = img_g[:, :]

        image_page, page_coord, cont_page = self.extract_page()

        if self.tables:
            table_prediction = self.get_tables_from_model(image_page, num_col_classifier)
        else:
            table_prediction = np.zeros((image_page.shape[0], image_page.shape[1])).astype(np.int16)

        if self.plotter:
            self.plotter.save_page_image(image_page)

        text_regions_p_1 = text_regions_p_1[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        mask_images = (text_regions_p_1[:, :] == 2) * 1
        mask_images = mask_images.astype(np.uint8)
        mask_images = cv2.erode(mask_images[:, :], KERNEL, iterations=10)
        mask_lines = (text_regions_p_1[:, :] == 3) * 1
        mask_lines = mask_lines.astype(np.uint8)
        img_only_regions_with_sep = ((text_regions_p_1[:, :] != 3) & (text_regions_p_1[:, :] != 0)) * 1
        img_only_regions_with_sep = img_only_regions_with_sep.astype(np.uint8)

        if erosion_hurts:
            img_only_regions = np.copy(img_only_regions_with_sep[:,:])
        else:
            img_only_regions = cv2.erode(img_only_regions_with_sep[:,:], KERNEL, iterations=6)
        try:
            num_col, _ = find_num_col(img_only_regions, num_col_classifier, self.tables, multiplier=6.0)
            num_col = num_col + 1
            if not num_column_is_classified:
                num_col_classifier = num_col + 1
        except Exception as why:
            self.logger.error(why)
            num_col = None
        return (num_col, num_col_classifier, img_only_regions, page_coord, image_page, mask_images, mask_lines,
                text_regions_p_1, cont_page, table_prediction)

    def run_enhancement(self, light_version):
        t_in = time.time()
        self.logger.info("Resizing and enhancing image...")
        is_image_enhanced, img_org, img_res, num_col_classifier, num_column_is_classified, img_bin = \
            self.resize_and_enhance_image_with_column_classifier(light_version)
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
            if self.allow_enhancement:
                self.get_image_and_scales(img_org, img_res, scale)
            else:
                self.get_image_and_scales(img_org, img_res, scale)
            if self.allow_scaling:
                img_org, img_res, is_image_enhanced = self.resize_image_with_column_classifier(is_image_enhanced, img_bin)
                self.get_image_and_scales_after_enhancing(img_org, img_res)
        #print("enhancement in ", time.time()-t_in)
        return img_res, is_image_enhanced, num_col_classifier, num_column_is_classified

    def run_textline(self, image_page, num_col_classifier=None):
        scaler_h_textline = 1#1.3  # 1.2#1.2
        scaler_w_textline = 1#1.3  # 0.9#1
        #print(image_page.shape)
        textline_mask_tot_ea, _ = self.textline_contours(image_page, True, scaler_h_textline, scaler_w_textline, num_col_classifier)
        if self.textline_light:
            textline_mask_tot_ea = textline_mask_tot_ea.astype(np.int16)

        if self.plotter:
            self.plotter.save_plot_of_textlines(textline_mask_tot_ea, image_page)
        return textline_mask_tot_ea

    def run_deskew(self, textline_mask_tot_ea):
        #print(textline_mask_tot_ea.shape, 'textline_mask_tot_ea deskew')
        slope_deskew = return_deskew_slop_old_mp(cv2.erode(textline_mask_tot_ea, KERNEL, iterations=2), 2, 30, True,
                                          logger=self.logger, plotter=self.plotter)
        slope_first = 0

        if self.plotter:
            self.plotter.save_deskewed_image(slope_deskew)
        self.logger.info("slope_deskew: %.2f°", slope_deskew)
        return slope_deskew, slope_first

    def run_marginals(
            self, image_page, textline_mask_tot_ea, mask_images, mask_lines,
            num_col_classifier, slope_deskew, text_regions_p_1, table_prediction):

        image_page_rotated, textline_mask_tot = image_page[:, :], textline_mask_tot_ea[:, :]
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
                    num_col_classifier, slope_deskew, light_version=self.light_version, kernel=KERNEL)
            except Exception as e:
                self.logger.error("exception %s", e)

        if self.plotter:
            self.plotter.save_plot_of_layout_main_all(text_regions_p, image_page)
            self.plotter.save_plot_of_layout_main(text_regions_p, image_page)
        return textline_mask_tot, text_regions_p, image_page_rotated

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
        regions_without_separators = (text_regions_p[:, :] == 1) * 1  # ( (text_regions_p[:,:]==1) | (text_regions_p[:,:]==2) )*1 #self.return_regions_without_separators_new(text_regions_p[:,:,0],img_only_regions)
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
                np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2),
                num_col_classifier, self.tables, pixel_lines)

        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            _, _, matrix_of_lines_ch_d, splitter_y_new_d, _ = find_number_of_columns_in_document(
                np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2),
                num_col_classifier, self.tables, pixel_lines)
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

            if self.tables:
                if self.light_version:
                    pass
                else:
                    text_regions_p_tables = np.copy(text_regions_p)
                    text_regions_p_tables[:,:][(table_prediction[:,:] == 1)] = 10
                    pixel_line = 3
                    img_revised_tab2 = self.add_tables_heuristic_to_layout(
                        text_regions_p_tables, boxes, 0, splitter_y_new, peaks_neg_tot_tables, text_regions_p_tables,
                        num_col_classifier , 0.000005, pixel_line)
                    #print(time.time()-t_0_box,'time box in 3.2')
                    img_revised_tab2, contoures_tables = self.check_iou_of_bounding_box_and_contour_for_tables(
                        img_revised_tab2, table_prediction, 10, num_col_classifier)
                    #print(time.time()-t_0_box,'time box in 3.3')
        else:
            boxes_d, peaks_neg_tot_tables_d = return_boxes_of_images_by_order_of_reading_new(
                splitter_y_new_d, regions_without_separators_d, matrix_of_lines_ch_d,
                num_col_classifier, erosion_hurts, self.tables, self.right2left)
            boxes = None
            self.logger.debug("len(boxes): %s", len(boxes_d))

            if self.tables:
                if self.light_version:
                    pass
                else:
                    text_regions_p_tables = np.copy(text_regions_p_1_n)
                    text_regions_p_tables =np.round(text_regions_p_tables)
                    text_regions_p_tables[:,:][(text_regions_p_tables[:,:] != 3) & (table_prediction_n[:,:] == 1)] = 10

                    pixel_line = 3
                    img_revised_tab2 = self.add_tables_heuristic_to_layout(
                        text_regions_p_tables, boxes_d, 0, splitter_y_new_d, peaks_neg_tot_tables_d, text_regions_p_tables,
                        num_col_classifier, 0.000005, pixel_line)
                    img_revised_tab2_d,_ = self.check_iou_of_bounding_box_and_contour_for_tables(
                        img_revised_tab2, table_prediction_n, 10, num_col_classifier)

                    img_revised_tab2_d_rotated = rotate_image(img_revised_tab2_d, -slope_deskew)
                    img_revised_tab2_d_rotated = np.round(img_revised_tab2_d_rotated)
                    img_revised_tab2_d_rotated = img_revised_tab2_d_rotated.astype(np.int8)
                    img_revised_tab2_d_rotated = resize_image(img_revised_tab2_d_rotated, text_regions_p.shape[0], text_regions_p.shape[1])
        #print(time.time()-t_0_box,'time box in 4')
        self.logger.info("detecting boxes took %.1fs", time.time() - t1)

        if self.tables:
            if self.light_version:
                text_regions_p[:,:][table_prediction[:,:]==1] = 10
                img_revised_tab=text_regions_p[:,:]
            else:
                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                    img_revised_tab = np.copy(img_revised_tab2[:,:,0])
                    img_revised_tab[:,:][(text_regions_p[:,:] == 1) & (img_revised_tab[:,:] != 10)] = 1
                else:
                    img_revised_tab = np.copy(text_regions_p[:,:])
                    img_revised_tab[:,:][img_revised_tab[:,:] == 10] = 0
                    img_revised_tab[:,:][img_revised_tab2_d_rotated[:,:,0] == 10] = 10

                text_regions_p[:,:][text_regions_p[:,:]==10] = 0
                text_regions_p[:,:][img_revised_tab[:,:]==10] = 10
        else:
            img_revised_tab=text_regions_p[:,:]
        #img_revised_tab = text_regions_p[:, :]
        if self.light_version:
            polygons_of_images = return_contours_of_interested_region(text_regions_p, 2)
        else:
            polygons_of_images = return_contours_of_interested_region(img_revised_tab, 2)

        pixel_img = 4
        min_area_mar = 0.00001
        if self.light_version:
            marginal_mask = (text_regions_p[:,:]==pixel_img)*1
            marginal_mask = marginal_mask.astype('uint8')
            marginal_mask = cv2.dilate(marginal_mask, KERNEL, iterations=2)

            polygons_of_marginals = return_contours_of_interested_region(marginal_mask, 1, min_area_mar)
        else:
            polygons_of_marginals = return_contours_of_interested_region(text_regions_p, pixel_img, min_area_mar)

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
            if self.light_version:
                text_regions_p[:,:][table_prediction[:,:]==1] = 10
                img_revised_tab = text_regions_p[:,:]
                if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                    image_page_rotated_n, textline_mask_tot_d, text_regions_p_1_n, table_prediction_n = \
                        rotation_not_90_func(image_page, textline_mask_tot, text_regions_p, table_prediction, slope_deskew)

                    text_regions_p_1_n = resize_image(text_regions_p_1_n,text_regions_p.shape[0],text_regions_p.shape[1])
                    textline_mask_tot_d = resize_image(textline_mask_tot_d,text_regions_p.shape[0],text_regions_p.shape[1])
                    table_prediction_n = resize_image(table_prediction_n,text_regions_p.shape[0],text_regions_p.shape[1])

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

            else:
                if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                    image_page_rotated_n, textline_mask_tot_d, text_regions_p_1_n, table_prediction_n = \
                        rotation_not_90_func(image_page, textline_mask_tot, text_regions_p, table_prediction, slope_deskew)

                    text_regions_p_1_n = resize_image(text_regions_p_1_n,text_regions_p.shape[0],text_regions_p.shape[1])
                    textline_mask_tot_d = resize_image(textline_mask_tot_d,text_regions_p.shape[0],text_regions_p.shape[1])
                    table_prediction_n = resize_image(table_prediction_n,text_regions_p.shape[0],text_regions_p.shape[1])

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

                pixel_lines=3
                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                    num_col, _, matrix_of_lines_ch, splitter_y_new, _ = find_number_of_columns_in_document(
                        np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2),
                        num_col_classifier, self.tables, pixel_lines)

                if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                    num_col_d, _, matrix_of_lines_ch_d, splitter_y_new_d, _ = find_number_of_columns_in_document(
                        np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2),
                        num_col_classifier, self.tables, pixel_lines)

                if num_col_classifier>=3:
                    if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                        regions_without_separators = regions_without_separators.astype(np.uint8)
                        regions_without_separators = cv2.erode(regions_without_separators[:,:], KERNEL, iterations=6)

                    if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                        regions_without_separators_d = regions_without_separators_d.astype(np.uint8)
                        regions_without_separators_d = cv2.erode(regions_without_separators_d[:,:], KERNEL, iterations=6)
                else:
                    pass

                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                    boxes, peaks_neg_tot_tables = return_boxes_of_images_by_order_of_reading_new(
                        splitter_y_new, regions_without_separators, matrix_of_lines_ch,
                        num_col_classifier, erosion_hurts, self.tables, self.right2left)
                    text_regions_p_tables = np.copy(text_regions_p)
                    text_regions_p_tables[:,:][(table_prediction[:,:]==1)] = 10
                    pixel_line = 3
                    img_revised_tab2 = self.add_tables_heuristic_to_layout(
                        text_regions_p_tables, boxes, 0, splitter_y_new, peaks_neg_tot_tables, text_regions_p_tables,
                        num_col_classifier , 0.000005, pixel_line)

                    img_revised_tab2,contoures_tables = self.check_iou_of_bounding_box_and_contour_for_tables(
                        img_revised_tab2, table_prediction, 10, num_col_classifier)
                else:
                    boxes_d, peaks_neg_tot_tables_d = return_boxes_of_images_by_order_of_reading_new(
                        splitter_y_new_d, regions_without_separators_d, matrix_of_lines_ch_d,
                        num_col_classifier, erosion_hurts, self.tables, self.right2left)
                    text_regions_p_tables = np.copy(text_regions_p_1_n)
                    text_regions_p_tables = np.round(text_regions_p_tables)
                    text_regions_p_tables[:,:][(text_regions_p_tables[:,:]!=3) & (table_prediction_n[:,:]==1)] = 10

                    pixel_line = 3
                    img_revised_tab2 = self.add_tables_heuristic_to_layout(
                        text_regions_p_tables, boxes_d, 0, splitter_y_new_d, peaks_neg_tot_tables_d, text_regions_p_tables,
                        num_col_classifier, 0.000005, pixel_line)

                    img_revised_tab2_d,_ = self.check_iou_of_bounding_box_and_contour_for_tables(
                        img_revised_tab2, table_prediction_n, 10, num_col_classifier)
                    img_revised_tab2_d_rotated = rotate_image(img_revised_tab2_d, -slope_deskew)

                    img_revised_tab2_d_rotated = np.round(img_revised_tab2_d_rotated)
                    img_revised_tab2_d_rotated = img_revised_tab2_d_rotated.astype(np.int8)

                    img_revised_tab2_d_rotated = resize_image(img_revised_tab2_d_rotated, text_regions_p.shape[0], text_regions_p.shape[1])

                if np.abs(slope_deskew) < 0.13:
                    img_revised_tab = np.copy(img_revised_tab2[:,:,0])
                else:
                    img_revised_tab = np.copy(text_regions_p[:,:])
                    img_revised_tab[:,:][img_revised_tab[:,:] == 10] = 0
                    img_revised_tab[:,:][img_revised_tab2_d_rotated[:,:,0] == 10] = 10

                ##img_revised_tab=img_revised_tab2[:,:,0]
                #img_revised_tab=text_regions_p[:,:]
                text_regions_p[:,:][text_regions_p[:,:]==10] = 0
                text_regions_p[:,:][img_revised_tab[:,:]==10] = 10
                #img_revised_tab[img_revised_tab2[:,:,0]==10] =10

        pixel_img = 4
        min_area_mar = 0.00001

        if self.light_version:
            marginal_mask = (text_regions_p[:,:]==pixel_img)*1
            marginal_mask = marginal_mask.astype('uint8')
            marginal_mask = cv2.dilate(marginal_mask, KERNEL, iterations=2)

            polygons_of_marginals = return_contours_of_interested_region(marginal_mask, 1, min_area_mar)
        else:
            polygons_of_marginals = return_contours_of_interested_region(text_regions_p, pixel_img, min_area_mar)

        pixel_img = 10
        contours_tables = return_contours_of_interested_region(text_regions_p, pixel_img, min_area_mar)

        # set first model with second model
        text_regions_p[:, :][text_regions_p[:, :] == 2] = 5
        text_regions_p[:, :][text_regions_p[:, :] == 3] = 6
        text_regions_p[:, :][text_regions_p[:, :] == 4] = 8

        image_page = image_page.astype(np.uint8)
        #print("full inside 1", time.time()- t_full0)
        regions_fully, regions_fully_only_drop = self.extract_text_regions_new(
            img_bin_light if self.light_version else image_page,
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
        ##regions_fully[:, :, 0][regions_fully_only_drop[:, :, 0] == 4] = 4
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

        ###regions_fully = boosting_headers_by_longshot_region_segmentation(regions_fully, regions_fully_np, img_only_regions)
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

    @staticmethod
    def our_load_model(model_file):
        if model_file.endswith('.h5') and Path(model_file[:-3]).exists():
            # prefer SavedModel over HDF5 format if it exists
            model_file = model_file[:-3]
        try:
            model = load_model(model_file, compile=False)
        except:
            model = load_model(model_file, compile=False, custom_objects={
                "PatchEncoder": PatchEncoder, "Patches": Patches})
        return model

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
        if len(contours_only_text_parent)>min_cont_size_to_be_dilated and self.light_version:
            cx_conts, cy_conts, x_min_conts, x_max_conts, y_min_conts, y_max_conts, _ = find_new_features_of_contours(contours_only_text_parent)
            args_cont_located = np.array(range(len(contours_only_text_parent)))
            
            diff_y_conts = np.abs(y_max_conts[:]-y_min_conts)
            diff_x_conts = np.abs(x_max_conts[:]-x_min_conts)
            
            mean_x = statistics.mean(diff_x_conts)
            median_x = statistics.median(diff_x_conts)
            
            
            diff_x_ratio= diff_x_conts/mean_x
            
            args_cont_located_excluded = args_cont_located[diff_x_ratio>=1.3]
            args_cont_located_included = args_cont_located[diff_x_ratio<1.3]
            
            contours_only_text_parent_excluded = [contours_only_text_parent[ind] for ind in range(len(contours_only_text_parent)) if diff_x_ratio[ind]>=1.3]#contours_only_text_parent[diff_x_ratio>=1.3]
            contours_only_text_parent_included = [contours_only_text_parent[ind] for ind in range(len(contours_only_text_parent)) if diff_x_ratio[ind]<1.3]#contours_only_text_parent[diff_x_ratio<1.3]
            
            
            cx_conts_excluded = [cx_conts[ind] for ind in range(len(cx_conts)) if diff_x_ratio[ind]>=1.3]#cx_conts[diff_x_ratio>=1.3]
            cx_conts_included = [cx_conts[ind] for ind in range(len(cx_conts)) if diff_x_ratio[ind]<1.3]#cx_conts[diff_x_ratio<1.3]
            
            cy_conts_excluded = [cy_conts[ind] for ind in range(len(cy_conts)) if diff_x_ratio[ind]>=1.3]#cy_conts[diff_x_ratio>=1.3]
            cy_conts_included = [cy_conts[ind] for ind in range(len(cy_conts)) if diff_x_ratio[ind]<1.3]#cy_conts[diff_x_ratio<1.3]
            
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
            
            indexes_of_located_cont, center_x_coordinates_of_located, center_y_coordinates_of_located = self.return_indexes_of_contours_loctaed_inside_another_list_of_contours(contours_only_dilated, contours_only_text_parent_included, cx_conts_included, cy_conts_included, args_cont_located_included)
            
            
            if len(args_cont_located_excluded)>0:
                for ind in args_cont_located_excluded:
                    indexes_of_located_cont.append(np.array([ind]))
                    contours_only_dilated.append(contours_only_text_parent[ind])
                    center_y_coordinates_of_located.append(0)
            
            array_list = [np.array([elem]) if isinstance(elem, int) else elem for elem in indexes_of_located_cont]
            flattened_array = np.concatenate([arr.ravel() for arr in array_list])
            #print(len( np.unique(flattened_array)), 'indexes_of_located_cont uniques')
            
            missing_textregions = list( set(np.array(range(len(contours_only_text_parent))) ) - set(np.unique(flattened_array)) )
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
            if len(contours_only_text_parent)>min_cont_size_to_be_dilated and self.light_version:
                co_text_all = contours_only_dilated + contours_only_text_parent_h
            else:
                co_text_all = contours_only_text_parent + contours_only_text_parent_h
        else:
            co_text_all_org = contours_only_text_parent
            if len(contours_only_text_parent)>min_cont_size_to_be_dilated and self.light_version:
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
                    y_pr = self.model_reading_order.predict(input_1 , verbose=0)
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
        
        if len(contours_only_text_parent)>min_cont_size_to_be_dilated and self.light_version:
            org_contours_indexes = []
            for ind in range(len(ordered)):
                region_with_curr_order = ordered[ind]
                if region_with_curr_order < len(contours_only_dilated):
                    if np.isscalar(indexes_of_located_cont[region_with_curr_order]):
                        org_contours_indexes = org_contours_indexes + [indexes_of_located_cont[region_with_curr_order]]
                    else:
                        arg_sort_located_cont = np.argsort(center_y_coordinates_of_located[region_with_curr_order])
                        org_contours_indexes = org_contours_indexes + list(np.array(indexes_of_located_cont[region_with_curr_order])[arg_sort_located_cont]) ##org_contours_indexes + list ( 
                else:
                    org_contours_indexes = org_contours_indexes + [indexes_of_located_cont[region_with_curr_order]]
            
            region_ids = ['region_%04d' % i for i in range(len(co_text_all_org))]
            return org_contours_indexes, region_ids
        else:
            region_ids = ['region_%04d' % i for i in range(len(co_text_all_org))]
            return ordered, region_ids

    def return_list_of_contours_with_desired_order(self, ls_cons, sorted_indexes):
        return [ls_cons[sorted_indexes[index]] for index in range(len(sorted_indexes))]

    def return_it_in_two_groups(self, x_differential):
        split = [ind if x_differential[ind]!=x_differential[ind+1] else -1
                 for ind in range(len(x_differential)-1)]
        split_masked = list( np.array(split[:])[np.array(split[:])!=-1] )
        if 0 not in split_masked:
            split_masked.insert(0, -1)
        split_masked.append(len(x_differential)-1)

        split_masked = np.array(split_masked) +1

        sums = [np.sum(x_differential[split_masked[ind]:split_masked[ind+1]])
                for ind in range(len(split_masked)-1)]

        indexes_to_bec_changed = [ind if (np.abs(sums[ind-1]) > np.abs(sums[ind]) and
                                          np.abs(sums[ind+1]) > np.abs(sums[ind])) else -1
                                  for ind in range(1,len(sums)-1)]
        indexes_to_bec_changed_filtered = np.array(indexes_to_bec_changed)[np.array(indexes_to_bec_changed)!=-1]

        x_differential_new = np.copy(x_differential)
        for i in indexes_to_bec_changed_filtered:
            i_slice = slice(split_masked[i], split_masked[i+1])
            x_differential_new[i_slice] = -1 * np.array(x_differential)[i_slice]

        return x_differential_new

    def dilate_textregions_contours_textline_version(self, all_found_textline_polygons):
        #print(all_found_textline_polygons)
        for j in range(len(all_found_textline_polygons)):
            for ij in range(len(all_found_textline_polygons[j])):
                con_ind = all_found_textline_polygons[j][ij]
                area = cv2.contourArea(con_ind)
                con_ind = con_ind.astype(float)

                x_differential = np.diff( con_ind[:,0,0])
                y_differential = np.diff( con_ind[:,0,1])

                x_differential = gaussian_filter1d(x_differential, 0.1)
                y_differential = gaussian_filter1d(y_differential, 0.1)

                x_min = float(np.min( con_ind[:,0,0] ))
                y_min = float(np.min( con_ind[:,0,1] ))

                x_max = float(np.max( con_ind[:,0,0] ))
                y_max = float(np.max( con_ind[:,0,1] ))

                x_differential_mask_nonzeros = [ ind/abs(ind) if ind!=0 else ind for ind in x_differential]
                y_differential_mask_nonzeros = [ ind/abs(ind) if ind!=0 else ind for ind in y_differential]

                abs_diff=abs(abs(x_differential)- abs(y_differential) )

                inc_x = np.zeros(len(x_differential)+1)
                inc_y = np.zeros(len(x_differential)+1)

                if (y_max-y_min) <= (x_max-x_min):
                    dilation_m1 = round(area / (x_max-x_min) * 0.12)
                else:
                    dilation_m1 = round(area / (y_max-y_min) * 0.12)

                if dilation_m1>8:
                    dilation_m1 = 8
                if dilation_m1<6:
                    dilation_m1 = 6
                #print(dilation_m1, 'dilation_m1')
                dilation_m1 = 6
                dilation_m2 = int(dilation_m1/2.) +1 

                for i in range(len(x_differential)):
                    if abs_diff[i]==0:
                        inc_x[i+1] = dilation_m2*(-1*y_differential_mask_nonzeros[i])
                        inc_y[i+1] = dilation_m2*(x_differential_mask_nonzeros[i])
                    elif abs_diff[i]!=0 and x_differential_mask_nonzeros[i]==0 and y_differential_mask_nonzeros[i]!=0:
                        inc_x[i+1]= dilation_m1*(-1*y_differential_mask_nonzeros[i])
                    elif abs_diff[i]!=0 and x_differential_mask_nonzeros[i]!=0 and y_differential_mask_nonzeros[i]==0:
                        inc_y[i+1] = dilation_m1*(x_differential_mask_nonzeros[i])

                    elif abs_diff[i]!=0 and abs_diff[i]>=3:
                        if abs(x_differential[i])>abs(y_differential[i]):
                            inc_y[i+1] = dilation_m1*(x_differential_mask_nonzeros[i])
                        else:
                            inc_x[i+1]= dilation_m1*(-1*y_differential_mask_nonzeros[i])
                    else:
                        inc_x[i+1] = dilation_m2*(-1*y_differential_mask_nonzeros[i])
                        inc_y[i+1] = dilation_m2*(x_differential_mask_nonzeros[i])

                inc_x[0] = inc_x[-1]
                inc_y[0] = inc_y[-1]

                con_scaled = con_ind*1

                con_scaled[:,0, 0] = con_ind[:,0,0] + np.array(inc_x)[:]
                con_scaled[:,0, 1] = con_ind[:,0,1] + np.array(inc_y)[:]

                con_scaled[:,0, 1][con_scaled[:,0, 1]<0] = 0
                con_scaled[:,0, 0][con_scaled[:,0, 0]<0] = 0

                area_scaled = cv2.contourArea(con_scaled.astype(np.int32))

                con_ind = con_ind.astype(np.int32)

                results = [cv2.pointPolygonTest(con_ind, (con_scaled[ind,0, 0], con_scaled[ind,0, 1]), False)
                           for ind in range(len(con_scaled[:,0, 1])) ]
                results = np.array(results)
                #print(results,'results')
                results[results==0] = 1

                diff_result = np.diff(results)

                indices_2 = [ind for ind in range(len(diff_result)) if diff_result[ind]==2]
                indices_m2 = [ind for ind in range(len(diff_result)) if diff_result[ind]==-2]

                if results[0]==1:
                    con_scaled[:indices_m2[0]+1,0, 1] = con_ind[:indices_m2[0]+1,0,1]
                    con_scaled[:indices_m2[0]+1,0, 0] = con_ind[:indices_m2[0]+1,0,0]
                    #indices_2 = indices_2[1:]
                    indices_m2 = indices_m2[1:]

                if len(indices_2)>len(indices_m2):
                    con_scaled[indices_2[-1]+1:,0, 1] = con_ind[indices_2[-1]+1:,0,1]
                    con_scaled[indices_2[-1]+1:,0, 0] = con_ind[indices_2[-1]+1:,0,0]
                    indices_2 = indices_2[:-1]

                for ii in range(len(indices_2)):
                    con_scaled[indices_2[ii]+1:indices_m2[ii]+1,0, 1] = con_scaled[indices_2[ii],0, 1]
                    con_scaled[indices_2[ii]+1:indices_m2[ii]+1,0, 0] = con_scaled[indices_2[ii],0, 0]

                all_found_textline_polygons[j][ij][:,0,1] = con_scaled[:,0, 1]
                all_found_textline_polygons[j][ij][:,0,0] = con_scaled[:,0, 0]
        return all_found_textline_polygons

    def dilate_textregions_contours(self, all_found_textline_polygons):
        #print(all_found_textline_polygons)
        for j in range(len(all_found_textline_polygons)):
            con_ind = all_found_textline_polygons[j]
            #print(len(con_ind[:,0,0]),'con_ind[:,0,0]')
            area = cv2.contourArea(con_ind)
            con_ind = con_ind.astype(float)

            x_differential = np.diff( con_ind[:,0,0])
            y_differential = np.diff( con_ind[:,0,1])

            x_differential = gaussian_filter1d(x_differential, 0.1)
            y_differential = gaussian_filter1d(y_differential, 0.1)

            x_min = float(np.min( con_ind[:,0,0] ))
            y_min = float(np.min( con_ind[:,0,1] ))

            x_max = float(np.max( con_ind[:,0,0] ))
            y_max = float(np.max( con_ind[:,0,1] ))

            x_differential_mask_nonzeros = [ ind/abs(ind) if ind!=0 else ind for ind in x_differential]
            y_differential_mask_nonzeros = [ ind/abs(ind) if ind!=0 else ind for ind in y_differential]

            abs_diff=abs(abs(x_differential)- abs(y_differential) )

            inc_x = np.zeros(len(x_differential)+1)
            inc_y = np.zeros(len(x_differential)+1)

            if (y_max-y_min) <= (x_max-x_min):
                dilation_m1 = round(area / (x_max-x_min) * 0.12)
            else:
                dilation_m1 = round(area / (y_max-y_min) * 0.12)

            if dilation_m1>8:
                dilation_m1 = 8
            if dilation_m1<6:
                dilation_m1 = 6
            #print(dilation_m1, 'dilation_m1')
            dilation_m1 = 4#6
            dilation_m2 = int(dilation_m1/2.) +1 

            for i in range(len(x_differential)):
                if abs_diff[i]==0:
                    inc_x[i+1] = dilation_m2*(-1*y_differential_mask_nonzeros[i])
                    inc_y[i+1] = dilation_m2*(x_differential_mask_nonzeros[i])
                elif abs_diff[i]!=0 and x_differential_mask_nonzeros[i]==0 and y_differential_mask_nonzeros[i]!=0:
                    inc_x[i+1]= dilation_m1*(-1*y_differential_mask_nonzeros[i])
                elif abs_diff[i]!=0 and x_differential_mask_nonzeros[i]!=0 and y_differential_mask_nonzeros[i]==0:
                    inc_y[i+1] = dilation_m1*(x_differential_mask_nonzeros[i])

                elif abs_diff[i]!=0 and abs_diff[i]>=3:
                    if abs(x_differential[i])>abs(y_differential[i]):
                        inc_y[i+1] = dilation_m1*(x_differential_mask_nonzeros[i])
                    else:
                        inc_x[i+1]= dilation_m1*(-1*y_differential_mask_nonzeros[i])
                else:
                    inc_x[i+1] = dilation_m2*(-1*y_differential_mask_nonzeros[i])
                    inc_y[i+1] = dilation_m2*(x_differential_mask_nonzeros[i])

            inc_x[0] = inc_x[-1]
            inc_y[0] = inc_y[-1]

            con_scaled = con_ind*1

            con_scaled[:,0, 0] = con_ind[:,0,0] + np.array(inc_x)[:]
            con_scaled[:,0, 1] = con_ind[:,0,1] + np.array(inc_y)[:]

            con_scaled[:,0, 1][con_scaled[:,0, 1]<0] = 0
            con_scaled[:,0, 0][con_scaled[:,0, 0]<0] = 0

            area_scaled = cv2.contourArea(con_scaled.astype(np.int32))

            con_ind = con_ind.astype(np.int32)

            results = [cv2.pointPolygonTest(con_ind, (con_scaled[ind,0, 0], con_scaled[ind,0, 1]), False)
                       for ind in range(len(con_scaled[:,0, 1])) ]
            results = np.array(results)
            #print(results,'results')
            results[results==0] = 1

            diff_result = np.diff(results)
            indices_2 = [ind for ind in range(len(diff_result)) if diff_result[ind]==2]
            indices_m2 = [ind for ind in range(len(diff_result)) if diff_result[ind]==-2]

            if results[0]==1:
                con_scaled[:indices_m2[0]+1,0, 1] = con_ind[:indices_m2[0]+1,0,1]
                con_scaled[:indices_m2[0]+1,0, 0] = con_ind[:indices_m2[0]+1,0,0]
                #indices_2 = indices_2[1:]
                indices_m2 = indices_m2[1:]

            if len(indices_2)>len(indices_m2):
                con_scaled[indices_2[-1]+1:,0, 1] = con_ind[indices_2[-1]+1:,0,1]
                con_scaled[indices_2[-1]+1:,0, 0] = con_ind[indices_2[-1]+1:,0,0]
                indices_2 = indices_2[:-1]

            for ii in range(len(indices_2)):
                con_scaled[indices_2[ii]+1:indices_m2[ii]+1,0, 1] = con_scaled[indices_2[ii],0, 1]
                con_scaled[indices_2[ii]+1:indices_m2[ii]+1,0, 0] = con_scaled[indices_2[ii],0, 0]

            all_found_textline_polygons[j][:,0,1] = con_scaled[:,0, 1]
            all_found_textline_polygons[j][:,0,0] = con_scaled[:,0, 0]
        return all_found_textline_polygons

    def dilate_textline_contours(self, all_found_textline_polygons):
        for j in range(len(all_found_textline_polygons)):
            for ij in range(len(all_found_textline_polygons[j])):
                con_ind = all_found_textline_polygons[j][ij]
                area = cv2.contourArea(con_ind)

                con_ind = con_ind.astype(float)

                x_differential = np.diff( con_ind[:,0,0])
                y_differential = np.diff( con_ind[:,0,1])

                x_differential = gaussian_filter1d(x_differential, 3)
                y_differential = gaussian_filter1d(y_differential, 3)

                x_min = float(np.min( con_ind[:,0,0] ))
                y_min = float(np.min( con_ind[:,0,1] ))

                x_max = float(np.max( con_ind[:,0,0] ))
                y_max = float(np.max( con_ind[:,0,1] ))

                x_differential_mask_nonzeros = [ ind/abs(ind) if ind!=0 else ind for ind in x_differential]
                y_differential_mask_nonzeros = [ ind/abs(ind) if ind!=0 else ind for ind in y_differential]

                abs_diff=abs(abs(x_differential)- abs(y_differential) )

                inc_x = np.zeros(len(x_differential)+1)
                inc_y = np.zeros(len(x_differential)+1)

                if (y_max-y_min) <= (x_max-x_min):
                    dilation_m1 = round(area / (x_max-x_min) * 0.35)
                else:
                    dilation_m1 = round(area / (y_max-y_min) * 0.35)

                if dilation_m1>12:
                    dilation_m1 = 12
                if dilation_m1<4:
                    dilation_m1 = 4
                #print(dilation_m1, 'dilation_m1')
                dilation_m2 = int(dilation_m1/2.) +1

                for i in range(len(x_differential)):
                    if abs_diff[i]==0:
                        inc_x[i+1] = dilation_m2*(-1*y_differential_mask_nonzeros[i])
                        inc_y[i+1] = dilation_m2*(x_differential_mask_nonzeros[i])
                    elif abs_diff[i]!=0 and x_differential_mask_nonzeros[i]==0 and y_differential_mask_nonzeros[i]!=0:
                        inc_x[i+1]= dilation_m1*(-1*y_differential_mask_nonzeros[i])
                    elif abs_diff[i]!=0 and x_differential_mask_nonzeros[i]!=0 and y_differential_mask_nonzeros[i]==0:
                        inc_y[i+1] = dilation_m1*(x_differential_mask_nonzeros[i])

                    elif abs_diff[i]!=0 and abs_diff[i]>=3:
                        if abs(x_differential[i])>abs(y_differential[i]):
                            inc_y[i+1] = dilation_m1*(x_differential_mask_nonzeros[i])
                        else:
                            inc_x[i+1]= dilation_m1*(-1*y_differential_mask_nonzeros[i])
                    else:
                        inc_x[i+1] = dilation_m2*(-1*y_differential_mask_nonzeros[i])
                        inc_y[i+1] = dilation_m2*(x_differential_mask_nonzeros[i])

                inc_x[0] = inc_x[-1]
                inc_y[0] = inc_y[-1]

                con_scaled = con_ind*1

                con_scaled[:,0, 0] = con_ind[:,0,0] + np.array(inc_x)[:]
                con_scaled[:,0, 1] = con_ind[:,0,1] + np.array(inc_y)[:]

                con_scaled[:,0, 1][con_scaled[:,0, 1]<0] = 0
                con_scaled[:,0, 0][con_scaled[:,0, 0]<0] = 0

                con_ind = con_ind.astype(np.int32)

                results = [cv2.pointPolygonTest(con_ind, (con_scaled[ind,0, 0], con_scaled[ind,0, 1]), False)
                           for ind in range(len(con_scaled[:,0, 1])) ]
                results = np.array(results)
                results[results==0] = 1

                diff_result = np.diff(results)

                indices_2 = [ind for ind in range(len(diff_result)) if diff_result[ind]==2]
                indices_m2 = [ind for ind in range(len(diff_result)) if diff_result[ind]==-2]

                if results[0]==1:
                    con_scaled[:indices_m2[0]+1,0, 1] = con_ind[:indices_m2[0]+1,0,1]
                    con_scaled[:indices_m2[0]+1,0, 0] = con_ind[:indices_m2[0]+1,0,0]
                    indices_m2 = indices_m2[1:]

                if len(indices_2)>len(indices_m2):
                    con_scaled[indices_2[-1]+1:,0, 1] = con_ind[indices_2[-1]+1:,0,1]
                    con_scaled[indices_2[-1]+1:,0, 0] = con_ind[indices_2[-1]+1:,0,0]
                    indices_2 = indices_2[:-1]

                for ii in range(len(indices_2)):
                    con_scaled[indices_2[ii]+1:indices_m2[ii]+1,0, 1] = con_scaled[indices_2[ii],0, 1]
                    con_scaled[indices_2[ii]+1:indices_m2[ii]+1,0, 0] = con_scaled[indices_2[ii],0, 0]

                all_found_textline_polygons[j][ij][:,0,1] = con_scaled[:,0, 1]
                all_found_textline_polygons[j][ij][:,0,0] = con_scaled[:,0, 0]
        return all_found_textline_polygons

    def filter_contours_inside_a_bigger_one(self,contours, contours_d_ordered, image, marginal_cnts=None, type_contour="textregion"):
        if type_contour=="textregion":
            areas = [cv2.contourArea(contours[j]) for j in range(len(contours))]
            area_tot = image.shape[0]*image.shape[1]

            M_main = [cv2.moments(contours[j])
                      for j in range(len(contours))]
            cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
            cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]

            areas_ratio = np.array(areas)/ area_tot
            contours_index_small = [ind for ind in range(len(contours)) if areas_ratio[ind] < 1e-3]
            contours_index_big = [ind  for ind in range(len(contours)) if areas_ratio[ind] >= 1e-3]

            #contours_> = [contours[ind] for ind in contours_index_big]
            indexes_to_be_removed = []
            for ind_small in contours_index_small:
                results = [cv2.pointPolygonTest(contours[ind], (cx_main[ind_small], cy_main[ind_small]), False)
                           for ind in contours_index_big]
                if marginal_cnts:
                    results_marginal = [cv2.pointPolygonTest(marginal_cnts[ind], (cx_main[ind_small], cy_main[ind_small]), False)
                                        for ind in range(len(marginal_cnts))]
                    results_marginal = np.array(results_marginal)

                    if np.any(results_marginal==1):
                        indexes_to_be_removed.append(ind_small)

                results = np.array(results)

                if np.any(results==1):
                    indexes_to_be_removed.append(ind_small)

            if len(indexes_to_be_removed)>0:
                indexes_to_be_removed = np.unique(indexes_to_be_removed)
                indexes_to_be_removed = np.sort(indexes_to_be_removed)[::-1]
                for ind in indexes_to_be_removed:
                    contours.pop(ind)
                    if len(contours_d_ordered)>0:
                        contours_d_ordered.pop(ind)

            return contours, contours_d_ordered

        else:
            contours_txtline_of_all_textregions = []
            indexes_of_textline_tot = []
            index_textline_inside_textregion = []

            for jj in range(len(contours)):
                contours_txtline_of_all_textregions = contours_txtline_of_all_textregions + contours[jj]

                ind_textline_inside_tr = list(range(len(contours[jj])))
                index_textline_inside_textregion = index_textline_inside_textregion + ind_textline_inside_tr
                ind_ins = [jj] * len(contours[jj])
                indexes_of_textline_tot = indexes_of_textline_tot + ind_ins

            M_main_tot = [cv2.moments(contours_txtline_of_all_textregions[j])
                          for j in range(len(contours_txtline_of_all_textregions))]
            cx_main_tot = [(M_main_tot[j]["m10"] / (M_main_tot[j]["m00"] + 1e-32)) for j in range(len(M_main_tot))]
            cy_main_tot = [(M_main_tot[j]["m01"] / (M_main_tot[j]["m00"] + 1e-32)) for j in range(len(M_main_tot))]

            areas_tot = [cv2.contourArea(con_ind) for con_ind in contours_txtline_of_all_textregions]
            area_tot_tot = image.shape[0]*image.shape[1]

            textregion_index_to_del = []
            textline_in_textregion_index_to_del = []
            for ij in range(len(contours_txtline_of_all_textregions)):
                args_all = list(np.array(range(len(contours_txtline_of_all_textregions))))
                args_all.pop(ij)

                areas_without = np.array(areas_tot)[args_all]
                area_of_con_interest = areas_tot[ij]

                args_with_bigger_area = np.array(args_all)[areas_without > 1.5*area_of_con_interest]

                if len(args_with_bigger_area)>0:
                    results = [cv2.pointPolygonTest(contours_txtline_of_all_textregions[ind], (cx_main_tot[ij], cy_main_tot[ij]), False)
                               for ind in args_with_bigger_area ]
                    results = np.array(results)
                    if np.any(results==1):
                        #print(indexes_of_textline_tot[ij], index_textline_inside_textregion[ij])
                        textregion_index_to_del.append(int(indexes_of_textline_tot[ij]))
                        textline_in_textregion_index_to_del.append(int(index_textline_inside_textregion[ij]))
                        #contours[int(indexes_of_textline_tot[ij])].pop(int(index_textline_inside_textregion[ij]))

            textregion_index_to_del = np.array(textregion_index_to_del)
            textline_in_textregion_index_to_del = np.array(textline_in_textregion_index_to_del)
            for ind_u_a_trs in np.unique(textregion_index_to_del):
                textline_in_textregion_index_to_del_ind = textline_in_textregion_index_to_del[textregion_index_to_del==ind_u_a_trs]
                textline_in_textregion_index_to_del_ind = np.sort(textline_in_textregion_index_to_del_ind)[::-1]
                for ittrd in textline_in_textregion_index_to_del_ind:
                    contours[ind_u_a_trs].pop(ittrd)

            return contours
        
    def return_indexes_of_contours_loctaed_inside_another_list_of_contours(self, contours, contours_loc, cx_main_loc, cy_main_loc, indexes_loc):
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
            indexes = indexes_loc[indexes_in]# [(results == 0) | (results == 1)]#np.where((results == 0) | (results == 1))

            indexes_of_located_cont.append(indexes)
            center_x_coordinates_of_located.append(np.array(cx_main_loc)[indexes_in] )
            center_y_coordinates_of_located.append(np.array(cy_main_loc)[indexes_in] )
            
        return indexes_of_located_cont, center_x_coordinates_of_located, center_y_coordinates_of_located
        

    def filter_contours_without_textline_inside(
            self, contours,text_con_org,  contours_textline, contours_only_text_parent_d_ordered, conf_contours_textregions):
        ###contours_txtline_of_all_textregions = []
        ###for jj in range(len(contours_textline)):
            ###contours_txtline_of_all_textregions = contours_txtline_of_all_textregions + contours_textline[jj]

        ###M_main_textline = [cv2.moments(contours_txtline_of_all_textregions[j])
        ###                   for j in range(len(contours_txtline_of_all_textregions))]
        ###cx_main_textline = [(M_main_textline[j]["m10"] / (M_main_textline[j]["m00"] + 1e-32))
        ###                    for j in range(len(M_main_textline))]
        ###cy_main_textline = [(M_main_textline[j]["m01"] / (M_main_textline[j]["m00"] + 1e-32))
        ###                    for j in range(len(M_main_textline))]

        ###M_main = [cv2.moments(contours[j]) for j in range(len(contours))]
        ###cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
        ###cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]

        ###contours_with_textline = []
        ###for ind_tr, con_tr in enumerate(contours):
            ###results = [cv2.pointPolygonTest(con_tr, (cx_main_textline[index_textline_con], cy_main_textline[index_textline_con]), False)
        ###               for index_textline_con in range(len(contours_txtline_of_all_textregions)) ]
            ###results = np.array(results)
            ###if np.any(results==1):
                ###contours_with_textline.append(con_tr)

        textregion_index_to_del = []
        for index_textregion, textlines_textregion in enumerate(contours_textline):
            if len(textlines_textregion)==0:
                textregion_index_to_del.append(index_textregion)

        uniqe_args_trs = np.unique(textregion_index_to_del)
        uniqe_args_trs_sorted = np.sort(uniqe_args_trs)[::-1]

        for ind_u_a_trs in uniqe_args_trs_sorted:
            conf_contours_textregions.pop(ind_u_a_trs)
            contours.pop(ind_u_a_trs)
            contours_textline.pop(ind_u_a_trs)
            text_con_org.pop(ind_u_a_trs)
            if len(contours_only_text_parent_d_ordered) > 0:
                contours_only_text_parent_d_ordered.pop(ind_u_a_trs)

        return contours, text_con_org, conf_contours_textregions, contours_textline, contours_only_text_parent_d_ordered, np.array(range(len(contours)))

    def dilate_textlines(self, all_found_textline_polygons):
        for j in range(len(all_found_textline_polygons)):
            for i in range(len(all_found_textline_polygons[j])):
                con_ind = all_found_textline_polygons[j][i]
                con_ind = con_ind.astype(float)

                x_differential = np.diff( con_ind[:,0,0])
                y_differential = np.diff( con_ind[:,0,1])

                x_min = float(np.min( con_ind[:,0,0] ))
                y_min = float(np.min( con_ind[:,0,1] ))

                x_max = float(np.max( con_ind[:,0,0] ))
                y_max = float(np.max( con_ind[:,0,1] ))

                if (y_max - y_min) > (x_max - x_min) and (x_max - x_min)<70:
                    x_biger_than_x = np.abs(x_differential) > np.abs(y_differential)
                    mult = x_biger_than_x*x_differential

                    arg_min_mult = np.argmin(mult)
                    arg_max_mult = np.argmax(mult)

                    if y_differential[0]==0:
                        y_differential[0] = 0.1
                    if y_differential[-1]==0:
                        y_differential[-1]= 0.1
                    y_differential = [y_differential[ind] if y_differential[ind] != 0
                                      else 0.5 * (y_differential[ind-1] + y_differential[ind+1])
                                      for ind in range(len(y_differential))]

                    if y_differential[0]==0.1:
                        y_differential[0] = y_differential[1]
                    if y_differential[-1]==0.1:
                        y_differential[-1] = y_differential[-2]
                    y_differential.append(y_differential[0])

                    y_differential = [-1 if y_differential[ind] < 0 else 1
                                      for ind in range(len(y_differential))]
                    y_differential = self.return_it_in_two_groups(y_differential)
                    y_differential = np.array(y_differential)

                    con_scaled = con_ind*1
                    con_scaled[:,0, 0] = con_ind[:,0,0] - 8*y_differential
                    con_scaled[arg_min_mult,0, 1] = con_ind[arg_min_mult,0,1] + 8
                    con_scaled[arg_min_mult+1,0, 1] = con_ind[arg_min_mult+1,0,1] + 8

                    try:
                        con_scaled[arg_min_mult-1,0, 1] = con_ind[arg_min_mult-1,0,1] + 5
                        con_scaled[arg_min_mult+2,0, 1] = con_ind[arg_min_mult+2,0,1] + 5
                    except:
                        pass

                    con_scaled[arg_max_mult,0, 1] = con_ind[arg_max_mult,0,1] - 8
                    con_scaled[arg_max_mult+1,0, 1] = con_ind[arg_max_mult+1,0,1] - 8

                    try:
                        con_scaled[arg_max_mult-1,0, 1] = con_ind[arg_max_mult-1,0,1] - 5
                        con_scaled[arg_max_mult+2,0, 1] = con_ind[arg_max_mult+2,0,1] - 5
                    except:
                        pass

                else:
                    y_biger_than_x = np.abs(y_differential) > np.abs(x_differential)
                    mult = y_biger_than_x*y_differential

                    arg_min_mult = np.argmin(mult)
                    arg_max_mult = np.argmax(mult)

                    if x_differential[0]==0:
                        x_differential[0] = 0.1
                    if x_differential[-1]==0:
                        x_differential[-1]= 0.1
                    x_differential = [x_differential[ind] if x_differential[ind] != 0
                                      else 0.5 * (x_differential[ind-1] + x_differential[ind+1])
                                      for ind in range(len(x_differential))]

                    if x_differential[0]==0.1:
                        x_differential[0] = x_differential[1]
                    if x_differential[-1]==0.1:
                        x_differential[-1] = x_differential[-2]
                    x_differential.append(x_differential[0])

                    x_differential = [-1 if x_differential[ind] < 0 else 1
                                      for ind in range(len(x_differential))]
                    x_differential = self.return_it_in_two_groups(x_differential)
                    x_differential = np.array(x_differential)

                    con_scaled = con_ind*1
                    con_scaled[:,0, 1] = con_ind[:,0,1] + 8*x_differential
                    con_scaled[arg_min_mult,0, 0] = con_ind[arg_min_mult,0,0] + 8
                    con_scaled[arg_min_mult+1,0, 0] = con_ind[arg_min_mult+1,0,0] + 8

                    try:
                        con_scaled[arg_min_mult-1,0, 0] = con_ind[arg_min_mult-1,0,0] + 5
                        con_scaled[arg_min_mult+2,0, 0] = con_ind[arg_min_mult+2,0,0] + 5
                    except:
                        pass

                    con_scaled[arg_max_mult,0, 0] = con_ind[arg_max_mult,0,0] - 8
                    con_scaled[arg_max_mult+1,0, 0] = con_ind[arg_max_mult+1,0,0] - 8

                    try:
                        con_scaled[arg_max_mult-1,0, 0] = con_ind[arg_max_mult-1,0,0] - 5
                        con_scaled[arg_max_mult+2,0, 0] = con_ind[arg_max_mult+2,0,0] - 5
                    except:
                        pass

                con_scaled[:,0, 1][con_scaled[:,0, 1]<0] = 0
                con_scaled[:,0, 0][con_scaled[:,0, 0]<0] = 0

                all_found_textline_polygons[j][i][:,0,1] = con_scaled[:,0, 1]
                all_found_textline_polygons[j][i][:,0,0] = con_scaled[:,0, 0]

        return all_found_textline_polygons

    def delete_regions_without_textlines(
            self, slopes, all_found_textline_polygons, boxes_text, txt_con_org,
            contours_only_text_parent, index_by_text_par_con):

        slopes_rem = []
        all_found_textline_polygons_rem = []
        boxes_text_rem = []
        txt_con_org_rem = []
        contours_only_text_parent_rem = []
        index_by_text_par_con_rem = []

        for i, ind_con in enumerate(all_found_textline_polygons):
            if len(ind_con):
                all_found_textline_polygons_rem.append(ind_con)
                slopes_rem.append(slopes[i])
                boxes_text_rem.append(boxes_text[i])
                txt_con_org_rem.append(txt_con_org[i])
                contours_only_text_parent_rem.append(contours_only_text_parent[i])
                index_by_text_par_con_rem.append(index_by_text_par_con[i])

        index_sort = np.argsort(index_by_text_par_con_rem)
        indexes_new = np.array(range(len(index_by_text_par_con_rem)))

        index_by_text_par_con_rem_sort = [indexes_new[index_sort==j][0]
                                          for j in range(len(index_by_text_par_con_rem))]

        return (slopes_rem, all_found_textline_polygons_rem, boxes_text_rem, txt_con_org_rem,
                contours_only_text_parent_rem, index_by_text_par_con_rem_sort)

    def run(self, image_filename : Optional[str] = None, dir_in : Optional[str] = None, overwrite : bool = False):
        """
        Get image and scales, then extract the page of scanned image
        """
        self.logger.debug("enter run")
        t0_tot = time.time()

        if dir_in:
            self.ls_imgs  = os.listdir(dir_in)
        elif image_filename:
            self.ls_imgs = [image_filename]
        else:
            raise ValueError("run requires either a single image filename or a directory")

        for img_filename in self.ls_imgs:
            print(img_filename, 'img_filename')
            self.logger.info(img_filename)
            t0 = time.time()

            self.reset_file_name_dir(os.path.join(dir_in or "", img_filename))
            #print("text region early -11 in %.1fs", time.time() - t0)
            if os.path.exists(self.writer.output_filename):
                if overwrite:
                    self.logger.warning("will overwrite existing output file '%s'", self.writer.output_filename)
                else:
                    self.logger.warning("will skip input for existing output file '%s'", self.writer.output_filename)
                    continue

            pcgts = self.run_single()
            self.logger.info("Job done in %.1fs", time.time() - t0)
            #print("Job done in %.1fs" % (time.time() - t0))
            self.writer.write_pagexml(pcgts)

        if dir_in:
            self.logger.info("All jobs done in %.1fs", time.time() - t0_tot)
            print("all Job done in %.1fs", time.time() - t0_tot)

    def run_single(self):
        t0 = time.time()
        img_res, is_image_enhanced, num_col_classifier, num_column_is_classified = self.run_enhancement(self.light_version)
        self.logger.info("Enhancing took %.1fs ", time.time() - t0)
        if self.extract_only_images:
            text_regions_p_1, erosion_hurts, polygons_lines_xml, polygons_of_images, image_page, page_coord, cont_page = \
                self.get_regions_light_v_extract_only_images(img_res, is_image_enhanced, num_col_classifier)
            pcgts = self.writer.build_pagexml_no_full_layout(
                [], page_coord, [], [], [], [],
                polygons_of_images, [], [], [], [], [],
                cont_page, [], [])
            if self.plotter:
                self.plotter.write_images_into_directory(polygons_of_images, image_page)
            return pcgts

        if self.skip_layout_and_reading_order:
            _ ,_, _, textline_mask_tot_ea, img_bin_light, _ = \
                self.get_regions_light_v(img_res, is_image_enhanced, num_col_classifier,
                                         skip_layout_and_reading_order=self.skip_layout_and_reading_order)

            page_coord, image_page, textline_mask_tot_ea, img_bin_light, cont_page = \
                self.run_graphics_and_columns_without_layout(textline_mask_tot_ea, img_bin_light)


            ##all_found_textline_polygons =self.scale_contours_new(textline_mask_tot_ea)

            cnt_clean_rot_raw, hir_on_cnt_clean_rot = return_contours_of_image(textline_mask_tot_ea)
            all_found_textline_polygons = filter_contours_area_of_image(
                textline_mask_tot_ea, cnt_clean_rot_raw, hir_on_cnt_clean_rot, max_area=1, min_area=0.00001)
            
            all_found_textline_polygons = all_found_textline_polygons[::-1]

            all_found_textline_polygons=[ all_found_textline_polygons ]

            all_found_textline_polygons = self.dilate_textregions_contours_textline_version(
                all_found_textline_polygons)
            all_found_textline_polygons = self.filter_contours_inside_a_bigger_one(
                all_found_textline_polygons, None, textline_mask_tot_ea, type_contour="textline")
            
            
            order_text_new = [0]
            slopes =[0]
            id_of_texts_tot =['region_0001']

            polygons_of_images = []
            slopes_marginals = []
            polygons_of_marginals = []
            all_found_textline_polygons_marginals = []
            all_box_coord_marginals = []
            polygons_lines_xml = []
            contours_tables = []
            conf_contours_textregions =[0]
            
            if self.ocr and not self.tr:
                gc.collect()
                ocr_all_textlines = return_rnn_cnn_ocr_of_given_textlines(image_page, all_found_textline_polygons, self.prediction_model, self.b_s_ocr, self.num_to_char, textline_light=True)
            else:
                ocr_all_textlines = None
            
            pcgts = self.writer.build_pagexml_no_full_layout(
                cont_page, page_coord, order_text_new, id_of_texts_tot,
                all_found_textline_polygons, page_coord, polygons_of_images, polygons_of_marginals,
                all_found_textline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals,
                cont_page, polygons_lines_xml, contours_tables, ocr_all_textlines=ocr_all_textlines, conf_contours_textregion=conf_contours_textregions, skip_layout_reading_order=self.skip_layout_and_reading_order)
            return pcgts

        #print("text region early -1 in %.1fs", time.time() - t0)
        t1 = time.time()
        if self.light_version:
            text_regions_p_1 ,erosion_hurts, polygons_lines_xml, textline_mask_tot_ea, img_bin_light, confidence_matrix = \
                self.get_regions_light_v(img_res, is_image_enhanced, num_col_classifier)
            #print("text region early -2 in %.1fs", time.time() - t0)

            if num_col_classifier == 1 or num_col_classifier ==2:
                if num_col_classifier == 1:
                    img_w_new = 1000
                else:
                    img_w_new = 1300
                img_h_new = img_w_new * textline_mask_tot_ea.shape[0] // textline_mask_tot_ea.shape[1]

                textline_mask_tot_ea_deskew = resize_image(textline_mask_tot_ea,img_h_new, img_w_new )

                slope_deskew, slope_first = self.run_deskew(textline_mask_tot_ea_deskew)
            else:
                slope_deskew, slope_first = self.run_deskew(textline_mask_tot_ea)
            #print("text region early -2,5 in %.1fs", time.time() - t0)
            #self.logger.info("Textregion detection took %.1fs ", time.time() - t1t)
            num_col, num_col_classifier, img_only_regions, page_coord, image_page, mask_images, mask_lines, \
                text_regions_p_1, cont_page, table_prediction, textline_mask_tot_ea, img_bin_light = \
                    self.run_graphics_and_columns_light(text_regions_p_1, textline_mask_tot_ea,
                                                        num_col_classifier, num_column_is_classified, erosion_hurts, img_bin_light)
            #self.logger.info("run graphics %.1fs ", time.time() - t1t)
            #print("text region early -3 in %.1fs", time.time() - t0)
            textline_mask_tot_ea_org = np.copy(textline_mask_tot_ea)
            #print("text region early -4 in %.1fs", time.time() - t0)
        else:
            text_regions_p_1 ,erosion_hurts, polygons_lines_xml = \
                self.get_regions_from_xy_2models(img_res, is_image_enhanced,
                                                 num_col_classifier)
            self.logger.info("Textregion detection took %.1fs ", time.time() - t1)
            confidence_matrix = np.zeros((text_regions_p_1.shape[:2]))

            t1 = time.time()
            num_col, num_col_classifier, img_only_regions, page_coord, image_page, mask_images, mask_lines, \
                text_regions_p_1, cont_page, table_prediction = \
                    self.run_graphics_and_columns(text_regions_p_1, num_col_classifier, num_column_is_classified, erosion_hurts)
            self.logger.info("Graphics detection took %.1fs ", time.time() - t1)
            #self.logger.info('cont_page %s', cont_page)
        #plt.imshow(table_prediction)
        #plt.show()

        if not num_col:
            self.logger.info("No columns detected, outputting an empty PAGE-XML")
            pcgts = self.writer.build_pagexml_no_full_layout(
                [], page_coord, [], [], [], [], [], [], [], [], [], [],
                cont_page, [], [])
            return pcgts

        #print("text region early in %.1fs", time.time() - t0)
        t1 = time.time()
        if not self.light_version:
            textline_mask_tot_ea = self.run_textline(image_page)
            self.logger.info("textline detection took %.1fs", time.time() - t1)
            t1 = time.time()
            slope_deskew, slope_first = self.run_deskew(textline_mask_tot_ea)
            self.logger.info("deskewing took %.1fs", time.time() - t1)
        elif num_col_classifier in (1,2):
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

        textline_mask_tot, text_regions_p, image_page_rotated = \
            self.run_marginals(image_page, textline_mask_tot_ea, mask_images, mask_lines,
                               num_col_classifier, slope_deskew, text_regions_p_1, table_prediction)

        if self.light_version and num_col_classifier in (1,2):
            image_page = resize_image(image_page,org_h_l_m, org_w_l_m )
            textline_mask_tot_ea = resize_image(textline_mask_tot_ea,org_h_l_m, org_w_l_m )
            text_regions_p = resize_image(text_regions_p,org_h_l_m, org_w_l_m )
            textline_mask_tot = resize_image(textline_mask_tot,org_h_l_m, org_w_l_m )
            text_regions_p_1 = resize_image(text_regions_p_1,org_h_l_m, org_w_l_m )
            table_prediction = resize_image(table_prediction,org_h_l_m, org_w_l_m )
            image_page_rotated = resize_image(image_page_rotated,org_h_l_m, org_w_l_m )

        self.logger.info("detection of marginals took %.1fs", time.time() - t1)
        #print("text region early 2 marginal in %.1fs", time.time() - t0)
        ## birdan sora chock chakir
        t1 = time.time()
        if not self.full_layout:
            polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_separators_d, \
                boxes, boxes_d, polygons_of_marginals, contours_tables = \
                self.run_boxes_no_full_layout(image_page, textline_mask_tot, text_regions_p, slope_deskew,
                                              num_col_classifier, table_prediction, erosion_hurts)
            ###polygons_of_marginals = self.dilate_textregions_contours(polygons_of_marginals)
        else:
            polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_separators_d, \
                regions_fully, regions_without_separators, polygons_of_marginals, contours_tables = \
                self.run_boxes_full_layout(image_page, textline_mask_tot, text_regions_p, slope_deskew,
                                           num_col_classifier, img_only_regions, table_prediction, erosion_hurts,
                                           img_bin_light if self.light_version else None)
            ###polygons_of_marginals = self.dilate_textregions_contours(polygons_of_marginals)
            if self.light_version:
                drop_label_in_full_layout = 4
                textline_mask_tot_ea_org[img_revised_tab==drop_label_in_full_layout] = 0


        text_only = ((img_revised_tab[:, :] == 1)) * 1
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            text_only_d = ((text_regions_p_1_n[:, :] == 1)) * 1

        #print("text region early 2 in %.1fs", time.time() - t0)
        ###min_con_area = 0.000005
        contours_only_text, hir_on_text = return_contours_of_image(text_only)
        contours_only_text_parent = return_parent_contours(contours_only_text, hir_on_text)
        if len(contours_only_text_parent) > 0:
            areas_cnt_text = np.array([cv2.contourArea(c) for c in contours_only_text_parent])
            areas_cnt_text = areas_cnt_text / float(text_only.shape[0] * text_only.shape[1])
            #self.logger.info('areas_cnt_text %s', areas_cnt_text)
            contours_biggest = contours_only_text_parent[np.argmax(areas_cnt_text)]
            contours_only_text_parent = [c for jz, c in enumerate(contours_only_text_parent)
                                         if areas_cnt_text[jz] > MIN_AREA_REGION]
            areas_cnt_text_parent = [area for area in areas_cnt_text if area > MIN_AREA_REGION]
            index_con_parents = np.argsort(areas_cnt_text_parent)

            contours_only_text_parent = self.return_list_of_contours_with_desired_order(
                contours_only_text_parent, index_con_parents)

            ##try:
                ##contours_only_text_parent = \
                    ##list(np.array(contours_only_text_parent,dtype=object)[index_con_parents])
            ##except:
                ##contours_only_text_parent = \
                    ##list(np.array(contours_only_text_parent,dtype=np.int32)[index_con_parents])
            ##areas_cnt_text_parent = list(np.array(areas_cnt_text_parent)[index_con_parents])
            areas_cnt_text_parent = self.return_list_of_contours_with_desired_order(
                areas_cnt_text_parent, index_con_parents)

            cx_bigest_big, cy_biggest_big, _, _, _, _, _ = find_new_features_of_contours([contours_biggest])
            cx_bigest, cy_biggest, _, _, _, _, _ = find_new_features_of_contours(contours_only_text_parent)

            if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                contours_only_text_d, hir_on_text_d = return_contours_of_image(text_only_d)
                contours_only_text_parent_d = return_parent_contours(contours_only_text_d, hir_on_text_d)

                areas_cnt_text_d = np.array([cv2.contourArea(c) for c in contours_only_text_parent_d])
                areas_cnt_text_d = areas_cnt_text_d / float(text_only_d.shape[0] * text_only_d.shape[1])

                if len(areas_cnt_text_d)>0:
                    contours_biggest_d = contours_only_text_parent_d[np.argmax(areas_cnt_text_d)]
                    index_con_parents_d = np.argsort(areas_cnt_text_d)
                    contours_only_text_parent_d = self.return_list_of_contours_with_desired_order(
                        contours_only_text_parent_d, index_con_parents_d)
                    #try:
                        #contours_only_text_parent_d = \
                            #list(np.array(contours_only_text_parent_d,dtype=object)[index_con_parents_d])
                    #except:
                        #contours_only_text_parent_d = \
                            #list(np.array(contours_only_text_parent_d,dtype=np.int32)[index_con_parents_d])
                    #areas_cnt_text_d = list(np.array(areas_cnt_text_d)[index_con_parents_d])
                    areas_cnt_text_d = self.return_list_of_contours_with_desired_order(
                        areas_cnt_text_d, index_con_parents_d)

                    cx_bigest_d_big, cy_biggest_d_big, _, _, _, _, _ = find_new_features_of_contours([contours_biggest_d])
                    cx_bigest_d, cy_biggest_d, _, _, _, _, _ = find_new_features_of_contours(contours_only_text_parent_d)
                    try:
                        if len(cx_bigest_d) >= 5:
                            cx_bigest_d_last5 = cx_bigest_d[-5:]
                            cy_biggest_d_last5 = cy_biggest_d[-5:]
                            dists_d = [math.sqrt((cx_bigest_big[0] - cx_bigest_d_last5[j]) ** 2 +
                                                 (cy_biggest_big[0] - cy_biggest_d_last5[j]) ** 2)
                                       for j in range(len(cy_biggest_d_last5))]
                            ind_largest = len(cx_bigest_d) -5 + np.argmin(dists_d)
                        else:
                            cx_bigest_d_last5 = cx_bigest_d[-len(cx_bigest_d):]
                            cy_biggest_d_last5 = cy_biggest_d[-len(cx_bigest_d):]
                            dists_d = [math.sqrt((cx_bigest_big[0]-cx_bigest_d_last5[j])**2 +
                                                 (cy_biggest_big[0]-cy_biggest_d_last5[j])**2)
                                       for j in range(len(cy_biggest_d_last5))]
                            ind_largest = len(cx_bigest_d) - len(cx_bigest_d) + np.argmin(dists_d)

                        cx_bigest_d_big[0] = cx_bigest_d[ind_largest]
                        cy_biggest_d_big[0] = cy_biggest_d[ind_largest]
                    except Exception as why:
                        self.logger.error(why)

                    (h, w) = text_only.shape[:2]
                    center = (w // 2.0, h // 2.0)
                    M = cv2.getRotationMatrix2D(center, slope_deskew, 1.0)
                    M_22 = np.array(M)[:2, :2]
                    p_big = np.dot(M_22, [cx_bigest_big, cy_biggest_big])
                    x_diff = p_big[0] - cx_bigest_d_big
                    y_diff = p_big[1] - cy_biggest_d_big

                    contours_only_text_parent_d_ordered = []
                    for i in range(len(contours_only_text_parent)):
                        p = np.dot(M_22, [cx_bigest[i], cy_biggest[i]])
                        p[0] = p[0] - x_diff[0]
                        p[1] = p[1] - y_diff[0]
                        dists = [math.sqrt((p[0] - cx_bigest_d[j]) ** 2 +
                                           (p[1] - cy_biggest_d[j]) ** 2)
                                 for j in range(len(cx_bigest_d))]
                        contours_only_text_parent_d_ordered.append(contours_only_text_parent_d[np.argmin(dists)])
                        # img2=np.zeros((text_only.shape[0],text_only.shape[1],3))
                        # img2=cv2.fillPoly(img2,pts=[contours_only_text_parent_d[np.argmin(dists)]] ,color=(1,1,1))
                        # plt.imshow(img2[:,:,0])
                        # plt.show()
                else:
                    contours_only_text_parent_d_ordered = []
                    contours_only_text_parent_d = []
                    contours_only_text_parent = []

            else:
                contours_only_text_parent_d_ordered = []
                contours_only_text_parent_d = []
                #contours_only_text_parent = []
        if not len(contours_only_text_parent):
            # stop early
            empty_marginals = [[]] * len(polygons_of_marginals)
            if self.full_layout:
                pcgts = self.writer.build_pagexml_full_layout(
                    [], [], page_coord, [], [], [], [], [], [],
                    polygons_of_images, contours_tables, [],
                    polygons_of_marginals, empty_marginals, empty_marginals, [], [], [],
                    cont_page, polygons_lines_xml)
            else:
                pcgts = self.writer.build_pagexml_no_full_layout(
                    [], page_coord, [], [], [], [],
                    polygons_of_images,
                    polygons_of_marginals, empty_marginals, empty_marginals, [], [],
                    cont_page, polygons_lines_xml, contours_tables)
            return pcgts



        #print("text region early 3 in %.1fs", time.time() - t0)
        if self.light_version:
            contours_only_text_parent = self.dilate_textregions_contours(
                contours_only_text_parent)
            contours_only_text_parent , contours_only_text_parent_d_ordered = self.filter_contours_inside_a_bigger_one(
                contours_only_text_parent, contours_only_text_parent_d_ordered, text_only, marginal_cnts=polygons_of_marginals)
            #print("text region early 3.5 in %.1fs", time.time() - t0)
            txt_con_org , conf_contours_textregions = get_textregion_contours_in_org_image_light(
                contours_only_text_parent, self.image, slope_first, confidence_matrix,  map=self.executor.map)
            #txt_con_org = self.dilate_textregions_contours(txt_con_org)
            #contours_only_text_parent = self.dilate_textregions_contours(contours_only_text_parent)
        else:
            txt_con_org , conf_contours_textregions = get_textregion_contours_in_org_image_light(
                contours_only_text_parent, self.image, slope_first, confidence_matrix,  map=self.executor.map)
        #print("text region early 4 in %.1fs", time.time() - t0)
        boxes_text, _ = get_text_region_boxes_by_given_contours(contours_only_text_parent)
        boxes_marginals, _ = get_text_region_boxes_by_given_contours(polygons_of_marginals)
        #print("text region early 5 in %.1fs", time.time() - t0)
        ## birdan sora chock chakir
        if not self.curved_line:
            if self.light_version:
                if self.textline_light:
                    all_found_textline_polygons, boxes_text, txt_con_org, contours_only_text_parent, \
                        all_box_coord, index_by_text_par_con, slopes = self.get_slopes_and_deskew_new_light2(
                            txt_con_org, contours_only_text_parent, textline_mask_tot_ea_org,
                            image_page_rotated, boxes_text, slope_deskew)
                    all_found_textline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, \
                        all_box_coord_marginals, _, slopes_marginals = self.get_slopes_and_deskew_new_light2(
                            polygons_of_marginals, polygons_of_marginals, textline_mask_tot_ea_org,
                            image_page_rotated, boxes_marginals, slope_deskew)

                    #slopes, all_found_textline_polygons, boxes_text, txt_con_org, contours_only_text_parent, index_by_text_par_con = \
                    #    self.delete_regions_without_textlines(slopes, all_found_textline_polygons,
                    #        boxes_text, txt_con_org, contours_only_text_parent, index_by_text_par_con)
                    #slopes_marginals, all_found_textline_polygons_marginals, boxes_marginals, polygons_of_marginals, polygons_of_marginals, _ = \
                    #    self.delete_regions_without_textlines(slopes_marginals, all_found_textline_polygons_marginals,
                    #        boxes_marginals, polygons_of_marginals, polygons_of_marginals, np.array(range(len(polygons_of_marginals))))
                    #all_found_textline_polygons = self.dilate_textlines(all_found_textline_polygons)
                    #####all_found_textline_polygons = self.dilate_textline_contours(all_found_textline_polygons)
                    all_found_textline_polygons = self.dilate_textregions_contours_textline_version(
                        all_found_textline_polygons)
                    all_found_textline_polygons = self.filter_contours_inside_a_bigger_one(
                        all_found_textline_polygons, None, textline_mask_tot_ea_org, type_contour="textline")
                    all_found_textline_polygons_marginals = self.dilate_textregions_contours_textline_version(
                        all_found_textline_polygons_marginals)
                    contours_only_text_parent, txt_con_org, conf_contours_textregions, all_found_textline_polygons, contours_only_text_parent_d_ordered, \
                        index_by_text_par_con = self.filter_contours_without_textline_inside(
                            contours_only_text_parent, txt_con_org, all_found_textline_polygons, contours_only_text_parent_d_ordered, conf_contours_textregions)
                else:
                    textline_mask_tot_ea = cv2.erode(textline_mask_tot_ea, kernel=KERNEL, iterations=1)
                    all_found_textline_polygons, boxes_text, txt_con_org, contours_only_text_parent, all_box_coord, \
                        index_by_text_par_con, slopes = self.get_slopes_and_deskew_new_light(
                            txt_con_org, contours_only_text_parent, textline_mask_tot_ea,
                            image_page_rotated, boxes_text, slope_deskew)
                    all_found_textline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, \
                        all_box_coord_marginals, _, slopes_marginals = self.get_slopes_and_deskew_new_light(
                            polygons_of_marginals, polygons_of_marginals, textline_mask_tot_ea,
                            image_page_rotated, boxes_marginals, slope_deskew)
                    #all_found_textline_polygons = self.filter_contours_inside_a_bigger_one(
                    #    all_found_textline_polygons, textline_mask_tot_ea_org, type_contour="textline")
            else:
                textline_mask_tot_ea = cv2.erode(textline_mask_tot_ea, kernel=KERNEL, iterations=1)
                all_found_textline_polygons, boxes_text, txt_con_org, contours_only_text_parent, \
                    all_box_coord, index_by_text_par_con, slopes = self.get_slopes_and_deskew_new(
                        txt_con_org, contours_only_text_parent, textline_mask_tot_ea,
                        image_page_rotated, boxes_text, slope_deskew)
                all_found_textline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, \
                    all_box_coord_marginals, _, slopes_marginals = self.get_slopes_and_deskew_new(
                        polygons_of_marginals, polygons_of_marginals, textline_mask_tot_ea,
                        image_page_rotated, boxes_marginals, slope_deskew)
        else:
            scale_param = 1
            textline_mask_tot_ea_erode = cv2.erode(textline_mask_tot_ea, kernel=KERNEL, iterations=2)
            all_found_textline_polygons, boxes_text, txt_con_org, contours_only_text_parent, \
                all_box_coord, index_by_text_par_con, slopes = self.get_slopes_and_deskew_new_curved(
                    txt_con_org, contours_only_text_parent, textline_mask_tot_ea_erode,
                    image_page_rotated, boxes_text, text_only,
                    num_col_classifier, scale_param, slope_deskew)
            all_found_textline_polygons = small_textlines_to_parent_adherence2(
                all_found_textline_polygons, textline_mask_tot_ea, num_col_classifier)
            all_found_textline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, \
                all_box_coord_marginals, _, slopes_marginals = self.get_slopes_and_deskew_new_curved(
                    polygons_of_marginals, polygons_of_marginals, textline_mask_tot_ea_erode,
                    image_page_rotated, boxes_marginals, text_only,
                    num_col_classifier, scale_param, slope_deskew)
            all_found_textline_polygons_marginals = small_textlines_to_parent_adherence2(
                all_found_textline_polygons_marginals, textline_mask_tot_ea, num_col_classifier)

        #print("text region early 6 in %.1fs", time.time() - t0)
        if self.full_layout:
            if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                contours_only_text_parent_d_ordered = self.return_list_of_contours_with_desired_order(
                    contours_only_text_parent_d_ordered, index_by_text_par_con)
                #try:
                    #contours_only_text_parent_d_ordered = \
                        #list(np.array(contours_only_text_parent_d_ordered, dtype=np.int32)[index_by_text_par_con])
                #except:
                    #contours_only_text_parent_d_ordered = \
                        #list(np.array(contours_only_text_parent_d_ordered, dtype=object)[index_by_text_par_con])
            else:
                #takes long timee
                contours_only_text_parent_d_ordered = None
            if self.light_version:
                fun = check_any_text_region_in_model_one_is_main_or_header_light
            else:
                fun = check_any_text_region_in_model_one_is_main_or_header
            text_regions_p, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, \
                all_found_textline_polygons, all_found_textline_polygons_h, slopes, slopes_h, \
                contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered, \
                    conf_contours_textregions, conf_contours_textregions_h = fun(
                    text_regions_p, regions_fully, contours_only_text_parent,
                    all_box_coord, all_found_textline_polygons, slopes, contours_only_text_parent_d_ordered, conf_contours_textregions)

            if self.plotter:
                self.plotter.save_plot_of_layout(text_regions_p, image_page)
                self.plotter.save_plot_of_layout_all(text_regions_p, image_page)

            pixel_img = 4
            polygons_of_drop_capitals = return_contours_of_interested_region_by_min_size(text_regions_p, pixel_img)
            ##all_found_textline_polygons = adhere_drop_capital_region_into_corresponding_textline(
                ##text_regions_p, polygons_of_drop_capitals, contours_only_text_parent, contours_only_text_parent_h,
                ##all_box_coord, all_box_coord_h, all_found_textline_polygons, all_found_textline_polygons_h,
                ##kernel=KERNEL, curved_line=self.curved_line, textline_light=self.textline_light)

            if not self.reading_order_machine_based:
                pixel_seps = 6
                if not self.headers_off:
                    if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                        num_col, _, matrix_of_lines_ch, splitter_y_new, _ = find_number_of_columns_in_document(
                            np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2),
                            num_col_classifier, self.tables,  pixel_seps, contours_only_text_parent_h)
                    else:
                        _, _, matrix_of_lines_ch_d, splitter_y_new_d, _ = find_number_of_columns_in_document(
                            np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2),
                            num_col_classifier, self.tables, pixel_seps, contours_only_text_parent_h_d_ordered)
                elif self.headers_off:
                    if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                        num_col, _, matrix_of_lines_ch, splitter_y_new, _ = find_number_of_columns_in_document(
                            np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2),
                            num_col_classifier, self.tables,  pixel_seps)
                    else:
                        _, _, matrix_of_lines_ch_d, splitter_y_new_d, _ = find_number_of_columns_in_document(
                            np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2),
                            num_col_classifier, self.tables, pixel_seps)

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
                        num_col_classifier, erosion_hurts, self.tables, self.right2left)
                else:
                    boxes_d, peaks_neg_tot_tables_d = return_boxes_of_images_by_order_of_reading_new(
                        splitter_y_new_d, regions_without_separators_d, matrix_of_lines_ch_d,
                        num_col_classifier, erosion_hurts, self.tables, self.right2left)

        if self.plotter:
            self.plotter.write_images_into_directory(polygons_of_images, image_page)
        t_order = time.time()

        if self.full_layout:
            if self.reading_order_machine_based:
                tror = time.time()
                order_text_new, id_of_texts_tot = self.do_order_of_regions_with_model(
                    contours_only_text_parent, contours_only_text_parent_h, text_regions_p)
                print('time spend for mb ro',  time.time()-tror)
            else:
                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                    order_text_new, id_of_texts_tot = self.do_order_of_regions(
                        contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot)
                else:
                    order_text_new, id_of_texts_tot = self.do_order_of_regions(
                        contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered, boxes_d, textline_mask_tot_d)
            self.logger.info("detection of reading order took %.1fs", time.time() - t_order)

            if self.ocr and not self.tr:
                gc.collect()
                if len(all_found_textline_polygons)>0:
                    ocr_all_textlines = return_rnn_cnn_ocr_of_given_textlines(image_page, all_found_textline_polygons, self.prediction_model, self.b_s_ocr, self.num_to_char, self.textline_light, self.curved_line)
                else:
                    ocr_all_textlines = None
                    
                if all_found_textline_polygons_marginals and len(all_found_textline_polygons_marginals)>0:
                    ocr_all_textlines_marginals = return_rnn_cnn_ocr_of_given_textlines(image_page, all_found_textline_polygons_marginals, self.prediction_model, self.b_s_ocr, self.num_to_char, self.textline_light, self.curved_line)
                else:
                    ocr_all_textlines_marginals = None
                
                if all_found_textline_polygons_h and len(all_found_textline_polygons)>0:
                    ocr_all_textlines_h = return_rnn_cnn_ocr_of_given_textlines(image_page, all_found_textline_polygons_h, self.prediction_model, self.b_s_ocr, self.num_to_char, self.textline_light, self.curved_line)
                else:
                    ocr_all_textlines_h = None
                    
                if polygons_of_drop_capitals and len(polygons_of_drop_capitals)>0:
                    ocr_all_textlines_drop = return_rnn_cnn_ocr_of_given_textlines(image_page, polygons_of_drop_capitals, self.prediction_model, self.b_s_ocr, self.num_to_char, self.textline_light, self.curved_line)
                else:
                    ocr_all_textlines_drop = None
            else:
                ocr_all_textlines = None
                ocr_all_textlines_marginals = None
                ocr_all_textlines_h = None
                ocr_all_textlines_drop = None
            pcgts = self.writer.build_pagexml_full_layout(
                contours_only_text_parent, contours_only_text_parent_h, page_coord, order_text_new, id_of_texts_tot,
                all_found_textline_polygons, all_found_textline_polygons_h, all_box_coord, all_box_coord_h,
                polygons_of_images, contours_tables, polygons_of_drop_capitals, polygons_of_marginals,
                all_found_textline_polygons_marginals, all_box_coord_marginals, slopes, slopes_h, slopes_marginals,
                cont_page, polygons_lines_xml, ocr_all_textlines, ocr_all_textlines_h, ocr_all_textlines_marginals, ocr_all_textlines_drop,  conf_contours_textregions, conf_contours_textregions_h)
            return pcgts

        contours_only_text_parent_h = None
        if self.reading_order_machine_based:
            order_text_new, id_of_texts_tot = self.do_order_of_regions_with_model(
                contours_only_text_parent, contours_only_text_parent_h, text_regions_p)
        else:
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                order_text_new, id_of_texts_tot = self.do_order_of_regions(
                    contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot)
            else:
                contours_only_text_parent_d_ordered = self.return_list_of_contours_with_desired_order(
                    contours_only_text_parent_d_ordered, index_by_text_par_con)
                #try:
                    #contours_only_text_parent_d_ordered = \
                        #list(np.array(contours_only_text_parent_d_ordered, dtype=object)[index_by_text_par_con])
                #except:
                    #contours_only_text_parent_d_ordered = \
                        #list(np.array(contours_only_text_parent_d_ordered, dtype=np.int32)[index_by_text_par_con])
                order_text_new, id_of_texts_tot = self.do_order_of_regions(
                    contours_only_text_parent_d_ordered, contours_only_text_parent_h, boxes_d, textline_mask_tot_d)

        if self.ocr and self.tr:
            device = cuda.get_current_device()
            device.reset()
            gc.collect()
            model_ocr = VisionEncoderDecoderModel.from_pretrained(self.model_ocr_dir)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            torch.cuda.empty_cache()
            model_ocr.to(device)

            ind_tot = 0
            #cv2.imwrite('./img_out.png', image_page)
            ocr_all_textlines = []
            for indexing, ind_poly_first in enumerate(all_found_textline_polygons):
                ocr_textline_in_textregion = []
                for indexing2, ind_poly in enumerate(ind_poly_first):
                    if not (self.textline_light or self.curved_line):
                        ind_poly = copy.deepcopy(ind_poly)
                        box_ind = all_box_coord[indexing]
                        #print(ind_poly,np.shape(ind_poly), 'ind_poly')
                        #print(box_ind)
                        ind_poly = self.return_textline_contour_with_added_box_coordinate(ind_poly, box_ind)
                        #print(ind_poly_copy)
                        ind_poly[ind_poly<0] = 0
                    x, y, w, h = cv2.boundingRect(ind_poly)
                    #print(ind_poly_copy, np.shape(ind_poly_copy))
                    #print(x, y, w, h, h/float(w),'ratio')
                    h2w_ratio = h/float(w)
                    mask_poly = np.zeros(image_page.shape)
                    if not self.light_version:
                        img_poly_on_img = np.copy(image_page)
                    else:
                        img_poly_on_img = np.copy(img_bin_light)
                    mask_poly = cv2.fillPoly(mask_poly, pts=[ind_poly], color=(1, 1, 1))

                    if self.textline_light:
                        mask_poly = cv2.dilate(mask_poly, KERNEL, iterations=1)
                    img_poly_on_img[:,:,0][mask_poly[:,:,0] ==0] = 255
                    img_poly_on_img[:,:,1][mask_poly[:,:,0] ==0] = 255
                    img_poly_on_img[:,:,2][mask_poly[:,:,0] ==0] = 255

                    img_croped = img_poly_on_img[y:y+h, x:x+w, :]
                    #cv2.imwrite('./extracted_lines/'+str(ind_tot)+'.jpg', img_croped)
                    text_ocr = self.return_ocr_of_textline_without_common_section(img_croped, model_ocr, processor, device, w, h2w_ratio, ind_tot)
                    ocr_textline_in_textregion.append(text_ocr)
                    ind_tot = ind_tot +1
                ocr_all_textlines.append(ocr_textline_in_textregion)
                
        elif self.ocr and not self.tr:
            gc.collect()
            if len(all_found_textline_polygons)>0:
                ocr_all_textlines = return_rnn_cnn_ocr_of_given_textlines(image_page, all_found_textline_polygons, self.prediction_model, self.b_s_ocr, self.num_to_char, self.textline_light, self.curved_line)
            if all_found_textline_polygons_marginals and len(all_found_textline_polygons_marginals)>0:
                ocr_all_textlines_marginals = return_rnn_cnn_ocr_of_given_textlines(image_page, all_found_textline_polygons_marginals, self.prediction_model, self.b_s_ocr, self.num_to_char, self.textline_light, self.curved_line)

        else:
            ocr_all_textlines = None
            ocr_all_textlines_marginals = None
        self.logger.info("detection of reading order took %.1fs", time.time() - t_order)

        pcgts = self.writer.build_pagexml_no_full_layout(
            txt_con_org, page_coord, order_text_new, id_of_texts_tot,
            all_found_textline_polygons, all_box_coord, polygons_of_images, polygons_of_marginals,
            all_found_textline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals,
            cont_page, polygons_lines_xml, contours_tables, ocr_all_textlines, ocr_all_textlines_marginals, conf_contours_textregions)
        return pcgts


class Eynollah_ocr:
    def __init__(
        self,
        dir_models,
        dir_xmls=None,
        dir_in=None,
        image_filename=None,
        dir_in_bin=None,
        dir_out=None,
        dir_out_image_text=None,
        tr_ocr=False,
        batch_size=None,
        export_textline_images_and_text=False,
        do_not_mask_with_textline_contour=False,
        draw_texts_on_image=False,
        prediction_with_both_of_rgb_and_bin=False,
        pref_of_dataset=None,
        min_conf_value_of_textline_text : Optional[float]=None,
        logger=None,
    ):
        self.dir_in = dir_in
        self.image_filename = image_filename
        self.dir_in_bin = dir_in_bin
        self.dir_out = dir_out
        self.dir_xmls = dir_xmls
        self.dir_models = dir_models
        self.tr_ocr = tr_ocr
        self.export_textline_images_and_text = export_textline_images_and_text
        self.do_not_mask_with_textline_contour = do_not_mask_with_textline_contour
        self.draw_texts_on_image = draw_texts_on_image
        self.dir_out_image_text = dir_out_image_text
        self.prediction_with_both_of_rgb_and_bin = prediction_with_both_of_rgb_and_bin
        self.pref_of_dataset = pref_of_dataset
        self.logger = logger if logger else getLogger('eynollah')
        
        if not export_textline_images_and_text:
            if min_conf_value_of_textline_text:
                self.min_conf_value_of_textline_text = float(min_conf_value_of_textline_text)
            else:
                self.min_conf_value_of_textline_text = 0.3
            if tr_ocr:
                self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model_ocr_dir = dir_models + "/trocr_model_ens_of_3_checkpoints_201124"
                self.model_ocr = VisionEncoderDecoderModel.from_pretrained(self.model_ocr_dir)
                self.model_ocr.to(self.device)
                if not batch_size:
                    self.b_s = 2
                else:
                    self.b_s = int(batch_size)

            else:
                self.model_ocr_dir = dir_models + "/model_eynollah_ocr_cnnrnn_20250805"
                model_ocr = load_model(self.model_ocr_dir , compile=False)
                
                self.prediction_model = tf.keras.models.Model(
                                model_ocr.get_layer(name = "image").input, 
                                model_ocr.get_layer(name = "dense2").output)
                if not batch_size:
                    self.b_s = 8
                else:
                    self.b_s = int(batch_size)
                    
                with open(os.path.join(self.model_ocr_dir, "characters_org.txt"),"r") as config_file:
                    characters = json.load(config_file)
                    
                AUTOTUNE = tf.data.AUTOTUNE

                # Mapping characters to integers.
                char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

                # Mapping integers back to original characters.
                self.num_to_char = StringLookup(
                    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
                )
                self.end_character = len(characters) + 2

    def run(self, overwrite : bool = False):
        if self.dir_in:
            ls_imgs = os.listdir(self.dir_in)
        else:
            ls_imgs = [self.image_filename]
        
        if self.tr_ocr:
            tr_ocr_input_height_and_width = 384
            for ind_img in ls_imgs:
                if self.dir_in:
                    file_name = Path(ind_img).stem
                    dir_img = os.path.join(self.dir_in, ind_img)
                else:
                    file_name = Path(self.image_filename).stem
                    dir_img = self.image_filename
                dir_xml = os.path.join(self.dir_xmls, file_name+'.xml')
                out_file_ocr = os.path.join(self.dir_out, file_name+'.xml')
                
                if os.path.exists(out_file_ocr):
                    if overwrite:
                        self.logger.warning("will overwrite existing output file '%s'", out_file_ocr)
                    else:
                        self.logger.warning("will skip input for existing output file '%s'", out_file_ocr)
                        continue
                    
                img = cv2.imread(dir_img)
                
                if self.draw_texts_on_image:
                    out_image_with_text = os.path.join(self.dir_out_image_text, file_name+'.png')
                    image_text = Image.new("RGB", (img.shape[1], img.shape[0]), "white")
                    draw = ImageDraw.Draw(image_text)
                    total_bb_coordinates = []

                ##file_name = Path(dir_xmls).stem
                tree1 = ET.parse(dir_xml, parser = ET.XMLParser(encoding="utf-8"))
                root1=tree1.getroot()
                alltags=[elem.tag for elem in root1.iter()]
                link=alltags[0].split('}')[0]+'}'

                name_space = alltags[0].split('}')[0]
                name_space = name_space.split('{')[1]

                region_tags=np.unique([x for x in alltags if x.endswith('TextRegion')]) 
                        
                    
                    
                cropped_lines = []
                cropped_lines_region_indexer = []
                cropped_lines_meging_indexing = []

                indexer_text_region = 0
                for nn in root1.iter(region_tags):
                    for child_textregion in nn:
                        if child_textregion.tag.endswith("TextLine"):
                            
                            for child_textlines in child_textregion:
                                if child_textlines.tag.endswith("Coords"):
                                    cropped_lines_region_indexer.append(indexer_text_region)
                                    p_h=child_textlines.attrib['points'].split(' ')
                                    textline_coords =  np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] )
                                    x,y,w,h = cv2.boundingRect(textline_coords)
                                    
                                    if self.draw_texts_on_image:
                                        total_bb_coordinates.append([x,y,w,h])
                                    
                                    h2w_ratio = h/float(w)
                                    
                                    img_poly_on_img = np.copy(img)
                                    mask_poly = np.zeros(img.shape)
                                    mask_poly = cv2.fillPoly(mask_poly, pts=[textline_coords], color=(1, 1, 1))
                                    
                                    mask_poly = mask_poly[y:y+h, x:x+w, :]
                                    img_crop = img_poly_on_img[y:y+h, x:x+w, :]
                                    img_crop[mask_poly==0] = 255
                                    
                                    if h2w_ratio > 0.1:
                                        cropped_lines.append(resize_image(img_crop, tr_ocr_input_height_and_width, tr_ocr_input_height_and_width)  )
                                        cropped_lines_meging_indexing.append(0)
                                    else:
                                        splited_images, _ = return_textlines_split_if_needed(img_crop, None)
                                        #print(splited_images)
                                        if splited_images:
                                            cropped_lines.append(resize_image(splited_images[0], tr_ocr_input_height_and_width, tr_ocr_input_height_and_width))
                                            cropped_lines_meging_indexing.append(1)
                                            cropped_lines.append(resize_image(splited_images[1], tr_ocr_input_height_and_width, tr_ocr_input_height_and_width))
                                            cropped_lines_meging_indexing.append(-1)
                                        else:
                                            cropped_lines.append(img_crop)
                                            cropped_lines_meging_indexing.append(0)
                    indexer_text_region = indexer_text_region +1
        
        
                extracted_texts = []
                n_iterations  = math.ceil(len(cropped_lines) / self.b_s) 

                for i in range(n_iterations):
                    if i==(n_iterations-1):
                        n_start = i*self.b_s
                        imgs = cropped_lines[n_start:]
                    else:
                        n_start = i*self.b_s
                        n_end = (i+1)*self.b_s
                        imgs = cropped_lines[n_start:n_end]
                    pixel_values_merged = self.processor(imgs, return_tensors="pt").pixel_values
                    generated_ids_merged = self.model_ocr.generate(pixel_values_merged.to(self.device))
                    generated_text_merged = self.processor.batch_decode(generated_ids_merged, skip_special_tokens=True)
                    
                    extracted_texts = extracted_texts + generated_text_merged
                    
                del cropped_lines
                gc.collect()

                extracted_texts_merged = [extracted_texts[ind]  if cropped_lines_meging_indexing[ind]==0 else extracted_texts[ind]+" "+extracted_texts[ind+1] if cropped_lines_meging_indexing[ind]==1 else None for ind in range(len(cropped_lines_meging_indexing))]

                extracted_texts_merged = [ind for ind in extracted_texts_merged if ind is not None]
                #print(extracted_texts_merged, len(extracted_texts_merged))

                unique_cropped_lines_region_indexer = np.unique(cropped_lines_region_indexer)
                
                if self.draw_texts_on_image:
                    
                    font_path = "Charis-7.000/Charis-Regular.ttf"  # Make sure this file exists!
                    font = ImageFont.truetype(font_path, 40)
                    
                    for indexer_text, bb_ind in enumerate(total_bb_coordinates):
                        
                        
                        x_bb = bb_ind[0]
                        y_bb = bb_ind[1]
                        w_bb = bb_ind[2]
                        h_bb = bb_ind[3]
                        
                        font = fit_text_single_line(draw, extracted_texts_merged[indexer_text], font_path, w_bb, int(h_bb*0.4) )
                        
                        ##draw.rectangle([x_bb, y_bb, x_bb + w_bb, y_bb + h_bb], outline="red", width=2)
                        
                        text_bbox = draw.textbbox((0, 0), extracted_texts_merged[indexer_text], font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]

                        text_x = x_bb + (w_bb - text_width) // 2  # Center horizontally
                        text_y = y_bb + (h_bb - text_height) // 2  # Center vertically

                        # Draw the text
                        draw.text((text_x, text_y), extracted_texts_merged[indexer_text], fill="black", font=font)
                    image_text.save(out_image_with_text)

                #print(len(unique_cropped_lines_region_indexer), 'unique_cropped_lines_region_indexer')
                text_by_textregion = []
                for ind in unique_cropped_lines_region_indexer:
                    extracted_texts_merged_un = np.array(extracted_texts_merged)[np.array(cropped_lines_region_indexer)==ind]
                    
                    text_by_textregion.append(" ".join(extracted_texts_merged_un))
                    
                #print(len(text_by_textregion) , indexer_text_region, "text_by_textregion")


                #print(time.time() - t0 ,'elapsed time')


                indexer = 0
                indexer_textregion = 0
                for nn in root1.iter(region_tags):
                    text_subelement_textregion = ET.SubElement(nn, 'TextEquiv')
                    unicode_textregion = ET.SubElement(text_subelement_textregion, 'Unicode')

                    
                    has_textline = False
                    for child_textregion in nn:
                        if child_textregion.tag.endswith("TextLine"):
                            text_subelement = ET.SubElement(child_textregion, 'TextEquiv')
                            unicode_textline = ET.SubElement(text_subelement, 'Unicode')
                            unicode_textline.text = extracted_texts_merged[indexer]
                            indexer = indexer + 1
                            has_textline = True
                    if has_textline:
                        unicode_textregion.text = text_by_textregion[indexer_textregion]
                        indexer_textregion = indexer_textregion + 1
                        


                ET.register_namespace("",name_space)
                tree1.write(out_file_ocr,xml_declaration=True,method='xml',encoding="utf8",default_namespace=None)
                #print("Job done in %.1fs", time.time() - t0)
        else:
            ###max_len = 280#512#280#512
            ###padding_token = 1500#299#1500#299
            image_width = 512#max_len * 4
            image_height = 32


            img_size=(image_width, image_height)
            
            for ind_img in ls_imgs:
                if self.dir_in:
                    file_name = Path(ind_img).stem
                    dir_img = os.path.join(self.dir_in, ind_img)
                else:
                    file_name = Path(self.image_filename).stem
                    dir_img = self.image_filename
                    
                #file_name = Path(ind_img).stem
                #dir_img = os.path.join(self.dir_in, ind_img)
                dir_xml = os.path.join(self.dir_xmls, file_name+'.xml')
                out_file_ocr = os.path.join(self.dir_out, file_name+'.xml')
                
                if os.path.exists(out_file_ocr):
                    if overwrite:
                        self.logger.warning("will overwrite existing output file '%s'", out_file_ocr)
                    else:
                        self.logger.warning("will skip input for existing output file '%s'", out_file_ocr)
                        continue
                
                img = cv2.imread(dir_img)
                if self.prediction_with_both_of_rgb_and_bin:
                    cropped_lines_bin = []
                    dir_img_bin = os.path.join(self.dir_in_bin, file_name+'.png')
                    img_bin = cv2.imread(dir_img_bin)
                
                if self.draw_texts_on_image:
                    out_image_with_text = os.path.join(self.dir_out_image_text, file_name+'.png')
                    image_text = Image.new("RGB", (img.shape[1], img.shape[0]), "white")
                    draw = ImageDraw.Draw(image_text)
                    total_bb_coordinates = []

                tree1 = ET.parse(dir_xml, parser = ET.XMLParser(encoding="utf-8"))
                root1=tree1.getroot()
                alltags=[elem.tag for elem in root1.iter()]
                link=alltags[0].split('}')[0]+'}'

                name_space = alltags[0].split('}')[0]
                name_space = name_space.split('{')[1]

                region_tags=np.unique([x for x in alltags if x.endswith('TextRegion')]) 
                    
                cropped_lines = []
                cropped_lines_ver_index = []
                cropped_lines_region_indexer = []
                cropped_lines_meging_indexing = []
                
                tinl = time.time()
                indexer_text_region = 0
                indexer_textlines = 0
                for nn in root1.iter(region_tags):
                    try:
                        type_textregion = nn.attrib['type']
                    except:
                        type_textregion = 'paragraph'
                    for child_textregion in nn:
                        if child_textregion.tag.endswith("TextLine"):
                            for child_textlines in child_textregion:
                                if child_textlines.tag.endswith("Coords"):
                                    cropped_lines_region_indexer.append(indexer_text_region)
                                    p_h=child_textlines.attrib['points'].split(' ')
                                    textline_coords =  np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] )
                                    
                                    x,y,w,h = cv2.boundingRect(textline_coords)
                                    
                                    angle_radians = math.atan2(h, w)
                                    # Convert to degrees
                                    angle_degrees = math.degrees(angle_radians)
                                    if type_textregion=='drop-capital':
                                        angle_degrees = 0
                                        
                                    if self.draw_texts_on_image:
                                        total_bb_coordinates.append([x,y,w,h])
                                       
                                    w_scaled = w *  image_height/float(h)
                                    
                                    img_poly_on_img = np.copy(img)
                                    if self.prediction_with_both_of_rgb_and_bin:
                                        img_poly_on_img_bin = np.copy(img_bin)
                                        img_crop_bin = img_poly_on_img_bin[y:y+h, x:x+w, :]
                                    
                                    mask_poly = np.zeros(img.shape)
                                    mask_poly = cv2.fillPoly(mask_poly, pts=[textline_coords], color=(1, 1, 1))
                                    
                                    
                                    mask_poly = mask_poly[y:y+h, x:x+w, :]
                                    img_crop = img_poly_on_img[y:y+h, x:x+w, :]
                                    
                                    if self.export_textline_images_and_text:
                                        if not self.do_not_mask_with_textline_contour:
                                            img_crop[mask_poly==0] = 255
                                        
                                    else:
                                        #print(file_name, angle_degrees,w*h , mask_poly[:,:,0].sum(),  mask_poly[:,:,0].sum() /float(w*h) , 'didi')
                                        
                                        if angle_degrees > 3:
                                            better_des_slope = get_orientation_moments(textline_coords)
                                            
                                            img_crop = rotate_image_with_padding(img_crop, better_des_slope )
                                            
                                            if self.prediction_with_both_of_rgb_and_bin:
                                                img_crop_bin = rotate_image_with_padding(img_crop_bin, better_des_slope )
                                                
                                            mask_poly = rotate_image_with_padding(mask_poly, better_des_slope )
                                            mask_poly = mask_poly.astype('uint8')
                                            
                                            #new bounding box
                                            x_n, y_n, w_n, h_n = get_contours_and_bounding_boxes(mask_poly[:,:,0])
                                            
                                            mask_poly = mask_poly[y_n:y_n+h_n, x_n:x_n+w_n, :]
                                            img_crop = img_crop[y_n:y_n+h_n, x_n:x_n+w_n, :]
                                                
                                            if not self.do_not_mask_with_textline_contour:
                                                img_crop[mask_poly==0] = 255
                                            
                                            if self.prediction_with_both_of_rgb_and_bin:
                                                img_crop_bin = img_crop_bin[y_n:y_n+h_n, x_n:x_n+w_n, :]
                                                if not self.do_not_mask_with_textline_contour:
                                                    img_crop_bin[mask_poly==0] = 255
                                            
                                            if mask_poly[:,:,0].sum() /float(w_n*h_n) < 0.50 and w_scaled > 90:
                                                if self.prediction_with_both_of_rgb_and_bin:
                                                    img_crop, img_crop_bin = break_curved_line_into_small_pieces_and_then_merge(img_crop, mask_poly, img_crop_bin)
                                                else:
                                                    img_crop, _ = break_curved_line_into_small_pieces_and_then_merge(img_crop, mask_poly)
        
                                                
                                        else:
                                            better_des_slope = 0
                                            if not self.do_not_mask_with_textline_contour:
                                                img_crop[mask_poly==0] = 255
                                            if self.prediction_with_both_of_rgb_and_bin:
                                                if not self.do_not_mask_with_textline_contour:
                                                    img_crop_bin[mask_poly==0] = 255
                                            if type_textregion=='drop-capital':
                                                pass
                                            else:
                                                if mask_poly[:,:,0].sum() /float(w*h) < 0.50 and w_scaled > 90:
                                                    if self.prediction_with_both_of_rgb_and_bin:
                                                        img_crop, img_crop_bin = break_curved_line_into_small_pieces_and_then_merge(img_crop, mask_poly, img_crop_bin)
                                                    else:
                                                        img_crop, _ = break_curved_line_into_small_pieces_and_then_merge(img_crop, mask_poly)
                                    
                                    if not self.export_textline_images_and_text:
                                        if w_scaled < 750:#1.5*image_width:
                                            img_fin = preprocess_and_resize_image_for_ocrcnn_model(img_crop, image_height, image_width)
                                            cropped_lines.append(img_fin)
                                            if abs(better_des_slope) > 45:
                                                cropped_lines_ver_index.append(1)
                                            else:
                                                cropped_lines_ver_index.append(0)
                                                
                                            cropped_lines_meging_indexing.append(0)
                                            if self.prediction_with_both_of_rgb_and_bin:
                                                img_fin = preprocess_and_resize_image_for_ocrcnn_model(img_crop_bin, image_height, image_width)
                                                cropped_lines_bin.append(img_fin)
                                        else:
                                            if self.prediction_with_both_of_rgb_and_bin:
                                                splited_images, splited_images_bin = return_textlines_split_if_needed(img_crop, img_crop_bin, prediction_with_both_of_rgb_and_bin=self.prediction_with_both_of_rgb_and_bin)
                                            else:
                                                splited_images, splited_images_bin = return_textlines_split_if_needed(img_crop, None)
                                            if splited_images:
                                                img_fin = preprocess_and_resize_image_for_ocrcnn_model(splited_images[0], image_height, image_width)
                                                cropped_lines.append(img_fin)
                                                cropped_lines_meging_indexing.append(1)
                                                
                                                if abs(better_des_slope) > 45:
                                                    cropped_lines_ver_index.append(1)
                                                else:
                                                    cropped_lines_ver_index.append(0)
                                                
                                                img_fin = preprocess_and_resize_image_for_ocrcnn_model(splited_images[1], image_height, image_width)
                                                
                                                cropped_lines.append(img_fin)
                                                cropped_lines_meging_indexing.append(-1)
                                                
                                                if abs(better_des_slope) > 45:
                                                    cropped_lines_ver_index.append(1)
                                                else:
                                                    cropped_lines_ver_index.append(0)
                                                
                                                if self.prediction_with_both_of_rgb_and_bin:
                                                    img_fin = preprocess_and_resize_image_for_ocrcnn_model(splited_images_bin[0], image_height, image_width)
                                                    cropped_lines_bin.append(img_fin)
                                                    img_fin = preprocess_and_resize_image_for_ocrcnn_model(splited_images_bin[1], image_height, image_width)
                                                    cropped_lines_bin.append(img_fin)
                                                    
                                            else:
                                                img_fin = preprocess_and_resize_image_for_ocrcnn_model(img_crop, image_height, image_width)
                                                cropped_lines.append(img_fin)
                                                cropped_lines_meging_indexing.append(0)
                                                
                                                if abs(better_des_slope) > 45:
                                                    cropped_lines_ver_index.append(1)
                                                else:
                                                    cropped_lines_ver_index.append(0)
                                                
                                                if self.prediction_with_both_of_rgb_and_bin:
                                                    img_fin = preprocess_and_resize_image_for_ocrcnn_model(img_crop_bin, image_height, image_width)
                                                    cropped_lines_bin.append(img_fin)
                                        
                                if self.export_textline_images_and_text:
                                    if img_crop.shape[0]==0 or img_crop.shape[1]==0:
                                        pass
                                    else:
                                        if child_textlines.tag.endswith("TextEquiv"):
                                            for cheild_text in child_textlines:
                                                if cheild_text.tag.endswith("Unicode"):
                                                    textline_text = cheild_text.text
                                                    if textline_text:
                                                        if self.do_not_mask_with_textline_contour:
                                                            if self.pref_of_dataset:
                                                                with open(os.path.join(self.dir_out, file_name+'_line_'+str(indexer_textlines)+'_'+self.pref_of_dataset+'.txt'), 'w') as text_file:
                                                                    text_file.write(textline_text)

                                                                cv2.imwrite(os.path.join(self.dir_out, file_name+'_line_'+str(indexer_textlines)+'_'+self.pref_of_dataset+'.png'), img_crop )
                                                            else:
                                                                with open(os.path.join(self.dir_out, file_name+'_line_'+str(indexer_textlines)+'.txt'), 'w') as text_file:
                                                                    text_file.write(textline_text)

                                                                cv2.imwrite(os.path.join(self.dir_out, file_name+'_line_'+str(indexer_textlines)+'.png'), img_crop )
                                                        else:
                                                            if self.pref_of_dataset:
                                                                with open(os.path.join(self.dir_out, file_name+'_line_'+str(indexer_textlines)+'_'+self.pref_of_dataset+'_masked.txt'), 'w') as text_file:
                                                                    text_file.write(textline_text)

                                                                cv2.imwrite(os.path.join(self.dir_out, file_name+'_line_'+str(indexer_textlines)+'_'+self.pref_of_dataset+'_masked.png'), img_crop )
                                                            else:
                                                                with open(os.path.join(self.dir_out, file_name+'_line_'+str(indexer_textlines)+'_masked.txt'), 'w') as text_file:
                                                                    text_file.write(textline_text)

                                                                cv2.imwrite(os.path.join(self.dir_out, file_name+'_line_'+str(indexer_textlines)+'_masked.png'), img_crop )
                                                            
                                                    indexer_textlines+=1

                    if not self.export_textline_images_and_text:
                        indexer_text_region = indexer_text_region +1
                    
                if not self.export_textline_images_and_text:
                    extracted_texts = []
                    extracted_conf_value = []

                    n_iterations  = math.ceil(len(cropped_lines) / self.b_s) 

                    for i in range(n_iterations):
                        if i==(n_iterations-1):
                            n_start = i*self.b_s
                            imgs = cropped_lines[n_start:]
                            imgs = np.array(imgs)
                            imgs = imgs.reshape(imgs.shape[0], image_height, image_width, 3)
                            
                            ver_imgs = np.array( cropped_lines_ver_index[n_start:] )
                            indices_ver = np.where(ver_imgs == 1)[0]
                            
                            #print(indices_ver, 'indices_ver')
                            if len(indices_ver)>0:
                                imgs_ver_flipped = imgs[indices_ver, : ,: ,:]
                                imgs_ver_flipped = imgs_ver_flipped[:,::-1,::-1,:]
                                #print(imgs_ver_flipped, 'imgs_ver_flipped')
                                
                            else:
                                imgs_ver_flipped = None
                            
                            if self.prediction_with_both_of_rgb_and_bin:
                                imgs_bin = cropped_lines_bin[n_start:]
                                imgs_bin = np.array(imgs_bin)
                                imgs_bin = imgs_bin.reshape(imgs_bin.shape[0], image_height, image_width, 3)
                                
                                if len(indices_ver)>0:
                                    imgs_bin_ver_flipped = imgs_bin[indices_ver, : ,: ,:]
                                    imgs_bin_ver_flipped = imgs_bin_ver_flipped[:,::-1,::-1,:]
                                    #print(imgs_ver_flipped, 'imgs_ver_flipped')
                                    
                                else:
                                    imgs_bin_ver_flipped = None
                        else:
                            n_start = i*self.b_s
                            n_end = (i+1)*self.b_s
                            imgs = cropped_lines[n_start:n_end]
                            imgs = np.array(imgs).reshape(self.b_s, image_height, image_width, 3)
                            
                            ver_imgs = np.array( cropped_lines_ver_index[n_start:n_end] )
                            indices_ver = np.where(ver_imgs == 1)[0]
                            #print(indices_ver, 'indices_ver')
                            
                            if len(indices_ver)>0:
                                imgs_ver_flipped = imgs[indices_ver, : ,: ,:]
                                imgs_ver_flipped = imgs_ver_flipped[:,::-1,::-1,:]
                                #print(imgs_ver_flipped, 'imgs_ver_flipped')
                            else:
                                imgs_ver_flipped = None

                            
                            if self.prediction_with_both_of_rgb_and_bin:
                                imgs_bin = cropped_lines_bin[n_start:n_end]
                                imgs_bin = np.array(imgs_bin).reshape(self.b_s, image_height, image_width, 3)
                                
                                
                                if len(indices_ver)>0:
                                    imgs_bin_ver_flipped = imgs_bin[indices_ver, : ,: ,:]
                                    imgs_bin_ver_flipped = imgs_bin_ver_flipped[:,::-1,::-1,:]
                                    #print(imgs_ver_flipped, 'imgs_ver_flipped')
                                else:
                                    imgs_bin_ver_flipped = None
                            

                        preds = self.prediction_model.predict(imgs, verbose=0)
                        
                        if len(indices_ver)>0:
                            preds_flipped = self.prediction_model.predict(imgs_ver_flipped, verbose=0)
                            preds_max_fliped = np.max(preds_flipped, axis=2 )
                            preds_max_args_flipped = np.argmax(preds_flipped, axis=2 )
                            pred_max_not_unk_mask_bool_flipped = preds_max_args_flipped[:,:]!=self.end_character
                            masked_means_flipped = np.sum(preds_max_fliped * pred_max_not_unk_mask_bool_flipped, axis=1) / np.sum(pred_max_not_unk_mask_bool_flipped, axis=1)
                            masked_means_flipped[np.isnan(masked_means_flipped)] = 0
                            
                            preds_max = np.max(preds, axis=2 )
                            preds_max_args = np.argmax(preds, axis=2 )
                            pred_max_not_unk_mask_bool = preds_max_args[:,:]!=self.end_character
                            
                            masked_means = np.sum(preds_max * pred_max_not_unk_mask_bool, axis=1) / np.sum(pred_max_not_unk_mask_bool, axis=1)
                            masked_means[np.isnan(masked_means)] = 0
                            
                            masked_means_ver = masked_means[indices_ver]
                            #print(masked_means_ver, 'pred_max_not_unk')
                            
                            indices_where_flipped_conf_value_is_higher = np.where(masked_means_flipped > masked_means_ver)[0]
                            
                            #print(indices_where_flipped_conf_value_is_higher, 'indices_where_flipped_conf_value_is_higher')
                            if len(indices_where_flipped_conf_value_is_higher)>0:
                                indices_to_be_replaced = indices_ver[indices_where_flipped_conf_value_is_higher]
                                preds[indices_to_be_replaced,:,:] = preds_flipped[indices_where_flipped_conf_value_is_higher, :, :]
                        if self.prediction_with_both_of_rgb_and_bin:
                            preds_bin = self.prediction_model.predict(imgs_bin, verbose=0)
                            
                            if len(indices_ver)>0:
                                preds_flipped = self.prediction_model.predict(imgs_bin_ver_flipped, verbose=0)
                                preds_max_fliped = np.max(preds_flipped, axis=2 )
                                preds_max_args_flipped = np.argmax(preds_flipped, axis=2 )
                                pred_max_not_unk_mask_bool_flipped = preds_max_args_flipped[:,:]!=self.end_character
                                masked_means_flipped = np.sum(preds_max_fliped * pred_max_not_unk_mask_bool_flipped, axis=1) / np.sum(pred_max_not_unk_mask_bool_flipped, axis=1)
                                masked_means_flipped[np.isnan(masked_means_flipped)] = 0
                                
                                preds_max = np.max(preds, axis=2 )
                                preds_max_args = np.argmax(preds, axis=2 )
                                pred_max_not_unk_mask_bool = preds_max_args[:,:]!=self.end_character
                                
                                masked_means = np.sum(preds_max * pred_max_not_unk_mask_bool, axis=1) / np.sum(pred_max_not_unk_mask_bool, axis=1)
                                masked_means[np.isnan(masked_means)] = 0
                                
                                masked_means_ver = masked_means[indices_ver]
                                #print(masked_means_ver, 'pred_max_not_unk')
                                
                                indices_where_flipped_conf_value_is_higher = np.where(masked_means_flipped > masked_means_ver)[0]
                                
                                #print(indices_where_flipped_conf_value_is_higher, 'indices_where_flipped_conf_value_is_higher')
                                if len(indices_where_flipped_conf_value_is_higher)>0:
                                    indices_to_be_replaced = indices_ver[indices_where_flipped_conf_value_is_higher]
                                    preds_bin[indices_to_be_replaced,:,:] = preds_flipped[indices_where_flipped_conf_value_is_higher, :, :]
                            
                            preds = (preds + preds_bin) / 2.
                            

                        pred_texts = decode_batch_predictions(preds, self.num_to_char)
                        
                        preds_max = np.max(preds, axis=2 )
                        preds_max_args = np.argmax(preds, axis=2 )
                        pred_max_not_unk_mask_bool = preds_max_args[:,:]!=self.end_character
                        masked_means = np.sum(preds_max * pred_max_not_unk_mask_bool, axis=1) / np.sum(pred_max_not_unk_mask_bool, axis=1)

                        for ib in range(imgs.shape[0]):
                            pred_texts_ib = pred_texts[ib].replace("[UNK]", "")
                            if masked_means[ib] >= self.min_conf_value_of_textline_text:
                                extracted_texts.append(pred_texts_ib)
                                extracted_conf_value.append(masked_means[ib])
                            else:
                                extracted_texts.append("")
                                extracted_conf_value.append(0)
                    del cropped_lines
                    if self.prediction_with_both_of_rgb_and_bin:
                        del cropped_lines_bin
                    gc.collect()
                    
                    extracted_texts_merged = [extracted_texts[ind]  if cropped_lines_meging_indexing[ind]==0 else extracted_texts[ind]+" "+extracted_texts[ind+1] if cropped_lines_meging_indexing[ind]==1 else None for ind in range(len(cropped_lines_meging_indexing))]
                    
                    extracted_conf_value_merged = [extracted_conf_value[ind]  if cropped_lines_meging_indexing[ind]==0 else (extracted_conf_value[ind]+extracted_conf_value[ind+1])/2. if cropped_lines_meging_indexing[ind]==1 else None for ind in range(len(cropped_lines_meging_indexing))]

                    extracted_conf_value_merged = [extracted_conf_value_merged[ind_cfm] for ind_cfm in range(len(extracted_texts_merged)) if extracted_texts_merged[ind_cfm] is not None]
                    extracted_texts_merged = [ind for ind in extracted_texts_merged if ind is not None]
                    unique_cropped_lines_region_indexer = np.unique(cropped_lines_region_indexer)
                    
                    
                    if self.draw_texts_on_image:
                        
                        font_path = "Charis-7.000/Charis-Regular.ttf"  # Make sure this file exists!
                        font = ImageFont.truetype(font_path, 40)
                        
                        for indexer_text, bb_ind in enumerate(total_bb_coordinates):
                            
                            
                            x_bb = bb_ind[0]
                            y_bb = bb_ind[1]
                            w_bb = bb_ind[2]
                            h_bb = bb_ind[3]
                            
                            font = fit_text_single_line(draw, extracted_texts_merged[indexer_text], font_path, w_bb, int(h_bb*0.4) )
                            
                            ##draw.rectangle([x_bb, y_bb, x_bb + w_bb, y_bb + h_bb], outline="red", width=2)
                            
                            text_bbox = draw.textbbox((0, 0), extracted_texts_merged[indexer_text], font=font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]

                            text_x = x_bb + (w_bb - text_width) // 2  # Center horizontally
                            text_y = y_bb + (h_bb - text_height) // 2  # Center vertically

                            # Draw the text
                            draw.text((text_x, text_y), extracted_texts_merged[indexer_text], fill="black", font=font)
                        image_text.save(out_image_with_text)

                    text_by_textregion = []
                    for ind in unique_cropped_lines_region_indexer:
                        extracted_texts_merged_un = np.array(extracted_texts_merged)[np.array(cropped_lines_region_indexer)==ind]
                        if len(extracted_texts_merged_un)>1:
                            text_by_textregion_ind = ""
                            next_glue = ""
                            for indt in range(len(extracted_texts_merged_un)):
                                if extracted_texts_merged_un[indt].endswith('⸗') or extracted_texts_merged_un[indt].endswith('-'):
                                    text_by_textregion_ind = text_by_textregion_ind + next_glue + extracted_texts_merged_un[indt][:-1]
                                    next_glue = ""
                                else:
                                    text_by_textregion_ind = text_by_textregion_ind + next_glue + extracted_texts_merged_un[indt]
                                    next_glue = " "
                            text_by_textregion.append(text_by_textregion_ind)
                                
                        else:
                            text_by_textregion.append(" ".join(extracted_texts_merged_un))
                        #print(text_by_textregion, 'text_by_textregiontext_by_textregiontext_by_textregiontext_by_textregiontext_by_textregion')
                        
                        
                    ###index_tot_regions = []
                    ###tot_region_ref = []

                    ###for jj in root1.iter(link+'RegionRefIndexed'):
                        ###index_tot_regions.append(jj.attrib['index'])
                        ###tot_region_ref.append(jj.attrib['regionRef'])
                        
                    ###id_to_order = {tid: ro for tid, ro in zip(tot_region_ref, index_tot_regions)}
        
                    #id_textregions = []
                    #textregions_by_existing_ids = []
                    indexer = 0
                    indexer_textregion = 0
                    for nn in root1.iter(region_tags):
                        #id_textregion = nn.attrib['id']
                        #id_textregions.append(id_textregion)
                        #textregions_by_existing_ids.append(text_by_textregion[indexer_textregion])
                        
                        is_textregion_text = False
                        for childtest in nn:
                            if childtest.tag.endswith("TextEquiv"):
                                is_textregion_text = True
                        
                        if not is_textregion_text:
                            text_subelement_textregion = ET.SubElement(nn, 'TextEquiv')
                            unicode_textregion = ET.SubElement(text_subelement_textregion, 'Unicode')

                        
                        has_textline = False
                        for child_textregion in nn:
                            if child_textregion.tag.endswith("TextLine"):
                                
                                is_textline_text = False
                                for childtest2 in child_textregion:
                                    if childtest2.tag.endswith("TextEquiv"):
                                        is_textline_text = True
                                
                                
                                if not is_textline_text:
                                    text_subelement = ET.SubElement(child_textregion, 'TextEquiv')
                                    text_subelement.set('conf', f"{extracted_conf_value_merged[indexer]:.2f}")
                                    unicode_textline = ET.SubElement(text_subelement, 'Unicode')
                                    unicode_textline.text = extracted_texts_merged[indexer]
                                else:
                                    for childtest3 in child_textregion:
                                        if childtest3.tag.endswith("TextEquiv"):
                                            for child_uc in childtest3:
                                                if child_uc.tag.endswith("Unicode"):
                                                    childtest3.set('conf', f"{extracted_conf_value_merged[indexer]:.2f}")
                                                    child_uc.text = extracted_texts_merged[indexer]
                                        
                                indexer = indexer + 1
                                has_textline = True
                        if has_textline:
                            if is_textregion_text:
                                for child4 in nn:
                                    if child4.tag.endswith("TextEquiv"):
                                        for childtr_uc in child4:
                                            if childtr_uc.tag.endswith("Unicode"):
                                                childtr_uc.text = text_by_textregion[indexer_textregion]
                            else:
                                unicode_textregion.text = text_by_textregion[indexer_textregion]
                            indexer_textregion = indexer_textregion + 1
                            
                    ###sample_order  = [(id_to_order[tid], text) for tid, text in zip(id_textregions, textregions_by_existing_ids) if tid in id_to_order]
                    
                    ##ordered_texts_sample = [text for _, text in sorted(sample_order)]
                    ##tot_page_text = ' '.join(ordered_texts_sample)
                    
                    ##for page_element in root1.iter(link+'Page'):
                        ##text_page = ET.SubElement(page_element, 'TextEquiv')
                        ##unicode_textpage = ET.SubElement(text_page, 'Unicode')
                        ##unicode_textpage.text = tot_page_text
                    
                    ET.register_namespace("",name_space)
                    tree1.write(out_file_ocr,xml_declaration=True,method='xml',encoding="utf8",default_namespace=None)
                    #print("Job done in %.1fs", time.time() - t0)
