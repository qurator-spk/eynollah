# pylint: disable=no-member,invalid-name,line-too-long,missing-function-docstring
"""
tool to extract table form data from alto xml data
"""

import gc
import math
import os
import sys
import time
import warnings
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count

from lxml import etree as ET
from ocrd_utils import getLogger
import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
from keras import backend as K
from keras.models import load_model
sys.stderr = stderr
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")

from .utils.contour import (
    contours_in_same_horizon,
    filter_contours_area_of_image_interiors,
    filter_contours_area_of_image_tables,
    filter_contours_area_of_image,
    find_contours_mean_y_diff,
    find_new_features_of_contoures,
    get_text_region_boxes_by_given_contours,
    get_textregion_contours_in_org_image,
    return_contours_of_image,
    return_contours_of_interested_region,
    return_contours_of_interested_region_by_min_size,
    return_contours_of_interested_textline,
    return_parent_contours,
    return_contours_of_interested_region_by_size,
)

from .utils.rotate import (
    rotate_image,
    rotate_max_area,
    rotate_max_area_new,
    rotatedRectWithMaxArea,
    rotation_image_new,
    rotation_not_90_func,
    rotation_not_90_func_full_layout,
    rotyate_image_different,
)

from .utils.separate_lines import (
    seperate_lines,
    seperate_lines_new_inside_teils,
    seperate_lines_new_inside_teils2,
    seperate_lines_vertical,
    seperate_lines_vertical_cont,
    textline_contours_postprocessing,
    seperate_lines_new2,
    return_deskew_slop,
)

from .utils.drop_capitals import (
    adhere_drop_capital_region_into_cprresponding_textline,
    filter_small_drop_capitals_from_no_patch_layout
)

from .utils.marginals import get_marginals

from .utils.resize import resize_image

from .utils import (
    boosting_headers_by_longshot_region_segmentation,
    crop_image_inside_box,
    find_features_of_lines,
    find_num_col,
    find_num_col_by_vertical_lines,
    find_num_col_deskew,
    find_num_col_only_image,
    isNaN,
    otsu_copy,
    otsu_copy_binary,
    return_hor_spliter_by_index_for_without_verticals,
    delete_seperator_around,
    return_regions_without_seperators,
    put_drop_out_from_only_drop_model,
    putt_bb_of_drop_capitals_of_model_in_patches_in_layout,
    check_any_text_region_in_model_one_is_main_or_header,
    small_textlines_to_parent_adherence2,
    order_and_id_of_texts,
    order_of_regions,
    implent_law_head_main_not_parallel,
    return_hor_spliter_by_index,
    combine_hor_lines_and_delete_cross_points_and_get_lines_features_back_new,
    return_points_with_boundies,
    find_number_of_columns_in_document,
    return_boxes_of_images_by_order_of_reading_new,
)

from .utils.xml import create_page_xml, add_textequiv
from .utils.pil_cv2 import check_dpi
from .plot import EynollahPlotter

SLOPE_THRESHOLD = 0.13

class eynollah:
    def __init__(
        self,
        image_filename,
        image_filename_stem,
        dir_out,
        dir_models,
        dir_of_cropped_images=None,
        dir_of_layout=None,
        dir_of_deskewed=None,
        dir_of_all=None,
        enable_plotting=False,
        allow_enhancement=False,
        curved_line=False,
        full_layout=False,
        allow_scaling=False,
        headers_off=False
    ):
        self.image_filename = image_filename  # XXX This does not seem to be a directory as the name suggests, but a file
        self.cont_page = []
        self.dir_out = dir_out
        self.image_filename_stem = image_filename_stem
        self.allow_enhancement = allow_enhancement
        self.curved_line = curved_line
        self.full_layout = full_layout
        self.allow_scaling = allow_scaling
        self.headers_off = headers_off
        if not self.image_filename_stem:
            self.image_filename_stem = Path(Path(image_filename).name).stem
        self.plotter = None if not enable_plotting else EynollahPlotter(
            dir_of_all=dir_of_all,
            dir_of_deskewed=dir_of_deskewed,
            dir_of_cropped_images=dir_of_cropped_images,
            dir_of_layout=dir_of_layout,
            image_filename=image_filename,
            image_filename_stem=image_filename_stem,
        )
        self.logger = getLogger('eynollah')
        self.dir_models = dir_models
        self.kernel = np.ones((5, 5), np.uint8)

        self.model_dir_of_enhancemnet = dir_models + "/model_enhancement.h5"
        self.model_dir_of_col_classifier = dir_models + "/model_scale_classifier.h5"
        self.model_region_dir_p = dir_models + "/model_main_covid19_lr5-5_scale_1_1_great.h5"
        self.model_region_dir_p2 = dir_models + "/model_main_home_corona3_rot.h5"
        self.model_region_dir_fully_np = dir_models + "/model_no_patches_class0_30eopch.h5"
        self.model_region_dir_fully = dir_models + "/model_3up_new_good_no_augmentation.h5"
        self.model_page_dir = dir_models + "/model_page_mixed_best.h5"
        self.model_region_dir_p_ens = dir_models + "/model_ensemble_s.h5"
        self.model_textline_dir = dir_models + "/model_textline_newspapers.h5" 

        self._imgs = {}

    def imread(self, grayscale=False, uint8=True):
        key = 'img'
        if grayscale:
            key += '_grayscale'
        if uint8:
            key += '_uint8'
        if key not in self._imgs:
            if grayscale:
                img = cv2.imread(self.image_filename, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(self.image_filename)
            if uint8:
                img = img.astype(np.uint8)
            self._imgs[key] = img
        return self._imgs[key].copy()

    def predict_enhancement(self, img):
        self.logger.debug("enter predict_enhancement")
        model_enhancement, session_enhancemnet = self.start_new_session_and_model(self.model_dir_of_enhancemnet)

        img_height_model = model_enhancement.layers[len(model_enhancement.layers) - 1].output_shape[1]
        img_width_model = model_enhancement.layers[len(model_enhancement.layers) - 1].output_shape[2]
        # n_classes = model_enhancement.layers[len(model_enhancement.layers) - 1].output_shape[3]
        if img.shape[0] < img_height_model:
            img = cv2.resize(img, (img.shape[1], img_width_model), interpolation=cv2.INTER_NEAREST)

        if img.shape[1] < img_width_model:
            img = cv2.resize(img, (img_height_model, img.shape[0]), interpolation=cv2.INTER_NEAREST)

        margin = True

        if margin:
            kernel = np.ones((5, 5), np.uint8)

            margin = int(0 * img_width_model)

            width_mid = img_width_model - 2 * margin
            height_mid = img_height_model - 2 * margin

            img = img / float(255.0)

            img_h = img.shape[0]
            img_w = img.shape[1]

            prediction_true = np.zeros((img_h, img_w, 3))
            mask_true = np.zeros((img_h, img_w))
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

                    img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
                    label_p_pred = model_enhancement.predict(img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))

                    seg = label_p_pred[0, :, :, :]
                    seg = seg * 255

                    if i == 0 and j == 0:
                        seg = seg[0 : seg.shape[0] - margin, 0 : seg.shape[1] - margin]
                        prediction_true[index_y_d + 0 : index_y_u - margin, index_x_d + 0 : index_x_u - margin, :] = seg
                    elif i == nxf - 1 and j == nyf - 1:
                        seg = seg[margin : seg.shape[0] - 0, margin : seg.shape[1] - 0]
                        prediction_true[index_y_d + margin : index_y_u - 0, index_x_d + margin : index_x_u - 0, :] = seg
                    elif i == 0 and j == nyf - 1:
                        seg = seg[margin : seg.shape[0] - 0, 0 : seg.shape[1] - margin]
                        prediction_true[index_y_d + margin : index_y_u - 0, index_x_d + 0 : index_x_u - margin, :] = seg
                    elif i == nxf - 1 and j == 0:
                        seg = seg[0 : seg.shape[0] - margin, margin : seg.shape[1] - 0]
                        prediction_true[index_y_d + 0 : index_y_u - margin, index_x_d + margin : index_x_u - 0, :] = seg
                    elif i == 0 and j != 0 and j != nyf - 1:
                        seg = seg[margin : seg.shape[0] - margin, 0 : seg.shape[1] - margin]
                        prediction_true[index_y_d + margin : index_y_u - margin, index_x_d + 0 : index_x_u - margin, :] = seg
                    elif i == nxf - 1 and j != 0 and j != nyf - 1:
                        seg = seg[margin : seg.shape[0] - margin, margin : seg.shape[1] - 0]
                        prediction_true[index_y_d + margin : index_y_u - margin, index_x_d + margin : index_x_u - 0, :] = seg
                    elif i != 0 and i != nxf - 1 and j == 0:
                        seg = seg[0 : seg.shape[0] - margin, margin : seg.shape[1] - margin]
                        prediction_true[index_y_d + 0 : index_y_u - margin, index_x_d + margin : index_x_u - margin, :] = seg
                    elif i != 0 and i != nxf - 1 and j == nyf - 1:
                        seg = seg[margin : seg.shape[0] - 0, margin : seg.shape[1] - margin]
                        prediction_true[index_y_d + margin : index_y_u - 0, index_x_d + margin : index_x_u - margin, :] = seg
                    else:
                        seg = seg[margin : seg.shape[0] - margin, margin : seg.shape[1] - margin]
                        prediction_true[index_y_d + margin : index_y_u - margin, index_x_d + margin : index_x_u - margin, :] = seg

            prediction_true = prediction_true.astype(int)

            del model_enhancement
            del session_enhancemnet

            return prediction_true

    def calculate_width_height_by_columns(self, img, num_col, width_early, label_p_pred):
        self.logger.debug("enter calculate_width_height_by_columns")
        if num_col == 1 and width_early < 1100:
            img_w_new = 2000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 2000)
        elif num_col == 1 and width_early >= 2500:
            img_w_new = 2000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 2000)
        elif num_col == 1 and width_early >= 1100 and width_early < 2500:
            img_w_new = width_early
            img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)
        elif num_col == 2 and width_early < 2000:
            img_w_new = 2400
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 2400)
        elif num_col == 2 and width_early >= 3500:
            img_w_new = 2400
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 2400)
        elif num_col == 2 and width_early >= 2000 and width_early < 3500:
            img_w_new = width_early
            img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)
        elif num_col == 3 and width_early < 2000:
            img_w_new = 3000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 3000)
        elif num_col == 3 and width_early >= 4000:
            img_w_new = 3000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 3000)
        elif num_col == 3 and width_early >= 2000 and width_early < 4000:
            img_w_new = width_early
            img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)
        elif num_col == 4 and width_early < 2500:
            img_w_new = 4000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 4000)
        elif num_col == 4 and width_early >= 5000:
            img_w_new = 4000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 4000)
        elif num_col == 4 and width_early >= 2500 and width_early < 5000:
            img_w_new = width_early
            img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)
        elif num_col == 5 and width_early < 3700:
            img_w_new = 5000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 5000)
        elif num_col == 5 and width_early >= 7000:
            img_w_new = 5000
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 5000)
        elif num_col == 5 and width_early >= 3700 and width_early < 7000:
            img_w_new = width_early
            img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)
        elif num_col == 6 and width_early < 4500:
            img_w_new = 6500  # 5400
            img_h_new = int(img.shape[0] / float(img.shape[1]) * 6500)
        else:
            img_w_new = width_early
            img_h_new = int(img.shape[0] / float(img.shape[1]) * width_early)

        if label_p_pred[0][int(num_col - 1)] < 0.9 and img_w_new < width_early:
            img_new = np.copy(img)
            num_column_is_classified = False
        else:
            img_new = resize_image(img, img_h_new, img_w_new)
            num_column_is_classified = True

        return img_new, num_column_is_classified

    def resize_image_with_column_classifier(self, is_image_enhanced):
        self.logger.debug("enter resize_image_with_column_classifier")
        img = self.imread()

        _, page_coord = self.early_page_for_num_of_column_classification()
        model_num_classifier, session_col_classifier = self.start_new_session_and_model(self.model_dir_of_col_classifier)

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

        label_p_pred = model_num_classifier.predict(img_in)
        num_col = np.argmax(label_p_pred[0]) + 1

        self.logger.info("Found %s columns (%s)", num_col, label_p_pred)

        session_col_classifier.close()
        del model_num_classifier
        del session_col_classifier

        K.clear_session()
        gc.collect()

        img_new, num_column_is_classified = self.calculate_width_height_by_columns(img, num_col, width_early, label_p_pred)

        if img_new.shape[1] > img.shape[1]:
            img_new = self.predict_enhancement(img_new)
            is_image_enhanced = True

        return img, img_new, is_image_enhanced

    def resize_and_enhance_image_with_column_classifier(self):
        self.logger.debug("enter resize_and_enhance_image_with_column_classifier")
        dpi = check_dpi(self.image_filename)
        self.logger.info("Detected %s DPI" % dpi)
        img = self.imread()

        _, page_coord = self.early_page_for_num_of_column_classification()
        model_num_classifier, session_col_classifier = self.start_new_session_and_model(self.model_dir_of_col_classifier)

        img_1ch = self.imread(grayscale=True)

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

        # plt.imshow(img_in[0,:,:,:])
        # plt.show()

        label_p_pred = model_num_classifier.predict(img_in)
        num_col = np.argmax(label_p_pred[0]) + 1

        self.logger.info("Found %s columns (%s)", num_col, label_p_pred)

        session_col_classifier.close()
        del model_num_classifier
        del session_col_classifier
        del img_in
        del img_1ch
        del page_coord
        K.clear_session()
        gc.collect()

        if dpi < 298:
            img_new, num_column_is_classified = self.calculate_width_height_by_columns(img, num_col, width_early, label_p_pred)
            image_res = self.predict_enhancement(img_new)
            is_image_enhanced = True
        else:
            is_image_enhanced = False
            num_column_is_classified = True
            image_res = np.copy(img)

        self.logger.debug("exit resize_and_enhance_image_with_column_classifier")
        return is_image_enhanced, img, image_res, num_col, num_column_is_classified

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
        # XXX TODO hacky
        if self.plotter:
            self.plotter.image_org = self.image_org
            self.plotter.scale_y = self.scale_y
            self.plotter.scale_x = self.scale_x


    def get_image_and_scales_after_enhancing(self, img_org, img_res):
        self.logger.debug("enter get_image_and_scales_after_enhancing")
        self.image = np.copy(img_res)
        self.image = self.image.astype(np.uint8)
        self.image_org = np.copy(img_org)
        self.height_org = self.image_org.shape[0]
        self.width_org = self.image_org.shape[1]

        self.scale_y = img_res.shape[0] / float(self.image_org.shape[0])
        self.scale_x = img_res.shape[1] / float(self.image_org.shape[1])
        

        del img_org
        del img_res

    def start_new_session_and_model(self, model_dir):
        self.logger.debug("enter start_new_session_and_model (model_dir=%s)", model_dir)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        session = tf.InteractiveSession()
        model = load_model(model_dir, compile=False)

        return model, session

    def do_prediction(self, patches, img, model, marginal_of_patch_percent=0.1):
        self.logger.debug("enter do_prediction")

        img_height_model = model.layers[len(model.layers) - 1].output_shape[1]
        img_width_model = model.layers[len(model.layers) - 1].output_shape[2]
        n_classes = model.layers[len(model.layers) - 1].output_shape[3]


        if not patches:
            img_h_page = img.shape[0]
            img_w_page = img.shape[1]
            img = img / float(255.0)
            img = resize_image(img, img_height_model, img_width_model)

            label_p_pred = model.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))

            seg = np.argmax(label_p_pred, axis=3)[0]
            seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = resize_image(seg_color, img_h_page, img_w_page)
            prediction_true = prediction_true.astype(np.uint8)

            del img
            del seg_color
            del label_p_pred
            del seg

        else:
            if img.shape[0] < img_height_model:
                img = resize_image(img, img_height_model, img.shape[1])

            if img.shape[1] < img_width_model:
                img = resize_image(img, img.shape[0], img_width_model)

            self.logger.info("Image dimensions: %sx%s", img_height_model, img_width_model)
            margin = int(marginal_of_patch_percent * img_height_model)
            width_mid = img_width_model - 2 * margin
            height_mid = img_height_model - 2 * margin
            img = img / float(255.0)
            img = img.astype(np.float16)
            img_h = img.shape[0]
            img_w = img.shape[1]
            prediction_true = np.zeros((img_h, img_w, 3))
            mask_true = np.zeros((img_h, img_w))
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

                    img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
                    label_p_pred = model.predict(img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]))
                    seg = np.argmax(label_p_pred, axis=3)[0]
                    seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)

                    if i == 0 and j == 0:
                        seg_color = seg_color[0 : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                        seg = seg[0 : seg.shape[0] - margin, 0 : seg.shape[1] - margin]
                        mask_true[index_y_d + 0 : index_y_u - margin, index_x_d + 0 : index_x_u - margin] = seg
                        prediction_true[index_y_d + 0 : index_y_u - margin, index_x_d + 0 : index_x_u - margin, :] = seg_color
                    elif i == nxf - 1 and j == nyf - 1:
                        seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - 0, :]
                        seg = seg[margin : seg.shape[0] - 0, margin : seg.shape[1] - 0]
                        mask_true[index_y_d + margin : index_y_u - 0, index_x_d + margin : index_x_u - 0] = seg
                        prediction_true[index_y_d + margin : index_y_u - 0, index_x_d + margin : index_x_u - 0, :] = seg_color
                    elif i == 0 and j == nyf - 1:
                        seg_color = seg_color[margin : seg_color.shape[0] - 0, 0 : seg_color.shape[1] - margin, :]
                        seg = seg[margin : seg.shape[0] - 0, 0 : seg.shape[1] - margin]
                        mask_true[index_y_d + margin : index_y_u - 0, index_x_d + 0 : index_x_u - margin] = seg
                        prediction_true[index_y_d + margin : index_y_u - 0, index_x_d + 0 : index_x_u - margin, :] = seg_color
                    elif i == nxf - 1 and j == 0:
                        seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                        seg = seg[0 : seg.shape[0] - margin, margin : seg.shape[1] - 0]
                        mask_true[index_y_d + 0 : index_y_u - margin, index_x_d + margin : index_x_u - 0] = seg
                        prediction_true[index_y_d + 0 : index_y_u - margin, index_x_d + margin : index_x_u - 0, :] = seg_color
                    elif i == 0 and j != 0 and j != nyf - 1:
                        seg_color = seg_color[margin : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                        seg = seg[margin : seg.shape[0] - margin, 0 : seg.shape[1] - margin]
                        mask_true[index_y_d + margin : index_y_u - margin, index_x_d + 0 : index_x_u - margin] = seg
                        prediction_true[index_y_d + margin : index_y_u - margin, index_x_d + 0 : index_x_u - margin, :] = seg_color
                    elif i == nxf - 1 and j != 0 and j != nyf - 1:
                        seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                        seg = seg[margin : seg.shape[0] - margin, margin : seg.shape[1] - 0]
                        mask_true[index_y_d + margin : index_y_u - margin, index_x_d + margin : index_x_u - 0] = seg
                        prediction_true[index_y_d + margin : index_y_u - margin, index_x_d + margin : index_x_u - 0, :] = seg_color
                    elif i != 0 and i != nxf - 1 and j == 0:
                        seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                        seg = seg[0 : seg.shape[0] - margin, margin : seg.shape[1] - margin]
                        mask_true[index_y_d + 0 : index_y_u - margin, index_x_d + margin : index_x_u - margin] = seg
                        prediction_true[index_y_d + 0 : index_y_u - margin, index_x_d + margin : index_x_u - margin, :] = seg_color
                    elif i != 0 and i != nxf - 1 and j == nyf - 1:
                        seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - margin, :]
                        seg = seg[margin : seg.shape[0] - 0, margin : seg.shape[1] - margin]
                        mask_true[index_y_d + margin : index_y_u - 0, index_x_d + margin : index_x_u - margin] = seg
                        prediction_true[index_y_d + margin : index_y_u - 0, index_x_d + margin : index_x_u - margin, :] = seg_color
                    else:
                        seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                        seg = seg[margin : seg.shape[0] - margin, margin : seg.shape[1] - margin]
                        mask_true[index_y_d + margin : index_y_u - margin, index_x_d + margin : index_x_u - margin] = seg
                        prediction_true[index_y_d + margin : index_y_u - margin, index_x_d + margin : index_x_u - margin, :] = seg_color

            prediction_true = prediction_true.astype(np.uint8)
            del img
            del mask_true
            del seg_color
            del seg
            del img_patch
        gc.collect()
        return prediction_true

    def early_page_for_num_of_column_classification(self):
        self.logger.debug("enter early_page_for_num_of_column_classification")
        img = self.imread()
        model_page, session_page = self.start_new_session_and_model(self.model_page_dir)
        for ii in range(1):
            img = cv2.GaussianBlur(img, (5, 5), 0)

        img_page_prediction = self.do_prediction(False, img, model_page)

        imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)
        thresh = cv2.dilate(thresh, self.kernel, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])
        cnt = contours[np.argmax(cnt_size)]
        x, y, w, h = cv2.boundingRect(cnt)
        box = [x, y, w, h]
        croped_page, page_coord = crop_image_inside_box(box, img)
        session_page.close()
        del model_page
        del session_page
        del contours
        del thresh
        del img
        del cnt_size
        del cnt
        del box
        del x
        del y
        del w
        del h
        del imgray
        del img_page_prediction

        gc.collect()
        self.logger.debug("exit early_page_for_num_of_column_classification")
        return croped_page, page_coord

    def extract_page(self):
        self.logger.debug("enter extract_page")
        model_page, session_page = self.start_new_session_and_model(self.model_page_dir)
        for ii in range(1):
            img = cv2.GaussianBlur(self.image, (5, 5), 0)

        img_page_prediction = self.do_prediction(False, img, model_page)

        imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        thresh = cv2.dilate(thresh, self.kernel, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])
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
        croped_page, page_coord = crop_image_inside_box(box, self.image)

        self.cont_page.append(np.array([[page_coord[2], page_coord[0]], [page_coord[3], page_coord[0]], [page_coord[3], page_coord[1]], [page_coord[2], page_coord[1]]]))

        session_page.close()
        del model_page
        del session_page
        del contours
        del thresh
        del img
        del imgray

        K.clear_session()
        gc.collect()
        self.logger.debug("exit extract_page")
        return croped_page, page_coord

    def extract_text_regions(self, img, patches, cols):
        self.logger.debug("enter extract_text_regions")
        img_height_h = img.shape[0]
        img_width_h = img.shape[1]

        model_region, session_region = self.start_new_session_and_model(self.model_region_dir_fully if patches else self.model_region_dir_fully_np)

        if not patches:
            img = otsu_copy_binary(img)
            img = img.astype(np.uint8)
            prediction_regions2 = None
        else:
            if cols == 1:
                img2 = otsu_copy_binary(img)
                img2 = img2.astype(np.uint8)
                img2 = resize_image(img2, int(img_height_h * 0.7), int(img_width_h * 0.7))
                marginal_of_patch_percent = 0.1
                prediction_regions2 = self.do_prediction(patches, img2, model_region, marginal_of_patch_percent)
                prediction_regions2 = resize_image(prediction_regions2, img_height_h, img_width_h)

            if cols == 2:
                img2 = otsu_copy_binary(img)
                img2 = img2.astype(np.uint8)
                img2 = resize_image(img2, int(img_height_h * 0.4), int(img_width_h * 0.4))
                marginal_of_patch_percent = 0.1
                prediction_regions2 = self.do_prediction(patches, img2, model_region, marginal_of_patch_percent)
                prediction_regions2 = resize_image(prediction_regions2, img_height_h, img_width_h)

            elif cols > 2:
                img2 = otsu_copy_binary(img)
                img2 = img2.astype(np.uint8)
                img2 = resize_image(img2, int(img_height_h * 0.3), int(img_width_h * 0.3))
                marginal_of_patch_percent = 0.1
                prediction_regions2 = self.do_prediction(patches, img2, model_region, marginal_of_patch_percent)
                prediction_regions2 = resize_image(prediction_regions2, img_height_h, img_width_h)

            if cols == 2:
                img = otsu_copy_binary(img)
                img = img.astype(np.uint8)
                if img_width_h >= 2000:
                    img = resize_image(img, int(img_height_h * 0.9), int(img_width_h * 0.9))
                img = img.astype(np.uint8)

            if cols == 1:
                img = otsu_copy_binary(img)
                img = img.astype(np.uint8)
                img = resize_image(img, int(img_height_h * 0.5), int(img_width_h * 0.5))
                img = img.astype(np.uint8)

            if cols == 3:
                if (self.scale_x == 1 and img_width_h > 3000) or (self.scale_x != 1 and img_width_h > 2800):
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)
                    img = resize_image(img, int(img_height_h * 2800 / float(img_width_h)), 2800)
                else:
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)

            if cols == 4:
                if (self.scale_x == 1 and img_width_h > 4000) or (self.scale_x != 1 and img_width_h > 3700):
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)
                    img= resize_image(img, int(img_height_h * 3700 / float(img_width_h)), 3700)
                else:
                    img = otsu_copy_binary(img)#self.otsu_copy(img)
                    img = img.astype(np.uint8)
                    img= resize_image(img, int(img_height_h * 0.9), int(img_width_h * 0.9))

            if cols == 5:
                if self.scale_x == 1 and img_width_h > 5000:
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)
                    img= resize_image(img, int(img_height_h * 0.7), int(img_width_h * 0.7))
                else:
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)
                    img= resize_image(img, int(img_height_h * 0.9), int(img_width_h * 0.9) )

            if cols >= 6:
                if img_width_h > 5600:
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)
                    img= resize_image(img, int(img_height_h * 5600 / float(img_width_h)), 5600)
                else:
                    img = otsu_copy_binary(img)
                    img = img.astype(np.uint8)
                    img= resize_image(img, int(img_height_h * 0.9), int(img_width_h * 0.9))

        marginal_of_patch_percent = 0.1
        prediction_regions = self.do_prediction(patches, img, model_region, marginal_of_patch_percent)
        prediction_regions = resize_image(prediction_regions, img_height_h, img_width_h)

        session_region.close()
        del model_region
        del session_region
        del img
        gc.collect()
        self.logger.debug("exit extract_text_regions")
        return prediction_regions, prediction_regions2

    def get_slopes_and_deskew_new(self, contours, contours_par, textline_mask_tot, image_page_rotated, boxes, slope_deskew):
        self.logger.debug("enter get_slopes_and_deskew_new")
        num_cores = cpu_count()
        queue_of_all_params = Queue()

        processes = []
        nh = np.linspace(0, len(boxes), num_cores + 1)
        indexes_by_text_con = np.array(range(len(contours_par)))

        for i in range(num_cores):
            boxes_per_process = boxes[int(nh[i]) : int(nh[i + 1])]
            contours_per_process = contours[int(nh[i]) : int(nh[i + 1])]
            contours_par_per_process = contours_par[int(nh[i]) : int(nh[i + 1])]
            indexes_text_con_per_process = indexes_by_text_con[int(nh[i]) : int(nh[i + 1])]

            processes.append(Process(target=self.do_work_of_slopes_new, args=(queue_of_all_params, boxes_per_process, textline_mask_tot, contours_per_process, contours_par_per_process, indexes_text_con_per_process, image_page_rotated, slope_deskew)))

        for i in range(num_cores):
            processes[i].start()

        slopes = []
        all_found_texline_polygons = []
        all_found_text_regions = []
        all_found_text_regions_par = []
        boxes = []
        all_box_coord = []
        all_index_text_con = []

        for i in range(num_cores):
            list_all_par = queue_of_all_params.get(True)
            slopes_for_sub_process = list_all_par[0]
            polys_for_sub_process = list_all_par[1]
            boxes_for_sub_process = list_all_par[2]
            contours_for_subprocess = list_all_par[3]
            contours_par_for_subprocess = list_all_par[4]
            boxes_coord_for_subprocess = list_all_par[5]
            indexes_for_subprocess = list_all_par[6]
            for j in range(len(slopes_for_sub_process)):
                slopes.append(slopes_for_sub_process[j])
                all_found_texline_polygons.append(polys_for_sub_process[j])
                boxes.append(boxes_for_sub_process[j])
                all_found_text_regions.append(contours_for_subprocess[j])
                all_found_text_regions_par.append(contours_par_for_subprocess[j])
                all_box_coord.append(boxes_coord_for_subprocess[j])
                all_index_text_con.append(indexes_for_subprocess[j])

        for i in range(num_cores):
            processes[i].join()
        self.logger.debug('slopes %s', slopes)
        self.logger.debug("exit get_slopes_and_deskew_new")
        return slopes, all_found_texline_polygons, boxes, all_found_text_regions, all_found_text_regions_par, all_box_coord, all_index_text_con

    def get_slopes_and_deskew_new_curved(self, contours, contours_par, textline_mask_tot, image_page_rotated, boxes, mask_texts_only, num_col, scale_par, slope_deskew):
        self.logger.debug("enter get_slopes_and_deskew_new_curved")
        num_cores = cpu_count()
        queue_of_all_params = Queue()

        processes = []
        nh = np.linspace(0, len(boxes), num_cores + 1)
        indexes_by_text_con = np.array(range(len(contours_par)))

        for i in range(num_cores):
            boxes_per_process = boxes[int(nh[i]) : int(nh[i + 1])]
            contours_per_process = contours[int(nh[i]) : int(nh[i + 1])]
            contours_par_per_process = contours_par[int(nh[i]) : int(nh[i + 1])]
            indexes_text_con_per_process = indexes_by_text_con[int(nh[i]) : int(nh[i + 1])]

            processes.append(Process(target=self.do_work_of_slopes_new_curved, args=(queue_of_all_params, boxes_per_process, textline_mask_tot, contours_per_process, contours_par_per_process, image_page_rotated, mask_texts_only, num_col, scale_par, indexes_text_con_per_process, slope_deskew)))

        for i in range(num_cores):
            processes[i].start()

        slopes = []
        all_found_texline_polygons = []
        all_found_text_regions = []
        all_found_text_regions_par = []
        boxes = []
        all_box_coord = []
        all_index_text_con = []

        for i in range(num_cores):
            list_all_par = queue_of_all_params.get(True)
            polys_for_sub_process = list_all_par[0]
            boxes_for_sub_process = list_all_par[1]
            contours_for_subprocess = list_all_par[2]
            contours_par_for_subprocess = list_all_par[3]
            boxes_coord_for_subprocess = list_all_par[4]
            indexes_for_subprocess = list_all_par[5]
            slopes_for_sub_process = list_all_par[6]
            for j in range(len(polys_for_sub_process)):
                slopes.append(slopes_for_sub_process[j])
                all_found_texline_polygons.append(polys_for_sub_process[j])
                boxes.append(boxes_for_sub_process[j])
                all_found_text_regions.append(contours_for_subprocess[j])
                all_found_text_regions_par.append(contours_par_for_subprocess[j])
                all_box_coord.append(boxes_coord_for_subprocess[j])
                all_index_text_con.append(indexes_for_subprocess[j])

        for i in range(num_cores):
            processes[i].join()
        # print(slopes,'slopes')
        return all_found_texline_polygons, boxes, all_found_text_regions, all_found_text_regions_par, all_box_coord, all_index_text_con, slopes

    def do_work_of_slopes_new_curved(self, queue_of_all_params, boxes_text, textline_mask_tot_ea, contours_per_process, contours_par_per_process, image_page_rotated, mask_texts_only, num_col, scale_par, indexes_r_con_per_pro, slope_deskew):
        self.logger.debug("enter do_work_of_slopes_new_curved")
        slopes_per_each_subprocess = []
        bounding_box_of_textregion_per_each_subprocess = []
        textlines_rectangles_per_each_subprocess = []
        contours_textregion_per_each_subprocess = []
        contours_textregion_par_per_each_subprocess = []
        all_box_coord_per_process = []
        index_by_text_region_contours = []

        textline_cnt_seperated = np.zeros(textline_mask_tot_ea.shape)

        for mv in range(len(boxes_text)):

            all_text_region_raw = textline_mask_tot_ea[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]]
            all_text_region_raw = all_text_region_raw.astype(np.uint8)
            img_int_p = all_text_region_raw[:, :]

            # img_int_p=cv2.erode(img_int_p,self.kernel,iterations = 2)
            # plt.imshow(img_int_p)
            # plt.show()

            if img_int_p.shape[0] / img_int_p.shape[1] < 0.1:
                slopes_per_each_subprocess.append(0)
                slope_for_all = [slope_deskew][0]
            else:
                try:
                    textline_con, hierachy = return_contours_of_image(img_int_p)
                    textline_con_fil = filter_contours_area_of_image(img_int_p, textline_con, hierachy, max_area=1, min_area=0.0008)
                    y_diff_mean = find_contours_mean_y_diff(textline_con_fil)
                    sigma_des = max(1, int(y_diff_mean * (4.0 / 40.0)))

                    img_int_p[img_int_p > 0] = 1
                    slope_for_all = return_deskew_slop(img_int_p, sigma_des, plotter=self.plotter)

                    if abs(slope_for_all) < 0.5:
                        slope_for_all = [slope_deskew][0]
                    # old method
                    # slope_for_all=self.textline_contours_to_get_slope_correctly(self.all_text_region_raw[mv],denoised,contours[mv])
                    # text_patch_processed=textline_contours_postprocessing(gada)
                except:
                    slope_for_all = 999

                if slope_for_all == 999:
                    slope_for_all = [slope_deskew][0]
                slopes_per_each_subprocess.append(slope_for_all)

            index_by_text_region_contours.append(indexes_r_con_per_pro[mv])
            crop_img, crop_coor = crop_image_inside_box(boxes_text[mv], image_page_rotated)

            if abs(slope_for_all) < 45:
                # all_box_coord.append(crop_coor)
                textline_region_in_image = np.zeros(textline_mask_tot_ea.shape)
                cnt_o_t_max = contours_par_per_process[mv]
                x, y, w, h = cv2.boundingRect(cnt_o_t_max)
                mask_biggest = np.zeros(mask_texts_only.shape)
                mask_biggest = cv2.fillPoly(mask_biggest, pts=[cnt_o_t_max], color=(1, 1, 1))
                mask_region_in_patch_region = mask_biggest[y : y + h, x : x + w]
                textline_biggest_region = mask_biggest * textline_mask_tot_ea

                # print(slope_for_all,'slope_for_all')
                textline_rotated_seperated = seperate_lines_new2(textline_biggest_region[y : y + h, x : x + w], 0, num_col, slope_for_all, plotter=self.plotter)

                # new line added
                ##print(np.shape(textline_rotated_seperated),np.shape(mask_biggest))
                textline_rotated_seperated[mask_region_in_patch_region[:, :] != 1] = 0
                # till here

                textline_cnt_seperated[y : y + h, x : x + w] = textline_rotated_seperated
                textline_region_in_image[y : y + h, x : x + w] = textline_rotated_seperated

                # plt.imshow(textline_region_in_image)
                # plt.show()
                # plt.imshow(textline_cnt_seperated)
                # plt.show()

                pixel_img = 1
                cnt_textlines_in_image = return_contours_of_interested_textline(textline_region_in_image, pixel_img)

                textlines_cnt_per_region = []
                for jjjj in range(len(cnt_textlines_in_image)):
                    mask_biggest2 = np.zeros(mask_texts_only.shape)
                    mask_biggest2 = cv2.fillPoly(mask_biggest2, pts=[cnt_textlines_in_image[jjjj]], color=(1, 1, 1))
                    if num_col + 1 == 1:
                        mask_biggest2 = cv2.dilate(mask_biggest2, self.kernel, iterations=5)
                    else:
                        mask_biggest2 = cv2.dilate(mask_biggest2, self.kernel, iterations=4)

                    pixel_img = 1
                    mask_biggest2 = resize_image(mask_biggest2, int(mask_biggest2.shape[0] * scale_par), int(mask_biggest2.shape[1] * scale_par))
                    cnt_textlines_in_image_ind = return_contours_of_interested_textline(mask_biggest2, pixel_img)
                    try:
                        # textlines_cnt_per_region.append(cnt_textlines_in_image_ind[0]/scale_par)
                        textlines_cnt_per_region.append(cnt_textlines_in_image_ind[0])
                    except:
                        pass
            else:
                add_boxes_coor_into_textlines = True
                textlines_cnt_per_region = textline_contours_postprocessing(all_text_region_raw, slope_for_all, contours_par_per_process[mv], boxes_text[mv], add_boxes_coor_into_textlines)
                add_boxes_coor_into_textlines = False
                # print(np.shape(textlines_cnt_per_region),'textlines_cnt_per_region')

            textlines_rectangles_per_each_subprocess.append(textlines_cnt_per_region)
            bounding_box_of_textregion_per_each_subprocess.append(boxes_text[mv])
            contours_textregion_per_each_subprocess.append(contours_per_process[mv])
            contours_textregion_par_per_each_subprocess.append(contours_par_per_process[mv])
            all_box_coord_per_process.append(crop_coor)

        queue_of_all_params.put([textlines_rectangles_per_each_subprocess, bounding_box_of_textregion_per_each_subprocess, contours_textregion_per_each_subprocess, contours_textregion_par_per_each_subprocess, all_box_coord_per_process, index_by_text_region_contours, slopes_per_each_subprocess])

    def do_work_of_slopes_new(self, queue_of_all_params, boxes_text, textline_mask_tot_ea, contours_per_process, contours_par_per_process, indexes_r_con_per_pro, image_page_rotated, slope_deskew):
        self.logger.debug('enter do_work_of_slopes_new')

        slopes_per_each_subprocess = []
        bounding_box_of_textregion_per_each_subprocess = []
        textlines_rectangles_per_each_subprocess = []
        contours_textregion_per_each_subprocess = []
        contours_textregion_par_per_each_subprocess = []
        all_box_coord_per_process = []
        index_by_text_region_contours = []

        for mv in range(len(boxes_text)):
            crop_img,crop_coor=crop_image_inside_box(boxes_text[mv],image_page_rotated)
            mask_textline=np.zeros((textline_mask_tot_ea.shape))
            mask_textline=cv2.fillPoly(mask_textline,pts=[contours_per_process[mv]],color=(1,1,1))
            denoised=None
            all_text_region_raw=(textline_mask_tot_ea*mask_textline[:,:])[boxes_text[mv][1]:boxes_text[mv][1]+boxes_text[mv][3] , boxes_text[mv][0]:boxes_text[mv][0]+boxes_text[mv][2] ]
            all_text_region_raw=all_text_region_raw.astype(np.uint8)
            img_int_p=all_text_region_raw[:,:]#self.all_text_region_raw[mv]
            img_int_p=cv2.erode(img_int_p,self.kernel,iterations = 2)

            if img_int_p.shape[0]/img_int_p.shape[1]<0.1:
                slopes_per_each_subprocess.append(0)
                slope_for_all = [slope_deskew][0]
                all_text_region_raw = textline_mask_tot_ea[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]]
                cnt_clean_rot = textline_contours_postprocessing(all_text_region_raw, slope_for_all, contours_par_per_process[mv], boxes_text[mv], 0)
                textlines_rectangles_per_each_subprocess.append(cnt_clean_rot)
                index_by_text_region_contours.append(indexes_r_con_per_pro[mv])
                bounding_box_of_textregion_per_each_subprocess.append(boxes_text[mv])
            else:
                try:
                    textline_con, hierachy = return_contours_of_image(img_int_p)
                    textline_con_fil = filter_contours_area_of_image(img_int_p, textline_con, hierachy, max_area=1, min_area=0.00008)
                    y_diff_mean = find_contours_mean_y_diff(textline_con_fil)
                    sigma_des = int(y_diff_mean * (4.0 / 40.0))
                    if sigma_des < 1:
                        sigma_des = 1
                    img_int_p[img_int_p > 0] = 1
                    slope_for_all = return_deskew_slop(img_int_p, sigma_des, plotter=self.plotter)
                    if abs(slope_for_all) <= 0.5:
                        slope_for_all = [slope_deskew][0]
                except:
                    slope_for_all = 999

                if slope_for_all == 999:
                    slope_for_all = [slope_deskew][0]
                slopes_per_each_subprocess.append(slope_for_all)
                mask_only_con_region = np.zeros(textline_mask_tot_ea.shape)
                mask_only_con_region = cv2.fillPoly(mask_only_con_region, pts=[contours_par_per_process[mv]], color=(1, 1, 1))

                # plt.imshow(mask_only_con_region)
                # plt.show()
                all_text_region_raw = np.copy(textline_mask_tot_ea[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]])
                mask_only_con_region = mask_only_con_region[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]]

                ##plt.imshow(textline_mask_tot_ea)
                ##plt.show()
                ##plt.imshow(all_text_region_raw)
                ##plt.show()
                ##plt.imshow(mask_only_con_region)
                ##plt.show()

                all_text_region_raw[mask_only_con_region == 0] = 0
                cnt_clean_rot = textline_contours_postprocessing(all_text_region_raw, slope_for_all, contours_par_per_process[mv], boxes_text[mv])

                textlines_rectangles_per_each_subprocess.append(cnt_clean_rot)
                index_by_text_region_contours.append(indexes_r_con_per_pro[mv])
                bounding_box_of_textregion_per_each_subprocess.append(boxes_text[mv])

            contours_textregion_per_each_subprocess.append(contours_per_process[mv])
            contours_textregion_par_per_each_subprocess.append(contours_par_per_process[mv])
            all_box_coord_per_process.append(crop_coor)

        queue_of_all_params.put([slopes_per_each_subprocess, textlines_rectangles_per_each_subprocess, bounding_box_of_textregion_per_each_subprocess, contours_textregion_per_each_subprocess, contours_textregion_par_per_each_subprocess, all_box_coord_per_process, index_by_text_region_contours])

    def textline_contours(self, img, patches, scaler_h, scaler_w):
        self.logger.debug('enter textline_contours')

        model_textline, session_textline = self.start_new_session_and_model(self.model_textline_dir if patches else self.model_textline_dir_np)
        img = img.astype(np.uint8)
        img_org = np.copy(img)
        img_h = img_org.shape[0]
        img_w = img_org.shape[1]
        img = resize_image(img_org, int(img_org.shape[0] * scaler_h), int(img_org.shape[1] * scaler_w))
        prediction_textline = self.do_prediction(patches, img, model_textline)
        prediction_textline = resize_image(prediction_textline, img_h, img_w)
        prediction_textline_longshot = self.do_prediction(False, img, model_textline)
        prediction_textline_longshot_true_size = resize_image(prediction_textline_longshot, img_h, img_w)
        ##plt.imshow(prediction_textline_streched[:,:,0])
        ##plt.show()

        session_textline.close()
        del model_textline
        del session_textline
        del img
        del img_org

        gc.collect()
        return prediction_textline[:, :, 0], prediction_textline_longshot_true_size[:, :, 0]

    def do_work_of_slopes(self, q, poly, box_sub, boxes_per_process, textline_mask_tot, contours_per_process):
        self.logger.debug('enter do_work_of_slopes')
        slope_biggest = 0
        slopes_sub = []
        boxes_sub_new = []
        poly_sub = []
        for mv in range(len(boxes_per_process)):

            crop_img, _ = crop_image_inside_box(boxes_per_process[mv], np.repeat(textline_mask_tot[:, :, np.newaxis], 3, axis=2))
            crop_img = crop_img[:, :, 0]
            crop_img = cv2.erode(crop_img, self.kernel, iterations=2)

            try:
                textline_con, hierachy = return_contours_of_image(crop_img)
                textline_con_fil = filter_contours_area_of_image(crop_img, textline_con, hierachy, max_area=1, min_area=0.0008)
                y_diff_mean = find_contours_mean_y_diff(textline_con_fil)

                sigma_des = int(y_diff_mean * (4.0 / 40.0))

                if sigma_des < 1:
                    sigma_des = 1

                crop_img[crop_img > 0] = 1
                slope_corresponding_textregion = return_deskew_slop(crop_img, sigma_des, plotter=self.plotter)

            except:
                slope_corresponding_textregion = 999

            if slope_corresponding_textregion == 999:
                slope_corresponding_textregion = slope_biggest
            slopes_sub.append(slope_corresponding_textregion)

            cnt_clean_rot = textline_contours_postprocessing(crop_img, slope_corresponding_textregion, contours_per_process[mv], boxes_per_process[mv])

            poly_sub.append(cnt_clean_rot)
            boxes_sub_new.append(boxes_per_process[mv])

        q.put(slopes_sub)
        poly.put(poly_sub)
        box_sub.put(boxes_sub_new)

    def serialize_lines_in_region(self, textregion, all_found_texline_polygons, region_idx, page_coord, all_box_coord, slopes, id_indexer_l):
        self.logger.debug('enter serialize_lines_in_region')
        for j in range(len(all_found_texline_polygons[region_idx])):
            textline = ET.SubElement(textregion, 'TextLine')
            textline.set('id', 'l%s' % id_indexer_l)
            id_indexer_l += 1
            coord = ET.SubElement(textline, 'Coords')
            add_textequiv(textline)

            points_co = ''
            for l in range(len(all_found_texline_polygons[region_idx][j])):
                if not self.curved_line:
                    if len(all_found_texline_polygons[region_idx][j][l])==2:
                        textline_x_coord = max(0, int((all_found_texline_polygons[region_idx][j][l][0] + all_box_coord[region_idx][2] + page_coord[2]) / self.scale_x))
                        textline_y_coord = max(0, int((all_found_texline_polygons[region_idx][j][l][1] + all_box_coord[region_idx][0] + page_coord[0]) / self.scale_y))
                    else:
                        textline_x_coord = max(0, int((all_found_texline_polygons[region_idx][j][l][0][0] + all_box_coord[region_idx][2] + page_coord[2]) / self.scale_x))
                        textline_y_coord = max(0, int((all_found_texline_polygons[region_idx][j][l][0][1] + all_box_coord[region_idx][0] + page_coord[0]) / self.scale_y))
                    points_co += str(textline_x_coord)
                    points_co += ','
                    points_co += str(textline_y_coord)

                if self.curved_line and np.abs(slopes[region_idx]) <= 45:
                    if len(all_found_texline_polygons[region_idx][j][l]) == 2:
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][0] + page_coord[2]) / self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][0][0] + page_coord[2]) / self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][0][1] + page_coord[0])/self.scale_y))
                elif self.curved_line and np.abs(slopes[region_idx]) > 45:
                    if len(all_found_texline_polygons[region_idx][j][l])==2:
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][0] + all_box_coord[region_idx][2]+page_coord[2])/self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][1] + all_box_coord[region_idx][0]+page_coord[0])/self.scale_y))
                    else:
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][0][0] + all_box_coord[region_idx][2]+page_coord[2])/self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][0][1] + all_box_coord[region_idx][0]+page_coord[0])/self.scale_y))

                if l < len(all_found_texline_polygons[region_idx][j]) - 1:
                    points_co += ' '
            coord.set('points',points_co)
        return id_indexer_l

    def calculate_polygon_coords(self, contour_list, i, page_coord):
        self.logger.debug('enter calculate_polygon_coords')
        coords = ''
        for j in range(len(contour_list[i])):
            if len(contour_list[i][j]) == 2:
                coords += str(int((contour_list[i][j][0] + page_coord[2]) / self.scale_x))
                coords += ','
                coords += str(int((contour_list[i][j][1] + page_coord[0]) / self.scale_y))
            else:
                coords += str(int((contour_list[i][j][0][0] + page_coord[2]) / self.scale_x))
                coords += ','
                coords += str(int((contour_list[i][j][0][1] + page_coord[0]) / self.scale_y))

            if j < len(contour_list[i]) - 1:
                coords=coords+' '
        #print(coords)
        return coords

    def calculate_page_coords(self):
        self.logger.debug('enter calculate_page_coords')
        points_page_print = ""
        for lmm in range(len(self.cont_page[0])):
            if len(self.cont_page[0][lmm]) == 2:
                points_page_print += str(int((self.cont_page[0][lmm][0] ) / self.scale_x))
                points_page_print += ','
                points_page_print += str(int((self.cont_page[0][lmm][1] ) / self.scale_y))
            else:
                points_page_print += str(int((self.cont_page[0][lmm][0][0]) / self.scale_x))
                points_page_print += ','
                points_page_print += str(int((self.cont_page[0][lmm][0][1] ) / self.scale_y))

            if lmm < len( self.cont_page[0] ) - 1:
                points_page_print = points_page_print + ' '
        return points_page_print

    def xml_reading_order(self, page, order_of_texts, id_of_texts, id_of_marginalia, found_polygons_marginals):
        """
        XXX side-effect: extends id_of_marginalia
        """
        region_order = ET.SubElement(page, 'ReadingOrder')
        region_order_sub = ET.SubElement(region_order, 'OrderedGroup')
        region_order_sub.set('id', "ro357564684568544579089")
        indexer_region = 0
        for vj in order_of_texts:
            name = "coord_text_%s" % vj
            name = ET.SubElement(region_order_sub, 'RegionRefIndexed')
            name.set('index', str(indexer_region))
            name.set('regionRef', id_of_texts[vj])
            indexer_region+=1
        for vm in range(len(found_polygons_marginals)):
            id_of_marginalia.append('r%s' % indexer_region)
            name = "coord_text_%s" % indexer_region
            name = ET.SubElement(region_order_sub, 'RegionRefIndexed')
            name.set('index', str(indexer_region))
            name.set('regionRef', 'r%s' % indexer_region)
            indexer_region += 1


    def write_into_page_xml(self, found_polygons_text_region, page_coord, dir_of_image, order_of_texts, id_of_texts, all_found_texline_polygons, all_box_coord, found_polygons_text_region_img, found_polygons_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, curved_line, slopes, slopes_marginals):
        self.logger.debug('enter write_into_page_xml')

        # create the file structure
        pcgts, page = create_page_xml(self.image_filename, self.height_org, self.width_org)
        page_print_sub = ET.SubElement(page, "Border")
        coord_page = ET.SubElement(page_print_sub, "Coords")
        coord_page.set('points', self.calculate_page_coords())

        id_of_marginalia = []
        id_indexer = 0
        id_indexer_l = 0
        if len(found_polygons_text_region) > 0:
            self.xml_reading_order(page, order_of_texts, id_of_texts, id_of_marginalia, found_polygons_marginals)

            for mm in range(len(found_polygons_text_region)):
                textregion = ET.SubElement(page, 'TextRegion')
                textregion.set('id', 'r%s' % id_indexer)
                id_indexer += 1
                textregion.set('type', 'paragraph')
                coord_text = ET.SubElement(textregion, 'Coords')
                coord_text.set('points', self.calculate_polygon_coords(found_polygons_text_region, mm, page_coord))
                for j in range(len(all_found_texline_polygons[mm])):
                    textline = ET.SubElement(textregion, 'TextLine')
                    textline.set('id', 'l%s'  % id_indexer_l)
                    id_indexer_l += 1
                    coord = ET.SubElement(textline, 'Coords')
                    add_textequiv(textline)
                    points_co = ''
                    for l in range(len(all_found_texline_polygons[mm][j])):
                        if not curved_line:
                            if len(all_found_texline_polygons[mm][j][l]) == 2:
                                textline_x_coord = max(0, int((all_found_texline_polygons[mm][j][l][0] + all_box_coord[mm][2] + page_coord[2]) / self.scale_x))
                                textline_y_coord = max(0, int((all_found_texline_polygons[mm][j][l][1] + all_box_coord[mm][0] + page_coord[0]) / self.scale_y))
                            else:
                                textline_x_coord = max(0, int((all_found_texline_polygons[mm][j][l][0][0] + all_box_coord[mm][2]+page_coord[2]) / self.scale_x))
                                textline_y_coord = max(0, int((all_found_texline_polygons[mm][j][l][0][1] + all_box_coord[mm][0]+page_coord[0]) / self.scale_y))
                            points_co += str(textline_x_coord) + ',' + str(textline_y_coord)
                        if curved_line and abs(slopes[mm]) <= 45:
                            if len(all_found_texline_polygons[mm][j][l]) == 2:
                                points_co += str(int((all_found_texline_polygons[mm][j][l][0] + page_coord[2]) / self.scale_x))
                                points_co += ','
                                points_co += str(int((all_found_texline_polygons[mm][j][l][1] + page_coord[0]) / self.scale_y))
                            else:
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][0] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ','
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][1] + page_coord[0]) / self.scale_y))
                        elif curved_line and abs(slopes[mm]) > 45:
                            if len(all_found_texline_polygons[mm][j][l]) == 2:
                                points_co += str(int((all_found_texline_polygons[mm][j][l][0] + all_box_coord[mm][2] + page_coord[2]) / self.scale_x))
                                points_co += ','
                                points_co += str(int((all_found_texline_polygons[mm][j][l][1] + all_box_coord[mm][0] + page_coord[0]) / self.scale_y))
                            else:
                                points_co += str(int((all_found_texline_polygons[mm][j][l][0][0] + all_box_coord[mm][2] + page_coord[2]) / self.scale_x))
                                points_co += ','
                                points_co += str(int((all_found_texline_polygons[mm][j][l][0][1] + all_box_coord[mm][0] + page_coord[0]) / self.scale_y))

                        if l < len(all_found_texline_polygons[mm][j]) - 1:
                            points_co += ' '
                    coord.set('points', points_co)

                add_textequiv(textregion)

        for mm in range(len(found_polygons_marginals)):
            textregion = ET.SubElement(page, 'TextRegion')
            textregion.set('id', id_of_marginalia[mm])
            textregion.set('type', 'marginalia')
            coord_text = ET.SubElement(textregion, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_marginals, mm, page_coord))
            for j in range(len(all_found_texline_polygons_marginals[mm])):
                textline = ET.SubElement(textregion, 'TextLine')
                textline.set('id','l'+str(id_indexer_l))
                id_indexer_l += 1
                coord = ET.SubElement(textline, 'Coords')
                add_textequiv(textline)
                points_co = ''
                for l in range(len(all_found_texline_polygons_marginals[mm][j])):
                    if not curved_line:
                        if len(all_found_texline_polygons_marginals[mm][j][l]) == 2:
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][0] + all_box_coord_marginals[mm][2] + page_coord[2]) / self.scale_x))
                            points_co += ','
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][1] + all_box_coord_marginals[mm][0] + page_coord[0]) / self.scale_y))
                        else:
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][0][0] + all_box_coord_marginals[mm][2] + page_coord[2]) / self.scale_x))
                            points_co += ','
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][0][1] + all_box_coord_marginals[mm][0] + page_coord[0])/self.scale_y))
                    else:
                        if len(all_found_texline_polygons_marginals[mm][j][l]) == 2:
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][0] + page_coord[2]) / self.scale_x))
                            points_co += ','
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][1] + page_coord[0]) / self.scale_y))
                        else:
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][0][0] + page_coord[2]) / self.scale_x))
                            points_co += ','
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][0][1] + page_coord[0]) / self.scale_y))
                    if l < len(all_found_texline_polygons_marginals[mm][j]) - 1:
                        points_co += ' '
                coord.set('points',points_co)

        id_indexer = len(found_polygons_text_region) + len(found_polygons_marginals)
        for mm in range(len(found_polygons_text_region_img)):
            textregion=ET.SubElement(page, 'ImageRegion')
            textregion.set('id', 'r%s' % id_indexer)
            id_indexer += 1
            coord_text = ET.SubElement(textregion, 'Coords')
            points_co = ''
            for lmm in range(len(found_polygons_text_region_img[mm])):
                points_co += str(int((found_polygons_text_region_img[mm][lmm,0,0] + page_coord[2]) / self.scale_x))
                points_co += ','
                points_co += str(int((found_polygons_text_region_img[mm][lmm,0,1] + page_coord[0]) / self.scale_y))
                if lmm < len(found_polygons_text_region_img[mm]) - 1:
                    points_co += ' '
            coord_text.set('points', points_co)

        self.logger.info("filename stem: '%s'", self.image_filename_stem)
        tree = ET.ElementTree(pcgts)
        tree.write(os.path.join(dir_of_image, self.image_filename_stem) + ".xml")

    def write_into_page_xml_full(self, found_polygons_text_region, found_polygons_text_region_h, page_coord, dir_of_image, order_of_texts, id_of_texts, all_found_texline_polygons, all_found_texline_polygons_h, all_box_coord, all_box_coord_h, found_polygons_text_region_img, found_polygons_tables, found_polygons_drop_capitals, found_polygons_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals):
        self.logger.debug('enter write_into_page_xml_full')

        # create the file structure
        pcgts, page = create_page_xml(self.image_filename, self.height_org, self.width_org)
        page_print_sub = ET.SubElement(page, "Border")
        coord_page = ET.SubElement(page_print_sub, "Coords")
        coord_page.set('points', self.calculate_page_coords())

        id_indexer = 0
        id_indexer_l = 0
        id_of_marginalia = []

        if len(found_polygons_text_region) > 0:
            self.xml_reading_order(page, order_of_texts, id_of_texts, id_of_marginalia, found_polygons_marginals)
            for mm in range(len(found_polygons_text_region)):
                textregion=ET.SubElement(page, 'TextRegion')
                textregion.set('id', 'r%s' % id_indexer)
                id_indexer += 1
                textregion.set('type', 'paragraph')
                coord_text = ET.SubElement(textregion, 'Coords')
                coord_text.set('points', self.calculate_polygon_coords(found_polygons_text_region, mm, page_coord))
                id_indexer_l = self.serialize_lines_in_region(textregion, all_found_texline_polygons, mm, page_coord, all_box_coord, slopes, id_indexer_l)
                add_textequiv(textregion)

        self.logger.debug('len(found_polygons_text_region_h) %s', len(found_polygons_text_region_h))
        if len(found_polygons_text_region_h) > 0:
            for mm in range(len(found_polygons_text_region_h)):
                textregion=ET.SubElement(page, 'TextRegion')
                textregion.set('id', 'r%s' % id_indexer)
                id_indexer += 1
                textregion.set('type','header')
                coord_text = ET.SubElement(textregion, 'Coords')
                coord_text.set('points', self.calculate_polygon_coords(found_polygons_text_region_h, mm, page_coord))
                id_indexer_l = self.serialize_lines_in_region(textregion, all_found_texline_polygons_h, mm, page_coord, all_box_coord_h, slopes, id_indexer_l)
                add_textequiv(textregion)

        if len(found_polygons_drop_capitals) > 0:
            id_indexer = len(found_polygons_text_region) + len(found_polygons_text_region_h) + len(found_polygons_marginals)
            for mm in range(len(found_polygons_drop_capitals)):
                textregion=ET.SubElement(page, 'TextRegion')
                textregion.set('id',' r%s' % id_indexer)
                id_indexer += 1
                textregion.set('type', 'drop-capital')
                coord_text = ET.SubElement(textregion, 'Coords')
                coord_text.set('points', self.calculate_polygon_coords(found_polygons_drop_capitals, mm, page_coord))
                add_textequiv(textregion)

        for mm in range(len(found_polygons_marginals)):
            textregion = ET.SubElement(page, 'TextRegion')
            textregion.set('id', id_of_marginalia[mm])
            textregion.set('type', 'marginalia')
            coord_text = ET.SubElement(textregion, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_marginals, mm, page_coord))

            for j in range(len(all_found_texline_polygons_marginals[mm])):
                textline = ET.SubElement(textregion, 'TextLine')
                textline.set('id', 'l%s' % id_indexer_l)
                id_indexer_l += 1
                coord = ET.SubElement(textline, 'Coords')
                add_textequiv(textline)
                points_co = ''
                for l in range(len(all_found_texline_polygons_marginals[mm][j])):
                    if not self.curved_line:
                        if len(all_found_texline_polygons_marginals[mm][j][l]) == 2:
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][0] + all_box_coord_marginals[mm][2] + page_coord[2]) / self.scale_x))
                            points_co += ','
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][1] + all_box_coord_marginals[mm][0] + page_coord[0]) / self.scale_y))
                        else:
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][0][0] + all_box_coord_marginals[mm][2] + page_coord[2]) / self.scale_x))
                            points_co += ','
                            points_co+= str(int((all_found_texline_polygons_marginals[mm][j][l][0][1] + all_box_coord_marginals[mm][0] + page_coord[0]) / self.scale_y))
                    else:
                        if len(all_found_texline_polygons_marginals[mm][j][l])==2:
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][0] + page_coord[2]) / self.scale_x))
                            points_co += ','
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][1] + page_coord[0]) / self.scale_y))
                        else:
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][0][0] + page_coord[2]) / self.scale_x))
                            points_co += ','
                            points_co += str(int((all_found_texline_polygons_marginals[mm][j][l][0][1] + page_coord[0]) / self.scale_y))

                    if l < len(all_found_texline_polygons_marginals[mm][j]) - 1:
                        points_co = points_co+' '
                coord.set('points',points_co)
            add_textequiv(textregion)

        id_indexer = len(found_polygons_text_region) + len(found_polygons_text_region_h) + len(found_polygons_marginals) + len(found_polygons_drop_capitals)
        for mm in range(len(found_polygons_text_region_img)):
            textregion=ET.SubElement(page, 'ImageRegion')
            textregion.set('id', 'r%s' % id_indexer)
            id_indexer += 1
            coord_text = ET.SubElement(textregion, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_text_region_img, mm, page_coord))

        for mm in range(len(found_polygons_tables)):
            textregion = ET.SubElement(page, 'TableRegion')
            textregion.set('id', 'r%s' %id_indexer)
            id_indexer += 1
            coord_text = ET.SubElement(textregion, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_tables, mm, page_coord))

        self.logger.info("filename stem: '%s'", self.image_filename_stem)
        tree = ET.ElementTree(pcgts)
        tree.write(os.path.join(dir_of_image, self.image_filename_stem) + ".xml")

    def get_regions_from_xy_2models(self,img,is_image_enhanced):
        self.logger.debug("enter get_regions_from_xy_2models")
        img_org = np.copy(img)
        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]

        model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_ens)

        gaussian_filter=False
        binary=False
        ratio_y=1.3
        ratio_x=1
        median_blur=False

        img = resize_image(img_org, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))

        if binary:
            img = otsu_copy_binary(img)
            img = img.astype(np.uint16)
        if median_blur:
            img = cv2.medianBlur(img,5)
        if gaussian_filter:
            img= cv2.GaussianBlur(img,(5,5),0)
            img = img.astype(np.uint16)

        prediction_regions_org_y = self.do_prediction(True, img, model_region)
        prediction_regions_org_y = resize_image(prediction_regions_org_y, img_height_h, img_width_h )

        #plt.imshow(prediction_regions_org_y[:,:,0])
        #plt.show()
        prediction_regions_org_y=prediction_regions_org_y[:,:,0]
        mask_zeros_y=(prediction_regions_org_y[:,:]==0)*1
        if is_image_enhanced:
            ratio_x = 1.2
        else:
            ratio_x = 1
        ratio_y = 1
        median_blur=False

        img = resize_image(img_org, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))

        if binary:
            img = otsu_copy_binary(img)#self.otsu_copy(img)
            img = img.astype(np.uint16)
        if median_blur:
            img = cv2.medianBlur(img, 5)
        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5,5 ), 0)
            img = img.astype(np.uint16)

        prediction_regions_org = self.do_prediction(True, img, model_region)
        prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h )

        ##plt.imshow(prediction_regions_org[:,:,0])
        ##plt.show()
        prediction_regions_org=prediction_regions_org[:,:,0]

        prediction_regions_org[(prediction_regions_org[:,:]==1) & (mask_zeros_y[:,:]==1)]=0
        session_region.close()
        del model_region
        del session_region
        gc.collect()

        model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p2)

        gaussian_filter=False
        binary=False
        ratio_x=1
        ratio_y=1
        median_blur=False

        img= resize_image(img_org, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))

        if binary:
            img = otsu_copy_binary(img)#self.otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img=cv2.medianBlur(img,5)
        if gaussian_filter:
            img= cv2.GaussianBlur(img,(5,5),0)
            img = img.astype(np.uint16)

        marginal_patch=0.2
        prediction_regions_org2=self.do_prediction(True, img, model_region, marginal_patch)

        prediction_regions_org2=resize_image(prediction_regions_org2, img_height_h, img_width_h )

        #plt.imshow(prediction_regions_org2[:,:,0])
        #plt.show()
        ##prediction_regions_org=prediction_regions_org[:,:,0]

        session_region.close()
        del model_region
        del session_region
        gc.collect()

        mask_zeros2=(prediction_regions_org2[:,:,0]==0)*1
        mask_lines2=(prediction_regions_org2[:,:,0]==3)*1

        text_sume_early=( (prediction_regions_org[:,:]==1)*1 ).sum()


        prediction_regions_org_copy=np.copy(prediction_regions_org)


        prediction_regions_org_copy[(prediction_regions_org_copy[:,:]==1) & (mask_zeros2[:,:]==1)]=0

        text_sume_second=( (prediction_regions_org_copy[:,:]==1)*1 ).sum()

        rate_two_models=text_sume_second/float(text_sume_early)*100

        self.logger.info("ratio_of_two_models: %s", rate_two_models)
        if not(is_image_enhanced and rate_two_models<95.50):#98.45:
            prediction_regions_org=np.copy(prediction_regions_org_copy)

        ##prediction_regions_org[mask_lines2[:,:]==1]=3
        prediction_regions_org[(mask_lines2[:,:]==1) & (prediction_regions_org[:,:]==0)]=3


        del mask_lines2
        del mask_zeros2
        del prediction_regions_org2

        mask_lines_only=(prediction_regions_org[:,:]==3)*1

        prediction_regions_org = cv2.erode(prediction_regions_org[:,:], self.kernel, iterations=2)

        #plt.imshow(text_region2_1st_channel)
        #plt.show()

        prediction_regions_org = cv2.dilate(prediction_regions_org[:,:], self.kernel, iterations=2)
        mask_texts_only=(prediction_regions_org[:,:]==1)*1
        mask_images_only=(prediction_regions_org[:,:]==2)*1

        pixel_img=1
        min_area_text=0.00001
        polygons_of_only_texts=return_contours_of_interested_region(mask_texts_only,pixel_img,min_area_text)
        polygons_of_only_images=return_contours_of_interested_region(mask_images_only,pixel_img)
        polygons_of_only_lines=return_contours_of_interested_region(mask_lines_only,pixel_img,min_area_text)

        text_regions_p_true=np.zeros(prediction_regions_org.shape)
        text_regions_p_true=cv2.fillPoly(text_regions_p_true,pts=polygons_of_only_lines, color=(3,3,3))
        text_regions_p_true[:,:][mask_images_only[:,:]==1]=2

        text_regions_p_true=cv2.fillPoly(text_regions_p_true,pts=polygons_of_only_texts, color=(1,1,1))

        del polygons_of_only_texts
        del polygons_of_only_images
        del polygons_of_only_lines
        del mask_images_only
        del prediction_regions_org
        del img
        del mask_zeros_y

        del prediction_regions_org_y
        del img_org
        gc.collect()

        K.clear_session()
        return text_regions_p_true

    def do_order_of_regions_full_layout(self, contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot):
        self.logger.debug("enter do_order_of_regions_full_layout")
        cx_text_only, cy_text_only, x_min_text_only, _, _, _, y_cor_x_min_main = find_new_features_of_contoures(contours_only_text_parent)
        cx_text_only_h, cy_text_only_h, x_min_text_only_h, _, _, _, y_cor_x_min_main_h = find_new_features_of_contoures(contours_only_text_parent_h)

        try:
            arg_text_con = []
            for ii in range(len(cx_text_only)):
                for jj in range(len(boxes)):
                    if (x_min_text_only[ii] + 80) >= boxes[jj][0] and (x_min_text_only[ii] + 80) < boxes[jj][1] and y_cor_x_min_main[ii] >= boxes[jj][2] and y_cor_x_min_main[ii] < boxes[jj][3]:
                        arg_text_con.append(jj)
                        break
            args_contours = np.array(range(len(arg_text_con)))

            arg_text_con_h = []
            for ii in range(len(cx_text_only_h)):
                for jj in range(len(boxes)):
                    if (x_min_text_only_h[ii] + 80) >= boxes[jj][0] and (x_min_text_only_h[ii] + 80) < boxes[jj][1] and y_cor_x_min_main_h[ii] >= boxes[jj][2] and y_cor_x_min_main_h[ii] < boxes[jj][3]:
                        arg_text_con_h.append(jj)
                        break
            args_contours_h = np.array(range(len(arg_text_con_h)))

            order_by_con_head = np.zeros(len(arg_text_con_h))
            order_by_con_main = np.zeros(len(arg_text_con))

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij in range(len(boxes)):

                args_contours_box = args_contours[np.array(arg_text_con) == iij]
                args_contours_box_h = args_contours_h[np.array(arg_text_con_h) == iij]
                con_inter_box = []
                con_inter_box_h = []

                for i in range(len(args_contours_box)):
                    con_inter_box.append(contours_only_text_parent[args_contours_box[i]])

                for i in range(len(args_contours_box_h)):
                    con_inter_box_h.append(contours_only_text_parent_h[args_contours_box_h[i]])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_sorted_head = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 2]
                indexes_by_type_head = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 2]

                zahler = 0
                for mtv in args_contours_box:
                    arg_order_v = indexes_sorted_main[zahler]
                    tartib = np.where(indexes_sorted == arg_order_v)[0][0]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = tartib + ref_point
                    zahler = zahler + 1

                zahler = 0
                for mtv in args_contours_box_h:
                    arg_order_v = indexes_sorted_head[zahler]
                    tartib = np.where(indexes_sorted == arg_order_v)[0][0]
                    # print(indexes_sorted,np.where(indexes_sorted==arg_order_v ),arg_order_v,tartib,'inshgalla')
                    order_by_con_head[args_contours_box_h[indexes_by_type_head[zahler]]] = tartib + ref_point
                    zahler = zahler + 1

                for jji in range(len(id_of_texts)):
                    order_of_texts_tot.append(order_of_texts[jji] + ref_point)
                    id_of_texts_tot.append(id_of_texts[jji])
                ref_point = ref_point + len(id_of_texts)

            order_of_texts_tot = []
            for tj1 in range(len(contours_only_text_parent)):
                order_of_texts_tot.append(int(order_by_con_main[tj1]))

            for tj1 in range(len(contours_only_text_parent_h)):
                order_of_texts_tot.append(int(order_by_con_head[tj1]))

            order_text_new = []
            for iii in range(len(order_of_texts_tot)):
                tartib_new = np.where(np.array(order_of_texts_tot) == iii)[0][0]
                order_text_new.append(tartib_new)

        except:
            arg_text_con = []
            for ii in range(len(cx_text_only)):
                for jj in range(len(boxes)):
                    if cx_text_only[ii] >= boxes[jj][0] and cx_text_only[ii] < boxes[jj][1] and cy_text_only[ii] >= boxes[jj][2] and cy_text_only[ii] < boxes[jj][3]:  # this is valid if the center of region identify in which box it is located
                        arg_text_con.append(jj)
                        break
            args_contours = np.array(range(len(arg_text_con)))

            order_by_con_main = np.zeros(len(arg_text_con))

            ############################# head

            arg_text_con_h = []
            for ii in range(len(cx_text_only_h)):
                for jj in range(len(boxes)):
                    if cx_text_only_h[ii] >= boxes[jj][0] and cx_text_only_h[ii] < boxes[jj][1] and cy_text_only_h[ii] >= boxes[jj][2] and cy_text_only_h[ii] < boxes[jj][3]:  # this is valid if the center of region identify in which box it is located
                        arg_text_con_h.append(jj)
                        break
            arg_arg_text_con_h = np.argsort(arg_text_con_h)
            args_contours_h = np.array(range(len(arg_text_con_h)))

            order_by_con_head = np.zeros(len(arg_text_con_h))

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij in range(len(boxes)):
                args_contours_box = args_contours[np.array(arg_text_con) == iij]
                args_contours_box_h = args_contours_h[np.array(arg_text_con_h) == iij]
                con_inter_box = []
                con_inter_box_h = []

                for i in range(len(args_contours_box)):

                    con_inter_box.append(contours_only_text_parent[args_contours_box[i]])
                for i in range(len(args_contours_box_h)):

                    con_inter_box_h.append(contours_only_text_parent_h[args_contours_box_h[i]])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_sorted_head = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 2]
                indexes_by_type_head = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 2]

                zahler = 0
                for mtv in args_contours_box:
                    arg_order_v = indexes_sorted_main[zahler]
                    tartib = np.where(indexes_sorted == arg_order_v)[0][0]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = tartib + ref_point
                    zahler = zahler + 1

                zahler = 0
                for mtv in args_contours_box_h:
                    arg_order_v = indexes_sorted_head[zahler]
                    tartib = np.where(indexes_sorted == arg_order_v)[0][0]
                    # print(indexes_sorted,np.where(indexes_sorted==arg_order_v ),arg_order_v,tartib,'inshgalla')
                    order_by_con_head[args_contours_box_h[indexes_by_type_head[zahler]]] = tartib + ref_point
                    zahler = zahler + 1

                for jji in range(len(id_of_texts)):
                    order_of_texts_tot.append(order_of_texts[jji] + ref_point)
                    id_of_texts_tot.append(id_of_texts[jji])
                ref_point = ref_point + len(id_of_texts)

            order_of_texts_tot = []
            for tj1 in range(len(contours_only_text_parent)):
                order_of_texts_tot.append(int(order_by_con_main[tj1]))

            for tj1 in range(len(contours_only_text_parent_h)):
                order_of_texts_tot.append(int(order_by_con_head[tj1]))

            order_text_new = []
            for iii in range(len(order_of_texts_tot)):
                tartib_new = np.where(np.array(order_of_texts_tot) == iii)[0][0]
                order_text_new.append(tartib_new)
        return order_text_new, id_of_texts_tot

    def do_order_of_regions_no_full_layout(self, contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot):
        self.logger.debug("enter do_order_of_regions_no_full_layout")
        cx_text_only, cy_text_only, x_min_text_only, _, _, _, y_cor_x_min_main = find_new_features_of_contoures(contours_only_text_parent)

        try:
            arg_text_con = []
            for ii in range(len(cx_text_only)):
                for jj in range(len(boxes)):
                    if (x_min_text_only[ii] + 80) >= boxes[jj][0] and (x_min_text_only[ii] + 80) < boxes[jj][1] and y_cor_x_min_main[ii] >= boxes[jj][2] and y_cor_x_min_main[ii] < boxes[jj][3]:
                        arg_text_con.append(jj)
                        break
            args_contours = np.array(range(len(arg_text_con)))

            order_by_con_main = np.zeros(len(arg_text_con))

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij in range(len(boxes)):

                args_contours_box = args_contours[np.array(arg_text_con) == iij]

                con_inter_box = []
                con_inter_box_h = []

                for i in range(len(args_contours_box)):
                    con_inter_box.append(contours_only_text_parent[args_contours_box[i]])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]

                zahler = 0
                for mtv in args_contours_box:
                    arg_order_v = indexes_sorted_main[zahler]
                    tartib = np.where(indexes_sorted == arg_order_v)[0][0]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = tartib + ref_point
                    zahler = zahler + 1

                for jji in range(len(id_of_texts)):
                    order_of_texts_tot.append(order_of_texts[jji] + ref_point)
                    id_of_texts_tot.append(id_of_texts[jji])
                ref_point = ref_point + len(id_of_texts)

            order_of_texts_tot = []
            for tj1 in range(len(contours_only_text_parent)):
                order_of_texts_tot.append(int(order_by_con_main[tj1]))

            order_text_new = []
            for iii in range(len(order_of_texts_tot)):
                tartib_new = np.where(np.array(order_of_texts_tot) == iii)[0][0]
                order_text_new.append(tartib_new)

        except:
            arg_text_con = []
            for ii in range(len(cx_text_only)):
                for jj in range(len(boxes)):
                    if cx_text_only[ii] >= boxes[jj][0] and cx_text_only[ii] < boxes[jj][1] and cy_text_only[ii] >= boxes[jj][2] and cy_text_only[ii] < boxes[jj][3]:  # this is valid if the center of region identify in which box it is located
                        arg_text_con.append(jj)
                        break
            args_contours = np.array(range(len(arg_text_con)))

            order_by_con_main = np.zeros(len(arg_text_con))

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij in range(len(boxes)):
                args_contours_box = args_contours[np.array(arg_text_con) == iij]
                con_inter_box = []
                con_inter_box_h = []

                for i in range(len(args_contours_box)):

                    con_inter_box.append(contours_only_text_parent[args_contours_box[i]])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_sorted_head = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 2]
                indexes_by_type_head = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 2]

                zahler = 0
                for mtv in args_contours_box:
                    arg_order_v = indexes_sorted_main[zahler]
                    tartib = np.where(indexes_sorted == arg_order_v)[0][0]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = tartib + ref_point
                    zahler = zahler + 1

                for jji in range(len(id_of_texts)):
                    order_of_texts_tot.append(order_of_texts[jji] + ref_point)
                    id_of_texts_tot.append(id_of_texts[jji])
                ref_point = ref_point + len(id_of_texts)

            order_of_texts_tot = []
            for tj1 in range(len(contours_only_text_parent)):
                order_of_texts_tot.append(int(order_by_con_main[tj1]))

            order_text_new = []
            for iii in range(len(order_of_texts_tot)):
                tartib_new = np.where(np.array(order_of_texts_tot) == iii)[0][0]
                order_text_new.append(tartib_new)

        return order_text_new, id_of_texts_tot

    def do_order_of_regions(self, *args, **kwargs):
        if self.full_layout:
            return self.do_order_of_regions_full_layout(*args, **kwargs)
        return self.do_order_of_regions_no_full_layout(*args, **kwargs)

    def run_graphics_and_columns(self, text_regions_p_1, num_col_classifier, num_column_is_classified):
        img_g = self.imread(grayscale=True, uint8=True)

        img_g3 = np.zeros((img_g.shape[0], img_g.shape[1], 3))
        img_g3 = img_g3.astype(np.uint8)
        img_g3[:, :, 0] = img_g[:, :]
        img_g3[:, :, 1] = img_g[:, :]
        img_g3[:, :, 2] = img_g[:, :]

        image_page, page_coord = self.extract_page()
        if self.plotter:
            self.plotter.save_page_image(image_page)

        img_g3_page = img_g3[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3], :]

        text_regions_p_1 = text_regions_p_1[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]

        mask_images = (text_regions_p_1[:, :] == 2) * 1
        mask_images = mask_images.astype(np.uint8)
        mask_images = cv2.erode(mask_images[:, :], self.kernel, iterations=10)

        mask_lines = (text_regions_p_1[:, :] == 3) * 1
        mask_lines = mask_lines.astype(np.uint8)

        img_only_regions_with_sep = ((text_regions_p_1[:, :] != 3) & (text_regions_p_1[:, :] != 0)) * 1
        img_only_regions_with_sep = img_only_regions_with_sep.astype(np.uint8)
        img_only_regions = cv2.erode(img_only_regions_with_sep[:, :], self.kernel, iterations=6)

        try:
            num_col, peaks_neg_fin = find_num_col(img_only_regions, multiplier=6.0)
            num_col = num_col + 1
            if not num_column_is_classified:
                num_col_classifier = num_col + 1
        except:
            num_col = None
            peaks_neg_fin = []
        return num_col, num_col_classifier, img_only_regions, page_coord, image_page, mask_images, mask_lines, text_regions_p_1

    def run_enhancement(self):
        self.logger.info("resize and enhance image")
        is_image_enhanced, img_org, img_res, num_col_classifier, num_column_is_classified = self.resize_and_enhance_image_with_column_classifier()
        self.logger.info("Image is %senhanced", '' if is_image_enhanced else 'not ')
        K.clear_session()
        scale = 1
        if is_image_enhanced:
            if self.allow_enhancement:
                cv2.imwrite(os.path.join(self.dir_out, self.image_filename_stem) + ".tif", img_res)
                img_res = img_res.astype(np.uint8)
                self.get_image_and_scales(img_org, img_res, scale)
            else:
                self.get_image_and_scales_after_enhancing(img_org, img_res)
        else:
            if self.allow_enhancement:
                self.get_image_and_scales(img_org, img_res, scale)
            else:
                self.get_image_and_scales(img_org, img_res, scale)
            if self.allow_scaling:
                img_org, img_res, is_image_enhanced = self.resize_image_with_column_classifier(is_image_enhanced)
                self.get_image_and_scales_after_enhancing(img_org, img_res)
        return img_res, is_image_enhanced, num_col_classifier, num_column_is_classified

    def run_textline(self, image_page):
        scaler_h_textline = 1  # 1.2#1.2
        scaler_w_textline = 1  # 0.9#1
        textline_mask_tot_ea, textline_mask_tot_long_shot = self.textline_contours(image_page, True, scaler_h_textline, scaler_w_textline)

        K.clear_session()
        gc.collect()
        #print(np.unique(textline_mask_tot_ea[:, :]), "textline")
        # plt.imshow(textline_mask_tot_ea)
        # plt.show()
        if self.plotter:
            self.plotter.save_plot_of_textlines(textline_mask_tot_ea, image_page)
        return textline_mask_tot_ea, textline_mask_tot_long_shot

    def run_deskew(self, textline_mask_tot_ea):
        sigma = 2
        main_page_deskew = True
        slope_deskew = return_deskew_slop(cv2.erode(textline_mask_tot_ea, self.kernel, iterations=2), sigma, main_page_deskew, plotter=self.plotter)
        slope_first = 0

        if self.plotter:
            self.plotter.save_deskewed_image(slope_deskew)
        self.logger.info("slope_deskew: %s", slope_deskew)
        return slope_deskew, slope_first

    def run_marginals(self, image_page, textline_mask_tot_ea, mask_images, mask_lines, num_col_classifier, slope_deskew, text_regions_p_1):
        image_page_rotated, textline_mask_tot = image_page[:, :], textline_mask_tot_ea[:, :]
        textline_mask_tot[mask_images[:, :] == 1] = 0

        pixel_img = 1
        min_area = 0.00001
        max_area = 0.0006
        textline_mask_tot_small_size = return_contours_of_interested_region_by_size(textline_mask_tot, pixel_img, min_area, max_area)
        text_regions_p_1[mask_lines[:, :] == 1] = 3
        text_regions_p = text_regions_p_1[:, :]  # long_short_region[:,:]#self.get_regions_from_2_models(image_page)
        text_regions_p = np.array(text_regions_p)

        if num_col_classifier == 1 or num_col_classifier == 2:
            try:
                regions_without_seperators = (text_regions_p[:, :] == 1) * 1
                regions_without_seperators = regions_without_seperators.astype(np.uint8)
                text_regions_p = get_marginals(rotate_image(regions_without_seperators, slope_deskew), text_regions_p, num_col_classifier, slope_deskew, kernel=self.kernel)
            except:
                pass

        if self.plotter:
            self.plotter.save_plot_of_layout_main_all(text_regions_p, image_page)
            self.plotter.save_plot_of_layout_main(text_regions_p, image_page)
        return textline_mask_tot, text_regions_p, image_page_rotated

    def run_boxes_no_full_layout(self, image_page, textline_mask_tot, text_regions_p, slope_deskew, num_col_classifier):
        self.logger.debug('enter run_boxes_no_full_layout')
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            image_page_rotated_n, textline_mask_tot_d, text_regions_p_1_n = rotation_not_90_func(image_page, textline_mask_tot, text_regions_p, slope_deskew)
            text_regions_p_1_n = resize_image(text_regions_p_1_n, text_regions_p.shape[0], text_regions_p.shape[1])
            textline_mask_tot_d = resize_image(textline_mask_tot_d, text_regions_p.shape[0], text_regions_p.shape[1])
            regions_without_seperators_d = (text_regions_p_1_n[:, :] == 1) * 1
        regions_without_seperators = (text_regions_p[:, :] == 1) * 1  # ( (text_regions_p[:,:]==1) | (text_regions_p[:,:]==2) )*1 #self.return_regions_without_seperators_new(text_regions_p[:,:,0],img_only_regions)
        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
            text_regions_p_1_n = None
            textline_mask_tot_d = None
            regions_without_seperators_d = None
        pixel_lines = 3
        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
            num_col, peaks_neg_fin, matrix_of_lines_ch, spliter_y_new, seperators_closeup_n = find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines)

        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            num_col_d, peaks_neg_fin_d, matrix_of_lines_ch_d, spliter_y_new_d, seperators_closeup_n_d = find_number_of_columns_in_document(np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines)
        K.clear_session()
        gc.collect()

        self.logger.info("num_col_classifier: %s", num_col_classifier)

        if num_col_classifier >= 3:
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                regions_without_seperators = regions_without_seperators.astype(np.uint8)
                regions_without_seperators = cv2.erode(regions_without_seperators[:, :], self.kernel, iterations=6)
                #random_pixels_for_image = np.random.randn(regions_without_seperators.shape[0], regions_without_seperators.shape[1])
                #random_pixels_for_image[random_pixels_for_image < -0.5] = 0
                #random_pixels_for_image[random_pixels_for_image != 0] = 1
                #regions_without_seperators[(random_pixels_for_image[:, :] == 1) & (text_regions_p[:, :] == 2)] = 1
            else:
                regions_without_seperators_d = regions_without_seperators_d.astype(np.uint8)
                regions_without_seperators_d = cv2.erode(regions_without_seperators_d[:, :], self.kernel, iterations=6)
                #random_pixels_for_image = np.random.randn(regions_without_seperators_d.shape[0], regions_without_seperators_d.shape[1])
                #random_pixels_for_image[random_pixels_for_image < -0.5] = 0
                #random_pixels_for_image[random_pixels_for_image != 0] = 1
                #regions_without_seperators_d[(random_pixels_for_image[:, :] == 1) & (text_regions_p_1_n[:, :] == 2)] = 1

        t1 = time.time()
        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
            boxes = return_boxes_of_images_by_order_of_reading_new(spliter_y_new, regions_without_seperators, matrix_of_lines_ch, num_col_classifier)
            boxes_d = None
            self.logger.debug("len(boxes): %s", len(boxes))
        else:
            boxes_d = return_boxes_of_images_by_order_of_reading_new(spliter_y_new_d, regions_without_seperators_d, matrix_of_lines_ch_d, num_col_classifier)
            boxes = None
            self.logger.debug("len(boxes): %s", len(boxes_d))

        self.logger.info("detecting boxes took %ss", str(time.time() - t1))
        img_revised_tab = text_regions_p[:, :]
        polygons_of_images = return_contours_of_interested_region(img_revised_tab, 2)

        # plt.imshow(img_revised_tab)
        # plt.show()
        K.clear_session()
        self.logger.debug('exit run_boxes_no_full_layout')
        return polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_seperators_d, boxes, boxes_d

    def run_boxes_full_layout(self, image_page, textline_mask_tot, text_regions_p, slope_deskew, num_col_classifier, img_only_regions):
        self.logger.debug('enter run_boxes_full_layout')
        # set first model with second model
        text_regions_p[:, :][text_regions_p[:, :] == 2] = 5
        text_regions_p[:, :][text_regions_p[:, :] == 3] = 6
        text_regions_p[:, :][text_regions_p[:, :] == 4] = 8

        K.clear_session()
        # gc.collect()
        image_page = image_page.astype(np.uint8)

        # print(type(image_page))
        regions_fully, regions_fully_only_drop = self.extract_text_regions(image_page, True, cols=num_col_classifier)
        text_regions_p[:,:][regions_fully[:,:,0]==6]=6

        regions_fully_only_drop = put_drop_out_from_only_drop_model(regions_fully_only_drop, text_regions_p)
        regions_fully[:, :, 0][regions_fully_only_drop[:, :, 0] == 4] = 4
        K.clear_session()
        gc.collect()

        # plt.imshow(regions_fully[:,:,0])
        # plt.show()

        regions_fully = putt_bb_of_drop_capitals_of_model_in_patches_in_layout(regions_fully)

        # plt.imshow(regions_fully[:,:,0])
        # plt.show()

        K.clear_session()
        gc.collect()
        regions_fully_np, _ = self.extract_text_regions(image_page, False, cols=num_col_classifier)

        # plt.imshow(regions_fully_np[:,:,0])
        # plt.show()

        if num_col_classifier > 2:
            regions_fully_np[:, :, 0][regions_fully_np[:, :, 0] == 4] = 0
        else:
            regions_fully_np = filter_small_drop_capitals_from_no_patch_layout(regions_fully_np, text_regions_p)

        # plt.imshow(regions_fully_np[:,:,0])
        # plt.show()

        K.clear_session()
        gc.collect()

        # plt.imshow(regions_fully[:,:,0])
        # plt.show()

        regions_fully = boosting_headers_by_longshot_region_segmentation(regions_fully, regions_fully_np, img_only_regions)

        # plt.imshow(regions_fully[:,:,0])
        # plt.show()

        text_regions_p[:, :][regions_fully[:, :, 0] == 4] = 4
        text_regions_p[:, :][regions_fully_np[:, :, 0] == 4] = 4

        #plt.imshow(text_regions_p)
        #plt.show()

        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            image_page_rotated_n, textline_mask_tot_d, text_regions_p_1_n, regions_fully_n = rotation_not_90_func_full_layout(image_page, textline_mask_tot, text_regions_p, regions_fully, slope_deskew)

            text_regions_p_1_n = resize_image(text_regions_p_1_n, text_regions_p.shape[0], text_regions_p.shape[1])
            textline_mask_tot_d = resize_image(textline_mask_tot_d, text_regions_p.shape[0], text_regions_p.shape[1])
            regions_fully_n = resize_image(regions_fully_n, text_regions_p.shape[0], text_regions_p.shape[1])
            regions_without_seperators_d = (text_regions_p_1_n[:, :] == 1) * 1
        else:
            text_regions_p_1_n = None
            textline_mask_tot_d = None
            regions_without_seperators_d = None

        regions_without_seperators = (text_regions_p[:, :] == 1) * 1  # ( (text_regions_p[:,:]==1) | (text_regions_p[:,:]==2) )*1 #self.return_regions_without_seperators_new(text_regions_p[:,:,0],img_only_regions)

        K.clear_session()
        gc.collect()
        img_revised_tab = np.copy(text_regions_p[:, :])
        polygons_of_images = return_contours_of_interested_region(img_revised_tab, 5)
        self.logger.debug('exit run_boxes_full_layout')
        return polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_seperators_d, regions_fully, regions_without_seperators

    def run(self):
        """
        Get image and scales, then extract the page of scanned image
        """
        self.logger.debug("enter run")

        t1 = time.time()
        img_res, is_image_enhanced, num_col_classifier, num_column_is_classified = self.run_enhancement()
        self.logger.info("Enhancing took %ss ", str(time.time() - t1))

        t1 = time.time()
        text_regions_p_1 = self.get_regions_from_xy_2models(img_res, is_image_enhanced)
        self.logger.info("Textregion detection took %ss ", str(time.time() - t1))

        t1 = time.time()
        num_col, num_col_classifier, img_only_regions, page_coord, image_page, mask_images, mask_lines, text_regions_p_1 = \
                self.run_graphics_and_columns(text_regions_p_1, num_col_classifier, num_column_is_classified)
        self.logger.info("Graphics detection took %ss ", str(time.time() - t1))

        if not num_col:
            self.logger.info("No columns detected, outputting an empty PAGE-XML")
            self.write_into_page_xml([], page_coord, self.dir_out, [], [], [], [], [], [], [], [], self.curved_line, [], [])
            self.logger.info("Job done in %ss", str(time.time() - t1))
            return

        t1 = time.time()
        textline_mask_tot_ea, textline_mask_tot_long_shot = self.run_textline(image_page)
        self.logger.info("textline detection took %ss", str(time.time() - t1))

        t1 = time.time()
        slope_deskew, slope_first = self.run_deskew(textline_mask_tot_ea)
        self.logger.info("deskewing took %ss", str(time.time() - t1))
        t1 = time.time()

        textline_mask_tot, text_regions_p, image_page_rotated = self.run_marginals(image_page, textline_mask_tot_ea, mask_images, mask_lines, num_col_classifier, slope_deskew, text_regions_p_1)
        self.logger.info("detection of marginals took %ss", str(time.time() - t1))
        t1 = time.time()

        if not self.full_layout:
            polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_seperators_d, boxes, boxes_d = self.run_boxes_no_full_layout(image_page, textline_mask_tot, text_regions_p, slope_deskew, num_col_classifier)

        pixel_img = 4
        min_area_mar = 0.00001
        polygons_of_marginals = return_contours_of_interested_region(text_regions_p, pixel_img, min_area_mar)

        if self.full_layout:
            polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_seperators_d, regions_fully, regions_without_seperators = self.run_boxes_full_layout(image_page, textline_mask_tot, text_regions_p, slope_deskew, num_col_classifier, img_only_regions)
        # plt.imshow(img_revised_tab)
        # plt.show()

        # print(img_revised_tab.shape,text_regions_p_1_n.shape)
        # text_regions_p_1_n=resize_image(text_regions_p_1_n,img_revised_tab.shape[0],img_revised_tab.shape[1])
        # print(np.unique(text_regions_p_1_n),'uni')

        text_only = ((img_revised_tab[:, :] == 1)) * 1
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            text_only_d = ((text_regions_p_1_n[:, :] == 1)) * 1
        ##text_only_h=( (img_revised_tab[:,:,0]==2) )*1

        # print(text_only.shape,text_only_d.shape)
        # plt.imshow(text_only)
        # plt.show()

        # plt.imshow(text_only_d)
        # plt.show()

        min_con_area = 0.000005
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            contours_only_text, hir_on_text = return_contours_of_image(text_only)
            contours_only_text_parent = return_parent_contours(contours_only_text, hir_on_text)
            areas_cnt_text = np.array([cv2.contourArea(contours_only_text_parent[j]) for j in range(len(contours_only_text_parent))])
            areas_cnt_text = areas_cnt_text / float(text_only.shape[0] * text_only.shape[1])
            self.logger.info('areas_cnt_text %s', areas_cnt_text)
            contours_biggest = contours_only_text_parent[np.argmax(areas_cnt_text)]
            contours_only_text_parent = [contours_only_text_parent[jz] for jz in range(len(contours_only_text_parent)) if areas_cnt_text[jz] > min_con_area]
            areas_cnt_text_parent = [areas_cnt_text[jz] for jz in range(len(areas_cnt_text)) if areas_cnt_text[jz] > min_con_area]

            index_con_parents = np.argsort(areas_cnt_text_parent)
            contours_only_text_parent = list(np.array(contours_only_text_parent)[index_con_parents])
            areas_cnt_text_parent = list(np.array(areas_cnt_text_parent)[index_con_parents])

            cx_bigest_big, cy_biggest_big, _, _, _, _, _ = find_new_features_of_contoures([contours_biggest])
            cx_bigest, cy_biggest, _, _, _, _, _ = find_new_features_of_contoures(contours_only_text_parent)

            contours_only_text_d, hir_on_text_d = return_contours_of_image(text_only_d)
            contours_only_text_parent_d = return_parent_contours(contours_only_text_d, hir_on_text_d)

            areas_cnt_text_d = np.array([cv2.contourArea(contours_only_text_parent_d[j]) for j in range(len(contours_only_text_parent_d))])
            areas_cnt_text_d = areas_cnt_text_d / float(text_only_d.shape[0] * text_only_d.shape[1])

            contours_biggest_d = contours_only_text_parent_d[np.argmax(areas_cnt_text_d)]
            index_con_parents_d=np.argsort(areas_cnt_text_d)
            contours_only_text_parent_d=list(np.array(contours_only_text_parent_d)[index_con_parents_d] )
            areas_cnt_text_d=list(np.array(areas_cnt_text_d)[index_con_parents_d] )

            cx_bigest_d_big, cy_biggest_d_big, _, _, _, _, _ = find_new_features_of_contoures([contours_biggest_d])
            cx_bigest_d, cy_biggest_d, _, _, _, _, _ = find_new_features_of_contoures(contours_only_text_parent_d)
            try:
                cx_bigest_d_last5=cx_bigest_d[-5:]
                cy_biggest_d_last5=cy_biggest_d[-5:]
                dists_d = [math.sqrt((cx_bigest_big[0]-cx_bigest_d_last5[j])**2 + (cy_biggest_big[0]-cy_biggest_d_last5[j])**2) for j in range(len(cy_biggest_d_last5))]
                ind_largest=len(cx_bigest_d)-5+np.argmin(dists_d)
                cx_bigest_d_big[0]=cx_bigest_d[ind_largest]
                cy_biggest_d_big[0]=cy_biggest_d[ind_largest]
            except:
                pass

            (h, w) = text_only.shape[:2]
            center = (w // 2.0, h // 2.0)
            M = cv2.getRotationMatrix2D(center, slope_deskew, 1.0)
            M_22 = np.array(M)[:2, :2]
            p_big = np.dot(M_22, [cx_bigest_big, cy_biggest_big])
            x_diff = p_big[0] - cx_bigest_d_big
            y_diff = p_big[1] - cy_biggest_d_big

            # print(p_big)
            # print(cx_bigest_d_big,cy_biggest_d_big)
            # print(x_diff,y_diff)

            contours_only_text_parent_d_ordered = []
            for i in range(len(contours_only_text_parent)):
                # img1=np.zeros((text_only.shape[0],text_only.shape[1],3))
                # img1=cv2.fillPoly(img1,pts=[contours_only_text_parent[i]] ,color=(1,1,1))
                # plt.imshow(img1[:,:,0])
                # plt.show()

                p = np.dot(M_22, [cx_bigest[i], cy_biggest[i]])
                # print(p)
                p[0] = p[0] - x_diff[0]
                p[1] = p[1] - y_diff[0]
                # print(p)
                # print(cx_bigest_d)
                # print(cy_biggest_d)
                dists = [math.sqrt((p[0] - cx_bigest_d[j]) ** 2 + (p[1] - cy_biggest_d[j]) ** 2) for j in range(len(cx_bigest_d))]
                # print(np.argmin(dists))
                contours_only_text_parent_d_ordered.append(contours_only_text_parent_d[np.argmin(dists)])
                # img2=np.zeros((text_only.shape[0],text_only.shape[1],3))
                # img2=cv2.fillPoly(img2,pts=[contours_only_text_parent_d[np.argmin(dists)]] ,color=(1,1,1))
                # plt.imshow(img2[:,:,0])
                # plt.show()
        else:
            contours_only_text, hir_on_text = return_contours_of_image(text_only)
            contours_only_text_parent = return_parent_contours(contours_only_text, hir_on_text)

            areas_cnt_text = np.array([cv2.contourArea(contours_only_text_parent[j]) for j in range(len(contours_only_text_parent))])
            areas_cnt_text = areas_cnt_text / float(text_only.shape[0] * text_only.shape[1])

            contours_biggest = contours_only_text_parent[np.argmax(areas_cnt_text)]
            contours_only_text_parent = [contours_only_text_parent[jz] for jz in range(len(contours_only_text_parent)) if areas_cnt_text[jz] > min_con_area]
            areas_cnt_text_parent = [areas_cnt_text[jz] for jz in range(len(areas_cnt_text)) if areas_cnt_text[jz] > min_con_area]

            index_con_parents = np.argsort(areas_cnt_text_parent)
            contours_only_text_parent = list(np.array(contours_only_text_parent)[index_con_parents])
            areas_cnt_text_parent = list(np.array(areas_cnt_text_parent)[index_con_parents])

            cx_bigest_big, cy_biggest_big, _, _, _, _, _ = find_new_features_of_contoures([contours_biggest])
            cx_bigest, cy_biggest, _, _, _, _, _ = find_new_features_of_contoures(contours_only_text_parent)
            self.logger.debug('areas_cnt_text_parent %s', areas_cnt_text_parent)
            # self.logger.debug('areas_cnt_text_parent_d %s', areas_cnt_text_parent_d)
            # self.logger.debug('len(contours_only_text_parent) %s', len(contours_only_text_parent_d))

        txt_con_org = get_textregion_contours_in_org_image(contours_only_text_parent, self.image, slope_first)
        boxes_text, _ = get_text_region_boxes_by_given_contours(contours_only_text_parent)
        boxes_marginals, _ = get_text_region_boxes_by_given_contours(polygons_of_marginals)

        if not self.curved_line:
            slopes, all_found_texline_polygons, boxes_text, txt_con_org, contours_only_text_parent, all_box_coord, index_by_text_par_con = self.get_slopes_and_deskew_new(txt_con_org, contours_only_text_parent, textline_mask_tot_ea, image_page_rotated, boxes_text, slope_deskew)
            slopes_marginals, all_found_texline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, all_box_coord_marginals, index_by_text_par_con_marginal = self.get_slopes_and_deskew_new(polygons_of_marginals, polygons_of_marginals, textline_mask_tot_ea, image_page_rotated, boxes_marginals, slope_deskew)

        else:
            scale_param = 1
            all_found_texline_polygons, boxes_text, txt_con_org, contours_only_text_parent, all_box_coord, index_by_text_par_con, slopes = self.get_slopes_and_deskew_new_curved(txt_con_org, contours_only_text_parent, cv2.erode(textline_mask_tot_ea, kernel=self.kernel, iterations=1), image_page_rotated, boxes_text, text_only, num_col_classifier, scale_param, slope_deskew)
            all_found_texline_polygons = small_textlines_to_parent_adherence2(all_found_texline_polygons, textline_mask_tot_ea, num_col_classifier)
            all_found_texline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, all_box_coord_marginals, index_by_text_par_con_marginal, slopes_marginals = self.get_slopes_and_deskew_new_curved(polygons_of_marginals, polygons_of_marginals, cv2.erode(textline_mask_tot_ea, kernel=self.kernel, iterations=1), image_page_rotated, boxes_marginals, text_only, num_col_classifier, scale_param, slope_deskew)
            all_found_texline_polygons_marginals = small_textlines_to_parent_adherence2(all_found_texline_polygons_marginals, textline_mask_tot_ea, num_col_classifier)
        index_of_vertical_text_contours = np.array(range(len(slopes)))[(abs(np.array(slopes)) > 60)]
        contours_text_vertical = [contours_only_text_parent[i] for i in index_of_vertical_text_contours]

        K.clear_session()
        gc.collect()
        # print(index_by_text_par_con,'index_by_text_par_con')

        if self.full_layout:
            if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                contours_only_text_parent_d_ordered = list(np.array(contours_only_text_parent_d_ordered)[index_by_text_par_con])
                text_regions_p, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_texline_polygons, all_found_texline_polygons_h, slopes, slopes_h, contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered = check_any_text_region_in_model_one_is_main_or_header(text_regions_p, regions_fully, contours_only_text_parent, all_box_coord, all_found_texline_polygons, slopes, contours_only_text_parent_d_ordered)
            else:
                contours_only_text_parent_d_ordered = None
                text_regions_p, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_texline_polygons, all_found_texline_polygons_h, slopes, slopes_h, contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered = check_any_text_region_in_model_one_is_main_or_header(text_regions_p, regions_fully, contours_only_text_parent, all_box_coord, all_found_texline_polygons, slopes, contours_only_text_parent_d_ordered)

            if self.plotter:
                self.plotter.save_plot_of_layout(text_regions_p, image_page)
                self.plotter.save_plot_of_layout_all(text_regions_p, image_page)

            K.clear_session()
            gc.collect()

            polygons_of_tabels = []
            pixel_img = 4
            polygons_of_drop_capitals = return_contours_of_interested_region_by_min_size(text_regions_p, pixel_img)
            all_found_texline_polygons = adhere_drop_capital_region_into_cprresponding_textline(text_regions_p, polygons_of_drop_capitals, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_texline_polygons, all_found_texline_polygons_h, kernel=self.kernel, curved_line=self.curved_line)

            # print(len(contours_only_text_parent_h),len(contours_only_text_parent_h_d_ordered),'contours_only_text_parent_h')
            pixel_lines = 6

            if not self.headers_off:
                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                    num_col, peaks_neg_fin, matrix_of_lines_ch, spliter_y_new, _ = find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines, contours_only_text_parent_h)
                else:
                    num_col_d, peaks_neg_fin_d, matrix_of_lines_ch_d, spliter_y_new_d, _ = find_number_of_columns_in_document(np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines, contours_only_text_parent_h_d_ordered)
            elif self.headers_off:
                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                    num_col, peaks_neg_fin, matrix_of_lines_ch, spliter_y_new, _ = find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines)
                else:
                    num_col_d, peaks_neg_fin_d, matrix_of_lines_ch_d, spliter_y_new_d, _ = find_number_of_columns_in_document(np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines)

            # print(peaks_neg_fin,peaks_neg_fin_d,'num_col2')
            # print(spliter_y_new,spliter_y_new_d,'num_col_classifier')
            # print(matrix_of_lines_ch.shape,matrix_of_lines_ch_d.shape,'matrix_of_lines_ch')

            if num_col_classifier >= 3:
                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                    regions_without_seperators = regions_without_seperators.astype(np.uint8)
                    regions_without_seperators = cv2.erode(regions_without_seperators[:, :], self.kernel, iterations=6)
                    random_pixels_for_image = np.random.randn(regions_without_seperators.shape[0], regions_without_seperators.shape[1])
                    random_pixels_for_image[random_pixels_for_image < -0.5] = 0
                    random_pixels_for_image[random_pixels_for_image != 0] = 1
                    regions_without_seperators[(random_pixels_for_image[:, :] == 1) & (text_regions_p[:, :] == 5)] = 1
                else:
                    regions_without_seperators_d = regions_without_seperators_d.astype(np.uint8)
                    regions_without_seperators_d = cv2.erode(regions_without_seperators_d[:, :], self.kernel, iterations=6)
                    random_pixels_for_image = np.random.randn(regions_without_seperators_d.shape[0], regions_without_seperators_d.shape[1])
                    random_pixels_for_image[random_pixels_for_image < -0.5] = 0
                    random_pixels_for_image[random_pixels_for_image != 0] = 1
                    regions_without_seperators_d[(random_pixels_for_image[:, :] == 1) & (text_regions_p_1_n[:, :] == 5)] = 1

            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                boxes = return_boxes_of_images_by_order_of_reading_new(spliter_y_new, regions_without_seperators, matrix_of_lines_ch, num_col_classifier)
            else:
                boxes_d = return_boxes_of_images_by_order_of_reading_new(spliter_y_new_d, regions_without_seperators_d, matrix_of_lines_ch_d, num_col_classifier)

        if self.plotter:
            self.plotter.write_images_into_directory(polygons_of_images, image_page)

        if self.full_layout:
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot)
            else:
                order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered, boxes_d, textline_mask_tot_d)

            self.write_into_page_xml_full(contours_only_text_parent, contours_only_text_parent_h, page_coord, self.dir_out, order_text_new, id_of_texts_tot, all_found_texline_polygons, all_found_texline_polygons_h, all_box_coord, all_box_coord_h, polygons_of_images, polygons_of_tabels, polygons_of_drop_capitals, polygons_of_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals)

        else:
            contours_only_text_parent_h = None
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot)
            else:
                contours_only_text_parent_d_ordered = list(np.array(contours_only_text_parent_d_ordered)[index_by_text_par_con])
                order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent_d_ordered, contours_only_text_parent_h, boxes_d, textline_mask_tot_d)
            self.write_into_page_xml(txt_con_org, page_coord, self.dir_out, order_text_new, id_of_texts_tot, all_found_texline_polygons, all_box_coord, polygons_of_images, polygons_of_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, self.curved_line, slopes, slopes_marginals)

        self.logger.info("Job done in %ss", str(time.time() - t1))
