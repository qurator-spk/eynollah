# pylint: disable=no-member,invalid-name,line-too-long,missing-function-docstring,missing-class-docstring,too-many-branches
# pylint: disable=too-many-locals,wrong-import-position,too-many-lines,too-many-statements,chained-comparison,fixme,broad-except,c-extension-no-member
# pylint: disable=too-many-public-methods,too-many-arguments,too-many-instance-attributes,too-many-public-methods,
# pylint: disable=consider-using-enumerate
"""
tool to extract table form data from alto xml data
"""

import math
import os
import sys
import time
import warnings
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count
import gc
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
    filter_contours_area_of_image,
    find_contours_mean_y_diff,
    find_new_features_of_contours,
    get_text_region_boxes_by_given_contours,
    get_textregion_contours_in_org_image,
    return_contours_of_image,
    return_contours_of_interested_region,
    return_contours_of_interested_region_by_min_size,
    return_contours_of_interested_textline,
    return_parent_contours,
)
from .utils.rotate import (
    rotate_image,
    rotation_not_90_func,
    rotation_not_90_func_full_layout)
from .utils.separate_lines import (
    textline_contours_postprocessing,
    separate_lines_new2,
    return_deskew_slop)
from .utils.drop_capitals import (
    adhere_drop_capital_region_into_corresponding_textline,
    filter_small_drop_capitals_from_no_patch_layout)
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
    small_textlines_to_parent_adherence2,
    order_of_regions,
    find_number_of_columns_in_document,
    return_boxes_of_images_by_order_of_reading_new)
from .utils.pil_cv2 import check_dpi, pil2cv
from .utils.xml import order_and_id_of_texts
from .plot import EynollahPlotter
from .writer import EynollahXmlWriter

SLOPE_THRESHOLD = 0.13
RATIO_OF_TWO_MODEL_THRESHOLD = 95.50 #98.45:
DPI_THRESHOLD = 298
MAX_SLOPE = 999
KERNEL = np.ones((5, 5), np.uint8)

class Eynollah:
    def __init__(
        self,
        dir_models,
        image_filename,
        image_pil=None,
        image_filename_stem=None,
        dir_out=None,
        dir_of_cropped_images=None,
        dir_of_layout=None,
        dir_of_deskewed=None,
        dir_of_all=None,
        enable_plotting=False,
        allow_enhancement=False,
        curved_line=False,
        full_layout=False,
        input_binary=False,
        allow_scaling=False,
        headers_off=False,
        override_dpi=None,
        logger=None,
        pcgts=None,
    ):
        if image_pil:
            self._imgs = self._cache_images(image_pil=image_pil)
        else:
            self._imgs = self._cache_images(image_filename=image_filename)
        if override_dpi:
            self.dpi = override_dpi
        self.image_filename = image_filename
        self.dir_out = dir_out
        self.allow_enhancement = allow_enhancement
        self.curved_line = curved_line
        self.full_layout = full_layout
        self.input_binary = input_binary
        self.allow_scaling = allow_scaling
        self.headers_off = headers_off
        self.plotter = None if not enable_plotting else EynollahPlotter(
            dir_of_all=dir_of_all,
            dir_of_deskewed=dir_of_deskewed,
            dir_of_cropped_images=dir_of_cropped_images,
            dir_of_layout=dir_of_layout,
            image_filename_stem=Path(Path(image_filename).name).stem)
        self.writer = EynollahXmlWriter(
            dir_out=self.dir_out,
            image_filename=self.image_filename,
            curved_line=self.curved_line,
            pcgts=pcgts)
        self.logger = logger if logger else getLogger('eynollah')
        self.dir_models = dir_models

        self.model_dir_of_enhancement = dir_models + "/model_enhancement.h5"
        self.model_dir_of_binarization = dir_models + "/model_bin_sbb_ens.h5"
        self.model_dir_of_col_classifier = dir_models + "/model_scale_classifier.h5"
        self.model_region_dir_p = dir_models + "/model_main_covid19_lr5-5_scale_1_1_great.h5"
        self.model_region_dir_p2 = dir_models + "/model_main_home_corona3_rot.h5"
        self.model_region_dir_fully_np = dir_models + "/model_no_patches_class0_30eopch.h5"
        self.model_region_dir_fully = dir_models + "/model_3up_new_good_no_augmentation.h5"
        self.model_page_dir = dir_models + "/model_page_mixed_best.h5"
        self.model_region_dir_p_ens = dir_models + "/model_ensemble_s.h5"
        self.model_textline_dir = dir_models + "/model_textline_newspapers.h5"
        
    def _cache_images(self, image_filename=None, image_pil=None):
        ret = {}
        if image_filename:
            ret['img'] = cv2.imread(image_filename)
            self.dpi = check_dpi(image_filename)
        else:
            ret['img'] = pil2cv(image_pil)
            self.dpi = check_dpi(image_pil)
        ret['img_grayscale'] = cv2.cvtColor(ret['img'], cv2.COLOR_BGR2GRAY)
        for prefix in ('',  '_grayscale'):
            ret[f'img{prefix}_uint8'] = ret[f'img{prefix}'].astype(np.uint8)
        return ret

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
        model_enhancement, session_enhancement = self.start_new_session_and_model(self.model_dir_of_enhancement)

        img_height_model = model_enhancement.layers[len(model_enhancement.layers) - 1].output_shape[1]
        img_width_model = model_enhancement.layers[len(model_enhancement.layers) - 1].output_shape[2]
        if img.shape[0] < img_height_model:
            img = cv2.resize(img, (img.shape[1], img_width_model), interpolation=cv2.INTER_NEAREST)

        if img.shape[1] < img_width_model:
            img = cv2.resize(img, (img_height_model, img.shape[0]), interpolation=cv2.INTER_NEAREST)
        margin = int(0 * img_width_model)
        width_mid = img_width_model - 2 * margin
        height_mid = img_height_model - 2 * margin
        img = img / float(255.0)

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
        session_enhancement.close()
        del model_enhancement
        del session_enhancement
        gc.collect()

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

    def resize_image_with_column_classifier(self, is_image_enhanced, img_bin):
        self.logger.debug("enter resize_image_with_column_classifier")
        if self.input_binary:
            img = np.copy(img_bin)
        else:
            img = self.imread()

        _, page_coord = self.early_page_for_num_of_column_classification(img)
        model_num_classifier, session_col_classifier = self.start_new_session_and_model(self.model_dir_of_col_classifier)
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

        label_p_pred = model_num_classifier.predict(img_in)
        num_col = np.argmax(label_p_pred[0]) + 1

        self.logger.info("Found %s columns (%s)", num_col, label_p_pred)

        session_col_classifier.close()
        
        del model_num_classifier
        del session_col_classifier
        
        K.clear_session()
        gc.collect()



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
            model_bin, session_bin = self.start_new_session_and_model(self.model_dir_of_binarization)
            prediction_bin = self.do_prediction(True, img, model_bin)
            
            prediction_bin=prediction_bin[:,:,0]
            prediction_bin = (prediction_bin[:,:]==0)*1
            prediction_bin = prediction_bin*255
            
            prediction_bin =np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2)

            session_bin.close()
            del model_bin
            del session_bin
            gc.collect()
            
            prediction_bin = prediction_bin.astype(np.uint8)
            img= np.copy(prediction_bin)
            img_bin = np.copy(prediction_bin)
        else:
            img = self.imread()
            img_bin = None

        _, page_coord = self.early_page_for_num_of_column_classification(img_bin)
        model_num_classifier, session_col_classifier = self.start_new_session_and_model(self.model_dir_of_col_classifier)
        
        if self.input_binary:
            img_in = np.copy(img)
            width_early = img_in.shape[1]
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



        label_p_pred = model_num_classifier.predict(img_in)
        num_col = np.argmax(label_p_pred[0]) + 1
        self.logger.info("Found %s columns (%s)", num_col, label_p_pred)
        session_col_classifier.close()
        K.clear_session()

        if dpi < DPI_THRESHOLD:
            img_new, num_column_is_classified = self.calculate_width_height_by_columns(img, num_col, width_early, label_p_pred)
            image_res = self.predict_enhancement(img_new)
            is_image_enhanced = True
        else:
            is_image_enhanced = False
            num_column_is_classified = True
            image_res = np.copy(img)
            
        session_col_classifier.close()

        
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

    def start_new_session_and_model_old(self, model_dir):
        self.logger.debug("enter start_new_session_and_model (model_dir=%s)", model_dir)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        session = tf.InteractiveSession()
        model = load_model(model_dir, compile=False)

        return model, session

    
    def start_new_session_and_model(self, model_dir):
        self.logger.debug("enter start_new_session_and_model (model_dir=%s)", model_dir)
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        #gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=7.7, allow_growth=True)
        session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        model = load_model(model_dir, compile=False)

        return model, session

    def do_prediction(self, patches, img, model, marginal_of_patch_percent=0.1):
        self.logger.debug("enter do_prediction")

        img_height_model = model.layers[len(model.layers) - 1].output_shape[1]
        img_width_model = model.layers[len(model.layers) - 1].output_shape[2]

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
        del model
        gc.collect()
        return prediction_true

    def early_page_for_num_of_column_classification(self,img_bin):
        self.logger.debug("enter early_page_for_num_of_column_classification")
        if self.input_binary:
            img =np.copy(img_bin)
            img = img.astype(np.uint8)
        else:
            img = self.imread()
        model_page, session_page = self.start_new_session_and_model(self.model_page_dir)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        img_page_prediction = self.do_prediction(False, img, model_page)

        imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)
        thresh = cv2.dilate(thresh, KERNEL, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])
            cnt = contours[np.argmax(cnt_size)]
            x, y, w, h = cv2.boundingRect(cnt)
            box = [x, y, w, h]
        else:
            box = [0, 0, img.shape[1], img.shape[0]]
        croped_page, page_coord = crop_image_inside_box(box, img)
        session_page.close()
        del model_page
        del session_page
        gc.collect()
        K.clear_session()
        self.logger.debug("exit early_page_for_num_of_column_classification")
        return croped_page, page_coord

    def extract_page(self):
        self.logger.debug("enter extract_page")
        cont_page = []
        model_page, session_page = self.start_new_session_and_model(self.model_page_dir)
        img = cv2.GaussianBlur(self.image, (5, 5), 0)
        img_page_prediction = self.do_prediction(False, img, model_page)
        imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)
        thresh = cv2.dilate(thresh, KERNEL, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours)>0:
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
        else:
            box = [0, 0, img.shape[1], img.shape[0]]
        croped_page, page_coord = crop_image_inside_box(box, self.image)
        cont_page.append(np.array([[page_coord[2], page_coord[0]], [page_coord[3], page_coord[0]], [page_coord[3], page_coord[1]], [page_coord[2], page_coord[1]]]))
        session_page.close()
        del model_page
        del session_page
        gc.collect()
        K.clear_session()
        self.logger.debug("exit extract_page")
        return croped_page, page_coord, cont_page

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
                    img = otsu_copy_binary(img)
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

        textline_cnt_separated = np.zeros(textline_mask_tot_ea.shape)

        for mv in range(len(boxes_text)):

            all_text_region_raw = textline_mask_tot_ea[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]]
            all_text_region_raw = all_text_region_raw.astype(np.uint8)
            img_int_p = all_text_region_raw[:, :]

            # img_int_p=cv2.erode(img_int_p,KERNEL,iterations = 2)
            # plt.imshow(img_int_p)
            # plt.show()

            if img_int_p.shape[0] / img_int_p.shape[1] < 0.1:
                slopes_per_each_subprocess.append(0)
                slope_for_all = [slope_deskew][0]
            else:
                try:
                    textline_con, hierarchy = return_contours_of_image(img_int_p)
                    textline_con_fil = filter_contours_area_of_image(img_int_p, textline_con, hierarchy, max_area=1, min_area=0.0008)
                    y_diff_mean = find_contours_mean_y_diff(textline_con_fil)
                    if self.isNaN(y_diff_mean):
                        slope_for_all = MAX_SLOPE
                    else:
                        sigma_des = max(1, int(y_diff_mean * (4.0 / 40.0)))
                        img_int_p[img_int_p > 0] = 1
                        slope_for_all = return_deskew_slop(img_int_p, sigma_des, plotter=self.plotter)

                        if abs(slope_for_all) < 0.5:
                            slope_for_all = [slope_deskew][0]

                except Exception as why:
                    self.logger.error(why)
                    slope_for_all = MAX_SLOPE

                if slope_for_all == MAX_SLOPE:
                    slope_for_all = [slope_deskew][0]
                slopes_per_each_subprocess.append(slope_for_all)

            index_by_text_region_contours.append(indexes_r_con_per_pro[mv])
            _, crop_coor = crop_image_inside_box(boxes_text[mv], image_page_rotated)

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
                textline_rotated_separated = separate_lines_new2(textline_biggest_region[y : y + h, x : x + w], 0, num_col, slope_for_all, plotter=self.plotter)

                # new line added
                ##print(np.shape(textline_rotated_separated),np.shape(mask_biggest))
                textline_rotated_separated[mask_region_in_patch_region[:, :] != 1] = 0
                # till here

                textline_cnt_separated[y : y + h, x : x + w] = textline_rotated_separated
                textline_region_in_image[y : y + h, x : x + w] = textline_rotated_separated

                # plt.imshow(textline_region_in_image)
                # plt.show()
                # plt.imshow(textline_cnt_separated)
                # plt.show()

                pixel_img = 1
                cnt_textlines_in_image = return_contours_of_interested_textline(textline_region_in_image, pixel_img)

                textlines_cnt_per_region = []
                for jjjj in range(len(cnt_textlines_in_image)):
                    mask_biggest2 = np.zeros(mask_texts_only.shape)
                    mask_biggest2 = cv2.fillPoly(mask_biggest2, pts=[cnt_textlines_in_image[jjjj]], color=(1, 1, 1))
                    if num_col + 1 == 1:
                        mask_biggest2 = cv2.dilate(mask_biggest2, KERNEL, iterations=5)
                    else:
                        mask_biggest2 = cv2.dilate(mask_biggest2, KERNEL, iterations=4)

                    pixel_img = 1
                    mask_biggest2 = resize_image(mask_biggest2, int(mask_biggest2.shape[0] * scale_par), int(mask_biggest2.shape[1] * scale_par))
                    cnt_textlines_in_image_ind = return_contours_of_interested_textline(mask_biggest2, pixel_img)
                    try:
                        textlines_cnt_per_region.append(cnt_textlines_in_image_ind[0])
                    except Exception as why:
                        self.logger.error(why)
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
            _, crop_coor = crop_image_inside_box(boxes_text[mv],image_page_rotated)
            mask_textline = np.zeros((textline_mask_tot_ea.shape))
            mask_textline = cv2.fillPoly(mask_textline,pts=[contours_per_process[mv]],color=(1,1,1))
            all_text_region_raw = (textline_mask_tot_ea*mask_textline[:,:])[boxes_text[mv][1]:boxes_text[mv][1]+boxes_text[mv][3] , boxes_text[mv][0]:boxes_text[mv][0]+boxes_text[mv][2] ]
            all_text_region_raw=all_text_region_raw.astype(np.uint8)
            img_int_p=all_text_region_raw[:,:]#self.all_text_region_raw[mv]
            img_int_p=cv2.erode(img_int_p,KERNEL,iterations = 2)

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
                    textline_con, hierarchy = return_contours_of_image(img_int_p)
                    textline_con_fil = filter_contours_area_of_image(img_int_p, textline_con, hierarchy, max_area=1, min_area=0.00008)
                    y_diff_mean = find_contours_mean_y_diff(textline_con_fil)
                    if self.isNaN(y_diff_mean):
                        slope_for_all = MAX_SLOPE
                    else:
                        sigma_des = int(y_diff_mean * (4.0 / 40.0))
                        if sigma_des < 1:
                            sigma_des = 1
                        img_int_p[img_int_p > 0] = 1
                        slope_for_all = return_deskew_slop(img_int_p, sigma_des, plotter=self.plotter)
                        if abs(slope_for_all) <= 0.5:
                            slope_for_all = [slope_deskew][0]
                except Exception as why:
                    self.logger.error(why)
                    slope_for_all = MAX_SLOPE
                if slope_for_all == MAX_SLOPE:
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

        session_textline.close()


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
            crop_img = cv2.erode(crop_img, KERNEL, iterations=2)
            try:
                textline_con, hierarchy = return_contours_of_image(crop_img)
                textline_con_fil = filter_contours_area_of_image(crop_img, textline_con, hierarchy, max_area=1, min_area=0.0008)
                y_diff_mean = find_contours_mean_y_diff(textline_con_fil)
                sigma_des = max(1, int(y_diff_mean * (4.0 / 40.0)))
                crop_img[crop_img > 0] = 1
                slope_corresponding_textregion = return_deskew_slop(crop_img, sigma_des, plotter=self.plotter)
            except Exception as why:
                self.logger.error(why)
                slope_corresponding_textregion = MAX_SLOPE

            if slope_corresponding_textregion == MAX_SLOPE:
                slope_corresponding_textregion = slope_biggest
            slopes_sub.append(slope_corresponding_textregion)

            cnt_clean_rot = textline_contours_postprocessing(crop_img, slope_corresponding_textregion, contours_per_process[mv], boxes_per_process[mv])

            poly_sub.append(cnt_clean_rot)
            boxes_sub_new.append(boxes_per_process[mv])

        q.put(slopes_sub)
        poly.put(poly_sub)
        box_sub.put(boxes_sub_new)

    def get_regions_from_xy_2models(self,img,is_image_enhanced, num_col_classifier):
        self.logger.debug("enter get_regions_from_xy_2models")
        erosion_hurts = False
        img_org = np.copy(img)
        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]

        model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_ens)

        ratio_y=1.3
        ratio_x=1

        img = resize_image(img_org, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))

        prediction_regions_org_y = self.do_prediction(True, img, model_region)
        prediction_regions_org_y = resize_image(prediction_regions_org_y, img_height_h, img_width_h )

        #plt.imshow(prediction_regions_org_y[:,:,0])
        #plt.show()
        prediction_regions_org_y = prediction_regions_org_y[:,:,0]
        mask_zeros_y = (prediction_regions_org_y[:,:]==0)*1
        
        ##img_only_regions_with_sep = ( (prediction_regions_org_y[:,:] != 3) & (prediction_regions_org_y[:,:] != 0) )*1
        img_only_regions_with_sep = ( prediction_regions_org_y[:,:] == 1 )*1
        img_only_regions_with_sep = img_only_regions_with_sep.astype(np.uint8)
        
        try:
            img_only_regions = cv2.erode(img_only_regions_with_sep[:,:], KERNEL, iterations=20)

            _, _ = find_num_col(img_only_regions, multiplier=6.0)
            
            img = resize_image(img_org, int(img_org.shape[0]), int(img_org.shape[1]*(1.2 if is_image_enhanced else 1)))

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
            img = resize_image(img_org, int(img_org.shape[0]), int(img_org.shape[1]))
            prediction_regions_org2 = self.do_prediction(True, img, model_region, 0.2)
            prediction_regions_org2=resize_image(prediction_regions_org2, img_height_h, img_width_h )


            session_region.close()
            del model_region
            del session_region
            gc.collect()

            mask_zeros2 = (prediction_regions_org2[:,:,0] == 0)
            mask_lines2 = (prediction_regions_org2[:,:,0] == 3)
            text_sume_early = (prediction_regions_org[:,:] == 1).sum()
            prediction_regions_org_copy = np.copy(prediction_regions_org)
            prediction_regions_org_copy[(prediction_regions_org_copy[:,:]==1) & (mask_zeros2[:,:]==1)] = 0
            text_sume_second = ((prediction_regions_org_copy[:,:]==1)*1).sum()

            rate_two_models = text_sume_second / float(text_sume_early) * 100

            self.logger.info("ratio_of_two_models: %s", rate_two_models)
            if not(is_image_enhanced and rate_two_models < RATIO_OF_TWO_MODEL_THRESHOLD):
                prediction_regions_org = np.copy(prediction_regions_org_copy)
                
            

            prediction_regions_org[(mask_lines2[:,:]==1) & (prediction_regions_org[:,:]==0)]=3
            mask_lines_only=(prediction_regions_org[:,:]==3)*1
            prediction_regions_org = cv2.erode(prediction_regions_org[:,:], KERNEL, iterations=2)

            #plt.imshow(text_region2_1st_channel)
            #plt.show()

            prediction_regions_org = cv2.dilate(prediction_regions_org[:,:], KERNEL, iterations=2)
            
            
            if rate_two_models<=40:
                if self.input_binary:
                    prediction_bin = np.copy(img_org)
                else:
                    model_bin, session_bin = self.start_new_session_and_model(self.model_dir_of_binarization)
                    prediction_bin = self.do_prediction(True, img_org, model_bin)
                    prediction_bin = resize_image(prediction_bin, img_height_h, img_width_h )
                    
                    prediction_bin=prediction_bin[:,:,0]
                    prediction_bin = (prediction_bin[:,:]==0)*1
                    prediction_bin = prediction_bin*255
                    
                    prediction_bin =np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2)

                    session_bin.close()
                    del model_bin
                    del session_bin
                    gc.collect()
                
                
                
                model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_ens)
                ratio_y=1
                ratio_x=1


                img = resize_image(prediction_bin, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))

                prediction_regions_org = self.do_prediction(True, img, model_region)
                prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h )
                prediction_regions_org=prediction_regions_org[:,:,0]
                
                mask_lines_only=(prediction_regions_org[:,:]==3)*1
                session_region.close()
                del model_region
                del session_region
                gc.collect()
                
                
            mask_texts_only=(prediction_regions_org[:,:]==1)*1
            mask_images_only=(prediction_regions_org[:,:]==2)*1
            
            
            
            polygons_lines_xml, hir_lines_xml = return_contours_of_image(mask_lines_only)
            polygons_lines_xml = textline_con_fil = filter_contours_area_of_image(mask_lines_only, polygons_lines_xml, hir_lines_xml, max_area=1, min_area=0.00001)

            polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only, 1, 0.00001)
            polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only, 1, 0.00001)

            text_regions_p_true = np.zeros(prediction_regions_org.shape)
            text_regions_p_true = cv2.fillPoly(text_regions_p_true,pts = polygons_of_only_lines, color=(3, 3, 3))
            text_regions_p_true[:,:][mask_images_only[:,:] == 1] = 2

            text_regions_p_true=cv2.fillPoly(text_regions_p_true,pts=polygons_of_only_texts, color=(1,1,1))

            

            K.clear_session()
            return text_regions_p_true, erosion_hurts, polygons_lines_xml
        except:
            
            if self.input_binary:
                prediction_bin = np.copy(img_org)
            else:
                session_region.close()
                del model_region
                del session_region
                gc.collect()
                
                model_bin, session_bin = self.start_new_session_and_model(self.model_dir_of_binarization)
                prediction_bin = self.do_prediction(True, img_org, model_bin)
                prediction_bin = resize_image(prediction_bin, img_height_h, img_width_h )
                prediction_bin=prediction_bin[:,:,0]
                
                prediction_bin = (prediction_bin[:,:]==0)*1
                
                prediction_bin = prediction_bin*255
                
                prediction_bin =np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2)

                
                
                session_bin.close()
                del model_bin
                del session_bin
                gc.collect()
            
            
            
                model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_ens)
            ratio_y=1
            ratio_x=1


            img = resize_image(prediction_bin, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))

            prediction_regions_org = self.do_prediction(True, img, model_region)
            prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h )
            prediction_regions_org=prediction_regions_org[:,:,0]
            
            #mask_lines_only=(prediction_regions_org[:,:]==3)*1
            session_region.close()
            del model_region
            del session_region
            gc.collect()
            
            #img = resize_image(img_org, int(img_org.shape[0]*1), int(img_org.shape[1]*1))
            
            #prediction_regions_org = self.do_prediction(True, img, model_region)
            
            #prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h )
            
            #prediction_regions_org = prediction_regions_org[:,:,0]
            
            #prediction_regions_org[(prediction_regions_org[:,:] == 1) & (mask_zeros_y[:,:] == 1)]=0
            #session_region.close()
            #del model_region
            #del session_region
            #gc.collect()
            
            
            
            
            mask_lines_only = (prediction_regions_org[:,:] ==3)*1
            
            mask_texts_only = (prediction_regions_org[:,:] ==1)*1
            
            mask_images_only=(prediction_regions_org[:,:] ==2)*1
            
            polygons_lines_xml, hir_lines_xml = return_contours_of_image(mask_lines_only)
            polygons_lines_xml = textline_con_fil = filter_contours_area_of_image(mask_lines_only, polygons_lines_xml, hir_lines_xml, max_area=1, min_area=0.00001)
            
            
            polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only,1,0.00001)
            
            polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only,1,0.00001)
            
            
            text_regions_p_true = np.zeros(prediction_regions_org.shape)
            
            text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_lines, color=(3,3,3))
            
            text_regions_p_true[:,:][mask_images_only[:,:] == 1] = 2
            
            text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_texts, color=(1,1,1))
            
            erosion_hurts = True
            K.clear_session()
            return text_regions_p_true, erosion_hurts, polygons_lines_xml

    def do_order_of_regions_full_layout(self, contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot):
        self.logger.debug("enter do_order_of_regions_full_layout")
        cx_text_only, cy_text_only, x_min_text_only, _, _, _, y_cor_x_min_main = find_new_features_of_contours(contours_only_text_parent)
        cx_text_only_h, cy_text_only_h, x_min_text_only_h, _, _, _, y_cor_x_min_main_h = find_new_features_of_contours(contours_only_text_parent_h)

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

                for box in args_contours_box:
                    con_inter_box.append(contours_only_text_parent[box])

                for box in args_contours_box_h:
                    con_inter_box_h.append(contours_only_text_parent_h[box])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_sorted_head = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 2]
                indexes_by_type_head = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 2]

                for zahler, _ in enumerate(args_contours_box):
                    arg_order_v = indexes_sorted_main[zahler]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for zahler, _ in enumerate(args_contours_box_h):
                    arg_order_v = indexes_sorted_head[zahler]
                    order_by_con_head[args_contours_box_h[indexes_by_type_head[zahler]]] = np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

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
            args_contours_h = np.array(range(len(arg_text_con_h)))

            order_by_con_head = np.zeros(len(arg_text_con_h))

            ref_point = 0
            order_of_texts_tot = []
            id_of_texts_tot = []
            for iij, _ in enumerate(boxes):
                args_contours_box = args_contours[np.array(arg_text_con) == iij]
                args_contours_box_h = args_contours_h[np.array(arg_text_con_h) == iij]
                con_inter_box = []
                con_inter_box_h = []

                for box in args_contours_box:
                    con_inter_box.append(contours_only_text_parent[box])

                for box in args_contours_box_h:
                    con_inter_box_h.append(contours_only_text_parent_h[box])

                indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                order_of_texts, id_of_texts = order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

                indexes_sorted_main = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_by_type_main = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 1]
                indexes_sorted_head = np.array(indexes_sorted)[np.array(kind_of_texts_sorted) == 2]
                indexes_by_type_head = np.array(index_by_kind_sorted)[np.array(kind_of_texts_sorted) == 2]

                for zahler, _ in enumerate(args_contours_box):
                    arg_order_v = indexes_sorted_main[zahler]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

                for zahler, _ in enumerate(args_contours_box_h):
                    arg_order_v = indexes_sorted_head[zahler]
                    order_by_con_head[args_contours_box_h[indexes_by_type_head[zahler]]] = np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

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
        return order_text_new, id_of_texts_tot

    def do_order_of_regions_no_full_layout(self, contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot):
        self.logger.debug("enter do_order_of_regions_no_full_layout")
        cx_text_only, cy_text_only, x_min_text_only, _, _, _, y_cor_x_min_main = find_new_features_of_contours(contours_only_text_parent)

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

                for zahler, _ in enumerate(args_contours_box):
                    arg_order_v = indexes_sorted_main[zahler]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

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

                for zahler, _ in enumerate(args_contours_box):
                    arg_order_v = indexes_sorted_main[zahler]
                    order_by_con_main[args_contours_box[indexes_by_type_main[zahler]]] = np.where(indexes_sorted == arg_order_v)[0][0] + ref_point

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
        
        return order_text_new, id_of_texts_tot

    def do_order_of_regions(self, *args, **kwargs):
        if self.full_layout:
            return self.do_order_of_regions_full_layout(*args, **kwargs)
        return self.do_order_of_regions_no_full_layout(*args, **kwargs)

    def run_graphics_and_columns(self, text_regions_p_1, num_col_classifier, num_column_is_classified, erosion_hurts):
        img_g = self.imread(grayscale=True, uint8=True)

        img_g3 = np.zeros((img_g.shape[0], img_g.shape[1], 3))
        img_g3 = img_g3.astype(np.uint8)
        img_g3[:, :, 0] = img_g[:, :]
        img_g3[:, :, 1] = img_g[:, :]
        img_g3[:, :, 2] = img_g[:, :]

        image_page, page_coord, cont_page = self.extract_page()
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
            num_col, _ = find_num_col(img_only_regions, multiplier=6.0)
            num_col = num_col + 1
            if not num_column_is_classified:
                num_col_classifier = num_col + 1
        except Exception as why:
            self.logger.error(why)
            num_col = None
        return num_col, num_col_classifier, img_only_regions, page_coord, image_page, mask_images, mask_lines, text_regions_p_1, cont_page

    def run_enhancement(self):
        self.logger.info("resize and enhance image")
        is_image_enhanced, img_org, img_res, num_col_classifier, num_column_is_classified, img_bin = self.resize_and_enhance_image_with_column_classifier()
        self.logger.info("Image is %senhanced", '' if is_image_enhanced else 'not ')
        K.clear_session()
        scale = 1
        if is_image_enhanced:
            if self.allow_enhancement:
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
                img_org, img_res, is_image_enhanced = self.resize_image_with_column_classifier(is_image_enhanced, img_bin)
                self.get_image_and_scales_after_enhancing(img_org, img_res)
        return img_res, is_image_enhanced, num_col_classifier, num_column_is_classified

    def run_textline(self, image_page):
        scaler_h_textline = 1  # 1.2#1.2
        scaler_w_textline = 1  # 0.9#1
        textline_mask_tot_ea, _ = self.textline_contours(image_page, True, scaler_h_textline, scaler_w_textline)
        K.clear_session()
        if self.plotter:
            self.plotter.save_plot_of_textlines(textline_mask_tot_ea, image_page)
        return textline_mask_tot_ea

    def run_deskew(self, textline_mask_tot_ea):
        sigma = 2
        main_page_deskew = True
        slope_deskew = return_deskew_slop(cv2.erode(textline_mask_tot_ea, KERNEL, iterations=2), sigma, main_page_deskew, plotter=self.plotter)
        slope_first = 0

        if self.plotter:
            self.plotter.save_deskewed_image(slope_deskew)
        self.logger.info("slope_deskew: %s", slope_deskew)
        return slope_deskew, slope_first

    def run_marginals(self, image_page, textline_mask_tot_ea, mask_images, mask_lines, num_col_classifier, slope_deskew, text_regions_p_1):
        image_page_rotated, textline_mask_tot = image_page[:, :], textline_mask_tot_ea[:, :]
        textline_mask_tot[mask_images[:, :] == 1] = 0

        text_regions_p_1[mask_lines[:, :] == 1] = 3
        text_regions_p = text_regions_p_1[:, :]
        text_regions_p = np.array(text_regions_p)

        if num_col_classifier in (1, 2):
            try:
                regions_without_separators = (text_regions_p[:, :] == 1) * 1
                regions_without_separators = regions_without_separators.astype(np.uint8)
                text_regions_p = get_marginals(rotate_image(regions_without_separators, slope_deskew), text_regions_p, num_col_classifier, slope_deskew, kernel=KERNEL)
            except Exception as e:
                self.logger.error("exception %s", e)

        if self.plotter:
            self.plotter.save_plot_of_layout_main_all(text_regions_p, image_page)
            self.plotter.save_plot_of_layout_main(text_regions_p, image_page)
        return textline_mask_tot, text_regions_p, image_page_rotated

    def run_boxes_no_full_layout(self, image_page, textline_mask_tot, text_regions_p, slope_deskew, num_col_classifier, erosion_hurts):
        self.logger.debug('enter run_boxes_no_full_layout')
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            _, textline_mask_tot_d, text_regions_p_1_n = rotation_not_90_func(image_page, textline_mask_tot, text_regions_p, slope_deskew)
            text_regions_p_1_n = resize_image(text_regions_p_1_n, text_regions_p.shape[0], text_regions_p.shape[1])
            textline_mask_tot_d = resize_image(textline_mask_tot_d, text_regions_p.shape[0], text_regions_p.shape[1])
            regions_without_separators_d = (text_regions_p_1_n[:, :] == 1) * 1
        regions_without_separators = (text_regions_p[:, :] == 1) * 1  # ( (text_regions_p[:,:]==1) | (text_regions_p[:,:]==2) )*1 #self.return_regions_without_separators_new(text_regions_p[:,:,0],img_only_regions)
        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
            text_regions_p_1_n = None
            textline_mask_tot_d = None
            regions_without_separators_d = None
        pixel_lines = 3
        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
            _, _, matrix_of_lines_ch, splitter_y_new, _ = find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines)

        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            _, _, matrix_of_lines_ch_d, splitter_y_new_d, _ = find_number_of_columns_in_document(np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines)
        K.clear_session()

        self.logger.info("num_col_classifier: %s", num_col_classifier)

        if num_col_classifier >= 3:
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                regions_without_separators = regions_without_separators.astype(np.uint8)
                regions_without_separators = cv2.erode(regions_without_separators[:, :], KERNEL, iterations=6)
            else:
                regions_without_separators_d = regions_without_separators_d.astype(np.uint8)
                regions_without_separators_d = cv2.erode(regions_without_separators_d[:, :], KERNEL, iterations=6)
        t1 = time.time()
        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
            boxes = return_boxes_of_images_by_order_of_reading_new(splitter_y_new, regions_without_separators, matrix_of_lines_ch, num_col_classifier, erosion_hurts)
            boxes_d = None
            self.logger.debug("len(boxes): %s", len(boxes))
        else:
            boxes_d = return_boxes_of_images_by_order_of_reading_new(splitter_y_new_d, regions_without_separators_d, matrix_of_lines_ch_d, num_col_classifier, erosion_hurts)
            boxes = None
            self.logger.debug("len(boxes): %s", len(boxes_d))

        self.logger.info("detecting boxes took %ss", str(time.time() - t1))
        img_revised_tab = text_regions_p[:, :]
        polygons_of_images = return_contours_of_interested_region(img_revised_tab, 2)

        # plt.imshow(img_revised_tab)
        # plt.show()
        K.clear_session()
        self.logger.debug('exit run_boxes_no_full_layout')
        return polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_separators_d, boxes, boxes_d

    def run_boxes_full_layout(self, image_page, textline_mask_tot, text_regions_p, slope_deskew, num_col_classifier, img_only_regions):
        self.logger.debug('enter run_boxes_full_layout')
        # set first model with second model
        text_regions_p[:, :][text_regions_p[:, :] == 2] = 5
        text_regions_p[:, :][text_regions_p[:, :] == 3] = 6
        text_regions_p[:, :][text_regions_p[:, :] == 4] = 8

        K.clear_session()
        image_page = image_page.astype(np.uint8)

        regions_fully, regions_fully_only_drop = self.extract_text_regions(image_page, True, cols=num_col_classifier)
        text_regions_p[:,:][regions_fully[:,:,0]==6]=6
        regions_fully_only_drop = put_drop_out_from_only_drop_model(regions_fully_only_drop, text_regions_p)
        regions_fully[:, :, 0][regions_fully_only_drop[:, :, 0] == 4] = 4
        K.clear_session()

        # plt.imshow(regions_fully[:,:,0])
        # plt.show()
        regions_fully = putt_bb_of_drop_capitals_of_model_in_patches_in_layout(regions_fully)
        # plt.imshow(regions_fully[:,:,0])
        # plt.show()
        K.clear_session()
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
            _, textline_mask_tot_d, text_regions_p_1_n, regions_fully_n = rotation_not_90_func_full_layout(image_page, textline_mask_tot, text_regions_p, regions_fully, slope_deskew)

            text_regions_p_1_n = resize_image(text_regions_p_1_n, text_regions_p.shape[0], text_regions_p.shape[1])
            textline_mask_tot_d = resize_image(textline_mask_tot_d, text_regions_p.shape[0], text_regions_p.shape[1])
            regions_fully_n = resize_image(regions_fully_n, text_regions_p.shape[0], text_regions_p.shape[1])
            regions_without_separators_d = (text_regions_p_1_n[:, :] == 1) * 1
        else:
            text_regions_p_1_n = None
            textline_mask_tot_d = None
            regions_without_separators_d = None

        regions_without_separators = (text_regions_p[:, :] == 1) * 1  # ( (text_regions_p[:,:]==1) | (text_regions_p[:,:]==2) )*1 #self.return_regions_without_separators_new(text_regions_p[:,:,0],img_only_regions)

        K.clear_session()
        img_revised_tab = np.copy(text_regions_p[:, :])
        polygons_of_images = return_contours_of_interested_region(img_revised_tab, 5)
        self.logger.debug('exit run_boxes_full_layout')
        return polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_separators_d, regions_fully, regions_without_separators

    def run(self):
        """
        Get image and scales, then extract the page of scanned image
        """
        self.logger.debug("enter run")

        t0 = time.time()
        img_res, is_image_enhanced, num_col_classifier, num_column_is_classified = self.run_enhancement()
        
        self.logger.info("Enhancing took %ss ", str(time.time() - t0))

        t1 = time.time()
        text_regions_p_1 ,erosion_hurts, polygons_lines_xml = self.get_regions_from_xy_2models(img_res, is_image_enhanced, num_col_classifier)
        self.logger.info("Textregion detection took %ss ", str(time.time() - t1))

        t1 = time.time()
        num_col, num_col_classifier, img_only_regions, page_coord, image_page, mask_images, mask_lines, text_regions_p_1, cont_page = \
                self.run_graphics_and_columns(text_regions_p_1, num_col_classifier, num_column_is_classified, erosion_hurts)
        self.logger.info("Graphics detection took %ss ", str(time.time() - t1))
        self.logger.info('cont_page %s', cont_page)

        if not num_col:
            self.logger.info("No columns detected, outputting an empty PAGE-XML")
            pcgts = self.writer.build_pagexml_no_full_layout([], page_coord, [], [], [], [], [], [], [], [], [], [], cont_page, [])
            self.logger.info("Job done in %ss", str(time.time() - t1))
            return pcgts

        t1 = time.time()
        textline_mask_tot_ea = self.run_textline(image_page)
        self.logger.info("textline detection took %ss", str(time.time() - t1))

        t1 = time.time()
        slope_deskew, slope_first = self.run_deskew(textline_mask_tot_ea)
        self.logger.info("deskewing took %ss", str(time.time() - t1))
        t1 = time.time()

        textline_mask_tot, text_regions_p, image_page_rotated = self.run_marginals(image_page, textline_mask_tot_ea, mask_images, mask_lines, num_col_classifier, slope_deskew, text_regions_p_1)
        self.logger.info("detection of marginals took %ss", str(time.time() - t1))
        t1 = time.time()

        if not self.full_layout:
            polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_separators_d, boxes, boxes_d = self.run_boxes_no_full_layout(image_page, textline_mask_tot, text_regions_p, slope_deskew, num_col_classifier, erosion_hurts)

        pixel_img = 4
        min_area_mar = 0.00001
        polygons_of_marginals = return_contours_of_interested_region(text_regions_p, pixel_img, min_area_mar)
        
        if self.full_layout:
            polygons_of_images, img_revised_tab, text_regions_p_1_n, textline_mask_tot_d, regions_without_separators_d, regions_fully, regions_without_separators = self.run_boxes_full_layout(image_page, textline_mask_tot, text_regions_p, slope_deskew, num_col_classifier, img_only_regions)

        text_only = ((img_revised_tab[:, :] == 1)) * 1
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            text_only_d = ((text_regions_p_1_n[:, :] == 1)) * 1

        min_con_area = 0.000005
        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
            contours_only_text, hir_on_text = return_contours_of_image(text_only)
            contours_only_text_parent = return_parent_contours(contours_only_text, hir_on_text)
                        
            if len(contours_only_text_parent) > 0:
                areas_cnt_text = np.array([cv2.contourArea(contours_only_text_parent[j]) for j in range(len(contours_only_text_parent))])
                areas_cnt_text = areas_cnt_text / float(text_only.shape[0] * text_only.shape[1])
                self.logger.info('areas_cnt_text %s', areas_cnt_text)
                contours_biggest = contours_only_text_parent[np.argmax(areas_cnt_text)]
                contours_only_text_parent = [contours_only_text_parent[jz] for jz in range(len(contours_only_text_parent)) if areas_cnt_text[jz] > min_con_area]
                areas_cnt_text_parent = [areas_cnt_text[jz] for jz in range(len(areas_cnt_text)) if areas_cnt_text[jz] > min_con_area]

                index_con_parents = np.argsort(areas_cnt_text_parent)
                contours_only_text_parent = list(np.array(contours_only_text_parent)[index_con_parents])
                areas_cnt_text_parent = list(np.array(areas_cnt_text_parent)[index_con_parents])

                cx_bigest_big, cy_biggest_big, _, _, _, _, _ = find_new_features_of_contours([contours_biggest])
                cx_bigest, cy_biggest, _, _, _, _, _ = find_new_features_of_contours(contours_only_text_parent)

                contours_only_text_d, hir_on_text_d = return_contours_of_image(text_only_d)
                contours_only_text_parent_d = return_parent_contours(contours_only_text_d, hir_on_text_d)

                areas_cnt_text_d = np.array([cv2.contourArea(contours_only_text_parent_d[j]) for j in range(len(contours_only_text_parent_d))])
                areas_cnt_text_d = areas_cnt_text_d / float(text_only_d.shape[0] * text_only_d.shape[1])
                
                if len(areas_cnt_text_d)>0:
                    contours_biggest_d = contours_only_text_parent_d[np.argmax(areas_cnt_text_d)]
                    index_con_parents_d=np.argsort(areas_cnt_text_d)
                    contours_only_text_parent_d=list(np.array(contours_only_text_parent_d)[index_con_parents_d] )
                    areas_cnt_text_d=list(np.array(areas_cnt_text_d)[index_con_parents_d] )

                    cx_bigest_d_big, cy_biggest_d_big, _, _, _, _, _ = find_new_features_of_contours([contours_biggest_d])
                    cx_bigest_d, cy_biggest_d, _, _, _, _, _ = find_new_features_of_contours(contours_only_text_parent_d)
                    try:
                        if len(cx_bigest_d) >= 5:
                            cx_bigest_d_last5 = cx_bigest_d[-5:]
                            cy_biggest_d_last5 = cy_biggest_d[-5:]
                            dists_d = [math.sqrt((cx_bigest_big[0] - cx_bigest_d_last5[j]) ** 2 + (cy_biggest_big[0] - cy_biggest_d_last5[j]) ** 2) for j in range(len(cy_biggest_d_last5))]
                            ind_largest = len(cx_bigest_d) -5 + np.argmin(dists_d)
                        else:
                            cx_bigest_d_last5 = cx_bigest_d[-len(cx_bigest_d):]
                            cy_biggest_d_last5 = cy_biggest_d[-len(cx_bigest_d):]
                            dists_d = [math.sqrt((cx_bigest_big[0]-cx_bigest_d_last5[j])**2 + (cy_biggest_big[0]-cy_biggest_d_last5[j])**2) for j in range(len(cy_biggest_d_last5))]
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
                        dists = [math.sqrt((p[0] - cx_bigest_d[j]) ** 2 + (p[1] - cy_biggest_d[j]) ** 2) for j in range(len(cx_bigest_d))]
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
                contours_only_text_parent = []
                
        else:
            contours_only_text, hir_on_text = return_contours_of_image(text_only)
            contours_only_text_parent = return_parent_contours(contours_only_text, hir_on_text)
            
            if len(contours_only_text_parent) > 0:
                areas_cnt_text = np.array([cv2.contourArea(contours_only_text_parent[j]) for j in range(len(contours_only_text_parent))])
                areas_cnt_text = areas_cnt_text / float(text_only.shape[0] * text_only.shape[1])

                contours_biggest = contours_only_text_parent[np.argmax(areas_cnt_text)]
                contours_only_text_parent = [contours_only_text_parent[jz] for jz in range(len(contours_only_text_parent)) if areas_cnt_text[jz] > min_con_area]
                areas_cnt_text_parent = [areas_cnt_text[jz] for jz in range(len(areas_cnt_text)) if areas_cnt_text[jz] > min_con_area]

                index_con_parents = np.argsort(areas_cnt_text_parent)
                contours_only_text_parent = list(np.array(contours_only_text_parent)[index_con_parents])
                areas_cnt_text_parent = list(np.array(areas_cnt_text_parent)[index_con_parents])

                cx_bigest_big, cy_biggest_big, _, _, _, _, _ = find_new_features_of_contours([contours_biggest])
                cx_bigest, cy_biggest, _, _, _, _, _ = find_new_features_of_contours(contours_only_text_parent)
                self.logger.debug('areas_cnt_text_parent %s', areas_cnt_text_parent)
                # self.logger.debug('areas_cnt_text_parent_d %s', areas_cnt_text_parent_d)
                # self.logger.debug('len(contours_only_text_parent) %s', len(contours_only_text_parent_d))
            else:
                pass
        txt_con_org = get_textregion_contours_in_org_image(contours_only_text_parent, self.image, slope_first)
        boxes_text, _ = get_text_region_boxes_by_given_contours(contours_only_text_parent)
        boxes_marginals, _ = get_text_region_boxes_by_given_contours(polygons_of_marginals)
        
        if not self.curved_line:
            slopes, all_found_texline_polygons, boxes_text, txt_con_org, contours_only_text_parent, all_box_coord, index_by_text_par_con = self.get_slopes_and_deskew_new(txt_con_org, contours_only_text_parent, textline_mask_tot_ea, image_page_rotated, boxes_text, slope_deskew)
            slopes_marginals, all_found_texline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, all_box_coord_marginals, _ = self.get_slopes_and_deskew_new(polygons_of_marginals, polygons_of_marginals, textline_mask_tot_ea, image_page_rotated, boxes_marginals, slope_deskew)
        else:
            
            scale_param = 1
            all_found_texline_polygons, boxes_text, txt_con_org, contours_only_text_parent, all_box_coord, index_by_text_par_con, slopes = self.get_slopes_and_deskew_new_curved(txt_con_org, contours_only_text_parent, cv2.erode(textline_mask_tot_ea, kernel=KERNEL, iterations=1), image_page_rotated, boxes_text, text_only, num_col_classifier, scale_param, slope_deskew)
            all_found_texline_polygons = small_textlines_to_parent_adherence2(all_found_texline_polygons, textline_mask_tot_ea, num_col_classifier)
            all_found_texline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, all_box_coord_marginals, _, slopes_marginals = self.get_slopes_and_deskew_new_curved(polygons_of_marginals, polygons_of_marginals, cv2.erode(textline_mask_tot_ea, kernel=KERNEL, iterations=1), image_page_rotated, boxes_marginals, text_only, num_col_classifier, scale_param, slope_deskew)
            all_found_texline_polygons_marginals = small_textlines_to_parent_adherence2(all_found_texline_polygons_marginals, textline_mask_tot_ea, num_col_classifier)
        K.clear_session()
        if self.full_layout:
            if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                contours_only_text_parent_d_ordered = list(np.array(contours_only_text_parent_d_ordered)[index_by_text_par_con])
                text_regions_p, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_texline_polygons, all_found_texline_polygons_h, slopes, _, contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered = check_any_text_region_in_model_one_is_main_or_header(text_regions_p, regions_fully, contours_only_text_parent, all_box_coord, all_found_texline_polygons, slopes, contours_only_text_parent_d_ordered)
            else:
                contours_only_text_parent_d_ordered = None
                text_regions_p, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_texline_polygons, all_found_texline_polygons_h, slopes, _, contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered = check_any_text_region_in_model_one_is_main_or_header(text_regions_p, regions_fully, contours_only_text_parent, all_box_coord, all_found_texline_polygons, slopes, contours_only_text_parent_d_ordered)

            if self.plotter:
                self.plotter.save_plot_of_layout(text_regions_p, image_page)
                self.plotter.save_plot_of_layout_all(text_regions_p, image_page)

            K.clear_session()

            polygons_of_tabels = []
            pixel_img = 4
            polygons_of_drop_capitals = return_contours_of_interested_region_by_min_size(text_regions_p, pixel_img)
            all_found_texline_polygons = adhere_drop_capital_region_into_corresponding_textline(text_regions_p, polygons_of_drop_capitals, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_texline_polygons, all_found_texline_polygons_h, kernel=KERNEL, curved_line=self.curved_line)

            # print(len(contours_only_text_parent_h),len(contours_only_text_parent_h_d_ordered),'contours_only_text_parent_h')
            pixel_lines = 6

            if not self.headers_off:
                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                    num_col, _, matrix_of_lines_ch, splitter_y_new, _ = find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines, contours_only_text_parent_h)
                else:
                    _, _, matrix_of_lines_ch_d, splitter_y_new_d, _ = find_number_of_columns_in_document(np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines, contours_only_text_parent_h_d_ordered)
            elif self.headers_off:
                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                    num_col, _, matrix_of_lines_ch, splitter_y_new, _ = find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines)
                else:
                    _, _, matrix_of_lines_ch_d, splitter_y_new_d, _ = find_number_of_columns_in_document(np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines)

            # print(peaks_neg_fin,peaks_neg_fin_d,'num_col2')
            # print(splitter_y_new,splitter_y_new_d,'num_col_classifier')
            # print(matrix_of_lines_ch.shape,matrix_of_lines_ch_d.shape,'matrix_of_lines_ch')

            if num_col_classifier >= 3:
                if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                    regions_without_separators = regions_without_separators.astype(np.uint8)
                    regions_without_separators = cv2.erode(regions_without_separators[:, :], KERNEL, iterations=6)
                    random_pixels_for_image = np.random.randn(regions_without_separators.shape[0], regions_without_separators.shape[1])
                    random_pixels_for_image[random_pixels_for_image < -0.5] = 0
                    random_pixels_for_image[random_pixels_for_image != 0] = 1
                    regions_without_separators[(random_pixels_for_image[:, :] == 1) & (text_regions_p[:, :] == 5)] = 1
                else:
                    regions_without_separators_d = regions_without_separators_d.astype(np.uint8)
                    regions_without_separators_d = cv2.erode(regions_without_separators_d[:, :], KERNEL, iterations=6)
                    random_pixels_for_image = np.random.randn(regions_without_separators_d.shape[0], regions_without_separators_d.shape[1])
                    random_pixels_for_image[random_pixels_for_image < -0.5] = 0
                    random_pixels_for_image[random_pixels_for_image != 0] = 1
                    regions_without_separators_d[(random_pixels_for_image[:, :] == 1) & (text_regions_p_1_n[:, :] == 5)] = 1

            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                boxes = return_boxes_of_images_by_order_of_reading_new(splitter_y_new, regions_without_separators, matrix_of_lines_ch, num_col_classifier, erosion_hurts)
            else:
                boxes_d = return_boxes_of_images_by_order_of_reading_new(splitter_y_new_d, regions_without_separators_d, matrix_of_lines_ch_d, num_col_classifier, erosion_hurts)

        if self.plotter:
            self.plotter.write_images_into_directory(polygons_of_images, image_page)

        if self.full_layout:
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot)
            else:
                order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered, boxes_d, textline_mask_tot_d)

            pcgts = self.writer.build_pagexml_full_layout(contours_only_text_parent, contours_only_text_parent_h, page_coord, order_text_new, id_of_texts_tot, all_found_texline_polygons, all_found_texline_polygons_h, all_box_coord, all_box_coord_h, polygons_of_images, polygons_of_tabels, polygons_of_drop_capitals, polygons_of_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals, cont_page, polygons_lines_xml)
            self.logger.info("Job done in %ss", str(time.time() - t0))
            return pcgts
        else:
            contours_only_text_parent_h = None
            if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot)
            else:
                contours_only_text_parent_d_ordered = list(np.array(contours_only_text_parent_d_ordered)[index_by_text_par_con])
                order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent_d_ordered, contours_only_text_parent_h, boxes_d, textline_mask_tot_d)
            pcgts = self.writer.build_pagexml_no_full_layout(txt_con_org, page_coord, order_text_new, id_of_texts_tot, all_found_texline_polygons, all_box_coord, polygons_of_images, polygons_of_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals, cont_page, polygons_lines_xml)
            self.logger.info("Job done in %ss", str(time.time() - t0))
            return pcgts
