"""
tool to extract table form data from alto xml data
"""

import gc
import math
import os
import random
import sys
import time
import warnings
from multiprocessing import Process, Queue, cpu_count
from sys import getsizeof

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
from keras import backend as K
from keras.models import load_model
sys.stderr = stderr
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from shapely import geometry
from lxml import etree as ET
from matplotlib import pyplot, transforms
import matplotlib.patches as mpatches
import imutils

from .utils import (
    resize_image,
    filter_contours_area_of_image_tables,
    filter_contours_area_of_image_interiors,
    rotatedRectWithMaxArea,
    rotate_image,
    rotate_max_area_new,
    rotation_image_new,
    crop_image_inside_box,
    otsu_copy,
    otsu_copy_binary,
    return_bonding_box_of_contours,
    find_features_of_lines,
    isNaN,
    return_parent_contours,
    return_contours_of_interested_region,
    return_contours_of_interested_region_by_min_size,
    return_contours_of_interested_textline,
    boosting_headers_by_longshot_region_segmentation,
    return_contours_of_image,
    get_textregion_contours_in_org_image,
    seperate_lines_vertical_cont,
    seperate_lines,
    seperate_lines_new_inside_teils2,
    filter_small_drop_capitals_from_no_patch_layout,
    find_num_col_deskew,
    return_hor_spliter_by_index_for_without_verticals,
    find_new_features_of_contoures,
)


SLOPE_THRESHOLD = 0.13
VERY_LARGE_NUMBER = 1000000000000000000000


class eynollah:
    def __init__(
        self,
        image_dir,
        f_name,
        dir_out,
        dir_models,
        dir_of_cropped_images=None,
        dir_of_layout=None,
        dir_of_deskewed=None,
        dir_of_all=None,
        allow_enhancement=False,
        curved_line=False,
        full_layout=False,
        allow_scaling=False,
        headers_off=False
    ):
        self.image_dir = image_dir  # XXX This does not seem to be a directory as the name suggests, but a file
        self.dir_out = dir_out
        self.f_name = f_name
        self.dir_of_cropped_images = dir_of_cropped_images
        self.allow_enhancement = allow_enhancement
        self.curved_line = curved_line
        self.full_layout = full_layout
        self.allow_scaling = allow_scaling
        self.dir_of_layout = dir_of_layout
        self.headers_off = headers_off
        self.dir_of_deskewed = dir_of_deskewed
        self.dir_of_all = dir_of_all
        if self.f_name is None:
            try:
                self.f_name = image_dir.split("/")[len(image_dir.split("/")) - 1]
                self.f_name = self.f_name.split(".")[0]
            except:
                self.f_name = self.f_name.split(".")[0]
        self.dir_models = dir_models
        self.kernel = np.ones((5, 5), np.uint8)

        self.model_dir_of_enhancemnet = dir_models + "/model_enhancement.h5"
        self.model_dir_of_col_classifier = dir_models + "/model_scale_classifier.h5"
        self.model_region_dir_p = dir_models + "/model_main_covid19_lr5-5_scale_1_1_great.h5"  # dir_models +'/model_main_covid_19_many_scalin_down_lr5-5_the_best.h5'#'/model_main_covid19_lr5-5_scale_1_1_great.h5'#'/model_main_scale_1_1und_1_2_corona_great.h5'
        # self.model_region_dir_p_ens = dir_models +'/model_ensemble_s.h5'#'/model_main_covid19_lr5-5_scale_1_1_great.h5'#'/model_main_scale_1_1und_1_2_corona_great.h5'
        self.model_region_dir_p2 = dir_models + "/model_main_home_corona3_rot.h5"

        self.model_region_dir_fully_np = dir_models + "/model_no_patches_class0_30eopch.h5"
        self.model_region_dir_fully = dir_models + "/model_3up_new_good_no_augmentation.h5"  # "model_3col_p_soft_10_less_aug_binarization_only.h5"

        self.model_page_dir = dir_models + "/model_page_mixed_best.h5"
        self.model_region_dir_p_ens = dir_models + "/model_ensemble_s.h5"  # dir_models +'/model_main_covid_19_many_scalin_down_lr5-5_the_best.h5' #dir_models +'/model_ensemble_s.h5'
        ###self.model_region_dir_p = dir_models +'/model_layout_newspapers.h5'#'/model_ensemble_s.h5'#'/model_layout_newspapers.h5'#'/model_ensemble_s.h5'#'/model_main_home_5_soft_new.h5'#'/model_home_soft_5_all_data.h5' #'/model_main_office_long_soft.h5'#'/model_20_cat_main.h5'
        self.model_textline_dir = dir_models + "/model_textline_newspapers.h5"  #'/model_hor_ver_home_trextline_very_good.h5'# '/model_hor_ver_1_great.h5'#'/model_curved_office_works_great.h5'

    def predict_enhancement(self, img):
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

            if nxf > int(nxf):
                nxf = int(nxf) + 1
            else:
                nxf = int(nxf)

            if nyf > int(nyf):
                nyf = int(nyf) + 1
            else:
                nyf = int(nyf)

            for i in range(nxf):
                for j in range(nyf):

                    if i == 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model
                    elif i > 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model

                    if j == 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model
                    elif j > 0:
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

    def check_dpi(self):
        dpi = os.popen('identify -format "%x " ' + self.image_dir).read()
        return int(float(dpi))

    def resize_image_with_column_classifier(self, is_image_enhanced):
        dpi = self.check_dpi()
        img = cv2.imread(self.image_dir)
        img = img.astype(np.uint8)

        _, page_coord = self.early_page_for_num_of_column_classification()
        model_num_classifier, session_col_classifier = self.start_new_session_and_model(self.model_dir_of_col_classifier)

        img_1ch = cv2.imread(self.image_dir, 0)

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

        print(num_col, label_p_pred, "num_col_classifier")

        session_col_classifier.close()
        del model_num_classifier
        del session_col_classifier

        K.clear_session()
        gc.collect()

        # sys.exit()
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

        if img_new.shape[1] > img.shape[1]:
            img_new = self.predict_enhancement(img_new)
            is_image_enhanced = True

        return img, img_new, is_image_enhanced

    def resize_and_enhance_image_with_column_classifier(self, is_image_enhanced):
        dpi = self.check_dpi()
        img = cv2.imread(self.image_dir)

        img = img.astype(np.uint8)

        _, page_coord = self.early_page_for_num_of_column_classification()
        model_num_classifier, session_col_classifier = self.start_new_session_and_model(self.model_dir_of_col_classifier)

        img_1ch = cv2.imread(self.image_dir, 0)
        img_1ch = img_1ch.astype(np.uint8)

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

        print(num_col, label_p_pred, "num_col_classifier")

        session_col_classifier.close()
        del model_num_classifier
        del session_col_classifier
        del img_in
        del img_1ch
        del page_coord

        K.clear_session()
        gc.collect()

        print(dpi)

        if dpi < 298:

            # sys.exit()
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

            # img_new=resize_image(img,img_h_new,img_w_new)
            image_res = self.predict_enhancement(img_new)
            # cv2.imwrite(os.path.join(self.dir_out, self.f_name) + ".tif",self.image)
            # self.image=self.image.astype(np.uint16)

            # self.scale_x=1
            # self.scale_y=1
            # self.height_org = self.image.shape[0]
            # self.width_org = self.image.shape[1]
            is_image_enhanced = True
        else:
            """
            if img.shape[0]<=2530 and img.shape[0]>=img.shape[1]:
                img_h_new=3000
                img_w_new=int(img.shape[1]/float(img.shape[0]) * 3000)
                img_new=resize_image(img,img_h_new,img_w_new)
                image_res=self.predict_enhancement(img_new)
                #cv2.imwrite(os.path.join(self.dir_out, self.f_name) + ".tif",self.image)
                #self.image=self.image.astype(np.uint16)
                ##self.scale_x=1
                ##self.scale_y=1
                ##self.height_org = self.image.shape[0]
                ##self.width_org = self.image.shape[1]
                is_image_enhanced=True
            else:
                is_image_enhanced=False
                image_res=np.copy(img)

            """
            is_image_enhanced = False
            num_column_is_classified = True
            image_res = np.copy(img)

        return is_image_enhanced, img, image_res, num_col, num_column_is_classified

    def resize_and_enhance_image(self, is_image_enhanced):
        dpi = self.check_dpi()
        img = cv2.imread(self.image_dir)
        img = img.astype(np.uint8)
        # sys.exit()

        print(dpi)

        if dpi < 298:
            if img.shape[0] < 1000:
                img_h_new = int(img.shape[0] * 3)
                img_w_new = int(img.shape[1] * 3)
                if img_h_new < 2800:
                    img_h_new = 3000
                    img_w_new = int(img.shape[1] / float(img.shape[0]) * 3000)
            elif img.shape[0] >= 1000 and img.shape[0] < 2000:
                img_h_new = int(img.shape[0] * 2)
                img_w_new = int(img.shape[1] * 2)
                if img_h_new < 2800:
                    img_h_new = 3000
                    img_w_new = int(img.shape[1] / float(img.shape[0]) * 3000)
            else:
                img_h_new = int(img.shape[0] * 1.5)
                img_w_new = int(img.shape[1] * 1.5)
            img_new = resize_image(img, img_h_new, img_w_new)
            image_res = self.predict_enhancement(img_new)
            # cv2.imwrite(os.path.join(self.dir_out, self.f_name) + ".tif",self.image)
            # self.image=self.image.astype(np.uint16)

            # self.scale_x=1
            # self.scale_y=1
            # self.height_org = self.image.shape[0]
            # self.width_org = self.image.shape[1]
            is_image_enhanced = True
        else:
            is_image_enhanced = False
            image_res = np.copy(img)

        return is_image_enhanced, img, image_res

    def resize_and_enhance_image_new(self, is_image_enhanced):
        # self.check_dpi()
        img = cv2.imread(self.image_dir)
        img = img.astype(np.uint8)
        # sys.exit()

        image_res = np.copy(img)

        return is_image_enhanced, img, image_res

    def get_image_and_scales(self, img_org, img_res, scale):
        self.image = np.copy(img_res)
        self.image_org = np.copy(img_org)
        self.height_org = self.image.shape[0]
        self.width_org = self.image.shape[1]

        self.img_hight_int = int(self.image.shape[0] * scale)
        self.img_width_int = int(self.image.shape[1] * scale)
        self.scale_y = self.img_hight_int / float(self.image.shape[0])
        self.scale_x = self.img_width_int / float(self.image.shape[1])

        self.image = resize_image(self.image, self.img_hight_int, self.img_width_int)
        del img_res
        del img_org

    def get_image_and_scales_after_enhancing(self, img_org, img_res):

        # self.image = cv2.imread(self.image_dir)

        self.image = np.copy(img_res)
        self.image = self.image.astype(np.uint8)
        self.image_org = np.copy(img_org)
        self.height_org = self.image_org.shape[0]
        self.width_org = self.image_org.shape[1]

        self.scale_y = img_res.shape[0] / float(self.image_org.shape[0])
        self.scale_x = img_res.shape[1] / float(self.image_org.shape[1])

        del img_org
        del img_res

    def get_image_and_scales_deskewd(self, img_deskewd):

        self.image = img_deskewd
        self.image_org = np.copy(self.image)
        self.height_org = self.image.shape[0]
        self.width_org = self.image.shape[1]

        self.img_hight_int = int(self.image.shape[0] * 1)
        self.img_width_int = int(self.image.shape[1] * 1)
        self.scale_y = self.img_hight_int / float(self.image.shape[0])
        self.scale_x = self.img_width_int / float(self.image.shape[1])

        self.image = resize_image(self.image, self.img_hight_int, self.img_width_int)

    def start_new_session_and_model(self, model_dir):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        session = tf.InteractiveSession()
        model = load_model(model_dir, compile=False)

        return model, session

    def find_images_contours_and_replace_table_and_graphic_pixels_by_image(self, region_pre_p):

        # pixels of images are identified by 5
        cnts_images = (region_pre_p[:, :, 0] == 5) * 1
        cnts_images = cnts_images.astype(np.uint8)
        cnts_images = np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
        imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_imgs = return_parent_contours(contours_imgs, hiearchy)
        # print(len(contours_imgs),'contours_imgs')
        contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=1, min_area=0.0003)

        # print(len(contours_imgs),'contours_imgs')

        boxes_imgs = return_bonding_box_of_contours(contours_imgs)

        for i in range(len(boxes_imgs)):
            x1 = int(boxes_imgs[i][0])
            x2 = int(boxes_imgs[i][0] + boxes_imgs[i][2])
            y1 = int(boxes_imgs[i][1])
            y2 = int(boxes_imgs[i][1] + boxes_imgs[i][3])
            region_pre_p[y1:y2, x1:x2, 0][region_pre_p[y1:y2, x1:x2, 0] == 8] = 5
            region_pre_p[y1:y2, x1:x2, 0][region_pre_p[y1:y2, x1:x2, 0] == 7] = 5
        return region_pre_p

    def do_prediction(self, patches, img, model, marginal_of_patch_percent=0.1):

        img_height_model = model.layers[len(model.layers) - 1].output_shape[1]
        img_width_model = model.layers[len(model.layers) - 1].output_shape[2]
        n_classes = model.layers[len(model.layers) - 1].output_shape[3]

        if patches:
            if img.shape[0] < img_height_model:
                img = resize_image(img, img_height_model, img.shape[1])

            if img.shape[1] < img_width_model:
                img = resize_image(img, img.shape[0], img_width_model)

            # print(img_height_model,img_width_model)
            # margin = int(0.2 * img_width_model)
            margin = int(marginal_of_patch_percent * img_height_model)

            width_mid = img_width_model - 2 * margin
            height_mid = img_height_model - 2 * margin

            img = img / float(255.0)
            # print(sys.getsizeof(img))
            # print(np.max(img))

            img = img.astype(np.float16)

            # print(sys.getsizeof(img))

            img_h = img.shape[0]
            img_w = img.shape[1]

            prediction_true = np.zeros((img_h, img_w, 3))
            mask_true = np.zeros((img_h, img_w))
            nxf = img_w / float(width_mid)
            nyf = img_h / float(height_mid)

            if nxf > int(nxf):
                nxf = int(nxf) + 1
            else:
                nxf = int(nxf)

            if nyf > int(nyf):
                nyf = int(nyf) + 1
            else:
                nyf = int(nyf)

            for i in range(nxf):
                for j in range(nyf):

                    if i == 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model
                    elif i > 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + img_width_model

                    if j == 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + img_height_model
                    elif j > 0:
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
        del model
        gc.collect()

        return prediction_true

    def early_page_for_num_of_column_classification(self):
        img = cv2.imread(self.image_dir)
        img = img.astype(np.uint8)
        patches = False
        model_page, session_page = self.start_new_session_and_model(self.model_page_dir)
        ###img = otsu_copy(self.image)
        for ii in range(1):
            img = cv2.GaussianBlur(img, (5, 5), 0)

        img_page_prediction = self.do_prediction(patches, img, model_page)

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
        return croped_page, page_coord

    def extract_page(self):
        patches = False
        model_page, session_page = self.start_new_session_and_model(self.model_page_dir)
        ###img = otsu_copy(self.image)
        for ii in range(1):
            img = cv2.GaussianBlur(self.image, (5, 5), 0)

        img_page_prediction = self.do_prediction(patches, img, model_page)

        imgray = cv2.cvtColor(img_page_prediction, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        thresh = cv2.dilate(thresh, self.kernel, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])

        cnt = contours[np.argmax(cnt_size)]

        x, y, w, h = cv2.boundingRect(cnt)

        if x <= 30:
            w = w + x
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

        self.cont_page = []
        self.cont_page.append(np.array([[page_coord[2], page_coord[0]], [page_coord[3], page_coord[0]], [page_coord[3], page_coord[1]], [page_coord[2], page_coord[1]]]))

        session_page.close()
        del model_page
        del session_page
        del contours
        del thresh
        del img
        del imgray

        gc.collect()
        return croped_page, page_coord

    def extract_drop_capital_13(self, img, patches, cols):

        img_height_h = img.shape[0]
        img_width_h = img.shape[1]
        patches = False

        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint16)

        model_region, session_region = self.start_new_session_and_model(self.model_region_dir_fully_np)

        img_1 = img[: int(img.shape[0] / 3.0), :, :]
        img_2 = img[int(img.shape[0] / 3.0) : int(2 * img.shape[0] / 3.0), :, :]
        img_3 = img[int(2 * img.shape[0] / 3.0) :, :, :]

        # img_1 = otsu_copy_binary(img_1)#otsu_copy(img)
        # img_1 = img_1.astype(np.uint16)

        plt.imshow(img_1)
        plt.show()
        # img_2 = otsu_copy_binary(img_2)#otsu_copy(img)
        # img_2 = img_2.astype(np.uint16)

        plt.imshow(img_2)
        plt.show()
        # img_3 = otsu_copy_binary(img_3)#otsu_copy(img)
        # img_3 = img_3.astype(np.uint16)

        plt.imshow(img_3)
        plt.show()

        prediction_regions_1 = self.do_prediction(patches, img_1, model_region)

        plt.imshow(prediction_regions_1)
        plt.show()

        prediction_regions_2 = self.do_prediction(patches, img_2, model_region)

        plt.imshow(prediction_regions_2)
        plt.show()
        prediction_regions_3 = self.do_prediction(patches, img_3, model_region)

        plt.imshow(prediction_regions_3)
        plt.show()
        prediction_regions = np.zeros((img_height_h, img_width_h))

        prediction_regions[: int(img.shape[0] / 3.0), :] = prediction_regions_1[:, :, 0]
        prediction_regions[int(img.shape[0] / 3.0) : int(2 * img.shape[0] / 3.0), :] = prediction_regions_2[:, :, 0]
        prediction_regions[int(2 * img.shape[0] / 3.0) :, :] = prediction_regions_3[:, :, 0]

        session_region.close()
        del img_1
        del img_2
        del img_3
        del prediction_regions_1
        del prediction_regions_2
        del prediction_regions_3
        del model_region
        del session_region
        del img
        gc.collect()
        return prediction_regions

    def extract_text_regions(self, img, patches, cols):
        img_height_h = img.shape[0]
        img_width_h = img.shape[1]

        ###if patches and cols>=3 :
        ###model_region, session_region = self.start_new_session_and_model(self.model_region_dir_fully)
        ###if not patches:
        ###model_region, session_region = self.start_new_session_and_model(self.model_region_dir_fully_np)

        ###if patches and cols==2 :
        ###model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_2col)

        ###if patches and cols==1 :
        ###model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_2col)

        ###if patches and cols>=2:

        ###img = otsu_copy_binary(img)#otsu_copy(img)
        ###img = img.astype(np.uint8)

        ###if patches and cols==1:

        ###img = otsu_copy_binary(img)#otsu_copy(img)
        ###img = img.astype(np.uint8)
        ###img= resize_image(img, int(img_height_h*1), int(img_width_h*1) )

        if patches:
            model_region, session_region = self.start_new_session_and_model(self.model_region_dir_fully)
        if not patches:
            model_region, session_region = self.start_new_session_and_model(self.model_region_dir_fully_np)

        if patches and cols == 1:
            img2 = otsu_copy_binary(img)  # otsu_copy(img)
            img2 = img2.astype(np.uint8)
            img2 = resize_image(img2, int(img_height_h * 0.7), int(img_width_h * 0.7))

            marginal_of_patch_percent = 0.1
            prediction_regions2 = self.do_prediction(patches, img2, model_region, marginal_of_patch_percent)
            prediction_regions2 = resize_image(prediction_regions2, img_height_h, img_width_h)

        if patches and cols == 2:
            img2 = otsu_copy_binary(img)  # otsu_copy(img)
            img2 = img2.astype(np.uint8)
            img2 = resize_image(img2, int(img_height_h * 0.4), int(img_width_h * 0.4))

            marginal_of_patch_percent = 0.1
            prediction_regions2 = self.do_prediction(patches, img2, model_region, marginal_of_patch_percent)
            prediction_regions2 = resize_image(prediction_regions2, img_height_h, img_width_h)
        elif patches and cols > 2:
            img2 = otsu_copy_binary(img)  # otsu_copy(img)
            img2 = img2.astype(np.uint8)
            img2 = resize_image(img2, int(img_height_h * 0.3), int(img_width_h * 0.3))

            marginal_of_patch_percent = 0.1
            prediction_regions2 = self.do_prediction(patches, img2, model_region, marginal_of_patch_percent)
            prediction_regions2 = resize_image(prediction_regions2, img_height_h, img_width_h)

        if patches and cols == 2:
            img = otsu_copy_binary(img)  # otsu_copy(img)

            img = img.astype(np.uint8)

            if img_width_h >= 2000:
                img = resize_image(img, int(img_height_h * 0.9), int(img_width_h * 0.9))
            else:
                pass  # img= resize_image(img, int(img_height_h*1), int(img_width_h*1) )
            img = img.astype(np.uint8)

        if patches and cols == 1:
            img = otsu_copy_binary(img)  # otsu_copy(img)

            img = img.astype(np.uint8)
            img = resize_image(img, int(img_height_h * 0.5), int(img_width_h * 0.5))
            img = img.astype(np.uint8)

        if patches and cols == 3:

            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint8)
            # img= resize_image(img, int(img_height_h*0.9), int(img_width_h*0.9) )

        if patches and cols == 4:

            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint8)
            # img= resize_image(img, int(img_height_h*0.9), int(img_width_h*0.9) )

        if patches and cols >= 5:

            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint8)
            # img= resize_image(img, int(img_height_h*0.9), int(img_width_h*0.9) )

        if not patches:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint8)
            prediction_regions2 = None

        marginal_of_patch_percent = 0.1
        prediction_regions = self.do_prediction(patches, img, model_region, marginal_of_patch_percent)
        prediction_regions = resize_image(prediction_regions, img_height_h, img_width_h)

        session_region.close()
        del model_region
        del session_region
        del img
        gc.collect()
        return prediction_regions, prediction_regions2

    def extract_only_text_regions(self, img, patches):

        model_region, session_region = self.start_new_session_and_model(self.model_only_text)
        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint8)
        img_org = np.copy(img)

        img_h = img_org.shape[0]
        img_w = img_org.shape[1]

        img = resize_image(img_org, int(img_org.shape[0] * 1), int(img_org.shape[1] * 1))

        prediction_regions1 = self.do_prediction(patches, img, model_region)

        prediction_regions1 = resize_image(prediction_regions1, img_h, img_w)

        # prediction_regions1 = cv2.dilate(prediction_regions1, self.kernel, iterations=4)
        # prediction_regions1 = cv2.erode(prediction_regions1, self.kernel, iterations=7)
        # prediction_regions1 = cv2.dilate(prediction_regions1, self.kernel, iterations=2)

        img = resize_image(img_org, int(img_org.shape[0] * 1), int(img_org.shape[1] * 1))

        prediction_regions2 = self.do_prediction(patches, img, model_region)

        prediction_regions2 = resize_image(prediction_regions2, img_h, img_w)

        # prediction_regions2 = cv2.dilate(prediction_regions2, self.kernel, iterations=2)
        prediction_regions2 = cv2.erode(prediction_regions2, self.kernel, iterations=2)
        prediction_regions2 = cv2.dilate(prediction_regions2, self.kernel, iterations=2)

        # prediction_regions=(  (prediction_regions2[:,:,0]==1) & (prediction_regions1[:,:,0]==1) )
        # prediction_regions=(prediction_regions1[:,:,0]==1)

        session_region.close()
        del model_region
        del session_region
        gc.collect()
        return prediction_regions1[:, :, 0]

    def extract_binarization(self, img, patches):

        model_bin, session_bin = self.start_new_session_and_model(self.model_binafrization)

        img_h = img.shape[0]
        img_w = img.shape[1]

        img = resize_image(img, int(img.shape[0] * 1), int(img.shape[1] * 1))

        prediction_regions = self.do_prediction(patches, img, model_bin)

        res = (prediction_regions[:, :, 0] != 0) * 1

        img_fin = np.zeros((res.shape[0], res.shape[1], 3))
        res[:, :][res[:, :] == 0] = 2
        res = res - 1
        res = res * 255
        img_fin[:, :, 0] = res
        img_fin[:, :, 1] = res
        img_fin[:, :, 2] = res

        session_bin.close()
        del model_bin
        del session_bin
        gc.collect()
        # plt.imshow(img_fin[:,:,0])
        # plt.show()
        return img_fin

    def get_slopes_and_deskew_new(self, contours, contours_par, textline_mask_tot, image_page_rotated, boxes, slope_deskew):
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
        # print(slopes,'slopes')
        return slopes, all_found_texline_polygons, boxes, all_found_text_regions, all_found_text_regions_par, all_box_coord, all_index_text_con

    def get_slopes_and_deskew_new_curved(self, contours, contours_par, textline_mask_tot, image_page_rotated, boxes, mask_texts_only, num_col, scale_par, slope_deskew):
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
        slopes_per_each_subprocess = []
        bounding_box_of_textregion_per_each_subprocess = []
        textlines_rectangles_per_each_subprocess = []
        contours_textregion_per_each_subprocess = []
        contours_textregion_par_per_each_subprocess = []
        all_box_coord_per_process = []
        index_by_text_region_contours = []
        slope_biggest = 0

        textline_cnt_seperated = np.zeros(textline_mask_tot_ea.shape)

        for mv in range(len(boxes_text)):

            all_text_region_raw = textline_mask_tot_ea[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]]
            all_text_region_raw = all_text_region_raw.astype(np.uint8)

            img_int_p = all_text_region_raw[:, :]  # self.all_text_region_raw[mv]

            ##img_int_p=cv2.erode(img_int_p,self.kernel,iterations = 2)

            # plt.imshow(img_int_p)
            # plt.show()

            if img_int_p.shape[0] / img_int_p.shape[1] < 0.1:

                slopes_per_each_subprocess.append(0)

                slope_first = 0
                slope_for_all = [slope_deskew][0]

            else:

                try:
                    textline_con, hierachy = return_contours_of_image(img_int_p)
                    textline_con_fil = filter_contours_area_of_image(img_int_p, textline_con, hierachy, max_area=1, min_area=0.0008)
                    y_diff_mean = self.find_contours_mean_y_diff(textline_con_fil)

                    sigma_des = int(y_diff_mean * (4.0 / 40.0))

                    if sigma_des < 1:
                        sigma_des = 1

                    img_int_p[img_int_p > 0] = 1
                    # slope_for_all=self.return_deskew_slope_new(img_int_p,sigma_des)
                    slope_for_all = self.return_deskew_slop(img_int_p, sigma_des)

                    if abs(slope_for_all) < 0.5:
                        slope_for_all = [slope_deskew][0]
                    # old method
                    # slope_for_all=self.textline_contours_to_get_slope_correctly(self.all_text_region_raw[mv],denoised,contours[mv])
                    # text_patch_processed=textline_contours_postprocessing(gada)

                except:
                    slope_for_all = 999

                ##slope_for_all=self.return_deskew_slop(img_int_p,sigma_des)

                if slope_for_all == 999:
                    slope_for_all = [slope_deskew][0]
                ##if np.abs(slope_for_all)>32.5 and slope_for_all!=999:
                ##slope_for_all=slope_biggest
                ##elif slope_for_all==999:
                ##slope_for_all=slope_biggest
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
                textline_rotated_seperated = self.seperate_lines_new2(textline_biggest_region[y : y + h, x : x + w], 0, num_col, slope_for_all)

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
                slope_first = 0
                add_boxes_coor_into_textlines = True
                textlines_cnt_per_region = self.textline_contours_postprocessing(all_text_region_raw, slope_for_all, contours_par_per_process[mv], boxes_text[mv], slope_first, add_boxes_coor_into_textlines)
                add_boxes_coor_into_textlines = False
                # print(np.shape(textlines_cnt_per_region),'textlines_cnt_per_region')

            # textlines_cnt_tot_per_process.append(textlines_cnt_per_region)
            # index_polygons_per_process_per_process.append(index_polygons_per_process[iiii])

            textlines_rectangles_per_each_subprocess.append(textlines_cnt_per_region)
            # all_found_texline_polygons.append(cnt_clean_rot)
            bounding_box_of_textregion_per_each_subprocess.append(boxes_text[mv])

            contours_textregion_per_each_subprocess.append(contours_per_process[mv])
            contours_textregion_par_per_each_subprocess.append(contours_par_per_process[mv])
            all_box_coord_per_process.append(crop_coor)

        queue_of_all_params.put([textlines_rectangles_per_each_subprocess, bounding_box_of_textregion_per_each_subprocess, contours_textregion_per_each_subprocess, contours_textregion_par_per_each_subprocess, all_box_coord_per_process, index_by_text_region_contours, slopes_per_each_subprocess])

    def do_work_of_slopes_new(self, queue_of_all_params, boxes_text, textline_mask_tot_ea, contours_per_process, contours_par_per_process, indexes_r_con_per_pro, image_page_rotated, slope_deskew):

        slopes_per_each_subprocess = []
        bounding_box_of_textregion_per_each_subprocess = []
        textlines_rectangles_per_each_subprocess = []
        contours_textregion_per_each_subprocess = []
        contours_textregion_par_per_each_subprocess = []
        all_box_coord_per_process = []
        index_by_text_region_contours = []
        slope_biggest = 0

        for mv in range(len(boxes_text)):

            crop_img, crop_coor = crop_image_inside_box(boxes_text[mv], image_page_rotated)

            # all_box_coord.append(crop_coor)

            denoised = None
            all_text_region_raw = textline_mask_tot_ea[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]]
            all_text_region_raw = all_text_region_raw.astype(np.uint8)

            img_int_p = all_text_region_raw[:, :]  # self.all_text_region_raw[mv]

            img_int_p = cv2.erode(img_int_p, self.kernel, iterations=2)

            if img_int_p.shape[0] / img_int_p.shape[1] < 0.1:

                slopes_per_each_subprocess.append(0)

                slope_for_all = [slope_deskew][0]

                all_text_region_raw = textline_mask_tot_ea[boxes_text[mv][1] : boxes_text[mv][1] + boxes_text[mv][3], boxes_text[mv][0] : boxes_text[mv][0] + boxes_text[mv][2]]
                ###cnt_clean_rot=self.textline_contours_postprocessing(all_text_region_raw,slopes[jj],contours_only_text_parent[jj],boxes_text[jj],slope_first)
                cnt_clean_rot = self.textline_contours_postprocessing(all_text_region_raw, slope_for_all, contours_par_per_process[mv], boxes_text[mv], 0)

                textlines_rectangles_per_each_subprocess.append(cnt_clean_rot)

                index_by_text_region_contours.append(indexes_r_con_per_pro[mv])
                # all_found_texline_polygons.append(cnt_clean_rot)
                bounding_box_of_textregion_per_each_subprocess.append(boxes_text[mv])
            else:

                try:
                    textline_con, hierachy = return_contours_of_image(img_int_p)
                    textline_con_fil = filter_contours_area_of_image(img_int_p, textline_con, hierachy, max_area=1, min_area=0.00008)

                    y_diff_mean = self.find_contours_mean_y_diff(textline_con_fil)

                    sigma_des = int(y_diff_mean * (4.0 / 40.0))

                    if sigma_des < 1:
                        sigma_des = 1

                    img_int_p[img_int_p > 0] = 1
                    # slope_for_all=self.return_deskew_slope_new(img_int_p,sigma_des)
                    slope_for_all = self.return_deskew_slop(img_int_p, sigma_des)

                    if abs(slope_for_all) <= 0.5:
                        slope_for_all = [slope_deskew][0]

                except:
                    slope_for_all = 999

                ##slope_for_all=self.return_deskew_slop(img_int_p,sigma_des)

                if slope_for_all == 999:
                    slope_for_all = [slope_deskew][0]
                ##if np.abs(slope_for_all)>32.5 and slope_for_all!=999:
                ##slope_for_all=slope_biggest
                ##elif slope_for_all==999:
                ##slope_for_all=slope_biggest
                slopes_per_each_subprocess.append(slope_for_all)

                slope_first = 0

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
                ###cnt_clean_rot=self.textline_contours_postprocessing(all_text_region_raw,slopes[jj],contours_only_text_parent[jj],boxes_text[jj],slope_first)
                cnt_clean_rot = self.textline_contours_postprocessing(all_text_region_raw, slope_for_all, contours_par_per_process[mv], boxes_text[mv], slope_first)

                textlines_rectangles_per_each_subprocess.append(cnt_clean_rot)
                index_by_text_region_contours.append(indexes_r_con_per_pro[mv])
                # all_found_texline_polygons.append(cnt_clean_rot)
                bounding_box_of_textregion_per_each_subprocess.append(boxes_text[mv])

            contours_textregion_per_each_subprocess.append(contours_per_process[mv])
            contours_textregion_par_per_each_subprocess.append(contours_par_per_process[mv])
            all_box_coord_per_process.append(crop_coor)

        queue_of_all_params.put([slopes_per_each_subprocess, textlines_rectangles_per_each_subprocess, bounding_box_of_textregion_per_each_subprocess, contours_textregion_per_each_subprocess, contours_textregion_par_per_each_subprocess, all_box_coord_per_process, index_by_text_region_contours])

    def get_text_region_contours_and_boxes(self, image):
        rgb_class_of_texts = (1, 1, 1)
        mask_texts = np.all(image == rgb_class_of_texts, axis=-1)

        image = np.repeat(mask_texts[:, :, np.newaxis], 3, axis=2) * 255
        image = image.astype(np.uint8)

        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.kernel)

        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours, hirarchy = cv2.findContours(thresh.copy(), cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        main_contours = filter_contours_area_of_image(thresh, contours, hirarchy, max_area=1, min_area=0.00001)
        self.boxes = []

        for jj in range(len(main_contours)):
            x, y, w, h = cv2.boundingRect(main_contours[jj])
            self.boxes.append([x, y, w, h])

        return main_contours

    def textline_contours(self, img, patches, scaler_h, scaler_w):

        if patches:
            model_textline, session_textline = self.start_new_session_and_model(self.model_textline_dir)
        if not patches:
            model_textline, session_textline = self.start_new_session_and_model(self.model_textline_dir_np)

        ##img = otsu_copy(img)
        img = img.astype(np.uint8)

        img_org = np.copy(img)
        img_h = img_org.shape[0]
        img_w = img_org.shape[1]

        img = resize_image(img_org, int(img_org.shape[0] * scaler_h), int(img_org.shape[1] * scaler_w))

        prediction_textline = self.do_prediction(patches, img, model_textline)

        prediction_textline = resize_image(prediction_textline, img_h, img_w)

        patches = False
        prediction_textline_longshot = self.do_prediction(patches, img, model_textline)

        prediction_textline_longshot_true_size = resize_image(prediction_textline_longshot, img_h, img_w)

        # scaler_w=1.5
        # scaler_h=1.5
        # patches=True
        # img= resize_image(img_org, int(img_org.shape[0]*scaler_h), int(img_org.shape[1]*scaler_w))

        # prediction_textline_streched=self.do_prediction(patches,img,model_textline)

        # prediction_textline_streched= resize_image(prediction_textline_streched, img_h, img_w)

        ##plt.imshow(prediction_textline_streched[:,:,0])
        ##plt.show()

        # sys.exit()
        session_textline.close()

        del model_textline
        del session_textline
        del img
        del img_org

        gc.collect()
        return prediction_textline[:, :, 0], prediction_textline_longshot_true_size[:, :, 0]

    def seperate_lines_new_inside_teils(self, img_path, thetha):
        (h, w) = img_path.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
        x_d = M[0, 2]
        y_d = M[1, 2]

        thetha = thetha / 180.0 * np.pi
        rotation_matrix = np.array([[np.cos(thetha), -np.sin(thetha)], [np.sin(thetha), np.cos(thetha)]])

        x_min_cont = 0
        x_max_cont = img_path.shape[1]
        y_min_cont = 0
        y_max_cont = img_path.shape[0]

        xv = np.linspace(x_min_cont, x_max_cont, 1000)

        mada_n = img_path.sum(axis=1)

        ##plt.plot(mada_n)
        ##plt.show()

        first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

        y = mada_n[:]  # [first_nonzero:last_nonzero]
        y_help = np.zeros(len(y) + 40)
        y_help[20 : len(y) + 20] = y
        x = np.array(range(len(y)))

        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
        if len(peaks_real) <= 2 and len(peaks_real) > 1:
            sigma_gaus = 10
        else:
            sigma_gaus = 5

        z = gaussian_filter1d(y_help, sigma_gaus)
        zneg_rev = -y_help + np.max(y_help)
        zneg = np.zeros(len(zneg_rev) + 40)
        zneg[20 : len(zneg_rev) + 20] = zneg_rev
        zneg = gaussian_filter1d(zneg, sigma_gaus)

        peaks, _ = find_peaks(z, height=0)
        peaks_neg, _ = find_peaks(zneg, height=0)

        for nn in range(len(peaks_neg)):
            if peaks_neg[nn] > len(z) - 1:
                peaks_neg[nn] = len(z) - 1
            if peaks_neg[nn] < 0:
                peaks_neg[nn] = 0

        diff_peaks = np.abs(np.diff(peaks_neg))

        cut_off = 20
        peaks_neg_true = []
        forest = []

        for i in range(len(peaks_neg)):
            if i == 0:
                forest.append(peaks_neg[i])
            if i < (len(peaks_neg) - 1):
                if diff_peaks[i] <= cut_off:
                    forest.append(peaks_neg[i + 1])
                if diff_peaks[i] > cut_off:
                    # print(forest[np.argmin(z[forest]) ] )
                    if not isNaN(forest[np.argmin(z[forest])]):
                        peaks_neg_true.append(forest[np.argmin(z[forest])])
                    forest = []
                    forest.append(peaks_neg[i + 1])
            if i == (len(peaks_neg) - 1):
                # print(print(forest[np.argmin(z[forest]) ] ))
                if not isNaN(forest[np.argmin(z[forest])]):
                    peaks_neg_true.append(forest[np.argmin(z[forest])])

        diff_peaks_pos = np.abs(np.diff(peaks))

        cut_off = 20
        peaks_pos_true = []
        forest = []

        for i in range(len(peaks)):
            if i == 0:
                forest.append(peaks[i])
            if i < (len(peaks) - 1):
                if diff_peaks_pos[i] <= cut_off:
                    forest.append(peaks[i + 1])
                if diff_peaks_pos[i] > cut_off:
                    # print(forest[np.argmin(z[forest]) ] )
                    if not isNaN(forest[np.argmax(z[forest])]):
                        peaks_pos_true.append(forest[np.argmax(z[forest])])
                    forest = []
                    forest.append(peaks[i + 1])
            if i == (len(peaks) - 1):
                # print(print(forest[np.argmin(z[forest]) ] ))
                if not isNaN(forest[np.argmax(z[forest])]):
                    peaks_pos_true.append(forest[np.argmax(z[forest])])

        # print(len(peaks_neg_true) ,len(peaks_pos_true) ,'lensss')

        if len(peaks_neg_true) > 0:
            peaks_neg_true = np.array(peaks_neg_true)
            """
            #plt.figure(figsize=(40,40))
            #plt.subplot(1,2,1)
            #plt.title('Textline segmentation von Textregion')
            #plt.imshow(img_path)
            #plt.xlabel('X')
            #plt.ylabel('Y')
            #plt.subplot(1,2,2)
            #plt.title('Dichte entlang X')
            #base = pyplot.gca().transData
            #rot = transforms.Affine2D().rotate_deg(90)
            #plt.plot(zneg,np.array(range(len(zneg))))
            #plt.plot(zneg[peaks_neg_true],peaks_neg_true,'*')
            #plt.gca().invert_yaxis()

            #plt.xlabel('Dichte')
            #plt.ylabel('Y')
            ##plt.plot([0,len(y)], [grenze,grenze])
            #plt.show()
            """
            peaks_neg_true = peaks_neg_true - 20 - 20

            # print(peaks_neg_true)
            for i in range(len(peaks_neg_true)):
                img_path[peaks_neg_true[i] - 6 : peaks_neg_true[i] + 6, :] = 0

        else:
            pass

        if len(peaks_pos_true) > 0:
            peaks_pos_true = np.array(peaks_pos_true)
            peaks_pos_true = peaks_pos_true - 20

            for i in range(len(peaks_pos_true)):
                img_path[peaks_pos_true[i] - 8 : peaks_pos_true[i] + 8, :] = 1
        else:
            pass
        kernel = np.ones((5, 5), np.uint8)

        # img_path = cv2.erode(img_path,kernel,iterations = 3)
        img_path = cv2.erode(img_path, kernel, iterations=2)
        return img_path

    def seperate_lines_new(self, img_path, thetha, num_col):

        if num_col == 1:
            num_patches = int(img_path.shape[1] / 200.0)
        else:
            num_patches = int(img_path.shape[1] / 100.0)
        # num_patches=int(img_path.shape[1]/200.)
        if num_patches == 0:
            num_patches = 1
        (h, w) = img_path.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
        x_d = M[0, 2]
        y_d = M[1, 2]

        thetha = thetha / 180.0 * np.pi
        rotation_matrix = np.array([[np.cos(thetha), -np.sin(thetha)], [np.sin(thetha), np.cos(thetha)]])

        x_min_cont = 0
        x_max_cont = img_path.shape[1]
        y_min_cont = 0
        y_max_cont = img_path.shape[0]

        xv = np.linspace(x_min_cont, x_max_cont, 1000)

        mada_n = img_path.sum(axis=1)

        ##plt.plot(mada_n)
        ##plt.show()
        first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

        y = mada_n[:]  # [first_nonzero:last_nonzero]
        y_help = np.zeros(len(y) + 40)
        y_help[20 : len(y) + 20] = y
        x = np.array(range(len(y)))

        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
        if len(peaks_real) <= 2 and len(peaks_real) > 1:
            sigma_gaus = 10
        else:
            sigma_gaus = 6

        z = gaussian_filter1d(y_help, sigma_gaus)
        zneg_rev = -y_help + np.max(y_help)
        zneg = np.zeros(len(zneg_rev) + 40)
        zneg[20 : len(zneg_rev) + 20] = zneg_rev
        zneg = gaussian_filter1d(zneg, sigma_gaus)

        peaks, _ = find_peaks(z, height=0)
        peaks_neg, _ = find_peaks(zneg, height=0)

        for nn in range(len(peaks_neg)):
            if peaks_neg[nn] > len(z) - 1:
                peaks_neg[nn] = len(z) - 1
            if peaks_neg[nn] < 0:
                peaks_neg[nn] = 0

        diff_peaks = np.abs(np.diff(peaks_neg))
        cut_off = 20
        peaks_neg_true = []
        forest = []

        for i in range(len(peaks_neg)):
            if i == 0:
                forest.append(peaks_neg[i])
            if i < (len(peaks_neg) - 1):
                if diff_peaks[i] <= cut_off:
                    forest.append(peaks_neg[i + 1])
                if diff_peaks[i] > cut_off:
                    # print(forest[np.argmin(z[forest]) ] )
                    if not isNaN(forest[np.argmin(z[forest])]):
                        # print(len(z),forest)
                        peaks_neg_true.append(forest[np.argmin(z[forest])])
                    forest = []
                    forest.append(peaks_neg[i + 1])
            if i == (len(peaks_neg) - 1):
                # print(print(forest[np.argmin(z[forest]) ] ))
                if not isNaN(forest[np.argmin(z[forest])]):

                    peaks_neg_true.append(forest[np.argmin(z[forest])])

        peaks_neg_true = np.array(peaks_neg_true)

        """
        #plt.figure(figsize=(40,40))
        #plt.subplot(1,2,1)
        #plt.title('Textline segmentation von Textregion')
        #plt.imshow(img_path)
        #plt.xlabel('X')
        #plt.ylabel('Y')
        #plt.subplot(1,2,2)
        #plt.title('Dichte entlang X')
        #base = pyplot.gca().transData
        #rot = transforms.Affine2D().rotate_deg(90)
        #plt.plot(zneg,np.array(range(len(zneg))))
        #plt.plot(zneg[peaks_neg_true],peaks_neg_true,'*')
        #plt.gca().invert_yaxis()

        #plt.xlabel('Dichte')
        #plt.ylabel('Y')
        ##plt.plot([0,len(y)], [grenze,grenze])
        #plt.show()
        """

        peaks_neg_true = peaks_neg_true - 20 - 20
        peaks = peaks - 20

        # dis_up=peaks_neg_true[14]-peaks_neg_true[0]
        # dis_down=peaks_neg_true[18]-peaks_neg_true[14]

        img_patch_ineterst = img_path[:, :]  # [peaks_neg_true[14]-dis_up:peaks_neg_true[15]+dis_down ,:]

        ##plt.imshow(img_patch_ineterst)
        ##plt.show()

        length_x = int(img_path.shape[1] / float(num_patches))
        margin = int(0.04 * length_x)

        width_mid = length_x - 2 * margin

        nxf = img_path.shape[1] / float(width_mid)

        if nxf > int(nxf):
            nxf = int(nxf) + 1
        else:
            nxf = int(nxf)

        slopes_tile_wise = []
        for i in range(nxf):
            if i == 0:
                index_x_d = i * width_mid
                index_x_u = index_x_d + length_x
            elif i > 0:
                index_x_d = i * width_mid
                index_x_u = index_x_d + length_x

            if index_x_u > img_path.shape[1]:
                index_x_u = img_path.shape[1]
                index_x_d = img_path.shape[1] - length_x

            # img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
            img_xline = img_patch_ineterst[:, index_x_d:index_x_u]

            sigma = 2
            try:
                slope_xline = self.return_deskew_slop(img_xline, sigma)
            except:
                slope_xline = 0
            slopes_tile_wise.append(slope_xline)
            # print(slope_xline,'xlineeee')
            img_line_rotated = rotate_image(img_xline, slope_xline)
            img_line_rotated[:, :][img_line_rotated[:, :] != 0] = 1

        """

        xline=np.linspace(0,img_path.shape[1],nx)
        slopes_tile_wise=[]

        for ui in range( nx-1 ):
            img_xline=img_patch_ineterst[:,int(xline[ui]):int(xline[ui+1])]


            ##plt.imshow(img_xline)
            ##plt.show()

            sigma=3
            try:
                slope_xline=self.return_deskew_slop(img_xline,sigma)
            except:
                slope_xline=0
            slopes_tile_wise.append(slope_xline)
            print(slope_xline,'xlineeee')
            img_line_rotated=rotate_image(img_xline,slope_xline)

            ##plt.imshow(img_line_rotated)
            ##plt.show()
        """

        # dis_up=peaks_neg_true[14]-peaks_neg_true[0]
        # dis_down=peaks_neg_true[18]-peaks_neg_true[14]

        img_patch_ineterst = img_path[:, :]  # [peaks_neg_true[14]-dis_up:peaks_neg_true[14]+dis_down ,:]

        img_patch_ineterst_revised = np.zeros(img_patch_ineterst.shape)

        for i in range(nxf):
            if i == 0:
                index_x_d = i * width_mid
                index_x_u = index_x_d + length_x
            elif i > 0:
                index_x_d = i * width_mid
                index_x_u = index_x_d + length_x

            if index_x_u > img_path.shape[1]:
                index_x_u = img_path.shape[1]
                index_x_d = img_path.shape[1] - length_x

            img_xline = img_patch_ineterst[:, index_x_d:index_x_u]

            img_int = np.zeros((img_xline.shape[0], img_xline.shape[1]))
            img_int[:, :] = img_xline[:, :]  # img_patch_org[:,:,0]

            img_resized = np.zeros((int(img_int.shape[0] * (1.2)), int(img_int.shape[1] * (3))))

            img_resized[int(img_int.shape[0] * (0.1)) : int(img_int.shape[0] * (0.1)) + img_int.shape[0], int(img_int.shape[1] * (1)) : int(img_int.shape[1] * (1)) + img_int.shape[1]] = img_int[:, :]
            ##plt.imshow(img_xline)
            ##plt.show()
            img_line_rotated = rotate_image(img_resized, slopes_tile_wise[i])
            img_line_rotated[:, :][img_line_rotated[:, :] != 0] = 1

            img_patch_seperated = self.seperate_lines_new_inside_teils(img_line_rotated, 0)

            ##plt.imshow(img_patch_seperated)
            ##plt.show()
            img_patch_seperated_returned = rotate_image(img_patch_seperated, -slopes_tile_wise[i])
            img_patch_seperated_returned[:, :][img_patch_seperated_returned[:, :] != 0] = 1

            img_patch_seperated_returned_true_size = img_patch_seperated_returned[int(img_int.shape[0] * (0.1)) : int(img_int.shape[0] * (0.1)) + img_int.shape[0], int(img_int.shape[1] * (1)) : int(img_int.shape[1] * (1)) + img_int.shape[1]]

            img_patch_seperated_returned_true_size = img_patch_seperated_returned_true_size[:, margin : length_x - margin]
            img_patch_ineterst_revised[:, index_x_d + margin : index_x_u - margin] = img_patch_seperated_returned_true_size

        """
        for ui in range( nx-1 ):
            img_xline=img_patch_ineterst[:,int(xline[ui]):int(xline[ui+1])]


            img_int=np.zeros((img_xline.shape[0],img_xline.shape[1]))
            img_int[:,:]=img_xline[:,:]#img_patch_org[:,:,0]

            img_resized=np.zeros((int( img_int.shape[0]*(1.2) ) , int( img_int.shape[1]*(3) ) ))

            img_resized[ int( img_int.shape[0]*(.1)):int( img_int.shape[0]*(.1))+img_int.shape[0] , int( img_int.shape[1]*(1)):int( img_int.shape[1]*(1))+img_int.shape[1] ]=img_int[:,:]
            ##plt.imshow(img_xline)
            ##plt.show()
            img_line_rotated=rotate_image(img_resized,slopes_tile_wise[ui])


            #img_patch_seperated=self.seperate_lines_new_inside_teils(img_line_rotated,0)

            img_patch_seperated=self.seperate_lines_new_inside_teils(img_line_rotated,0)

            img_patch_seperated_returned=rotate_image(img_patch_seperated,-slopes_tile_wise[ui])
            ##plt.imshow(img_patch_seperated)
            ##plt.show()
            print(img_patch_seperated_returned.shape)
            #plt.imshow(img_patch_seperated_returned[ int( img_int.shape[0]*(.1)):int( img_int.shape[0]*(.1))+img_int.shape[0] , int( img_int.shape[1]*(1)):int( img_int.shape[1]*(1))+img_int.shape[1] ])
            #plt.show()

            img_patch_ineterst_revised[:,int(xline[ui]):int(xline[ui+1])]=img_patch_seperated_returned[ int( img_int.shape[0]*(.1)):int( img_int.shape[0]*(.1))+img_int.shape[0] , int( img_int.shape[1]*(1)):int( img_int.shape[1]*(1))+img_int.shape[1] ]


        """

        # print(img_patch_ineterst_revised.shape,np.unique(img_patch_ineterst_revised))
        ##plt.imshow(img_patch_ineterst_revised)
        ##plt.show()
        return img_patch_ineterst_revised

    def seperate_lines_new2(self, img_path, thetha, num_col, slope_region):

        if num_col == 1:
            num_patches = int(img_path.shape[1] / 200.0)
        else:
            num_patches = int(img_path.shape[1] / 140.0)
        # num_patches=int(img_path.shape[1]/200.)
        if num_patches == 0:
            num_patches = 1

        img_patch_ineterst = img_path[:, :]  # [peaks_neg_true[14]-dis_up:peaks_neg_true[15]+dis_down ,:]

        # plt.imshow(img_patch_ineterst)
        # plt.show()

        length_x = int(img_path.shape[1] / float(num_patches))
        # margin = int(0.04 * length_x) just recently this was changed because it break lines into 2
        margin = int(0.04 * length_x)
        # print(margin,'margin')
        # if margin<=4:
        # margin = int(0.08 * length_x)

        # margin=0

        width_mid = length_x - 2 * margin

        nxf = img_path.shape[1] / float(width_mid)

        if nxf > int(nxf):
            nxf = int(nxf) + 1
        else:
            nxf = int(nxf)

        slopes_tile_wise = []
        for i in range(nxf):
            if i == 0:
                index_x_d = i * width_mid
                index_x_u = index_x_d + length_x
            elif i > 0:
                index_x_d = i * width_mid
                index_x_u = index_x_d + length_x

            if index_x_u > img_path.shape[1]:
                index_x_u = img_path.shape[1]
                index_x_d = img_path.shape[1] - length_x

            # img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
            img_xline = img_patch_ineterst[:, index_x_d:index_x_u]

            sigma = 2
            try:
                slope_xline = self.return_deskew_slop(img_xline, sigma)
            except:
                slope_xline = 0

            if abs(slope_region) < 25 and abs(slope_xline) > 25:
                slope_xline = [slope_region][0]
            # if abs(slope_region)>70 and abs(slope_xline)<25:
            # slope_xline=[slope_region][0]
            slopes_tile_wise.append(slope_xline)
            # print(slope_xline,'xlineeee')
            img_line_rotated = rotate_image(img_xline, slope_xline)
            img_line_rotated[:, :][img_line_rotated[:, :] != 0] = 1

        # print(slopes_tile_wise,'slopes_tile_wise')
        img_patch_ineterst = img_path[:, :]  # [peaks_neg_true[14]-dis_up:peaks_neg_true[14]+dis_down ,:]

        img_patch_ineterst_revised = np.zeros(img_patch_ineterst.shape)

        for i in range(nxf):
            if i == 0:
                index_x_d = i * width_mid
                index_x_u = index_x_d + length_x
            elif i > 0:
                index_x_d = i * width_mid
                index_x_u = index_x_d + length_x

            if index_x_u > img_path.shape[1]:
                index_x_u = img_path.shape[1]
                index_x_d = img_path.shape[1] - length_x

            img_xline = img_patch_ineterst[:, index_x_d:index_x_u]

            img_int = np.zeros((img_xline.shape[0], img_xline.shape[1]))
            img_int[:, :] = img_xline[:, :]  # img_patch_org[:,:,0]

            img_resized = np.zeros((int(img_int.shape[0] * (1.2)), int(img_int.shape[1] * (3))))

            img_resized[int(img_int.shape[0] * (0.1)) : int(img_int.shape[0] * (0.1)) + img_int.shape[0], int(img_int.shape[1] * (1)) : int(img_int.shape[1] * (1)) + img_int.shape[1]] = img_int[:, :]
            # plt.imshow(img_xline)
            # plt.show()
            img_line_rotated = rotate_image(img_resized, slopes_tile_wise[i])
            img_line_rotated[:, :][img_line_rotated[:, :] != 0] = 1

            img_patch_seperated = seperate_lines_new_inside_teils2(img_line_rotated, 0)

            img_patch_seperated_returned = rotate_image(img_patch_seperated, -slopes_tile_wise[i])
            img_patch_seperated_returned[:, :][img_patch_seperated_returned[:, :] != 0] = 1

            img_patch_seperated_returned_true_size = img_patch_seperated_returned[int(img_int.shape[0] * (0.1)) : int(img_int.shape[0] * (0.1)) + img_int.shape[0], int(img_int.shape[1] * (1)) : int(img_int.shape[1] * (1)) + img_int.shape[1]]

            img_patch_seperated_returned_true_size = img_patch_seperated_returned_true_size[:, margin : length_x - margin]
            img_patch_ineterst_revised[:, index_x_d + margin : index_x_u - margin] = img_patch_seperated_returned_true_size

        # plt.imshow(img_patch_ineterst_revised)
        # plt.show()
        return img_patch_ineterst_revised


    def textline_contours_postprocessing(self, textline_mask, slope, contour_text_interest, box_ind, slope_first, add_boxes_coor_into_textlines=False):

        textline_mask = np.repeat(textline_mask[:, :, np.newaxis], 3, axis=2) * 255
        textline_mask = textline_mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_OPEN, kernel)
        textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_CLOSE, kernel)
        textline_mask = cv2.erode(textline_mask, kernel, iterations=2)
        # textline_mask = cv2.erode(textline_mask, kernel, iterations=1)

        # print(textline_mask.shape[0]/float(textline_mask.shape[1]),'miz')
        try:
            # if np.abs(slope)>.5 and textline_mask.shape[0]/float(textline_mask.shape[1])>3:
            # plt.imshow(textline_mask)
            # plt.show()

            # if abs(slope)>1:
            # x_help=30
            # y_help=2
            # else:
            # x_help=2
            # y_help=2

            x_help = 30
            y_help = 2

            textline_mask_help = np.zeros((textline_mask.shape[0] + int(2 * y_help), textline_mask.shape[1] + int(2 * x_help), 3))
            textline_mask_help[y_help : y_help + textline_mask.shape[0], x_help : x_help + textline_mask.shape[1], :] = np.copy(textline_mask[:, :, :])

            dst = rotate_image(textline_mask_help, slope)
            dst = dst[:, :, 0]
            dst[dst != 0] = 1

            # if np.abs(slope)>.5 and textline_mask.shape[0]/float(textline_mask.shape[1])>3:
            # plt.imshow(dst)
            # plt.show()

            contour_text_copy = contour_text_interest.copy()

            contour_text_copy[:, 0, 0] = contour_text_copy[:, 0, 0] - box_ind[0]
            contour_text_copy[:, 0, 1] = contour_text_copy[:, 0, 1] - box_ind[1]

            img_contour = np.zeros((box_ind[3], box_ind[2], 3))
            img_contour = cv2.fillPoly(img_contour, pts=[contour_text_copy], color=(255, 255, 255))

            # if np.abs(slope)>.5 and textline_mask.shape[0]/float(textline_mask.shape[1])>3:
            # plt.imshow(img_contour)
            # plt.show()

            img_contour_help = np.zeros((img_contour.shape[0] + int(2 * y_help), img_contour.shape[1] + int(2 * x_help), 3))

            img_contour_help[y_help : y_help + img_contour.shape[0], x_help : x_help + img_contour.shape[1], :] = np.copy(img_contour[:, :, :])

            img_contour_rot = rotate_image(img_contour_help, slope)

            # plt.imshow(img_contour_rot_help)
            # plt.show()

            # plt.imshow(dst_help)
            # plt.show()

            # if np.abs(slope)>.5 and textline_mask.shape[0]/float(textline_mask.shape[1])>3:
            # plt.imshow(img_contour_rot_help)
            # plt.show()

            # plt.imshow(dst_help)
            # plt.show()

            img_contour_rot = img_contour_rot.astype(np.uint8)
            # dst_help = dst_help.astype(np.uint8)
            imgrayrot = cv2.cvtColor(img_contour_rot, cv2.COLOR_BGR2GRAY)
            _, threshrot = cv2.threshold(imgrayrot, 0, 255, 0)
            contours_text_rot, _ = cv2.findContours(threshrot.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            len_con_text_rot = [len(contours_text_rot[ib]) for ib in range(len(contours_text_rot))]
            ind_big_con = np.argmax(len_con_text_rot)

            # print('juzaa')
            if abs(slope) > 45:
                # print(add_boxes_coor_into_textlines,'avval')
                _, contours_rotated_clean = seperate_lines_vertical_cont(textline_mask, contours_text_rot[ind_big_con], box_ind, slope, add_boxes_coor_into_textlines=add_boxes_coor_into_textlines)
            else:
                _, contours_rotated_clean = seperate_lines(dst, contours_text_rot[ind_big_con], slope, x_help, y_help)

        except:

            contours_rotated_clean = []

        return contours_rotated_clean

    def textline_contours_to_get_slope_correctly(self, textline_mask, img_patch, contour_interest):

        slope_new = 0  # deskew_images(img_patch)

        textline_mask = np.repeat(textline_mask[:, :, np.newaxis], 3, axis=2) * 255

        textline_mask = textline_mask.astype(np.uint8)
        textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_OPEN, self.kernel)
        textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_CLOSE, self.kernel)
        textline_mask = cv2.erode(textline_mask, self.kernel, iterations=1)
        imgray = cv2.cvtColor(textline_mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 0, 255, 0)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)

        contours, hirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        main_contours = filter_contours_area_of_image_tables(thresh, contours, hirarchy, max_area=1, min_area=0.003)

        textline_maskt = textline_mask[:, :, 0]
        textline_maskt[textline_maskt != 0] = 1

        peaks_point, _ = seperate_lines(textline_maskt, contour_interest, slope_new)

        mean_dis = np.mean(np.diff(peaks_point))

        len_x = thresh.shape[1]

        slope_lines = []
        contours_slope_new = []

        for kk in range(len(main_contours)):

            if len(main_contours[kk].shape) == 2:
                xminh = np.min(main_contours[kk][:, 0])
                xmaxh = np.max(main_contours[kk][:, 0])

                yminh = np.min(main_contours[kk][:, 1])
                ymaxh = np.max(main_contours[kk][:, 1])
            elif len(main_contours[kk].shape) == 3:
                xminh = np.min(main_contours[kk][:, 0, 0])
                xmaxh = np.max(main_contours[kk][:, 0, 0])

                yminh = np.min(main_contours[kk][:, 0, 1])
                ymaxh = np.max(main_contours[kk][:, 0, 1])

            if ymaxh - yminh <= mean_dis and (xmaxh - xminh) >= 0.3 * len_x:  # xminh>=0.05*len_x and xminh<=0.4*len_x and xmaxh<=0.95*len_x and xmaxh>=0.6*len_x:
                contours_slope_new.append(main_contours[kk])

                rows, cols = thresh.shape[:2]
                [vx, vy, x, y] = cv2.fitLine(main_contours[kk], cv2.DIST_L2, 0, 0.01, 0.01)

                slope_lines.append((vy / vx) / np.pi * 180)

            if len(slope_lines) >= 2:

                slope = np.mean(slope_lines)  # slope_true/np.pi*180
            else:
                slope = 999

        else:
            slope = 0

        return slope

    def find_contours_mean_y_diff(self, contours_main):
        M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
        cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
        return np.mean(np.diff(np.sort(np.array(cy_main))))


    def return_deskew_slop(self, img_patch_org, sigma_des, main_page=False):

        if main_page and self.dir_of_all is not None:

            plt.figure(figsize=(70, 40))
            plt.rcParams["font.size"] = "50"
            plt.subplot(1, 2, 1)
            plt.imshow(img_patch_org)
            plt.subplot(1, 2, 2)
            plt.plot(gaussian_filter1d(img_patch_org.sum(axis=1), 3), np.array(range(len(gaussian_filter1d(img_patch_org.sum(axis=1), 3)))), linewidth=8)
            plt.xlabel("Density of textline prediction in direction of X axis", fontsize=60)
            plt.ylabel("Height", fontsize=60)
            plt.yticks([0, len(gaussian_filter1d(img_patch_org.sum(axis=1), 3))])
            plt.gca().invert_yaxis()

            plt.savefig(os.path.join(self.dir_of_all, self.f_name + "_density_of_textline.png"))
        # print(np.max(img_patch_org.sum(axis=0)) ,np.max(img_patch_org.sum(axis=1)),'axislar')

        # img_patch_org=resize_image(img_patch_org,int(img_patch_org.shape[0]*2.5),int(img_patch_org.shape[1]/2.5))

        # print(np.max(img_patch_org.sum(axis=0)) ,np.max(img_patch_org.sum(axis=1)),'axislar2')

        img_int = np.zeros((img_patch_org.shape[0], img_patch_org.shape[1]))
        img_int[:, :] = img_patch_org[:, :]  # img_patch_org[:,:,0]

        img_resized = np.zeros((int(img_int.shape[0] * (1.8)), int(img_int.shape[1] * (2.6))))

        img_resized[int(img_int.shape[0] * (0.4)) : int(img_int.shape[0] * (0.4)) + img_int.shape[0], int(img_int.shape[1] * (0.8)) : int(img_int.shape[1] * (0.8)) + img_int.shape[1]] = img_int[:, :]

        if main_page and img_patch_org.shape[1] > img_patch_org.shape[0]:

            # plt.imshow(img_resized)
            # plt.show()
            angels = np.array(
                [
                    -45,
                    0,
                    45,
                    90,
                ]
            )  # np.linspace(-12,12,100)#np.array([0 , 45 , 90 , -45])

            res = []
            num_of_peaks = []
            index_cor = []
            var_res = []

            indexer = 0
            for rot in angels:
                img_rot = rotate_image(img_resized, rot)
                # plt.imshow(img_rot)
                # plt.show()
                img_rot[img_rot != 0] = 1
                # res_me=np.mean(find_num_col_deskew(img_rot,sigma_des,2.0  ))

                # neg_peaks,var_spectrum=find_num_col_deskew(img_rot,sigma_des,20.3  )
                # print(var_spectrum,'var_spectrum')
                try:
                    neg_peaks, var_spectrum = find_num_col_deskew(img_rot, sigma_des, 20.3)
                    # print(rot,var_spectrum,'var_spectrum')
                    res_me = np.mean(neg_peaks)
                    if res_me == 0:
                        res_me = VERY_LARGE_NUMBER
                    else:
                        pass

                    res_num = len(neg_peaks)
                except:
                    res_me = VERY_LARGE_NUMBER
                    res_num = 0
                    var_spectrum = 0
                if isNaN(res_me):
                    pass
                else:
                    res.append(res_me)
                    var_res.append(var_spectrum)
                    num_of_peaks.append(res_num)
                    index_cor.append(indexer)
                indexer = indexer + 1

            try:
                var_res = np.array(var_res)

                ang_int = angels[np.argmax(var_res)]  # angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
            except:
                ang_int = 0

            angels = np.linspace(ang_int - 22.5, ang_int + 22.5, 100)

            res = []
            num_of_peaks = []
            index_cor = []
            var_res = []

            indexer = 0
            for rot in angels:
                img_rot = rotate_image(img_resized, rot)
                ##plt.imshow(img_rot)
                ##plt.show()
                img_rot[img_rot != 0] = 1
                # res_me=np.mean(find_num_col_deskew(img_rot,sigma_des,2.0  ))
                try:
                    neg_peaks, var_spectrum = find_num_col_deskew(img_rot, sigma_des, 20.3)
                    # print(indexer,'indexer')
                    res_me = np.mean(neg_peaks)
                    if res_me == 0:
                        res_me = VERY_LARGE_NUMBER
                    else:
                        pass

                    res_num = len(neg_peaks)
                except:
                    res_me = VERY_LARGE_NUMBER
                    res_num = 0
                    var_spectrum = 0
                if isNaN(res_me):
                    pass
                else:
                    res.append(res_me)
                    var_res.append(var_spectrum)
                    num_of_peaks.append(res_num)
                    index_cor.append(indexer)
                indexer = indexer + 1

            try:
                var_res = np.array(var_res)

                ang_int = angels[np.argmax(var_res)]  # angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
            except:
                ang_int = 0

        elif main_page and img_patch_org.shape[1] <= img_patch_org.shape[0]:

            # plt.imshow(img_resized)
            # plt.show()
            angels = np.linspace(-12, 12, 100)  # np.array([0 , 45 , 90 , -45])

            res = []
            num_of_peaks = []
            index_cor = []
            var_res = []

            indexer = 0
            for rot in angels:
                img_rot = rotate_image(img_resized, rot)
                # plt.imshow(img_rot)
                # plt.show()
                img_rot[img_rot != 0] = 1
                # res_me=np.mean(find_num_col_deskew(img_rot,sigma_des,2.0  ))

                # neg_peaks,var_spectrum=find_num_col_deskew(img_rot,sigma_des,20.3  )
                # print(var_spectrum,'var_spectrum')
                try:
                    neg_peaks, var_spectrum = find_num_col_deskew(img_rot, sigma_des, 20.3)
                    # print(rot,var_spectrum,'var_spectrum')
                    res_me = np.mean(neg_peaks)
                    if res_me == 0:
                        res_me = VERY_LARGE_NUMBER
                    else:
                        pass

                    res_num = len(neg_peaks)
                except:
                    res_me = VERY_LARGE_NUMBER
                    res_num = 0
                    var_spectrum = 0
                if isNaN(res_me):
                    pass
                else:
                    res.append(res_me)
                    var_res.append(var_spectrum)
                    num_of_peaks.append(res_num)
                    index_cor.append(indexer)
                indexer = indexer + 1

            if self.dir_of_all is not None:
                print("galdi?")
                plt.figure(figsize=(60, 30))
                plt.rcParams["font.size"] = "50"
                plt.plot(angels, np.array(var_res), "-o", markersize=25, linewidth=4)
                plt.xlabel("angle", fontsize=50)
                plt.ylabel("variance of sum of rotated textline in direction of x axis", fontsize=50)

                plt.plot(angels[np.argmax(var_res)], var_res[np.argmax(np.array(var_res))], "*", markersize=50, label="Angle of deskewing=" + str("{:.2f}".format(angels[np.argmax(var_res)])) + r"$\degree$")
                plt.legend(loc="best")
                plt.savefig(os.path.join(self.dir_of_all, self.f_name + "_rotation_angle.png"))

            try:
                var_res = np.array(var_res)

                ang_int = angels[np.argmax(var_res)]  # angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
            except:
                ang_int = 0

            early_slope_edge = 11
            if abs(ang_int) > early_slope_edge and ang_int < 0:

                angels = np.linspace(-90, -12, 100)

                res = []
                num_of_peaks = []
                index_cor = []
                var_res = []

                indexer = 0
                for rot in angels:
                    img_rot = rotate_image(img_resized, rot)
                    ##plt.imshow(img_rot)
                    ##plt.show()
                    img_rot[img_rot != 0] = 1
                    # res_me=np.mean(find_num_col_deskew(img_rot,sigma_des,2.0  ))
                    try:
                        neg_peaks, var_spectrum = find_num_col_deskew(img_rot, sigma_des, 20.3)
                        # print(indexer,'indexer')
                        res_me = np.mean(neg_peaks)
                        if res_me == 0:
                            res_me = VERY_LARGE_NUMBER
                        else:
                            pass

                        res_num = len(neg_peaks)
                    except:
                        res_me = VERY_LARGE_NUMBER
                        res_num = 0
                        var_spectrum = 0
                    if isNaN(res_me):
                        pass
                    else:
                        res.append(res_me)
                        var_res.append(var_spectrum)
                        num_of_peaks.append(res_num)
                        index_cor.append(indexer)
                    indexer = indexer + 1

                try:
                    var_res = np.array(var_res)

                    ang_int = angels[np.argmax(var_res)]  # angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
                except:
                    ang_int = 0

            elif abs(ang_int) > early_slope_edge and ang_int > 0:

                angels = np.linspace(90, 12, 100)

                res = []
                num_of_peaks = []
                index_cor = []
                var_res = []

                indexer = 0
                for rot in angels:
                    img_rot = rotate_image(img_resized, rot)
                    ##plt.imshow(img_rot)
                    ##plt.show()
                    img_rot[img_rot != 0] = 1
                    # res_me=np.mean(find_num_col_deskew(img_rot,sigma_des,2.0  ))
                    try:
                        neg_peaks, var_spectrum = find_num_col_deskew(img_rot, sigma_des, 20.3)
                        # print(indexer,'indexer')
                        res_me = np.mean(neg_peaks)
                        if res_me == 0:
                            res_me = VERY_LARGE_NUMBER
                        else:
                            pass

                        res_num = len(neg_peaks)
                    except:
                        res_me = VERY_LARGE_NUMBER
                        res_num = 0
                        var_spectrum = 0
                    if isNaN(res_me):
                        pass
                    else:
                        res.append(res_me)
                        var_res.append(var_spectrum)
                        num_of_peaks.append(res_num)
                        index_cor.append(indexer)
                    indexer = indexer + 1

                try:
                    var_res = np.array(var_res)

                    ang_int = angels[np.argmax(var_res)]  # angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
                except:
                    ang_int = 0
        else:

            angels = np.linspace(-25, 25, 60)

            res = []
            num_of_peaks = []
            index_cor = []
            var_res = []

            indexer = 0
            for rot in angels:
                img_rot = rotate_image(img_resized, rot)
                # plt.imshow(img_rot)
                # plt.show()
                img_rot[img_rot != 0] = 1
                # res_me=np.mean(find_num_col_deskew(img_rot,sigma_des,2.0  ))

                # neg_peaks,var_spectrum=find_num_col_deskew(img_rot,sigma_des,20.3  )
                # print(var_spectrum,'var_spectrum')
                try:
                    neg_peaks, var_spectrum = find_num_col_deskew(img_rot, sigma_des, 20.3)
                    # print(rot,var_spectrum,'var_spectrum')
                    res_me = np.mean(neg_peaks)
                    if res_me == 0:
                        res_me = VERY_LARGE_NUMBER
                    else:
                        pass

                    res_num = len(neg_peaks)
                except:
                    res_me = VERY_LARGE_NUMBER
                    res_num = 0
                    var_spectrum = 0
                if isNaN(res_me):
                    pass
                else:
                    res.append(res_me)
                    var_res.append(var_spectrum)
                    num_of_peaks.append(res_num)
                    index_cor.append(indexer)
                indexer = indexer + 1

            try:
                var_res = np.array(var_res)

                ang_int = angels[np.argmax(var_res)]  # angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
            except:
                ang_int = 0

            # print(ang_int,'ang_int')

            early_slope_edge = 22
            if abs(ang_int) > early_slope_edge and ang_int < 0:

                angels = np.linspace(-90, -25, 60)

                res = []
                num_of_peaks = []
                index_cor = []
                var_res = []

                indexer = 0
                for rot in angels:
                    img_rot = rotate_image(img_resized, rot)
                    ##plt.imshow(img_rot)
                    ##plt.show()
                    img_rot[img_rot != 0] = 1
                    # res_me=np.mean(find_num_col_deskew(img_rot,sigma_des,2.0  ))
                    try:
                        neg_peaks, var_spectrum = find_num_col_deskew(img_rot, sigma_des, 20.3)
                        # print(indexer,'indexer')
                        res_me = np.mean(neg_peaks)
                        if res_me == 0:
                            res_me = VERY_LARGE_NUMBER
                        else:
                            pass

                        res_num = len(neg_peaks)
                    except:
                        res_me = VERY_LARGE_NUMBER
                        res_num = 0
                        var_spectrum = 0
                    if isNaN(res_me):
                        pass
                    else:
                        res.append(res_me)
                        var_res.append(var_spectrum)
                        num_of_peaks.append(res_num)
                        index_cor.append(indexer)
                    indexer = indexer + 1

                try:
                    var_res = np.array(var_res)

                    ang_int = angels[np.argmax(var_res)]  # angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
                except:
                    ang_int = 0

            elif abs(ang_int) > early_slope_edge and ang_int > 0:

                angels = np.linspace(90, 25, 60)

                res = []
                num_of_peaks = []
                index_cor = []
                var_res = []

                indexer = 0
                for rot in angels:
                    img_rot = rotate_image(img_resized, rot)
                    ##plt.imshow(img_rot)
                    ##plt.show()
                    img_rot[img_rot != 0] = 1
                    # res_me=np.mean(find_num_col_deskew(img_rot,sigma_des,2.0  ))
                    try:
                        neg_peaks, var_spectrum = find_num_col_deskew(img_rot, sigma_des, 20.3)
                        # print(indexer,'indexer')
                        res_me = np.mean(neg_peaks)
                        if res_me == 0:
                            res_me = VERY_LARGE_NUMBER
                        else:
                            pass

                        res_num = len(neg_peaks)
                    except:
                        res_me = VERY_LARGE_NUMBER
                        res_num = 0
                        var_spectrum = 0
                    if isNaN(res_me):
                        pass
                    else:
                        res.append(res_me)
                        var_res.append(var_spectrum)
                        num_of_peaks.append(res_num)
                        index_cor.append(indexer)
                    indexer = indexer + 1

                try:
                    var_res = np.array(var_res)

                    ang_int = angels[np.argmax(var_res)]  # angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
                except:
                    ang_int = 0

        return ang_int

    def return_deskew_slope_new(self, img_patch, sigma_des):
        max_x_y = max(img_patch.shape[0], img_patch.shape[1])

        ##img_patch=resize_image(img_patch,max_x_y,max_x_y)

        img_patch_copy = np.zeros((img_patch.shape[0], img_patch.shape[1]))
        img_patch_copy[:, :] = img_patch[:, :]  # img_patch_org[:,:,0]

        img_patch_padded = np.zeros((int(max_x_y * (1.4)), int(max_x_y * (1.4))))

        img_patch_padded_center_p = int(img_patch_padded.shape[0] / 2.0)
        len_x_org_patch_half = int(img_patch_copy.shape[1] / 2.0)
        len_y_org_patch_half = int(img_patch_copy.shape[0] / 2.0)

        img_patch_padded[img_patch_padded_center_p - len_y_org_patch_half : img_patch_padded_center_p - len_y_org_patch_half + img_patch_copy.shape[0], img_patch_padded_center_p - len_x_org_patch_half : img_patch_padded_center_p - len_x_org_patch_half + img_patch_copy.shape[1]] = img_patch_copy[:, :]
        # img_patch_padded[ int( img_patch_copy.shape[0]*(.1)):int( img_patch_copy.shape[0]*(.1))+img_patch_copy.shape[0] , int( img_patch_copy.shape[1]*(.8)):int( img_patch_copy.shape[1]*(.8))+img_patch_copy.shape[1] ]=img_patch_copy[:,:]
        angles = np.linspace(-25, 25, 80)

        res = []
        num_of_peaks = []
        index_cor = []
        var_res = []

        # plt.imshow(img_patch)
        # plt.show()
        indexer = 0
        for rot in angles:
            # print(rot,'rot')
            img_rotated = rotate_image(img_patch_padded, rot)
            img_rotated[img_rotated != 0] = 1

            # plt.imshow(img_rotated)
            # plt.show()

            try:
                neg_peaks, var_spectrum = self.get_standard_deviation_of_summed_textline_patch_along_width(img_rotated, sigma_des, 20.3)
                res_me = np.mean(neg_peaks)
                if res_me == 0:
                    res_me = VERY_LARGE_NUMBER
                else:
                    pass

                res_num = len(neg_peaks)
            except:
                res_me = VERY_LARGE_NUMBER
                res_num = 0
                var_spectrum = 0
            if isNaN(res_me):
                pass
            else:
                res.append(res_me)
                var_res.append(var_spectrum)
                num_of_peaks.append(res_num)
                index_cor.append(indexer)
            indexer = indexer + 1

        try:
            var_res = np.array(var_res)
            # print(var_res)

            ang_int = angles[np.argmax(var_res)]  # angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
        except:
            ang_int = 0

        if abs(ang_int) > 15:
            angles = np.linspace(-90, -50, 30)
            res = []
            num_of_peaks = []
            index_cor = []
            var_res = []

            # plt.imshow(img_patch)
            # plt.show()
            indexer = 0
            for rot in angles:
                # print(rot,'rot')
                img_rotated = rotate_image(img_patch_padded, rot)
                img_rotated[img_rotated != 0] = 1

                # plt.imshow(img_rotated)
                # plt.show()

                try:
                    neg_peaks, var_spectrum = self.get_standard_deviation_of_summed_textline_patch_along_width(img_rotated, sigma_des, 20.3)
                    res_me = np.mean(neg_peaks)
                    if res_me == 0:
                        res_me = VERY_LARGE_NUMBER
                    else:
                        pass

                    res_num = len(neg_peaks)
                except:
                    res_me = VERY_LARGE_NUMBER
                    res_num = 0
                    var_spectrum = 0
                if isNaN(res_me):
                    pass
                else:
                    res.append(res_me)
                    var_res.append(var_spectrum)
                    num_of_peaks.append(res_num)
                    index_cor.append(indexer)
                indexer = indexer + 1

            try:
                var_res = np.array(var_res)
                # print(var_res)

                ang_int = angles[np.argmax(var_res)]  # angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
            except:
                ang_int = 0

        return ang_int

    def do_work_of_slopes(self, q, poly, box_sub, boxes_per_process, textline_mask_tot, contours_per_process):
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
                y_diff_mean = self.find_contours_mean_y_diff(textline_con_fil)

                sigma_des = int(y_diff_mean * (4.0 / 40.0))

                if sigma_des < 1:
                    sigma_des = 1

                crop_img[crop_img > 0] = 1
                slope_corresponding_textregion = self.return_deskew_slop(crop_img, sigma_des)

            except:
                slope_corresponding_textregion = 999

            if slope_corresponding_textregion == 999:
                slope_corresponding_textregion = slope_biggest
            ##if np.abs(slope_corresponding_textregion)>12.5 and slope_corresponding_textregion!=999:
            ##slope_corresponding_textregion=slope_biggest
            ##elif slope_corresponding_textregion==999:
            ##slope_corresponding_textregion=slope_biggest
            slopes_sub.append(slope_corresponding_textregion)

            cnt_clean_rot = self.textline_contours_postprocessing(crop_img, slope_corresponding_textregion, contours_per_process[mv], boxes_per_process[mv])

            poly_sub.append(cnt_clean_rot)
            boxes_sub_new.append(boxes_per_process[mv])

        q.put(slopes_sub)
        poly.put(poly_sub)
        box_sub.put(boxes_sub_new)

    def get_slopes_and_deskew(self, contours, textline_mask_tot):

        slope_biggest = 0  # self.return_deskew_slop(img_int_p,sigma_des)

        num_cores = cpu_count()
        q = Queue()
        poly = Queue()
        box_sub = Queue()

        processes = []
        nh = np.linspace(0, len(self.boxes), num_cores + 1)

        for i in range(num_cores):
            boxes_per_process = self.boxes[int(nh[i]) : int(nh[i + 1])]
            contours_per_process = contours[int(nh[i]) : int(nh[i + 1])]
            processes.append(Process(target=self.do_work_of_slopes, args=(q, poly, box_sub, boxes_per_process, textline_mask_tot, contours_per_process)))

        for i in range(num_cores):
            processes[i].start()

        self.slopes = []
        self.all_found_texline_polygons = []
        self.boxes = []

        for i in range(num_cores):
            slopes_for_sub_process = q.get(True)
            boxes_for_sub_process = box_sub.get(True)
            polys_for_sub_process = poly.get(True)

            for j in range(len(slopes_for_sub_process)):
                self.slopes.append(slopes_for_sub_process[j])
                self.all_found_texline_polygons.append(polys_for_sub_process[j])
                self.boxes.append(boxes_for_sub_process[j])

        for i in range(num_cores):
            processes[i].join()

    def order_of_regions_old(self, textline_mask, contours_main):
        mada_n = textline_mask.sum(axis=1)
        y = mada_n[:]

        y_help = np.zeros(len(y) + 40)
        y_help[20 : len(y) + 20] = y
        x = np.array(range(len(y)))

        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)

        sigma_gaus = 8

        z = gaussian_filter1d(y_help, sigma_gaus)
        zneg_rev = -y_help + np.max(y_help)

        zneg = np.zeros(len(zneg_rev) + 40)
        zneg[20 : len(zneg_rev) + 20] = zneg_rev
        zneg = gaussian_filter1d(zneg, sigma_gaus)

        peaks, _ = find_peaks(z, height=0)
        peaks_neg, _ = find_peaks(zneg, height=0)

        peaks_neg = peaks_neg - 20 - 20
        peaks = peaks - 20

        if contours_main != None:
            areas_main = np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
            M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
            cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
            cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
            x_min_main = np.array([np.min(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])
            x_max_main = np.array([np.max(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

            y_min_main = np.array([np.min(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
            y_max_main = np.array([np.max(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])

        if contours_main != None:
            indexer_main = np.array(range(len(contours_main)))

        if contours_main != None:
            len_main = len(contours_main)
        else:
            len_main = 0

        matrix_of_orders = np.zeros((len_main, 5))

        matrix_of_orders[:, 0] = np.array(range(len_main))

        matrix_of_orders[:len_main, 1] = 1
        matrix_of_orders[len_main:, 1] = 2

        matrix_of_orders[:len_main, 2] = cx_main
        matrix_of_orders[:len_main, 3] = cy_main

        matrix_of_orders[:len_main, 4] = np.array(range(len_main))

        peaks_neg_new = []
        peaks_neg_new.append(0)
        for iii in range(len(peaks_neg)):
            peaks_neg_new.append(peaks_neg[iii])
        peaks_neg_new.append(textline_mask.shape[0])

        final_indexers_sorted = []
        for i in range(len(peaks_neg_new) - 1):
            top = peaks_neg_new[i]
            down = peaks_neg_new[i + 1]

            indexes_in = matrix_of_orders[:, 0][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]
            cxs_in = matrix_of_orders[:, 2][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]

            sorted_inside = np.argsort(cxs_in)

            ind_in_int = indexes_in[sorted_inside]

            for j in range(len(ind_in_int)):
                final_indexers_sorted.append(int(ind_in_int[j]))

        return final_indexers_sorted, matrix_of_orders

    def order_and_id_of_texts_old(self, found_polygons_text_region, matrix_of_orders, indexes_sorted):
        id_of_texts = []
        order_of_texts = []
        index_b = 0
        for mm in range(len(found_polygons_text_region)):
            id_of_texts.append("r" + str(index_b))
            index_matrix = matrix_of_orders[:, 0][(matrix_of_orders[:, 1] == 1) & (matrix_of_orders[:, 4] == mm)]
            order_of_texts.append(np.where(indexes_sorted == index_matrix)[0][0])

            index_b += 1

        order_of_texts
        return order_of_texts, id_of_texts

    def write_into_page_xml_only_textlines(self, contours, page_coord, all_found_texline_polygons, all_box_coord, dir_of_image):

        found_polygons_text_region = contours

        # create the file structure
        data = ET.Element("PcGts")

        data.set("xmlns", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")
        data.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        data.set("xsi:schemaLocation", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")

        metadata = ET.SubElement(data, "Metadata")

        author = ET.SubElement(metadata, "Creator")
        author.text = "SBB_QURATOR"

        created = ET.SubElement(metadata, "Created")
        created.text = "2019-06-17T18:15:12"

        changetime = ET.SubElement(metadata, "LastChange")
        changetime.text = "2019-06-17T18:15:12"

        page = ET.SubElement(data, "Page")

        page.set("imageFilename", self.image_dir)
        page.set("imageHeight", str(self.height_org))
        page.set("imageWidth", str(self.width_org))
        page.set("type", "content")
        page.set("readingDirection", "left-to-right")
        page.set("textLineOrder", "top-to-bottom")

        page_print_sub = ET.SubElement(page, "PrintSpace")
        coord_page = ET.SubElement(page_print_sub, "Coords")
        points_page_print = ""

        for lmm in range(len(self.cont_page[0])):
            if len(self.cont_page[0][lmm]) == 2:
                points_page_print = points_page_print + str(int((self.cont_page[0][lmm][0]) / self.scale_x))
                points_page_print = points_page_print + ","
                points_page_print = points_page_print + str(int((self.cont_page[0][lmm][1]) / self.scale_y))
            else:
                points_page_print = points_page_print + str(int((self.cont_page[0][lmm][0][0]) / self.scale_x))
                points_page_print = points_page_print + ","
                points_page_print = points_page_print + str(int((self.cont_page[0][lmm][0][1]) / self.scale_y))

            if lmm < (len(self.cont_page[0]) - 1):
                points_page_print = points_page_print + " "
        coord_page.set("points", points_page_print)

        if len(contours) > 0:

            id_indexer = 0
            id_indexer_l = 0

            for mm in range(len(found_polygons_text_region)):
                textregion = ET.SubElement(page, "TextRegion")

                textregion.set("id", "r" + str(id_indexer))
                id_indexer += 1

                textregion.set("type", "paragraph")
                # if mm==0:
                #    textregion.set('type','header')
                # else:
                #    textregion.set('type','paragraph')
                coord_text = ET.SubElement(textregion, "Coords")

                points_co = ""
                for lmm in range(len(found_polygons_text_region[mm])):
                    if len(found_polygons_text_region[mm][lmm]) == 2:
                        points_co = points_co + str(int((found_polygons_text_region[mm][lmm][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_text_region[mm][lmm][1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co = points_co + str(int((found_polygons_text_region[mm][lmm][0][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_text_region[mm][lmm][0][1] + page_coord[0]) / self.scale_y))

                    if lmm < (len(found_polygons_text_region[mm]) - 1):
                        points_co = points_co + " "
                # print(points_co)
                coord_text.set("points", points_co)

                for j in range(len(all_found_texline_polygons[mm])):

                    textline = ET.SubElement(textregion, "TextLine")

                    textline.set("id", "l" + str(id_indexer_l))

                    id_indexer_l += 1

                    coord = ET.SubElement(textline, "Coords")

                    texteq = ET.SubElement(textline, "TextEquiv")

                    uni = ET.SubElement(texteq, "Unicode")
                    uni.text = " "

                    # points = ET.SubElement(coord, 'Points')

                    points_co = ""
                    for l in range(len(all_found_texline_polygons[mm][j])):
                        # point = ET.SubElement(coord, 'Point')

                        # point.set('x',str(found_polygons[j][l][0]))
                        # point.set('y',str(found_polygons[j][l][1]))
                        if len(all_found_texline_polygons[mm][j][l]) == 2:
                            points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0] + page_coord[2]) / self.scale_x))
                            points_co = points_co + ","
                            points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][1] + page_coord[0]) / self.scale_y))
                        else:
                            points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][0] + page_coord[2]) / self.scale_x))
                            points_co = points_co + ","
                            points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][1] + page_coord[0]) / self.scale_y))

                        if l < (len(all_found_texline_polygons[mm][j]) - 1):
                            points_co = points_co + " "
                    # print(points_co)
                    coord.set("points", points_co)

                texteqreg = ET.SubElement(textregion, "TextEquiv")

                unireg = ET.SubElement(texteqreg, "Unicode")
                unireg.text = " "

        # print(dir_of_image)
        print(self.f_name)
        # print(os.path.join(dir_of_image, self.f_name) + ".xml")
        tree = ET.ElementTree(data)
        tree.write(os.path.join(dir_of_image, self.f_name) + ".xml")

    def write_into_page_xml_full(self, contours, contours_h, page_coord, dir_of_image, order_of_texts, id_of_texts, all_found_texline_polygons, all_found_texline_polygons_h, all_box_coord, all_box_coord_h, found_polygons_text_region_img, found_polygons_tables, found_polygons_drop_capitals, found_polygons_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals):

        found_polygons_text_region = contours
        found_polygons_text_region_h = contours_h

        # create the file structure
        data = ET.Element("PcGts")

        data.set("xmlns", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")
        data.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        data.set("xsi:schemaLocation", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")

        metadata = ET.SubElement(data, "Metadata")

        author = ET.SubElement(metadata, "Creator")
        author.text = "SBB_QURATOR"

        created = ET.SubElement(metadata, "Created")
        created.text = "2019-06-17T18:15:12"

        changetime = ET.SubElement(metadata, "LastChange")
        changetime.text = "2019-06-17T18:15:12"

        page = ET.SubElement(data, "Page")

        page.set("imageFilename", self.image_dir)
        page.set("imageHeight", str(self.height_org))
        page.set("imageWidth", str(self.width_org))
        page.set("type", "content")
        page.set("readingDirection", "left-to-right")
        page.set("textLineOrder", "top-to-bottom")

        page_print_sub = ET.SubElement(page, "PrintSpace")
        coord_page = ET.SubElement(page_print_sub, "Coords")
        points_page_print = ""

        for lmm in range(len(self.cont_page[0])):
            if len(self.cont_page[0][lmm]) == 2:
                points_page_print = points_page_print + str(int((self.cont_page[0][lmm][0]) / self.scale_x))
                points_page_print = points_page_print + ","
                points_page_print = points_page_print + str(int((self.cont_page[0][lmm][1]) / self.scale_y))
            else:
                points_page_print = points_page_print + str(int((self.cont_page[0][lmm][0][0]) / self.scale_x))
                points_page_print = points_page_print + ","
                points_page_print = points_page_print + str(int((self.cont_page[0][lmm][0][1]) / self.scale_y))

            if lmm < (len(self.cont_page[0]) - 1):
                points_page_print = points_page_print + " "
        coord_page.set("points", points_page_print)

        if len(contours) > 0:
            region_order = ET.SubElement(page, "ReadingOrder")
            region_order_sub = ET.SubElement(region_order, "OrderedGroup")

            region_order_sub.set("id", "ro357564684568544579089")

            # args_sort=order_of_texts
            for vj in order_of_texts:
                name = "coord_text_" + str(vj)
                name = ET.SubElement(region_order_sub, "RegionRefIndexed")
                name.set("index", str(order_of_texts[vj]))
                name.set("regionRef", id_of_texts[vj])

            id_of_marginalia = []
            indexer_region = len(contours) + len(contours_h)
            for vm in range(len(found_polygons_marginals)):
                id_of_marginalia.append("r" + str(indexer_region))

                name = "coord_text_" + str(indexer_region)
                name = ET.SubElement(region_order_sub, "RegionRefIndexed")
                name.set("index", str(indexer_region))
                name.set("regionRef", "r" + str(indexer_region))
                indexer_region += 1

            id_indexer = 0
            id_indexer_l = 0

            for mm in range(len(found_polygons_text_region)):
                textregion = ET.SubElement(page, "TextRegion")

                textregion.set("id", "r" + str(id_indexer))
                id_indexer += 1

                textregion.set("type", "paragraph")
                # if mm==0:
                #    textregion.set('type','header')
                # else:
                #    textregion.set('type','paragraph')
                coord_text = ET.SubElement(textregion, "Coords")

                points_co = ""
                for lmm in range(len(found_polygons_text_region[mm])):
                    if len(found_polygons_text_region[mm][lmm]) == 2:
                        points_co = points_co + str(int((found_polygons_text_region[mm][lmm][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_text_region[mm][lmm][1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co = points_co + str(int((found_polygons_text_region[mm][lmm][0][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_text_region[mm][lmm][0][1] + page_coord[0]) / self.scale_y))

                    if lmm < (len(found_polygons_text_region[mm]) - 1):
                        points_co = points_co + " "
                # print(points_co)
                coord_text.set("points", points_co)

                for j in range(len(all_found_texline_polygons[mm])):

                    textline = ET.SubElement(textregion, "TextLine")

                    textline.set("id", "l" + str(id_indexer_l))

                    id_indexer_l += 1

                    coord = ET.SubElement(textline, "Coords")

                    texteq = ET.SubElement(textline, "TextEquiv")

                    uni = ET.SubElement(texteq, "Unicode")
                    uni.text = " "

                    # points = ET.SubElement(coord, 'Points')

                    points_co = ""
                    for l in range(len(all_found_texline_polygons[mm][j])):
                        # point = ET.SubElement(coord, 'Point')

                        if not self.curved_line:
                            # point.set('x',str(found_polygons[j][l][0]))
                            # point.set('y',str(found_polygons[j][l][1]))
                            if len(all_found_texline_polygons[mm][j][l]) == 2:
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0] + all_box_coord[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][1] + all_box_coord[mm][0] + page_coord[0]) / self.scale_y))
                            else:
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][0] + all_box_coord[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][1] + all_box_coord[mm][0] + page_coord[0]) / self.scale_y))

                        if (self.curved_line) and np.abs(slopes[mm]) <= 45:
                            if len(all_found_texline_polygons[mm][j][l]) == 2:
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][1] + page_coord[0]) / self.scale_y))
                            else:
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][0] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][1] + page_coord[0]) / self.scale_y))
                        elif (self.curved_line) and np.abs(slopes[mm]) > 45:
                            if len(all_found_texline_polygons[mm][j][l]) == 2:
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0] + all_box_coord[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][1] + all_box_coord[mm][0] + page_coord[0]) / self.scale_y))
                            else:
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][0] + all_box_coord[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][1] + all_box_coord[mm][0] + page_coord[0]) / self.scale_y))

                        if l < (len(all_found_texline_polygons[mm][j]) - 1):
                            points_co = points_co + " "
                    # print(points_co)
                    coord.set("points", points_co)

                texteqreg = ET.SubElement(textregion, "TextEquiv")

                unireg = ET.SubElement(texteqreg, "Unicode")
                unireg.text = " "

        print(len(contours_h))
        if len(contours_h) > 0:
            for mm in range(len(found_polygons_text_region_h)):
                textregion = ET.SubElement(page, "TextRegion")
                try:
                    id_indexer = id_indexer
                    id_indexer_l = id_indexer_l
                except:
                    id_indexer = 0
                    id_indexer_l = 0
                textregion.set("id", "r" + str(id_indexer))
                id_indexer += 1

                textregion.set("type", "header")
                # if mm==0:
                #    textregion.set('type','header')
                # else:
                #    textregion.set('type','paragraph')
                coord_text = ET.SubElement(textregion, "Coords")

                points_co = ""
                for lmm in range(len(found_polygons_text_region_h[mm])):

                    if len(found_polygons_text_region_h[mm][lmm]) == 2:
                        points_co = points_co + str(int((found_polygons_text_region_h[mm][lmm][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_text_region_h[mm][lmm][1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co = points_co + str(int((found_polygons_text_region_h[mm][lmm][0][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_text_region_h[mm][lmm][0][1] + page_coord[0]) / self.scale_y))

                    if lmm < (len(found_polygons_text_region_h[mm]) - 1):
                        points_co = points_co + " "
                # print(points_co)
                coord_text.set("points", points_co)

                for j in range(len(all_found_texline_polygons_h[mm])):

                    textline = ET.SubElement(textregion, "TextLine")

                    textline.set("id", "l" + str(id_indexer_l))

                    id_indexer_l += 1

                    coord = ET.SubElement(textline, "Coords")

                    texteq = ET.SubElement(textline, "TextEquiv")

                    uni = ET.SubElement(texteq, "Unicode")
                    uni.text = " "

                    # points = ET.SubElement(coord, 'Points')

                    points_co = ""
                    for l in range(len(all_found_texline_polygons_h[mm][j])):
                        # point = ET.SubElement(coord, 'Point')

                        if not self.curved_line:
                            # point.set('x',str(found_polygons[j][l][0]))
                            # point.set('y',str(found_polygons[j][l][1]))
                            if len(all_found_texline_polygons_h[mm][j][l]) == 2:
                                points_co = points_co + str(int((all_found_texline_polygons_h[mm][j][l][0] + all_box_coord_h[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons_h[mm][j][l][1] + all_box_coord_h[mm][0] + page_coord[0]) / self.scale_y))
                            else:
                                points_co = points_co + str(int((all_found_texline_polygons_h[mm][j][l][0][0] + all_box_coord_h[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons_h[mm][j][l][0][1] + all_box_coord_h[mm][0] + page_coord[0]) / self.scale_y))

                        if self.curved_line:
                            if len(all_found_texline_polygons_h[mm][j][l]) == 2:
                                points_co = points_co + str(int((all_found_texline_polygons_h[mm][j][l][0] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons_h[mm][j][l][1] + page_coord[0]) / self.scale_y))
                            else:
                                points_co = points_co + str(int((all_found_texline_polygons_h[mm][j][l][0][0] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons_h[mm][j][l][0][1] + page_coord[0]) / self.scale_y))

                        if l < (len(all_found_texline_polygons_h[mm][j]) - 1):
                            points_co = points_co + " "
                    # print(points_co)
                    coord.set("points", points_co)

                texteqreg = ET.SubElement(textregion, "TextEquiv")

                unireg = ET.SubElement(texteqreg, "Unicode")
                unireg.text = " "

        if len(found_polygons_drop_capitals) > 0:
            id_indexer = len(contours_h) + len(contours) + len(found_polygons_marginals)
            for mm in range(len(found_polygons_drop_capitals)):
                textregion = ET.SubElement(page, "TextRegion")

                # id_indexer_l=id_indexer_l

                textregion.set("id", "r" + str(id_indexer))
                id_indexer += 1

                textregion.set("type", "drop-capital")
                # if mm==0:
                #    textregion.set('type','header')
                # else:
                #    textregion.set('type','paragraph')
                coord_text = ET.SubElement(textregion, "Coords")

                points_co = ""
                for lmm in range(len(found_polygons_drop_capitals[mm])):

                    if len(found_polygons_drop_capitals[mm][lmm]) == 2:
                        points_co = points_co + str(int((found_polygons_drop_capitals[mm][lmm][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_drop_capitals[mm][lmm][1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co = points_co + str(int((found_polygons_drop_capitals[mm][lmm][0][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_drop_capitals[mm][lmm][0][1] + page_coord[0]) / self.scale_y))

                    if lmm < (len(found_polygons_drop_capitals[mm]) - 1):
                        points_co = points_co + " "
                # print(points_co)
                coord_text.set("points", points_co)

                ##for j in range(len(all_found_texline_polygons_h[mm])):

                ##textline=ET.SubElement(textregion, 'TextLine')

                ##textline.set('id','l'+str(id_indexer_l))

                ##id_indexer_l+=1

                ##coord = ET.SubElement(textline, 'Coords')

                ##texteq=ET.SubElement(textline, 'TextEquiv')

                ##uni=ET.SubElement(texteq, 'Unicode')
                ##uni.text = ' '

                ###points = ET.SubElement(coord, 'Points')

                ##points_co=''
                ##for l in range(len(all_found_texline_polygons_h[mm][j])):
                ###point = ET.SubElement(coord, 'Point')

                ##if not curved_line:
                ###point.set('x',str(found_polygons[j][l][0]))
                ###point.set('y',str(found_polygons[j][l][1]))
                ##if len(all_found_texline_polygons_h[mm][j][l])==2:
                ##points_co=points_co+str( int( (all_found_texline_polygons_h[mm][j][l][0]
                ##+all_box_coord_h[mm][2]+page_coord[2])/self.scale_x) )
                ##points_co=points_co+','
                ##points_co=points_co+str( int( (all_found_texline_polygons_h[mm][j][l][1]
                ##+all_box_coord_h[mm][0]+page_coord[0])/self.scale_y) )
                ##else:
                ##points_co=points_co+str( int( ( all_found_texline_polygons_h[mm][j][l][0][0]
                ##+all_box_coord_h[mm][2]+page_coord[2])/self.scale_x ) )
                ##points_co=points_co+','
                ##points_co=points_co+str( int( ( all_found_texline_polygons_h[mm][j][l][0][1]
                ##+all_box_coord_h[mm][0]+page_coord[0])/self.scale_y) )

                ##if curved_line:
                ##if len(all_found_texline_polygons_h[mm][j][l])==2:
                ##points_co=points_co+str( int( (all_found_texline_polygons_h[mm][j][l][0]
                ##+page_coord[2])/self.scale_x) )
                ##points_co=points_co+','
                ##points_co=points_co+str( int( (all_found_texline_polygons_h[mm][j][l][1]
                ##+page_coord[0])/self.scale_y) )
                ##else:
                ##points_co=points_co+str( int( ( all_found_texline_polygons_h[mm][j][l][0][0]
                ##+page_coord[2])/self.scale_x ) )
                ##points_co=points_co+','
                ##points_co=points_co+str( int( ( all_found_texline_polygons_h[mm][j][l][0][1]
                ##+page_coord[0])/self.scale_y) )

                ##if l<(len(all_found_texline_polygons_h[mm][j])-1):
                ##points_co=points_co+' '
                ###print(points_co)
                ####coord.set('points',points_co)

                texteqreg = ET.SubElement(textregion, "TextEquiv")

                unireg = ET.SubElement(texteqreg, "Unicode")
                unireg.text = " "

        try:

            try:
                ###id_indexer=id_indexer
                id_indexer_l = id_indexer_l
            except:
                ###id_indexer=0
                id_indexer_l = 0
            for mm in range(len(found_polygons_marginals)):
                textregion = ET.SubElement(page, "TextRegion")

                textregion.set("id", id_of_marginalia[mm])

                textregion.set("type", "marginalia")
                # if mm==0:
                #    textregion.set('type','header')
                # else:
                #    textregion.set('type','paragraph')
                coord_text = ET.SubElement(textregion, "Coords")

                points_co = ""
                for lmm in range(len(found_polygons_marginals[mm])):
                    if len(found_polygons_marginals[mm][lmm]) == 2:
                        points_co = points_co + str(int((found_polygons_marginals[mm][lmm][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_marginals[mm][lmm][1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co = points_co + str(int((found_polygons_marginals[mm][lmm][0][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_marginals[mm][lmm][0][1] + page_coord[0]) / self.scale_y))

                    if lmm < (len(found_polygons_marginals[mm]) - 1):
                        points_co = points_co + " "
                # print(points_co)
                coord_text.set("points", points_co)

                for j in range(len(all_found_texline_polygons_marginals[mm])):

                    textline = ET.SubElement(textregion, "TextLine")

                    textline.set("id", "l" + str(id_indexer_l))

                    id_indexer_l += 1

                    coord = ET.SubElement(textline, "Coords")

                    texteq = ET.SubElement(textline, "TextEquiv")

                    uni = ET.SubElement(texteq, "Unicode")
                    uni.text = " "

                    # points = ET.SubElement(coord, 'Points')

                    points_co = ""
                    for l in range(len(all_found_texline_polygons_marginals[mm][j])):
                        # point = ET.SubElement(coord, 'Point')

                        if not self.curved_line:
                            # point.set('x',str(found_polygons[j][l][0]))
                            # point.set('y',str(found_polygons[j][l][1]))
                            if len(all_found_texline_polygons_marginals[mm][j][l]) == 2:
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][0] + all_box_coord_marginals[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][1] + all_box_coord_marginals[mm][0] + page_coord[0]) / self.scale_y))
                            else:
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][0][0] + all_box_coord_marginals[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][0][1] + all_box_coord_marginals[mm][0] + page_coord[0]) / self.scale_y))

                        if self.curved_line:
                            if len(all_found_texline_polygons_marginals[mm][j][l]) == 2:
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][0] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][1] + page_coord[0]) / self.scale_y))
                            else:
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][0][0] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][0][1] + page_coord[0]) / self.scale_y))

                        if l < (len(all_found_texline_polygons_marginals[mm][j]) - 1):
                            points_co = points_co + " "
                    # print(points_co)
                    coord.set("points", points_co)

                texteqreg = ET.SubElement(textregion, "TextEquiv")

                unireg = ET.SubElement(texteqreg, "Unicode")
                unireg.text = " "
        except:
            pass

        try:
            id_indexer = len(contours_h) + len(contours) + len(found_polygons_marginals) + len(found_polygons_drop_capitals)
            for mm in range(len(found_polygons_text_region_img)):
                textregion = ET.SubElement(page, "ImageRegion")

                textregion.set("id", "r" + str(id_indexer))
                id_indexer += 1

                coord_text = ET.SubElement(textregion, "Coords")

                points_co = ""
                for lmm in range(len(found_polygons_text_region_img[mm])):

                    if len(found_polygons_text_region_img[mm][lmm]) == 2:
                        points_co = points_co + str(int((found_polygons_text_region_img[mm][lmm][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_text_region_img[mm][lmm][1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co = points_co + str(int((found_polygons_text_region_img[mm][lmm][0][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_text_region_img[mm][lmm][0][1] + page_coord[0]) / self.scale_y))

                    if lmm < (len(found_polygons_text_region_img[mm]) - 1):
                        points_co = points_co + " "

                coord_text.set("points", points_co)
        except:
            pass

        try:
            for mm in range(len(found_polygons_tables)):
                textregion = ET.SubElement(page, "TableRegion")

                textregion.set("id", "r" + str(id_indexer))
                id_indexer += 1

                coord_text = ET.SubElement(textregion, "Coords")

                points_co = ""
                for lmm in range(len(found_polygons_tables[mm])):

                    if len(found_polygons_tables[mm][lmm]) == 2:
                        points_co = points_co + str(int((found_polygons_tables[mm][lmm][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_tables[mm][lmm][1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co = points_co + str(int((found_polygons_tables[mm][lmm][0][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_tables[mm][lmm][0][1] + page_coord[0]) / self.scale_y))

                    if lmm < (len(found_polygons_tables[mm]) - 1):
                        points_co = points_co + " "

                coord_text.set("points", points_co)
        except:
            pass

        print(dir_of_image)
        print(self.f_name)
        print(os.path.join(dir_of_image, self.f_name) + ".xml")
        tree = ET.ElementTree(data)
        tree.write(os.path.join(dir_of_image, self.f_name) + ".xml")

    def write_into_page_xml(self, contours, page_coord, dir_of_image, order_of_texts, id_of_texts, all_found_texline_polygons, all_box_coord, found_polygons_text_region_img, found_polygons_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, curved_line, slopes, slopes_marginals):

        found_polygons_text_region = contours
        ##found_polygons_text_region_h=contours_h

        # create the file structure
        data = ET.Element("PcGts")

        data.set("xmlns", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")
        data.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        data.set("xsi:schemaLocation", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")

        metadata = ET.SubElement(data, "Metadata")

        author = ET.SubElement(metadata, "Creator")
        author.text = "SBB_QURATOR"

        created = ET.SubElement(metadata, "Created")
        created.text = "2019-06-17T18:15:12"

        changetime = ET.SubElement(metadata, "LastChange")
        changetime.text = "2019-06-17T18:15:12"

        page = ET.SubElement(data, "Page")

        page.set("imageFilename", self.image_dir)
        page.set("imageHeight", str(self.height_org))
        page.set("imageWidth", str(self.width_org))
        page.set("type", "content")
        page.set("readingDirection", "left-to-right")
        page.set("textLineOrder", "top-to-bottom")

        page_print_sub = ET.SubElement(page, "PrintSpace")
        coord_page = ET.SubElement(page_print_sub, "Coords")
        points_page_print = ""

        for lmm in range(len(self.cont_page[0])):
            if len(self.cont_page[0][lmm]) == 2:
                points_page_print = points_page_print + str(int((self.cont_page[0][lmm][0]) / self.scale_x))
                points_page_print = points_page_print + ","
                points_page_print = points_page_print + str(int((self.cont_page[0][lmm][1]) / self.scale_y))
            else:
                points_page_print = points_page_print + str(int((self.cont_page[0][lmm][0][0]) / self.scale_x))
                points_page_print = points_page_print + ","
                points_page_print = points_page_print + str(int((self.cont_page[0][lmm][0][1]) / self.scale_y))

            if lmm < (len(self.cont_page[0]) - 1):
                points_page_print = points_page_print + " "
        coord_page.set("points", points_page_print)

        if len(contours) > 0:
            region_order = ET.SubElement(page, "ReadingOrder")
            region_order_sub = ET.SubElement(region_order, "OrderedGroup")

            region_order_sub.set("id", "ro357564684568544579089")

            indexer_region = 0

            for vj in order_of_texts:
                name = "coord_text_" + str(vj)
                name = ET.SubElement(region_order_sub, "RegionRefIndexed")

                name.set("index", str(indexer_region))
                name.set("regionRef", id_of_texts[vj])
                indexer_region += 1

            id_of_marginalia = []
            for vm in range(len(found_polygons_marginals)):
                id_of_marginalia.append("r" + str(indexer_region))

                name = "coord_text_" + str(indexer_region)
                name = ET.SubElement(region_order_sub, "RegionRefIndexed")
                name.set("index", str(indexer_region))
                name.set("regionRef", "r" + str(indexer_region))
                indexer_region += 1

            id_indexer = 0
            id_indexer_l = 0

            for mm in range(len(found_polygons_text_region)):
                textregion = ET.SubElement(page, "TextRegion")

                textregion.set("id", "r" + str(id_indexer))
                id_indexer += 1

                textregion.set("type", "paragraph")
                # if mm==0:
                #    textregion.set('type','header')
                # else:
                #    textregion.set('type','paragraph')
                coord_text = ET.SubElement(textregion, "Coords")

                points_co = ""
                for lmm in range(len(found_polygons_text_region[mm])):
                    if len(found_polygons_text_region[mm][lmm]) == 2:
                        points_co = points_co + str(int((found_polygons_text_region[mm][lmm][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_text_region[mm][lmm][1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co = points_co + str(int((found_polygons_text_region[mm][lmm][0][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_text_region[mm][lmm][0][1] + page_coord[0]) / self.scale_y))

                    if lmm < (len(found_polygons_text_region[mm]) - 1):
                        points_co = points_co + " "
                # print(points_co)
                coord_text.set("points", points_co)

                for j in range(len(all_found_texline_polygons[mm])):

                    textline = ET.SubElement(textregion, "TextLine")

                    textline.set("id", "l" + str(id_indexer_l))

                    id_indexer_l += 1

                    coord = ET.SubElement(textline, "Coords")

                    texteq = ET.SubElement(textline, "TextEquiv")

                    uni = ET.SubElement(texteq, "Unicode")
                    uni.text = " "

                    # points = ET.SubElement(coord, 'Points')

                    points_co = ""
                    for l in range(len(all_found_texline_polygons[mm][j])):
                        # point = ET.SubElement(coord, 'Point')

                        if not self.curved_line:
                            # point.set('x',str(found_polygons[j][l][0]))
                            # point.set('y',str(found_polygons[j][l][1]))
                            if len(all_found_texline_polygons[mm][j][l]) == 2:
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0] + all_box_coord[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][1] + all_box_coord[mm][0] + page_coord[0]) / self.scale_y))
                            else:
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][0] + all_box_coord[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][1] + all_box_coord[mm][0] + page_coord[0]) / self.scale_y))

                        if (self.curved_line) and abs(slopes[mm]) <= 45:
                            if len(all_found_texline_polygons[mm][j][l]) == 2:
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][1] + page_coord[0]) / self.scale_y))
                            else:
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][0] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][1] + page_coord[0]) / self.scale_y))

                        elif (self.curved_line) and abs(slopes[mm]) > 45:
                            if len(all_found_texline_polygons[mm][j][l]) == 2:
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0] + all_box_coord[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][1] + all_box_coord[mm][0] + page_coord[0]) / self.scale_y))
                            else:
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][0] + all_box_coord[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][1] + all_box_coord[mm][0] + page_coord[0]) / self.scale_y))

                        if l < (len(all_found_texline_polygons[mm][j]) - 1):
                            points_co = points_co + " "
                    # print(points_co)
                    coord.set("points", points_co)

                texteqreg = ET.SubElement(textregion, "TextEquiv")

                unireg = ET.SubElement(texteqreg, "Unicode")
                unireg.text = " "

        ###print(len(contours_h))
        ###if len(contours_h)>0:
        ###for mm in range(len(found_polygons_text_region_h)):
        ###textregion=ET.SubElement(page, 'TextRegion')
        ###try:
        ###id_indexer=id_indexer
        ###id_indexer_l=id_indexer_l
        ###except:
        ###id_indexer=0
        ###id_indexer_l=0
        ###textregion.set('id','r'+str(id_indexer))
        ###id_indexer+=1

        ###textregion.set('type','header')
        ####if mm==0:
        ####    textregion.set('type','header')
        ####else:
        ####    textregion.set('type','paragraph')
        ###coord_text = ET.SubElement(textregion, 'Coords')

        ###points_co=''
        ###for lmm in range(len(found_polygons_text_region_h[mm])):

        ###if len(found_polygons_text_region_h[mm][lmm])==2:
        ###points_co=points_co+str( int( (found_polygons_text_region_h[mm][lmm][0] +page_coord[2])/self.scale_x ) )
        ###points_co=points_co+','
        ###points_co=points_co+str( int( (found_polygons_text_region_h[mm][lmm][1] +page_coord[0])/self.scale_y ) )
        ###else:
        ###points_co=points_co+str( int((found_polygons_text_region_h[mm][lmm][0][0] +page_coord[2])/self.scale_x) )
        ###points_co=points_co+','
        ###points_co=points_co+str( int((found_polygons_text_region_h[mm][lmm][0][1] +page_coord[0])/self.scale_y) )

        ###if lmm<(len(found_polygons_text_region_h[mm])-1):
        ###points_co=points_co+' '
        ####print(points_co)
        ###coord_text.set('points',points_co)

        ###for j in range(len(all_found_texline_polygons_h[mm])):

        ###textline=ET.SubElement(textregion, 'TextLine')

        ###textline.set('id','l'+str(id_indexer_l))

        ###id_indexer_l+=1

        ###coord = ET.SubElement(textline, 'Coords')

        ###texteq=ET.SubElement(textline, 'TextEquiv')

        ###uni=ET.SubElement(texteq, 'Unicode')
        ###uni.text = ' '

        ####points = ET.SubElement(coord, 'Points')

        ###points_co=''
        ###for l in range(len(all_found_texline_polygons_h[mm][j])):
        ####point = ET.SubElement(coord, 'Point')

        ####point.set('x',str(found_polygons[j][l][0]))
        ####point.set('y',str(found_polygons[j][l][1]))
        ###if len(all_found_texline_polygons_h[mm][j][l])==2:
        ###points_co=points_co+str( int( (all_found_texline_polygons_h[mm][j][l][0] +page_coord[2]
        ###+all_box_coord_h[mm][2])/self.scale_x) )
        ###points_co=points_co+','
        ###points_co=points_co+str( int( (all_found_texline_polygons_h[mm][j][l][1] +page_coord[0]
        ###+all_box_coord_h[mm][0])/self.scale_y) )
        ###else:
        ###points_co=points_co+str( int( ( all_found_texline_polygons_h[mm][j][l][0][0] +page_coord[2]
        ###+all_box_coord_h[mm][2])/self.scale_x ) )
        ###points_co=points_co+','
        ###points_co=points_co+str( int( ( all_found_texline_polygons_h[mm][j][l][0][1] +page_coord[0]
        ###+all_box_coord_h[mm][0])/self.scale_y) )

        ###if l<(len(all_found_texline_polygons_h[mm][j])-1):
        ###points_co=points_co+' '
        ####print(points_co)
        ###coord.set('points',points_co)

        ###texteqreg=ET.SubElement(textregion, 'TextEquiv')

        ###unireg=ET.SubElement(texteqreg, 'Unicode')
        ###unireg.text = ' '
        try:
            # id_indexer_l=0

            try:
                ###id_indexer=id_indexer
                id_indexer_l = id_indexer_l
            except:
                ###id_indexer=0
                id_indexer_l = 0

            for mm in range(len(found_polygons_marginals)):
                textregion = ET.SubElement(page, "TextRegion")

                textregion.set("id", id_of_marginalia[mm])

                textregion.set("type", "marginalia")
                # if mm==0:
                #    textregion.set('type','header')
                # else:
                #    textregion.set('type','paragraph')
                coord_text = ET.SubElement(textregion, "Coords")

                points_co = ""
                for lmm in range(len(found_polygons_marginals[mm])):
                    if len(found_polygons_marginals[mm][lmm]) == 2:
                        points_co = points_co + str(int((found_polygons_marginals[mm][lmm][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_marginals[mm][lmm][1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co = points_co + str(int((found_polygons_marginals[mm][lmm][0][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((found_polygons_marginals[mm][lmm][0][1] + page_coord[0]) / self.scale_y))

                    if lmm < (len(found_polygons_marginals[mm]) - 1):
                        points_co = points_co + " "
                # print(points_co)
                coord_text.set("points", points_co)

                for j in range(len(all_found_texline_polygons_marginals[mm])):

                    textline = ET.SubElement(textregion, "TextLine")

                    textline.set("id", "l" + str(id_indexer_l))

                    id_indexer_l += 1

                    coord = ET.SubElement(textline, "Coords")

                    texteq = ET.SubElement(textline, "TextEquiv")

                    uni = ET.SubElement(texteq, "Unicode")
                    uni.text = " "

                    # points = ET.SubElement(coord, 'Points')

                    points_co = ""
                    for l in range(len(all_found_texline_polygons_marginals[mm][j])):
                        # point = ET.SubElement(coord, 'Point')

                        if not self.curved_line:
                            # point.set('x',str(found_polygons[j][l][0]))
                            # point.set('y',str(found_polygons[j][l][1]))
                            if len(all_found_texline_polygons_marginals[mm][j][l]) == 2:
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][0] + all_box_coord_marginals[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][1] + all_box_coord_marginals[mm][0] + page_coord[0]) / self.scale_y))
                            else:
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][0][0] + all_box_coord_marginals[mm][2] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][0][1] + all_box_coord_marginals[mm][0] + page_coord[0]) / self.scale_y))

                        if self.curved_line:
                            if len(all_found_texline_polygons_marginals[mm][j][l]) == 2:
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][0] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][1] + page_coord[0]) / self.scale_y))
                            else:
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][0][0] + page_coord[2]) / self.scale_x))
                                points_co = points_co + ","
                                points_co = points_co + str(int((all_found_texline_polygons_marginals[mm][j][l][0][1] + page_coord[0]) / self.scale_y))

                        if l < (len(all_found_texline_polygons_marginals[mm][j]) - 1):
                            points_co = points_co + " "
                    # print(points_co)
                    coord.set("points", points_co)
        except:
            pass

        try:

            for mm in range(len(found_polygons_text_region_img)):
                textregion = ET.SubElement(page, "ImageRegion")

                textregion.set("id", "r" + str(id_indexer))
                id_indexer += 1

                coord_text = ET.SubElement(textregion, "Coords")
                points_co = ""
                for lmm in range(len(found_polygons_text_region_img[mm])):
                    points_co = points_co + str(int((found_polygons_text_region_img[mm][lmm, 0, 0] + page_coord[2]) / self.scale_x))
                    points_co = points_co + ","
                    points_co = points_co + str(int((found_polygons_text_region_img[mm][lmm, 0, 1] + page_coord[0]) / self.scale_y))

                    if lmm < (len(found_polygons_text_region_img[mm]) - 1):
                        points_co = points_co + " "

                coord_text.set("points", points_co)
            ###for mm in range(len(found_polygons_text_region_img)):
            ###textregion=ET.SubElement(page, 'ImageRegion')

            ###textregion.set('id','r'+str(id_indexer))
            ###id_indexer+=1

            ###coord_text = ET.SubElement(textregion, 'Coords')
            ###print(found_polygons_text_region_img[mm])
            ###points_co=''
            ###for lmm in range(len(found_polygons_text_region_img[mm])):
            ###print(len(found_polygons_text_region_img[mm][lmm]))

            ###if len(found_polygons_text_region_img[mm][lmm])==2:
            ###points_co=points_co+str( int( (found_polygons_text_region_img[mm][lmm][0]+page_coord[2] )/self.scale_x ) )
            ###points_co=points_co+','
            ###points_co=points_co+str( int( (found_polygons_text_region_img[mm][lmm][1]+page_coord[0] )/self.scale_y ) )
            ###else:
            ###points_co=points_co+str( int((found_polygons_text_region_img[mm][lmm][0][0]+page_coord[2] )/self.scale_x) )
            ###points_co=points_co+','
            ###points_co=points_co+str( int((found_polygons_text_region_img[mm][lmm][0][1]+page_coord[0] )/self.scale_y) )

            ###if lmm<(len(found_polygons_text_region_img[mm])-1):
            ###points_co=points_co+' '

            ###coord_text.set('points',points_co)
        except:
            pass

        ####try:
        ####for mm in range(len(found_polygons_tables)):
        ####textregion=ET.SubElement(page, 'TableRegion')

        ####textregion.set('id','r'+str(id_indexer))
        ####id_indexer+=1

        ####coord_text = ET.SubElement(textregion, 'Coords')

        ####points_co=''
        ####for lmm in range(len(found_polygons_tables[mm])):

        ####if len(found_polygons_tables[mm][lmm])==2:
        ####points_co=points_co+str( int( (found_polygons_tables[mm][lmm][0] +page_coord[2])/self.scale_x ) )
        ####points_co=points_co+','
        ####points_co=points_co+str( int( (found_polygons_tables[mm][lmm][1] +page_coord[0])/self.scale_y ) )
        ####else:
        ####points_co=points_co+str( int((found_polygons_tables[mm][lmm][0][0] +page_coord[2])/self.scale_x) )
        ####points_co=points_co+','
        ####points_co=points_co+str( int((found_polygons_tables[mm][lmm][0][1] +page_coord[0])/self.scale_y) )

        ####if lmm<(len(found_polygons_tables[mm])-1):
        ####points_co=points_co+' '

        ####coord_text.set('points',points_co)
        ####except:
        ####pass
        """

        try:
            for mm in range(len(found_polygons_drop_capitals)):
                textregion=ET.SubElement(page, 'DropCapitals')

                textregion.set('id','r'+str(id_indexer))
                id_indexer+=1


                coord_text = ET.SubElement(textregion, 'Coords')

                points_co=''
                for lmm in range(len(found_polygons_drop_capitals[mm])):

                    if len(found_polygons_drop_capitals[mm][lmm])==2:
                        points_co=points_co+str( int( (found_polygons_drop_capitals[mm][lmm][0] +page_coord[2])/self.scale_x ) )
                        points_co=points_co+','
                        points_co=points_co+str( int( (found_polygons_drop_capitals[mm][lmm][1] +page_coord[0])/self.scale_y ) )
                    else:
                        points_co=points_co+str( int((found_polygons_drop_capitals[mm][lmm][0][0] +page_coord[2])/self.scale_x) )
                        points_co=points_co+','
                        points_co=points_co+str( int((found_polygons_drop_capitals[mm][lmm][0][1] +page_coord[0])/self.scale_y) )

                    if lmm<(len(found_polygons_drop_capitals[mm])-1):
                        points_co=points_co+' '


                coord_text.set('points',points_co)
        except:
            pass
        """

        # print(dir_of_image)
        print(self.f_name)
        # print(os.path.join(dir_of_image, self.f_name) + ".xml")
        tree = ET.ElementTree(data)
        tree.write(os.path.join(dir_of_image, self.f_name) + ".xml")
        # cv2.imwrite(os.path.join(dir_of_image, self.f_name) + ".tif",self.image_org)


    def return_regions_without_seperators(self, regions_pre):
        kernel = np.ones((5, 5), np.uint8)
        regions_without_seperators = ((regions_pre[:, :] != 6) & (regions_pre[:, :] != 0)) * 1
        # regions_without_seperators=( (image_regions_eraly_p[:,:,:]!=6) & (image_regions_eraly_p[:,:,:]!=0) & (image_regions_eraly_p[:,:,:]!=5) & (image_regions_eraly_p[:,:,:]!=8) & (image_regions_eraly_p[:,:,:]!=7))*1

        regions_without_seperators = regions_without_seperators.astype(np.uint8)

        regions_without_seperators = cv2.erode(regions_without_seperators, kernel, iterations=6)

        return regions_without_seperators


    def image_change_background_pixels_to_zero(self, image_page):
        image_back_zero = np.zeros((image_page.shape[0], image_page.shape[1]))
        image_back_zero[:, :] = image_page[:, :, 0]
        image_back_zero[:, :][image_back_zero[:, :] == 0] = -255
        image_back_zero[:, :][image_back_zero[:, :] == 255] = 0
        image_back_zero[:, :][image_back_zero[:, :] == -255] = 255
        return image_back_zero

    def find_num_col_only_image(self, regions_without_seperators, multiplier=3.8):
        regions_without_seperators_0 = regions_without_seperators[:, :].sum(axis=0)

        ##plt.plot(regions_without_seperators_0)
        ##plt.show()

        sigma_ = 15

        meda_n_updown = regions_without_seperators_0[len(regions_without_seperators_0) :: -1]

        first_nonzero = next((i for i, x in enumerate(regions_without_seperators_0) if x), 0)
        last_nonzero = next((i for i, x in enumerate(meda_n_updown) if x), 0)

        last_nonzero = len(regions_without_seperators_0) - last_nonzero

        y = regions_without_seperators_0  # [first_nonzero:last_nonzero]

        y_help = np.zeros(len(y) + 20)

        y_help[10 : len(y) + 10] = y

        x = np.array(range(len(y)))

        zneg_rev = -y_help + np.max(y_help)

        zneg = np.zeros(len(zneg_rev) + 20)

        zneg[10 : len(zneg_rev) + 10] = zneg_rev

        z = gaussian_filter1d(y, sigma_)
        zneg = gaussian_filter1d(zneg, sigma_)

        peaks_neg, _ = find_peaks(zneg, height=0)
        peaks, _ = find_peaks(z, height=0)

        peaks_neg = peaks_neg - 10 - 10

        peaks_neg_org = np.copy(peaks_neg)

        peaks_neg = peaks_neg[(peaks_neg > first_nonzero) & (peaks_neg < last_nonzero)]

        peaks = peaks[(peaks > 0.09 * regions_without_seperators.shape[1]) & (peaks < 0.91 * regions_without_seperators.shape[1])]

        peaks_neg = peaks_neg[(peaks_neg > 500) & (peaks_neg < (regions_without_seperators.shape[1] - 500))]
        # print(peaks)
        interest_pos = z[peaks]

        interest_pos = interest_pos[interest_pos > 10]

        interest_neg = z[peaks_neg]
        min_peaks_pos = np.mean(interest_pos)  # np.min(interest_pos)
        min_peaks_neg = 0  # np.min(interest_neg)

        # $print(min_peaks_pos)
        dis_talaei = (min_peaks_pos - min_peaks_neg) / multiplier
        # print(interest_pos)
        grenze = min_peaks_pos - dis_talaei  # np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

        interest_neg_fin = interest_neg[(interest_neg < grenze)]
        peaks_neg_fin = peaks_neg[(interest_neg < grenze)]

        num_col = (len(interest_neg_fin)) + 1

        p_l = 0
        p_u = len(y) - 1
        p_m = int(len(y) / 2.0)
        p_g_l = int(len(y) / 3.0)
        p_g_u = len(y) - int(len(y) / 3.0)

        if num_col == 3:
            if (peaks_neg_fin[0] > p_g_u and peaks_neg_fin[1] > p_g_u) or (peaks_neg_fin[0] < p_g_l and peaks_neg_fin[1] < p_g_l) or (peaks_neg_fin[0] < p_m and peaks_neg_fin[1] < p_m) or (peaks_neg_fin[0] > p_m and peaks_neg_fin[1] > p_m):
                num_col = 1
            else:
                pass

        if num_col == 2:
            if (peaks_neg_fin[0] > p_g_u) or (peaks_neg_fin[0] < p_g_l):
                num_col = 1
            else:
                pass

        diff_peaks = np.abs(np.diff(peaks_neg_fin))

        cut_off = 400
        peaks_neg_true = []
        forest = []

        for i in range(len(peaks_neg_fin)):
            if i == 0:
                forest.append(peaks_neg_fin[i])
            if i < (len(peaks_neg_fin) - 1):
                if diff_peaks[i] <= cut_off:
                    forest.append(peaks_neg_fin[i + 1])
                if diff_peaks[i] > cut_off:
                    # print(forest[np.argmin(z[forest]) ] )
                    if not isNaN(forest[np.argmin(z[forest])]):
                        peaks_neg_true.append(forest[np.argmin(z[forest])])
                    forest = []
                    forest.append(peaks_neg_fin[i + 1])
            if i == (len(peaks_neg_fin) - 1):
                # print(print(forest[np.argmin(z[forest]) ] ))
                if not isNaN(forest[np.argmin(z[forest])]):
                    peaks_neg_true.append(forest[np.argmin(z[forest])])

        num_col = (len(peaks_neg_true)) + 1
        p_l = 0
        p_u = len(y) - 1
        p_m = int(len(y) / 2.0)
        p_quarter = int(len(y) / 4.0)
        p_g_l = int(len(y) / 3.0)
        p_g_u = len(y) - int(len(y) / 3.0)

        p_u_quarter = len(y) - p_quarter

        if num_col == 3:
            if (peaks_neg_true[0] > p_g_u and peaks_neg_true[1] > p_g_u) or (peaks_neg_true[0] < p_g_l and peaks_neg_true[1] < p_g_l) or (peaks_neg_true[0] < p_m and peaks_neg_true[1] < p_m) or (peaks_neg_true[0] > p_m and peaks_neg_true[1] > p_m):
                num_col = 1
                peaks_neg_true = []
            elif (peaks_neg_true[0] < p_g_u and peaks_neg_true[0] > p_g_l) and (peaks_neg_true[1] > p_u_quarter):
                peaks_neg_true = [peaks_neg_true[0]]
            elif (peaks_neg_true[1] < p_g_u and peaks_neg_true[1] > p_g_l) and (peaks_neg_true[0] < p_quarter):
                peaks_neg_true = [peaks_neg_true[1]]
            else:
                pass

        if num_col == 2:
            if (peaks_neg_true[0] > p_g_u) or (peaks_neg_true[0] < p_g_l):
                num_col = 1
                peaks_neg_true = []

        if num_col == 4:
            if len(np.array(peaks_neg_true)[np.array(peaks_neg_true) < p_g_l]) == 2 or len(np.array(peaks_neg_true)[np.array(peaks_neg_true) > (len(y) - p_g_l)]) == 2:
                num_col = 1
                peaks_neg_true = []
            else:
                pass

        # no deeper hill around found hills

        peaks_fin_true = []
        for i in range(len(peaks_neg_true)):
            hill_main = peaks_neg_true[i]
            # deep_depth=z[peaks_neg]
            hills_around = peaks_neg_org[((peaks_neg_org > hill_main) & (peaks_neg_org <= hill_main + 400)) | ((peaks_neg_org < hill_main) & (peaks_neg_org >= hill_main - 400))]
            deep_depth_around = z[hills_around]

            # print(hill_main,z[hill_main],hills_around,deep_depth_around,'manoooo')
            try:
                if np.min(deep_depth_around) < z[hill_main]:
                    pass
                else:
                    peaks_fin_true.append(hill_main)
            except:
                pass

        diff_peaks_annormal = diff_peaks[diff_peaks < 360]

        if len(diff_peaks_annormal) > 0:
            arg_help = np.array(range(len(diff_peaks)))
            arg_help_ann = arg_help[diff_peaks < 360]

            peaks_neg_fin_new = []

            for ii in range(len(peaks_neg_fin)):
                if ii in arg_help_ann:
                    arg_min = np.argmin([interest_neg_fin[ii], interest_neg_fin[ii + 1]])
                    if arg_min == 0:
                        peaks_neg_fin_new.append(peaks_neg_fin[ii])
                    else:
                        peaks_neg_fin_new.append(peaks_neg_fin[ii + 1])

                elif (ii - 1) in arg_help_ann:
                    pass
                else:
                    peaks_neg_fin_new.append(peaks_neg_fin[ii])
        else:
            peaks_neg_fin_new = peaks_neg_fin

        # sometime pages with one columns gives also some negative peaks. delete those peaks
        param = z[peaks_neg_true] / float(min_peaks_pos) * 100

        if len(param[param <= 41]) == 0:
            peaks_neg_true = []

        return len(peaks_fin_true), peaks_fin_true

    def find_num_col_by_vertical_lines(self, regions_without_seperators, multiplier=3.8):
        regions_without_seperators_0 = regions_without_seperators[:, :, 0].sum(axis=0)

        ##plt.plot(regions_without_seperators_0)
        ##plt.show()

        sigma_ = 35  # 70#35

        z = gaussian_filter1d(regions_without_seperators_0, sigma_)

        peaks, _ = find_peaks(z, height=0)

        # print(peaks,'peaksnew')
        return peaks

    def find_num_col(self, regions_without_seperators, multiplier=3.8):
        regions_without_seperators_0 = regions_without_seperators[:, :].sum(axis=0)

        ##plt.plot(regions_without_seperators_0)
        ##plt.show()

        sigma_ = 35  # 70#35

        meda_n_updown = regions_without_seperators_0[len(regions_without_seperators_0) :: -1]

        first_nonzero = next((i for i, x in enumerate(regions_without_seperators_0) if x), 0)
        last_nonzero = next((i for i, x in enumerate(meda_n_updown) if x), 0)

        # print(last_nonzero)
        # print(isNaN(last_nonzero))
        # last_nonzero=0#halalikh
        last_nonzero = len(regions_without_seperators_0) - last_nonzero

        y = regions_without_seperators_0  # [first_nonzero:last_nonzero]

        y_help = np.zeros(len(y) + 20)

        y_help[10 : len(y) + 10] = y

        x = np.array(range(len(y)))

        zneg_rev = -y_help + np.max(y_help)

        zneg = np.zeros(len(zneg_rev) + 20)

        zneg[10 : len(zneg_rev) + 10] = zneg_rev

        z = gaussian_filter1d(y, sigma_)
        zneg = gaussian_filter1d(zneg, sigma_)

        peaks_neg, _ = find_peaks(zneg, height=0)
        peaks, _ = find_peaks(z, height=0)

        peaks_neg = peaks_neg - 10 - 10

        last_nonzero = last_nonzero - 100
        first_nonzero = first_nonzero + 200

        peaks_neg = peaks_neg[(peaks_neg > first_nonzero) & (peaks_neg < last_nonzero)]

        peaks = peaks[(peaks > 0.06 * regions_without_seperators.shape[1]) & (peaks < 0.94 * regions_without_seperators.shape[1])]
        peaks_neg = peaks_neg[(peaks_neg > 370) & (peaks_neg < (regions_without_seperators.shape[1] - 370))]

        # print(peaks)
        interest_pos = z[peaks]

        interest_pos = interest_pos[interest_pos > 10]

        # plt.plot(z)
        # plt.show()
        interest_neg = z[peaks_neg]

        min_peaks_pos = np.min(interest_pos)
        max_peaks_pos = np.max(interest_pos)

        if max_peaks_pos / min_peaks_pos >= 35:
            min_peaks_pos = np.mean(interest_pos)

        min_peaks_neg = 0  # np.min(interest_neg)

        # print(np.min(interest_pos),np.max(interest_pos),np.max(interest_pos)/np.min(interest_pos),'minmax')
        # $print(min_peaks_pos)
        dis_talaei = (min_peaks_pos - min_peaks_neg) / multiplier
        # print(interest_pos)
        grenze = min_peaks_pos - dis_talaei  # np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

        # print(interest_neg,'interest_neg')
        # print(grenze,'grenze')
        # print(min_peaks_pos,'min_peaks_pos')
        # print(dis_talaei,'dis_talaei')
        # print(peaks_neg,'peaks_neg')

        interest_neg_fin = interest_neg[(interest_neg < grenze)]
        peaks_neg_fin = peaks_neg[(interest_neg < grenze)]
        # interest_neg_fin=interest_neg[(interest_neg<grenze)]

        num_col = (len(interest_neg_fin)) + 1

        # print(peaks_neg_fin,'peaks_neg_fin')
        # print(num_col,'diz')
        p_l = 0
        p_u = len(y) - 1
        p_m = int(len(y) / 2.0)
        p_g_l = int(len(y) / 4.0)
        p_g_u = len(y) - int(len(y) / 4.0)

        if num_col == 3:
            if (peaks_neg_fin[0] > p_g_u and peaks_neg_fin[1] > p_g_u) or (peaks_neg_fin[0] < p_g_l and peaks_neg_fin[1] < p_g_l) or ((peaks_neg_fin[0] + 200) < p_m and peaks_neg_fin[1] < p_m) or ((peaks_neg_fin[0] - 200) > p_m and peaks_neg_fin[1] > p_m):
                num_col = 1
                peaks_neg_fin = []
            else:
                pass

        if num_col == 2:
            if (peaks_neg_fin[0] > p_g_u) or (peaks_neg_fin[0] < p_g_l):
                num_col = 1
                peaks_neg_fin = []
            else:
                pass

        ##print(len(peaks_neg_fin))

        diff_peaks = np.abs(np.diff(peaks_neg_fin))

        cut_off = 400
        peaks_neg_true = []
        forest = []

        # print(len(peaks_neg_fin),'len_')

        for i in range(len(peaks_neg_fin)):
            if i == 0:
                forest.append(peaks_neg_fin[i])
            if i < (len(peaks_neg_fin) - 1):
                if diff_peaks[i] <= cut_off:
                    forest.append(peaks_neg_fin[i + 1])
                if diff_peaks[i] > cut_off:
                    # print(forest[np.argmin(z[forest]) ] )
                    if not isNaN(forest[np.argmin(z[forest])]):
                        peaks_neg_true.append(forest[np.argmin(z[forest])])
                    forest = []
                    forest.append(peaks_neg_fin[i + 1])
            if i == (len(peaks_neg_fin) - 1):
                # print(print(forest[np.argmin(z[forest]) ] ))
                if not isNaN(forest[np.argmin(z[forest])]):
                    peaks_neg_true.append(forest[np.argmin(z[forest])])

        num_col = (len(peaks_neg_true)) + 1
        p_l = 0
        p_u = len(y) - 1
        p_m = int(len(y) / 2.0)
        p_quarter = int(len(y) / 5.0)
        p_g_l = int(len(y) / 4.0)
        p_g_u = len(y) - int(len(y) / 4.0)

        p_u_quarter = len(y) - p_quarter

        ##print(num_col,'early')
        if num_col == 3:
            if (peaks_neg_true[0] > p_g_u and peaks_neg_true[1] > p_g_u) or (peaks_neg_true[0] < p_g_l and peaks_neg_true[1] < p_g_l) or (peaks_neg_true[0] < p_m and (peaks_neg_true[1] + 200) < p_m) or ((peaks_neg_true[0] - 200) > p_m and peaks_neg_true[1] > p_m):
                num_col = 1
                peaks_neg_true = []
            elif (peaks_neg_true[0] < p_g_u and peaks_neg_true[0] > p_g_l) and (peaks_neg_true[1] > p_u_quarter):
                peaks_neg_true = [peaks_neg_true[0]]
            elif (peaks_neg_true[1] < p_g_u and peaks_neg_true[1] > p_g_l) and (peaks_neg_true[0] < p_quarter):
                peaks_neg_true = [peaks_neg_true[1]]
            else:
                pass

        if num_col == 2:
            if (peaks_neg_true[0] > p_g_u) or (peaks_neg_true[0] < p_g_l):
                num_col = 1
                peaks_neg_true = []
            else:
                pass

        diff_peaks_annormal = diff_peaks[diff_peaks < 360]

        if len(diff_peaks_annormal) > 0:
            arg_help = np.array(range(len(diff_peaks)))
            arg_help_ann = arg_help[diff_peaks < 360]

            peaks_neg_fin_new = []

            for ii in range(len(peaks_neg_fin)):
                if ii in arg_help_ann:
                    arg_min = np.argmin([interest_neg_fin[ii], interest_neg_fin[ii + 1]])
                    if arg_min == 0:
                        peaks_neg_fin_new.append(peaks_neg_fin[ii])
                    else:
                        peaks_neg_fin_new.append(peaks_neg_fin[ii + 1])

                elif (ii - 1) in arg_help_ann:
                    pass
                else:
                    peaks_neg_fin_new.append(peaks_neg_fin[ii])
        else:
            peaks_neg_fin_new = peaks_neg_fin

        # plt.plot(gaussian_filter1d(y, sigma_))
        # plt.plot(peaks_neg_true,z[peaks_neg_true],'*')
        # plt.plot([0,len(y)], [grenze,grenze])
        # plt.show()

        ##print(len(peaks_neg_true))
        return len(peaks_neg_true), peaks_neg_true


    def return_points_with_boundies(self, peaks_neg_fin, first_point, last_point):
        peaks_neg_tot = []
        peaks_neg_tot.append(first_point)
        for ii in range(len(peaks_neg_fin)):
            peaks_neg_tot.append(peaks_neg_fin[ii])
        peaks_neg_tot.append(last_point)
        return peaks_neg_tot

    def contours_in_same_horizon(self, cy_main_hor):
        X1 = np.zeros((len(cy_main_hor), len(cy_main_hor)))
        X2 = np.zeros((len(cy_main_hor), len(cy_main_hor)))

        X1[0::1, :] = cy_main_hor[:]
        X2 = X1.T

        X_dif = np.abs(X2 - X1)
        args_help = np.array(range(len(cy_main_hor)))
        all_args = []
        for i in range(len(cy_main_hor)):
            list_h = list(args_help[X_dif[i, :] <= 20])
            list_h.append(i)
            if len(list_h) > 1:
                all_args.append(list(set(list_h)))
        return np.unique(all_args)

    def return_boxes_of_images_by_order_of_reading_without_seperators(self, spliter_y_new, image_p_rev, regions_without_seperators, matrix_of_lines_ch, seperators_closeup_n):

        boxes = []

        # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
        # holes in the text and also finding spliter which covers more than one columns.
        for i in range(len(spliter_y_new) - 1):
            # print(spliter_y_new[i],spliter_y_new[i+1])
            matrix_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 6] > spliter_y_new[i]) & (matrix_of_lines_ch[:, 7] < spliter_y_new[i + 1])]
            # print(len( matrix_new[:,9][matrix_new[:,9]==1] ))

            # print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')

            # check to see is there any vertical seperator to find holes.
            if np.abs(spliter_y_new[i + 1] - spliter_y_new[i]) > 1.0 / 3.0 * regions_without_seperators.shape[0]:  # len( matrix_new[:,9][matrix_new[:,9]==1] )>0 and np.max(matrix_new[:,8][matrix_new[:,9]==1])>=0.1*(np.abs(spliter_y_new[i+1]-spliter_y_new[i] )):

                # org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
                # org_img_dichte=org_img_dichte-np.min(org_img_dichte)
                ##plt.figure(figsize=(20,20))
                ##plt.plot(org_img_dichte)
                ##plt.show()
                ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)

                num_col, peaks_neg_fin = self.find_num_col_only_image(image_p_rev[int(spliter_y_new[i]) : int(spliter_y_new[i + 1]), :], multiplier=2.4)

                # num_col, peaks_neg_fin=find_num_col(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=7.0)
                x_min_hor_some = matrix_new[:, 2][(matrix_new[:, 9] == 0)]
                x_max_hor_some = matrix_new[:, 3][(matrix_new[:, 9] == 0)]
                cy_hor_some = matrix_new[:, 5][(matrix_new[:, 9] == 0)]
                arg_org_hor_some = matrix_new[:, 0][(matrix_new[:, 9] == 0)]

                peaks_neg_tot = self.return_points_with_boundies(peaks_neg_fin, 0, seperators_closeup_n[:, :, 0].shape[1])

                start_index_of_hor, newest_peaks, arg_min_hor_sort, lines_length_dels, lines_indexes_deleted = return_hor_spliter_by_index_for_without_verticals(peaks_neg_tot, x_min_hor_some, x_max_hor_some)

                arg_org_hor_some_sort = arg_org_hor_some[arg_min_hor_sort]

                start_index_of_hor_with_subset = [start_index_of_hor[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]  # start_index_of_hor[lines_length_dels>0]
                arg_min_hor_sort_with_subset = [arg_min_hor_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]
                lines_indexes_deleted_with_subset = [lines_indexes_deleted[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]
                lines_length_dels_with_subset = [lines_length_dels[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]

                arg_org_hor_some_sort_subset = [arg_org_hor_some_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]

                # arg_min_hor_sort_with_subset=arg_min_hor_sort[lines_length_dels>0]
                # lines_indexes_deleted_with_subset=lines_indexes_deleted[lines_length_dels>0]
                # lines_length_dels_with_subset=lines_length_dels[lines_length_dels>0]

                # print(len(arg_min_hor_sort),len(arg_org_hor_some_sort),'vizzzzzz')

                vahid_subset = np.zeros((len(start_index_of_hor_with_subset), len(start_index_of_hor_with_subset))) - 1
                for kkk1 in range(len(start_index_of_hor_with_subset)):

                    # print(lines_indexes_deleted,'hiii')
                    index_del_sub = np.unique(lines_indexes_deleted_with_subset[kkk1])

                    for kkk2 in range(len(start_index_of_hor_with_subset)):

                        if set(lines_indexes_deleted_with_subset[kkk2][0]) < set(lines_indexes_deleted_with_subset[kkk1][0]):
                            vahid_subset[kkk1, kkk2] = kkk1
                        else:
                            pass
                    # print(set(lines_indexes_deleted[kkk2][0]), set(lines_indexes_deleted[kkk1][0]))

                # check the len of matrix if it has no length means that there is no spliter at all

                if len(vahid_subset > 0):
                    # print('hihoo')

                    # find parenets args
                    line_int = np.zeros(vahid_subset.shape[0])

                    childs_id = []
                    arg_child = []
                    for li in range(vahid_subset.shape[0]):
                        if np.all(vahid_subset[:, li] == -1):
                            line_int[li] = -1
                        else:
                            line_int[li] = 1

                            # childs_args_in=[ idd for idd in range(vahid_subset.shape[0]) if vahid_subset[idd,li]!=-1]
                            # helpi=[]
                            # for nad in range(len(childs_args_in)):
                            #    helpi.append(arg_min_hor_sort_with_subset[childs_args_in[nad]])

                            arg_child.append(arg_min_hor_sort_with_subset[li])

                    arg_parent = [arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] == -1]
                    start_index_of_hor_parent = [start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] == -1]
                    # arg_parent=[lines_indexes_deleted_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    # arg_parent=[lines_length_dels_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]

                    # arg_child=[arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]!=-1]
                    start_index_of_hor_child = [start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] != -1]

                    cy_hor_some_sort = cy_hor_some[arg_parent]

                    newest_y_spliter_tot = []

                    for tj in range(len(newest_peaks) - 1):
                        newest_y_spliter = []
                        newest_y_spliter.append(spliter_y_new[i])
                        if tj in np.unique(start_index_of_hor_parent):
                            cy_help = np.array(cy_hor_some_sort)[np.array(start_index_of_hor_parent) == tj]
                            cy_help_sort = np.sort(cy_help)

                            # print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                            for mj in range(len(cy_help_sort)):
                                newest_y_spliter.append(cy_help_sort[mj])
                        newest_y_spliter.append(spliter_y_new[i + 1])

                        newest_y_spliter_tot.append(newest_y_spliter)

                else:
                    line_int = []
                    newest_y_spliter_tot = []

                    for tj in range(len(newest_peaks) - 1):
                        newest_y_spliter = []
                        newest_y_spliter.append(spliter_y_new[i])

                        newest_y_spliter.append(spliter_y_new[i + 1])

                        newest_y_spliter_tot.append(newest_y_spliter)

                # if line_int is all -1 means that big spliters have no child and we can easily go through
                if np.all(np.array(line_int) == -1):
                    for j in range(len(newest_peaks) - 1):
                        newest_y_spliter = newest_y_spliter_tot[j]

                        for n in range(len(newest_y_spliter) - 1):
                            # print(j,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'maaaa')
                            ##plt.imshow(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]])
                            ##plt.show()

                            # print(matrix_new[:,0][ (matrix_new[:,9]==1 )])
                            for jvt in matrix_new[:, 0][(matrix_new[:, 9] == 1) & (matrix_new[:, 6] > newest_y_spliter[n]) & (matrix_new[:, 7] < newest_y_spliter[n + 1]) & ((matrix_new[:, 1]) < newest_peaks[j + 1]) & ((matrix_new[:, 1]) > newest_peaks[j])]:
                                pass

                                ###plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                            # print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                            matrix_new_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter[n]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter[n + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < newest_peaks[j + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > newest_peaks[j])]
                            # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                            if 1 > 0:  # len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                # num_col_sub, peaks_neg_fin_sub=find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=2.3)
                                num_col_sub, peaks_neg_fin_sub = self.find_num_col_only_image(image_p_rev[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=2.4)
                            else:
                                peaks_neg_fin_sub = []

                            peaks_sub = []
                            peaks_sub.append(newest_peaks[j])

                            for kj in range(len(peaks_neg_fin_sub)):
                                peaks_sub.append(peaks_neg_fin_sub[kj] + newest_peaks[j])

                            peaks_sub.append(newest_peaks[j + 1])

                            # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                            for kh in range(len(peaks_sub) - 1):
                                boxes.append([peaks_sub[kh], peaks_sub[kh + 1], newest_y_spliter[n], newest_y_spliter[n + 1]])

                else:
                    for j in range(len(newest_peaks) - 1):
                        newest_y_spliter = newest_y_spliter_tot[j]

                        if j in start_index_of_hor_parent:

                            x_min_ch = x_min_hor_some[arg_child]
                            x_max_ch = x_max_hor_some[arg_child]
                            cy_hor_some_sort_child = cy_hor_some[arg_child]
                            cy_hor_some_sort_child = np.sort(cy_hor_some_sort_child)

                            for n in range(len(newest_y_spliter) - 1):

                                cy_child_in = cy_hor_some_sort_child[(cy_hor_some_sort_child > newest_y_spliter[n]) & (cy_hor_some_sort_child < newest_y_spliter[n + 1])]

                                if len(cy_child_in) > 0:
                                    ###num_col_ch, peaks_neg_ch=find_num_col( regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=2.3)

                                    num_col_ch, peaks_neg_ch = self.find_num_col_only_image(image_p_rev[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=2.3)

                                    peaks_neg_ch = peaks_neg_ch[:] + newest_peaks[j]

                                    peaks_neg_ch_tot = self.return_points_with_boundies(peaks_neg_ch, newest_peaks[j], newest_peaks[j + 1])

                                    ss_in_ch, nst_p_ch, arg_n_ch, lines_l_del_ch, lines_in_del_ch = return_hor_spliter_by_index_for_without_verticals(peaks_neg_ch_tot, x_min_ch, x_max_ch)

                                    newest_y_spliter_ch_tot = []

                                    for tjj in range(len(nst_p_ch) - 1):
                                        newest_y_spliter_new = []
                                        newest_y_spliter_new.append(newest_y_spliter[n])
                                        if tjj in np.unique(ss_in_ch):

                                            # print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                                            for mjj in range(len(cy_child_in)):
                                                newest_y_spliter_new.append(cy_child_in[mjj])
                                        newest_y_spliter_new.append(newest_y_spliter[n + 1])

                                        newest_y_spliter_ch_tot.append(newest_y_spliter_new)

                                    for jn in range(len(nst_p_ch) - 1):
                                        newest_y_spliter_h = newest_y_spliter_ch_tot[jn]

                                        for nd in range(len(newest_y_spliter_h) - 1):

                                            matrix_new_new2 = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter_h[nd]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter_h[nd + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < nst_p_ch[jn + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > nst_p_ch[jn])]
                                            # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                            if 1 > 0:  # len( matrix_new_new2[:,9][matrix_new_new2[:,9]==1] )>0 and np.max(matrix_new_new2[:,8][matrix_new_new2[:,9]==1])>=0.2*(np.abs(newest_y_spliter_h[nd+1]-newest_y_spliter_h[nd] )):
                                                # num_col_sub_ch, peaks_neg_fin_sub_ch=find_num_col(regions_without_seperators[int(newest_y_spliter_h[nd]):int(newest_y_spliter_h[nd+1]),nst_p_ch[jn]:nst_p_ch[jn+1]],multiplier=2.3)

                                                num_col_sub_ch, peaks_neg_fin_sub_ch = self.find_num_col_only_image(image_p_rev[int(newest_y_spliter_h[nd]) : int(newest_y_spliter_h[nd + 1]), nst_p_ch[jn] : nst_p_ch[jn + 1]], multiplier=2.3)
                                                # print(peaks_neg_fin_sub_ch,'gada kutullllllll')
                                            else:
                                                peaks_neg_fin_sub_ch = []

                                            peaks_sub_ch = []
                                            peaks_sub_ch.append(nst_p_ch[jn])

                                            for kjj in range(len(peaks_neg_fin_sub_ch)):
                                                peaks_sub_ch.append(peaks_neg_fin_sub_ch[kjj] + nst_p_ch[jn])

                                            peaks_sub_ch.append(nst_p_ch[jn + 1])

                                            # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                            for khh in range(len(peaks_sub_ch) - 1):
                                                boxes.append([peaks_sub_ch[khh], peaks_sub_ch[khh + 1], newest_y_spliter_h[nd], newest_y_spliter_h[nd + 1]])

                                else:

                                    matrix_new_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter[n]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter[n + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < newest_peaks[j + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > newest_peaks[j])]
                                    # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                    if 1 > 0:  # len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                        ###num_col_sub, peaks_neg_fin_sub=find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=2.3)
                                        num_col_sub, peaks_neg_fin_sub = self.find_num_col_only_image(image_p_rev[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=2.3)
                                    else:
                                        peaks_neg_fin_sub = []

                                    peaks_sub = []
                                    peaks_sub.append(newest_peaks[j])

                                    for kj in range(len(peaks_neg_fin_sub)):
                                        peaks_sub.append(peaks_neg_fin_sub[kj] + newest_peaks[j])

                                    peaks_sub.append(newest_peaks[j + 1])

                                    # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                    for kh in range(len(peaks_sub) - 1):
                                        boxes.append([peaks_sub[kh], peaks_sub[kh + 1], newest_y_spliter[n], newest_y_spliter[n + 1]])

                        else:
                            for n in range(len(newest_y_spliter) - 1):

                                for jvt in matrix_new[:, 0][(matrix_new[:, 9] == 1) & (matrix_new[:, 6] > newest_y_spliter[n]) & (matrix_new[:, 7] < newest_y_spliter[n + 1]) & ((matrix_new[:, 1]) < newest_peaks[j + 1]) & ((matrix_new[:, 1]) > newest_peaks[j])]:
                                    pass

                                    # plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                                # print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                                matrix_new_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter[n]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter[n + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < newest_peaks[j + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > newest_peaks[j])]
                                # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                if 1 > 0:  # len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                    ###num_col_sub, peaks_neg_fin_sub=find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=5.0)
                                    num_col_sub, peaks_neg_fin_sub = self.find_num_col_only_image(image_p_rev[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=2.3)
                                else:
                                    peaks_neg_fin_sub = []

                                peaks_sub = []
                                peaks_sub.append(newest_peaks[j])

                                for kj in range(len(peaks_neg_fin_sub)):
                                    peaks_sub.append(peaks_neg_fin_sub[kj] + newest_peaks[j])

                                peaks_sub.append(newest_peaks[j + 1])

                                # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                for kh in range(len(peaks_sub) - 1):
                                    boxes.append([peaks_sub[kh], peaks_sub[kh + 1], newest_y_spliter[n], newest_y_spliter[n + 1]])

            else:
                boxes.append([0, seperators_closeup_n[:, :, 0].shape[1], spliter_y_new[i], spliter_y_new[i + 1]])
        return boxes

    def return_boxes_of_images_by_order_of_reading_without_seperators_2cols(self, spliter_y_new, image_p_rev, regions_without_seperators, matrix_of_lines_ch, seperators_closeup_n):

        boxes = []

        # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
        # holes in the text and also finding spliter which covers more than one columns.
        for i in range(len(spliter_y_new) - 1):
            # print(spliter_y_new[i],spliter_y_new[i+1])
            matrix_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 6] > spliter_y_new[i]) & (matrix_of_lines_ch[:, 7] < spliter_y_new[i + 1])]
            # print(len( matrix_new[:,9][matrix_new[:,9]==1] ))

            # print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')

            # check to see is there any vertical seperator to find holes.
            if np.abs(spliter_y_new[i + 1] - spliter_y_new[i]) > 1.0 / 3.0 * regions_without_seperators.shape[0]:  # len( matrix_new[:,9][matrix_new[:,9]==1] )>0 and np.max(matrix_new[:,8][matrix_new[:,9]==1])>=0.1*(np.abs(spliter_y_new[i+1]-spliter_y_new[i] )):

                # org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
                # org_img_dichte=org_img_dichte-np.min(org_img_dichte)
                ##plt.figure(figsize=(20,20))
                ##plt.plot(org_img_dichte)
                ##plt.show()
                ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)

                try:
                    num_col, peaks_neg_fin = self.find_num_col_only_image(image_p_rev[int(spliter_y_new[i]) : int(spliter_y_new[i + 1]), :], multiplier=2.4)
                except:
                    peaks_neg_fin = []
                    num_col = 0

                peaks_neg_tot = self.return_points_with_boundies(peaks_neg_fin, 0, seperators_closeup_n[:, :, 0].shape[1])

                for kh in range(len(peaks_neg_tot) - 1):
                    boxes.append([peaks_neg_tot[kh], peaks_neg_tot[kh + 1], spliter_y_new[i], spliter_y_new[i + 1]])
            else:
                boxes.append([0, seperators_closeup_n[:, :, 0].shape[1], spliter_y_new[i], spliter_y_new[i + 1]])

        return boxes

    def combine_hor_lines_and_delete_cross_points_and_get_lines_features_back(self, regions_pre_p):
        seperators_closeup = ((regions_pre_p[:, :] == 6)) * 1

        seperators_closeup = seperators_closeup.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)

        seperators_closeup = cv2.dilate(seperators_closeup, kernel, iterations=1)
        seperators_closeup = cv2.erode(seperators_closeup, kernel, iterations=1)

        seperators_closeup = cv2.erode(seperators_closeup, kernel, iterations=1)
        seperators_closeup = cv2.dilate(seperators_closeup, kernel, iterations=1)

        if len(seperators_closeup.shape) == 2:
            seperators_closeup_n = np.zeros((seperators_closeup.shape[0], seperators_closeup.shape[1], 3))
            seperators_closeup_n[:, :, 0] = seperators_closeup
            seperators_closeup_n[:, :, 1] = seperators_closeup
            seperators_closeup_n[:, :, 2] = seperators_closeup
        else:
            seperators_closeup_n = seperators_closeup[:, :, :]
        # seperators_closeup=seperators_closeup.astype(np.uint8)
        seperators_closeup_n = seperators_closeup_n.astype(np.uint8)
        imgray = cv2.cvtColor(seperators_closeup_n, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_lines, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        slope_lines, dist_x, x_min_main, x_max_main, cy_main, slope_lines_org, y_min_main, y_max_main, cx_main = find_features_of_lines(contours_lines)

        dist_y = np.abs(y_max_main - y_min_main)

        slope_lines_org_hor = slope_lines_org[slope_lines == 0]
        args = np.array(range(len(slope_lines)))
        len_x = seperators_closeup.shape[1] * 0
        len_y = seperators_closeup.shape[0] * 0.01

        args_hor = args[slope_lines == 0]
        dist_x_hor = dist_x[slope_lines == 0]
        dist_y_hor = dist_y[slope_lines == 0]
        x_min_main_hor = x_min_main[slope_lines == 0]
        x_max_main_hor = x_max_main[slope_lines == 0]
        cy_main_hor = cy_main[slope_lines == 0]
        y_min_main_hor = y_min_main[slope_lines == 0]
        y_max_main_hor = y_max_main[slope_lines == 0]

        args_hor = args_hor[dist_x_hor >= len_x]
        x_max_main_hor = x_max_main_hor[dist_x_hor >= len_x]
        x_min_main_hor = x_min_main_hor[dist_x_hor >= len_x]
        cy_main_hor = cy_main_hor[dist_x_hor >= len_x]
        y_min_main_hor = y_min_main_hor[dist_x_hor >= len_x]
        y_max_main_hor = y_max_main_hor[dist_x_hor >= len_x]
        slope_lines_org_hor = slope_lines_org_hor[dist_x_hor >= len_x]
        dist_y_hor = dist_y_hor[dist_x_hor >= len_x]
        dist_x_hor = dist_x_hor[dist_x_hor >= len_x]

        args_ver = args[slope_lines == 1]
        dist_y_ver = dist_y[slope_lines == 1]
        dist_x_ver = dist_x[slope_lines == 1]
        x_min_main_ver = x_min_main[slope_lines == 1]
        x_max_main_ver = x_max_main[slope_lines == 1]
        y_min_main_ver = y_min_main[slope_lines == 1]
        y_max_main_ver = y_max_main[slope_lines == 1]
        cx_main_ver = cx_main[slope_lines == 1]

        args_ver = args_ver[dist_y_ver >= len_y]
        x_max_main_ver = x_max_main_ver[dist_y_ver >= len_y]
        x_min_main_ver = x_min_main_ver[dist_y_ver >= len_y]
        cx_main_ver = cx_main_ver[dist_y_ver >= len_y]
        y_min_main_ver = y_min_main_ver[dist_y_ver >= len_y]
        y_max_main_ver = y_max_main_ver[dist_y_ver >= len_y]
        dist_x_ver = dist_x_ver[dist_y_ver >= len_y]
        dist_y_ver = dist_y_ver[dist_y_ver >= len_y]

        img_p_in_ver = np.zeros(seperators_closeup_n[:, :, 2].shape)
        for jv in range(len(args_ver)):
            img_p_in_ver = cv2.fillPoly(img_p_in_ver, pts=[contours_lines[args_ver[jv]]], color=(1, 1, 1))

        img_in_hor = np.zeros(seperators_closeup_n[:, :, 2].shape)
        for jv in range(len(args_hor)):
            img_p_in_hor = cv2.fillPoly(img_in_hor, pts=[contours_lines[args_hor[jv]]], color=(1, 1, 1))

        all_args_uniq = self.contours_in_same_horizon(cy_main_hor)
        # print(all_args_uniq,'all_args_uniq')
        if len(all_args_uniq) > 0:
            if type(all_args_uniq[0]) is list:
                contours_new = []
                for dd in range(len(all_args_uniq)):
                    merged_all = None
                    some_args = args_hor[all_args_uniq[dd]]
                    some_cy = cy_main_hor[all_args_uniq[dd]]
                    some_x_min = x_min_main_hor[all_args_uniq[dd]]
                    some_x_max = x_max_main_hor[all_args_uniq[dd]]

                    img_in = np.zeros(seperators_closeup_n[:, :, 2].shape)
                    for jv in range(len(some_args)):

                        img_p_in = cv2.fillPoly(img_p_in_hor, pts=[contours_lines[some_args[jv]]], color=(1, 1, 1))
                        img_p_in[int(np.mean(some_cy)) - 5 : int(np.mean(some_cy)) + 5, int(np.min(some_x_min)) : int(np.max(some_x_max))] = 1

            else:
                img_p_in = seperators_closeup
        else:
            img_p_in = seperators_closeup

        sep_ver_hor = img_p_in + img_p_in_ver
        sep_ver_hor_cross = (sep_ver_hor == 2) * 1

        sep_ver_hor_cross = np.repeat(sep_ver_hor_cross[:, :, np.newaxis], 3, axis=2)
        sep_ver_hor_cross = sep_ver_hor_cross.astype(np.uint8)
        imgray = cv2.cvtColor(sep_ver_hor_cross, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_cross, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cx_cross, cy_cross, _, _, _, _, _ = find_new_features_of_contoures(contours_cross)

        for ii in range(len(cx_cross)):
            sep_ver_hor[int(cy_cross[ii]) - 15 : int(cy_cross[ii]) + 15, int(cx_cross[ii]) + 5 : int(cx_cross[ii]) + 40] = 0
            sep_ver_hor[int(cy_cross[ii]) - 15 : int(cy_cross[ii]) + 15, int(cx_cross[ii]) - 40 : int(cx_cross[ii]) - 4] = 0

        img_p_in[:, :] = sep_ver_hor[:, :]

        if len(img_p_in.shape) == 2:
            seperators_closeup_n = np.zeros((img_p_in.shape[0], img_p_in.shape[1], 3))
            seperators_closeup_n[:, :, 0] = img_p_in
            seperators_closeup_n[:, :, 1] = img_p_in
            seperators_closeup_n[:, :, 2] = img_p_in
        else:
            seperators_closeup_n = img_p_in[:, :, :]
        # seperators_closeup=seperators_closeup.astype(np.uint8)
        seperators_closeup_n = seperators_closeup_n.astype(np.uint8)
        imgray = cv2.cvtColor(seperators_closeup_n, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours_lines, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        slope_lines, dist_x, x_min_main, x_max_main, cy_main, slope_lines_org, y_min_main, y_max_main, cx_main = find_features_of_lines(contours_lines)

        dist_y = np.abs(y_max_main - y_min_main)

        slope_lines_org_hor = slope_lines_org[slope_lines == 0]
        args = np.array(range(len(slope_lines)))
        len_x = seperators_closeup.shape[1] * 0.04
        len_y = seperators_closeup.shape[0] * 0.08

        args_hor = args[slope_lines == 0]
        dist_x_hor = dist_x[slope_lines == 0]
        dist_y_hor = dist_y[slope_lines == 0]
        x_min_main_hor = x_min_main[slope_lines == 0]
        x_max_main_hor = x_max_main[slope_lines == 0]
        cy_main_hor = cy_main[slope_lines == 0]
        y_min_main_hor = y_min_main[slope_lines == 0]
        y_max_main_hor = y_max_main[slope_lines == 0]

        args_hor = args_hor[dist_x_hor >= len_x]
        x_max_main_hor = x_max_main_hor[dist_x_hor >= len_x]
        x_min_main_hor = x_min_main_hor[dist_x_hor >= len_x]
        cy_main_hor = cy_main_hor[dist_x_hor >= len_x]
        y_min_main_hor = y_min_main_hor[dist_x_hor >= len_x]
        y_max_main_hor = y_max_main_hor[dist_x_hor >= len_x]
        slope_lines_org_hor = slope_lines_org_hor[dist_x_hor >= len_x]
        dist_y_hor = dist_y_hor[dist_x_hor >= len_x]
        dist_x_hor = dist_x_hor[dist_x_hor >= len_x]

        args_ver = args[slope_lines == 1]
        dist_y_ver = dist_y[slope_lines == 1]
        dist_x_ver = dist_x[slope_lines == 1]
        x_min_main_ver = x_min_main[slope_lines == 1]
        x_max_main_ver = x_max_main[slope_lines == 1]
        y_min_main_ver = y_min_main[slope_lines == 1]
        y_max_main_ver = y_max_main[slope_lines == 1]
        cx_main_ver = cx_main[slope_lines == 1]

        args_ver = args_ver[dist_y_ver >= len_y]
        x_max_main_ver = x_max_main_ver[dist_y_ver >= len_y]
        x_min_main_ver = x_min_main_ver[dist_y_ver >= len_y]
        cx_main_ver = cx_main_ver[dist_y_ver >= len_y]
        y_min_main_ver = y_min_main_ver[dist_y_ver >= len_y]
        y_max_main_ver = y_max_main_ver[dist_y_ver >= len_y]
        dist_x_ver = dist_x_ver[dist_y_ver >= len_y]
        dist_y_ver = dist_y_ver[dist_y_ver >= len_y]

        matrix_of_lines_ch = np.zeros((len(cy_main_hor) + len(cx_main_ver), 10))

        matrix_of_lines_ch[: len(cy_main_hor), 0] = args_hor
        matrix_of_lines_ch[len(cy_main_hor) :, 0] = args_ver

        matrix_of_lines_ch[len(cy_main_hor) :, 1] = cx_main_ver

        matrix_of_lines_ch[: len(cy_main_hor), 2] = x_min_main_hor
        matrix_of_lines_ch[len(cy_main_hor) :, 2] = x_min_main_ver

        matrix_of_lines_ch[: len(cy_main_hor), 3] = x_max_main_hor
        matrix_of_lines_ch[len(cy_main_hor) :, 3] = x_max_main_ver

        matrix_of_lines_ch[: len(cy_main_hor), 4] = dist_x_hor
        matrix_of_lines_ch[len(cy_main_hor) :, 4] = dist_x_ver

        matrix_of_lines_ch[: len(cy_main_hor), 5] = cy_main_hor

        matrix_of_lines_ch[: len(cy_main_hor), 6] = y_min_main_hor
        matrix_of_lines_ch[len(cy_main_hor) :, 6] = y_min_main_ver

        matrix_of_lines_ch[: len(cy_main_hor), 7] = y_max_main_hor
        matrix_of_lines_ch[len(cy_main_hor) :, 7] = y_max_main_ver

        matrix_of_lines_ch[: len(cy_main_hor), 8] = dist_y_hor
        matrix_of_lines_ch[len(cy_main_hor) :, 8] = dist_y_ver

        matrix_of_lines_ch[len(cy_main_hor) :, 9] = 1

        return matrix_of_lines_ch, seperators_closeup_n

    def combine_hor_lines_and_delete_cross_points_and_get_lines_features_back_new(self, img_p_in_ver, img_in_hor):

        # plt.imshow(img_in_hor)
        # plt.show()

        # img_p_in_ver = cv2.erode(img_p_in_ver, self.kernel, iterations=2)
        img_p_in_ver = img_p_in_ver.astype(np.uint8)
        img_p_in_ver = np.repeat(img_p_in_ver[:, :, np.newaxis], 3, axis=2)
        imgray = cv2.cvtColor(img_p_in_ver, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours_lines_ver, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        slope_lines_ver, dist_x_ver, x_min_main_ver, x_max_main_ver, cy_main_ver, slope_lines_org_ver, y_min_main_ver, y_max_main_ver, cx_main_ver = find_features_of_lines(contours_lines_ver)

        for i in range(len(x_min_main_ver)):
            img_p_in_ver[int(y_min_main_ver[i]) : int(y_min_main_ver[i]) + 30, int(cx_main_ver[i]) - 25 : int(cx_main_ver[i]) + 25, 0] = 0
            img_p_in_ver[int(y_max_main_ver[i]) - 30 : int(y_max_main_ver[i]), int(cx_main_ver[i]) - 25 : int(cx_main_ver[i]) + 25, 0] = 0

        # plt.imshow(img_p_in_ver[:,:,0])
        # plt.show()
        img_in_hor = img_in_hor.astype(np.uint8)
        img_in_hor = np.repeat(img_in_hor[:, :, np.newaxis], 3, axis=2)
        imgray = cv2.cvtColor(img_in_hor, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours_lines_hor, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        slope_lines_hor, dist_x_hor, x_min_main_hor, x_max_main_hor, cy_main_hor, slope_lines_org_hor, y_min_main_hor, y_max_main_hor, cx_main_hor = find_features_of_lines(contours_lines_hor)

        args_hor = np.array(range(len(slope_lines_hor)))
        all_args_uniq = self.contours_in_same_horizon(cy_main_hor)
        # print(all_args_uniq,'all_args_uniq')
        if len(all_args_uniq) > 0:
            if type(all_args_uniq[0]) is list:
                special_seperators = []
                contours_new = []
                for dd in range(len(all_args_uniq)):
                    merged_all = None
                    some_args = args_hor[all_args_uniq[dd]]
                    some_cy = cy_main_hor[all_args_uniq[dd]]
                    some_x_min = x_min_main_hor[all_args_uniq[dd]]
                    some_x_max = x_max_main_hor[all_args_uniq[dd]]

                    # img_in=np.zeros(seperators_closeup_n[:,:,2].shape)
                    for jv in range(len(some_args)):

                        img_p_in = cv2.fillPoly(img_in_hor, pts=[contours_lines_hor[some_args[jv]]], color=(1, 1, 1))
                        img_p_in[int(np.mean(some_cy)) - 5 : int(np.mean(some_cy)) + 5, int(np.min(some_x_min)) : int(np.max(some_x_max))] = 1

                    sum_dis = dist_x_hor[some_args].sum()
                    diff_max_min_uniques = np.max(x_max_main_hor[some_args]) - np.min(x_min_main_hor[some_args])

                    # print( sum_dis/float(diff_max_min_uniques) ,diff_max_min_uniques/float(img_p_in_ver.shape[1]),dist_x_hor[some_args].sum(),diff_max_min_uniques,np.mean( dist_x_hor[some_args]),np.std( dist_x_hor[some_args]) )

                    if diff_max_min_uniques > sum_dis and ((sum_dis / float(diff_max_min_uniques)) > 0.85) and ((diff_max_min_uniques / float(img_p_in_ver.shape[1])) > 0.85) and np.std(dist_x_hor[some_args]) < (0.55 * np.mean(dist_x_hor[some_args])):
                        # print(dist_x_hor[some_args],dist_x_hor[some_args].sum(),np.min(x_min_main_hor[some_args]) ,np.max(x_max_main_hor[some_args]),'jalibdi')
                        # print(np.mean( dist_x_hor[some_args] ),np.std( dist_x_hor[some_args] ),np.var( dist_x_hor[some_args] ),'jalibdiha')
                        special_seperators.append(np.mean(cy_main_hor[some_args]))

            else:
                img_p_in = img_in_hor
                special_seperators = []
        else:
            img_p_in = img_in_hor
            special_seperators = []

        img_p_in_ver[:, :, 0][img_p_in_ver[:, :, 0] == 255] = 1
        # print(img_p_in_ver.shape,np.unique(img_p_in_ver[:,:,0]))

        # plt.imshow(img_p_in[:,:,0])
        # plt.show()

        # plt.imshow(img_p_in_ver[:,:,0])
        # plt.show()
        sep_ver_hor = img_p_in + img_p_in_ver
        # print(sep_ver_hor.shape,np.unique(sep_ver_hor[:,:,0]),'sep_ver_horsep_ver_horsep_ver_hor')
        # plt.imshow(sep_ver_hor[:,:,0])
        # plt.show()

        sep_ver_hor_cross = (sep_ver_hor[:, :, 0] == 2) * 1

        sep_ver_hor_cross = np.repeat(sep_ver_hor_cross[:, :, np.newaxis], 3, axis=2)
        sep_ver_hor_cross = sep_ver_hor_cross.astype(np.uint8)
        imgray = cv2.cvtColor(sep_ver_hor_cross, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_cross, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cx_cross, cy_cross, _, _, _, _, _ = find_new_features_of_contoures(contours_cross)

        for ii in range(len(cx_cross)):
            img_p_in[int(cy_cross[ii]) - 30 : int(cy_cross[ii]) + 30, int(cx_cross[ii]) + 5 : int(cx_cross[ii]) + 40, 0] = 0
            img_p_in[int(cy_cross[ii]) - 30 : int(cy_cross[ii]) + 30, int(cx_cross[ii]) - 40 : int(cx_cross[ii]) - 4, 0] = 0

        # plt.imshow(img_p_in[:,:,0])
        # plt.show()

        return img_p_in[:, :, 0], special_seperators

    def return_boxes_of_images_by_order_of_reading(self, spliter_y_new, regions_without_seperators, matrix_of_lines_ch, seperators_closeup_n):
        boxes = []

        # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
        # holes in the text and also finding spliter which covers more than one columns.
        for i in range(len(spliter_y_new) - 1):
            # print(spliter_y_new[i],spliter_y_new[i+1])
            matrix_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 6] > spliter_y_new[i]) & (matrix_of_lines_ch[:, 7] < spliter_y_new[i + 1])]
            # print(len( matrix_new[:,9][matrix_new[:,9]==1] ))

            # print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')

            # check to see is there any vertical seperator to find holes.
            if len(matrix_new[:, 9][matrix_new[:, 9] == 1]) > 0 and np.max(matrix_new[:, 8][matrix_new[:, 9] == 1]) >= 0.1 * (np.abs(spliter_y_new[i + 1] - spliter_y_new[i])):

                # org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
                # org_img_dichte=org_img_dichte-np.min(org_img_dichte)
                ##plt.figure(figsize=(20,20))
                ##plt.plot(org_img_dichte)
                ##plt.show()
                ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)

                num_col, peaks_neg_fin = self.find_num_col(regions_without_seperators[int(spliter_y_new[i]) : int(spliter_y_new[i + 1]), :], multiplier=7.0)

                # num_col, peaks_neg_fin=find_num_col(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=7.0)
                x_min_hor_some = matrix_new[:, 2][(matrix_new[:, 9] == 0)]
                x_max_hor_some = matrix_new[:, 3][(matrix_new[:, 9] == 0)]
                cy_hor_some = matrix_new[:, 5][(matrix_new[:, 9] == 0)]
                arg_org_hor_some = matrix_new[:, 0][(matrix_new[:, 9] == 0)]

                peaks_neg_tot = self.return_points_with_boundies(peaks_neg_fin, 0, seperators_closeup_n[:, :, 0].shape[1])

                start_index_of_hor, newest_peaks, arg_min_hor_sort, lines_length_dels, lines_indexes_deleted = self.return_hor_spliter_by_index(peaks_neg_tot, x_min_hor_some, x_max_hor_some)

                arg_org_hor_some_sort = arg_org_hor_some[arg_min_hor_sort]

                start_index_of_hor_with_subset = [start_index_of_hor[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]  # start_index_of_hor[lines_length_dels>0]
                arg_min_hor_sort_with_subset = [arg_min_hor_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]
                lines_indexes_deleted_with_subset = [lines_indexes_deleted[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]
                lines_length_dels_with_subset = [lines_length_dels[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]

                arg_org_hor_some_sort_subset = [arg_org_hor_some_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]

                # arg_min_hor_sort_with_subset=arg_min_hor_sort[lines_length_dels>0]
                # lines_indexes_deleted_with_subset=lines_indexes_deleted[lines_length_dels>0]
                # lines_length_dels_with_subset=lines_length_dels[lines_length_dels>0]

                vahid_subset = np.zeros((len(start_index_of_hor_with_subset), len(start_index_of_hor_with_subset))) - 1
                for kkk1 in range(len(start_index_of_hor_with_subset)):

                    index_del_sub = np.unique(lines_indexes_deleted_with_subset[kkk1])

                    for kkk2 in range(len(start_index_of_hor_with_subset)):

                        if set(lines_indexes_deleted_with_subset[kkk2][0]) < set(lines_indexes_deleted_with_subset[kkk1][0]):
                            vahid_subset[kkk1, kkk2] = kkk1
                        else:
                            pass
                    # print(set(lines_indexes_deleted[kkk2][0]), set(lines_indexes_deleted[kkk1][0]))

                # print(vahid_subset,'zartt222')

                # check the len of matrix if it has no length means that there is no spliter at all

                if len(vahid_subset > 0):
                    # print('hihoo')

                    # find parenets args
                    line_int = np.zeros(vahid_subset.shape[0])

                    childs_id = []
                    arg_child = []
                    for li in range(vahid_subset.shape[0]):
                        # print(vahid_subset[:,li])
                        if np.all(vahid_subset[:, li] == -1):
                            line_int[li] = -1
                        else:
                            line_int[li] = 1

                            # childs_args_in=[ idd for idd in range(vahid_subset.shape[0]) if vahid_subset[idd,li]!=-1]
                            # helpi=[]
                            # for nad in range(len(childs_args_in)):
                            #    helpi.append(arg_min_hor_sort_with_subset[childs_args_in[nad]])

                            arg_child.append(arg_min_hor_sort_with_subset[li])

                    # line_int=vahid_subset[0,:]

                    # print(arg_child,line_int[0],'zartt33333')
                    arg_parent = [arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] == -1]
                    start_index_of_hor_parent = [start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] == -1]
                    # arg_parent=[lines_indexes_deleted_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    # arg_parent=[lines_length_dels_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]

                    # arg_child=[arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]!=-1]
                    start_index_of_hor_child = [start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] != -1]

                    cy_hor_some_sort = cy_hor_some[arg_parent]

                    # print(start_index_of_hor, lines_length_dels ,lines_indexes_deleted,'zartt')

                    # args_indexes=np.array(range(len(start_index_of_hor) ))

                    newest_y_spliter_tot = []

                    for tj in range(len(newest_peaks) - 1):
                        newest_y_spliter = []
                        newest_y_spliter.append(spliter_y_new[i])
                        if tj in np.unique(start_index_of_hor_parent):
                            ##print(cy_hor_some_sort)
                            cy_help = np.array(cy_hor_some_sort)[np.array(start_index_of_hor_parent) == tj]
                            cy_help_sort = np.sort(cy_help)

                            # print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                            for mj in range(len(cy_help_sort)):
                                newest_y_spliter.append(cy_help_sort[mj])
                        newest_y_spliter.append(spliter_y_new[i + 1])

                        newest_y_spliter_tot.append(newest_y_spliter)

                else:
                    line_int = []
                    newest_y_spliter_tot = []

                    for tj in range(len(newest_peaks) - 1):
                        newest_y_spliter = []
                        newest_y_spliter.append(spliter_y_new[i])

                        newest_y_spliter.append(spliter_y_new[i + 1])

                        newest_y_spliter_tot.append(newest_y_spliter)

                # if line_int is all -1 means that big spliters have no child and we can easily go through
                if np.all(np.array(line_int) == -1):
                    for j in range(len(newest_peaks) - 1):
                        newest_y_spliter = newest_y_spliter_tot[j]

                        for n in range(len(newest_y_spliter) - 1):
                            # print(j,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'maaaa')
                            ##plt.imshow(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]])
                            ##plt.show()

                            # print(matrix_new[:,0][ (matrix_new[:,9]==1 )])
                            for jvt in matrix_new[:, 0][(matrix_new[:, 9] == 1) & (matrix_new[:, 6] > newest_y_spliter[n]) & (matrix_new[:, 7] < newest_y_spliter[n + 1]) & ((matrix_new[:, 1]) < newest_peaks[j + 1]) & ((matrix_new[:, 1]) > newest_peaks[j])]:
                                pass

                                ###plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                            # print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                            matrix_new_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter[n]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter[n + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < newest_peaks[j + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > newest_peaks[j])]
                            # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                            if len(matrix_new_new[:, 9][matrix_new_new[:, 9] == 1]) > 0 and np.max(matrix_new_new[:, 8][matrix_new_new[:, 9] == 1]) >= 0.2 * (np.abs(newest_y_spliter[n + 1] - newest_y_spliter[n])):
                                num_col_sub, peaks_neg_fin_sub = self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=5.0)
                            else:
                                peaks_neg_fin_sub = []

                            peaks_sub = []
                            peaks_sub.append(newest_peaks[j])

                            for kj in range(len(peaks_neg_fin_sub)):
                                peaks_sub.append(peaks_neg_fin_sub[kj] + newest_peaks[j])

                            peaks_sub.append(newest_peaks[j + 1])

                            # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                            for kh in range(len(peaks_sub) - 1):
                                boxes.append([peaks_sub[kh], peaks_sub[kh + 1], newest_y_spliter[n], newest_y_spliter[n + 1]])

                else:
                    for j in range(len(newest_peaks) - 1):
                        newest_y_spliter = newest_y_spliter_tot[j]

                        if j in start_index_of_hor_parent:

                            x_min_ch = x_min_hor_some[arg_child]
                            x_max_ch = x_max_hor_some[arg_child]
                            cy_hor_some_sort_child = cy_hor_some[arg_child]
                            cy_hor_some_sort_child = np.sort(cy_hor_some_sort_child)

                            # print(cy_hor_some_sort_child,'ychilds')

                            for n in range(len(newest_y_spliter) - 1):

                                cy_child_in = cy_hor_some_sort_child[(cy_hor_some_sort_child > newest_y_spliter[n]) & (cy_hor_some_sort_child < newest_y_spliter[n + 1])]

                                if len(cy_child_in) > 0:
                                    num_col_ch, peaks_neg_ch = self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=5.0)
                                    # print(peaks_neg_ch,'mizzzz')
                                    # peaks_neg_ch=[]
                                    # for djh in range(len(peaks_neg_ch)):
                                    #    peaks_neg_ch.append( peaks_neg_ch[djh]+newest_peaks[j] )

                                    peaks_neg_ch_tot = self.return_points_with_boundies(peaks_neg_ch, newest_peaks[j], newest_peaks[j + 1])

                                    ss_in_ch, nst_p_ch, arg_n_ch, lines_l_del_ch, lines_in_del_ch = self.return_hor_spliter_by_index(peaks_neg_ch_tot, x_min_ch, x_max_ch)

                                    newest_y_spliter_ch_tot = []

                                    for tjj in range(len(nst_p_ch) - 1):
                                        newest_y_spliter_new = []
                                        newest_y_spliter_new.append(newest_y_spliter[n])
                                        if tjj in np.unique(ss_in_ch):

                                            # print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                                            for mjj in range(len(cy_child_in)):
                                                newest_y_spliter_new.append(cy_child_in[mjj])
                                        newest_y_spliter_new.append(newest_y_spliter[n + 1])

                                        newest_y_spliter_ch_tot.append(newest_y_spliter_new)

                                    for jn in range(len(nst_p_ch) - 1):
                                        newest_y_spliter_h = newest_y_spliter_ch_tot[jn]

                                        for nd in range(len(newest_y_spliter_h) - 1):

                                            matrix_new_new2 = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter_h[nd]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter_h[nd + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < nst_p_ch[jn + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > nst_p_ch[jn])]
                                            # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                            if len(matrix_new_new2[:, 9][matrix_new_new2[:, 9] == 1]) > 0 and np.max(matrix_new_new2[:, 8][matrix_new_new2[:, 9] == 1]) >= 0.2 * (np.abs(newest_y_spliter_h[nd + 1] - newest_y_spliter_h[nd])):
                                                num_col_sub_ch, peaks_neg_fin_sub_ch = self.find_num_col(regions_without_seperators[int(newest_y_spliter_h[nd]) : int(newest_y_spliter_h[nd + 1]), nst_p_ch[jn] : nst_p_ch[jn + 1]], multiplier=5.0)

                                            else:
                                                peaks_neg_fin_sub_ch = []

                                            peaks_sub_ch = []
                                            peaks_sub_ch.append(nst_p_ch[jn])

                                            for kjj in range(len(peaks_neg_fin_sub_ch)):
                                                peaks_sub_ch.append(peaks_neg_fin_sub_ch[kjj] + nst_p_ch[jn])

                                            peaks_sub_ch.append(nst_p_ch[jn + 1])

                                            # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                            for khh in range(len(peaks_sub_ch) - 1):
                                                boxes.append([peaks_sub_ch[khh], peaks_sub_ch[khh + 1], newest_y_spliter_h[nd], newest_y_spliter_h[nd + 1]])

                                else:

                                    matrix_new_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter[n]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter[n + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < newest_peaks[j + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > newest_peaks[j])]
                                    # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                    if len(matrix_new_new[:, 9][matrix_new_new[:, 9] == 1]) > 0 and np.max(matrix_new_new[:, 8][matrix_new_new[:, 9] == 1]) >= 0.2 * (np.abs(newest_y_spliter[n + 1] - newest_y_spliter[n])):
                                        num_col_sub, peaks_neg_fin_sub = self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=5.0)
                                    else:
                                        peaks_neg_fin_sub = []

                                    peaks_sub = []
                                    peaks_sub.append(newest_peaks[j])

                                    for kj in range(len(peaks_neg_fin_sub)):
                                        peaks_sub.append(peaks_neg_fin_sub[kj] + newest_peaks[j])

                                    peaks_sub.append(newest_peaks[j + 1])

                                    # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                    for kh in range(len(peaks_sub) - 1):
                                        boxes.append([peaks_sub[kh], peaks_sub[kh + 1], newest_y_spliter[n], newest_y_spliter[n + 1]])

                        else:
                            for n in range(len(newest_y_spliter) - 1):

                                # plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                                # print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                                matrix_new_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter[n]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter[n + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < newest_peaks[j + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > newest_peaks[j])]
                                # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                if len(matrix_new_new[:, 9][matrix_new_new[:, 9] == 1]) > 0 and np.max(matrix_new_new[:, 8][matrix_new_new[:, 9] == 1]) >= 0.2 * (np.abs(newest_y_spliter[n + 1] - newest_y_spliter[n])):
                                    num_col_sub, peaks_neg_fin_sub = self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=5.0)
                                else:
                                    peaks_neg_fin_sub = []

                                peaks_sub = []
                                peaks_sub.append(newest_peaks[j])

                                for kj in range(len(peaks_neg_fin_sub)):
                                    peaks_sub.append(peaks_neg_fin_sub[kj] + newest_peaks[j])

                                peaks_sub.append(newest_peaks[j + 1])

                                # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                for kh in range(len(peaks_sub) - 1):
                                    boxes.append([peaks_sub[kh], peaks_sub[kh + 1], newest_y_spliter[n], newest_y_spliter[n + 1]])

            else:
                boxes.append([0, seperators_closeup_n[:, :, 0].shape[1], spliter_y_new[i], spliter_y_new[i + 1]])

        return boxes

    def return_boxes_of_images_by_order_of_reading_new(self, spliter_y_new, regions_without_seperators, matrix_of_lines_ch):
        boxes = []

        # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
        # holes in the text and also finding spliter which covers more than one columns.
        for i in range(len(spliter_y_new) - 1):
            # print(spliter_y_new[i],spliter_y_new[i+1])
            matrix_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 6] > spliter_y_new[i]) & (matrix_of_lines_ch[:, 7] < spliter_y_new[i + 1])]
            # print(len( matrix_new[:,9][matrix_new[:,9]==1] ))

            # print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')

            # check to see is there any vertical seperator to find holes.
            if 1 > 0:  # len( matrix_new[:,9][matrix_new[:,9]==1] )>0 and np.max(matrix_new[:,8][matrix_new[:,9]==1])>=0.1*(np.abs(spliter_y_new[i+1]-spliter_y_new[i] )):

                # org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
                # org_img_dichte=org_img_dichte-np.min(org_img_dichte)
                ##plt.figure(figsize=(20,20))
                ##plt.plot(org_img_dichte)
                ##plt.show()
                ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)

                # print(int(spliter_y_new[i]),int(spliter_y_new[i+1]),'firssst')

                # plt.imshow(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:])
                # plt.show()
                try:
                    num_col, peaks_neg_fin = self.find_num_col(regions_without_seperators[int(spliter_y_new[i]) : int(spliter_y_new[i + 1]), :], multiplier=7.0)
                except:
                    peaks_neg_fin = []

                # print(peaks_neg_fin,'peaks_neg_fin')
                # num_col, peaks_neg_fin=find_num_col(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=7.0)
                x_min_hor_some = matrix_new[:, 2][(matrix_new[:, 9] == 0)]
                x_max_hor_some = matrix_new[:, 3][(matrix_new[:, 9] == 0)]
                cy_hor_some = matrix_new[:, 5][(matrix_new[:, 9] == 0)]
                arg_org_hor_some = matrix_new[:, 0][(matrix_new[:, 9] == 0)]

                peaks_neg_tot = self.return_points_with_boundies(peaks_neg_fin, 0, regions_without_seperators[:, :].shape[1])

                start_index_of_hor, newest_peaks, arg_min_hor_sort, lines_length_dels, lines_indexes_deleted = return_hor_spliter_by_index_for_without_verticals(peaks_neg_tot, x_min_hor_some, x_max_hor_some)

                arg_org_hor_some_sort = arg_org_hor_some[arg_min_hor_sort]

                start_index_of_hor_with_subset = [start_index_of_hor[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]  # start_index_of_hor[lines_length_dels>0]
                arg_min_hor_sort_with_subset = [arg_min_hor_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]
                lines_indexes_deleted_with_subset = [lines_indexes_deleted[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]
                lines_length_dels_with_subset = [lines_length_dels[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]

                arg_org_hor_some_sort_subset = [arg_org_hor_some_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]

                # arg_min_hor_sort_with_subset=arg_min_hor_sort[lines_length_dels>0]
                # lines_indexes_deleted_with_subset=lines_indexes_deleted[lines_length_dels>0]
                # lines_length_dels_with_subset=lines_length_dels[lines_length_dels>0]

                vahid_subset = np.zeros((len(start_index_of_hor_with_subset), len(start_index_of_hor_with_subset))) - 1
                for kkk1 in range(len(start_index_of_hor_with_subset)):

                    index_del_sub = np.unique(lines_indexes_deleted_with_subset[kkk1])

                    for kkk2 in range(len(start_index_of_hor_with_subset)):

                        if set(lines_indexes_deleted_with_subset[kkk2][0]) < set(lines_indexes_deleted_with_subset[kkk1][0]):
                            vahid_subset[kkk1, kkk2] = kkk1
                        else:
                            pass
                    # print(set(lines_indexes_deleted[kkk2][0]), set(lines_indexes_deleted[kkk1][0]))

                # check the len of matrix if it has no length means that there is no spliter at all

                if len(vahid_subset > 0):
                    # print('hihoo')

                    # find parenets args
                    line_int = np.zeros(vahid_subset.shape[0])

                    childs_id = []
                    arg_child = []
                    for li in range(vahid_subset.shape[0]):
                        # print(vahid_subset[:,li])
                        if np.all(vahid_subset[:, li] == -1):
                            line_int[li] = -1
                        else:
                            line_int[li] = 1

                            # childs_args_in=[ idd for idd in range(vahid_subset.shape[0]) if vahid_subset[idd,li]!=-1]
                            # helpi=[]
                            # for nad in range(len(childs_args_in)):
                            #    helpi.append(arg_min_hor_sort_with_subset[childs_args_in[nad]])

                            arg_child.append(arg_min_hor_sort_with_subset[li])

                    # line_int=vahid_subset[0,:]

                    arg_parent = [arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] == -1]
                    start_index_of_hor_parent = [start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] == -1]
                    # arg_parent=[lines_indexes_deleted_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                    # arg_parent=[lines_length_dels_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]

                    # arg_child=[arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]!=-1]
                    start_index_of_hor_child = [start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] != -1]

                    cy_hor_some_sort = cy_hor_some[arg_parent]

                    # print(start_index_of_hor, lines_length_dels ,lines_indexes_deleted,'zartt')

                    # args_indexes=np.array(range(len(start_index_of_hor) ))

                    newest_y_spliter_tot = []

                    for tj in range(len(newest_peaks) - 1):
                        newest_y_spliter = []
                        newest_y_spliter.append(spliter_y_new[i])
                        if tj in np.unique(start_index_of_hor_parent):
                            # print(cy_hor_some_sort)
                            cy_help = np.array(cy_hor_some_sort)[np.array(start_index_of_hor_parent) == tj]
                            cy_help_sort = np.sort(cy_help)

                            # print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                            for mj in range(len(cy_help_sort)):
                                newest_y_spliter.append(cy_help_sort[mj])
                        newest_y_spliter.append(spliter_y_new[i + 1])

                        newest_y_spliter_tot.append(newest_y_spliter)

                else:
                    line_int = []
                    newest_y_spliter_tot = []

                    for tj in range(len(newest_peaks) - 1):
                        newest_y_spliter = []
                        newest_y_spliter.append(spliter_y_new[i])

                        newest_y_spliter.append(spliter_y_new[i + 1])

                        newest_y_spliter_tot.append(newest_y_spliter)

                # if line_int is all -1 means that big spliters have no child and we can easily go through
                if np.all(np.array(line_int) == -1):
                    for j in range(len(newest_peaks) - 1):
                        newest_y_spliter = newest_y_spliter_tot[j]

                        for n in range(len(newest_y_spliter) - 1):
                            # print(j,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'maaaa')
                            ##plt.imshow(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]])
                            ##plt.show()

                            # print(matrix_new[:,0][ (matrix_new[:,9]==1 )])
                            for jvt in matrix_new[:, 0][(matrix_new[:, 9] == 1) & (matrix_new[:, 6] > newest_y_spliter[n]) & (matrix_new[:, 7] < newest_y_spliter[n + 1]) & ((matrix_new[:, 1]) < newest_peaks[j + 1]) & ((matrix_new[:, 1]) > newest_peaks[j])]:
                                pass

                                ###plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                            # print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                            matrix_new_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter[n]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter[n + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < newest_peaks[j + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > newest_peaks[j])]
                            # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                            if 1 > 0:  # len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                # print( int(newest_y_spliter[n]),int(newest_y_spliter[n+1]),newest_peaks[j],newest_peaks[j+1] )
                                try:
                                    num_col_sub, peaks_neg_fin_sub = self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=7.0)
                                except:
                                    peaks_neg_fin_sub = []
                            else:
                                peaks_neg_fin_sub = []

                            peaks_sub = []
                            peaks_sub.append(newest_peaks[j])

                            for kj in range(len(peaks_neg_fin_sub)):
                                peaks_sub.append(peaks_neg_fin_sub[kj] + newest_peaks[j])

                            peaks_sub.append(newest_peaks[j + 1])

                            # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                            for kh in range(len(peaks_sub) - 1):
                                boxes.append([peaks_sub[kh], peaks_sub[kh + 1], newest_y_spliter[n], newest_y_spliter[n + 1]])

                else:
                    for j in range(len(newest_peaks) - 1):

                        newest_y_spliter = newest_y_spliter_tot[j]

                        if j in start_index_of_hor_parent:

                            x_min_ch = x_min_hor_some[arg_child]
                            x_max_ch = x_max_hor_some[arg_child]
                            cy_hor_some_sort_child = cy_hor_some[arg_child]
                            cy_hor_some_sort_child = np.sort(cy_hor_some_sort_child)

                            for n in range(len(newest_y_spliter) - 1):

                                cy_child_in = cy_hor_some_sort_child[(cy_hor_some_sort_child > newest_y_spliter[n]) & (cy_hor_some_sort_child < newest_y_spliter[n + 1])]

                                if len(cy_child_in) > 0:
                                    try:
                                        num_col_ch, peaks_neg_ch = self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=7.0)
                                    except:
                                        peaks_neg_ch = []
                                    # print(peaks_neg_ch,'mizzzz')
                                    # peaks_neg_ch=[]
                                    # for djh in range(len(peaks_neg_ch)):
                                    #    peaks_neg_ch.append( peaks_neg_ch[djh]+newest_peaks[j] )

                                    peaks_neg_ch_tot = self.return_points_with_boundies(peaks_neg_ch, newest_peaks[j], newest_peaks[j + 1])

                                    ss_in_ch, nst_p_ch, arg_n_ch, lines_l_del_ch, lines_in_del_ch = return_hor_spliter_by_index_for_without_verticals(peaks_neg_ch_tot, x_min_ch, x_max_ch)

                                    newest_y_spliter_ch_tot = []

                                    for tjj in range(len(nst_p_ch) - 1):
                                        newest_y_spliter_new = []
                                        newest_y_spliter_new.append(newest_y_spliter[n])
                                        if tjj in np.unique(ss_in_ch):

                                            # print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                                            for mjj in range(len(cy_child_in)):
                                                newest_y_spliter_new.append(cy_child_in[mjj])
                                        newest_y_spliter_new.append(newest_y_spliter[n + 1])

                                        newest_y_spliter_ch_tot.append(newest_y_spliter_new)

                                    for jn in range(len(nst_p_ch) - 1):
                                        newest_y_spliter_h = newest_y_spliter_ch_tot[jn]

                                        for nd in range(len(newest_y_spliter_h) - 1):

                                            matrix_new_new2 = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter_h[nd]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter_h[nd + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < nst_p_ch[jn + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > nst_p_ch[jn])]
                                            # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                            if 1 > 0:  # len( matrix_new_new2[:,9][matrix_new_new2[:,9]==1] )>0 and np.max(matrix_new_new2[:,8][matrix_new_new2[:,9]==1])>=0.2*(np.abs(newest_y_spliter_h[nd+1]-newest_y_spliter_h[nd] )):
                                                try:
                                                    num_col_sub_ch, peaks_neg_fin_sub_ch = self.find_num_col(regions_without_seperators[int(newest_y_spliter_h[nd]) : int(newest_y_spliter_h[nd + 1]), nst_p_ch[jn] : nst_p_ch[jn + 1]], multiplier=7.0)
                                                except:
                                                    peaks_neg_fin_sub_ch = []

                                            else:
                                                peaks_neg_fin_sub_ch = []

                                            peaks_sub_ch = []
                                            peaks_sub_ch.append(nst_p_ch[jn])

                                            for kjj in range(len(peaks_neg_fin_sub_ch)):
                                                peaks_sub_ch.append(peaks_neg_fin_sub_ch[kjj] + nst_p_ch[jn])

                                            peaks_sub_ch.append(nst_p_ch[jn + 1])

                                            # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                            for khh in range(len(peaks_sub_ch) - 1):
                                                boxes.append([peaks_sub_ch[khh], peaks_sub_ch[khh + 1], newest_y_spliter_h[nd], newest_y_spliter_h[nd + 1]])

                                else:

                                    matrix_new_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter[n]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter[n + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < newest_peaks[j + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > newest_peaks[j])]
                                    # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                    if 1 > 0:  # len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                        try:
                                            num_col_sub, peaks_neg_fin_sub = self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=7.0)
                                        except:
                                            peaks_neg_fin_sub = []
                                    else:
                                        peaks_neg_fin_sub = []

                                    peaks_sub = []
                                    peaks_sub.append(newest_peaks[j])

                                    for kj in range(len(peaks_neg_fin_sub)):
                                        peaks_sub.append(peaks_neg_fin_sub[kj] + newest_peaks[j])

                                    peaks_sub.append(newest_peaks[j + 1])

                                    # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                    for kh in range(len(peaks_sub) - 1):
                                        boxes.append([peaks_sub[kh], peaks_sub[kh + 1], newest_y_spliter[n], newest_y_spliter[n + 1]])

                        else:
                            for n in range(len(newest_y_spliter) - 1):

                                # plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                                # print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                                matrix_new_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter[n]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter[n + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < newest_peaks[j + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > newest_peaks[j])]
                                # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                if 1 > 0:  # len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                    try:
                                        num_col_sub, peaks_neg_fin_sub = self.find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=5.0)
                                    except:
                                        peaks_neg_fin_sub = []
                                else:
                                    peaks_neg_fin_sub = []

                                peaks_sub = []
                                peaks_sub.append(newest_peaks[j])

                                for kj in range(len(peaks_neg_fin_sub)):
                                    peaks_sub.append(peaks_neg_fin_sub[kj] + newest_peaks[j])

                                peaks_sub.append(newest_peaks[j + 1])

                                # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                for kh in range(len(peaks_sub) - 1):
                                    boxes.append([peaks_sub[kh], peaks_sub[kh + 1], newest_y_spliter[n], newest_y_spliter[n + 1]])

            else:
                boxes.append([0, regions_without_seperators[:, :].shape[1], spliter_y_new[i], spliter_y_new[i + 1]])

        return boxes

    def return_boxes_of_images_by_order_of_reading_2cols(self, spliter_y_new, regions_without_seperators, matrix_of_lines_ch, seperators_closeup_n):
        boxes = []

        # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
        # holes in the text and also finding spliter which covers more than one columns.
        for i in range(len(spliter_y_new) - 1):
            # print(spliter_y_new[i],spliter_y_new[i+1])
            matrix_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 6] > spliter_y_new[i]) & (matrix_of_lines_ch[:, 7] < spliter_y_new[i + 1])]
            # print(len( matrix_new[:,9][matrix_new[:,9]==1] ))

            # print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')

            # check to see is there any vertical seperator to find holes.
            if 1 > 0:  # len( matrix_new[:,9][matrix_new[:,9]==1] )>0 and np.max(matrix_new[:,8][matrix_new[:,9]==1])>=0.1*(np.abs(spliter_y_new[i+1]-spliter_y_new[i] )):
                # print(int(spliter_y_new[i]),int(spliter_y_new[i+1]),'burayaaaa galimiirrrrrrrrrrrrrrrrrrrrrrrrrrr')
                # org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
                # org_img_dichte=org_img_dichte-np.min(org_img_dichte)
                ##plt.figure(figsize=(20,20))
                ##plt.plot(org_img_dichte)
                ##plt.show()
                ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)

                try:
                    num_col, peaks_neg_fin = self.find_num_col(regions_without_seperators[int(spliter_y_new[i]) : int(spliter_y_new[i + 1]), :], multiplier=7.0)

                except:
                    peaks_neg_fin = []
                    num_col = 0

                peaks_neg_tot = self.return_points_with_boundies(peaks_neg_fin, 0, seperators_closeup_n[:, :, 0].shape[1])

                for kh in range(len(peaks_neg_tot) - 1):
                    boxes.append([peaks_neg_tot[kh], peaks_neg_tot[kh + 1], spliter_y_new[i], spliter_y_new[i + 1]])

            else:
                boxes.append([0, seperators_closeup_n[:, :, 0].shape[1], spliter_y_new[i], spliter_y_new[i + 1]])

        return boxes

    def return_hor_spliter_by_index(self, peaks_neg_fin_t, x_min_hor_some, x_max_hor_some):

        arg_min_hor_sort = np.argsort(x_min_hor_some)
        x_min_hor_some_sort = np.sort(x_min_hor_some)
        x_max_hor_some_sort = x_max_hor_some[arg_min_hor_sort]

        arg_minmax = np.array(range(len(peaks_neg_fin_t)))
        indexer_lines = []
        indexes_to_delete = []
        indexer_lines_deletions_len = []
        indexr_uniq_ind = []
        for i in range(len(x_min_hor_some_sort)):
            min_h = peaks_neg_fin_t - x_min_hor_some_sort[i]
            max_h = peaks_neg_fin_t - x_max_hor_some_sort[i]

            min_h[0] = min_h[0]  # +20
            max_h[len(max_h) - 1] = max_h[len(max_h) - 1]  ##-20

            min_h_neg = arg_minmax[(min_h < 0) & (np.abs(min_h) < 360)]
            max_h_neg = arg_minmax[(max_h >= 0) & (np.abs(max_h) < 360)]

            if len(min_h_neg) > 0 and len(max_h_neg) > 0:
                deletions = list(range(min_h_neg[0] + 1, max_h_neg[0]))
                unique_delets_int = []
                # print(deletions,len(deletions),'delii')
                if len(deletions) > 0:
                    # print(deletions,len(deletions),'delii2')

                    for j in range(len(deletions)):
                        indexes_to_delete.append(deletions[j])
                        # print(deletions,indexes_to_delete,'badiii')
                        unique_delets = np.unique(indexes_to_delete)
                        # print(min_h_neg[0],unique_delets)
                        unique_delets_int = unique_delets[unique_delets < min_h_neg[0]]

                    indexer_lines_deletions_len.append(len(deletions))
                    indexr_uniq_ind.append([deletions])

                else:
                    indexer_lines_deletions_len.append(0)
                    indexr_uniq_ind.append(-999)

                index_line_true = min_h_neg[0] - len(unique_delets_int)
                # print(index_line_true)
                if index_line_true > 0 and min_h_neg[0] >= 2:
                    index_line_true = index_line_true
                else:
                    index_line_true = min_h_neg[0]

                indexer_lines.append(index_line_true)

                if len(unique_delets_int) > 0:
                    for dd in range(len(unique_delets_int)):
                        indexes_to_delete.append(unique_delets_int[dd])
            else:
                indexer_lines.append(-999)
                indexer_lines_deletions_len.append(-999)
                indexr_uniq_ind.append(-999)

        peaks_true = []
        for m in range(len(peaks_neg_fin_t)):
            if m in indexes_to_delete:
                pass
            else:
                peaks_true.append(peaks_neg_fin_t[m])
        return indexer_lines, peaks_true, arg_min_hor_sort, indexer_lines_deletions_len, indexr_uniq_ind

    def return_region_segmentation_after_implementing_not_head_maintext_parallel(self, image_regions_eraly_p, boxes):
        image_revised = np.zeros((image_regions_eraly_p.shape[0], image_regions_eraly_p.shape[1]))
        for i in range(len(boxes)):

            image_box = image_regions_eraly_p[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][0]) : int(boxes[i][1])]
            image_box = np.array(image_box)
            # plt.imshow(image_box)
            # plt.show()

            # print(int(boxes[i][2]),int(boxes[i][3]),int(boxes[i][0]),int(boxes[i][1]),'addaa')
            image_box = self.implent_law_head_main_not_parallel(image_box)
            image_box = self.implent_law_head_main_not_parallel(image_box)
            image_box = self.implent_law_head_main_not_parallel(image_box)

            image_revised[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][0]) : int(boxes[i][1])] = image_box[:, :]
        return image_revised

    def tear_main_texts_on_the_boundaries_of_boxes(self, img_revised_tab, boxes):
        for i in range(len(boxes)):
            img_revised_tab[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][1] - 10) : int(boxes[i][1]), 0][img_revised_tab[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][1] - 10) : int(boxes[i][1]), 0] == 1] = 0
            img_revised_tab[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][1] - 10) : int(boxes[i][1]), 1][img_revised_tab[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][1] - 10) : int(boxes[i][1]), 1] == 1] = 0
            img_revised_tab[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][1] - 10) : int(boxes[i][1]), 2][img_revised_tab[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][1] - 10) : int(boxes[i][1]), 2] == 1] = 0
        return img_revised_tab

    def implent_law_head_main_not_parallel(self, text_regions):
        # print(text_regions.shape)
        text_indexes = [1, 2]  # 1: main text , 2: header , 3: comments

        for t_i in text_indexes:
            textline_mask = text_regions[:, :] == t_i
            textline_mask = textline_mask * 255.0

            textline_mask = textline_mask.astype(np.uint8)
            textline_mask = np.repeat(textline_mask[:, :, np.newaxis], 3, axis=2)
            kernel = np.ones((5, 5), np.uint8)

            # print(type(textline_mask),np.unique(textline_mask),textline_mask.shape)
            imgray = cv2.cvtColor(textline_mask, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 0, 255, 0)

            if t_i == 1:
                contours_main, hirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # print(type(contours_main))
                areas_main = np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
                M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
                cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
                cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
                x_min_main = np.array([np.min(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])
                x_max_main = np.array([np.max(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

                y_min_main = np.array([np.min(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
                y_max_main = np.array([np.max(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
                # print(contours_main[0],np.shape(contours_main[0]),contours_main[0][:,0,0])
            elif t_i == 2:
                contours_header, hirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # print(type(contours_header))
                areas_header = np.array([cv2.contourArea(contours_header[j]) for j in range(len(contours_header))])
                M_header = [cv2.moments(contours_header[j]) for j in range(len(contours_header))]
                cx_header = [(M_header[j]["m10"] / (M_header[j]["m00"] + 1e-32)) for j in range(len(M_header))]
                cy_header = [(M_header[j]["m01"] / (M_header[j]["m00"] + 1e-32)) for j in range(len(M_header))]

                x_min_header = np.array([np.min(contours_header[j][:, 0, 0]) for j in range(len(contours_header))])
                x_max_header = np.array([np.max(contours_header[j][:, 0, 0]) for j in range(len(contours_header))])

                y_min_header = np.array([np.min(contours_header[j][:, 0, 1]) for j in range(len(contours_header))])
                y_max_header = np.array([np.max(contours_header[j][:, 0, 1]) for j in range(len(contours_header))])

        args = np.array(range(1, len(cy_header) + 1))
        args_main = np.array(range(1, len(cy_main) + 1))
        for jj in range(len(contours_main)):
            headers_in_main = [(cy_header > y_min_main[jj]) & ((cy_header < y_max_main[jj]))]
            mains_in_main = [(cy_main > y_min_main[jj]) & ((cy_main < y_max_main[jj]))]
            args_log = args * headers_in_main
            res = args_log[args_log > 0]
            res_true = res - 1

            args_log_main = args_main * mains_in_main
            res_main = args_log_main[args_log_main > 0]
            res_true_main = res_main - 1

            if len(res_true) > 0:
                sum_header = np.sum(areas_header[res_true])
                sum_main = np.sum(areas_main[res_true_main])
                if sum_main > sum_header:
                    cnt_int = [contours_header[j] for j in res_true]
                    text_regions = cv2.fillPoly(text_regions, pts=cnt_int, color=(1, 1, 1))
                else:
                    cnt_int = [contours_main[j] for j in res_true_main]
                    text_regions = cv2.fillPoly(text_regions, pts=cnt_int, color=(2, 2, 2))

        for jj in range(len(contours_header)):
            main_in_header = [(cy_main > y_min_header[jj]) & ((cy_main < y_max_header[jj]))]
            header_in_header = [(cy_header > y_min_header[jj]) & ((cy_header < y_max_header[jj]))]
            args_log = args_main * main_in_header
            res = args_log[args_log > 0]
            res_true = res - 1

            args_log_header = args * header_in_header
            res_header = args_log_header[args_log_header > 0]
            res_true_header = res_header - 1

            if len(res_true) > 0:

                sum_header = np.sum(areas_header[res_true_header])
                sum_main = np.sum(areas_main[res_true])

                if sum_main > sum_header:

                    cnt_int = [contours_header[j] for j in res_true_header]
                    text_regions = cv2.fillPoly(text_regions, pts=cnt_int, color=(1, 1, 1))
                else:
                    cnt_int = [contours_main[j] for j in res_true]
                    text_regions = cv2.fillPoly(text_regions, pts=cnt_int, color=(2, 2, 2))

        return text_regions

    def delete_seperator_around(self, spliter_y, peaks_neg, image_by_region):
        # format of subboxes box=[x1, x2 , y1, y2]

        if len(image_by_region.shape) == 3:
            for i in range(len(spliter_y) - 1):
                for j in range(1, len(peaks_neg[i]) - 1):
                    image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0] == 6] = 0
                    image_by_region[spliter_y[i] : spliter_y[i + 1], peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 1] == 6] = 0
                    image_by_region[spliter_y[i] : spliter_y[i + 1], peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 2] == 6] = 0

                    image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0] == 7] = 0
                    image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 1] == 7] = 0
                    image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 2] == 7] = 0
        else:
            for i in range(len(spliter_y) - 1):
                for j in range(1, len(peaks_neg[i]) - 1):
                    image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j])][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j])] == 6] = 0

                    image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j])][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j])] == 7] = 0
        return image_by_region

    def find_features_of_contoures(self, contours_main):

        areas_main = np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
        M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
        cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
        cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
        x_min_main = np.array([np.min(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])
        x_max_main = np.array([np.max(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

        y_min_main = np.array([np.min(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
        y_max_main = np.array([np.max(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])

        return y_min_main, y_max_main

    def add_tables_heuristic_to_layout(self, image_regions_eraly_p, boxes, slope_mean_hor, spliter_y, peaks_neg_tot, image_revised):

        image_revised_1 = self.delete_seperator_around(spliter_y, peaks_neg_tot, image_revised)
        img_comm_e = np.zeros(image_revised_1.shape)
        img_comm = np.repeat(img_comm_e[:, :, np.newaxis], 3, axis=2)

        for indiv in np.unique(image_revised_1):

            # print(indiv,'indd')
            image_col = (image_revised_1 == indiv) * 255
            img_comm_in = np.repeat(image_col[:, :, np.newaxis], 3, axis=2)
            img_comm_in = img_comm_in.astype(np.uint8)

            imgray = cv2.cvtColor(img_comm_in, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(imgray, 0, 255, 0)

            contours, hirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            main_contours = filter_contours_area_of_image_tables(thresh, contours, hirarchy, max_area=1, min_area=0.0001)

            img_comm = cv2.fillPoly(img_comm, pts=main_contours, color=(indiv, indiv, indiv))
            ###img_comm_in=cv2.fillPoly(img_comm, pts =interior_contours, color=(0,0,0))

            # img_comm=np.repeat(img_comm[:, :, np.newaxis], 3, axis=2)
            img_comm = img_comm.astype(np.uint8)

        if not isNaN(slope_mean_hor):
            image_revised_last = np.zeros((image_regions_eraly_p.shape[0], image_regions_eraly_p.shape[1], 3))
            for i in range(len(boxes)):

                image_box = img_comm[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][0]) : int(boxes[i][1]), :]

                image_box_tabels_1 = (image_box[:, :, 0] == 7) * 1

                contours_tab, _ = return_contours_of_image(image_box_tabels_1)

                contours_tab = filter_contours_area_of_image_tables(image_box_tabels_1, contours_tab, _, 1, 0.001)

                image_box_tabels_1 = (image_box[:, :, 0] == 6) * 1

                image_box_tabels_and_m_text = ((image_box[:, :, 0] == 7) | (image_box[:, :, 0] == 1)) * 1
                image_box_tabels_and_m_text = image_box_tabels_and_m_text.astype(np.uint8)

                image_box_tabels_1 = image_box_tabels_1.astype(np.uint8)
                image_box_tabels_1 = cv2.dilate(image_box_tabels_1, self.kernel, iterations=5)

                contours_table_m_text, _ = return_contours_of_image(image_box_tabels_and_m_text)

                image_box_tabels = np.repeat(image_box_tabels_1[:, :, np.newaxis], 3, axis=2)

                image_box_tabels = image_box_tabels.astype(np.uint8)
                imgray = cv2.cvtColor(image_box_tabels, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                contours_line, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                y_min_main_line, y_max_main_line = self.find_features_of_contoures(contours_line)
                # _,_,y_min_main_line ,y_max_main_line,x_min_main_line,x_max_main_line=find_new_features_of_contoures(contours_line)
                y_min_main_tab, y_max_main_tab = self.find_features_of_contoures(contours_tab)

                cx_tab_m_text, cy_tab_m_text, x_min_tab_m_text, x_max_tab_m_text, y_min_tab_m_text, y_max_tab_m_text = find_new_features_of_contoures(contours_table_m_text)
                cx_tabl, cy_tabl, x_min_tabl, x_max_tabl, y_min_tabl, y_max_tabl, _ = find_new_features_of_contoures(contours_tab)

                if len(y_min_main_tab) > 0:
                    y_down_tabs = []
                    y_up_tabs = []

                    for i_t in range(len(y_min_main_tab)):
                        y_down_tab = []
                        y_up_tab = []
                        for i_l in range(len(y_min_main_line)):
                            if y_min_main_tab[i_t] > y_min_main_line[i_l] and y_max_main_tab[i_t] > y_min_main_line[i_l] and y_min_main_tab[i_t] > y_max_main_line[i_l] and y_max_main_tab[i_t] > y_min_main_line[i_l]:
                                pass
                            elif y_min_main_tab[i_t] < y_max_main_line[i_l] and y_max_main_tab[i_t] < y_max_main_line[i_l] and y_max_main_tab[i_t] < y_min_main_line[i_l] and y_min_main_tab[i_t] < y_min_main_line[i_l]:
                                pass
                            elif np.abs(y_max_main_line[i_l] - y_min_main_line[i_l]) < 100:
                                pass

                            else:
                                y_up_tab.append(np.min([y_min_main_line[i_l], y_min_main_tab[i_t]]))
                                y_down_tab.append(np.max([y_max_main_line[i_l], y_max_main_tab[i_t]]))

                        if len(y_up_tab) == 0:
                            for v_n in range(len(cx_tab_m_text)):
                                if cx_tabl[i_t] <= x_max_tab_m_text[v_n] and cx_tabl[i_t] >= x_min_tab_m_text[v_n] and cy_tabl[i_t] <= y_max_tab_m_text[v_n] and cy_tabl[i_t] >= y_min_tab_m_text[v_n] and cx_tabl[i_t] != cx_tab_m_text[v_n] and cy_tabl[i_t] != cy_tab_m_text[v_n]:
                                    y_up_tabs.append(y_min_tab_m_text[v_n])
                                    y_down_tabs.append(y_max_tab_m_text[v_n])
                            # y_up_tabs.append(y_min_main_tab[i_t])
                            # y_down_tabs.append(y_max_main_tab[i_t])
                        else:
                            y_up_tabs.append(np.min(y_up_tab))
                            y_down_tabs.append(np.max(y_down_tab))

                else:
                    y_down_tabs = []
                    y_up_tabs = []
                    pass

                for ii in range(len(y_up_tabs)):
                    image_box[y_up_tabs[ii] : y_down_tabs[ii], :, 0] = 7

                image_revised_last[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][0]) : int(boxes[i][1]), :] = image_box[:, :, :]

        else:
            for i in range(len(boxes)):

                image_box = img_comm[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][0]) : int(boxes[i][1]), :]
                image_revised_last[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][0]) : int(boxes[i][1]), :] = image_box[:, :, :]

                ##plt.figure(figsize=(20,20))
                ##plt.imshow(image_box[:,:,0])
                ##plt.show()
        return image_revised_last

    def find_features_of_contours(self, contours_main):

        areas_main = np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
        M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
        cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
        cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
        x_min_main = np.array([np.min(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])
        x_max_main = np.array([np.max(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

        y_min_main = np.array([np.min(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
        y_max_main = np.array([np.max(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])

        return y_min_main, y_max_main, areas_main

    def remove_headers_and_mains_intersection(self, seperators_closeup_n, img_revised_tab, boxes):
        for ind in range(len(boxes)):
            asp = np.zeros((img_revised_tab[:, :, 0].shape[0], seperators_closeup_n[:, :, 0].shape[1]))
            asp[int(boxes[ind][2]) : int(boxes[ind][3]), int(boxes[ind][0]) : int(boxes[ind][1])] = img_revised_tab[int(boxes[ind][2]) : int(boxes[ind][3]), int(boxes[ind][0]) : int(boxes[ind][1]), 0]

            head_patch_con = (asp[:, :] == 2) * 1
            main_patch_con = (asp[:, :] == 1) * 1
            # print(head_patch_con)
            head_patch_con = head_patch_con.astype(np.uint8)
            main_patch_con = main_patch_con.astype(np.uint8)

            head_patch_con = np.repeat(head_patch_con[:, :, np.newaxis], 3, axis=2)
            main_patch_con = np.repeat(main_patch_con[:, :, np.newaxis], 3, axis=2)

            imgray = cv2.cvtColor(head_patch_con, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 0, 255, 0)

            contours_head_patch_con, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_head_patch_con = return_parent_contours(contours_head_patch_con, hiearchy)

            imgray = cv2.cvtColor(main_patch_con, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 0, 255, 0)

            contours_main_patch_con, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_main_patch_con = return_parent_contours(contours_main_patch_con, hiearchy)

            y_patch_head_min, y_patch_head_max, _ = self.find_features_of_contours(contours_head_patch_con)
            y_patch_main_min, y_patch_main_max, _ = self.find_features_of_contours(contours_main_patch_con)

            for i in range(len(y_patch_head_min)):
                for j in range(len(y_patch_main_min)):
                    if y_patch_head_max[i] > y_patch_main_min[j] and y_patch_head_min[i] < y_patch_main_min[j]:
                        y_down = y_patch_head_max[i]
                        y_up = y_patch_main_min[j]

                        patch_intersection = np.zeros(asp.shape)
                        patch_intersection[y_up:y_down, :] = asp[y_up:y_down, :]

                        head_patch_con = (patch_intersection[:, :] == 2) * 1
                        main_patch_con = (patch_intersection[:, :] == 1) * 1
                        head_patch_con = head_patch_con.astype(np.uint8)
                        main_patch_con = main_patch_con.astype(np.uint8)

                        head_patch_con = np.repeat(head_patch_con[:, :, np.newaxis], 3, axis=2)
                        main_patch_con = np.repeat(main_patch_con[:, :, np.newaxis], 3, axis=2)

                        imgray = cv2.cvtColor(head_patch_con, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                        contours_head_patch_con, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        contours_head_patch_con = return_parent_contours(contours_head_patch_con, hiearchy)

                        imgray = cv2.cvtColor(main_patch_con, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                        contours_main_patch_con, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        contours_main_patch_con = return_parent_contours(contours_main_patch_con, hiearchy)

                        _, _, areas_head = self.find_features_of_contours(contours_head_patch_con)
                        _, _, areas_main = self.find_features_of_contours(contours_main_patch_con)

                        if np.sum(areas_head) > np.sum(areas_main):
                            img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0][img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0] == 1] = 2
                        else:
                            img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0][img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0] == 2] = 1

                    elif y_patch_head_min[i] < y_patch_main_max[j] and y_patch_head_max[i] > y_patch_main_max[j]:
                        y_down = y_patch_main_max[j]
                        y_up = y_patch_head_min[i]

                        patch_intersection = np.zeros(asp.shape)
                        patch_intersection[y_up:y_down, :] = asp[y_up:y_down, :]

                        head_patch_con = (patch_intersection[:, :] == 2) * 1
                        main_patch_con = (patch_intersection[:, :] == 1) * 1
                        head_patch_con = head_patch_con.astype(np.uint8)
                        main_patch_con = main_patch_con.astype(np.uint8)

                        head_patch_con = np.repeat(head_patch_con[:, :, np.newaxis], 3, axis=2)
                        main_patch_con = np.repeat(main_patch_con[:, :, np.newaxis], 3, axis=2)

                        imgray = cv2.cvtColor(head_patch_con, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                        contours_head_patch_con, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        contours_head_patch_con = return_parent_contours(contours_head_patch_con, hiearchy)

                        imgray = cv2.cvtColor(main_patch_con, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                        contours_main_patch_con, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        contours_main_patch_con = return_parent_contours(contours_main_patch_con, hiearchy)

                        _, _, areas_head = self.find_features_of_contours(contours_head_patch_con)
                        _, _, areas_main = self.find_features_of_contours(contours_main_patch_con)

                        if np.sum(areas_head) > np.sum(areas_main):
                            img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0][img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0] == 1] = 2
                        else:
                            img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0][img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0] == 2] = 1

                        # print(np.unique(patch_intersection) )
                        ##plt.figure(figsize=(20,20))
                        ##plt.imshow(patch_intersection)
                        ##plt.show()
                    else:
                        pass

        return img_revised_tab

    def order_of_regions(self, textline_mask, contours_main, contours_header, y_ref):

        ##plt.imshow(textline_mask)
        ##plt.show()
        """
        print(len(contours_main),'contours_main')
        mada_n=textline_mask.sum(axis=1)
        y=mada_n[:]

        y_help=np.zeros(len(y)+40)
        y_help[20:len(y)+20]=y
        x=np.array( range(len(y)) )


        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)

        ##plt.imshow(textline_mask[:,:])
        ##plt.show()


        sigma_gaus=8

        z= gaussian_filter1d(y_help, sigma_gaus)
        zneg_rev=-y_help+np.max(y_help)

        zneg=np.zeros(len(zneg_rev)+40)
        zneg[20:len(zneg_rev)+20]=zneg_rev
        zneg= gaussian_filter1d(zneg, sigma_gaus)


        peaks, _ = find_peaks(z, height=0)
        peaks_neg, _ = find_peaks(zneg, height=0)

        peaks_neg=peaks_neg-20-20
        peaks=peaks-20
        """

        textline_sum_along_width = textline_mask.sum(axis=1)

        y = textline_sum_along_width[:]
        y_padded = np.zeros(len(y) + 40)
        y_padded[20 : len(y) + 20] = y
        x = np.array(range(len(y)))

        peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)

        sigma_gaus = 8

        z = gaussian_filter1d(y_padded, sigma_gaus)
        zneg_rev = -y_padded + np.max(y_padded)

        zneg = np.zeros(len(zneg_rev) + 40)
        zneg[20 : len(zneg_rev) + 20] = zneg_rev
        zneg = gaussian_filter1d(zneg, sigma_gaus)

        peaks, _ = find_peaks(z, height=0)
        peaks_neg, _ = find_peaks(zneg, height=0)

        peaks_neg = peaks_neg - 20 - 20
        peaks = peaks - 20

        ##plt.plot(z)
        ##plt.show()

        if contours_main != None:
            areas_main = np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
            M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
            cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
            cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
            x_min_main = np.array([np.min(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])
            x_max_main = np.array([np.max(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

            y_min_main = np.array([np.min(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
            y_max_main = np.array([np.max(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])

        if len(contours_header) != None:
            areas_header = np.array([cv2.contourArea(contours_header[j]) for j in range(len(contours_header))])
            M_header = [cv2.moments(contours_header[j]) for j in range(len(contours_header))]
            cx_header = [(M_header[j]["m10"] / (M_header[j]["m00"] + 1e-32)) for j in range(len(M_header))]
            cy_header = [(M_header[j]["m01"] / (M_header[j]["m00"] + 1e-32)) for j in range(len(M_header))]

            x_min_header = np.array([np.min(contours_header[j][:, 0, 0]) for j in range(len(contours_header))])
            x_max_header = np.array([np.max(contours_header[j][:, 0, 0]) for j in range(len(contours_header))])

            y_min_header = np.array([np.min(contours_header[j][:, 0, 1]) for j in range(len(contours_header))])
            y_max_header = np.array([np.max(contours_header[j][:, 0, 1]) for j in range(len(contours_header))])
            # print(cy_main,'mainy')

        peaks_neg_new = []

        peaks_neg_new.append(0 + y_ref)
        for iii in range(len(peaks_neg)):
            peaks_neg_new.append(peaks_neg[iii] + y_ref)

        peaks_neg_new.append(textline_mask.shape[0] + y_ref)

        if len(cy_main) > 0 and np.max(cy_main) > np.max(peaks_neg_new):
            cy_main = np.array(cy_main) * (np.max(peaks_neg_new) / np.max(cy_main)) - 10

        if contours_main != None:
            indexer_main = np.array(range(len(contours_main)))

        if contours_main != None:
            len_main = len(contours_main)
        else:
            len_main = 0

        matrix_of_orders = np.zeros((len(contours_main) + len(contours_header), 5))

        matrix_of_orders[:, 0] = np.array(range(len(contours_main) + len(contours_header)))

        matrix_of_orders[: len(contours_main), 1] = 1
        matrix_of_orders[len(contours_main) :, 1] = 2

        matrix_of_orders[: len(contours_main), 2] = cx_main
        matrix_of_orders[len(contours_main) :, 2] = cx_header

        matrix_of_orders[: len(contours_main), 3] = cy_main
        matrix_of_orders[len(contours_main) :, 3] = cy_header

        matrix_of_orders[: len(contours_main), 4] = np.array(range(len(contours_main)))
        matrix_of_orders[len(contours_main) :, 4] = np.array(range(len(contours_header)))

        # print(peaks_neg_new,'peaks_neg_new')

        # print(matrix_of_orders,'matrix_of_orders')
        # print(peaks_neg_new,np.max(peaks_neg_new))
        final_indexers_sorted = []
        final_types = []
        final_index_type = []
        for i in range(len(peaks_neg_new) - 1):
            top = peaks_neg_new[i]
            down = peaks_neg_new[i + 1]

            # print(top,down,'topdown')

            indexes_in = matrix_of_orders[:, 0][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]
            cxs_in = matrix_of_orders[:, 2][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]
            cys_in = matrix_of_orders[:, 3][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]
            types_of_text = matrix_of_orders[:, 1][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]
            index_types_of_text = matrix_of_orders[:, 4][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]

            # print(top,down)
            # print(cys_in,'cyyyins')
            # print(indexes_in,'indexes')
            sorted_inside = np.argsort(cxs_in)

            ind_in_int = indexes_in[sorted_inside]
            ind_in_type = types_of_text[sorted_inside]
            ind_ind_type = index_types_of_text[sorted_inside]

            for j in range(len(ind_in_int)):
                final_indexers_sorted.append(int(ind_in_int[j]))
                final_types.append(int(ind_in_type[j]))
                final_index_type.append(int(ind_ind_type[j]))

        ##matrix_of_orders[:len_main,4]=final_indexers_sorted[:]

        # print(peaks_neg_new,'peaks')
        # print(final_indexers_sorted,'indexsorted')
        # print(final_types,'types')
        # print(final_index_type,'final_index_type')

        return final_indexers_sorted, matrix_of_orders, final_types, final_index_type

    def order_and_id_of_texts(self, found_polygons_text_region, found_polygons_text_region_h, matrix_of_orders, indexes_sorted, index_of_types, kind_of_texts, ref_point):
        indexes_sorted = np.array(indexes_sorted)
        index_of_types = np.array(index_of_types)
        kind_of_texts = np.array(kind_of_texts)

        id_of_texts = []
        order_of_texts = []

        index_of_types_1 = index_of_types[kind_of_texts == 1]
        indexes_sorted_1 = indexes_sorted[kind_of_texts == 1]

        index_of_types_2 = index_of_types[kind_of_texts == 2]
        indexes_sorted_2 = indexes_sorted[kind_of_texts == 2]

        ##print(index_of_types,'index_of_types')
        ##print(kind_of_texts,'kind_of_texts')
        ##print(len(found_polygons_text_region),'found_polygons_text_region')
        ##print(index_of_types_1,'index_of_types_1')
        ##print(indexes_sorted_1,'indexes_sorted_1')
        index_b = 0 + ref_point
        for mm in range(len(found_polygons_text_region)):

            id_of_texts.append("r" + str(index_b))
            interest = indexes_sorted_1[indexes_sorted_1 == index_of_types_1[mm]]

            if len(interest) > 0:
                order_of_texts.append(interest[0])
                index_b += 1
            else:
                pass

        for mm in range(len(found_polygons_text_region_h)):
            id_of_texts.append("r" + str(index_b))
            interest = indexes_sorted_2[index_of_types_2[mm]]
            order_of_texts.append(interest)
            index_b += 1

        return order_of_texts, id_of_texts

    def get_text_region_boxes_by_given_contours(self, contours):

        kernel = np.ones((5, 5), np.uint8)
        boxes = []
        contours_new = []
        for jj in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[jj])

            boxes.append([x, y, w, h])
            contours_new.append(contours[jj])

        del contours
        return boxes, contours_new

    def return_teilwiese_deskewed_lines(self, text_regions_p, textline_rotated):

        kernel = np.ones((5, 5), np.uint8)
        textline_rotated = cv2.erode(textline_rotated, kernel, iterations=1)

        textline_rotated_new = np.zeros(textline_rotated.shape)
        rgb_m = 1
        rgb_h = 2

        cnt_m, boxes_m = self.return_contours_of_interested_region_and_bounding_box(text_regions_p, rgb_m)
        cnt_h, boxes_h = self.return_contours_of_interested_region_and_bounding_box(text_regions_p, rgb_h)

        areas_cnt_m = np.array([cv2.contourArea(cnt_m[j]) for j in range(len(cnt_m))])

        argmax = np.argmax(areas_cnt_m)

        # plt.imshow(textline_rotated[ boxes_m[argmax][1]:boxes_m[argmax][1]+boxes_m[argmax][3] ,boxes_m[argmax][0]:boxes_m[argmax][0]+boxes_m[argmax][2]])
        # plt.show()

        for argmax in range(len(boxes_m)):

            textline_text_region = textline_rotated[boxes_m[argmax][1] : boxes_m[argmax][1] + boxes_m[argmax][3], boxes_m[argmax][0] : boxes_m[argmax][0] + boxes_m[argmax][2]]

            textline_text_region_revised = self.seperate_lines_new(textline_text_region, 0)
            # except:
            #    textline_text_region_revised=textline_rotated[ boxes_m[argmax][1]:boxes_m[argmax][1]+boxes_m[argmax][3] ,boxes_m[argmax][0]:boxes_m[argmax][0]+boxes_m[argmax][2]  ]
            textline_rotated_new[boxes_m[argmax][1] : boxes_m[argmax][1] + boxes_m[argmax][3], boxes_m[argmax][0] : boxes_m[argmax][0] + boxes_m[argmax][2]] = textline_text_region_revised[:, :]

        # textline_rotated_new[textline_rotated_new>0]=1
        # textline_rotated_new[textline_rotated_new<0]=0
        # plt.imshow(textline_rotated_new)
        # plt.show()

    def return_contours_of_interested_region_and_bounding_box(self, region_pre_p, pixel):

        # pixels of images are identified by 5
        cnts_images = (region_pre_p[:, :, 0] == pixel) * 1
        cnts_images = cnts_images.astype(np.uint8)
        cnts_images = np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
        imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_imgs = return_parent_contours(contours_imgs, hiearchy)
        contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=1, min_area=0.0003)

        boxes = []

        for jj in range(len(contours_imgs)):
            x, y, w, h = cv2.boundingRect(contours_imgs[jj])
            boxes.append([int(x), int(y), int(w), int(h)])
        return contours_imgs, boxes

    def find_number_of_columns_in_document(self, region_pre_p, num_col_classifier, pixel_lines, contours_h=None):

        seperators_closeup = ((region_pre_p[:, :, :] == pixel_lines)) * 1

        seperators_closeup[0:110, :, :] = 0
        seperators_closeup[seperators_closeup.shape[0] - 150 :, :, :] = 0

        kernel = np.ones((5, 5), np.uint8)

        seperators_closeup = seperators_closeup.astype(np.uint8)
        seperators_closeup = cv2.dilate(seperators_closeup, kernel, iterations=1)
        seperators_closeup = cv2.erode(seperators_closeup, kernel, iterations=1)

        ##plt.imshow(seperators_closeup[:,:,0])
        ##plt.show()
        seperators_closeup_new = np.zeros((seperators_closeup.shape[0], seperators_closeup.shape[1]))

        ##_,seperators_closeup_n=self.combine_hor_lines_and_delete_cross_points_and_get_lines_features_back(region_pre_p[:,:,0])
        seperators_closeup_n = np.copy(seperators_closeup)

        seperators_closeup_n = seperators_closeup_n.astype(np.uint8)
        ##plt.imshow(seperators_closeup_n[:,:,0])
        ##plt.show()

        seperators_closeup_n_binary = np.zeros((seperators_closeup_n.shape[0], seperators_closeup_n.shape[1]))
        seperators_closeup_n_binary[:, :] = seperators_closeup_n[:, :, 0]

        seperators_closeup_n_binary[:, :][seperators_closeup_n_binary[:, :] != 0] = 1
        # seperators_closeup_n_binary[:,:][seperators_closeup_n_binary[:,:]==0]=255
        # seperators_closeup_n_binary[:,:][seperators_closeup_n_binary[:,:]==-255]=0

        # seperators_closeup_n_binary=(seperators_closeup_n_binary[:,:]==2)*1

        # gray = cv2.cvtColor(seperators_closeup_n, cv2.COLOR_BGR2GRAY)

        # print(np.unique(seperators_closeup_n_binary))

        ##plt.imshow(seperators_closeup_n_binary)
        ##plt.show()

        # print( np.unique(gray),np.unique(seperators_closeup_n[:,:,1]) )

        gray = cv2.bitwise_not(seperators_closeup_n_binary)
        gray = gray.astype(np.uint8)

        ##plt.imshow(gray)
        ##plt.show()
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        ##plt.imshow(bw[:,:])
        ##plt.show()

        horizontal = np.copy(bw)
        vertical = np.copy(bw)

        cols = horizontal.shape[1]
        horizontal_size = cols // 30
        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)

        kernel = np.ones((5, 5), np.uint8)

        horizontal = cv2.dilate(horizontal, kernel, iterations=2)
        horizontal = cv2.erode(horizontal, kernel, iterations=2)
        # plt.imshow(horizontal)
        # plt.show()

        rows = vertical.shape[0]
        verticalsize = rows // 30
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)

        vertical = cv2.dilate(vertical, kernel, iterations=1)
        # Show extracted vertical lines

        horizontal, special_seperators = self.combine_hor_lines_and_delete_cross_points_and_get_lines_features_back_new(vertical, horizontal)

        ##plt.imshow(vertical)
        ##plt.show()
        # print(vertical.shape,np.unique(vertical),'verticalvertical')
        seperators_closeup_new[:, :][vertical[:, :] != 0] = 1
        seperators_closeup_new[:, :][horizontal[:, :] != 0] = 1

        ##plt.imshow(seperators_closeup_new)
        ##plt.show()
        ##seperators_closeup_n
        vertical = np.repeat(vertical[:, :, np.newaxis], 3, axis=2)
        vertical = vertical.astype(np.uint8)

        ##plt.plot(vertical[:,:,0].sum(axis=0))
        ##plt.show()

        # plt.plot(vertical[:,:,0].sum(axis=1))
        # plt.show()

        imgray = cv2.cvtColor(vertical, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours_line_vers, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        slope_lines, dist_x, x_min_main, x_max_main, cy_main, slope_lines_org, y_min_main, y_max_main, cx_main = find_features_of_lines(contours_line_vers)
        # print(slope_lines,'vertical')
        args = np.array(range(len(slope_lines)))
        args_ver = args[slope_lines == 1]
        dist_x_ver = dist_x[slope_lines == 1]
        y_min_main_ver = y_min_main[slope_lines == 1]
        y_max_main_ver = y_max_main[slope_lines == 1]
        x_min_main_ver = x_min_main[slope_lines == 1]
        x_max_main_ver = x_max_main[slope_lines == 1]
        cx_main_ver = cx_main[slope_lines == 1]
        dist_y_ver = y_max_main_ver - y_min_main_ver
        len_y = seperators_closeup.shape[0] / 3.0

        # plt.imshow(horizontal)
        # plt.show()

        horizontal = np.repeat(horizontal[:, :, np.newaxis], 3, axis=2)
        horizontal = horizontal.astype(np.uint8)
        imgray = cv2.cvtColor(horizontal, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours_line_hors, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        slope_lines, dist_x, x_min_main, x_max_main, cy_main, slope_lines_org, y_min_main, y_max_main, cx_main = find_features_of_lines(contours_line_hors)

        slope_lines_org_hor = slope_lines_org[slope_lines == 0]
        args = np.array(range(len(slope_lines)))
        len_x = seperators_closeup.shape[1] / 5.0

        dist_y = np.abs(y_max_main - y_min_main)

        args_hor = args[slope_lines == 0]
        dist_x_hor = dist_x[slope_lines == 0]
        y_min_main_hor = y_min_main[slope_lines == 0]
        y_max_main_hor = y_max_main[slope_lines == 0]
        x_min_main_hor = x_min_main[slope_lines == 0]
        x_max_main_hor = x_max_main[slope_lines == 0]
        dist_y_hor = dist_y[slope_lines == 0]
        cy_main_hor = cy_main[slope_lines == 0]

        args_hor = args_hor[dist_x_hor >= len_x / 2.0]
        x_max_main_hor = x_max_main_hor[dist_x_hor >= len_x / 2.0]
        x_min_main_hor = x_min_main_hor[dist_x_hor >= len_x / 2.0]
        cy_main_hor = cy_main_hor[dist_x_hor >= len_x / 2.0]
        y_min_main_hor = y_min_main_hor[dist_x_hor >= len_x / 2.0]
        y_max_main_hor = y_max_main_hor[dist_x_hor >= len_x / 2.0]
        dist_y_hor = dist_y_hor[dist_x_hor >= len_x / 2.0]

        slope_lines_org_hor = slope_lines_org_hor[dist_x_hor >= len_x / 2.0]
        dist_x_hor = dist_x_hor[dist_x_hor >= len_x / 2.0]

        matrix_of_lines_ch = np.zeros((len(cy_main_hor) + len(cx_main_ver), 10))

        matrix_of_lines_ch[: len(cy_main_hor), 0] = args_hor
        matrix_of_lines_ch[len(cy_main_hor) :, 0] = args_ver

        matrix_of_lines_ch[len(cy_main_hor) :, 1] = cx_main_ver

        matrix_of_lines_ch[: len(cy_main_hor), 2] = x_min_main_hor + 50  # x_min_main_hor+150
        matrix_of_lines_ch[len(cy_main_hor) :, 2] = x_min_main_ver

        matrix_of_lines_ch[: len(cy_main_hor), 3] = x_max_main_hor - 50  # x_max_main_hor-150
        matrix_of_lines_ch[len(cy_main_hor) :, 3] = x_max_main_ver

        matrix_of_lines_ch[: len(cy_main_hor), 4] = dist_x_hor
        matrix_of_lines_ch[len(cy_main_hor) :, 4] = dist_x_ver

        matrix_of_lines_ch[: len(cy_main_hor), 5] = cy_main_hor

        matrix_of_lines_ch[: len(cy_main_hor), 6] = y_min_main_hor
        matrix_of_lines_ch[len(cy_main_hor) :, 6] = y_min_main_ver

        matrix_of_lines_ch[: len(cy_main_hor), 7] = y_max_main_hor
        matrix_of_lines_ch[len(cy_main_hor) :, 7] = y_max_main_ver

        matrix_of_lines_ch[: len(cy_main_hor), 8] = dist_y_hor
        matrix_of_lines_ch[len(cy_main_hor) :, 8] = dist_y_ver

        matrix_of_lines_ch[len(cy_main_hor) :, 9] = 1

        if contours_h is not None:
            slope_lines_head, dist_x_head, x_min_main_head, x_max_main_head, cy_main_head, slope_lines_org_head, y_min_main_head, y_max_main_head, cx_main_head = find_features_of_lines(contours_h)
            matrix_l_n = np.zeros((matrix_of_lines_ch.shape[0] + len(cy_main_head), matrix_of_lines_ch.shape[1]))
            matrix_l_n[: matrix_of_lines_ch.shape[0], :] = np.copy(matrix_of_lines_ch[:, :])
            args_head = np.array(range(len(cy_main_head))) + len(cy_main_hor)

            matrix_l_n[matrix_of_lines_ch.shape[0] :, 0] = args_head
            matrix_l_n[matrix_of_lines_ch.shape[0] :, 2] = x_min_main_head + 30
            matrix_l_n[matrix_of_lines_ch.shape[0] :, 3] = x_max_main_head - 30

            matrix_l_n[matrix_of_lines_ch.shape[0] :, 4] = dist_x_head

            matrix_l_n[matrix_of_lines_ch.shape[0] :, 5] = y_min_main_head - 3 - 8
            matrix_l_n[matrix_of_lines_ch.shape[0] :, 6] = y_min_main_head - 5 - 8
            matrix_l_n[matrix_of_lines_ch.shape[0] :, 7] = y_min_main_head + 1 - 8
            matrix_l_n[matrix_of_lines_ch.shape[0] :, 8] = 4

            matrix_of_lines_ch = np.copy(matrix_l_n)

        # print(matrix_of_lines_ch)

        """



        seperators_closeup=seperators_closeup.astype(np.uint8)
        imgray = cv2.cvtColor(seperators_closeup, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours_lines,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        slope_lines,dist_x, x_min_main ,x_max_main ,cy_main,slope_lines_org,y_min_main, y_max_main, cx_main=find_features_of_lines(contours_lines)

        slope_lines_org_hor=slope_lines_org[slope_lines==0]
        args=np.array( range(len(slope_lines) ))
        len_x=seperators_closeup.shape[1]/4.0

        args_hor=args[slope_lines==0]
        dist_x_hor=dist_x[slope_lines==0]
        x_min_main_hor=x_min_main[slope_lines==0]
        x_max_main_hor=x_max_main[slope_lines==0]
        cy_main_hor=cy_main[slope_lines==0]

        args_hor=args_hor[dist_x_hor>=len_x/2.0]
        x_max_main_hor=x_max_main_hor[dist_x_hor>=len_x/2.0]
        x_min_main_hor=x_min_main_hor[dist_x_hor>=len_x/2.0]
        cy_main_hor=cy_main_hor[dist_x_hor>=len_x/2.0]
        slope_lines_org_hor=slope_lines_org_hor[dist_x_hor>=len_x/2.0]


        slope_lines_org_hor=slope_lines_org_hor[np.abs(slope_lines_org_hor)<1.2]
        slope_mean_hor=np.mean(slope_lines_org_hor)



        args_ver=args[slope_lines==1]
        y_min_main_ver=y_min_main[slope_lines==1]
        y_max_main_ver=y_max_main[slope_lines==1]
        x_min_main_ver=x_min_main[slope_lines==1]
        x_max_main_ver=x_max_main[slope_lines==1]
        cx_main_ver=cx_main[slope_lines==1]
        dist_y_ver=y_max_main_ver-y_min_main_ver
        len_y=seperators_closeup.shape[0]/3.0



        print(matrix_of_lines_ch[:,8][matrix_of_lines_ch[:,9]==0],'khatlarrrr')
        args_main_spliters=matrix_of_lines_ch[:,0][ (matrix_of_lines_ch[:,9]==0) & ((matrix_of_lines_ch[:,8]<=290)) & ((matrix_of_lines_ch[:,2]<=.16*region_pre_p.shape[1])) & ((matrix_of_lines_ch[:,3]>=.84*region_pre_p.shape[1]))]

        cy_main_spliters=matrix_of_lines_ch[:,5][ (matrix_of_lines_ch[:,9]==0) & ((matrix_of_lines_ch[:,8]<=290)) & ((matrix_of_lines_ch[:,2]<=.16*region_pre_p.shape[1])) & ((matrix_of_lines_ch[:,3]>=.84*region_pre_p.shape[1]))]
        """

        cy_main_spliters = cy_main_hor[(x_min_main_hor <= 0.16 * region_pre_p.shape[1]) & (x_max_main_hor >= 0.84 * region_pre_p.shape[1])]

        cy_main_spliters = np.array(list(cy_main_spliters) + list(special_seperators))

        if contours_h is not None:
            try:
                cy_main_spliters_head = cy_main_head[(x_min_main_head <= 0.16 * region_pre_p.shape[1]) & (x_max_main_head >= 0.84 * region_pre_p.shape[1])]
                cy_main_spliters = np.array(list(cy_main_spliters) + list(cy_main_spliters_head))
            except:
                pass
        args_cy_spliter = np.argsort(cy_main_spliters)

        cy_main_spliters_sort = cy_main_spliters[args_cy_spliter]

        spliter_y_new = []
        spliter_y_new.append(0)
        for i in range(len(cy_main_spliters_sort)):
            spliter_y_new.append(cy_main_spliters_sort[i])

        spliter_y_new.append(region_pre_p.shape[0])

        spliter_y_new_diff = np.diff(spliter_y_new) / float(region_pre_p.shape[0]) * 100

        args_big_parts = np.array(range(len(spliter_y_new_diff)))[spliter_y_new_diff > 22]

        regions_without_seperators = self.return_regions_without_seperators(region_pre_p)

        ##print(args_big_parts,'args_big_parts')
        # image_page_otsu=otsu_copy(image_page_deskewd)
        # print(np.unique(image_page_otsu[:,:,0]))
        # image_page_background_zero=self.image_change_background_pixels_to_zero(image_page_otsu)

        length_y_threshold = regions_without_seperators.shape[0] / 4.0

        num_col_fin = 0
        peaks_neg_fin_fin = []

        for iteils in args_big_parts:

            regions_without_seperators_teil = regions_without_seperators[int(spliter_y_new[iteils]) : int(spliter_y_new[iteils + 1]), :, 0]
            # image_page_background_zero_teil=image_page_background_zero[int(spliter_y_new[iteils]):int(spliter_y_new[iteils+1]),:]

            # print(regions_without_seperators_teil.shape)
            ##plt.imshow(regions_without_seperators_teil)
            ##plt.show()

            # num_col, peaks_neg_fin=self.find_num_col(regions_without_seperators_teil,multiplier=6.0)

            # regions_without_seperators_teil=cv2.erode(regions_without_seperators_teil,kernel,iterations = 3)
            #
            num_col, peaks_neg_fin = self.find_num_col(regions_without_seperators_teil, multiplier=7.0)

            if num_col > num_col_fin:
                num_col_fin = num_col
                peaks_neg_fin_fin = peaks_neg_fin
            """
            #print(length_y_vertical_lines,length_y_threshold,'x_center_of_ver_linesx_center_of_ver_linesx_center_of_ver_lines')
            if len(cx_main_ver)>0 and len( dist_y_ver[dist_y_ver>=length_y_threshold] ) >=1:
                num_col, peaks_neg_fin=self.find_num_col(regions_without_seperators_teil,multiplier=6.0)
            else:
                #plt.imshow(image_page_background_zero_teil)
                #plt.show()
                #num_col, peaks_neg_fin=self.find_num_col_only_image(image_page_background_zero,multiplier=2.4)#2.3)
                num_col, peaks_neg_fin=self.find_num_col_only_image(image_page_background_zero_teil,multiplier=3.4)#2.3)

                print(num_col,'birda')
                if num_col>0:
                    pass
                elif num_col==0:
                    print(num_col,'birda2222')
                    num_col_regions, peaks_neg_fin_regions=self.find_num_col(regions_without_seperators_teil,multiplier=10.0)
                    if num_col_regions==0:
                        pass
                    else:

                        num_col=num_col_regions
                        peaks_neg_fin=peaks_neg_fin_regions[:]
            """

            # print(num_col+1,'num colmsssssssss')

        if len(args_big_parts) == 1 and (len(peaks_neg_fin_fin) + 1) < num_col_classifier:
            peaks_neg_fin = self.find_num_col_by_vertical_lines(vertical)
            peaks_neg_fin = peaks_neg_fin[peaks_neg_fin >= 500]
            peaks_neg_fin = peaks_neg_fin[peaks_neg_fin <= (vertical.shape[1] - 500)]
            peaks_neg_fin_fin = peaks_neg_fin[:]

            # print(peaks_neg_fin_fin,'peaks_neg_fin_fintaza')

        return num_col_fin, peaks_neg_fin_fin, matrix_of_lines_ch, spliter_y_new, seperators_closeup_n

    def return_contours_of_interested_region_by_size(self, region_pre_p, pixel, min_area, max_area):

        # pixels of images are identified by 5
        if len(region_pre_p.shape) == 3:
            cnts_images = (region_pre_p[:, :, 0] == pixel) * 1
        else:
            cnts_images = (region_pre_p[:, :] == pixel) * 1
        cnts_images = cnts_images.astype(np.uint8)
        cnts_images = np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
        imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)
        contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_imgs = return_parent_contours(contours_imgs, hiearchy)
        contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=max_area, min_area=min_area)

        img_ret = np.zeros((region_pre_p.shape[0], region_pre_p.shape[1], 3))
        img_ret = cv2.fillPoly(img_ret, pts=contours_imgs, color=(1, 1, 1))
        return img_ret[:, :, 0]

    def get_regions_from_xy_neu(self, img):
        img_org = np.copy(img)

        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]

        model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p)

        gaussian_filter = False
        patches = True
        binary = True

        ratio_x = 1
        ratio_y = 1
        median_blur = False

        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)
        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)
        prediction_regions_org = self.do_prediction(patches, img, model_region)

        prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h)

        # plt.imshow(prediction_regions_org[:,:,0])
        # plt.show()
        # sys.exit()
        prediction_regions_org = prediction_regions_org[:, :, 0]

        gaussian_filter = False
        patches = False
        binary = False

        ratio_x = 1
        ratio_y = 1
        median_blur = False

        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)
            img = cv2.medianBlur(img, 5)
        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)
        prediction_regions_orgt = self.do_prediction(patches, img, model_region)

        prediction_regions_orgt = resize_image(prediction_regions_orgt, img_height_h, img_width_h)

        # plt.imshow(prediction_regions_orgt[:,:,0])
        # plt.show()
        # sys.exit()
        prediction_regions_orgt = prediction_regions_orgt[:, :, 0]

        mask_texts_longshot = (prediction_regions_orgt[:, :] == 1) * 1

        mask_texts_longshot = np.uint8(mask_texts_longshot)
        # mask_texts_longshot = cv2.dilate(mask_texts_longshot[:,:], self.kernel, iterations=2)

        pixel_img = 1
        polygons_of_only_texts_longshot = return_contours_of_interested_region(mask_texts_longshot, pixel_img)

        longshot_true = np.zeros(mask_texts_longshot.shape)
        # text_regions_p_true[:,:]=text_regions_p_1[:,:]

        longshot_true = cv2.fillPoly(longshot_true, pts=polygons_of_only_texts_longshot, color=(1, 1, 1))

        # plt.imshow(longshot_true)
        # plt.show()

        gaussian_filter = False
        patches = False
        binary = False

        ratio_x = 1
        ratio_y = 1
        median_blur = False

        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

        one_third_upper_ny = int(img.shape[0] / 3.0)

        img = img[0:one_third_upper_ny, :, :]

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)

        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)
        prediction_regions_longshot_one_third = self.do_prediction(patches, img, model_region)

        prediction_regions_longshot_one_third = resize_image(prediction_regions_longshot_one_third, one_third_upper_ny, img_width_h)

        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))
        img = img[one_third_upper_ny : int(2 * one_third_upper_ny), :, :]

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)

        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)
        prediction_regions_longshot_one_third_middle = self.do_prediction(patches, img, model_region)

        prediction_regions_longshot_one_third_middle = resize_image(prediction_regions_longshot_one_third_middle, one_third_upper_ny, img_width_h)

        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))
        img = img[int(2 * one_third_upper_ny) :, :, :]

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)

        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)
        prediction_regions_longshot_one_third_down = self.do_prediction(patches, img, model_region)

        prediction_regions_longshot_one_third_down = resize_image(prediction_regions_longshot_one_third_down, img_height_h - int(2 * one_third_upper_ny), img_width_h)

        # plt.imshow(prediction_regions_org[:,:,0])
        # plt.show()
        # sys.exit()
        prediction_regions_longshot = np.zeros((img_height_h, img_width_h))

        # prediction_regions_longshot=prediction_regions_longshot[:,:,0]

        # prediction_regions_longshot[0:one_third_upper_ny,:]=prediction_regions_longshot_one_third[:,:,0]
        # prediction_regions_longshot[one_third_upper_ny:int(2*one_third_upper_ny):,:]=prediction_regions_longshot_one_third_middle[:,:,0]
        # prediction_regions_longshot[int(2*one_third_upper_ny):,:]=prediction_regions_longshot_one_third_down[:,:,0]

        prediction_regions_longshot = longshot_true[:, :]
        # plt.imshow(prediction_regions_longshot)
        # plt.show()

        gaussian_filter = False
        patches = True
        binary = False

        ratio_x = 1  # 1.1
        ratio_y = 1
        median_blur = False

        # img= resize_image(img_org, int(img_org.shape[0]*0.8), int(img_org.shape[1]*1.6))
        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)
        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)

        prediction_regions = self.do_prediction(patches, img, model_region)
        text_region1 = resize_image(prediction_regions, img_height_h, img_width_h)

        # plt.imshow(text_region1[:,:,0])
        # plt.show()
        ratio_x = 1
        ratio_y = 1.2  # 1.3
        binary = False
        median_blur = False

        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)
        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)

        prediction_regions = self.do_prediction(patches, img, model_region)
        text_region2 = resize_image(prediction_regions, img_height_h, img_width_h)

        # plt.imshow(text_region2[:,:,0])
        # plt.show()
        session_region.close()
        del model_region
        del session_region
        gc.collect()

        # text_region1=text_region1[:,:,0]
        # text_region2=text_region2[:,:,0]

        # text_region1[(text_region1[:,:]==2) & (text_region2[:,:]==1)]=1

        mask_zeros_from_1 = (text_region2[:, :, 0] == 0) * 1
        # mask_text_from_1=(text_region1[:,:,0]==1)*1

        mask_img_text_region1 = (text_region1[:, :, 0] == 2) * 1
        text_region2_1st_channel = text_region1[:, :, 0]

        text_region2_1st_channel[mask_zeros_from_1 == 1] = 0

        ##text_region2_1st_channel[mask_img_text_region1[:,:]==1]=2
        # text_region2_1st_channel[(mask_text_from_1==1) & (text_region2_1st_channel==2)]=1

        mask_lines1 = (text_region1[:, :, 0] == 3) * 1
        mask_lines2 = (text_region2[:, :, 0] == 3) * 1

        mask_lines2[mask_lines1[:, :] == 1] = 1

        # plt.imshow(text_region2_1st_channel)
        # plt.show()

        text_region2_1st_channel = cv2.erode(text_region2_1st_channel[:, :], self.kernel, iterations=4)

        # plt.imshow(text_region2_1st_channel)
        # plt.show()

        text_region2_1st_channel = cv2.dilate(text_region2_1st_channel[:, :], self.kernel, iterations=4)

        text_region2_1st_channel[mask_lines2[:, :] == 1] = 3

        # text_region2_1st_channel[ (prediction_regions_org[:,:]==1) & (text_region2_1st_channel[:,:]==2)]=1

        # only in the case of model 3

        text_region2_1st_channel[(prediction_regions_longshot[:, :] == 1) & (text_region2_1st_channel[:, :] == 2)] = 1

        text_region2_1st_channel[(prediction_regions_org[:, :] == 2) & (text_region2_1st_channel[:, :] == 0)] = 2

        # text_region2_1st_channel[prediction_regions_org[:,:]==0]=0

        # plt.imshow(text_region2_1st_channel)
        # plt.show()

        # text_region2_1st_channel[:,:400]=0

        mask_texts_only = (text_region2_1st_channel[:, :] == 1) * 1

        mask_images_only = (text_region2_1st_channel[:, :] == 2) * 1

        mask_lines_only = (text_region2_1st_channel[:, :] == 3) * 1

        pixel_img = 1
        polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only, pixel_img)

        polygons_of_only_images = return_contours_of_interested_region(mask_images_only, pixel_img)

        polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only, pixel_img)

        text_regions_p_true = np.zeros(text_region2_1st_channel.shape)
        # text_regions_p_true[:,:]=text_regions_p_1[:,:]

        text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts=polygons_of_only_lines, color=(3, 3, 3))

        text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts=polygons_of_only_images, color=(2, 2, 2))

        text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts=polygons_of_only_texts, color=(1, 1, 1))

        ##print(np.unique(text_regions_p_true))

        # text_regions_p_true_3d=np.repeat(text_regions_p_1[:, :, np.newaxis], 3, axis=2)
        # text_regions_p_true_3d=text_regions_p_true_3d.astype(np.uint8)

        return text_regions_p_true  # text_region2_1st_channel

    def get_regions_from_xy(self, img):
        img_org = np.copy(img)

        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]

        model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p)

        gaussian_filter = False
        patches = True
        binary = True

        ratio_x = 1
        ratio_y = 1
        median_blur = False

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)

        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)
        prediction_regions_org = self.do_prediction(patches, img, model_region)

        ###plt.imshow(prediction_regions_org[:,:,0])
        ###plt.show()
        ##sys.exit()
        prediction_regions_org = prediction_regions_org[:, :, 0]

        gaussian_filter = False
        patches = True
        binary = False

        ratio_x = 1.1
        ratio_y = 1
        median_blur = False

        # img= resize_image(img_org, int(img_org.shape[0]*0.8), int(img_org.shape[1]*1.6))
        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)
        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)

        prediction_regions = self.do_prediction(patches, img, model_region)
        text_region1 = resize_image(prediction_regions, img_height_h, img_width_h)

        ratio_x = 1
        ratio_y = 1.1
        binary = False
        median_blur = False

        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)
        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)

        prediction_regions = self.do_prediction(patches, img, model_region)
        text_region2 = resize_image(prediction_regions, img_height_h, img_width_h)

        session_region.close()
        del model_region
        del session_region
        gc.collect()

        mask_zeros_from_1 = (text_region1[:, :, 0] == 0) * 1
        # mask_text_from_1=(text_region1[:,:,0]==1)*1

        mask_img_text_region1 = (text_region1[:, :, 0] == 2) * 1
        text_region2_1st_channel = text_region2[:, :, 0]

        text_region2_1st_channel[mask_zeros_from_1 == 1] = 0

        text_region2_1st_channel[mask_img_text_region1[:, :] == 1] = 2
        # text_region2_1st_channel[(mask_text_from_1==1) & (text_region2_1st_channel==2)]=1

        mask_lines1 = (text_region1[:, :, 0] == 3) * 1
        mask_lines2 = (text_region2[:, :, 0] == 3) * 1

        mask_lines2[mask_lines1[:, :] == 1] = 1

        ##plt.imshow(text_region2_1st_channel)
        ##plt.show()

        text_region2_1st_channel = cv2.erode(text_region2_1st_channel[:, :], self.kernel, iterations=5)

        ##plt.imshow(text_region2_1st_channel)
        ##plt.show()

        text_region2_1st_channel = cv2.dilate(text_region2_1st_channel[:, :], self.kernel, iterations=5)

        text_region2_1st_channel[mask_lines2[:, :] == 1] = 3

        text_region2_1st_channel[(prediction_regions_org[:, :] == 1) & (text_region2_1st_channel[:, :] == 2)] = 1
        text_region2_1st_channel[prediction_regions_org[:, :] == 3] = 3

        ##plt.imshow(text_region2_1st_channel)
        ##plt.show()
        return text_region2_1st_channel

    def rotation_not_90_func(self, img, textline, text_regions_p_1, thetha):
        rotated = imutils.rotate(img, thetha)
        rotated_textline = imutils.rotate(textline, thetha)
        rotated_layout = imutils.rotate(text_regions_p_1, thetha)
        return self.rotate_max_area(img, rotated, rotated_textline, rotated_layout, thetha)

    def rotate_max_area(self, image, rotated, rotated_textline, rotated_layout, angle):
        wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0], math.radians(angle))
        h, w, _ = rotated.shape
        y1 = h // 2 - int(hr / 2)
        y2 = y1 + int(hr)
        x1 = w // 2 - int(wr / 2)
        x2 = x1 + int(wr)
        return rotated[y1:y2, x1:x2], rotated_textline[y1:y2, x1:x2], rotated_layout[y1:y2, x1:x2]

    def rotation_not_90_func_full_layout(self, img, textline, text_regions_p_1, text_regions_p_fully, thetha):
        rotated = imutils.rotate(img, thetha)
        rotated_textline = imutils.rotate(textline, thetha)
        rotated_layout = imutils.rotate(text_regions_p_1, thetha)
        rotated_layout_full = imutils.rotate(text_regions_p_fully, thetha)
        return self.rotate_max_area_full_layout(img, rotated, rotated_textline, rotated_layout, rotated_layout_full, thetha)

    def rotate_max_area_full_layout(self, image, rotated, rotated_textline, rotated_layout, rotated_layout_full, angle):
        wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0], math.radians(angle))
        h, w, _ = rotated.shape
        y1 = h // 2 - int(hr / 2)
        y2 = y1 + int(hr)
        x1 = w // 2 - int(wr / 2)
        x2 = x1 + int(wr)
        return rotated[y1:y2, x1:x2], rotated_textline[y1:y2, x1:x2], rotated_layout[y1:y2, x1:x2], rotated_layout_full[y1:y2, x1:x2]

    def get_regions_from_xy_2models_ens(self, img):
        img_org = np.copy(img)

        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]

        model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_ens)

        gaussian_filter = False
        patches = False
        binary = False

        ratio_x = 1
        ratio_y = 1
        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

        prediction_regions_long = self.do_prediction(patches, img, model_region)

        prediction_regions_long = resize_image(prediction_regions_long, img_height_h, img_width_h)

        gaussian_filter = False
        patches = True
        binary = False

        ratio_x = 1
        ratio_y = 1.2
        median_blur = False

        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)
        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)
        prediction_regions_org_y = self.do_prediction(patches, img, model_region)

        prediction_regions_org_y = resize_image(prediction_regions_org_y, img_height_h, img_width_h)

        # plt.imshow(prediction_regions_org[:,:,0])
        # plt.show()
        # sys.exit()
        prediction_regions_org_y = prediction_regions_org_y[:, :, 0]

        mask_zeros_y = (prediction_regions_org_y[:, :] == 0) * 1

        ratio_x = 1.2
        ratio_y = 1
        median_blur = False

        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)
        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)
        prediction_regions_org = self.do_prediction(patches, img, model_region)

        prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h)

        # plt.imshow(prediction_regions_org[:,:,0])
        # plt.show()
        # sys.exit()
        prediction_regions_org = prediction_regions_org[:, :, 0]

        prediction_regions_org[(prediction_regions_org[:, :] == 1) & (mask_zeros_y[:, :] == 1)] = 0

        prediction_regions_org[(prediction_regions_long[:, :, 0] == 1) & (prediction_regions_org[:, :] == 2)] = 1

        session_region.close()
        del model_region
        del session_region
        gc.collect()

        return prediction_regions_org

    def get_regions_from_xy_2models(self, img, is_image_enhanced):
        img_org = np.copy(img)

        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]

        model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_ens)

        gaussian_filter = False
        patches = True
        binary = False

        ratio_y = 1.3
        ratio_x = 1

        median_blur = False

        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)
        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)
        prediction_regions_org_y = self.do_prediction(patches, img, model_region)

        prediction_regions_org_y = resize_image(prediction_regions_org_y, img_height_h, img_width_h)

        # plt.imshow(prediction_regions_org_y[:,:,0])
        # plt.show()
        # sys.exit()
        prediction_regions_org_y = prediction_regions_org_y[:, :, 0]

        mask_zeros_y = (prediction_regions_org_y[:, :] == 0) * 1

        if is_image_enhanced:
            ratio_x = 1.2
        else:
            ratio_x = 1

        ratio_y = 1
        median_blur = False

        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)
        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)
        prediction_regions_org = self.do_prediction(patches, img, model_region)

        prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h)

        ##plt.imshow(prediction_regions_org[:,:,0])
        ##plt.show()
        ##sys.exit()
        prediction_regions_org = prediction_regions_org[:, :, 0]

        prediction_regions_org[(prediction_regions_org[:, :] == 1) & (mask_zeros_y[:, :] == 1)] = 0
        session_region.close()
        del model_region
        del session_region
        gc.collect()
        ###K.clear_session()

        model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p2)

        gaussian_filter = False
        patches = True
        binary = False

        ratio_x = 1
        ratio_y = 1
        median_blur = False

        img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

        if binary:
            img = otsu_copy_binary(img)  # otsu_copy(img)
            img = img.astype(np.uint16)

        if median_blur:
            img = cv2.medianBlur(img, 5)
        if gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = img.astype(np.uint16)
        prediction_regions_org2 = self.do_prediction(patches, img, model_region)

        prediction_regions_org2 = resize_image(prediction_regions_org2, img_height_h, img_width_h)

        # plt.imshow(prediction_regions_org2[:,:,0])
        # plt.show()
        # sys.exit()
        ##prediction_regions_org=prediction_regions_org[:,:,0]

        session_region.close()
        del model_region
        del session_region
        gc.collect()
        ###K.clear_session()

        mask_zeros2 = (prediction_regions_org2[:, :, 0] == 0) * 1
        mask_lines2 = (prediction_regions_org2[:, :, 0] == 3) * 1

        text_sume_early = ((prediction_regions_org[:, :] == 1) * 1).sum()

        prediction_regions_org_copy = np.copy(prediction_regions_org)

        prediction_regions_org_copy[(prediction_regions_org_copy[:, :] == 1) & (mask_zeros2[:, :] == 1)] = 0

        text_sume_second = ((prediction_regions_org_copy[:, :] == 1) * 1).sum()

        rate_two_models = text_sume_second / float(text_sume_early) * 100

        print(rate_two_models, "ratio_of_two_models")
        if is_image_enhanced and rate_two_models < 95.50:  # 98.45:
            pass
        else:
            prediction_regions_org = np.copy(prediction_regions_org_copy)

        ##prediction_regions_org[mask_lines2[:,:]==1]=3
        prediction_regions_org[(mask_lines2[:, :] == 1) & (prediction_regions_org[:, :] == 0)] = 3

        del mask_lines2
        del mask_zeros2
        del prediction_regions_org2

        # if is_image_enhanced:
        # pass
        # else:
        # model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p2)

        # gaussian_filter=False
        # patches=True
        # binary=False

        # ratio_x=1
        # ratio_y=1
        # median_blur=False

        # img= resize_image(img_org, int(img_org.shape[0]*ratio_y), int(img_org.shape[1]*ratio_x))

        # if binary:
        # img = otsu_copy_binary(img)#otsu_copy(img)
        # img = img.astype(np.uint16)

        # if median_blur:
        # img=cv2.medianBlur(img,5)
        # if gaussian_filter:
        # img= cv2.GaussianBlur(img,(5,5),0)
        # img = img.astype(np.uint16)
        # prediction_regions_org2=self.do_prediction(patches,img,model_region)

        # prediction_regions_org2=resize_image(prediction_regions_org2, img_height_h, img_width_h )

        ##plt.imshow(prediction_regions_org2[:,:,0])
        ##plt.show()
        ##sys.exit()
        ###prediction_regions_org=prediction_regions_org[:,:,0]

        # session_region.close()
        # del model_region
        # del session_region
        # gc.collect()
        ####K.clear_session()

        # mask_zeros2=(prediction_regions_org2[:,:,0]==0)*1
        # mask_lines2=(prediction_regions_org2[:,:,0]==3)*1

        # text_sume_early=( (prediction_regions_org[:,:]==1)*1 ).sum()

        # prediction_regions_org[(prediction_regions_org[:,:]==1) & (mask_zeros2[:,:]==1)]=0

        ###prediction_regions_org[mask_lines2[:,:]==1]=3
        # prediction_regions_org[(mask_lines2[:,:]==1) & (prediction_regions_org[:,:]==0)]=3

        # text_sume_second=( (prediction_regions_org[:,:]==1)*1 ).sum()

        # print(text_sume_second/float(text_sume_early)*100,'twomodelsratio')

        # del mask_lines2
        # del mask_zeros2
        # del prediction_regions_org2

        mask_lines_only = (prediction_regions_org[:, :] == 3) * 1

        prediction_regions_org = cv2.erode(prediction_regions_org[:, :], self.kernel, iterations=2)

        # plt.imshow(text_region2_1st_channel)
        # plt.show()

        prediction_regions_org = cv2.dilate(prediction_regions_org[:, :], self.kernel, iterations=2)

        mask_texts_only = (prediction_regions_org[:, :] == 1) * 1

        mask_images_only = (prediction_regions_org[:, :] == 2) * 1

        pixel_img = 1
        min_area_text = 0.00001
        polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only, pixel_img, min_area_text)

        polygons_of_only_images = return_contours_of_interested_region(mask_images_only, pixel_img)

        polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only, pixel_img, min_area_text)

        text_regions_p_true = np.zeros(prediction_regions_org.shape)
        # text_regions_p_true[:,:]=text_regions_p_1[:,:]

        text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts=polygons_of_only_lines, color=(3, 3, 3))

        ##text_regions_p_true=cv2.fillPoly(text_regions_p_true,pts=polygons_of_only_images, color=(2,2,2))
        text_regions_p_true[:, :][mask_images_only[:, :] == 1] = 2

        text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts=polygons_of_only_texts, color=(1, 1, 1))

        ##print(np.unique(text_regions_p_true))

        # text_regions_p_true_3d=np.repeat(text_regions_p_1[:, :, np.newaxis], 3, axis=2)
        # text_regions_p_true_3d=text_regions_p_true_3d.astype(np.uint8)

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

        return text_regions_p_true

    def write_images_into_directory(self, img_contoures, dir_of_cropped_imgs, image_page):
        index = 0
        for cont_ind in img_contoures:
            # cont_ind[:,0,0]=cont_ind[:,0,0]/self.scale_x
            # cont_ind[:,0,1]=cont_ind[:,0,1]/self.scale_y

            x, y, w, h = cv2.boundingRect(cont_ind)
            box = [x, y, w, h]
            croped_page, page_coord = crop_image_inside_box(box, image_page)

            croped_page = resize_image(croped_page, int(croped_page.shape[0] / self.scale_y), int(croped_page.shape[1] / self.scale_x))

            path = os.path.join(dir_of_cropped_imgs, self.f_name + "_" + str(index) + ".jpg")
            cv2.imwrite(path, croped_page)
            index += 1

    def get_marginals(self, text_with_lines, text_regions, num_col, slope_deskew):
        mask_marginals = np.zeros((text_with_lines.shape[0], text_with_lines.shape[1]))
        mask_marginals = mask_marginals.astype(np.uint8)

        text_with_lines = text_with_lines.astype(np.uint8)
        ##text_with_lines=cv2.erode(text_with_lines,self.kernel,iterations=3)

        text_with_lines_eroded = cv2.erode(text_with_lines, self.kernel, iterations=5)

        if text_with_lines.shape[0] <= 1500:
            pass
        elif text_with_lines.shape[0] > 1500 and text_with_lines.shape[0] <= 1800:
            text_with_lines = resize_image(text_with_lines, int(text_with_lines.shape[0] * 1.5), text_with_lines.shape[1])
            text_with_lines = cv2.erode(text_with_lines, self.kernel, iterations=5)
            text_with_lines = resize_image(text_with_lines, text_with_lines_eroded.shape[0], text_with_lines_eroded.shape[1])
        else:
            text_with_lines = resize_image(text_with_lines, int(text_with_lines.shape[0] * 1.8), text_with_lines.shape[1])
            text_with_lines = cv2.erode(text_with_lines, self.kernel, iterations=7)
            text_with_lines = resize_image(text_with_lines, text_with_lines_eroded.shape[0], text_with_lines_eroded.shape[1])

        text_with_lines_y = text_with_lines.sum(axis=0)
        text_with_lines_y_eroded = text_with_lines_eroded.sum(axis=0)

        thickness_along_y_percent = text_with_lines_y_eroded.max() / (float(text_with_lines.shape[0])) * 100

        # print(thickness_along_y_percent,'thickness_along_y_percent')

        if thickness_along_y_percent < 30:
            min_textline_thickness = 8
        elif thickness_along_y_percent >= 30 and thickness_along_y_percent < 50:
            min_textline_thickness = 20
        else:
            min_textline_thickness = 40

        if thickness_along_y_percent >= 14:

            text_with_lines_y_rev = -1 * text_with_lines_y[:]
            # print(text_with_lines_y)
            # print(text_with_lines_y_rev)

            # plt.plot(text_with_lines_y)
            # plt.show()

            text_with_lines_y_rev = text_with_lines_y_rev - np.min(text_with_lines_y_rev)

            # plt.plot(text_with_lines_y_rev)
            # plt.show()
            sigma_gaus = 1
            region_sum_0 = gaussian_filter1d(text_with_lines_y, sigma_gaus)

            region_sum_0_rev = gaussian_filter1d(text_with_lines_y_rev, sigma_gaus)

            # plt.plot(region_sum_0_rev)
            # plt.show()
            region_sum_0_updown = region_sum_0[len(region_sum_0) :: -1]

            first_nonzero = next((i for i, x in enumerate(region_sum_0) if x), None)
            last_nonzero = next((i for i, x in enumerate(region_sum_0_updown) if x), None)

            last_nonzero = len(region_sum_0) - last_nonzero

            ##img_sum_0_smooth_rev=-region_sum_0

            mid_point = (last_nonzero + first_nonzero) / 2.0

            one_third_right = (last_nonzero - mid_point) / 3.0
            one_third_left = (mid_point - first_nonzero) / 3.0

            # img_sum_0_smooth_rev=img_sum_0_smooth_rev-np.min(img_sum_0_smooth_rev)

            peaks, _ = find_peaks(text_with_lines_y_rev, height=0)

            peaks = np.array(peaks)

            # print(region_sum_0[peaks])
            ##plt.plot(region_sum_0)
            ##plt.plot(peaks,region_sum_0[peaks],'*')
            ##plt.show()
            # print(first_nonzero,last_nonzero,peaks)
            peaks = peaks[(peaks > first_nonzero) & ((peaks < last_nonzero))]

            # print(first_nonzero,last_nonzero,peaks)

            # print(region_sum_0[peaks]<10)
            ####peaks=peaks[region_sum_0[peaks]<25 ]

            # print(region_sum_0[peaks])
            peaks = peaks[region_sum_0[peaks] < min_textline_thickness]
            # print(peaks)
            # print(first_nonzero,last_nonzero,one_third_right,one_third_left)

            if num_col == 1:
                peaks_right = peaks[peaks > mid_point]
                peaks_left = peaks[peaks < mid_point]
            if num_col == 2:
                peaks_right = peaks[peaks > (mid_point + one_third_right)]
                peaks_left = peaks[peaks < (mid_point - one_third_left)]

            try:
                point_right = np.min(peaks_right)
            except:
                point_right = last_nonzero

            try:
                point_left = np.max(peaks_left)
            except:
                point_left = first_nonzero

            # print(point_left,point_right)
            # print(text_regions.shape)
            if point_right >= mask_marginals.shape[1]:
                point_right = mask_marginals.shape[1] - 1

            try:
                mask_marginals[:, point_left:point_right] = 1
            except:
                mask_marginals[:, :] = 1

            # print(mask_marginals.shape,point_left,point_right,'nadosh')
            mask_marginals_rotated = rotate_image(mask_marginals, -slope_deskew)

            # print(mask_marginals_rotated.shape,'nadosh')
            mask_marginals_rotated_sum = mask_marginals_rotated.sum(axis=0)

            mask_marginals_rotated_sum[mask_marginals_rotated_sum != 0] = 1
            index_x = np.array(range(len(mask_marginals_rotated_sum))) + 1

            index_x_interest = index_x[mask_marginals_rotated_sum == 1]

            min_point_of_left_marginal = np.min(index_x_interest) - 16
            max_point_of_right_marginal = np.max(index_x_interest) + 16

            if min_point_of_left_marginal < 0:
                min_point_of_left_marginal = 0
            if max_point_of_right_marginal >= text_regions.shape[1]:
                max_point_of_right_marginal = text_regions.shape[1] - 1

            # print(np.min(index_x_interest) ,np.max(index_x_interest),'minmaxnew')
            # print(mask_marginals_rotated.shape,text_regions.shape,'mask_marginals_rotated')
            # plt.imshow(mask_marginals)
            # plt.show()

            # plt.imshow(mask_marginals_rotated)
            # plt.show()

            text_regions[(mask_marginals_rotated[:, :] != 1) & (text_regions[:, :] == 1)] = 4

            pixel_img = 4
            min_area_text = 0.00001
            polygons_of_marginals = return_contours_of_interested_region(text_regions, pixel_img, min_area_text)

            cx_text_only, cy_text_only, x_min_text_only, x_max_text_only, y_min_text_only, y_max_text_only, y_cor_x_min_main = find_new_features_of_contoures(polygons_of_marginals)

            text_regions[(text_regions[:, :] == 4)] = 1

            marginlas_should_be_main_text = []

            x_min_marginals_left = []
            x_min_marginals_right = []

            for i in range(len(cx_text_only)):

                x_width_mar = abs(x_min_text_only[i] - x_max_text_only[i])
                y_height_mar = abs(y_min_text_only[i] - y_max_text_only[i])
                # print(x_width_mar,y_height_mar,'y_height_mar')
                if x_width_mar > 16 and y_height_mar / x_width_mar < 10:
                    marginlas_should_be_main_text.append(polygons_of_marginals[i])
                    if x_min_text_only[i] < (mid_point - one_third_left):
                        x_min_marginals_left_new = x_min_text_only[i]
                        if len(x_min_marginals_left) == 0:
                            x_min_marginals_left.append(x_min_marginals_left_new)
                        else:
                            x_min_marginals_left[0] = min(x_min_marginals_left[0], x_min_marginals_left_new)
                    else:
                        x_min_marginals_right_new = x_min_text_only[i]
                        if len(x_min_marginals_right) == 0:
                            x_min_marginals_right.append(x_min_marginals_right_new)
                        else:
                            x_min_marginals_right[0] = min(x_min_marginals_right[0], x_min_marginals_right_new)

            if len(x_min_marginals_left) == 0:
                x_min_marginals_left = [0]
            if len(x_min_marginals_right) == 0:
                x_min_marginals_right = [text_regions.shape[1] - 1]

            # print(x_min_marginals_left[0],x_min_marginals_right[0],'margo')

            # print(marginlas_should_be_main_text,'marginlas_should_be_main_text')
            text_regions = cv2.fillPoly(text_regions, pts=marginlas_should_be_main_text, color=(4, 4))

            # print(np.unique(text_regions))

            # text_regions[:,:int(x_min_marginals_left[0])][text_regions[:,:int(x_min_marginals_left[0])]==1]=0
            # text_regions[:,int(x_min_marginals_right[0]):][text_regions[:,int(x_min_marginals_right[0]):]==1]=0

            text_regions[:, : int(min_point_of_left_marginal)][text_regions[:, : int(min_point_of_left_marginal)] == 1] = 0
            text_regions[:, int(max_point_of_right_marginal) :][text_regions[:, int(max_point_of_right_marginal) :] == 1] = 0

            ###text_regions[:,0:point_left][text_regions[:,0:point_left]==1]=4

            ###text_regions[:,point_right:][ text_regions[:,point_right:]==1]=4
            # plt.plot(region_sum_0)
            # plt.plot(peaks,region_sum_0[peaks],'*')
            # plt.show()

            # plt.imshow(text_regions)
            # plt.show()

            # sys.exit()
        else:
            pass
        return text_regions

    def do_work_of_textline_seperation(self, queue_of_all_params, polygons_per_process, index_polygons_per_process, con_par_org, textline_mask_tot, mask_texts_only, num_col, scale_par, boxes_text):

        textregions_cnt_tot_per_process = []
        textlines_cnt_tot_per_process = []
        index_polygons_per_process_per_process = []
        polygons_per_par_process_per_process = []
        textline_cnt_seperated = np.zeros(textline_mask_tot.shape)
        for iiii in range(len(polygons_per_process)):
            # crop_img,crop_coor=crop_image_inside_box(boxes_text[mv],image_page_rotated)
            # arg_max=np.argmax(areas_cnt_only_text)
            textregions_cnt_tot_per_process.append(polygons_per_process[iiii] / scale_par)
            textline_region_in_image = np.zeros(textline_mask_tot.shape)
            cnt_o_t_max = polygons_per_process[iiii]

            x, y, w, h = cv2.boundingRect(cnt_o_t_max)

            mask_biggest = np.zeros(mask_texts_only.shape)
            mask_biggest = cv2.fillPoly(mask_biggest, pts=[cnt_o_t_max], color=(1, 1, 1))

            mask_region_in_patch_region = mask_biggest[y : y + h, x : x + w]

            textline_biggest_region = mask_biggest * textline_mask_tot

            textline_rotated_seperated = self.seperate_lines_new2(textline_biggest_region[y : y + h, x : x + w], 0, num_col)

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
                cnt_textlines_in_image_ind = return_contours_of_interested_textline(mask_biggest2, pixel_img)

                try:
                    textlines_cnt_per_region.append(cnt_textlines_in_image_ind[0] / scale_par)
                except:
                    pass
                # print(len(cnt_textlines_in_image_ind))

                # plt.imshow(mask_biggest2)
                # plt.show()
            textlines_cnt_tot_per_process.append(textlines_cnt_per_region)
            index_polygons_per_process_per_process.append(index_polygons_per_process[iiii])
            polygons_per_par_process_per_process.append(con_par_org[iiii])

        queue_of_all_params.put([index_polygons_per_process_per_process, polygons_per_par_process_per_process, textregions_cnt_tot_per_process, textlines_cnt_tot_per_process])

    def small_textlines_to_parent_adherence2(self, textlines_con, textline_iamge, num_col):
        # print(textlines_con)
        # textlines_con=textlines_con.astype(np.uint32)

        textlines_con_changed = []
        for m1 in range(len(textlines_con)):

            # textlines_tot=textlines_con[m1]
            # textlines_tot=textlines_tot.astype()
            textlines_tot = []
            textlines_tot_org_form = []
            # print(textlines_tot)

            for nn in range(len(textlines_con[m1])):
                textlines_tot.append(np.array(textlines_con[m1][nn], dtype=np.int32))
                textlines_tot_org_form.append(textlines_con[m1][nn])

            ##img_text_all=np.zeros((textline_iamge.shape[0],textline_iamge.shape[1]))
            ##img_text_all=cv2.fillPoly(img_text_all, pts =textlines_tot , color=(1,1,1))

            ##plt.imshow(img_text_all)
            ##plt.show()
            areas_cnt_text = np.array([cv2.contourArea(textlines_tot[j]) for j in range(len(textlines_tot))])
            areas_cnt_text = areas_cnt_text / float(textline_iamge.shape[0] * textline_iamge.shape[1])
            indexes_textlines = np.array(range(len(textlines_tot)))

            # print(areas_cnt_text,np.min(areas_cnt_text),np.max(areas_cnt_text))
            if num_col == 0:
                min_area = 0.0004
            elif num_col == 1:
                min_area = 0.0003
            else:
                min_area = 0.0001
            indexes_textlines_small = indexes_textlines[areas_cnt_text < min_area]

            # print(indexes_textlines)

            textlines_small = []
            textlines_small_org_form = []
            for i in indexes_textlines_small:
                textlines_small.append(textlines_tot[i])
                textlines_small_org_form.append(textlines_tot_org_form[i])

            textlines_big = []
            textlines_big_org_form = []
            for i in list(set(indexes_textlines) - set(indexes_textlines_small)):
                textlines_big.append(textlines_tot[i])
                textlines_big_org_form.append(textlines_tot_org_form[i])

            img_textline_s = np.zeros((textline_iamge.shape[0], textline_iamge.shape[1]))
            img_textline_s = cv2.fillPoly(img_textline_s, pts=textlines_small, color=(1, 1, 1))

            img_textline_b = np.zeros((textline_iamge.shape[0], textline_iamge.shape[1]))
            img_textline_b = cv2.fillPoly(img_textline_b, pts=textlines_big, color=(1, 1, 1))

            sum_small_big_all = img_textline_s + img_textline_b
            sum_small_big_all2 = (sum_small_big_all[:, :] == 2) * 1

            sum_intersection_sb = sum_small_big_all2.sum(axis=1).sum()

            if sum_intersection_sb > 0:

                dis_small_from_bigs_tot = []
                for z1 in range(len(textlines_small)):
                    # print(len(textlines_small),'small')
                    intersections = []
                    for z2 in range(len(textlines_big)):
                        img_text = np.zeros((textline_iamge.shape[0], textline_iamge.shape[1]))
                        img_text = cv2.fillPoly(img_text, pts=[textlines_small[z1]], color=(1, 1, 1))

                        img_text2 = np.zeros((textline_iamge.shape[0], textline_iamge.shape[1]))
                        img_text2 = cv2.fillPoly(img_text2, pts=[textlines_big[z2]], color=(1, 1, 1))

                        sum_small_big = img_text2 + img_text
                        sum_small_big_2 = (sum_small_big[:, :] == 2) * 1

                        sum_intersection = sum_small_big_2.sum(axis=1).sum()

                        # print(sum_intersection)

                        intersections.append(sum_intersection)

                    if len(np.array(intersections)[np.array(intersections) > 0]) == 0:
                        intersections = []

                    try:
                        dis_small_from_bigs_tot.append(np.argmax(intersections))
                    except:
                        dis_small_from_bigs_tot.append(-1)

                smalls_list = np.array(dis_small_from_bigs_tot)[np.array(dis_small_from_bigs_tot) >= 0]

                # index_small_textlines_rest=list( set(indexes_textlines_small)-set(smalls_list) )

                textlines_big_with_change = []
                textlines_big_with_change_con = []
                textlines_small_with_change = []

                for z in list(set(smalls_list)):
                    index_small_textlines = list(np.where(np.array(dis_small_from_bigs_tot) == z)[0])
                    # print(z,index_small_textlines)

                    img_text2 = np.zeros((textline_iamge.shape[0], textline_iamge.shape[1], 3))
                    img_text2 = cv2.fillPoly(img_text2, pts=[textlines_big[z]], color=(255, 255, 255))

                    textlines_big_with_change.append(z)

                    for k in index_small_textlines:
                        img_text2 = cv2.fillPoly(img_text2, pts=[textlines_small[k]], color=(255, 255, 255))
                        textlines_small_with_change.append(k)

                    img_text2 = img_text2.astype(np.uint8)
                    imgray = cv2.cvtColor(img_text2, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
                    cont, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # print(cont[0],type(cont))

                    textlines_big_with_change_con.append(cont)
                    textlines_big_org_form[z] = cont[0]

                    # plt.imshow(img_text2)
                    # plt.show()

                # print(textlines_big_with_change,'textlines_big_with_change')
                # print(textlines_small_with_change,'textlines_small_with_change')
                # print(textlines_big)
                textlines_con_changed.append(textlines_big_org_form)

            else:
                textlines_con_changed.append(textlines_big_org_form)
        return textlines_con_changed

    def check_any_text_region_in_model_one_is_main_or_header(self, regions_model_1, regions_model_full, contours_only_text_parent, all_box_coord, all_found_texline_polygons, slopes, contours_only_text_parent_d_ordered):
        text_only = (regions_model_1[:, :] == 1) * 1
        contours_only_text, hir_on_text = return_contours_of_image(text_only)

        """
        contours_only_text_parent=return_parent_contours( contours_only_text,hir_on_text)

        areas_cnt_text=np.array([cv2.contourArea(contours_only_text_parent[j]) for j in range(len(contours_only_text_parent))])
        areas_cnt_text=areas_cnt_text/float(text_only.shape[0]*text_only.shape[1])

        ###areas_cnt_text_h=np.array([cv2.contourArea(contours_only_text_parent_h[j]) for j in range(len(contours_only_text_parent_h))])
        ###areas_cnt_text_h=areas_cnt_text_h/float(text_only_h.shape[0]*text_only_h.shape[1])

        ###contours_only_text_parent=[contours_only_text_parent[jz] for jz in range(len(contours_only_text_parent)) if areas_cnt_text[jz]>0.0002]
        contours_only_text_parent=[contours_only_text_parent[jz] for jz in range(len(contours_only_text_parent)) if areas_cnt_text[jz]>0.00001]
        """

        cx_main, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, y_corr_x_min_from_argmin = find_new_features_of_contoures(contours_only_text_parent)

        length_con = x_max_main - x_min_main
        height_con = y_max_main - y_min_main

        all_found_texline_polygons_main = []
        all_found_texline_polygons_head = []

        all_box_coord_main = []
        all_box_coord_head = []

        slopes_main = []
        slopes_head = []

        contours_only_text_parent_main = []
        contours_only_text_parent_head = []

        contours_only_text_parent_main_d = []
        contours_only_text_parent_head_d = []

        for ii in range(len(contours_only_text_parent)):
            con = contours_only_text_parent[ii]
            img = np.zeros((regions_model_1.shape[0], regions_model_1.shape[1], 3))
            img = cv2.fillPoly(img, pts=[con], color=(255, 255, 255))

            all_pixels = ((img[:, :, 0] == 255) * 1).sum()

            pixels_header = (((img[:, :, 0] == 255) & (regions_model_full[:, :, 0] == 2)) * 1).sum()
            pixels_main = all_pixels - pixels_header

            if (pixels_header >= pixels_main) and ((length_con[ii] / float(height_con[ii])) >= 1.3):
                regions_model_1[:, :][(regions_model_1[:, :] == 1) & (img[:, :, 0] == 255)] = 2
                contours_only_text_parent_head.append(con)
                if contours_only_text_parent_d_ordered is not None:
                    contours_only_text_parent_head_d.append(contours_only_text_parent_d_ordered[ii])
                all_box_coord_head.append(all_box_coord[ii])
                slopes_head.append(slopes[ii])
                all_found_texline_polygons_head.append(all_found_texline_polygons[ii])
            else:
                regions_model_1[:, :][(regions_model_1[:, :] == 1) & (img[:, :, 0] == 255)] = 1
                contours_only_text_parent_main.append(con)
                if contours_only_text_parent_d_ordered is not None:
                    contours_only_text_parent_main_d.append(contours_only_text_parent_d_ordered[ii])
                all_box_coord_main.append(all_box_coord[ii])
                slopes_main.append(slopes[ii])
                all_found_texline_polygons_main.append(all_found_texline_polygons[ii])

            # print(all_pixels,pixels_main,pixels_header)

            # plt.imshow(img[:,:,0])
            # plt.show()
        return regions_model_1, contours_only_text_parent_main, contours_only_text_parent_head, all_box_coord_main, all_box_coord_head, all_found_texline_polygons_main, all_found_texline_polygons_head, slopes_main, slopes_head, contours_only_text_parent_main_d, contours_only_text_parent_head_d

    def putt_bb_of_drop_capitals_of_model_in_patches_in_layout(self, layout_in_patch):

        drop_only = (layout_in_patch[:, :, 0] == 4) * 1
        contours_drop, hir_on_drop = return_contours_of_image(drop_only)
        contours_drop_parent = return_parent_contours(contours_drop, hir_on_drop)

        areas_cnt_text = np.array([cv2.contourArea(contours_drop_parent[j]) for j in range(len(contours_drop_parent))])
        areas_cnt_text = areas_cnt_text / float(drop_only.shape[0] * drop_only.shape[1])

        contours_drop_parent = [contours_drop_parent[jz] for jz in range(len(contours_drop_parent)) if areas_cnt_text[jz] > 0.00001]

        areas_cnt_text = [areas_cnt_text[jz] for jz in range(len(areas_cnt_text)) if areas_cnt_text[jz] > 0.001]

        contours_drop_parent_final = []

        for jj in range(len(contours_drop_parent)):
            x, y, w, h = cv2.boundingRect(contours_drop_parent[jj])
            layout_in_patch[y : y + h, x : x + w, 0] = 4

        return layout_in_patch

    def put_drop_out_from_only_drop_model(self, layout_no_patch, layout1):

        drop_only = (layout_no_patch[:, :, 0] == 4) * 1
        contours_drop, hir_on_drop = return_contours_of_image(drop_only)
        contours_drop_parent = return_parent_contours(contours_drop, hir_on_drop)

        areas_cnt_text = np.array([cv2.contourArea(contours_drop_parent[j]) for j in range(len(contours_drop_parent))])
        areas_cnt_text = areas_cnt_text / float(drop_only.shape[0] * drop_only.shape[1])

        contours_drop_parent = [contours_drop_parent[jz] for jz in range(len(contours_drop_parent)) if areas_cnt_text[jz] > 0.00001]

        areas_cnt_text = [areas_cnt_text[jz] for jz in range(len(areas_cnt_text)) if areas_cnt_text[jz] > 0.00001]

        contours_drop_parent_final = []

        for jj in range(len(contours_drop_parent)):
            x, y, w, h = cv2.boundingRect(contours_drop_parent[jj])
            # boxes.append([int(x), int(y), int(w), int(h)])

            map_of_drop_contour_bb = np.zeros((layout1.shape[0], layout1.shape[1]))
            map_of_drop_contour_bb[y : y + h, x : x + w] = layout1[y : y + h, x : x + w]

            if (((map_of_drop_contour_bb == 1) * 1).sum() / float(((map_of_drop_contour_bb == 5) * 1).sum()) * 100) >= 15:
                contours_drop_parent_final.append(contours_drop_parent[jj])

        layout_no_patch[:, :, 0][layout_no_patch[:, :, 0] == 4] = 0

        layout_no_patch = cv2.fillPoly(layout_no_patch, pts=contours_drop_parent_final, color=(4, 4, 4))

        return layout_no_patch

    def do_order_of_regions(self, contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot):

        if self.full_layout:
            cx_text_only, cy_text_only, x_min_text_only, _, _, _, y_cor_x_min_main = find_new_features_of_contoures(contours_only_text_parent)
            cx_text_only_h, cy_text_only_h, x_min_text_only_h, _, _, _, y_cor_x_min_main_h = find_new_features_of_contoures(contours_only_text_parent_h)

            try:
                arg_text_con = []
                for ii in range(len(cx_text_only)):
                    for jj in range(len(boxes)):
                        if (x_min_text_only[ii] + 80) >= boxes[jj][0] and (x_min_text_only[ii] + 80) < boxes[jj][1] and y_cor_x_min_main[ii] >= boxes[jj][2] and y_cor_x_min_main[ii] < boxes[jj][3]:
                            arg_text_con.append(jj)
                            break
                arg_arg_text_con = np.argsort(arg_text_con)
                args_contours = np.array(range(len(arg_text_con)))

                arg_text_con_h = []
                for ii in range(len(cx_text_only_h)):
                    for jj in range(len(boxes)):
                        if (x_min_text_only_h[ii] + 80) >= boxes[jj][0] and (x_min_text_only_h[ii] + 80) < boxes[jj][1] and y_cor_x_min_main_h[ii] >= boxes[jj][2] and y_cor_x_min_main_h[ii] < boxes[jj][3]:
                            arg_text_con_h.append(jj)
                            break
                arg_arg_text_con = np.argsort(arg_text_con_h)
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

                    indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = self.order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                    order_of_texts, id_of_texts = self.order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

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
                arg_arg_text_con = np.argsort(arg_text_con)
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
                #####

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

                    indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = self.order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                    order_of_texts, id_of_texts = self.order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

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

        else:
            cx_text_only, cy_text_only, x_min_text_only, _, _, _, y_cor_x_min_main = find_new_features_of_contoures(contours_only_text_parent)

            try:
                arg_text_con = []
                for ii in range(len(cx_text_only)):
                    for jj in range(len(boxes)):
                        if (x_min_text_only[ii] + 80) >= boxes[jj][0] and (x_min_text_only[ii] + 80) < boxes[jj][1] and y_cor_x_min_main[ii] >= boxes[jj][2] and y_cor_x_min_main[ii] < boxes[jj][3]:
                            arg_text_con.append(jj)
                            break
                arg_arg_text_con = np.argsort(arg_text_con)
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

                    indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = self.order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                    order_of_texts, id_of_texts = self.order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

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

            except:
                arg_text_con = []
                for ii in range(len(cx_text_only)):
                    for jj in range(len(boxes)):
                        if cx_text_only[ii] >= boxes[jj][0] and cx_text_only[ii] < boxes[jj][1] and cy_text_only[ii] >= boxes[jj][2] and cy_text_only[ii] < boxes[jj][3]:  # this is valid if the center of region identify in which box it is located
                            arg_text_con.append(jj)
                            break
                arg_arg_text_con = np.argsort(arg_text_con)
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

                    indexes_sorted, matrix_of_orders, kind_of_texts_sorted, index_by_kind_sorted = self.order_of_regions(textline_mask_tot[int(boxes[iij][2]) : int(boxes[iij][3]), int(boxes[iij][0]) : int(boxes[iij][1])], con_inter_box, con_inter_box_h, boxes[iij][2])

                    order_of_texts, id_of_texts = self.order_and_id_of_texts(con_inter_box, con_inter_box_h, matrix_of_orders, indexes_sorted, index_by_kind_sorted, kind_of_texts_sorted, ref_point)

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

    def adhere_drop_capital_region_into_cprresponding_textline(self, text_regions_p, polygons_of_drop_capitals, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_texline_polygons, all_found_texline_polygons_h):
        # print(np.shape(all_found_texline_polygons),np.shape(all_found_texline_polygons[3]),'all_found_texline_polygonsshape')
        # print(all_found_texline_polygons[3])
        cx_m, cy_m, _, _, _, _, _ = find_new_features_of_contoures(contours_only_text_parent)
        cx_h, cy_h, _, _, _, _, _ = find_new_features_of_contoures(contours_only_text_parent_h)
        cx_d, cy_d, _, _, y_min_d, y_max_d, _ = find_new_features_of_contoures(polygons_of_drop_capitals)

        img_con_all = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
        for j_cont in range(len(contours_only_text_parent)):
            img_con_all[all_box_coord[j_cont][0] : all_box_coord[j_cont][1], all_box_coord[j_cont][2] : all_box_coord[j_cont][3], 0] = (j_cont + 1) * 3
            # img_con_all=cv2.fillPoly(img_con_all,pts=[contours_only_text_parent[j_cont]],color=((j_cont+1)*3,(j_cont+1)*3,(j_cont+1)*3))

        # plt.imshow(img_con_all[:,:,0])
        # plt.show()
        # img_con_all=cv2.dilate(img_con_all, self.kernel, iterations=3)

        # plt.imshow(img_con_all[:,:,0])
        # plt.show()
        # print(np.unique(img_con_all[:,:,0]))
        for i_drop in range(len(polygons_of_drop_capitals)):
            # print(i_drop,'i_drop')
            img_con_all_copy = np.copy(img_con_all)
            img_con = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
            img_con = cv2.fillPoly(img_con, pts=[polygons_of_drop_capitals[i_drop]], color=(1, 1, 1))

            # plt.imshow(img_con[:,:,0])
            # plt.show()
            ##img_con=cv2.dilate(img_con, self.kernel, iterations=30)

            # plt.imshow(img_con[:,:,0])
            # plt.show()

            # print(np.unique(img_con[:,:,0]))

            img_con_all_copy[:, :, 0] = img_con_all_copy[:, :, 0] + img_con[:, :, 0]

            img_con_all_copy[:, :, 0][img_con_all_copy[:, :, 0] == 1] = 0

            kherej_ghesmat = np.unique(img_con_all_copy[:, :, 0]) / 3
            res_summed_pixels = np.unique(img_con_all_copy[:, :, 0]) % 3
            region_with_intersected_drop = kherej_ghesmat[res_summed_pixels == 1]
            # region_with_intersected_drop=region_with_intersected_drop/3
            region_with_intersected_drop = region_with_intersected_drop.astype(np.uint8)

            # print(len(region_with_intersected_drop),'region_with_intersected_drop1')
            if len(region_with_intersected_drop) == 0:
                img_con_all_copy = np.copy(img_con_all)
                img_con = cv2.dilate(img_con, self.kernel, iterations=4)

                img_con_all_copy[:, :, 0] = img_con_all_copy[:, :, 0] + img_con[:, :, 0]

                img_con_all_copy[:, :, 0][img_con_all_copy[:, :, 0] == 1] = 0

                kherej_ghesmat = np.unique(img_con_all_copy[:, :, 0]) / 3
                res_summed_pixels = np.unique(img_con_all_copy[:, :, 0]) % 3
                region_with_intersected_drop = kherej_ghesmat[res_summed_pixels == 1]
                # region_with_intersected_drop=region_with_intersected_drop/3
                region_with_intersected_drop = region_with_intersected_drop.astype(np.uint8)
            # print(np.unique(img_con_all_copy[:,:,0]))
            if self.curved_line:

                if len(region_with_intersected_drop) > 1:
                    sum_pixels_of_intersection = []
                    for i in range(len(region_with_intersected_drop)):
                        # print((region_with_intersected_drop[i]*3+1))
                        sum_pixels_of_intersection.append(((img_con_all_copy[:, :, 0] == (region_with_intersected_drop[i] * 3 + 1)) * 1).sum())
                    # print(sum_pixels_of_intersection)
                    region_final = region_with_intersected_drop[np.argmax(sum_pixels_of_intersection)] - 1

                    # print(region_final,'region_final')
                    # cx_t,cy_t ,_, _, _ ,_,_= find_new_features_of_contoures(all_found_texline_polygons[int(region_final)])
                    try:
                        cx_t, cy_t, _, _, _, _, _ = find_new_features_of_contoures(all_found_texline_polygons[int(region_final)])
                        # print(all_box_coord[j_cont])
                        # print(cx_t)
                        # print(cy_t)
                        # print(cx_d[i_drop])
                        # print(cy_d[i_drop])
                        y_lines = np.array(cy_t)  # all_box_coord[int(region_final)][0]+np.array(cy_t)

                        # print(y_lines)

                        y_lines[y_lines < y_min_d[i_drop]] = 0
                        # print(y_lines)

                        arg_min = np.argmin(np.abs(y_lines - y_min_d[i_drop]))
                        # print(arg_min)

                        cnt_nearest = np.copy(all_found_texline_polygons[int(region_final)][arg_min])
                        cnt_nearest[:, 0, 0] = all_found_texline_polygons[int(region_final)][arg_min][:, 0, 0]  # +all_box_coord[int(region_final)][2]
                        cnt_nearest[:, 0, 1] = all_found_texline_polygons[int(region_final)][arg_min][:, 0, 1]  # +all_box_coord[int(region_final)][0]

                        img_textlines = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
                        img_textlines = cv2.fillPoly(img_textlines, pts=[cnt_nearest], color=(255, 255, 255))
                        img_textlines = cv2.fillPoly(img_textlines, pts=[polygons_of_drop_capitals[i_drop]], color=(255, 255, 255))

                        img_textlines = img_textlines.astype(np.uint8)
                        imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                        contours_combined, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        # print(len(contours_combined),'len textlines mixed')
                        areas_cnt_text = np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

                        contours_biggest = contours_combined[np.argmax(areas_cnt_text)]

                        # print(np.shape(contours_biggest))
                        # print(contours_biggest[:])
                        # contours_biggest[:,0,0]=contours_biggest[:,0,0]#-all_box_coord[int(region_final)][2]
                        # contours_biggest[:,0,1]=contours_biggest[:,0,1]#-all_box_coord[int(region_final)][0]

                        # contours_biggest=contours_biggest.reshape(np.shape(contours_biggest)[0],np.shape(contours_biggest)[2])

                        all_found_texline_polygons[int(region_final)][arg_min] = contours_biggest

                    except:
                        # print('gordun1')
                        pass
                elif len(region_with_intersected_drop) == 1:
                    region_final = region_with_intersected_drop[0] - 1

                    # areas_main=np.array([cv2.contourArea(all_found_texline_polygons[int(region_final)][0][j] ) for j in range(len(all_found_texline_polygons[int(region_final)]))])

                    # cx_t,cy_t ,_, _, _ ,_,_= find_new_features_of_contoures(all_found_texline_polygons[int(region_final)])

                    cx_t, cy_t, _, _, _, _, _ = find_new_features_of_contoures(all_found_texline_polygons[int(region_final)])
                    # print(all_box_coord[j_cont])
                    # print(cx_t)
                    # print(cy_t)
                    # print(cx_d[i_drop])
                    # print(cy_d[i_drop])
                    y_lines = np.array(cy_t)  # all_box_coord[int(region_final)][0]+np.array(cy_t)

                    y_lines[y_lines < y_min_d[i_drop]] = 0
                    # print(y_lines)

                    arg_min = np.argmin(np.abs(y_lines - y_min_d[i_drop]))
                    # print(arg_min)

                    cnt_nearest = np.copy(all_found_texline_polygons[int(region_final)][arg_min])
                    cnt_nearest[:, 0, 0] = all_found_texline_polygons[int(region_final)][arg_min][:, 0, 0]  # +all_box_coord[int(region_final)][2]
                    cnt_nearest[:, 0, 1] = all_found_texline_polygons[int(region_final)][arg_min][:, 0, 1]  # +all_box_coord[int(region_final)][0]

                    img_textlines = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
                    img_textlines = cv2.fillPoly(img_textlines, pts=[cnt_nearest], color=(255, 255, 255))
                    img_textlines = cv2.fillPoly(img_textlines, pts=[polygons_of_drop_capitals[i_drop]], color=(255, 255, 255))

                    img_textlines = img_textlines.astype(np.uint8)

                    # plt.imshow(img_textlines)
                    # plt.show()
                    imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                    contours_combined, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # print(len(contours_combined),'len textlines mixed')
                    areas_cnt_text = np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

                    contours_biggest = contours_combined[np.argmax(areas_cnt_text)]

                    # print(np.shape(contours_biggest))
                    # print(contours_biggest[:])
                    # contours_biggest[:,0,0]=contours_biggest[:,0,0]#-all_box_coord[int(region_final)][2]
                    # contours_biggest[:,0,1]=contours_biggest[:,0,1]#-all_box_coord[int(region_final)][0]
                    # print(np.shape(contours_biggest),'contours_biggest')
                    # print(np.shape(all_found_texline_polygons[int(region_final)][arg_min]))
                    ##contours_biggest=contours_biggest.reshape(np.shape(contours_biggest)[0],np.shape(contours_biggest)[2])
                    all_found_texline_polygons[int(region_final)][arg_min] = contours_biggest

                    # print(cx_t,'print')
                    try:
                        # print(all_found_texline_polygons[j_cont][0])
                        cx_t, cy_t, _, _, _, _, _ = find_new_features_of_contoures(all_found_texline_polygons[int(region_final)])
                        # print(all_box_coord[j_cont])
                        # print(cx_t)
                        # print(cy_t)
                        # print(cx_d[i_drop])
                        # print(cy_d[i_drop])
                        y_lines = all_box_coord[int(region_final)][0] + np.array(cy_t)

                        y_lines[y_lines < y_min_d[i_drop]] = 0
                        # print(y_lines)

                        arg_min = np.argmin(np.abs(y_lines - y_min_d[i_drop]))
                        # print(arg_min)

                        cnt_nearest = np.copy(all_found_texline_polygons[int(region_final)][arg_min])
                        cnt_nearest[:, 0, 0] = all_found_texline_polygons[int(region_final)][arg_min][:, 0, 0]  # +all_box_coord[int(region_final)][2]
                        cnt_nearest[:, 0, 1] = all_found_texline_polygons[int(region_final)][arg_min][:, 0, 1]  # +all_box_coord[int(region_final)][0]

                        img_textlines = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
                        img_textlines = cv2.fillPoly(img_textlines, pts=[cnt_nearest], color=(255, 255, 255))
                        img_textlines = cv2.fillPoly(img_textlines, pts=[polygons_of_drop_capitals[i_drop]], color=(255, 255, 255))

                        img_textlines = img_textlines.astype(np.uint8)
                        imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                        contours_combined, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        # print(len(contours_combined),'len textlines mixed')
                        areas_cnt_text = np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

                        contours_biggest = contours_combined[np.argmax(areas_cnt_text)]

                        # print(np.shape(contours_biggest))
                        # print(contours_biggest[:])
                        contours_biggest[:, 0, 0] = contours_biggest[:, 0, 0]  # -all_box_coord[int(region_final)][2]
                        contours_biggest[:, 0, 1] = contours_biggest[:, 0, 1]  # -all_box_coord[int(region_final)][0]

                        ##contours_biggest=contours_biggest.reshape(np.shape(contours_biggest)[0],np.shape(contours_biggest)[2])
                        all_found_texline_polygons[int(region_final)][arg_min] = contours_biggest
                        # all_found_texline_polygons[int(region_final)][arg_min]=contours_biggest

                    except:
                        pass
                else:
                    pass

                ##cx_t,cy_t ,_, _, _ ,_,_= find_new_features_of_contoures(all_found_texline_polygons[int(region_final)])
                ###print(all_box_coord[j_cont])
                ###print(cx_t)
                ###print(cy_t)
                ###print(cx_d[i_drop])
                ###print(cy_d[i_drop])
                ##y_lines=all_box_coord[int(region_final)][0]+np.array(cy_t)

                ##y_lines[y_lines<y_min_d[i_drop]]=0
                ###print(y_lines)

                ##arg_min=np.argmin(np.abs(y_lines-y_min_d[i_drop])  )
                ###print(arg_min)

                ##cnt_nearest=np.copy(all_found_texline_polygons[int(region_final)][arg_min])
                ##cnt_nearest[:,0,0]=all_found_texline_polygons[int(region_final)][arg_min][:,0,0]#+all_box_coord[int(region_final)][2]
                ##cnt_nearest[:,0,1]=all_found_texline_polygons[int(region_final)][arg_min][:,0,1]#+all_box_coord[int(region_final)][0]

                ##img_textlines=np.zeros((text_regions_p.shape[0],text_regions_p.shape[1],3))
                ##img_textlines=cv2.fillPoly(img_textlines,pts=[cnt_nearest],color=(255,255,255))
                ##img_textlines=cv2.fillPoly(img_textlines,pts=[polygons_of_drop_capitals[i_drop] ],color=(255,255,255))

                ##img_textlines=img_textlines.astype(np.uint8)

                ##plt.imshow(img_textlines)
                ##plt.show()
                ##imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
                ##ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                ##contours_combined,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                ##print(len(contours_combined),'len textlines mixed')
                ##areas_cnt_text=np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

                ##contours_biggest=contours_combined[np.argmax(areas_cnt_text)]

                ###print(np.shape(contours_biggest))
                ###print(contours_biggest[:])
                ##contours_biggest[:,0,0]=contours_biggest[:,0,0]#-all_box_coord[int(region_final)][2]
                ##contours_biggest[:,0,1]=contours_biggest[:,0,1]#-all_box_coord[int(region_final)][0]

                ##contours_biggest=contours_biggest.reshape(np.shape(contours_biggest)[0],np.shape(contours_biggest)[2])
                ##all_found_texline_polygons[int(region_final)][arg_min]=contours_biggest

            else:
                if len(region_with_intersected_drop) > 1:
                    sum_pixels_of_intersection = []
                    for i in range(len(region_with_intersected_drop)):
                        # print((region_with_intersected_drop[i]*3+1))
                        sum_pixels_of_intersection.append(((img_con_all_copy[:, :, 0] == (region_with_intersected_drop[i] * 3 + 1)) * 1).sum())
                    # print(sum_pixels_of_intersection)
                    region_final = region_with_intersected_drop[np.argmax(sum_pixels_of_intersection)] - 1

                    # print(region_final,'region_final')
                    # cx_t,cy_t ,_, _, _ ,_,_= find_new_features_of_contoures(all_found_texline_polygons[int(region_final)])
                    try:
                        cx_t, cy_t, _, _, _, _, _ = find_new_features_of_contoures(all_found_texline_polygons[int(region_final)])
                        # print(all_box_coord[j_cont])
                        # print(cx_t)
                        # print(cy_t)
                        # print(cx_d[i_drop])
                        # print(cy_d[i_drop])
                        y_lines = all_box_coord[int(region_final)][0] + np.array(cy_t)

                        # print(y_lines)

                        y_lines[y_lines < y_min_d[i_drop]] = 0
                        # print(y_lines)

                        arg_min = np.argmin(np.abs(y_lines - y_min_d[i_drop]))
                        # print(arg_min)

                        cnt_nearest = np.copy(all_found_texline_polygons[int(region_final)][arg_min])
                        cnt_nearest[:, 0] = all_found_texline_polygons[int(region_final)][arg_min][:, 0] + all_box_coord[int(region_final)][2]
                        cnt_nearest[:, 1] = all_found_texline_polygons[int(region_final)][arg_min][:, 1] + all_box_coord[int(region_final)][0]

                        img_textlines = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
                        img_textlines = cv2.fillPoly(img_textlines, pts=[cnt_nearest], color=(255, 255, 255))
                        img_textlines = cv2.fillPoly(img_textlines, pts=[polygons_of_drop_capitals[i_drop]], color=(255, 255, 255))

                        img_textlines = img_textlines.astype(np.uint8)
                        imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                        contours_combined, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        # print(len(contours_combined),'len textlines mixed')
                        areas_cnt_text = np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

                        contours_biggest = contours_combined[np.argmax(areas_cnt_text)]

                        # print(np.shape(contours_biggest))
                        # print(contours_biggest[:])
                        contours_biggest[:, 0, 0] = contours_biggest[:, 0, 0] - all_box_coord[int(region_final)][2]
                        contours_biggest[:, 0, 1] = contours_biggest[:, 0, 1] - all_box_coord[int(region_final)][0]

                        contours_biggest = contours_biggest.reshape(np.shape(contours_biggest)[0], np.shape(contours_biggest)[2])

                        all_found_texline_polygons[int(region_final)][arg_min] = contours_biggest

                    except:
                        # print('gordun1')
                        pass
                elif len(region_with_intersected_drop) == 1:
                    region_final = region_with_intersected_drop[0] - 1

                    # areas_main=np.array([cv2.contourArea(all_found_texline_polygons[int(region_final)][0][j] ) for j in range(len(all_found_texline_polygons[int(region_final)]))])

                    # cx_t,cy_t ,_, _, _ ,_,_= find_new_features_of_contoures(all_found_texline_polygons[int(region_final)])

                    # print(cx_t,'print')
                    try:
                        # print(all_found_texline_polygons[j_cont][0])
                        cx_t, cy_t, _, _, _, _, _ = find_new_features_of_contoures(all_found_texline_polygons[int(region_final)])
                        # print(all_box_coord[j_cont])
                        # print(cx_t)
                        # print(cy_t)
                        # print(cx_d[i_drop])
                        # print(cy_d[i_drop])
                        y_lines = all_box_coord[int(region_final)][0] + np.array(cy_t)

                        y_lines[y_lines < y_min_d[i_drop]] = 0
                        # print(y_lines)

                        arg_min = np.argmin(np.abs(y_lines - y_min_d[i_drop]))
                        # print(arg_min)

                        cnt_nearest = np.copy(all_found_texline_polygons[int(region_final)][arg_min])
                        cnt_nearest[:, 0] = all_found_texline_polygons[int(region_final)][arg_min][:, 0] + all_box_coord[int(region_final)][2]
                        cnt_nearest[:, 1] = all_found_texline_polygons[int(region_final)][arg_min][:, 1] + all_box_coord[int(region_final)][0]

                        img_textlines = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
                        img_textlines = cv2.fillPoly(img_textlines, pts=[cnt_nearest], color=(255, 255, 255))
                        img_textlines = cv2.fillPoly(img_textlines, pts=[polygons_of_drop_capitals[i_drop]], color=(255, 255, 255))

                        img_textlines = img_textlines.astype(np.uint8)
                        imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                        contours_combined, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        # print(len(contours_combined),'len textlines mixed')
                        areas_cnt_text = np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

                        contours_biggest = contours_combined[np.argmax(areas_cnt_text)]

                        # print(np.shape(contours_biggest))
                        # print(contours_biggest[:])
                        contours_biggest[:, 0, 0] = contours_biggest[:, 0, 0] - all_box_coord[int(region_final)][2]
                        contours_biggest[:, 0, 1] = contours_biggest[:, 0, 1] - all_box_coord[int(region_final)][0]

                        contours_biggest = contours_biggest.reshape(np.shape(contours_biggest)[0], np.shape(contours_biggest)[2])
                        all_found_texline_polygons[int(region_final)][arg_min] = contours_biggest
                        # all_found_texline_polygons[int(region_final)][arg_min]=contours_biggest

                    except:
                        pass
                else:
                    pass

        #####for i_drop in range(len(polygons_of_drop_capitals)):
        #####for j_cont in range(len(contours_only_text_parent)):
        #####img_con=np.zeros((text_regions_p.shape[0],text_regions_p.shape[1],3))
        #####img_con=cv2.fillPoly(img_con,pts=[polygons_of_drop_capitals[i_drop] ],color=(255,255,255))
        #####img_con=cv2.fillPoly(img_con,pts=[contours_only_text_parent[j_cont]],color=(255,255,255))

        #####img_con=img_con.astype(np.uint8)
        ######imgray = cv2.cvtColor(img_con, cv2.COLOR_BGR2GRAY)
        ######ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        ######contours_new,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #####contours_new,hir_new=return_contours_of_image(img_con)
        #####contours_new_parent=return_parent_contours( contours_new,hir_new)
        ######plt.imshow(img_con)
        ######plt.show()
        #####try:
        #####if len(contours_new_parent)==1:
        ######print(all_found_texline_polygons[j_cont][0])
        #####cx_t,cy_t ,_, _, _ ,_,_= find_new_features_of_contoures(all_found_texline_polygons[j_cont])
        ######print(all_box_coord[j_cont])
        ######print(cx_t)
        ######print(cy_t)
        ######print(cx_d[i_drop])
        ######print(cy_d[i_drop])
        #####y_lines=all_box_coord[j_cont][0]+np.array(cy_t)

        ######print(y_lines)

        #####arg_min=np.argmin(np.abs(y_lines-y_min_d[i_drop])  )
        ######print(arg_min)

        #####cnt_nearest=np.copy(all_found_texline_polygons[j_cont][arg_min])
        #####cnt_nearest[:,0]=all_found_texline_polygons[j_cont][arg_min][:,0]+all_box_coord[j_cont][2]
        #####cnt_nearest[:,1]=all_found_texline_polygons[j_cont][arg_min][:,1]+all_box_coord[j_cont][0]

        #####img_textlines=np.zeros((text_regions_p.shape[0],text_regions_p.shape[1],3))
        #####img_textlines=cv2.fillPoly(img_textlines,pts=[cnt_nearest],color=(255,255,255))
        #####img_textlines=cv2.fillPoly(img_textlines,pts=[polygons_of_drop_capitals[i_drop] ],color=(255,255,255))

        #####img_textlines=img_textlines.astype(np.uint8)
        #####imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
        #####ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        #####contours_combined,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #####areas_cnt_text=np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

        #####contours_biggest=contours_combined[np.argmax(areas_cnt_text)]

        ######print(np.shape(contours_biggest))
        ######print(contours_biggest[:])
        #####contours_biggest[:,0,0]=contours_biggest[:,0,0]-all_box_coord[j_cont][2]
        #####contours_biggest[:,0,1]=contours_biggest[:,0,1]-all_box_coord[j_cont][0]

        #####all_found_texline_polygons[j_cont][arg_min]=contours_biggest
        ######print(contours_biggest)
        ######plt.imshow(img_textlines[:,:,0])
        ######plt.show()
        #####else:
        #####pass
        #####except:
        #####pass
        return all_found_texline_polygons

    def save_plot_of_layout_main(self, text_regions_p, image_page):
        values = np.unique(text_regions_p[:, :])

        # pixels=['Background' , 'Main text' , 'Heading' , 'Marginalia' ,'Drop capitals' , 'Images' , 'Seperators' , 'Tables', 'Graphics']

        pixels = ["Background", "Main text", "Images", "Seperators", "Marginalia"]
        values_indexes = [0, 1, 2, 3, 4]
        plt.figure(figsize=(40, 40))
        plt.rcParams["font.size"] = "40"

        im = plt.imshow(text_regions_p[:, :])
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[np.where(values == i)[0][0]], label="{l}".format(l=pixels[int(np.where(values_indexes == i)[0][0])])) for i in values]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=40)
        plt.savefig(os.path.join(self.dir_of_layout, self.f_name + "_layout_main.png"))

    def save_plot_of_layout_main_all(self, text_regions_p, image_page):
        values = np.unique(text_regions_p[:, :])

        # pixels=['Background' , 'Main text' , 'Heading' , 'Marginalia' ,'Drop capitals' , 'Images' , 'Seperators' , 'Tables', 'Graphics']

        pixels = ["Background", "Main text", "Images", "Seperators", "Marginalia"]
        values_indexes = [0, 1, 2, 3, 4]

        plt.figure(figsize=(70, 40))
        plt.rcParams["font.size"] = "40"
        plt.subplot(1, 2, 1)
        plt.imshow(image_page)
        plt.subplot(1, 2, 2)
        im = plt.imshow(text_regions_p[:, :])
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[np.where(values == i)[0][0]], label="{l}".format(l=pixels[int(np.where(values_indexes == i)[0][0])])) for i in values]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=60)

        plt.savefig(os.path.join(self.dir_of_all, self.f_name + "_layout_main_and_page.png"))

    def save_plot_of_layout(self, text_regions_p, image_page):
        values = np.unique(text_regions_p[:, :])

        # pixels=['Background' , 'Main text' , 'Heading' , 'Marginalia' ,'Drop capitals' , 'Images' , 'Seperators' , 'Tables', 'Graphics']

        pixels = ["Background", "Main text", "Header", "Marginalia", "Drop capitals", "Images", "Seperators"]
        values_indexes = [0, 1, 2, 8, 4, 5, 6]
        plt.figure(figsize=(40, 40))
        plt.rcParams["font.size"] = "40"
        im = plt.imshow(text_regions_p[:, :])
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[np.where(values == i)[0][0]], label="{l}".format(l=pixels[int(np.where(values_indexes == i)[0][0])])) for i in values]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=40)
        plt.savefig(os.path.join(self.dir_of_layout, self.f_name + "_layout.png"))

    def save_plot_of_layout_all(self, text_regions_p, image_page):
        values = np.unique(text_regions_p[:, :])

        # pixels=['Background' , 'Main text' , 'Heading' , 'Marginalia' ,'Drop capitals' , 'Images' , 'Seperators' , 'Tables', 'Graphics']

        pixels = ["Background", "Main text", "Header", "Marginalia", "Drop capitals", "Images", "Seperators"]
        values_indexes = [0, 1, 2, 8, 4, 5, 6]

        plt.figure(figsize=(70, 40))
        plt.rcParams["font.size"] = "40"
        plt.subplot(1, 2, 1)
        plt.imshow(image_page)
        plt.subplot(1, 2, 2)
        im = plt.imshow(text_regions_p[:, :])
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[np.where(values == i)[0][0]], label="{l}".format(l=pixels[int(np.where(values_indexes == i)[0][0])])) for i in values]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=60)

        plt.savefig(os.path.join(self.dir_of_all, self.f_name + "_layout_and_page.png"))

    def save_deskewed_image(self, slope_deskew):
        img_rotated = self.rotyate_image_different(self.image_org, slope_deskew)

        if self.dir_of_all is not None:
            cv2.imwrite(os.path.join(self.dir_of_all, self.f_name + "_org.png"), self.image_org)

        cv2.imwrite(os.path.join(self.dir_of_deskewed, self.f_name + "_deskewed.png"), img_rotated)
        del img_rotated

    def run(self):
        is_image_enhanced = False
        # get image and sclaes, then extract the page of scanned image
        t1 = time.time()

        ##########

        ###is_image_enhanced,img_org,img_res=self.resize_and_enhance_image(is_image_enhanced)
        is_image_enhanced, img_org, img_res, num_col_classifier, num_column_is_classified = self.resize_and_enhance_image_with_column_classifier(is_image_enhanced)

        print(is_image_enhanced, "is_image_enhanced")
        K.clear_session()
        scale = 1
        if (self.allow_enhancement) and is_image_enhanced:
            cv2.imwrite(os.path.join(self.dir_out, self.f_name) + ".tif", img_res)
            img_res = img_res.astype(np.uint8)
            self.get_image_and_scales(img_org, img_res, scale)

        if (not self.allow_enhancement) and is_image_enhanced:
            self.get_image_and_scales_after_enhancing(img_org, img_res)

        if (self.allow_enhancement) and not is_image_enhanced:
            self.get_image_and_scales(img_org, img_res, scale)

        if (not self.allow_enhancement) and not is_image_enhanced:
            self.get_image_and_scales(img_org, img_res, scale)

        if (self.allow_scaling) and not is_image_enhanced:
            img_org, img_res, is_image_enhanced = self.resize_image_with_column_classifier(is_image_enhanced)
            self.get_image_and_scales_after_enhancing(img_org, img_res)

        # print(self.scale_x)

        print("enhancing: " + str(time.time() - t1))
        text_regions_p_1 = self.get_regions_from_xy_2models(img_res, is_image_enhanced)
        K.clear_session()
        gc.collect()

        print("textregion: " + str(time.time() - t1))

        img_g = cv2.imread(self.image_dir, 0)
        img_g = img_g.astype(np.uint8)

        img_g3 = np.zeros((img_g.shape[0], img_g.shape[1], 3))

        img_g3 = img_g3.astype(np.uint8)

        img_g3[:, :, 0] = img_g[:, :]
        img_g3[:, :, 1] = img_g[:, :]
        img_g3[:, :, 2] = img_g[:, :]

        ###self.produce_groundtruth_for_textline()
        image_page, page_coord = self.extract_page()

        # print(image_page.shape,'page')

        if self.dir_of_all is not None:
            cv2.imwrite(os.path.join(self.dir_of_all, self.f_name + "_page.png"), image_page)
        ##########
        K.clear_session()
        gc.collect()

        img_g3_page = img_g3[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3], :]
        del img_g3
        del img_g

        text_regions_p_1 = text_regions_p_1[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]

        mask_images = (text_regions_p_1[:, :] == 2) * 1
        mask_lines = (text_regions_p_1[:, :] == 3) * 1

        mask_images = mask_images.astype(np.uint8)
        mask_lines = mask_lines.astype(np.uint8)

        mask_images = cv2.erode(mask_images[:, :], self.kernel, iterations=10)

        img_only_regions_with_sep = ((text_regions_p_1[:, :] != 3) & (text_regions_p_1[:, :] != 0)) * 1
        img_only_regions_with_sep = img_only_regions_with_sep.astype(np.uint8)
        img_only_regions = cv2.erode(img_only_regions_with_sep[:, :], self.kernel, iterations=6)

        try:
            num_col, peaks_neg_fin = self.find_num_col(img_only_regions, multiplier=6.0)
            if not num_column_is_classified:
                num_col_classifier = num_col + 1
        except:
            num_col = None
            peaks_neg_fin = []

        print(num_col, "num_colnum_col")
        if num_col is None:
            txt_con_org = []
            order_text_new = []
            id_of_texts_tot = []
            all_found_texline_polygons = []
            all_box_coord = []
            polygons_of_images = []
            polygons_of_marginals = []
            all_found_texline_polygons_marginals = []
            all_box_coord_marginals = []
            slopes = []
            slopes_marginals = []
            self.write_into_page_xml(txt_con_org, page_coord, self.dir_out, order_text_new, id_of_texts_tot, all_found_texline_polygons, all_box_coord, polygons_of_images, polygons_of_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, self.curved_line, slopes, slopes_marginals)
        else:
            # pass
            try:
                patches = True
                scaler_h_textline = 1  # 1.2#1.2
                scaler_w_textline = 1  # 0.9#1
                textline_mask_tot_ea, textline_mask_tot_long_shot = self.textline_contours(image_page, patches, scaler_h_textline, scaler_w_textline)

                K.clear_session()
                gc.collect()

                print(np.unique(textline_mask_tot_ea[:, :]), "textline")

                if self.dir_of_all is not None:

                    values = np.unique(textline_mask_tot_ea[:, :])
                    pixels = ["Background", "Textlines"]
                    values_indexes = [0, 1]
                    plt.figure(figsize=(70, 40))
                    plt.rcParams["font.size"] = "40"
                    plt.subplot(1, 2, 1)
                    plt.imshow(image_page)
                    plt.subplot(1, 2, 2)
                    im = plt.imshow(textline_mask_tot_ea[:, :])
                    colors = [im.cmap(im.norm(value)) for value in values]
                    patches = [mpatches.Patch(color=colors[np.where(values == i)[0][0]], label="{l}".format(l=pixels[int(np.where(values_indexes == i)[0][0])])) for i in values]
                    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=60)

                    plt.savefig(os.path.join(self.dir_of_all, self.f_name + "_textline_and_page.png"))
                print("textline: " + str(time.time() - t1))
                # plt.imshow(textline_mask_tot_ea)
                # plt.show()
                # sys.exit()

                sigma = 2
                main_page_deskew = True
                slope_deskew = self.return_deskew_slop(cv2.erode(textline_mask_tot_ea, self.kernel, iterations=2), sigma, main_page_deskew)
                slope_first = 0  # self.return_deskew_slop(cv2.erode(textline_mask_tot_ea, self.kernel, iterations=2),sigma)

                if self.dir_of_deskewed is not None:
                    self.save_deskewed_image(slope_deskew)
                # img_rotated=self.rotyate_image_different(self.image_org,slope_deskew)
                print(slope_deskew, "slope_deskew")

                ##plt.imshow(img_rotated)
                ##plt.show()
                ##sys.exit()
                print("deskewing: " + str(time.time() - t1))

                image_page_rotated, textline_mask_tot = image_page[:, :], textline_mask_tot_ea[:, :]  # self.rotation_not_90_func(image_page,textline_mask_tot_ea,slope_first)
                textline_mask_tot[mask_images[:, :] == 1] = 0

                pixel_img = 1
                min_area = 0.00001
                max_area = 0.0006
                textline_mask_tot_small_size = self.return_contours_of_interested_region_by_size(textline_mask_tot, pixel_img, min_area, max_area)

                # text_regions_p_1[(textline_mask_tot[:,:]==1) & (text_regions_p_1[:,:]==2)]=1

                text_regions_p_1[mask_lines[:, :] == 1] = 3

                ##text_regions_p_1[textline_mask_tot_small_size[:,:]==1]=1

                text_regions_p = text_regions_p_1[:, :]  # long_short_region[:,:]#self.get_regions_from_2_models(image_page)

                text_regions_p = np.array(text_regions_p)

                if num_col_classifier == 1 or num_col_classifier == 2:

                    try:
                        regions_without_seperators = (text_regions_p[:, :] == 1) * 1
                        regions_without_seperators = regions_without_seperators.astype(np.uint8)

                        text_regions_p = self.get_marginals(rotate_image(regions_without_seperators, slope_deskew), text_regions_p, num_col_classifier, slope_deskew)

                    except:
                        pass
                else:
                    pass

                # plt.imshow(text_regions_p)
                # plt.show()

                if self.dir_of_all is not None:
                    self.save_plot_of_layout_main_all(text_regions_p, image_page)
                if self.dir_of_layout is not None:
                    self.save_plot_of_layout_main(text_regions_p, image_page)

                print("marginals: " + str(time.time() - t1))

                if not self.full_layout:

                    if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                        image_page_rotated_n, textline_mask_tot_d, text_regions_p_1_n = self.rotation_not_90_func(image_page, textline_mask_tot, text_regions_p, slope_deskew)

                        text_regions_p_1_n = resize_image(text_regions_p_1_n, text_regions_p.shape[0], text_regions_p.shape[1])
                        textline_mask_tot_d = resize_image(textline_mask_tot_d, text_regions_p.shape[0], text_regions_p.shape[1])

                        regions_without_seperators_d = (text_regions_p_1_n[:, :] == 1) * 1

                    regions_without_seperators = (text_regions_p[:, :] == 1) * 1  # ( (text_regions_p[:,:]==1) | (text_regions_p[:,:]==2) )*1 #self.return_regions_without_seperators_new(text_regions_p[:,:,0],img_only_regions)

                    pixel_lines = 3
                    if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                        num_col, peaks_neg_fin, matrix_of_lines_ch, spliter_y_new, seperators_closeup_n = self.find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines)

                    if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                        num_col_d, peaks_neg_fin_d, matrix_of_lines_ch_d, spliter_y_new_d, seperators_closeup_n_d = self.find_number_of_columns_in_document(np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines)
                    K.clear_session()
                    gc.collect()

                    # print(peaks_neg_fin,num_col,'num_col2')

                    print(num_col_classifier, "num_col_classifier")

                    if num_col_classifier >= 3:
                        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                            regions_without_seperators = regions_without_seperators.astype(np.uint8)
                            regions_without_seperators = cv2.erode(regions_without_seperators[:, :], self.kernel, iterations=6)

                            random_pixels_for_image = np.random.randn(regions_without_seperators.shape[0], regions_without_seperators.shape[1])
                            random_pixels_for_image[random_pixels_for_image < -0.5] = 0
                            random_pixels_for_image[random_pixels_for_image != 0] = 1

                            regions_without_seperators[(random_pixels_for_image[:, :] == 1) & (text_regions_p[:, :] == 2)] = 1

                        if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                            regions_without_seperators_d = regions_without_seperators_d.astype(np.uint8)
                            regions_without_seperators_d = cv2.erode(regions_without_seperators_d[:, :], self.kernel, iterations=6)

                            random_pixels_for_image = np.random.randn(regions_without_seperators_d.shape[0], regions_without_seperators_d.shape[1])
                            random_pixels_for_image[random_pixels_for_image < -0.5] = 0
                            random_pixels_for_image[random_pixels_for_image != 0] = 1

                            regions_without_seperators_d[(random_pixels_for_image[:, :] == 1) & (text_regions_p_1_n[:, :] == 2)] = 1
                    else:
                        pass

                    if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                        boxes = self.return_boxes_of_images_by_order_of_reading_new(spliter_y_new, regions_without_seperators, matrix_of_lines_ch)
                    else:
                        boxes_d = self.return_boxes_of_images_by_order_of_reading_new(spliter_y_new_d, regions_without_seperators_d, matrix_of_lines_ch_d)

                    # print(len(boxes),'boxes')

                    # sys.exit()

                    print("boxes in: " + str(time.time() - t1))
                    img_revised_tab = text_regions_p[:, :]

                    # plt.imshow(img_revised_tab)
                    # plt.show()
                    K.clear_session()

                pixel_img = 4
                min_area_mar = 0.00001
                polygons_of_marginals = return_contours_of_interested_region(text_regions_p, pixel_img, min_area_mar)

                if self.full_layout:
                    # set first model with second model
                    text_regions_p[:, :][text_regions_p[:, :] == 2] = 5
                    text_regions_p[:, :][text_regions_p[:, :] == 3] = 6
                    text_regions_p[:, :][text_regions_p[:, :] == 4] = 8

                    K.clear_session()
                    # gc.collect()

                    patches = True

                    image_page = image_page.astype(np.uint8)

                    # print(type(image_page))
                    regions_fully, regions_fully_only_drop = self.extract_text_regions(image_page, patches, cols=num_col_classifier)

                    regions_fully_only_drop = self.put_drop_out_from_only_drop_model(regions_fully_only_drop, text_regions_p)
                    regions_fully[:, :, 0][regions_fully_only_drop[:, :, 0] == 4] = 4
                    K.clear_session()
                    gc.collect()

                    # plt.imshow(regions_fully[:,:,0])
                    # plt.show()

                    regions_fully = self.putt_bb_of_drop_capitals_of_model_in_patches_in_layout(regions_fully)

                    # plt.imshow(regions_fully[:,:,0])
                    # plt.show()

                    K.clear_session()
                    gc.collect()
                    patches = False
                    regions_fully_np, _ = self.extract_text_regions(image_page, patches, cols=num_col_classifier)

                    # plt.imshow(regions_fully_np[:,:,0])
                    # plt.show()

                    if num_col_classifier > 2:
                        regions_fully_np[:, :, 0][regions_fully_np[:, :, 0] == 4] = 0
                    else:
                        regions_fully_np = filter_small_drop_capitals_from_no_patch_layout(regions_fully_np, text_regions_p)

                    # regions_fully_np=filter_small_drop_capitals_from_no_patch_layout(regions_fully_np,text_regions_p)
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
                    ##text_regions_p[:,:][(regions_fully[:,:,0]==7) & (text_regions_p[:,:]!=0)]=7

                    text_regions_p[:, :][regions_fully_np[:, :, 0] == 4] = 4

                    # plt.imshow(text_regions_p)
                    # plt.show()

                    if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                        image_page_rotated_n, textline_mask_tot_d, text_regions_p_1_n, regions_fully_n = self.rotation_not_90_func_full_layout(image_page, textline_mask_tot, text_regions_p, regions_fully, slope_deskew)

                        text_regions_p_1_n = resize_image(text_regions_p_1_n, text_regions_p.shape[0], text_regions_p.shape[1])
                        textline_mask_tot_d = resize_image(textline_mask_tot_d, text_regions_p.shape[0], text_regions_p.shape[1])
                        regions_fully_n = resize_image(regions_fully_n, text_regions_p.shape[0], text_regions_p.shape[1])

                        regions_without_seperators_d = (text_regions_p_1_n[:, :] == 1) * 1

                    regions_without_seperators = (text_regions_p[:, :] == 1) * 1  # ( (text_regions_p[:,:]==1) | (text_regions_p[:,:]==2) )*1 #self.return_regions_without_seperators_new(text_regions_p[:,:,0],img_only_regions)

                    K.clear_session()
                    gc.collect()

                    img_revised_tab = np.copy(text_regions_p[:, :])

                    print("full layout in: " + str(time.time() - t1))

                # sys.exit()

                pixel_img = 5
                polygons_of_images = return_contours_of_interested_region(img_revised_tab, pixel_img)

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

                    cx_bigest_d_big, cy_biggest_d_big, _, _, _, _, _ = find_new_features_of_contoures([contours_biggest_d])
                    cx_bigest_d, cy_biggest_d, _, _, _, _, _ = find_new_features_of_contoures(contours_only_text_parent_d)

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
                    # print(areas_cnt_text_parent,'areas_cnt_text_parent')

                    ###index_con_parents_d=np.argsort(areas_cnt_text_parent_d)
                    ##contours_only_text_parent_d=list(np.array(contours_only_text_parent_d)[index_con_parents_d])
                    ###areas_cnt_text_parent_d=list(np.array(areas_cnt_text_parent_d)[index_con_parents_d])

                    ##print(areas_cnt_text_parent_d,'areas_cnt_text_parent_d')

                    # print(len(contours_only_text_parent),len(contours_only_text_parent_d),'vizzz')

                txt_con_org = get_textregion_contours_in_org_image(contours_only_text_parent, self.image, slope_first)

                ###boxes_text,_=self.get_text_region_boxes_by_given_contours(contours_only_text_parent)
                boxes_text, _ = self.get_text_region_boxes_by_given_contours(contours_only_text_parent)
                boxes_marginals, _ = self.get_text_region_boxes_by_given_contours(polygons_of_marginals)
                ####boxes_text_h,_=self.get_text_region_boxes_by_given_contours(text_only_h,contours_only_text_parent_h,image_page)

                if not self.curved_line:
                    slopes, all_found_texline_polygons, boxes_text, txt_con_org, contours_only_text_parent, all_box_coord, index_by_text_par_con = self.get_slopes_and_deskew_new(txt_con_org, contours_only_text_parent, textline_mask_tot_ea, image_page_rotated, boxes_text, slope_deskew)

                    slopes_marginals, all_found_texline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, all_box_coord_marginals, index_by_text_par_con_marginal = self.get_slopes_and_deskew_new(polygons_of_marginals, polygons_of_marginals, textline_mask_tot_ea, image_page_rotated, boxes_marginals, slope_deskew)

                if self.curved_line:
                    scale_param = 1
                    all_found_texline_polygons, boxes_text, txt_con_org, contours_only_text_parent, all_box_coord, index_by_text_par_con, slopes = self.get_slopes_and_deskew_new_curved(txt_con_org, contours_only_text_parent, cv2.erode(textline_mask_tot_ea, kernel=self.kernel, iterations=1), image_page_rotated, boxes_text, text_only, num_col_classifier, scale_param, slope_deskew)

                    # all_found_texline_polygons,boxes_text,txt_con_org,contours_only_text_parent,all_box_coord=self.get_slopes_and_deskew_new_curved(txt_con_org,contours_only_text_parent,textline_mask_tot_ea,image_page_rotated,boxes_text,text_only,num_col,scale_param)
                    all_found_texline_polygons = self.small_textlines_to_parent_adherence2(all_found_texline_polygons, textline_mask_tot_ea, num_col_classifier)

                    # slopes=list(np.zeros(len(contours_only_text_parent)))

                    all_found_texline_polygons_marginals, boxes_marginals, _, polygons_of_marginals, all_box_coord_marginals, index_by_text_par_con_marginal, slopes_marginals = self.get_slopes_and_deskew_new_curved(polygons_of_marginals, polygons_of_marginals, cv2.erode(textline_mask_tot_ea, kernel=self.kernel, iterations=1), image_page_rotated, boxes_marginals, text_only, num_col_classifier, scale_param, slope_deskew)

                    # all_found_texline_polygons,boxes_text,txt_con_org,contours_only_text_parent,all_box_coord=self.get_slopes_and_deskew_new_curved(txt_con_org,contours_only_text_parent,textline_mask_tot_ea,image_page_rotated,boxes_text,text_only,num_col,scale_param)
                    all_found_texline_polygons_marginals = self.small_textlines_to_parent_adherence2(all_found_texline_polygons_marginals, textline_mask_tot_ea, num_col_classifier)

                index_of_vertical_text_contours = np.array(range(len(slopes)))[(abs(np.array(slopes)) > 60)]

                contours_text_vertical = [contours_only_text_parent[i] for i in index_of_vertical_text_contours]

                K.clear_session()
                gc.collect()

                # contours_only_text_parent_d_ordered=list(np.array(contours_only_text_parent_d_ordered)[index_by_text_par_con])
                ###print(index_by_text_par_con,'index_by_text_par_con')

                if self.full_layout:
                    ##for iii in range(len(contours_only_text_parent)):
                    ##img1=np.zeros((text_only.shape[0],text_only.shape[1],3))
                    ##img1=cv2.fillPoly(img1,pts=[contours_only_text_parent[iii]] ,color=(1,1,1))

                    ##plt.imshow(img1[:,:,0])
                    ##plt.show()

                    ##img2=np.zeros((text_only.shape[0],text_only.shape[1],3))
                    ##img2=cv2.fillPoly(img2,pts=[contours_only_text_parent_d_ordered[iii]] ,color=(1,1,1))

                    ##plt.imshow(img2[:,:,0])
                    ##plt.show()

                    if np.abs(slope_deskew) >= SLOPE_THRESHOLD:
                        contours_only_text_parent_d_ordered = list(np.array(contours_only_text_parent_d_ordered)[index_by_text_par_con])

                        text_regions_p, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_texline_polygons, all_found_texline_polygons_h, slopes, slopes_h, contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered = self.check_any_text_region_in_model_one_is_main_or_header(text_regions_p, regions_fully, contours_only_text_parent, all_box_coord, all_found_texline_polygons, slopes, contours_only_text_parent_d_ordered)
                    else:
                        contours_only_text_parent_d_ordered = None

                        text_regions_p, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_texline_polygons, all_found_texline_polygons_h, slopes, slopes_h, contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered = self.check_any_text_region_in_model_one_is_main_or_header(text_regions_p, regions_fully, contours_only_text_parent, all_box_coord, all_found_texline_polygons, slopes, contours_only_text_parent_d_ordered)

                    ###text_regions_p,contours_only_text_parent,contours_only_text_parent_h,all_box_coord,all_box_coord_h,all_found_texline_polygons,all_found_texline_polygons_h=self.check_any_text_region_in_model_one_is_main_or_header(text_regions_p,regions_fully,contours_only_text_parent,all_box_coord,all_found_texline_polygons)
                    # text_regions_p=self.return_region_segmentation_after_implementing_not_head_maintext_parallel(text_regions_p,boxes)

                    # if you want to save the layout result just uncommet following plot

                    if self.dir_of_layout is not None:
                        self.save_plot_of_layout(text_regions_p, image_page)
                    if self.dir_of_all is not None:
                        self.save_plot_of_layout_all(text_regions_p, image_page)

                    K.clear_session()
                    gc.collect()

                    ##print('Job done in: '+str(time.time()-t1))

                    polygons_of_tabels = []

                    pixel_img = 4
                    polygons_of_drop_capitals = return_contours_of_interested_region_by_min_size(text_regions_p, pixel_img)
                    # polygons_of_drop_capitals=[]

                    all_found_texline_polygons = self.adhere_drop_capital_region_into_cprresponding_textline(text_regions_p, polygons_of_drop_capitals, contours_only_text_parent, contours_only_text_parent_h, all_box_coord, all_box_coord_h, all_found_texline_polygons, all_found_texline_polygons_h)

                    # print(len(contours_only_text_parent_h),len(contours_only_text_parent_h_d_ordered),'contours_only_text_parent_h')
                    pixel_lines = 6

                    if not self.headers_off:
                        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                            num_col, peaks_neg_fin, matrix_of_lines_ch, spliter_y_new, seperators_closeup_n = self.find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines, contours_only_text_parent_h)
                        else:
                            num_col_d, peaks_neg_fin_d, matrix_of_lines_ch_d, spliter_y_new_d, seperators_closeup_n_d = self.find_number_of_columns_in_document(np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines, contours_only_text_parent_h_d_ordered)
                    elif self.headers_off:
                        if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                            num_col, peaks_neg_fin, matrix_of_lines_ch, spliter_y_new, seperators_closeup_n = self.find_number_of_columns_in_document(np.repeat(text_regions_p[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines)
                        else:
                            num_col_d, peaks_neg_fin_d, matrix_of_lines_ch_d, spliter_y_new_d, seperators_closeup_n_d = self.find_number_of_columns_in_document(np.repeat(text_regions_p_1_n[:, :, np.newaxis], 3, axis=2), num_col_classifier, pixel_lines)

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
                    else:
                        pass

                    if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                        boxes = self.return_boxes_of_images_by_order_of_reading_new(spliter_y_new, regions_without_seperators, matrix_of_lines_ch)
                    else:
                        boxes_d = self.return_boxes_of_images_by_order_of_reading_new(spliter_y_new_d, regions_without_seperators_d, matrix_of_lines_ch_d)

                # print(slopes)
                if self.dir_of_cropped_images is not None:
                    self.write_images_into_directory(polygons_of_images, self.dir_of_cropped_images, image_page)

                if self.full_layout:
                    if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                        order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot)
                    else:
                        order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent_d_ordered, contours_only_text_parent_h_d_ordered, boxes_d, textline_mask_tot_d)

                    self.write_into_page_xml_full(contours_only_text_parent, contours_only_text_parent_h, page_coord, self.dir_out, order_text_new, id_of_texts_tot, all_found_texline_polygons, all_found_texline_polygons_h, all_box_coord, all_box_coord_h, polygons_of_images, polygons_of_tabels, polygons_of_drop_capitals, polygons_of_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals)
                else:
                    contours_only_text_parent_h = None
                    # print('bura galmir?')
                    if np.abs(slope_deskew) < SLOPE_THRESHOLD:
                        contours_only_text_parent = list(np.array(contours_only_text_parent)[index_by_text_par_con])
                        order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent, contours_only_text_parent_h, boxes, textline_mask_tot)
                    else:
                        contours_only_text_parent_d_ordered = list(np.array(contours_only_text_parent_d_ordered)[index_by_text_par_con])
                        order_text_new, id_of_texts_tot = self.do_order_of_regions(contours_only_text_parent_d_ordered, contours_only_text_parent_h, boxes_d, textline_mask_tot_d)
                    # order_text_new , id_of_texts_tot=self.do_order_of_regions(contours_only_text_parent,contours_only_text_parent_h,boxes,textline_mask_tot)
                    self.write_into_page_xml(txt_con_org, page_coord, self.dir_out, order_text_new, id_of_texts_tot, all_found_texline_polygons, all_box_coord, polygons_of_images, polygons_of_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, self.curved_line, slopes, slopes_marginals)

            except:
                txt_con_org = []
                order_text_new = []
                id_of_texts_tot = []
                all_found_texline_polygons = []
                all_box_coord = []
                polygons_of_images = []
                polygons_of_marginals = []
                all_found_texline_polygons_marginals = []
                all_box_coord_marginals = []
                slopes = []
                slopes_marginals = []
                self.write_into_page_xml(txt_con_org, page_coord, self.dir_out, order_text_new, id_of_texts_tot, all_found_texline_polygons, all_box_coord, polygons_of_images, polygons_of_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, self.curved_line, slopes, slopes_marginals)

        print("Job done in: " + str(time.time() - t1))

