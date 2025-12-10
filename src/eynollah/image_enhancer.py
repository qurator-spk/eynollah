"""
Image enhancer. The output can be written as same scale of input or in new predicted scale.
"""

# FIXME: fix all of those...
# pyright: reportUnboundVariable=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false

import logging
import os
import time
from typing import Optional
from pathlib import Path
import gc

import cv2
from keras.models import Model
import numpy as np
import tensorflow as tf # type: ignore
from skimage.morphology import skeletonize

from .model_zoo import EynollahModelZoo
from .utils.resize import resize_image
from .utils.pil_cv2 import pil2cv
from .utils import (
    is_image_filename,
    crop_image_inside_box
)

DPI_THRESHOLD = 298
KERNEL = np.ones((5, 5), np.uint8)


class Enhancer:
    def __init__(
        self,
        *,
        model_zoo: EynollahModelZoo,
        num_col_upper : Optional[int] = None,
        num_col_lower : Optional[int] = None,
        save_org_scale : bool = False,
    ):
        self.input_binary = False
        self.save_org_scale = save_org_scale
        if num_col_upper:
            self.num_col_upper = int(num_col_upper)
        else:
            self.num_col_upper = num_col_upper
        if num_col_lower:
            self.num_col_lower = int(num_col_lower)
        else:
            self.num_col_lower = num_col_lower
            
        self.logger = logging.getLogger('eynollah.enhance')
        self.model_zoo = model_zoo
        for v in ['binarization', 'enhancement', 'col_classifier', 'page']:
            self.model_zoo.load_model(v)
        
        try:
            for device in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(device, True)
        except:
            self.logger.warning("no GPU device available")

    def cache_images(self, image_filename=None, image_pil=None, dpi=None):
        ret = {}
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
        self.cache_images(image_filename=image_filename)
        self.output_filename = os.path.join(dir_out, Path(image_filename).stem +'.png')

    def imread(self, grayscale=False, uint8=True):
        key = 'img'
        if grayscale:
            key += '_grayscale'
        if uint8:
            key += '_uint8'
        return self._imgs[key].copy()

    def predict_enhancement(self, img):
        self.logger.debug("enter predict_enhancement")

        img_height_model = self.model_zoo.get('enhancement', Model).layers[-1].output_shape[1]
        img_width_model = self.model_zoo.get('enhancement', Model).layers[-1].output_shape[2]
        if img.shape[0] < img_height_model:
            img = cv2.resize(img, (img.shape[1], img_width_model), interpolation=cv2.INTER_NEAREST)
        if img.shape[1] < img_width_model:
            img = cv2.resize(img, (img_height_model, img.shape[0]), interpolation=cv2.INTER_NEAREST)
        margin = int(0.1 * img_width_model)
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
                label_p_pred = self.model_zoo.get('enhancement', Model).predict(img_patch, verbose='0')
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
        if num_col == 1:
            img_w_new = 2000
        elif num_col == 2:
            img_w_new = 2400
        elif num_col == 3:
            img_w_new = 3000
        elif num_col == 4:
            img_w_new = 4000
        elif num_col == 5:
            img_w_new = 5000
        elif num_col == 6:
            img_w_new = 6500
        else:
            img_w_new = width_early
        img_h_new = img_w_new * img.shape[0] // img.shape[1]

        if img_h_new >= 8000:
            img_new = np.copy(img)
            num_column_is_classified = False
        else:
            img_new = resize_image(img, img_h_new, img_w_new)
            num_column_is_classified = True

        return img_new, num_column_is_classified
    
    def early_page_for_num_of_column_classification(self,img_bin):
        self.logger.debug("enter early_page_for_num_of_column_classification")
        if self.input_binary:
            img = np.copy(img_bin).astype(np.uint8)
        else:
            img = self.imread()
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img_page_prediction = self.do_prediction(False, img, self.model_zoo.get('page'))

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
        return cropped_page, page_coord
    
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
    
    def resize_and_enhance_image_with_column_classifier(self):
        self.logger.debug("enter resize_and_enhance_image_with_column_classifier")
        dpi = 0#self.dpi
        self.logger.info("Detected %s DPI", dpi)
        if self.input_binary:
            img = self.imread()
            prediction_bin = self.do_prediction(True, img, self.model_zoo.get('binarization'), n_batch_inference=5)
            prediction_bin = 255 * (prediction_bin[:,:,0]==0)
            prediction_bin = np.repeat(prediction_bin[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            img= np.copy(prediction_bin)
            img_bin = prediction_bin
        else:
            img = self.imread()
            self.h_org, self.w_org = img.shape[:2]
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

            label_p_pred = self.model_zoo.get('col_classifier').predict(img_in, verbose=0)
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

            label_p_pred = self.model_zoo.get('col_classifier').predict(img_in, verbose=0)
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
            num_column_is_classified = True
            image_res = np.copy(img)
            is_image_enhanced = False

        self.logger.debug("exit resize_and_enhance_image_with_column_classifier")
        return is_image_enhanced, img, image_res, num_col, num_column_is_classified, img_bin
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
    
    def run_enhancement(self):
        t_in = time.time()
        self.logger.info("Resizing and enhancing image...")
        is_image_enhanced, img_org, img_res, num_col_classifier, num_column_is_classified, img_bin = \
            self.resize_and_enhance_image_with_column_classifier()
        
        self.logger.info("Image was %senhanced.", '' if is_image_enhanced else 'not ')
        return img_res, is_image_enhanced, num_col_classifier, num_column_is_classified


    def run_single(self):
        t0 = time.time()
        img_res, is_image_enhanced, num_col_classifier, num_column_is_classified = self.run_enhancement()
        
        return img_res, is_image_enhanced
        
        
    def run(self,
            overwrite: bool = False,
            image_filename: Optional[str] = None,
            dir_in: Optional[str] = None,
            dir_out: Optional[str] = None,
    ):
        """
        Get image and scales, then extract the page of scanned image
        """
        self.logger.debug("enter run")
        t0_tot = time.time()

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
            #print("text region early -11 in %.1fs", time.time() - t0)
            
            if os.path.exists(self.output_filename):
                if overwrite:
                    self.logger.warning("will overwrite existing output file '%s'", self.output_filename)
                else:
                    self.logger.warning("will skip input for existing output file '%s'", self.output_filename)
                    continue

            did_resize = False
            image_enhanced, did_enhance = self.run_single()
            if self.save_org_scale:
                image_enhanced = resize_image(image_enhanced, self.h_org, self.w_org)
                did_resize = True

            self.logger.info(
                "Image %s was %senhanced%s.",
                img_filename,
                '' if did_enhance else 'not ',
                'and resized' if did_resize else ''
            )
            
            cv2.imwrite(self.output_filename, image_enhanced)
            
