"""
Image enhancer. The output can be written as same scale of input or in new predicted scale.
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
from loky import ProcessPoolExecutor
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from ocrd import OcrdPage
from ocrd_utils import getLogger, tf_disable_interactive_logs
import statistics
from tensorflow.keras.models import load_model
from .utils.resize import resize_image
from .utils import (
    crop_image_inside_box
)

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

DPI_THRESHOLD = 298
KERNEL = np.ones((5, 5), np.uint8)


class machine_based_reading_order_on_layout:
    def __init__(
        self,
        dir_models : str,
        dir_out : Optional[str] = None,
        logger : Optional[Logger] = None,
    ):
        self.dir_out = dir_out
            
        self.logger = logger if logger else getLogger('mbro on layout')
        # for parallelization of CPU-intensive tasks:
        self.executor = ProcessPoolExecutor(max_workers=cpu_count(), timeout=1200)
        atexit.register(self.executor.shutdown)
        self.dir_models = dir_models
        self.model_reading_order_dir = dir_models + "/model_eynollah_reading_order_20250824"#"/model_ens_reading_order_machine_based"
        
        try:
            for device in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(device, True)
        except:
            self.logger.warning("no GPU device available")
            
        self.model_reading_order = self.our_load_model(self.model_reading_order_dir)
        self.light_version = True


    def cache_images(self, image_filename=None, image_pil=None, dpi=None):
        ret = {}
        t_c0 = time.time()
        if image_filename:
            ret['img'] = cv2.imread(image_filename)
            if self.light_version:
                self.dpi = 100
            else:
                self.dpi = 0#check_dpi(image_filename)
        else:
            ret['img'] = pil2cv(image_pil)
            if self.light_version:
                self.dpi = 100
            else:
                self.dpi = 0#check_dpi(image_pil)
        ret['img_grayscale'] = cv2.cvtColor(ret['img'], cv2.COLOR_BGR2GRAY)
        for prefix in ('',  '_grayscale'):
            ret[f'img{prefix}_uint8'] = ret[f'img{prefix}'].astype(np.uint8)
        self._imgs = ret
        if dpi is not None:
            self.dpi = dpi

    def reset_file_name_dir(self, image_filename):
        t_c = time.time()
        self.cache_images(image_filename=image_filename)
        self.output_filename = os.path.join(self.dir_out, Path(image_filename).stem +'.png')

    def imread(self, grayscale=False, uint8=True):
        key = 'img'
        if grayscale:
            key += '_grayscale'
        if uint8:
            key += '_uint8'
        return self._imgs[key].copy()

    def isNaN(self, num):
        return num != num

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
    
    def predict_enhancement(self, img):
        self.logger.debug("enter predict_enhancement")

        img_height_model = self.model_enhancement.layers[-1].output_shape[1]
        img_width_model = self.model_enhancement.layers[-1].output_shape[2]
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
    
    def resize_and_enhance_image_with_column_classifier(self, light_version):
        self.logger.debug("enter resize_and_enhance_image_with_column_classifier")
        dpi = 0#self.dpi
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
            num_column_is_classified = True
            image_res = np.copy(img)
            is_image_enhanced = False

        self.logger.debug("exit resize_and_enhance_image_with_column_classifier")
        return is_image_enhanced, img, image_res, num_col, num_column_is_classified, img_bin
    def read_xml(self, xml_file):
        file_name = Path(xml_file).stem
        tree1 = ET.parse(xml_file, parser = ET.XMLParser(encoding='utf-8'))
        root1=tree1.getroot()
        alltags=[elem.tag for elem in root1.iter()]
        link=alltags[0].split('}')[0]+'}'

        index_tot_regions = []
        tot_region_ref = []

        for jj in root1.iter(link+'Page'):
            y_len=int(jj.attrib['imageHeight'])
            x_len=int(jj.attrib['imageWidth'])

        for jj in root1.iter(link+'RegionRefIndexed'):
            index_tot_regions.append(jj.attrib['index'])
            tot_region_ref.append(jj.attrib['regionRef'])
            
        if (link+'PrintSpace' in alltags) or  (link+'Border' in alltags):
            co_printspace = []
            if link+'PrintSpace' in alltags:
                region_tags_printspace = np.unique([x for x in alltags if x.endswith('PrintSpace')])
            elif link+'Border' in alltags:
                region_tags_printspace = np.unique([x for x in alltags if x.endswith('Border')])
                
            for tag in region_tags_printspace:
                if link+'PrintSpace' in alltags:
                    tag_endings_printspace = ['}PrintSpace','}printspace']
                elif link+'Border' in alltags:
                    tag_endings_printspace = ['}Border','}border']
                    
                if tag.endswith(tag_endings_printspace[0]) or tag.endswith(tag_endings_printspace[1]):
                    for nn in root1.iter(tag):
                        c_t_in = []
                        sumi = 0
                        for vv in nn.iter():
                            # check the format of coords
                            if vv.tag == link + 'Coords':
                                coords = bool(vv.attrib)
                                if coords:
                                    p_h = vv.attrib['points'].split(' ')
                                    c_t_in.append(
                                        np.array([[int(x.split(',')[0]), int(x.split(',')[1])] for x in p_h]))
                                    break
                                else:
                                    pass

                            if vv.tag == link + 'Point':
                                c_t_in.append([int(float(vv.attrib['x'])), int(float(vv.attrib['y']))])
                                sumi += 1
                            elif vv.tag != link + 'Point' and sumi >= 1:
                                break
                        co_printspace.append(np.array(c_t_in))
            img_printspace = np.zeros( (y_len,x_len,3) ) 
            img_printspace=cv2.fillPoly(img_printspace, pts =co_printspace, color=(1,1,1))
            img_printspace = img_printspace.astype(np.uint8)
            
            imgray = cv2.cvtColor(img_printspace, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgray, 0, 255, 0)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])
            cnt = contours[np.argmax(cnt_size)]
            x, y, w, h = cv2.boundingRect(cnt)
            
            bb_coord_printspace = [x, y, w, h]
                        
        else:
            bb_coord_printspace = None
                        

        region_tags=np.unique([x for x in alltags if x.endswith('Region')])   
        co_text_paragraph=[]
        co_text_drop=[]
        co_text_heading=[]
        co_text_header=[]
        co_text_marginalia=[]
        co_text_catch=[]
        co_text_page_number=[]
        co_text_signature_mark=[]
        co_sep=[]
        co_img=[]
        co_table=[]
        co_graphic=[]
        co_graphic_text_annotation=[]
        co_graphic_decoration=[]
        co_noise=[]

        co_text_paragraph_text=[]
        co_text_drop_text=[]
        co_text_heading_text=[]
        co_text_header_text=[]
        co_text_marginalia_text=[]
        co_text_catch_text=[]
        co_text_page_number_text=[]
        co_text_signature_mark_text=[]
        co_sep_text=[]
        co_img_text=[]
        co_table_text=[]
        co_graphic_text=[]
        co_graphic_text_annotation_text=[]
        co_graphic_decoration_text=[]
        co_noise_text=[]

        id_paragraph = []
        id_header = []
        id_heading = []
        id_marginalia = []

        for tag in region_tags:
            if tag.endswith('}TextRegion') or tag.endswith('}Textregion'):
                for nn in root1.iter(tag):
                    for child2 in nn:
                        tag2 = child2.tag
                        if tag2.endswith('}TextEquiv') or tag2.endswith('}TextEquiv'):
                            for childtext2 in child2:
                                if childtext2.tag.endswith('}Unicode') or childtext2.tag.endswith('}Unicode'):
                                    if "type" in nn.attrib and nn.attrib['type']=='drop-capital':
                                        co_text_drop_text.append(childtext2.text)
                                    elif "type" in nn.attrib and nn.attrib['type']=='heading':
                                        co_text_heading_text.append(childtext2.text)
                                    elif "type" in nn.attrib and nn.attrib['type']=='signature-mark':
                                        co_text_signature_mark_text.append(childtext2.text)
                                    elif "type" in nn.attrib and nn.attrib['type']=='header':
                                        co_text_header_text.append(childtext2.text)
                                    ###elif "type" in nn.attrib and nn.attrib['type']=='catch-word':
                                        ###co_text_catch_text.append(childtext2.text)
                                    ###elif "type" in nn.attrib and nn.attrib['type']=='page-number':
                                        ###co_text_page_number_text.append(childtext2.text)
                                    elif "type" in nn.attrib and nn.attrib['type']=='marginalia':
                                        co_text_marginalia_text.append(childtext2.text)
                                    else:
                                        co_text_paragraph_text.append(childtext2.text)
                    c_t_in_drop=[]
                    c_t_in_paragraph=[]
                    c_t_in_heading=[]
                    c_t_in_header=[]
                    c_t_in_page_number=[]
                    c_t_in_signature_mark=[]
                    c_t_in_catch=[]
                    c_t_in_marginalia=[]


                    sumi=0
                    for vv in nn.iter():
                        # check the format of coords
                        if vv.tag==link+'Coords':

                            coords=bool(vv.attrib)
                            if coords:
                                #print('birda1')
                                p_h=vv.attrib['points'].split(' ')



                                if "type" in nn.attrib and nn.attrib['type']=='drop-capital':

                                    c_t_in_drop.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )

                                elif "type" in nn.attrib and nn.attrib['type']=='heading':
                                    ##id_heading.append(nn.attrib['id'])
                                    c_t_in_heading.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )


                                elif "type" in nn.attrib and nn.attrib['type']=='signature-mark':

                                    c_t_in_signature_mark.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                    #print(c_t_in_paragraph)
                                elif "type" in nn.attrib and nn.attrib['type']=='header':
                                    #id_header.append(nn.attrib['id'])
                                    c_t_in_header.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )


                                ###elif "type" in nn.attrib and nn.attrib['type']=='catch-word':
                                    ###c_t_in_catch.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )


                                ###elif "type" in nn.attrib and nn.attrib['type']=='page-number':

                                    ###c_t_in_page_number.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )

                                elif "type" in nn.attrib and nn.attrib['type']=='marginalia':
                                    #id_marginalia.append(nn.attrib['id'])

                                    c_t_in_marginalia.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                else:
                                    #id_paragraph.append(nn.attrib['id'])

                                    c_t_in_paragraph.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )

                                break
                            else:
                                pass


                        if vv.tag==link+'Point':
                            if "type" in nn.attrib and nn.attrib['type']=='drop-capital':

                                c_t_in_drop.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1

                            elif "type" in nn.attrib and nn.attrib['type']=='heading':
                                #id_heading.append(nn.attrib['id'])
                                c_t_in_heading.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1


                            elif "type" in nn.attrib and nn.attrib['type']=='signature-mark':

                                c_t_in_signature_mark.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1
                            elif "type" in nn.attrib and nn.attrib['type']=='header':
                                #id_header.append(nn.attrib['id'])
                                c_t_in_header.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1


                            ###elif "type" in nn.attrib and nn.attrib['type']=='catch-word':
                                ###c_t_in_catch.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                ###sumi+=1

                            ###elif "type" in nn.attrib and nn.attrib['type']=='page-number':

                                ###c_t_in_page_number.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                ###sumi+=1

                            elif "type" in nn.attrib and nn.attrib['type']=='marginalia':
                                #id_marginalia.append(nn.attrib['id'])

                                c_t_in_marginalia.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1

                            else:
                                #id_paragraph.append(nn.attrib['id'])
                                c_t_in_paragraph.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1

                        elif vv.tag!=link+'Point' and sumi>=1:
                            break

                    if len(c_t_in_drop)>0:
                        co_text_drop.append(np.array(c_t_in_drop))
                    if len(c_t_in_paragraph)>0:
                        co_text_paragraph.append(np.array(c_t_in_paragraph))
                        id_paragraph.append(nn.attrib['id'])
                    if len(c_t_in_heading)>0:
                        co_text_heading.append(np.array(c_t_in_heading))
                        id_heading.append(nn.attrib['id'])

                    if len(c_t_in_header)>0:
                        co_text_header.append(np.array(c_t_in_header))
                        id_header.append(nn.attrib['id'])
                    if len(c_t_in_page_number)>0:
                        co_text_page_number.append(np.array(c_t_in_page_number))
                    if len(c_t_in_catch)>0:
                        co_text_catch.append(np.array(c_t_in_catch))

                    if len(c_t_in_signature_mark)>0:
                        co_text_signature_mark.append(np.array(c_t_in_signature_mark))

                    if len(c_t_in_marginalia)>0:
                        co_text_marginalia.append(np.array(c_t_in_marginalia))
                        id_marginalia.append(nn.attrib['id'])


            elif tag.endswith('}GraphicRegion') or tag.endswith('}graphicregion'):
                for nn in root1.iter(tag):
                    c_t_in=[]
                    c_t_in_text_annotation=[]
                    c_t_in_decoration=[]
                    sumi=0
                    for vv in nn.iter():
                        # check the format of coords
                        if vv.tag==link+'Coords':
                            coords=bool(vv.attrib)
                            if coords:
                                p_h=vv.attrib['points'].split(' ')

                                if "type" in nn.attrib and nn.attrib['type']=='handwritten-annotation':
                                    c_t_in_text_annotation.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                    
                                elif "type" in nn.attrib and nn.attrib['type']=='decoration':
                                    c_t_in_decoration.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                    
                                else:
                                    c_t_in.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )


                                break
                            else:
                                pass


                        if vv.tag==link+'Point':
                            if "type" in nn.attrib and nn.attrib['type']=='handwritten-annotation':
                                c_t_in_text_annotation.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1

                            elif "type" in nn.attrib and nn.attrib['type']=='decoration':
                                c_t_in_decoration.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1
                                
                            else:
                                c_t_in.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1

                    if len(c_t_in_text_annotation)>0:
                        co_graphic_text_annotation.append(np.array(c_t_in_text_annotation))
                    if len(c_t_in_decoration)>0:
                        co_graphic_decoration.append(np.array(c_t_in_decoration))
                    if len(c_t_in)>0:
                        co_graphic.append(np.array(c_t_in))



            elif tag.endswith('}ImageRegion') or tag.endswith('}imageregion'):
                for nn in root1.iter(tag):
                    c_t_in=[]
                    sumi=0
                    for vv in nn.iter():
                        # check the format of coords
                        if vv.tag==link+'Coords':
                            coords=bool(vv.attrib)
                            if coords:
                                p_h=vv.attrib['points'].split(' ')
                                c_t_in.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                break
                            else:
                                pass


                        if vv.tag==link+'Point':
                            c_t_in.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                            sumi+=1
                        elif vv.tag!=link+'Point' and sumi>=1:
                            break
                    co_img.append(np.array(c_t_in))
                    co_img_text.append(' ')


            elif tag.endswith('}SeparatorRegion') or tag.endswith('}separatorregion'):
                for nn in root1.iter(tag):
                    c_t_in=[]
                    sumi=0
                    for vv in nn.iter():
                        # check the format of coords
                        if vv.tag==link+'Coords':
                            coords=bool(vv.attrib)
                            if coords:
                                p_h=vv.attrib['points'].split(' ')
                                c_t_in.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                break
                            else:
                                pass


                        if vv.tag==link+'Point':
                            c_t_in.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                            sumi+=1
                        elif vv.tag!=link+'Point' and sumi>=1:
                            break
                    co_sep.append(np.array(c_t_in))



            elif tag.endswith('}TableRegion') or tag.endswith('}tableregion'):
                for nn in root1.iter(tag):
                    c_t_in=[]
                    sumi=0
                    for vv in nn.iter():
                        # check the format of coords
                        if vv.tag==link+'Coords':
                            coords=bool(vv.attrib)
                            if coords:
                                p_h=vv.attrib['points'].split(' ')
                                c_t_in.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                break
                            else:
                                pass


                        if vv.tag==link+'Point':
                            c_t_in.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                            sumi+=1
                            
                        elif vv.tag!=link+'Point' and sumi>=1:
                            break
                    co_table.append(np.array(c_t_in))
                    co_table_text.append(' ')

            elif tag.endswith('}NoiseRegion') or tag.endswith('}noiseregion'):
                for nn in root1.iter(tag):
                    c_t_in=[]
                    sumi=0
                    for vv in nn.iter():
                        # check the format of coords
                        if vv.tag==link+'Coords':
                            coords=bool(vv.attrib)
                            if coords:
                                p_h=vv.attrib['points'].split(' ')
                                c_t_in.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                break
                            else:
                                pass


                        if vv.tag==link+'Point':
                            c_t_in.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                            sumi+=1

                        elif vv.tag!=link+'Point' and sumi>=1:
                            break
                    co_noise.append(np.array(c_t_in))
                    co_noise_text.append(' ')

        img = np.zeros( (y_len,x_len,3) ) 
        img_poly=cv2.fillPoly(img, pts =co_text_paragraph, color=(1,1,1))

        img_poly=cv2.fillPoly(img, pts =co_text_heading, color=(2,2,2))
        img_poly=cv2.fillPoly(img, pts =co_text_header, color=(2,2,2))
        img_poly=cv2.fillPoly(img, pts =co_text_marginalia, color=(3,3,3))
        img_poly=cv2.fillPoly(img, pts =co_img, color=(4,4,4))
        img_poly=cv2.fillPoly(img, pts =co_sep, color=(5,5,5))

        return tree1, root1, bb_coord_printspace, file_name, id_paragraph, id_header+id_heading, co_text_paragraph, co_text_header+co_text_heading,\
    tot_region_ref,x_len, y_len,index_tot_regions, img_poly

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
        ###img_poly[text_regions_p[:,:]==1] = 1
        ###img_poly[text_regions_p[:,:]==2] = 2
        ###img_poly[text_regions_p[:,:]==3] = 4
        ###img_poly[text_regions_p[:,:]==6] = 5
        
        ##img_poly[text_regions_p[:,:]==1] = 1
        ##img_poly[text_regions_p[:,:]==2] = 2
        ##img_poly[text_regions_p[:,:]==3] = 3
        ##img_poly[text_regions_p[:,:]==4] = 4
        ##img_poly[text_regions_p[:,:]==5] = 5
        
        img_poly = np.copy(text_regions_p)
        
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
        
        ##id_all_text = np.array(id_all_text)[index_sort]
        
        
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
    

        
        
    def run(self, xml_filename : Optional[str] = None, dir_in : Optional[str] = None, overwrite : bool = False):
        """
        Get image and scales, then extract the page of scanned image
        """
        self.logger.debug("enter run")
        t0_tot = time.time()

        if dir_in:
            self.ls_xmls  = os.listdir(dir_in)
        elif xml_filename:
            self.ls_xmls = [xml_filename]
        else:
            raise ValueError("run requires either a single image filename or a directory")

        for xml_filename in self.ls_xmls:
            self.logger.info(xml_filename)
            t0 = time.time()
            
            if dir_in:
                xml_file = os.path.join(dir_in, xml_filename)
            else:
                xml_file = xml_filename
            
            tree_xml, root_xml, bb_coord_printspace, file_name, id_paragraph, id_header, co_text_paragraph, co_text_header, tot_region_ref, x_len, y_len, index_tot_regions, img_poly = self.read_xml(xml_file)
            
            id_all_text = id_paragraph + id_header
            
            order_text_new, id_of_texts_tot = self.do_order_of_regions_with_model(co_text_paragraph, co_text_header, img_poly[:,:,0])
            
            id_all_text = np.array(id_all_text)[order_text_new]
            
            alltags=[elem.tag for elem in root_xml.iter()]
            
            
            
            link=alltags[0].split('}')[0]+'}'
            name_space = alltags[0].split('}')[0]
            name_space = name_space.split('{')[1]
            
            page_element = root_xml.find(link+'Page')
            
            
            old_ro = root_xml.find(".//{*}ReadingOrder")
            
            if old_ro is not None:
                page_element.remove(old_ro)
            
            #print(old_ro, 'old_ro')
            ro_subelement = ET.Element('ReadingOrder')
            
            ro_subelement2 = ET.SubElement(ro_subelement, 'OrderedGroup')
            ro_subelement2.set('id', "ro357564684568544579089")
            
            for index, id_text in enumerate(id_all_text):
                new_element_2 = ET.SubElement(ro_subelement2, 'RegionRefIndexed')
                new_element_2.set('regionRef', id_all_text[index])
                new_element_2.set('index', str(index))
            
            if (link+'PrintSpace' in alltags) or  (link+'Border' in alltags):
                page_element.insert(1, ro_subelement)
            else:
                page_element.insert(0, ro_subelement)
            
            alltags=[elem.tag for elem in root_xml.iter()]
            
            ET.register_namespace("",name_space)
            tree_xml.write(os.path.join(self.dir_out, file_name+'.xml'),xml_declaration=True,method='xml',encoding="utf8",default_namespace=None)
            
            #sys.exit()
            
