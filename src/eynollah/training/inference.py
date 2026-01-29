import sys
import os
from typing import Tuple
import warnings
import json

import numpy as np
import cv2
from numpy._typing import NDArray
import tensorflow as tf
from keras.models import Model, load_model
from keras import backend as K
import click
from tensorflow.python.keras import backend as tensorflow_backend
import xml.etree.ElementTree as ET

from .gt_gen_utils import (
    filter_contours_area_of_image,
    find_new_features_of_contours,
    read_xml,
    resize_image,
    update_list_and_return_first_with_length_bigger_than_one
)
from .models import (
    PatchEncoder,
    Patches
)

from.utils import (scale_padd_image_for_ocr)
from eynollah.utils.utils_ocr import (decode_batch_predictions)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
__doc__=\
"""
Tool to load model and predict for given image.
"""

class sbb_predict:
    def __init__(self,image, dir_in, model, task, config_params_model, patches, save, save_layout, ground_truth, xml_file, cpu, out, min_area):
        self.image=image
        self.dir_in=dir_in
        self.patches=patches
        self.save=save
        self.save_layout=save_layout
        self.model_dir=model
        self.ground_truth=ground_truth
        self.task=task
        self.config_params_model=config_params_model
        self.xml_file = xml_file
        self.out = out
        self.cpu = cpu
        if min_area:
            self.min_area = float(min_area)
        else:
            self.min_area = 0

    def resize_image(self,img_in,input_height,input_width):
        return cv2.resize( img_in, ( input_width,input_height) ,interpolation=cv2.INTER_NEAREST)
    
    
    def color_images(self,seg):
        ann_u=range(self.n_classes)
        if len(np.shape(seg))==3:
            seg=seg[:,:,0]
            
        seg_img=np.zeros((np.shape(seg)[0],np.shape(seg)[1],3)).astype(np.uint8)
        
        for c in ann_u:
            c=int(c)
            seg_img[:,:,0][seg==c]=c
            seg_img[:,:,1][seg==c]=c
            seg_img[:,:,2][seg==c]=c
        return seg_img
    
    def otsu_copy_binary(self,img):
        img_r=np.zeros((img.shape[0],img.shape[1],3))
        img1=img[:,:,0]

        #print(img.min())
        #print(img[:,:,0].min())
        #blur = cv2.GaussianBlur(img,(5,5))
        #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, threshold1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        

        img_r[:,:,0]=threshold1
        img_r[:,:,1]=threshold1
        img_r[:,:,2]=threshold1
        #img_r=img_r/float(np.max(img_r))*255
        return img_r
    
    def otsu_copy(self,img):
        img_r=np.zeros((img.shape[0],img.shape[1],3))
        #img1=img[:,:,0]

        #print(img.min())
        #print(img[:,:,0].min())
        #blur = cv2.GaussianBlur(img,(5,5))
        #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, threshold1 = cv2.threshold(img[:,:,0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, threshold2 = cv2.threshold(img[:,:,1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, threshold3 = cv2.threshold(img[:,:,2], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        

        img_r[:,:,0]=threshold1
        img_r[:,:,1]=threshold2
        img_r[:,:,2]=threshold3
        ###img_r=img_r/float(np.max(img_r))*255
        return img_r
    
    def soft_dice_loss(self,y_true, y_pred, epsilon=1e-6): 

        axes = tuple(range(1, len(y_pred.shape)-1))
        
        numerator = 2. * K.sum(y_pred * y_true, axes)
    
        denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
        return 1.00 - K.mean(numerator / (denominator + epsilon)) # average over classes and batch
    
    # def weighted_categorical_crossentropy(self,weights=None):
    #
    #     def loss(y_true, y_pred):
    #         labels_floats = tf.cast(y_true, tf.float32)
    #         per_pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_floats,logits=y_pred)
    #     
    #         if weights is not None:
    #             weight_mask = tf.maximum(tf.reduce_max(tf.constant(
    #                 np.array(weights, dtype=np.float32)[None, None, None])
    #                 * labels_floats, axis=-1), 1.0)
    #             per_pixel_loss = per_pixel_loss * weight_mask[:, :, :, None]
    #         return tf.reduce_mean(per_pixel_loss)
    #     return self.loss


    def IoU(self,Yi,y_predi):
        ## mean Intersection over Union
        ## Mean IoU = TP/(FN + TP + FP)
    
        IoUs = []
        Nclass = np.unique(Yi)
        for c in Nclass:
            TP = np.sum( (Yi == c)&(y_predi==c) )
            FP = np.sum( (Yi != c)&(y_predi==c) )
            FN = np.sum( (Yi == c)&(y_predi != c)) 
            IoU = TP/float(TP + FP + FN)
            if self.n_classes>2:
                print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c,TP,FP,FN,IoU))
            IoUs.append(IoU)
        if self.n_classes>2:
            mIoU = np.mean(IoUs)
            print("_________________")
            print("Mean IoU: {:4.3f}".format(mIoU))
            return mIoU
        elif self.n_classes==2:
            mIoU = IoUs[1]
            print("_________________")
            print("IoU: {:4.3f}".format(mIoU))
            return mIoU
            
    def start_new_session_and_model(self):
        if self.task == "cnn-rnn-ocr":
            if self.cpu:
                os.environ['CUDA_VISIBLE_DEVICES']='-1'
            self.model = load_model(self.model_dir)
            self.model = tf.keras.models.Model(
                            self.model.get_layer(name = "image").input, 
                            self.model.get_layer(name = "dense2").output)
        else:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True

            session = tf.compat.v1.Session(config=config)  # tf.InteractiveSession()
            tensorflow_backend.set_session(session)

        
        ##if self.weights_dir!=None:
            ##self.model.load_weights(self.weights_dir)
            
        assert isinstance(self.model, Model)
        if self.task != 'classification' and self.task != 'reading_order':
            self.img_height=self.model.layers[len(self.model.layers)-1].output_shape[1]
            self.img_width=self.model.layers[len(self.model.layers)-1].output_shape[2]
            self.n_classes=self.model.layers[len(self.model.layers)-1].output_shape[3]
        
    def visualize_model_output(self, prediction, img, task) -> Tuple[NDArray, NDArray]:
        if task == "binarization":
            prediction = prediction * -1
            prediction = prediction + 1
            added_image = prediction * 255
            layout_only = None
        else:
            unique_classes = np.unique(prediction[:,:,0])
            rgb_colors = {'0' : [255, 255, 255],
                        '1' : [255, 0, 0],
                        '2' : [255, 125, 0],
                        '3' : [255, 0, 125],
                        '4' : [125, 125, 125],
                        '5' : [125, 125, 0],
                        '6' : [0, 125, 255],
                        '7' : [0, 125, 0],
                        '8' : [125, 125, 125],
                        '9' : [0, 125, 255],
                        '10' : [125, 0, 125],
                        '11' : [0, 255, 0],
                        '12' : [0, 0, 255],
                        '13' : [0, 255, 255],
                        '14' : [255, 125, 125],
                        '15' : [255, 0, 255]}
        
            layout_only = np.zeros(prediction.shape)
        
            for unq_class in unique_classes:
                rgb_class_unique = rgb_colors[str(int(unq_class))]
                layout_only[:,:,0][prediction[:,:,0]==unq_class] = rgb_class_unique[0]
                layout_only[:,:,1][prediction[:,:,0]==unq_class] = rgb_class_unique[1]
                layout_only[:,:,2][prediction[:,:,0]==unq_class] = rgb_class_unique[2]
        
        
        
            img = self.resize_image(img, layout_only.shape[0], layout_only.shape[1])
        
            layout_only = layout_only.astype(np.int32)
            img = img.astype(np.int32)
        
            
            
            added_image = cv2.addWeighted(img,0.5,layout_only,0.1,0)
            
        assert isinstance(added_image, np.ndarray)
        assert isinstance(layout_only, np.ndarray)
        return added_image, layout_only

    def predict(self, image_dir):
        assert isinstance(self.model, Model)
        if self.task == 'classification':
            classes_names = self.config_params_model['classification_classes_name']
            img_1ch = img=cv2.imread(image_dir, 0)

            img_1ch = img_1ch / 255.0
            img_1ch = cv2.resize(img_1ch, (self.config_params_model['input_height'], self.config_params_model['input_width']), interpolation=cv2.INTER_NEAREST)
            img_in = np.zeros((1, img_1ch.shape[0], img_1ch.shape[1], 3))
            img_in[0, :, :, 0] = img_1ch[:, :]
            img_in[0, :, :, 1] = img_1ch[:, :]
            img_in[0, :, :, 2] = img_1ch[:, :]
                      
            label_p_pred = self.model.predict(img_in, verbose='0')
            index_class = np.argmax(label_p_pred[0])
            
            print("Predicted Class: {}".format(classes_names[str(int(index_class))]))
        elif self.task == "cnn-rnn-ocr":
            img=cv2.imread(image_dir)
            img = scale_padd_image_for_ocr(img, self.config_params_model['input_height'], self.config_params_model['input_width'])
            
            img = img / 255.
            
            with open(os.path.join(self.model_dir, "characters_org.txt"), 'r') as char_txt_f:
                characters = json.load(char_txt_f)
                
            AUTOTUNE = tf.data.AUTOTUNE

            # Mapping characters to integers.
            char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
            
            # Mapping integers back to original characters.
            num_to_char = StringLookup(
                vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
            )
            preds = self.model.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]), verbose=0)
            pred_texts = decode_batch_predictions(preds, num_to_char)
            pred_texts = pred_texts[0].replace("[UNK]", "")
            return pred_texts
            
            
        elif self.task == 'reading_order':
            img_height = self.config_params_model['input_height']
            img_width = self.config_params_model['input_width']
            
            tree_xml, root_xml, bb_coord_printspace, file_name, id_paragraph, id_header, co_text_paragraph, co_text_header, tot_region_ref, x_len, y_len, index_tot_regions, img_poly = read_xml(self.xml_file)
            _, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, _ = find_new_features_of_contours(co_text_header)
            
            img_header_and_sep = np.zeros((y_len,x_len), dtype='uint8')
            

            for j in range(len(cy_main)):
                img_header_and_sep[int(y_max_main[j]):int(y_max_main[j])+12,int(x_min_main[j]):int(x_max_main[j]) ] = 1
                
            co_text_all = co_text_paragraph + co_text_header
            id_all_text = id_paragraph + id_header
            

            ##texts_corr_order_index  = [index_tot_regions[tot_region_ref.index(i)] for i in id_all_text ]
            ##texts_corr_order_index_int = [int(x) for x in texts_corr_order_index]
            texts_corr_order_index_int = list(np.array(range(len(co_text_all))))
            
            #print(texts_corr_order_index_int)
            
            max_area = 1
            #print(np.shape(co_text_all[0]), len( np.shape(co_text_all[0]) ),'co_text_all')
            #co_text_all = filter_contours_area_of_image_tables(img_poly, co_text_all, _, max_area, min_area)
            #print(co_text_all,'co_text_all')
            co_text_all, texts_corr_order_index_int, _ = filter_contours_area_of_image(img_poly, co_text_all, texts_corr_order_index_int, max_area, self.min_area)
            
            #print(texts_corr_order_index_int)
            
            #co_text_all = [co_text_all[index] for index in texts_corr_order_index_int]
            id_all_text = [id_all_text[index] for index in texts_corr_order_index_int]
            
            labels_con = np.zeros((y_len,x_len,len(co_text_all)),dtype='uint8')
            for i in range(len(co_text_all)):
                img_label = np.zeros((y_len,x_len,3),dtype='uint8')
                img_label=cv2.fillPoly(img_label, pts =[co_text_all[i]], color=(1,1,1))
                labels_con[:,:,i] = img_label[:,:,0]
                
            if bb_coord_printspace:
                #bb_coord_printspace[x,y,w,h,_,_]
                x = bb_coord_printspace[0]
                y = bb_coord_printspace[1]
                w = bb_coord_printspace[2]
                h = bb_coord_printspace[3]
                labels_con = labels_con[y:y+h, x:x+w, :]
                img_poly = img_poly[y:y+h, x:x+w, :]
                img_header_and_sep = img_header_and_sep[y:y+h, x:x+w]
                

                
            img3= np.copy(img_poly)
            labels_con = resize_image(labels_con, img_height, img_width)

            img_header_and_sep = resize_image(img_header_and_sep, img_height, img_width)

            img3= resize_image (img3, img_height, img_width)
            img3 = img3.astype(np.uint16)
            
            inference_bs = 1#4

            input_1= np.zeros( (inference_bs, img_height, img_width,3))


            starting_list_of_regions = [list(range(labels_con.shape[2]))]

            index_update = 0
            index_selected = starting_list_of_regions[0]
            
            scalibility_num = 0
            while index_update>=0:
                ij_list = starting_list_of_regions[index_update] 
                i = ij_list[0]
                ij_list.pop(0)
                
                
                pr_list = []
                post_list = []
                
                batch_counter = 0
                tot_counter = 1
                
                tot_iteration = len(ij_list)
                full_bs_ite= tot_iteration//inference_bs
                last_bs = tot_iteration % inference_bs
                
                jbatch_indexer =[]
                for j in ij_list:
                    img1= np.repeat(labels_con[:,:,i][:, :, np.newaxis], 3, axis=2)
                    img2 = np.repeat(labels_con[:,:,j][:, :, np.newaxis], 3, axis=2)

                    
                    img2[:,:,0][img3[:,:,0]==5] = 2
                    img2[:,:,0][img_header_and_sep[:,:]==1] = 3
                    
                    
                    
                    img1[:,:,0][img3[:,:,0]==5] = 2
                    img1[:,:,0][img_header_and_sep[:,:]==1] = 3
                    
                    #input_1= np.zeros( (height1, width1,3))
                    

                    jbatch_indexer.append(j)
                        
                    input_1[batch_counter,:,:,0] = img1[:,:,0]/3.
                    input_1[batch_counter,:,:,2] = img2[:,:,0]/3.
                    input_1[batch_counter,:,:,1] = img3[:,:,0]/5.
                    #input_1[batch_counter,:,:,:]= np.zeros( (batch_counter, height1, width1,3))
                    batch_counter = batch_counter+1
                    
                    #input_1[:,:,0] = img1[:,:,0]/3.
                    #input_1[:,:,2] = img2[:,:,0]/3.
                    #input_1[:,:,1] = img3[:,:,0]/5.
                    
                    if batch_counter==inference_bs or ( (tot_counter//inference_bs)==full_bs_ite and tot_counter%inference_bs==last_bs):
                        y_pr = self.model.predict(input_1 , verbose='0')
                        scalibility_num = scalibility_num+1
                        
                        if batch_counter==inference_bs:
                            iteration_batches = inference_bs
                        else:
                            iteration_batches = last_bs
                        for jb in range(iteration_batches):
                            if y_pr[jb][0]>=0.5:
                                post_list.append(jbatch_indexer[jb])
                            else:
                                pr_list.append(jbatch_indexer[jb])
                                
                        batch_counter = 0
                        jbatch_indexer = []
                        
                    tot_counter = tot_counter+1
                        
                starting_list_of_regions, index_update = update_list_and_return_first_with_length_bigger_than_one(index_update, i, pr_list, post_list,starting_list_of_regions)
            
            
            index_sort = [i[0] for i in starting_list_of_regions ]
            
            id_all_text = np.array(id_all_text)[index_sort]
            
            alltags=[elem.tag for elem in root_xml.iter()]
            
            
            
            link=alltags[0].split('}')[0]+'}'
            name_space = alltags[0].split('}')[0]
            name_space = name_space.split('{')[1]
            
            page_element = root_xml.find(link+'Page')
            assert isinstance(page_element, ET.Element)
            
            """
            ro_subelement = ET.SubElement(page_element, 'ReadingOrder')
            #print(page_element, 'page_element')
            
            #new_element = ET.Element('ReadingOrder')
            
            new_element_element = ET.Element('OrderedGroup')
            new_element_element.set('id', "ro357564684568544579089")
            
            for index, id_text in enumerate(id_all_text):
                new_element_2 = ET.Element('RegionRefIndexed')
                new_element_2.set('regionRef', id_all_text[index])
                new_element_2.set('index', str(index_sort[index]))
            
                new_element_element.append(new_element_2)
            
            ro_subelement.append(new_element_element)
            """
            ##ro_subelement = ET.SubElement(page_element, 'ReadingOrder')
            
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
            tree_xml.write(os.path.join(self.out, file_name+'.xml'),xml_declaration=True,method='xml',encoding="utf8",default_namespace=None)
            #tree_xml.write('library2.xml')
            
        else:
            if self.patches:
                #def textline_contours(img,input_width,input_height,n_classes,model):
                
                img=cv2.imread(image_dir)
                self.img_org = np.copy(img)
                
                if img.shape[0] < self.img_height:
                    img = self.resize_image(img, self.img_height, img.shape[1])

                if img.shape[1] < self.img_width:
                    img = self.resize_image(img, img.shape[0], self.img_width)
                    
                margin = int(0.1 * self.img_width)
                width_mid = self.img_width - 2 * margin
                height_mid = self.img_height - 2 * margin
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
                            index_x_u = index_x_d + self.img_width
                        else:
                            index_x_d = i * width_mid
                            index_x_u = index_x_d + self.img_width
                        if j == 0:
                            index_y_d = j * height_mid
                            index_y_u = index_y_d + self.img_height
                        else:
                            index_y_d = j * height_mid
                            index_y_u = index_y_d + self.img_height

                        if index_x_u > img_w:
                            index_x_u = img_w
                            index_x_d = img_w - self.img_width
                        if index_y_u > img_h:
                            index_y_u = img_h
                            index_y_d = img_h - self.img_height

                        img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
                        label_p_pred = self.model.predict(img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2]),
                                                                verbose='0')
                        
                        if self.task == 'enhancement':
                            seg = label_p_pred[0, :, :, :]
                            seg = seg * 255
                        elif self.task == 'segmentation' or self.task == 'binarization':
                            seg = np.argmax(label_p_pred, axis=3)[0]
                            seg = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
                        else:
                            raise ValueError(f"Unhandled task {self.task}")
                            

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
                prediction_true = cv2.resize(prediction_true, (self.img_org.shape[1], self.img_org.shape[0]), interpolation=cv2.INTER_NEAREST)
                return prediction_true

            else:

                img=cv2.imread(image_dir)
                self.img_org = np.copy(img)

                width=self.img_width
                height=self.img_height

                img=img/255.0
                img=self.resize_image(img,self.img_height,self.img_width)
                

                label_p_pred=self.model.predict(
                    img.reshape(1,img.shape[0],img.shape[1],img.shape[2]))

                if self.task == 'enhancement':
                    seg = label_p_pred[0, :, :, :]
                    seg = seg * 255
                elif self.task == 'segmentation' or self.task == 'binarization':
                    seg = np.argmax(label_p_pred, axis=3)[0]
                    seg = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
                else:
                    raise ValueError(f"Unhandled task {self.task}")
                    
                prediction_true = seg.astype(int)

                prediction_true = cv2.resize(prediction_true, (self.img_org.shape[1], self.img_org.shape[0]), interpolation=cv2.INTER_NEAREST)
                return prediction_true



    def run(self):
        self.start_new_session_and_model()
        if self.image:
            res=self.predict(image_dir = self.image)
            
            if self.task == 'classification' or self.task == 'reading_order':
                pass
            elif self.task == 'enhancement':
                if self.save:
                    cv2.imwrite(self.save,res)
            elif self.task == "cnn-rnn-ocr":
                print(f"Detected text: {res}")
            else:
                img_seg_overlayed, only_layout  = self.visualize_model_output(res, self.img_org, self.task)
                if self.save:
                    cv2.imwrite(self.save,img_seg_overlayed)
                if self.save_layout:
                    cv2.imwrite(self.save_layout, only_layout)
                    
                if self.ground_truth:
                    gt_img=cv2.imread(self.ground_truth)
                    self.IoU(gt_img[:,:,0],res[:,:,0])
            
        else:
            ls_images = os.listdir(self.dir_in)
            for ind_image in ls_images:
                f_name = ind_image.split('.')[0]
                image_dir = os.path.join(self.dir_in, ind_image)
                res=self.predict(image_dir)
                
                if self.task == 'classification' or self.task == 'reading_order':
                    pass
                elif self.task == 'enhancement':
                    self.save = os.path.join(self.out, f_name+'.png')
                    cv2.imwrite(self.save,res)
                elif self.task == "cnn-rnn-ocr":
                    print(f"Detected text for file name {f_name} is: {res}")
                else:
                    img_seg_overlayed, only_layout  = self.visualize_model_output(res, self.img_org, self.task)
                    self.save = os.path.join(self.out, f_name+'_overlayed.png')
                    cv2.imwrite(self.save,img_seg_overlayed)
                    self.save_layout = os.path.join(self.out, f_name+'_layout.png')
                    cv2.imwrite(self.save_layout, only_layout)
                        
                    if self.ground_truth:
                        gt_img=cv2.imread(self.ground_truth)
                        self.IoU(gt_img[:,:,0],res[:,:,0])
            

        
@click.command()
@click.option(
    "--image",
    "-i",
    help="image filename",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--dir_in",
    "-di",
    help="directory of images",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--out",
    "-o",
    help="output directory where xml with detected reading order will be written.",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--patches/--no-patches",
    "-p/-nop",
    is_flag=True,
    help="if this parameter set to true, this tool will try to do inference in patches.",
)
@click.option(
    "--save",
    "-s",
    help="save prediction as a png file in current folder.",
)
@click.option(
    "--save_layout",
    "-sl",
    help="save layout prediction only as a png file in current folder.",
)
@click.option(
    "--model",
    "-m",
    help="directory of models",
    type=click.Path(exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--ground_truth",
    "-gt",
    help="ground truth directory if you want to see the iou of prediction.",
)
@click.option(
    "--xml_file",
    "-xml",
    help="xml file with layout coordinates that reading order detection will be implemented on. The result will be written in the same xml file.",
)
@click.option(
    "--cpu",
    "-cpu",
    help="For OCR, the default device is the GPU. If this parameter is set to true, inference will be performed on the CPU",
    is_flag=True,
)
@click.option(
    "--min_area",
    "-min",
    help="min area size of regions considered for reading order detection. The default value is zero and means that all text regions are considered for reading order.",
)
def main(image, dir_in, model, patches, save, save_layout, ground_truth, xml_file, cpu, out, min_area):
    assert image or dir_in, "Either a single image -i or a dir_in -di is required"
    with open(os.path.join(model,'config.json')) as f:
        config_params_model = json.load(f)
    task = config_params_model['task']
    if task != 'classification' and task != 'reading_order' and task != "cnn-rnn-ocr":
        if image and not save:
            print("Error: You used one of segmentation or binarization task with image input but not set -s, you need a filename to save visualized output with -s")
            sys.exit(1)
        if dir_in and not out:
            print("Error: You used one of segmentation or binarization task with dir_in but not set -out")
            sys.exit(1)
    x=sbb_predict(image, dir_in, model, task, config_params_model, patches, save, save_layout, ground_truth, xml_file, cpu, out, min_area)
    x.run()

