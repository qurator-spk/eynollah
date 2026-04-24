"""
Tool to load model and binarize a given image.
"""

# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportPossiblyUnboundVariable=false

import os
import logging
from pathlib import Path 
from typing import Optional

import numpy as np
import cv2

os.environ['TF_USE_LEGACY_KERAS'] = '1' # avoid Keras 3 after TF 2.15
from ocrd_utils import tf_disable_interactive_logs
tf_disable_interactive_logs()
import tensorflow as tf

from .model_zoo import EynollahModelZoo
from .utils import is_image_filename

def resize_image(img_in, input_height, input_width):
    return cv2.resize(img_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)

class SbbBinarizer:

    def __init__(
        self,
        *,
        model_zoo: EynollahModelZoo,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger if logger else logging.getLogger('eynollah.binarization')
        try:
            for device in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(device, True)
        except:
            self.logger.warning("no GPU device available")
        self.models = (model_zoo.model_path('binarization'), model_zoo.load_model('binarization'))
        self.logger.info('Loaded model %s [%s]', self.models[1], self.models[0])

    def predict(self, model, img, use_patches, n_batch_inference=5):
        model_height = model.layers[len(model.layers)-1].output_shape[1]
        model_width = model.layers[len(model.layers)-1].output_shape[2]
        
        img_org_h = img.shape[0]
        img_org_w = img.shape[1]
        
        if img.shape[0] < model_height and img.shape[1] >= model_width:
            img_padded = np.zeros(( model_height, img.shape[1], img.shape[2] ))
            
            index_start_h =  int( abs( img.shape[0] - model_height) /2.)
            index_start_w = 0
            
            img_padded [ index_start_h: index_start_h+img.shape[0], :, : ] = img[:,:,:]
            
        elif img.shape[0] >= model_height and img.shape[1] < model_width:
            img_padded = np.zeros(( img.shape[0], model_width, img.shape[2] ))
            
            index_start_h =  0 
            index_start_w = int( abs( img.shape[1] - model_width) /2.)
            
            img_padded [ :, index_start_w: index_start_w+img.shape[1], : ] = img[:,:,:]
            
            
        elif img.shape[0] < model_height and img.shape[1] < model_width:
            img_padded = np.zeros(( model_height, model_width, img.shape[2] ))
            
            index_start_h =  int( abs( img.shape[0] - model_height) /2.)
            index_start_w = int( abs( img.shape[1] - model_width) /2.)
            
            img_padded [ index_start_h: index_start_h+img.shape[0], index_start_w: index_start_w+img.shape[1], : ] = img[:,:,:]
            
        else:
            index_start_h = 0
            index_start_w  = 0
            img_padded = np.copy(img)
            
            
        img = np.copy(img_padded)
        
            

        if use_patches:

            margin = int(0.1 * model_width)

            width_mid = model_width - 2 * margin
            height_mid = model_height - 2 * margin


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
                
                
            list_i_s = []
            list_j_s = []
            list_x_u = []
            list_x_d = []
            list_y_u = []
            list_y_d = []
            
            batch_indexer = 0
            
            img_patch = np.zeros((n_batch_inference, model_height, model_width,3))

            for i in range(nxf):
                for j in range(nyf):

                    if i == 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + model_width
                    elif i > 0:
                        index_x_d = i * width_mid
                        index_x_u = index_x_d + model_width

                    if j == 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + model_height
                    elif j > 0:
                        index_y_d = j * height_mid
                        index_y_u = index_y_d + model_height

                    if index_x_u > img_w:
                        index_x_u = img_w
                        index_x_d = img_w - model_width
                    if index_y_u > img_h:
                        index_y_u = img_h
                        index_y_d = img_h - model_height
                        
                        
                    list_i_s.append(i)
                    list_j_s.append(j)
                    list_x_u.append(index_x_u)
                    list_x_d.append(index_x_d)
                    list_y_d.append(index_y_d)
                    list_y_u.append(index_y_u)
                    

                    img_patch[batch_indexer,:,:,:] = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
                    
                    batch_indexer = batch_indexer + 1
                    
                    

                    if batch_indexer == n_batch_inference:
                        
                        label_p_pred = model.predict(img_patch,verbose=0)
                        
                        seg = np.argmax(label_p_pred, axis=3)
                        
                        #print(seg.shape, len(seg), len(list_i_s))
                        
                        indexer_inside_batch = 0
                        for i_batch, j_batch in zip(list_i_s, list_j_s):
                            seg_in = seg[indexer_inside_batch,:,:]
                            seg_color = np.repeat(seg_in[:, :, np.newaxis], 3, axis=2)
                            
                            index_y_u_in = list_y_u[indexer_inside_batch]
                            index_y_d_in = list_y_d[indexer_inside_batch]
                            
                            index_x_u_in = list_x_u[indexer_inside_batch]
                            index_x_d_in = list_x_d[indexer_inside_batch]
                            
                            if i_batch == 0 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch == 0 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch == 0 and j_batch != 0 and j_batch != nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch != 0 and j_batch != nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch != 0 and i_batch != nxf - 1 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                            elif i_batch != 0 and i_batch != nxf - 1 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                            else:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                                
                            indexer_inside_batch = indexer_inside_batch +1
                                
                        
                        list_i_s = []
                        list_j_s = []
                        list_x_u = []
                        list_x_d = []
                        list_y_u = []
                        list_y_d = []
                        
                        batch_indexer = 0
                        
                        img_patch = np.zeros((n_batch_inference, model_height, model_width,3))
                        
                    elif i==(nxf-1) and j==(nyf-1):
                        label_p_pred = model.predict(img_patch,verbose=0)
                        
                        seg = np.argmax(label_p_pred, axis=3)
                        
                        #print(seg.shape, len(seg), len(list_i_s))
                        
                        indexer_inside_batch = 0
                        for i_batch, j_batch in zip(list_i_s, list_j_s):
                            seg_in = seg[indexer_inside_batch,:,:]
                            seg_color = np.repeat(seg_in[:, :, np.newaxis], 3, axis=2)
                            
                            index_y_u_in = list_y_u[indexer_inside_batch]
                            index_y_d_in = list_y_d[indexer_inside_batch]
                            
                            index_x_u_in = list_x_u[indexer_inside_batch]
                            index_x_d_in = list_x_d[indexer_inside_batch]
                            
                            if i_batch == 0 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch == 0 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch == 0 and j_batch != 0 and j_batch != nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, 0 : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + 0 : index_x_u_in - margin, :] = seg_color
                            elif i_batch == nxf - 1 and j_batch != 0 and j_batch != nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - 0, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - 0, :] = seg_color
                            elif i_batch != 0 and i_batch != nxf - 1 and j_batch == 0:
                                seg_color = seg_color[0 : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + 0 : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                            elif i_batch != 0 and i_batch != nxf - 1 and j_batch == nyf - 1:
                                seg_color = seg_color[margin : seg_color.shape[0] - 0, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - 0, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                            else:
                                seg_color = seg_color[margin : seg_color.shape[0] - margin, margin : seg_color.shape[1] - margin, :]
                                prediction_true[index_y_d_in + margin : index_y_u_in - margin, index_x_d_in + margin : index_x_u_in - margin, :] = seg_color
                                
                            indexer_inside_batch = indexer_inside_batch +1
                                
                        
                        list_i_s = []
                        list_j_s = []
                        list_x_u = []
                        list_x_d = []
                        list_y_u = []
                        list_y_d = []
                        
                        batch_indexer = 0
                        
                        img_patch = np.zeros((n_batch_inference, model_height, model_width,3))
            
            
            
            prediction_true = prediction_true[index_start_h: index_start_h+img_org_h, index_start_w: index_start_w+img_org_w,:]
            prediction_true = prediction_true.astype(np.uint8)

        else:
            img_h_page = img.shape[0]
            img_w_page = img.shape[1]
            img = img / float(255.0)
            img = resize_image(img, model_height, model_width)

            label_p_pred = model.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))

            seg = np.argmax(label_p_pred, axis=3)[0]
            seg_color = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
            prediction_true = resize_image(seg_color, img_h_page, img_w_page)
            prediction_true = prediction_true.astype(np.uint8)
        return prediction_true[:,:,0]

    def run(self, image=None, image_path=None, output=None, use_patches=False, dir_in=None, overwrite=False):
        if not dir_in:
            if (image is None) == (image_path is None):
                raise ValueError("Must pass either a opencv2 image or an image_path")
            if image_path is not None:
                image = cv2.imread(image_path)
            img_last = self.run_single(image, use_patches)
            if output:
                if os.path.exists(output):
                    if overwrite:
                        self.logger.warning("will overwrite existing output file '%s'", output)
                    else:
                        self.logger.warning("output file already exists '%s'", output)
                        return img_last
                self.logger.info('Writing binarized image to %s', output)
                cv2.imwrite(output, img_last)
            return img_last
        else:
            ls_imgs = list(filter(is_image_filename, os.listdir(dir_in)))
            self.logger.info("Found %d image files to binarize in %s", len(ls_imgs), dir_in)
            for i, image_path in enumerate(ls_imgs):
                image_stem = os.path.splitext(image_path)[0]
                output_path = os.path.join(output, image_stem + '.png')
                if os.path.exists(output_path):
                    if overwrite:
                        self.logger.warning("will overwrite existing output file '%s'", output_path)
                    else:
                        self.logger.warning("will skip input for existing output file '%s'", output_path)
                        continue
                self.logger.info('Binarizing [%3d/%d] %s', i + 1, len(ls_imgs), image_path)
                image = cv2.imread(os.path.join(dir_in, image_path))
                img_last = self.run_single(image, use_patches)
                self.logger.info('Writing binarized image to %s', output_path)
                cv2.imwrite(output_path, img_last)

    def run_single(self, image: np.ndarray, use_patches=False):
        img_last = 0
        model_file, model = self.models
        res = self.predict(model, image, use_patches)

        img_fin = np.zeros((res.shape[0], res.shape[1], 3))
        res[:, :][res[:, :] == 0] = 2
        res = res - 1
        res = res * 255
        img_fin[:, :, 0] = res
        img_fin[:, :, 1] = res
        img_fin[:, :, 2] = res

        img_fin = img_fin.astype(np.uint8)
        img_fin = (res[:, :] == 0) * 255
        img_last = img_last + img_fin

        kernel = np.ones((5, 5), np.uint8)
        img_last[:, :][img_last[:, :] > 0] = 255
        img_last = (img_last[:, :] == 0) * 255
        return img_last
