import sys
import os
import numpy as np
import warnings
import cv2
import seaborn as sns
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow.keras.losses
from tensorflow.keras.layers import *
from models import *
import click
import json
from tensorflow.python.keras import backend as tensorflow_backend






with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
__doc__=\
"""
Tool to load model and predict for given image.
"""

class sbb_predict:
    def __init__(self,image, model, task, config_params_model, patches, save, ground_truth):
        self.image=image
        self.patches=patches
        self.save=save
        self.model_dir=model
        self.ground_truth=ground_truth
        self.task=task
        self.config_params_model=config_params_model

    def resize_image(self,img_in,input_height,input_width):
        return cv2.resize( img_in, ( input_width,input_height) ,interpolation=cv2.INTER_NEAREST)
    
    
    def color_images(self,seg):
        ann_u=range(self.n_classes)
        if len(np.shape(seg))==3:
            seg=seg[:,:,0]
            
        seg_img=np.zeros((np.shape(seg)[0],np.shape(seg)[1],3)).astype(np.uint8)
        colors=sns.color_palette("hls", self.n_classes)
        
        for c in ann_u:
            c=int(c)
            segl=(seg==c)
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
        retval1, threshold1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        

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
    
    def weighted_categorical_crossentropy(self,weights=None):

        def loss(y_true, y_pred):
            labels_floats = tf.cast(y_true, tf.float32)
            per_pixel_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_floats,logits=y_pred)
        
            if weights is not None:
                weight_mask = tf.maximum(tf.reduce_max(tf.constant(
                    np.array(weights, dtype=np.float32)[None, None, None])
                    * labels_floats, axis=-1), 1.0)
                per_pixel_loss = per_pixel_loss * weight_mask[:, :, :, None]
            return tf.reduce_mean(per_pixel_loss)
        return self.loss


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
        
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        session = tf.compat.v1.Session(config=config)  # tf.InteractiveSession()
        tensorflow_backend.set_session(session)
        #tensorflow.keras.layers.custom_layer = PatchEncoder
        #tensorflow.keras.layers.custom_layer = Patches
        self.model = load_model(self.model_dir , compile=False,custom_objects = {"PatchEncoder": PatchEncoder, "Patches": Patches})
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth=True
    
        #self.session = tf.InteractiveSession()
        #keras.losses.custom_loss = self.weighted_categorical_crossentropy
        #self.model = load_model(self.model_dir , compile=False)

        
        ##if self.weights_dir!=None:
            ##self.model.load_weights(self.weights_dir)
            
        if self.task != 'classification':
            self.img_height=self.model.layers[len(self.model.layers)-1].output_shape[1]
            self.img_width=self.model.layers[len(self.model.layers)-1].output_shape[2]
            self.n_classes=self.model.layers[len(self.model.layers)-1].output_shape[3]
        
    def visualize_model_output(self, prediction, img, task):
        if task == "binarization":
            prediction = prediction * -1
            prediction = prediction + 1
            added_image = prediction * 255
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
        
            output = np.zeros(prediction.shape)
        
            for unq_class in unique_classes:
                rgb_class_unique = rgb_colors[str(int(unq_class))]
                output[:,:,0][prediction[:,:,0]==unq_class] = rgb_class_unique[0]
                output[:,:,1][prediction[:,:,0]==unq_class] = rgb_class_unique[1]
                output[:,:,2][prediction[:,:,0]==unq_class] = rgb_class_unique[2]
        
        
        
            img = self.resize_image(img, output.shape[0], output.shape[1])
        
            output = output.astype(np.int32)
            img = img.astype(np.int32)
        
            
            
            added_image = cv2.addWeighted(img,0.5,output,0.1,0)
            
        return added_image

    def predict(self):
        self.start_new_session_and_model()
        if self.task == 'classification':
            classes_names = self.config_params_model['classification_classes_name']
            img_1ch = img=cv2.imread(self.image, 0)

            img_1ch = img_1ch / 255.0
            img_1ch = cv2.resize(img_1ch, (self.config_params_model['input_height'], self.config_params_model['input_width']), interpolation=cv2.INTER_NEAREST)
            img_in = np.zeros((1, img_1ch.shape[0], img_1ch.shape[1], 3))
            img_in[0, :, :, 0] = img_1ch[:, :]
            img_in[0, :, :, 1] = img_1ch[:, :]
            img_in[0, :, :, 2] = img_1ch[:, :]
                      
            label_p_pred = self.model.predict(img_in, verbose=0)
            index_class = np.argmax(label_p_pred[0])
            
            print("Predicted Class: {}".format(classes_names[str(int(index_class))]))
        else:
            if self.patches:
                #def textline_contours(img,input_width,input_height,n_classes,model):
                
                img=cv2.imread(self.image)
                self.img_org = np.copy(img)
                
                if img.shape[0] < self.img_height:
                    img = cv2.resize(img, (img.shape[1], self.img_width), interpolation=cv2.INTER_NEAREST)

                if img.shape[1] < self.img_width:
                    img = cv2.resize(img, (self.img_height, img.shape[0]), interpolation=cv2.INTER_NEAREST)
                margin = int(0 * self.img_width)
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
                                                                verbose=0)
                        
                        if self.task == 'enhancement':
                            seg = label_p_pred[0, :, :, :]
                            seg = seg * 255
                        elif self.task == 'segmentation' or self.task == 'binarization':
                            seg = np.argmax(label_p_pred, axis=3)[0]
                            seg = np.repeat(seg[:, :, np.newaxis], 3, axis=2)
                            

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

                img=cv2.imread(self.image)
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
                    
                prediction_true = seg.astype(int)

                prediction_true = cv2.resize(prediction_true, (self.img_org.shape[1], self.img_org.shape[0]), interpolation=cv2.INTER_NEAREST)
                return prediction_true



    def run(self):
        res=self.predict()
        if self.task == 'classification':
            pass
        else:
            img_seg_overlayed = self.visualize_model_output(res, self.img_org, self.task)
            if self.save:
                cv2.imwrite(self.save,img_seg_overlayed)
                
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
def main(image, model, patches, save, ground_truth):
    with open(os.path.join(model,'config.json')) as f:
        config_params_model = json.load(f)
    task = config_params_model['task']
    if task != 'classification':
        if not save:
            print("Error: You used one of segmentation or binarization task but not set -s, you need a filename to save visualized output with -s")
            sys.exit(1)
    x=sbb_predict(image, model, task, config_params_model, patches, save, ground_truth)
    x.run()

if __name__=="__main__":
    main()

    
    
    
