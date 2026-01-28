import os
import math
import random
from logging import getLogger

import cv2
import numpy as np
import seaborn as sns
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
import imutils
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageEnhance


def return_shuffled_channels(img, channels_order):
    """
    channels order in ordinary case is like this [0, 1, 2]. In the case of shuffling the order should be provided.
    """
    img_sh = np.copy(img)
    
    img_sh[:,:,0]= img[:,:,channels_order[0]]
    img_sh[:,:,1]= img[:,:,channels_order[1]]
    img_sh[:,:,2]= img[:,:,channels_order[2]]
    return img_sh

def return_binary_image_with_red_textlines(img_bin):
    img_red = np.copy(img_bin)
    
    img_red[:,:,0][img_bin[:,:,0] == 0] = 255
    return img_red

def return_binary_image_with_given_rgb_background(img_bin, img_rgb_background):
    img_rgb_background = resize_image(img_rgb_background ,img_bin.shape[0], img_bin.shape[1])
    
    img_final = np.copy(img_bin)
    
    img_final[:,:,0][img_bin[:,:,0] != 0] = img_rgb_background[:,:,0][img_bin[:,:,0] != 0]
    img_final[:,:,1][img_bin[:,:,1] != 0] = img_rgb_background[:,:,1][img_bin[:,:,1] != 0]
    img_final[:,:,2][img_bin[:,:,2] != 0] = img_rgb_background[:,:,2][img_bin[:,:,2] != 0]
    
    return img_final

def return_binary_image_with_given_rgb_background_and_given_foreground_rgb(img_bin, img_rgb_background, rgb_foreground):
    img_rgb_background = resize_image(img_rgb_background ,img_bin.shape[0], img_bin.shape[1])
    
    img_final = np.copy(img_bin)
    img_foreground = np.zeros(img_bin.shape)
    
    
    img_foreground[:,:,0][img_bin[:,:,0] == 0] = rgb_foreground[0]
    img_foreground[:,:,1][img_bin[:,:,0] == 0] = rgb_foreground[1]
    img_foreground[:,:,2][img_bin[:,:,0] == 0] = rgb_foreground[2]
    
    
    img_final[:,:,0][img_bin[:,:,0] != 0] = img_rgb_background[:,:,0][img_bin[:,:,0] != 0]
    img_final[:,:,1][img_bin[:,:,1] != 0] = img_rgb_background[:,:,1][img_bin[:,:,1] != 0]
    img_final[:,:,2][img_bin[:,:,2] != 0] = img_rgb_background[:,:,2][img_bin[:,:,2] != 0]
    
    img_final = img_final + img_foreground
    return img_final

def return_binary_image_with_given_rgb_background_red_textlines(img_bin, img_rgb_background, img_color):
    img_rgb_background = resize_image(img_rgb_background ,img_bin.shape[0], img_bin.shape[1])
    
    img_final = np.copy(img_color)
    
    img_final[:,:,0][img_bin[:,:,0] != 0] = img_rgb_background[:,:,0][img_bin[:,:,0] != 0]
    img_final[:,:,1][img_bin[:,:,1] != 0] = img_rgb_background[:,:,1][img_bin[:,:,1] != 0]
    img_final[:,:,2][img_bin[:,:,2] != 0] = img_rgb_background[:,:,2][img_bin[:,:,2] != 0]
    
    return img_final

def return_image_with_red_elements(img, img_bin):
    img_final = np.copy(img)
    
    img_final[:,:,0][img_bin[:,:,0]==0] = 0
    img_final[:,:,1][img_bin[:,:,0]==0] = 0
    img_final[:,:,2][img_bin[:,:,0]==0] = 255
    return img_final
    
def shift_image_and_label(img, label, type_shift):
    h_n = int(img.shape[0]*1.06)
    w_n = int(img.shape[1]*1.06)

    channel0_avg = int( np.mean(img[:,:,0]) )
    channel1_avg = int( np.mean(img[:,:,1]) )
    channel2_avg = int( np.mean(img[:,:,2]) )
    
    h_diff = abs( img.shape[0] - h_n )
    w_diff = abs( img.shape[1] - w_n )

    h_start = int(h_diff / 2.)
    w_start = int(w_diff / 2.)
    
    img_scaled_padded = np.zeros((h_n, w_n, 3))
    label_scaled_padded = np.zeros((h_n, w_n, 3))

    img_scaled_padded[:,:,0] = channel0_avg
    img_scaled_padded[:,:,1] = channel1_avg
    img_scaled_padded[:,:,2] = channel2_avg
    
    img_scaled_padded[h_start:h_start+img.shape[0], w_start:w_start+img.shape[1],:] = img[:,:,:]
    label_scaled_padded[h_start:h_start+img.shape[0], w_start:w_start+img.shape[1],:] = label[:,:,:]
    
    
    if type_shift=="xpos":
        img_dis = img_scaled_padded[h_start:h_start+img.shape[0],2*w_start:2*w_start+img.shape[1],:]
        label_dis = label_scaled_padded[h_start:h_start+img.shape[0],2*w_start:2*w_start+img.shape[1],:]
    elif type_shift=="xmin":
        img_dis = img_scaled_padded[h_start:h_start+img.shape[0],:img.shape[1],:]
        label_dis = label_scaled_padded[h_start:h_start+img.shape[0],:img.shape[1],:]
    elif type_shift=="ypos":
        img_dis = img_scaled_padded[2*h_start:2*h_start+img.shape[0],w_start:w_start+img.shape[1],:]
        label_dis = label_scaled_padded[2*h_start:2*h_start+img.shape[0],w_start:w_start+img.shape[1],:]
    elif type_shift=="ymin":
        img_dis = img_scaled_padded[:img.shape[0],w_start:w_start+img.shape[1],:]
        label_dis = label_scaled_padded[:img.shape[0],w_start:w_start+img.shape[1],:]
    elif type_shift=="xypos":
        img_dis = img_scaled_padded[2*h_start:2*h_start+img.shape[0],2*w_start:2*w_start+img.shape[1],:]
        label_dis = label_scaled_padded[2*h_start:2*h_start+img.shape[0],2*w_start:2*w_start+img.shape[1],:]
    elif type_shift=="xymin":
        img_dis = img_scaled_padded[:img.shape[0],:img.shape[1],:]
        label_dis = label_scaled_padded[:img.shape[0],:img.shape[1],:]
    return img_dis, label_dis

def scale_image_for_no_patch(img, label, scale):
    h_n = int(img.shape[0]*scale)
    w_n = int(img.shape[1]*scale)
    
    channel0_avg = int( np.mean(img[:,:,0]) )
    channel1_avg = int( np.mean(img[:,:,1]) )
    channel2_avg = int( np.mean(img[:,:,2]) )
    
    h_diff = img.shape[0] - h_n
    w_diff = img.shape[1] - w_n
    
    h_start = int(h_diff / 2.)
    w_start = int(w_diff / 2.)
    
    img_res = resize_image(img, h_n, w_n)
    label_res = resize_image(label, h_n, w_n)
    
    img_scaled_padded = np.copy(img)
    
    label_scaled_padded = np.zeros(label.shape)
    
    img_scaled_padded[:,:,0] = channel0_avg
    img_scaled_padded[:,:,1] = channel1_avg
    img_scaled_padded[:,:,2] = channel2_avg
    
    img_scaled_padded[h_start:h_start+h_n, w_start:w_start+w_n,:] = img_res[:,:,:]
    label_scaled_padded[h_start:h_start+h_n, w_start:w_start+w_n,:] = label_res[:,:,:]
    
    return img_scaled_padded, label_scaled_padded


def return_number_of_total_training_data(path_classes):
    sub_classes = os.listdir(path_classes)
    n_tot = 0
    for sub_c in sub_classes:
        sub_files =  os.listdir(os.path.join(path_classes,sub_c))
        n_tot = n_tot + len(sub_files)
    return n_tot
        
    
    
def generate_data_from_folder_evaluation(path_classes, height, width, n_classes, list_classes):
    #sub_classes = os.listdir(path_classes)
    #n_classes = len(sub_classes)
    all_imgs = []
    labels = []
    #dicts =dict()
    #indexer= 0
    for indexer, sub_c in enumerate(list_classes):
        sub_files =  os.listdir(os.path.join(path_classes,sub_c  )) 
        sub_files = [os.path.join(path_classes,sub_c  )+'/' + x for x in sub_files]
        #print(     os.listdir(os.path.join(path_classes,sub_c  ))     )
        all_imgs = all_imgs + sub_files
        sub_labels = list( np.zeros( len(sub_files) ) +indexer )

        #print( len(sub_labels) )
        labels = labels + sub_labels
        #dicts[sub_c] = indexer
        #indexer +=1 
        

    categories =  to_categorical(range(n_classes)).astype(np.int16)#[  [1 , 0, 0 , 0 , 0 , 0]  , [0 , 1, 0 , 0 , 0 , 0]  , [0 , 0, 1 , 0 , 0 , 0] , [0 , 0, 0 , 1 , 0 , 0] , [0 , 0, 0 , 0 , 1 , 0]  , [0 , 0, 0 , 0 , 0 , 1] ]
    ret_x= np.zeros((len(labels), height,width, 3)).astype(np.int16)
    ret_y= np.zeros((len(labels), n_classes)).astype(np.int16)
    
    #print(all_imgs)
    for i in range(len(all_imgs)):
        row = all_imgs[i]
        #####img = cv2.imread(row, 0)
        #####img= resize_image (img, height, width)
        #####img = img.astype(np.uint16)
        #####ret_x[i, :,:,0] = img[:,:]
        #####ret_x[i, :,:,1] = img[:,:]
        #####ret_x[i, :,:,2] = img[:,:]
        
        img = cv2.imread(row)
        img= resize_image (img, height, width)
        img = img.astype(np.uint16)
        ret_x[i, :,:] = img[:,:,:]
        
        ret_y[i, :] =  categories[ int( labels[i] ) ][:]
    
    return ret_x/255., ret_y

def generate_data_from_folder_training(path_classes, batchsize, height, width, n_classes, list_classes):
    #sub_classes = os.listdir(path_classes)
    #n_classes = len(sub_classes)

    all_imgs = []
    labels = []
    #dicts =dict()
    #indexer= 0
    for indexer, sub_c in enumerate(list_classes):
        sub_files =  os.listdir(os.path.join(path_classes,sub_c  )) 
        sub_files = [os.path.join(path_classes,sub_c  )+'/' + x for x in sub_files]
        #print(     os.listdir(os.path.join(path_classes,sub_c  ))     )
        all_imgs = all_imgs + sub_files
        sub_labels = list( np.zeros( len(sub_files) ) +indexer )

        #print( len(sub_labels) )
        labels = labels + sub_labels
        #dicts[sub_c] = indexer
        #indexer +=1 
        
    ids = np.array(range(len(labels)))
    random.shuffle(ids)
    
    shuffled_labels = np.array(labels)[ids]
    shuffled_files = np.array(all_imgs)[ids]
    categories = to_categorical(range(n_classes)).astype(np.int16)#[  [1 , 0, 0 , 0 , 0 , 0]  , [0 , 1, 0 , 0 , 0 , 0]  , [0 , 0, 1 , 0 , 0 , 0] , [0 , 0, 0 , 1 , 0 , 0] , [0 , 0, 0 , 0 , 1 , 0]  , [0 , 0, 0 , 0 , 0 , 1] ]
    ret_x= np.zeros((batchsize, height,width, 3)).astype(np.int16)
    ret_y= np.zeros((batchsize, n_classes)).astype(np.int16)
    batchcount = 0
    while True:
        for i in range(len(shuffled_files)):
            row = shuffled_files[i]
            #print(row)
            ###img = cv2.imread(row, 0)
            ###img= resize_image (img, height, width)
            ###img = img.astype(np.uint16)
            ###ret_x[batchcount, :,:,0] = img[:,:]
            ###ret_x[batchcount, :,:,1] = img[:,:]
            ###ret_x[batchcount, :,:,2] = img[:,:]
            
            img = cv2.imread(row)
            img= resize_image (img, height, width)
            img = img.astype(np.uint16)
            ret_x[batchcount, :,:,:] = img[:,:,:]
            
            #print(int(shuffled_labels[i]) )
            #print( categories[int(shuffled_labels[i])] )
            ret_y[batchcount, :] =  categories[ int( shuffled_labels[i] ) ][:]
            
            batchcount+=1
            
            if batchcount>=batchsize:
                ret_x = ret_x/255.
                yield ret_x, ret_y
                ret_x= np.zeros((batchsize, height,width, 3)).astype(np.int16)
                ret_y= np.zeros((batchsize, n_classes)).astype(np.int16)
                batchcount = 0

def do_brightening(img, factor):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img_rgb)
    enhancer = ImageEnhance.Brightness(im)
    out_img = enhancer.enhance(factor)
    out_img = out_img.convert('RGB')
    opencv_img = np.array(out_img)
    opencv_img = opencv_img[:,:,::-1].copy()
    return opencv_img


def bluring(img_in, kind):
    if kind == 'gauss':
        img_blur = cv2.GaussianBlur(img_in, (5, 5), 0)
    elif kind == "median":
        img_blur = cv2.medianBlur(img_in, 5)
    elif kind == 'blur':
        img_blur = cv2.blur(img_in, (5, 5))
    return img_blur


def elastic_transform(image, alpha, sigma, seedj, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(seedj)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


def rotation_90(img):
    img_rot = np.zeros((img.shape[1], img.shape[0], img.shape[2]))
    img_rot[:, :, 0] = img[:, :, 0].T
    img_rot[:, :, 1] = img[:, :, 1].T
    img_rot[:, :, 2] = img[:, :, 2].T
    return img_rot


def rotatedRectWithMaxArea(w, h, angle):
    """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def rotate_max_area(image, rotated, rotated_label, angle):
    """ image: cv2 image matrix object
        angle: in degree
    """
    wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0],
                                    math.radians(angle))
    h, w, _ = rotated.shape
    y1 = h // 2 - int(hr / 2)
    y2 = y1 + int(hr)
    x1 = w // 2 - int(wr / 2)
    x2 = x1 + int(wr)
    return rotated[y1:y2, x1:x2], rotated_label[y1:y2, x1:x2]

def rotate_max_area_single_image(image, rotated, angle):
    """ image: cv2 image matrix object
        angle: in degree
    """
    wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0],
                                    math.radians(angle))
    h, w, _ = rotated.shape
    y1 = h // 2 - int(hr / 2)
    y2 = y1 + int(hr)
    x1 = w // 2 - int(wr / 2)
    x2 = x1 + int(wr)
    return rotated[y1:y2, x1:x2]

def rotation_not_90_func(img, label, thetha):
    rotated = imutils.rotate(img, thetha)
    rotated_label = imutils.rotate(label, thetha)
    return rotate_max_area(img, rotated, rotated_label, thetha)


def rotation_not_90_func_single_image(img, thetha):
    rotated = imutils.rotate(img, thetha)
    return rotate_max_area_single_image(img, rotated, thetha)


def color_images(seg, n_classes):
    ann_u = range(n_classes)
    if len(np.shape(seg)) == 3:
        seg = seg[:, :, 0]

    seg_img = np.zeros((np.shape(seg)[0], np.shape(seg)[1], 3)).astype(float)
    colors = sns.color_palette("hls", n_classes)

    for c in ann_u:
        c = int(c)
        segl = (seg == c)
        seg_img[:, :, 0] += segl * (colors[c][0])
        seg_img[:, :, 1] += segl * (colors[c][1])
        seg_img[:, :, 2] += segl * (colors[c][2])
    return seg_img


def resize_image(seg_in, input_height, input_width):
    return cv2.resize(seg_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)


def get_one_hot(seg, input_height, input_width, n_classes):
    seg = seg[:, :, 0]
    seg_f = np.zeros((input_height, input_width, n_classes))
    for j in range(n_classes):
        seg_f[:, :, j] = (seg == j).astype(int)
    return seg_f


def IoU(Yi, y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    IoUs = []
    classes_true = np.unique(Yi)
    for c in classes_true:
        TP = np.sum((Yi == c) & (y_predi == c))
        FP = np.sum((Yi != c) & (y_predi == c))
        FN = np.sum((Yi == c) & (y_predi != c))
        IoU = TP / float(TP + FP + FN)
        #print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c, TP, FP, FN, IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    #print("_________________")
    #print("Mean IoU: {:4.3f}".format(mIoU))
    return mIoU

def generate_arrays_from_folder_reading_order(classes_file_dir, modal_dir, batchsize, height, width, n_classes, thetha, augmentation=False):
    all_labels_files = os.listdir(classes_file_dir)
    ret_x= np.zeros((batchsize, height, width, 3))#.astype(np.int16)
    ret_y= np.zeros((batchsize, n_classes)).astype(np.int16)
    batchcount = 0
    while True:
        for i in all_labels_files:
            file_name = os.path.splitext(i)[0]
            img = cv2.imread(os.path.join(modal_dir,file_name+'.png'))

            label_class = int( np.load(os.path.join(classes_file_dir,i)) )

            ret_x[batchcount, :,:,0] = img[:,:,0]/3.0
            ret_x[batchcount, :,:,2] = img[:,:,2]/3.0
            ret_x[batchcount, :,:,1] = img[:,:,1]/5.0

            ret_y[batchcount, :] =  label_class
            batchcount+=1
            if batchcount>=batchsize:
                yield ret_x, ret_y
                ret_x= np.zeros((batchsize, height, width, 3))#.astype(np.int16)
                ret_y= np.zeros((batchsize, n_classes)).astype(np.int16)
                batchcount = 0
                
            if augmentation:
                for thetha_i in thetha:
                    img_rot = rotation_not_90_func_single_image(img, thetha_i)
                    
                    img_rot = resize_image(img_rot, height, width)
                    
                    ret_x[batchcount, :,:,0] = img_rot[:,:,0]/3.0
                    ret_x[batchcount, :,:,2] = img_rot[:,:,2]/3.0
                    ret_x[batchcount, :,:,1] = img_rot[:,:,1]/5.0

                    ret_y[batchcount, :] =  label_class
                    batchcount+=1
                    if batchcount>=batchsize:
                        yield ret_x, ret_y
                        ret_x= np.zeros((batchsize, height, width, 3))#.astype(np.int16)
                        ret_y= np.zeros((batchsize, n_classes)).astype(np.int16)
                        batchcount = 0

def data_gen(img_folder, mask_folder, batch_size, input_height, input_width, n_classes, task='segmentation'):
    c = 0
    n = [f for f in os.listdir(img_folder) if not f.startswith('.')]  # os.listdir(img_folder) #List of training images
    random.shuffle(n)
    while True:
        img = np.zeros((batch_size, input_height, input_width, 3)).astype('float')
        mask = np.zeros((batch_size, input_height, input_width, n_classes)).astype('float')

        for i in range(c, c + batch_size):  # initially from 0 to 16, c = 0.
            try:
                filename = os.path.splitext(n[i])[0]

                train_img = cv2.imread(img_folder + '/' + n[i]) / 255.
                train_img = cv2.resize(train_img, (input_width, input_height),
                                       interpolation=cv2.INTER_NEAREST)  # Read an image from folder and resize

                img[i - c] = train_img  # add to array - img[0], img[1], and so on.
                if task == "segmentation" or task=="binarization":
                    train_mask = cv2.imread(mask_folder + '/' + filename + '.png')
                    train_mask = get_one_hot(resize_image(train_mask, input_height, input_width), input_height, input_width,
                                            n_classes)
                elif task == "enhancement":
                    train_mask = cv2.imread(mask_folder + '/' + filename + '.png')/255.
                    train_mask = resize_image(train_mask, input_height, input_width)
                    
                # train_mask = train_mask.reshape(224, 224, 1) # Add extra dimension for parity with train_img size [512 * 512 * 3]

                mask[i - c] = train_mask
            except:
                img[i - c] = np.ones((input_height, input_width, 3)).astype('float')
                mask[i - c] = np.zeros((input_height, input_width, n_classes)).astype('float')

        c += batch_size
        if c + batch_size >= len(os.listdir(img_folder)):
            c = 0
            random.shuffle(n)
        yield img, mask


def otsu_copy(img):
    img_r = np.zeros(img.shape)
    img1 = img[:, :, 0]
    img2 = img[:, :, 1]
    img3 = img[:, :, 2]
    _, threshold1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, threshold2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, threshold3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_r[:, :, 0] = threshold1
    img_r[:, :, 1] = threshold1
    img_r[:, :, 2] = threshold1
    return img_r


def get_patches(dir_img_f, dir_seg_f, img, label, height, width, indexer):
    if img.shape[0] < height or img.shape[1] < width:
        img, label = do_padding(img, label, height, width)

    img_h = img.shape[0]
    img_w = img.shape[1]

    nxf = img_w / float(width)
    nyf = img_h / float(height)

    if nxf > int(nxf):
        nxf = int(nxf) + 1
    if nyf > int(nyf):
        nyf = int(nyf) + 1

    nxf = int(nxf)
    nyf = int(nyf)

    for i in range(nxf):
        for j in range(nyf):
            index_x_d = i * width
            index_x_u = (i + 1) * width

            index_y_d = j * height
            index_y_u = (j + 1) * height

            if index_x_u > img_w:
                index_x_u = img_w
                index_x_d = img_w - width
            if index_y_u > img_h:
                index_y_u = img_h
                index_y_d = img_h - height

            img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
            label_patch = label[index_y_d:index_y_u, index_x_d:index_x_u, :]

            cv2.imwrite(dir_img_f + '/img_' + str(indexer) + '.png', img_patch)
            cv2.imwrite(dir_seg_f + '/img_' + str(indexer) + '.png', label_patch)
            indexer += 1

    return indexer


def do_padding_white(img):
    img_org_h = img.shape[0]
    img_org_w = img.shape[1]
    
    index_start_h = 4
    index_start_w = 4
    
    img_padded = np.zeros((img.shape[0] + 2*index_start_h, img.shape[1]+ 2*index_start_w, img.shape[2])) + 255
    img_padded[index_start_h: index_start_h + img.shape[0], index_start_w: index_start_w + img.shape[1], :] = img[:, :, :]
    
    return img_padded.astype(float)


def do_degrading(img, scale):
    img_org_h = img.shape[0]
    img_org_w = img.shape[1]
    
    img_res = resize_image(img, int(img_org_h * scale), int(img_org_w * scale))
    
    return resize_image(img_res, img_org_h, img_org_w)
    
    
def do_padding_black(img):
    img_org_h = img.shape[0]
    img_org_w = img.shape[1]
    
    index_start_h = 4
    index_start_w = 4
    
    img_padded = np.zeros((img.shape[0] + 2*index_start_h, img.shape[1] + 2*index_start_w, img.shape[2]))
    img_padded[index_start_h: index_start_h + img.shape[0], index_start_w: index_start_w + img.shape[1], :] = img[:, :, :]
    
    return img_padded.astype(float)


def do_padding_label(img):
    img_org_h = img.shape[0]
    img_org_w = img.shape[1]
    
    index_start_h = 4
    index_start_w = 4
    
    img_padded = np.zeros((img.shape[0] + 2*index_start_h, img.shape[1] + 2*index_start_w, img.shape[2]))
    img_padded[index_start_h: index_start_h + img.shape[0], index_start_w: index_start_w + img.shape[1], :] = img[:, :, :]
    
    return img_padded.astype(np.int16)

def do_padding(img, label, height, width):
    height_new=img.shape[0]
    width_new=img.shape[1]
    
    h_start = 0
    w_start = 0
    
    if img.shape[0] < height:
        h_start = int(abs(height - img.shape[0]) / 2.)
        height_new = height
        
    if img.shape[1] < width:
        w_start = int(abs(width - img.shape[1]) / 2.)
        width_new = width
    
    img_new = np.ones((height_new, width_new, img.shape[2])).astype(float) * 255
    label_new = np.zeros((height_new, width_new, label.shape[2])).astype(float)
    
    img_new[h_start:h_start + img.shape[0], w_start:w_start + img.shape[1], :] = np.copy(img[:, :, :])
    label_new[h_start:h_start + label.shape[0], w_start:w_start + label.shape[1], :] = np.copy(label[:, :, :])
    
    return img_new,label_new


def get_patches_num_scale(dir_img_f, dir_seg_f, img, label, height, width, indexer, n_patches, scaler):
    if img.shape[0] < height or img.shape[1] < width:
        img, label = do_padding(img, label, height, width)
    
    img_h = img.shape[0]
    img_w = img.shape[1]
    
    height_scale = int(height * scaler)
    width_scale = int(width * scaler)
    
    
    nxf = img_w / float(width_scale)
    nyf = img_h / float(height_scale)
    
    if nxf > int(nxf):
        nxf = int(nxf) + 1
    if nyf > int(nyf):
        nyf = int(nyf) + 1
        
    nxf = int(nxf)
    nyf = int(nyf)
        
    for i in range(nxf):
        for j in range(nyf):
            index_x_d = i * width_scale
            index_x_u = (i + 1) * width_scale
            
            index_y_d = j * height_scale
            index_y_u = (j + 1) * height_scale
            
            if index_x_u > img_w:
                index_x_u = img_w
                index_x_d = img_w - width_scale
            if index_y_u > img_h:
                index_y_u = img_h
                index_y_d = img_h - height_scale
                
            
            img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
            label_patch = label[index_y_d:index_y_u, index_x_d:index_x_u, :]
            
            img_patch = resize_image(img_patch, height, width)
            label_patch = resize_image(label_patch, height, width)
            
            cv2.imwrite(dir_img_f + '/img_' + str(indexer) + '.png', img_patch)
            cv2.imwrite(dir_seg_f + '/img_' + str(indexer) + '.png', label_patch)
            indexer += 1
            
    return indexer


def get_patches_num_scale_new(dir_img_f, dir_seg_f, img, label, height, width, indexer, scaler):
    img = resize_image(img, int(img.shape[0] * scaler), int(img.shape[1] * scaler))
    label = resize_image(label, int(label.shape[0] * scaler), int(label.shape[1] * scaler))
    
    if img.shape[0] < height or img.shape[1] < width:
        img, label = do_padding(img, label, height, width)
    
    img_h = img.shape[0]
    img_w = img.shape[1]
    
    height_scale = int(height * 1)
    width_scale = int(width * 1)
    
    nxf = img_w / float(width_scale)
    nyf = img_h / float(height_scale)
    
    if nxf > int(nxf):
        nxf = int(nxf) + 1
    if nyf > int(nyf):
        nyf = int(nyf) + 1
        
    nxf = int(nxf)
    nyf = int(nyf)
        
    for i in range(nxf):
        for j in range(nyf):
            index_x_d = i * width_scale
            index_x_u = (i + 1) * width_scale
            
            index_y_d = j * height_scale
            index_y_u = (j + 1) * height_scale
            
            if index_x_u > img_w:
                index_x_u = img_w
                index_x_d = img_w - width_scale
            if index_y_u > img_h:
                index_y_u = img_h
                index_y_d = img_h - height_scale
            
            img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
            label_patch = label[index_y_d:index_y_u, index_x_d:index_x_u, :]
            
            cv2.imwrite(dir_img_f + '/img_' + str(indexer) + '.png', img_patch)
            cv2.imwrite(dir_seg_f + '/img_' + str(indexer) + '.png', label_patch)
            indexer += 1

    return indexer


def preprocess_imgs(config,
                    imgs_list,
                    segs_list,
                    dir_img,
                    dir_seg,
                    dir_flow_imgs,
                    dir_flow_labels,
                    logger=None,
                    **kwargs,
):
    if logger is None:
        logger = getLogger('')

    # make a copy for this run
    config = dict(config)
    # add derived keys not part of config
    if config.get('dir_rgb_backgrounds', None):
        config['list_all_possible_background_images'] = \
            os.listdir(config['dir_rgb_backgrounds'])
    if config.get('dir_rgb_foregrounds', None):
        config['list_all_possible_foreground_rgbs'] = \
            os.listdir(config['dir_rgb_foregrounds'])
    # override keys from call
    config.update(kwargs)

    indexer = 0
    for im, seg_i in tqdm(zip(imgs_list, segs_list)):
        img = cv2.imread(os.path.join(dir_img, im))
        img_name = os.path.splitext(im)[0]
        if config['task'] in ["segmentation", "binarization"]:
            lab = cv2.imread(os.path.join(dir_seg, img_name + '.png'))
        elif config['task'] == "enhancement":
            lab = cv2.imread(os.path.join(dir_seg, im))
        else:
            lab = None

        try:
            indexer = preprocess_img(indexer, img, img_name, lab,
                                     dir_flow_imgs,
                                     dir_flow_labels,
                                     **config)

        except:
            logger.exception("skipping image %s", img_name)

def preprocess_img(indexer,
                   img,
                   img_name,
                   lab,
                   dir_flow_train_imgs,
                   dir_flow_train_labels,
                   input_height=None,
                   input_width=None,
                   augmentation=False,
                   flip_aug=False,
                   flip_index=None,
                   blur_aug=False,
                   blur_k=None,
                   padding_white=False,
                   padding_black=False,
                   scaling=False,
                   scaling_bluring=False,
                   scaling_brightness=False,
                   scaling_binarization=False,
                   scaling_flip=False,
                   scales=None,
                   shifting=False,
                   degrading=False,
                   degrade_scales=None,
                   brightening=False,
                   brightness=None,
                   binarization=False,
                   dir_img_bin=None,
                   add_red_textlines=False,
                   adding_rgb_background=False,
                   dir_rgb_backgrounds=None,
                   adding_rgb_foreground=False,
                   dir_rgb_foregrounds=None,
                   number_of_backgrounds_per_image=None,
                   channels_shuffling=False,
                   shuffle_indexes=None,
                   rotation=False,
                   rotation_not_90=False,
                   thetha=None,
                   patches=False,
                   list_all_possible_background_images=None,
                   list_all_possible_foreground_rgbs=None,
                   **kwargs,
):
    if not patches:
        cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                    resize_image(img,
                                 input_height,
                                 input_width))
        cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                    resize_image(lab,
                                 input_height,
                                 input_width))
        indexer += 1
        if augmentation:
            if flip_aug:
                for f_i in flip_index:
                    cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                resize_image(cv2.flip(img, f_i),
                                             input_height,
                                             input_width))
                    cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                resize_image(cv2.flip(lab, f_i),
                                             input_height,
                                             input_width))
                    indexer += 1
            if blur_aug:
                for blur_i in blur_k:
                    cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                (resize_image(bluring(img, blur_i),
                                              input_height,
                                              input_width)))
                    cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                resize_image(lab,
                                             input_height,
                                             input_width))
                    indexer += 1
            if brightening:
                for factor in brightness:
                    cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                (resize_image(do_brightening(img, factor),
                                              input_height,
                                              input_width)))
                    cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                resize_image(lab,
                                             input_height,
                                             input_width))
                    indexer += 1
            if binarization:
                if dir_img_bin:
                    img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                    cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                resize_image(img_bin_corr,
                                             input_height,
                                             input_width))
                else:
                    cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                resize_image(otsu_copy(img),
                                             input_height,
                                             input_width))
                cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                            resize_image(lab,
                                         input_height,
                                         input_width))
                indexer += 1
            if degrading:
                for degrade_scale_ind in degrade_scales:
                    cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                (resize_image(do_degrading(img, degrade_scale_ind),
                                              input_height,
                                              input_width)))
                    cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                resize_image(lab,
                                             input_height,
                                             input_width))
                    indexer += 1
            if rotation_not_90:
                for thetha_i in thetha:
                    img_max_rotated, label_max_rotated = \
                        rotation_not_90_func(img, lab, thetha_i)
                    cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                resize_image(img_max_rotated,
                                             input_height,
                                             input_width))
                    cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                resize_image(label_max_rotated,
                                             input_height,
                                             input_width))
                    indexer += 1
            if channels_shuffling:
                for shuffle_index in shuffle_indexes:
                    cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                (resize_image(return_shuffled_channels(img, shuffle_index),
                                              input_height,
                                              input_width)))
                    cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                resize_image(lab,
                                             input_height,
                                             input_width))
                    indexer += 1
            if scaling:
                for sc_ind in scales:
                    img_scaled, label_scaled = \
                        scale_image_for_no_patch(img, lab, sc_ind)
                    cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                resize_image(img_scaled,
                                             input_height,
                                             input_width))
                    cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                resize_image(label_scaled,
                                             input_height,
                                             input_width))
                    indexer += 1
            if shifting:
                shift_types = ['xpos', 'xmin', 'ypos', 'ymin', 'xypos', 'xymin']
                for st_ind in shift_types:
                    img_shifted, label_shifted = \
                        shift_image_and_label(img, lab, st_ind)
                    cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                resize_image(img_shifted,
                                             input_height,
                                             input_width))
                    cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                resize_image(label_shifted,
                                             input_height,
                                             input_width))
                    indexer += 1
            if adding_rgb_background:
                img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                for i_n in range(number_of_backgrounds_per_image):
                    background_image_chosen_name = random.choice(list_all_possible_background_images)
                    img_rgb_background_chosen = \
                        cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
                    img_with_overlayed_background = \
                        return_binary_image_with_given_rgb_background(
                            img_bin_corr, img_rgb_background_chosen)
                    cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                resize_image(img_with_overlayed_background,
                                             input_height,
                                             input_width))
                    cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                resize_image(lab,
                                             input_height,
                                             input_width))
                    indexer += 1
            if adding_rgb_foreground:
                img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                for i_n in range(number_of_backgrounds_per_image):
                    background_image_chosen_name = random.choice(list_all_possible_background_images)
                    foreground_rgb_chosen_name = random.choice(list_all_possible_foreground_rgbs)
                    img_rgb_background_chosen = \
                        cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
                    foreground_rgb_chosen = \
                        np.load(dir_rgb_foregrounds + '/' + foreground_rgb_chosen_name)
                    img_with_overlayed_background = \
                        return_binary_image_with_given_rgb_background_and_given_foreground_rgb(
                            img_bin_corr, img_rgb_background_chosen, foreground_rgb_chosen)
                    cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                resize_image(img_with_overlayed_background,
                                             input_height,
                                             input_width))
                    cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                resize_image(lab,
                                             input_height,
                                             input_width))
                    indexer += 1
            if add_red_textlines:
                img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                img_red_context = \
                    return_image_with_red_elements(img, img_bin_corr)
                cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                            resize_image(img_red_context,
                                         input_height,
                                         input_width))
                cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                            resize_image(lab,
                                         input_height,
                                         input_width))
                indexer += 1
    else:
        indexer = get_patches(dir_flow_train_imgs,
                              dir_flow_train_labels,
                              img,
                              lab,
                              input_height,
                              input_width,
                              indexer=indexer)
        if augmentation:
            if rotation:
                indexer = get_patches(dir_flow_train_imgs,
                                      dir_flow_train_labels,
                                      rotation_90(img),
                                      rotation_90(lab),
                                      input_height,
                                      input_width,
                                      indexer=indexer)
            if rotation_not_90:
                for thetha_i in thetha:
                    img_max_rotated, label_max_rotated = \
                        rotation_not_90_func(img, lab, thetha_i)
                    indexer = get_patches(dir_flow_train_imgs,
                                          dir_flow_train_labels,
                                          img_max_rotated,
                                          label_max_rotated,
                                          input_height,
                                          input_width,
                                          indexer=indexer)
            if channels_shuffling:
                for shuffle_index in shuffle_indexes:
                    img_shuffled = \
                        return_shuffled_channels(img, shuffle_index),
                    indexer = get_patches(dir_flow_train_imgs,
                                          dir_flow_train_labels,
                                          img_shuffled,
                                          lab,
                                          input_height,
                                          input_width,
                                          indexer=indexer)
            if adding_rgb_background:
                img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                for i_n in range(number_of_backgrounds_per_image):
                    background_image_chosen_name = random.choice(list_all_possible_background_images)
                    img_rgb_background_chosen = \
                        cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
                    img_with_overlayed_background = \
                        return_binary_image_with_given_rgb_background(
                            img_bin_corr, img_rgb_background_chosen)
                    indexer = get_patches(dir_flow_train_imgs,
                                          dir_flow_train_labels,
                                          img_with_overlayed_background,
                                          lab,
                                          input_height,
                                          input_width,
                                          indexer=indexer)
            if adding_rgb_foreground:
                img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                for i_n in range(number_of_backgrounds_per_image):
                    background_image_chosen_name = random.choice(list_all_possible_background_images)
                    foreground_rgb_chosen_name = random.choice(list_all_possible_foreground_rgbs)
                    img_rgb_background_chosen = \
                        cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
                    foreground_rgb_chosen = \
                        np.load(dir_rgb_foregrounds + '/' + foreground_rgb_chosen_name)
                    img_with_overlayed_background = \
                        return_binary_image_with_given_rgb_background_and_given_foreground_rgb(
                            img_bin_corr, img_rgb_background_chosen, foreground_rgb_chosen)
                    indexer = get_patches(dir_flow_train_imgs,
                                          dir_flow_train_labels,
                                          img_with_overlayed_background,
                                          lab,
                                          input_height,
                                          input_width,
                                          indexer=indexer)
            if add_red_textlines:
                img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                img_red_context = \
                    return_image_with_red_elements(img, img_bin_corr)
                indexer = get_patches(dir_flow_train_imgs,
                                      dir_flow_train_labels,
                                      img_red_context,
                                      lab,
                                      input_height,
                                      input_width,
                                      indexer=indexer)
            if flip_aug:
                for f_i in flip_index:
                    indexer = get_patches(dir_flow_train_imgs,
                                          dir_flow_train_labels,
                                          cv2.flip(img, f_i),
                                          cv2.flip(lab, f_i),
                                          input_height,
                                          input_width,
                                          indexer=indexer)
            if blur_aug:
                for blur_i in blur_k:
                    indexer = get_patches(dir_flow_train_imgs,
                                          dir_flow_train_labels,
                                          bluring(img, blur_i),
                                          lab,
                                          input_height,
                                          input_width,
                                          indexer=indexer)
            if padding_black:
                indexer = get_patches(dir_flow_train_imgs,
                                      dir_flow_train_labels,
                                      do_padding_black(img),
                                      do_padding_label(lab),
                                      input_height,
                                      input_width,
                                      indexer=indexer)
            if padding_white:
                indexer = get_patches(dir_flow_train_imgs,
                                      dir_flow_train_labels,
                                      do_padding_white(img),
                                      do_padding_label(lab),
                                      input_height,
                                      input_width,
                                      indexer=indexer)
            if brightening:
                for factor in brightness:
                    indexer = get_patches(dir_flow_train_imgs,
                                          dir_flow_train_labels,
                                          do_brightening(img, factor),
                                          lab,
                                          input_height,
                                          input_width,
                                          indexer=indexer)
            if scaling:
                for sc_ind in scales:
                    indexer = get_patches_num_scale_new(
                        dir_flow_train_imgs,
                        dir_flow_train_labels,
                        img ,
                        lab,
                        input_height,
                        input_width,
                        indexer=indexer,
                        scaler=sc_ind)
            if degrading:
                for degrade_scale_ind in degrade_scales:
                    img_deg = \
                        do_degrading(img, degrade_scale_ind),
                    indexer = get_patches(dir_flow_train_imgs,
                                          dir_flow_train_labels,
                                          img_deg,
                                          lab,
                                          input_height,
                                          input_width,
                                          indexer=indexer)
            if binarization:
                if dir_img_bin:
                    img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                    indexer = get_patches(dir_flow_train_imgs,
                                          dir_flow_train_labels,
                                          img_bin_corr,
                                          lab,
                                          input_height,
                                          input_width,
                                          indexer=indexer)
                else:
                    indexer = get_patches(dir_flow_train_imgs,
                                          dir_flow_train_labels,
                                          otsu_copy(img),
                                          lab,
                                          input_height,
                                          input_width,
                                          indexer=indexer)
            if scaling_brightness:
                for sc_ind in scales:
                    for factor in brightness:
                        img_bright = do_brightening(img, factor)
                        indexer = get_patches_num_scale_new(
                            dir_flow_train_imgs,
                            dir_flow_train_labels,
                            img_bright,
                            lab,
                            input_height,
                            input_width,
                            indexer=indexer,
                            scaler=sc_ind)
            if scaling_bluring:
                for sc_ind in scales:
                    for blur_i in blur_k:
                        img_blur = bluring(img, blur_i),
                        indexer = get_patches_num_scale_new(
                            dir_flow_train_imgs,
                            dir_flow_train_labels,
                            img_blur,
                            lab,
                            input_height,
                            input_width,
                            indexer=indexer,
                            scaler=sc_ind)
            if scaling_binarization:
                for sc_ind in scales:
                    img_bin = otsu_copy(img),
                    indexer = get_patches_num_scale_new(
                        dir_flow_train_imgs,
                        dir_flow_train_labels,
                        img_bin,
                        lab,
                        input_height,
                        input_width,
                        indexer=indexer,
                        scaler=sc_ind)
            if scaling_flip:
                for sc_ind in scales:
                    for f_i in flip_index:
                        indexer = get_patches_num_scale_new(
                            dir_flow_train_imgs,
                            dir_flow_train_labels,
                            cv2.flip(img, f_i),
                            cv2.flip(lab, f_i),
                            input_height,
                            input_width,
                            indexer=indexer,
                            scaler=sc_ind)
    return indexer
