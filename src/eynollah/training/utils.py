import os
import math
import random
from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
import seaborn as sns
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
import imutils
import tensorflow as tf

from PIL import Image, ImageFile, ImageEnhance

ImageFile.LOAD_TRUNCATED_IMAGES = True


def vectorize_label(label, char_to_num, padding_token, max_len):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label

def scale_padd_image_for_ocr(img, height, width):
    ratio = height /float(img.shape[0])

    w_ratio = int(ratio * img.shape[1])

    if w_ratio<=width:
        width_new = w_ratio
    else:
        width_new = width

    if width_new <= 0:
        width_new = width

    img_res= resize_image (img, height, width_new)
    img_fin = np.ones((height, width, 3))*255

    img_fin[:,:width_new,:] = img_res[:,:,:]
    return img_fin

# TODO: document where this is from
def add_salt_and_pepper_noise(img, salt_prob, pepper_prob):
    """
    Add salt-and-pepper noise to an image.
    
    Parameters:
        image: ndarray
            Input image.
        salt_prob: float
            Probability of salt noise.
        pepper_prob: float
            Probability of pepper noise.
            
    Returns:
        noisy_image: ndarray
            Image with salt-and-pepper noise.
    """
    # Make a copy of the image
    noisy_image = np.copy(img)
    
    # Generate random noise
    total_pixels = img.size
    num_salt = int(salt_prob * total_pixels)
    num_pepper = int(pepper_prob * total_pixels)
    
    # Add salt noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    noisy_image[coords[0], coords[1]] = 255  # white pixels
    
    # Add pepper noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    noisy_image[coords[0], coords[1]] = 0  # black pixels
    
    return noisy_image

def invert_image(img):
    img_inv = 255 - img
    return img_inv

def return_image_with_strapped_white_noises(img):
    img_w_noised = np.copy(img)
    img_h, img_width = img.shape[0], img.shape[1]
    n = 9
    p = 0.3
    num_windows = np.random.binomial(n, p, 1)[0]
    
    if num_windows<1:
        num_windows = 1
        
    loc_of_windows = np.random.uniform(0,img_width,num_windows).astype(np.int64)
    width_windows = np.random.uniform(10,50,num_windows).astype(np.int64)
    
    for i, loc in enumerate(loc_of_windows):
        noise = np.random.normal(0, 50, (img_h, width_windows[i], 3))
        
        try:
            img_w_noised[:, loc:loc+width_windows[i], : ] = noise[:,:,:]
        except:
            pass
    return img_w_noised

def do_padding_for_ocr(img, percent_height, padding_color):
    padding_size = int( img.shape[0]*percent_height/2. )
    height_new = img.shape[0] + 2*padding_size
    width_new = img.shape[1] + 2*padding_size

    h_start = padding_size
    w_start = padding_size

    if padding_color == 'white':
        img_new = np.ones((height_new, width_new, img.shape[2])).astype(float) * 255
    elif padding_color == 'black':
        img_new = np.zeros((height_new, width_new, img.shape[2])).astype(float)
    else:
        raise ValueError("padding_color must be 'white' or 'black'")

    img_new[h_start:h_start + img.shape[0], w_start:w_start + img.shape[1], :] = np.copy(img[:, :, :])


    return img_new

# TODO: document where this is from
def do_deskewing(img, amplitude):
    height, width = img.shape[:2]

    # Generate sinusoidal wave distortion with reduced amplitude
    #amplitude = 8 # 5 # Reduce the amplitude for less curvature
    frequency = 300  # Increase frequency to stretch the curve
    x_indices = np.tile(np.arange(width), (height, 1))
    y_indices = np.arange(height).reshape(-1, 1) + amplitude * np.sin(2 * np.pi * x_indices / frequency)

    # Convert indices to float32 for remapping
    map_x = x_indices.astype(np.float32)
    map_y = y_indices.astype(np.float32)

    # Apply the remap to create the curve
    curved_image = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return curved_image

# TODO: document where this is from
def do_direction_in_depth(img, direction: str):
    height, width = img.shape[:2]

    if direction == 'left':
        # Define the original corner points of the image
        src_points = np.float32([
            [0, 0],          # Top-left corner
            [width, 0],      # Top-right corner
            [0, height],     # Bottom-left corner
            [width, height]  # Bottom-right corner
        ])

        # Define the new corner points for a subtle right-to-left tilt
        dst_points = np.float32([
            [2, 13],                # Slight inward shift for top-left
            [width, 0],            # Slight downward shift for top-right
            [2, height-13],           # Slight inward shift for bottom-left
            [width, height]    # Slight upward shift for bottom-right
        ])
    elif direction == 'right':
        # Define the original corner points of the image
        src_points = np.float32([
            [0, 0],          # Top-left corner
            [width, 0],      # Top-right corner
            [0, height],     # Bottom-left corner
            [width, height]  # Bottom-right corner
        ])

        # Define the new corner points for a subtle right-to-left tilt
        dst_points = np.float32([
            [0, 0],                # Slight inward shift for top-left
            [width, 13],            # Slight downward shift for top-right
            [0, height],           # Slight inward shift for bottom-left
            [width, height - 13]    # Slight upward shift for bottom-right
        ])

    elif direction == 'up':
        # Define the original corner points of the image
        src_points = np.float32([
            [0, 0],          # Top-left corner
            [width, 0],      # Top-right corner
            [0, height],     # Bottom-left corner
            [width, height]  # Bottom-right corner
        ])

        # Define the new corner points to simulate a tilted perspective
        # Make the top part appear closer and the bottom part farther
        dst_points = np.float32([
            [50, 0],                 # Top-left moved inward
            [width - 50, 0],         # Top-right moved inward
            [0, height],             # Bottom-left remains the same
            [width, height]          # Bottom-right remains the same
        ])
    elif direction == 'down':
        # Define the original corner points of the image
        src_points = np.float32([
            [0, 0],          # Top-left corner
            [width, 0],      # Top-right corner
            [0, height],     # Bottom-left corner
            [width, height]  # Bottom-right corner
        ])

        # Define the new corner points to simulate a tilted perspective
        # Make the top part appear closer and the bottom part farther
        dst_points = np.float32([
            [0, 0],                 # Top-left moved inward
            [width, 0],         # Top-right moved inward
            [50, height],             # Bottom-left remains the same
            [width - 50, height]          # Bottom-right remains the same
        ])
    else:
        raise ValueError("direction must be 'left', 'right', 'up' or 'down'")

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective warp
    warped_image = cv2.warpPerspective(img, matrix, (width, height))
    return warped_image


def return_shuffled_channels(img, channels_order):
    """
    channels order in ordinary case is like this [0, 1, 2]. In the case of shuffling the order should be provided.
    """
    img_sh = np.copy(img)
    
    img_sh[:,:,0]= img[:,:,channels_order[0]]
    img_sh[:,:,1]= img[:,:,channels_order[1]]
    img_sh[:,:,2]= img[:,:,channels_order[2]]
    return img_sh

# TODO: Refactor into one {{{
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

# }}}
    
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
    else:
        raise ValueError("kind must be 'gauss', 'median' or 'blur'")
    return img_blur


# TODO: document where this is from
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


# TODO: Use one of the utils/rotate.py functions for this
def rotation_90(img):
    img_rot = np.zeros((img.shape[1], img.shape[0], img.shape[2]))
    img_rot[:, :, 0] = img[:, :, 0].T
    img_rot[:, :, 1] = img[:, :, 1].T
    img_rot[:, :, 2] = img[:, :, 2].T
    return img_rot


# TODO: document where this is from
# TODO: Use one of the utils/rotate.py functions for this
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


# TODO: Use one of the utils/rotate.py functions for this
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

# TODO: Use one of the utils/rotate.py functions for this
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

# TODO: Use one of the utils/rotate.py functions for this
def rotation_not_90_func(img, label, thetha):
    rotated = imutils.rotate(img, thetha)
    rotated_label = imutils.rotate(label, thetha)
    return rotate_max_area(img, rotated, rotated_label, thetha)


# TODO: Use one of the utils/rotate.py functions for this
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


# TODO: use resize_image from utils
def resize_image(seg_in, input_height, input_width):
    return cv2.resize(seg_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)


def get_one_hot(seg, input_height, input_width, n_classes):
    seg = seg[:, :, 0]
    seg_f = np.zeros((input_height, input_width, n_classes))
    for j in range(n_classes):
        seg_f[:, :, j] = (seg == j).astype(int)
    return seg_f


# TODO: document where this is from
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

def generate_arrays_from_folder_reading_order(classes_file_dir, modal_dir, n_batch, height, width, n_classes, thetha, augmentation=False):
    all_labels_files = os.listdir(classes_file_dir)
    ret_x= np.zeros((n_batch, height, width, 3))#.astype(np.int16)
    ret_y= np.zeros((n_batch, n_classes)).astype(np.int16)
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
            if batchcount>=n_batch:
                yield ret_x, ret_y
                ret_x= np.zeros((n_batch, height, width, 3))#.astype(np.int16)
                ret_y= np.zeros((n_batch, n_classes)).astype(np.int16)
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
                    if batchcount>=n_batch:
                        yield ret_x, ret_y
                        ret_x= np.zeros((n_batch, height, width, 3))#.astype(np.int16)
                        ret_y= np.zeros((n_batch, n_classes)).astype(np.int16)
                        batchcount = 0

def data_gen(img_folder, mask_folder, batch_size, input_height, input_width, n_classes, task='segmentation'):
    c = 0
    n = [f for f in os.listdir(img_folder) if not f.startswith('.')]  # os.listdir(img_folder) #List of training images
    random.shuffle(n)
    img = np.zeros((batch_size, input_height, input_width, 3), dtype=float)
    mask = np.zeros((batch_size, input_height, input_width, n_classes), dtype=float)
    while True:
        for i in range(c, c + batch_size):  # initially from 0 to 16, c = 0.
            try:
                filename = os.path.splitext(n[i])[0]

                train_img = cv2.imread(img_folder + '/' + n[i]) / 255.
                train_img = cv2.resize(train_img, (input_width, input_height),
                                       interpolation=cv2.INTER_NEAREST)  # Read an image from folder and resize

                img[i - c, :] = train_img  # add to array - img[0], img[1], and so on.
                if task == "segmentation" or task=="binarization":
                    train_mask = cv2.imread(mask_folder + '/' + filename + '.png')
                    train_mask = resize_image(train_mask, input_height, input_width)
                    train_mask = get_one_hot(train_mask, input_height, input_width, n_classes)
                elif task == "enhancement":
                    train_mask = cv2.imread(mask_folder + '/' + filename + '.png')/255.
                    train_mask = resize_image(train_mask, input_height, input_width)
                    
                # train_mask = train_mask.reshape(224, 224, 1) # Add extra dimension for parity with train_img size [512 * 512 * 3]

                mask[i - c, :] = train_mask
            except Exception as e:
                print(str(e))
                img[i - c, :] = 1.
                mask[i - c, :] = 0.

        c += batch_size
        if c + batch_size >= len(os.listdir(img_folder)):
            c = 0
            random.shuffle(n)
        yield img, mask


# TODO: Use otsu_copy from utils
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


def get_patches(img, label, height, width):
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

            yield img_patch, label_patch


def do_padding_with_color(img, padding_color='black'):
    index_start_h = 4
    index_start_w = 4
    
    img_padded = np.zeros((img.shape[0] + 2*index_start_h, img.shape[1]+ 2*index_start_w, img.shape[2]))
    if padding_color == 'white':
        img_padded += 255
    img_padded[index_start_h: index_start_h + img.shape[0], index_start_w: index_start_w + img.shape[1], :] = img[:, :, :]
    
    return img_padded.astype(float)


def do_degrading(img, scale):
    img_org_h = img.shape[0]
    img_org_w = img.shape[1]
    
    img_res = resize_image(img, int(img_org_h * scale), int(img_org_w * scale))
    
    return resize_image(img_res, img_org_h, img_org_w)
    
# TODO: How is this different from do_padding_black?
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


def get_patches_num_scale_new(img, label, height, width, scaler=1.0):
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
            
            yield img_patch, label_patch


# TODO: refactor to combine with data_gen_ocr
def preprocess_imgs(config,
                    imgs_list,
                    labs_list,
                    dir_img,
                    dir_lab,
                    dir_flow_imgs,
                    dir_flow_lbls,
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

    seed = random.random()
    random.shuffle(imgs_list, random=lambda: seed)
    random.shuffle(labs_list, random=lambda: seed)

    # labs_list not used because stem matching more robust
    indexer = 0
    for img, lab in tqdm(zip(imgs_list, labs_list)):
        img = cv2.imread(os.path.join(dir_img, img))
        img_name = os.path.splitext(img)[0]
        if config['task'] in ["segmentation", "binarization"]:
            # assert lab == img_name + '.png'
            lab = cv2.imread(os.path.join(dir_lab, img_name + '.png'))
        elif config['task'] == "enhancement":
            lab = cv2.imread(os.path.join(dir_lab, img))
        elif config['task'] == "cnn-rnn-ocr":
            # assert lab == 'img_name + '.txt'
            with open(os.path.join(dir_lab, img_name + '.txt'), 'r') as f:
                lab = f.read().split('\n')[0]
        else:
            lab = None

        try:
            if config['task'] == "cnn-rnn-ocr":
                yield from preprocess_img_ocr(img, img_name, lab,
                                              **config)
                continue
            for img, lab in preprocess_img(img, img_name, lab,
                                           **config):
                cv2.imwrite(os.path.join(dir_flow_imgs, '/img_%d.png' % indexer),
                            resize_image(img,
                                         config['input_height'],
                                         config['input_width']))
                cv2.imwrite(os.path.join(dir_flow_lbls, '/img_%d.png' % indexer),
                            resize_image(lab,
                                         config['input_height'],
                                         config['input_width']))
                indexer += 1
        except:
            logger.exception("skipping image %s", img_name)

def preprocess_img(img,
                   img_name,
                   lab,
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
        yield img, lab
        if augmentation:
            if flip_aug:
                for f_i in flip_index:
                    yield cv2.flip(img, f_i), cv2.flip(lab, f_i)
            if blur_aug:
                for blur_i in blur_k:
                    yield bluring(img, blur_i), lab
            if brightening:
                for factor in brightness:
                    yield do_brightening(img, factor), lab
            if binarization:
                if dir_img_bin:
                    img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                else:
                    img_bin_corr = otsu_copy(img)
                yield img_bin_corr, lab
            if degrading:
                for degrade_scale_ind in degrade_scales:
                    yield do_degrading(img, degrade_scale_ind), lab
            if rotation_not_90:
                for thetha_i in thetha:
                    yield rotation_not_90_func(img, lab, thetha_i)
            if channels_shuffling:
                for shuffle_index in shuffle_indexes:
                    yield return_shuffled_channels(img, shuffle_index), lab
            if scaling:
                for sc_ind in scales:
                    yield scale_image_for_no_patch(img, lab, sc_ind)
            if shifting:
                shift_types = ['xpos', 'xmin', 'ypos', 'ymin', 'xypos', 'xymin']
                for st_ind in shift_types:
                    yield shift_image_and_label(img, lab, st_ind)
            if adding_rgb_background:
                img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                for i_n in range(number_of_backgrounds_per_image):
                    background_image_chosen_name = random.choice(list_all_possible_background_images)
                    img_rgb_background_chosen = \
                        cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
                    img_with_overlayed_background = \
                        return_binary_image_with_given_rgb_background(
                            img_bin_corr, img_rgb_background_chosen)
                    yield img_with_overlayed_background, lab
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
                    yield img_with_overlayed_background, lab
            if add_red_textlines:
                img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                yield return_image_with_red_elements(img, img_bin_corr), lab
    else:
        yield from get_patches(img,
                               lab,
                               input_height,
                               input_width)
        if augmentation:
            if rotation:
                yield from get_patches(rotation_90(img),
                                       rotation_90(lab),
                                       input_height,
                                       input_width)
            if rotation_not_90:
                for thetha_i in thetha:
                    img_max_rotated, label_max_rotated = \
                        rotation_not_90_func(img, lab, thetha_i)
                    yield from get_patches(img_max_rotated,
                                           label_max_rotated,
                                           input_height,
                                           input_width)
            if channels_shuffling:
                for shuffle_index in shuffle_indexes:
                    img_shuffled = \
                        return_shuffled_channels(img, shuffle_index),
                    yield from get_patches(img_shuffled,
                                           lab,
                                           input_height,
                                           input_width)
            if adding_rgb_background:
                img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                for i_n in range(number_of_backgrounds_per_image):
                    background_image_chosen_name = random.choice(list_all_possible_background_images)
                    img_rgb_background_chosen = \
                        cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
                    img_with_overlayed_background = \
                        return_binary_image_with_given_rgb_background(
                            img_bin_corr, img_rgb_background_chosen)
                    yield from get_patches(img_with_overlayed_background,
                                           lab,
                                           input_height,
                                           input_width)
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
                    yield from get_patches(img_with_overlayed_background,
                                           lab,
                                           input_height,
                                           input_width)
            if add_red_textlines:
                img_bin_corr = cv2.imread(os.path.join(dir_img_bin, img_name + '.png'))
                img_red_context = \
                    return_image_with_red_elements(img, img_bin_corr)
                yield from get_patches(img_red_context,
                                       lab,
                                       input_height,
                                       input_width)
            if flip_aug:
                for f_i in flip_index:
                    yield from get_patches(cv2.flip(img, f_i),
                                           cv2.flip(lab, f_i),
                                           input_height,
                                           input_width)
            if blur_aug:
                for blur_i in blur_k:
                    yield from get_patches(bluring(img, blur_i),
                                           lab,
                                           input_height,
                                           input_width)
            if padding_black:
                yield from get_patches(do_padding_black(img),
                                       do_padding_label(lab),
                                       input_height,
                                       input_width)
            if padding_white:
                yield from get_patches(do_padding_white(img),
                                       do_padding_label(lab),
                                       input_height,
                                       input_width)
            if brightening:
                for factor in brightness:
                    yield from get_patches(do_brightening(img, factor),
                                           lab,
                                           input_height,
                                           input_width)
            if scaling:
                for sc_ind in scales:
                    yield from get_patches_num_scale_new(img,
                                                         lab,
                                                         input_height,
                                                         input_width,
                                                         scaler=sc_ind)
            if degrading:
                for degrade_scale_ind in degrade_scales:
                    img_deg = \
                        do_degrading(img, degrade_scale_ind),
                    yield from get_patches(img_deg,
                                           lab,
                                           input_height,
                                           input_width)
            if binarization:
                if dir_img_bin:
                    img_bin_corr = cv2.imread(os.path.join(dir_img_bin, img_name + '.png'))
                else:
                    img_bin_corr = otsu_copy(img)
                yield from get_patches(img_bin_corr,
                                       lab,
                                       input_height,
                                       input_width)
            if scaling_brightness:
                for sc_ind in scales:
                    for factor in brightness:
                        img_bright = do_brightening(img, factor)
                        yield from get_patches_num_scale_new(img_bright,
                                                             lab,
                                                             input_height,
                                                             input_width,
                                                             scaler=sc_ind)
            if scaling_bluring:
                for sc_ind in scales:
                    for blur_i in blur_k:
                        img_blur = bluring(img, blur_i),
                        yield from get_patches_num_scale_new(img_blur,
                                                             lab,
                                                             input_height,
                                                             input_width,
                                                             scaler=sc_ind)
            if scaling_binarization:
                for sc_ind in scales:
                    img_bin = otsu_copy(img),
                    yield from get_patches_num_scale_new(img_bin,
                                                         lab,
                                                         input_height,
                                                         input_width,
                                                         scaler=sc_ind)
            if scaling_flip:
                for sc_ind in scales:
                    for f_i in flip_index:
                        yield from get_patches_num_scale_new(cv2.flip(img, f_i),
                                                             cv2.flip(lab, f_i),
                                                             input_height,
                                                             input_width,
                                                             scaler=sc_ind)
                            
def preprocess_img_ocr(
    img,
    img_name,
    lab,
    char_to_num=None,
    padding_token=-1,
    max_len=500,
    n_batch=1,
    input_height=None,
    input_width=None,
    augmentation=False,
    color_padding_rotation=None,
    thetha_padd=None,
    padd_colors=None,
    rotation_not_90=None,
    thetha=None,
    padding_white=None,
    white_padds=None,
    degrading=False,
    bin_deg=None,
    degrade_scales=None,
    blur_aug=False,
    blur_k=None,
    brightening=False,
    brightness=None,
    binarization=False,
    image_inversion=False,
    channels_shuffling=False,
    shuffle_indexes=None,
    white_noise_strap=False,
    textline_skewing=False,
    textline_skewing_bin=False,
    skewing_amplitudes=None,
    textline_left_in_depth=False,
    textline_left_in_depth_bin=False,
    textline_right_in_depth=False,
    textline_right_in_depth_bin=False,
    textline_up_in_depth=False,
    textline_up_in_depth_bin=False,
    textline_down_in_depth=False,
    textline_down_in_depth_bin=False,
    pepper_aug=False,
    pepper_bin_aug=False,
    pepper_indexes=None,
    dir_img_bin=None,
    add_red_textlines=False,
    adding_rgb_background=False,
    dir_rgb_backgrounds=None,
    adding_rgb_foreground=False,
    dir_rgb_foregrounds=None,
    number_of_backgrounds_per_image=None,
    list_all_possible_background_images=None,
    list_all_possible_foreground_rgbs=None,
):
    def scale_image(img):
        return scale_padd_image_for_ocr(img, input_height, input_width).astype(np.float32) / 255.
    #lab = vectorize_label(lab, char_to_num, padding_token, max_len)
    # now padded at Dataset.padded_batch
    lab = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    yield scale_image(img), lab
    #to_yield = {"image": ret_x, "label": ret_y}

    if dir_img_bin:
        img_bin_corr = cv2.imread(os.path.join(dir_img_bin, img_name + '.png'))
    else:
        img_bin_corr = None

    if not augmentation:
        return

    if color_padding_rotation:
        for thetha_ind in thetha_padd:
            for padd_col in padd_colors:
                img_pad = do_padding_for_ocr(img, 1.2, padd_col)
                img_rot = rotation_not_90_func_single_image(img_pad, thetha_ind)
                yield scale_image(img_rot), lab
    if rotation_not_90:
        for thetha_ind in thetha:
            img_rot = rotation_not_90_func_single_image(img, thetha_ind)
            yield scale_image(img_rot), lab
    if blur_aug:
        for blur_type in blur_k:
            img_blur = bluring(img, blur_type)
            yield scale_image(img_blur), lab
    if degrading:
        for deg_scale_ind in degrade_scales:
            img_deg = do_degrading(img, deg_scale_ind)
            yield scale_image(img_deg), lab
    if bin_deg:
        for deg_scale_ind in degrade_scales:
            img_deg  = do_degrading(img_bin_corr, deg_scale_ind)
            yield scale_image(img_deg), lab
    if brightening:
        for bright_scale_ind in brightness:
            img_bright  = do_brightening(img, bright_scale_ind)
            yield scale_image(img_bright), lab
    if padding_white:
        for padding_size in white_padds:
            for padd_col in padd_colors:
                img_pad = do_padding_for_ocr(img, padding_size, padd_col)
                yield scale_image(img_pad), lab
    if adding_rgb_foreground:
        for i_n in range(number_of_backgrounds_per_image):
            background_image_chosen_name = random.choice(list_all_possible_background_images)
            foreground_rgb_chosen_name = random.choice(list_all_possible_foreground_rgbs)

            img_rgb_background_chosen = \
                cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
            foreground_rgb_chosen = \
                np.load(dir_rgb_foregrounds + '/' + foreground_rgb_chosen_name)

            img_fg = \
                return_binary_image_with_given_rgb_background_and_given_foreground_rgb(
                    img_bin_corr, img_rgb_background_chosen, foreground_rgb_chosen)
            yield scale_image(img_fg), lab
    if adding_rgb_background:
        for i_n in range(number_of_backgrounds_per_image):
            background_image_chosen_name = random.choice(list_all_possible_background_images)
            img_rgb_background_chosen = \
                cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
            img_bg = \
                return_binary_image_with_given_rgb_background(img_bin_corr, img_rgb_background_chosen)
            yield scale_image(img_bg), lab
    if binarization:
        yield scale_image(img_bin_corr), lab
    if image_inversion:
        img_inv = invert_image(img_bin_corr)
        yield scale_image(img_inv), lab
    if channels_shuffling:
        for shuffle_index in shuffle_indexes:
            img_shuf = return_shuffled_channels(img, shuffle_index)
            yield scale_image(img_shuf), lab
    if add_red_textlines:
        img_red = return_image_with_red_elements(img, img_bin_corr)
        yield scale_image(img_red), lab
    if white_noise_strap:
        img_noisy = return_image_with_strapped_white_noises(img)
        yield scale_image(img_noisy), lab
    if textline_skewing:
        for des_scale_ind in skewing_amplitudes:
            img_rot  = do_deskewing(img, des_scale_ind)
            yield scale_image(img_rot), lab
    if textline_skewing_bin:
        for des_scale_ind in skewing_amplitudes:
            img_rot  = do_deskewing(img_bin_corr, des_scale_ind)
            yield scale_image(img_rot), lab
    if textline_left_in_depth:
        img_warp  = do_direction_in_depth(img, 'left')
        yield scale_image(img_warp), lab
    if textline_left_in_depth_bin:
        img_warp  = do_direction_in_depth(img_bin_corr, 'left')
        yield scale_image(img_warp), lab
    if textline_right_in_depth:
        img_warp  = do_direction_in_depth(img, 'right')
        yield scale_image(img_warp), lab
    if textline_right_in_depth_bin:
        img_warp  = do_direction_in_depth(img_bin_corr, 'right')
        yield scale_image(img_warp), lab
    if textline_up_in_depth:
        img_warp  = do_direction_in_depth(img, 'up')
        yield scale_image(img_warp), lab
    if textline_up_in_depth_bin:
        img_warp  = do_direction_in_depth(img_bin_corr, 'up')
        yield scale_image(img_warp), lab
    if textline_down_in_depth:
        img_warp  = do_direction_in_depth(img, 'down')
        yield scale_image(img_warp), lab
    if textline_down_in_depth_bin:
        img_warp  = do_direction_in_depth(img_bin_corr, 'down')
        yield scale_image(img_warp), lab
    if pepper_aug:
        for pepper_ind in pepper_indexes:
            img_noisy = add_salt_and_pepper_noise(img, pepper_ind, pepper_ind)
            yield scale_image(img_noisy), lab
    if pepper_bin_aug:
        for pepper_ind in pepper_indexes:
            img_noisy = add_salt_and_pepper_noise(img_bin_corr, pepper_ind, pepper_ind)
            yield scale_image(img_noisy), lab
