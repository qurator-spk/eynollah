import os
import math
import random
from pathlib import Path
import cv2
import numpy as np
import seaborn as sns
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
import imutils
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
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

def generate_data_from_folder_training(path_classes, n_batch, height, width, n_classes, list_classes):
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
    ret_x= np.zeros((n_batch, height,width, 3)).astype(np.int16)
    ret_y= np.zeros((n_batch, n_classes)).astype(np.int16)
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
            
            if batchcount>=n_batch:
                ret_x = ret_x/255.
                yield ret_x, ret_y
                ret_x= np.zeros((n_batch, height,width, 3)).astype(np.int16)
                ret_y= np.zeros((n_batch, n_classes)).astype(np.int16)
                batchcount = 0

def do_brightening(img_in_dir, factor):
    im = Image.open(img_in_dir)
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


# TODO: (far) too many args
# TODO: refactor to combine with data_gen_ocr
def provide_patches(
    imgs_list_train,
    segs_list_train,
    dir_img,
    dir_seg,
    dir_flow_train_imgs,
    dir_flow_train_labels,
    input_height,
    input_width,
    blur_k,
    blur_aug,
    padding_white,
    padding_black,
    flip_aug,
    binarization,
    adding_rgb_background,
    adding_rgb_foreground,
    add_red_textlines,
    channels_shuffling,
    scaling,
    shifting,
    degrading,
    brightening,
    scales,
    degrade_scales,
    brightness,
    flip_index,
    shuffle_indexes,
    scaling_bluring,
    scaling_brightness,
    scaling_binarization,
    rotation,
    rotation_not_90,
    thetha,
    scaling_flip,
    task,
    augmentation=False,
    patches=False,
    dir_img_bin=None,
    number_of_backgrounds_per_image=None,
    list_all_possible_background_images=None,
    dir_rgb_backgrounds=None,
    dir_rgb_foregrounds=None,
    list_all_possible_foreground_rgbs=None,
):
    
    # TODO: why sepoarate var if you have seg_i?
    indexer = 0
    for im, seg_i in tqdm(zip(imgs_list_train, segs_list_train)):
        img_name = os.path.splitext(im)[0]
        if task == "segmentation" or task == "binarization":
            dir_of_label_file = os.path.join(dir_seg, img_name + '.png')
        elif task=="enhancement":
            dir_of_label_file = os.path.join(dir_seg, im)
            
        if not patches:
            cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png', resize_image(cv2.imread(dir_img + '/' + im), input_height, input_width))
            cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png', resize_image(cv2.imread(dir_of_label_file), input_height, input_width))
            indexer += 1
            
            if augmentation:
                if flip_aug:
                    for f_i in flip_index:
                        cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                    resize_image(cv2.flip(cv2.imread(dir_img+'/'+im),f_i),input_height,input_width) )
                        
                        cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                    resize_image(cv2.flip(cv2.imread(dir_of_label_file), f_i), input_height, input_width)) 
                        indexer += 1
                        
                if blur_aug:   
                    for blur_i in blur_k:
                        cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                    (resize_image(bluring(cv2.imread(dir_img + '/' + im), blur_i), input_height, input_width)))
                        
                        cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                    resize_image(cv2.imread(dir_of_label_file), input_height, input_width))
                        indexer += 1
                if brightening:
                    for factor in brightness:
                        try:
                            cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                        (resize_image(do_brightening(dir_img + '/' +im, factor), input_height, input_width)))
                        
                            cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                        resize_image(cv2.imread(dir_of_label_file), input_height, input_width))
                            indexer += 1
                        except:
                            pass
                    
                if binarization:
                    
                    if dir_img_bin:
                        img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                        
                        cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                    resize_image(img_bin_corr, input_height, input_width))
                    else:
                        cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                    resize_image(otsu_copy(cv2.imread(dir_img + '/' + im)), input_height, input_width))
                    
                    cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                resize_image(cv2.imread(dir_of_label_file), input_height, input_width))
                    indexer += 1
                    
                if degrading:  
                    for degrade_scale_ind in degrade_scales:
                        cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                    (resize_image(do_degrading(cv2.imread(dir_img + '/' + im), degrade_scale_ind), input_height, input_width)))
                        
                        cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                    resize_image(cv2.imread(dir_of_label_file), input_height, input_width))
                        indexer += 1
                        
                if rotation_not_90:
                    for thetha_i in thetha:
                        img_max_rotated, label_max_rotated = rotation_not_90_func(cv2.imread(dir_img + '/'+im),
                                                                                  cv2.imread(dir_of_label_file), thetha_i)
                        
                        cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png', resize_image(img_max_rotated, input_height, input_width))
                        
                        cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png', resize_image(label_max_rotated, input_height, input_width))
                        indexer += 1
                        
                if channels_shuffling:
                    for shuffle_index in shuffle_indexes:
                        cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png',
                                    (resize_image(return_shuffled_channels(cv2.imread(dir_img + '/' + im), shuffle_index), input_height, input_width)))
                        
                        cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                    resize_image(cv2.imread(dir_of_label_file), input_height, input_width))
                        indexer += 1
                        
                if scaling:
                    for sc_ind in scales:
                        img_scaled, label_scaled = scale_image_for_no_patch(cv2.imread(dir_img + '/'+im),
                                                                                  cv2.imread(dir_of_label_file), sc_ind)
                        
                        cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png', resize_image(img_scaled, input_height, input_width))
                        cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png', resize_image(label_scaled, input_height, input_width))
                        indexer += 1
                if shifting:
                    shift_types = ['xpos', 'xmin', 'ypos', 'ymin', 'xypos', 'xymin']
                    for st_ind in shift_types:
                        img_shifted, label_shifted = shift_image_and_label(cv2.imread(dir_img + '/'+im),
                                                                                  cv2.imread(dir_of_label_file), st_ind)
                        
                        cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png', resize_image(img_shifted, input_height, input_width))
                        cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png', resize_image(label_shifted, input_height, input_width))
                        indexer += 1
                        
                        
                if adding_rgb_background:
                    img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                    for i_n in range(number_of_backgrounds_per_image):
                        background_image_chosen_name = random.choice(list_all_possible_background_images)
                        img_rgb_background_chosen = cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
                        img_with_overlayed_background = return_binary_image_with_given_rgb_background(img_bin_corr, img_rgb_background_chosen)
                        
                        cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png', resize_image(img_with_overlayed_background, input_height, input_width))
                        cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                    resize_image(cv2.imread(dir_of_label_file), input_height, input_width))
                        
                        indexer += 1
                        
                if adding_rgb_foreground:
                    img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                    for i_n in range(number_of_backgrounds_per_image):
                        background_image_chosen_name = random.choice(list_all_possible_background_images)
                        foreground_rgb_chosen_name = random.choice(list_all_possible_foreground_rgbs)
                        
                        img_rgb_background_chosen = cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
                        foreground_rgb_chosen = np.load(dir_rgb_foregrounds + '/' + foreground_rgb_chosen_name)
                        
                        img_with_overlayed_background = return_binary_image_with_given_rgb_background_and_given_foreground_rgb(img_bin_corr, img_rgb_background_chosen, foreground_rgb_chosen)
                        
                        cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png', resize_image(img_with_overlayed_background, input_height, input_width))
                        cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                    resize_image(cv2.imread(dir_of_label_file), input_height, input_width))
                        
                        indexer += 1
                        
                if add_red_textlines:
                    img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                    img_red_context = return_image_with_red_elements(cv2.imread(dir_img + '/'+im), img_bin_corr)
                    
                    cv2.imwrite(dir_flow_train_imgs + '/img_' + str(indexer) + '.png', resize_image(img_red_context, input_height, input_width))
                    cv2.imwrite(dir_flow_train_labels + '/img_' + str(indexer) + '.png',
                                resize_image(cv2.imread(dir_of_label_file), input_height, input_width))
                    
                    indexer += 1
                        
                    
                    
                    
        if patches:
            indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                  cv2.imread(dir_img + '/' + im), cv2.imread(dir_of_label_file),
                                  input_height, input_width, indexer=indexer)
            
            if augmentation:
                if rotation:
                    indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                        rotation_90(cv2.imread(dir_img + '/' + im)),
                                        rotation_90(cv2.imread(dir_of_label_file)),
                                        input_height, input_width, indexer=indexer)
                    
                if rotation_not_90:
                    for thetha_i in thetha:
                        img_max_rotated, label_max_rotated = rotation_not_90_func(cv2.imread(dir_img + '/'+im),
                                                                                  cv2.imread(dir_of_label_file), thetha_i)
                        indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                              img_max_rotated,
                                              label_max_rotated,
                                              input_height, input_width, indexer=indexer)
                        
                if channels_shuffling:
                    for shuffle_index in shuffle_indexes:
                        indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                              return_shuffled_channels(cv2.imread(dir_img + '/' + im), shuffle_index),
                                              cv2.imread(dir_of_label_file),
                                              input_height, input_width, indexer=indexer)
                        
                if adding_rgb_background:
                    img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                    for i_n in range(number_of_backgrounds_per_image):
                        background_image_chosen_name = random.choice(list_all_possible_background_images)
                        img_rgb_background_chosen = cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
                        img_with_overlayed_background = return_binary_image_with_given_rgb_background(img_bin_corr, img_rgb_background_chosen)
                        
                        indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                              img_with_overlayed_background,
                                              cv2.imread(dir_of_label_file),
                                              input_height, input_width, indexer=indexer)
                        
                        
                if adding_rgb_foreground:
                    img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                    for i_n in range(number_of_backgrounds_per_image):
                        background_image_chosen_name = random.choice(list_all_possible_background_images)
                        foreground_rgb_chosen_name = random.choice(list_all_possible_foreground_rgbs)
                        
                        img_rgb_background_chosen = cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
                        foreground_rgb_chosen = np.load(dir_rgb_foregrounds + '/' + foreground_rgb_chosen_name)
                        
                        img_with_overlayed_background = return_binary_image_with_given_rgb_background_and_given_foreground_rgb(img_bin_corr, img_rgb_background_chosen, foreground_rgb_chosen)
                        
                        indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                              img_with_overlayed_background,
                                              cv2.imread(dir_of_label_file),
                                              input_height, input_width, indexer=indexer)
                        
                        
                if add_red_textlines:
                    img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                    img_red_context = return_image_with_red_elements(cv2.imread(dir_img + '/'+im), img_bin_corr)
                    
                    indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                            img_red_context,
                                            cv2.imread(dir_of_label_file),
                                            input_height, input_width, indexer=indexer)
                
                if flip_aug:
                    for f_i in flip_index:
                        indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                              cv2.flip(cv2.imread(dir_img + '/' + im), f_i),
                                              cv2.flip(cv2.imread(dir_of_label_file), f_i),
                                              input_height, input_width, indexer=indexer)
                if blur_aug:   
                    for blur_i in blur_k:
                        indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                              bluring(cv2.imread(dir_img + '/' + im), blur_i),
                                              cv2.imread(dir_of_label_file),
                                              input_height, input_width, indexer=indexer)          
                if padding_black:
                    indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                          do_padding_black(cv2.imread(dir_img + '/' + im)),
                                          do_padding_label(cv2.imread(dir_of_label_file)),
                                          input_height, input_width, indexer=indexer)       
        
                if padding_white:   
                    indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                          do_padding_white(cv2.imread(dir_img + '/'+im)),
                                          do_padding_label(cv2.imread(dir_of_label_file)),
                                          input_height, input_width, indexer=indexer)       
                    
                if brightening:
                    for factor in brightness:
                        try:
                            indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                                  do_brightening(dir_img + '/' +im, factor),
                                                  cv2.imread(dir_of_label_file),
                                                  input_height, input_width, indexer=indexer)
                        except:
                            pass
                if scaling:  
                    for sc_ind in scales:
                        indexer = get_patches_num_scale_new(dir_flow_train_imgs, dir_flow_train_labels,
                                                            cv2.imread(dir_img + '/' + im) ,
                                                            cv2.imread(dir_of_label_file),
                                                            input_height, input_width, indexer=indexer, scaler=sc_ind)
                        
                if degrading:  
                    for degrade_scale_ind in degrade_scales:
                        indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                              do_degrading(cv2.imread(dir_img + '/' + im), degrade_scale_ind),
                                              cv2.imread(dir_of_label_file),
                                              input_height, input_width, indexer=indexer)
                        
                if binarization:
                    if dir_img_bin:
                        img_bin_corr = cv2.imread(dir_img_bin + '/' + img_name+'.png')
                        
                        indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                            img_bin_corr,
                                            cv2.imread(dir_of_label_file),
                                            input_height, input_width, indexer=indexer)
                        
                    else:
                        indexer = get_patches(dir_flow_train_imgs, dir_flow_train_labels,
                                            otsu_copy(cv2.imread(dir_img + '/' + im)),
                                            cv2.imread(dir_of_label_file),
                                            input_height, input_width, indexer=indexer)

                if scaling_brightness:
                    for sc_ind in scales:
                        for factor in brightness:
                            try:
                                indexer = get_patches_num_scale_new(dir_flow_train_imgs,
                                                                    dir_flow_train_labels,
                                                                    do_brightening(dir_img + '/' + im, factor)
                                                                    ,cv2.imread(dir_of_label_file)
                                                                    ,input_height, input_width, indexer=indexer, scaler=sc_ind)
                            except:
                                pass
                        
                if scaling_bluring:  
                    for sc_ind in scales:
                        for blur_i in blur_k:
                            indexer = get_patches_num_scale_new(dir_flow_train_imgs, dir_flow_train_labels,
                                                                bluring(cv2.imread(dir_img + '/' + im), blur_i),
                                                                cv2.imread(dir_of_label_file),
                                                                input_height, input_width, indexer=indexer, scaler=sc_ind)

                if scaling_binarization:  
                    for sc_ind in scales:
                        indexer = get_patches_num_scale_new(dir_flow_train_imgs, dir_flow_train_labels,
                                                            otsu_copy(cv2.imread(dir_img + '/' + im)),
                                                            cv2.imread(dir_of_label_file),
                                                            input_height, input_width, indexer=indexer, scaler=sc_ind)
                        
                if scaling_flip:  
                    for sc_ind in scales:
                        for f_i in flip_index:
                            indexer = get_patches_num_scale_new(dir_flow_train_imgs, dir_flow_train_labels,
                                                                 
                                                                cv2.flip( cv2.imread(dir_img + '/' + im), f_i),
                                                                cv2.flip(cv2.imread(dir_of_label_file), f_i),
                                                                input_height, input_width, indexer=indexer, scaler=sc_ind)
                            
                            
                            
def data_gen_ocr(
    padding_token,
    n_batch,
    input_height,
    input_width,
    max_len,
    dir_train,
    ls_files_images,
    augmentation,
    color_padding_rotation,
    rotation_not_90,
    blur_aug,
    degrading,
    bin_deg,
    brightening,
    padding_white,
    adding_rgb_foreground,
    adding_rgb_background,
    binarization,
    image_inversion,
    channels_shuffling,
    add_red_textlines,
    white_noise_strap,
    textline_skewing,
    textline_skewing_bin,
    textline_left_in_depth,
    textline_left_in_depth_bin,
    textline_right_in_depth,
    textline_right_in_depth_bin,
    textline_up_in_depth,
    textline_up_in_depth_bin,
    textline_down_in_depth,
    textline_down_in_depth_bin,
    pepper_bin_aug,
    pepper_aug,
    degrade_scales,
    number_of_backgrounds_per_image,
    thetha,
    thetha_padd,
    brightness,
    padd_colors,
    shuffle_indexes,
    pepper_indexes,
    skewing_amplitudes,
    blur_k,
    char_to_num,
    list_all_possible_background_images,
    list_all_possible_foreground_rgbs,
    dir_rgb_backgrounds,
    dir_rgb_foregrounds,
    white_padds,
    dir_img_bin=None,
):
    
    random.shuffle(ls_files_images)

    ret_x= np.zeros((n_batch, input_height,  input_width, 3)).astype(np.float32)
    ret_y= np.zeros((n_batch, max_len)).astype(np.int16)+padding_token
    batchcount = 0

    def increment_batchcount(img_out, batchcount, ret_x, ret_y):
        to_yield = None
        img_out = scale_padd_image_for_ocr(img, input_height, input_width)
        ret_x[batchcount, :,:,:] = img_out[:,:,:]
        ret_y[batchcount, :] =  vectorize_label(txt_inp, char_to_num, padding_token, max_len)
        batchcount += 1
        if batchcount>=n_batch:
            ret_x = ret_x/255.
            to_yield = {"image": ret_x, "label": ret_y}
            ret_x= np.zeros((n_batch, input_height, input_width, 3)).astype(np.float32)
            ret_y= np.zeros((n_batch, max_len)).astype(np.int16)+padding_token
            batchcount = 0
        return img_out, batchcount, ret_x, ret_y, to_yield

    # TODO: Why while True + yield, why not return a list?
    while True:
        for i in ls_files_images:
            print(i, 'i')
            f_name = Path(i).stem#.split('.')[0]

            txt_inp  = open(os.path.join(dir_train, "labels/"+f_name+'.txt'),'r').read().split('\n')[0]
            
            img = cv2.imread(os.path.join(dir_train, "images/"+i) )
            if dir_img_bin:
                img_bin_corr = cv2.imread(os.path.join(dir_img_bin, f_name+'.png') )
            else:
                img_bin_corr = None

            
            if augmentation:
                img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img, batchcount, ret_x, ret_y)
                if to_yield: yield to_yield
                
                if color_padding_rotation:
                    for thetha_ind in thetha_padd:
                        for padd_col in padd_colors:
                            img_out = rotation_not_90_func_single_image(do_padding_for_ocr(img, 1.2, padd_col), thetha_ind)
                            img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                            if to_yield: yield to_yield
                        
                if rotation_not_90:
                    for thetha_ind in thetha:
                        img_out = rotation_not_90_func_single_image(img, thetha_ind)
                        img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                        if to_yield: yield to_yield
                    
                if blur_aug:
                    for blur_type in blur_k:
                        img_out = bluring(img, blur_type)
                        img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                        if to_yield: yield to_yield

                if degrading:
                    for deg_scale_ind in degrade_scales:
                        try:
                            img_out  = do_degrading(img, deg_scale_ind)
                        # TODO: qualify except
                        except:
                            img_out = np.copy(img)
                        img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                        if to_yield: yield to_yield
                            
                if bin_deg:
                    for deg_scale_ind in degrade_scales:
                        try:
                            img_out  = do_degrading(img_bin_corr, deg_scale_ind)
                        # TODO: qualify except
                        except:
                            img_out = np.copy(img_bin_corr)
                        img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                        if to_yield: yield to_yield
                
                if brightening:
                    for bright_scale_ind in brightness:
                        try:
                            # FIXME: dir_img is not defined in this scope, will always fail
                            img_out  = do_brightening(dir_img, bright_scale_ind)
                        # TODO: qualify except
                        except:
                            img_out = np.copy(img)
                        img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                        if to_yield: yield to_yield
                        
                if padding_white:
                    for padding_size in white_padds:
                        for padd_col in padd_colors:
                            img_out  = do_padding_for_ocr(img, padding_size, padd_col)
                            img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                            if to_yield: yield to_yield
                            
                if adding_rgb_foreground:
                    for i_n in range(number_of_backgrounds_per_image):
                        background_image_chosen_name = random.choice(list_all_possible_background_images)
                        foreground_rgb_chosen_name = random.choice(list_all_possible_foreground_rgbs)

                        img_rgb_background_chosen = cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
                        foreground_rgb_chosen = np.load(dir_rgb_foregrounds + '/' + foreground_rgb_chosen_name)

                        img_out = return_binary_image_with_given_rgb_background_and_given_foreground_rgb(img_bin_corr, img_rgb_background_chosen, foreground_rgb_chosen)
                        
                        img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                        if to_yield: yield to_yield
                        
                       
                if adding_rgb_background:
                    for i_n in range(number_of_backgrounds_per_image):
                        background_image_chosen_name = random.choice(list_all_possible_background_images)
                        img_rgb_background_chosen = cv2.imread(dir_rgb_backgrounds + '/' + background_image_chosen_name)
                        img_out = return_binary_image_with_given_rgb_background(img_bin_corr, img_rgb_background_chosen)
                        img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                        if to_yield: yield to_yield
                        
                if binarization:
                    img_out = scale_padd_image_for_ocr(img_bin_corr, input_height, input_width)
                    img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                    if to_yield: yield to_yield
                        
                if image_inversion:
                    img_out = invert_image(img_bin_corr)
                    img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                    if to_yield: yield to_yield

                if channels_shuffling:
                    for shuffle_index in shuffle_indexes:
                        img_out  = return_shuffled_channels(img, shuffle_index)
                        img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                        if to_yield: yield to_yield
                        
                if add_red_textlines:
                    img_out = return_image_with_red_elements(img, img_bin_corr)
                    img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                    if to_yield: yield to_yield
                        
                if white_noise_strap:
                    img_out  = return_image_with_strapped_white_noises(img)
                    img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                    if to_yield: yield to_yield
                        
                if textline_skewing:
                    for des_scale_ind in skewing_amplitudes:
                        try:
                            img_out  = do_deskewing(img, des_scale_ind)
                        # TODO: qualify except
                        except:
                            img_out = np.copy(img)
                        img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                        if to_yield: yield to_yield
                            
                if textline_skewing_bin:
                    for des_scale_ind in skewing_amplitudes:
                        try:
                            img_out  = do_deskewing(img_bin_corr, des_scale_ind)
                        # TODO: qualify except
                        except:
                            img_out = np.copy(img_bin_corr)
                        img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                        if to_yield: yield to_yield
                            
                if textline_left_in_depth:
                    try:
                        img_out  = do_direction_in_depth(img, 'left')
                    # TODO: qualify except
                    except:
                        img_out = np.copy(img)
                    img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                    if to_yield: yield to_yield
                        
                if textline_left_in_depth_bin:
                    try:
                        img_out  = do_direction_in_depth(img_bin_corr, 'left')
                    # TODO: qualify except
                    except:
                        img_out = np.copy(img_bin_corr)
                    img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                    if to_yield: yield to_yield
                        
                if textline_right_in_depth:
                    try:
                        img_out  = do_direction_in_depth(img_bin_corr, 'right')
                    # TODO: qualify except
                    except:
                        img_out = np.copy(img)
                    img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                    if to_yield: yield to_yield
                    
                        
                if textline_right_in_depth_bin:
                    try:
                        img_out  = do_direction_in_depth(img_bin_corr, 'right')
                    # TODO: qualify except
                    except:
                        img_out = np.copy(img_bin_corr)
                    img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                    if to_yield: yield to_yield
                        
                if textline_up_in_depth:
                    try:
                        img_out  = do_direction_in_depth(img, 'up')
                    # TODO: qualify except
                    except:
                        img_out = np.copy(img)
                    img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                    if to_yield: yield to_yield
                        
                if textline_up_in_depth_bin:
                    try:
                        img_out  = do_direction_in_depth(img_bin_corr, 'up')
                    # TODO: qualify except
                    except:
                        img_out = np.copy(img_bin_corr)
                    img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                    if to_yield: yield to_yield
                        
                if textline_down_in_depth:
                    try:
                        img_out  = do_direction_in_depth(img, 'down')
                    # TODO: qualify except
                    except:
                        img_out = np.copy(img)
                    img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                    if to_yield: yield to_yield
                        
                if textline_down_in_depth_bin:
                    try:
                        img_out  = do_direction_in_depth(img_bin_corr, 'down')
                    # TODO: qualify except
                    except:
                        img_out = np.copy(img_bin_corr)
                    img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                    if to_yield: yield to_yield
                        
                if pepper_bin_aug:
                    for pepper_ind in pepper_indexes:
                        img_out  = add_salt_and_pepper_noise(img_bin_corr, pepper_ind, pepper_ind)
                        img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                        if to_yield: yield to_yield
                            
                if pepper_aug:
                    for pepper_ind in pepper_indexes:
                        img_out  = add_salt_and_pepper_noise(img, pepper_ind, pepper_ind)
                        img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                        if to_yield: yield to_yield
                        
            else:
                img_out, batchcount, ret_x, ret_y, to_yield = increment_batchcount(img_out, batchcount, ret_x, ret_y)
                if to_yield: yield to_yield


# TODO: what is aug_multip and why calculate it in this way
def return_multiplier_based_on_augmnentations(
    augmentation,
    color_padding_rotation,
    rotation_not_90,
    blur_aug,
    degrading,
    bin_deg,
    brightening,
    padding_white,
    adding_rgb_foreground,
    adding_rgb_background,
    binarization,
    image_inversion,
    channels_shuffling,
    add_red_textlines,
    white_noise_strap,
    textline_skewing,
    textline_skewing_bin,
    textline_left_in_depth,
    textline_left_in_depth_bin,
    textline_right_in_depth,
    textline_right_in_depth_bin,
    textline_up_in_depth,
    textline_up_in_depth_bin,
    textline_down_in_depth,
    textline_down_in_depth_bin,
    pepper_bin_aug,
    pepper_aug,
    degrade_scales,
    number_of_backgrounds_per_image,
    thetha,
    thetha_padd,
    brightness,
    padd_colors,
    shuffle_indexes,
    pepper_indexes,
    skewing_amplitudes,
    blur_k,
    white_padds,
):
    aug_multip = 1
    if not augmentation:
        return 1

    if binarization:
        aug_multip += 1
    if image_inversion:
        aug_multip += 1
    if add_red_textlines:
        aug_multip += 1
    if white_noise_strap:
        aug_multip += 1
    if textline_right_in_depth:
        aug_multip += 1
    if textline_left_in_depth:
        aug_multip += 1
    if textline_up_in_depth:
        aug_multip += 1
    if textline_down_in_depth:
        aug_multip += 1
    if textline_right_in_depth_bin:
        aug_multip += 1
    if textline_left_in_depth_bin:
        aug_multip += 1
    if textline_up_in_depth_bin:
        aug_multip += 1
    if textline_down_in_depth_bin:
        aug_multip += 1
    if adding_rgb_foreground:
        aug_multip += number_of_backgrounds_per_image
    if adding_rgb_background:
        aug_multip += number_of_backgrounds_per_image
    if bin_deg:
        aug_multip += len(degrade_scales)
    if degrading:
        aug_multip += len(degrade_scales)
    if rotation_not_90:
        aug_multip += len(thetha)
    if textline_skewing:
        aug_multip += len(skewing_amplitudes)
    if textline_skewing_bin:
        aug_multip += len(skewing_amplitudes)
    if color_padding_rotation:
        aug_multip += len(thetha_padd)*len(padd_colors)
    if channels_shuffling:
        aug_multip += len(shuffle_indexes)
    if blur_aug:
        aug_multip += len(blur_k)
    if brightening:
        aug_multip += len(brightness)
    if padding_white:
        aug_multip += len(white_padds)*len(padd_colors)
    if pepper_aug:
        aug_multip += len(pepper_indexes)
    if pepper_bin_aug:
        aug_multip += len(pepper_indexes)
            
    return aug_multip
