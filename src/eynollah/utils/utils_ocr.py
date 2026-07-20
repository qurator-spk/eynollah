import math
import copy

import numpy as np
import cv2
# avoid module-level import:
# import tensorflow as tf
# (wait for tf-keras and logging setup in ModelZoo.load_model)
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFont

from . import pairwise
from .resize import resize_image


def decode_batch_predictions(pred, num_to_char, max_len = 128):
    import tensorflow as tf

    # input_len is the product of the batch size and the
    # number of time steps.
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    
    # Decode CTC predictions using greedy search.
    # decoded is a tuple with 2 elements.
    decoded = tf.keras.backend.ctc_decode(pred, 
                    input_length = input_len, 
                                beam_width = 100)
    # The outputs are in the first element of the tuple.
    # Additionally, the first element is actually a list,
    # therefore we take the first element of that list as well.
    #print(decoded,'decoded')
    decoded = decoded[0][0][:, :max_len]
    
    #print(decoded, decoded.shape,'decoded')

    output = []
    for d in decoded:
        # Convert the predicted indices to the corresponding chars.
        d = tf.strings.reduce_join(num_to_char(d))
        d = d.numpy().decode("utf-8")
        output.append(d)
    return output

def return_start_and_end_of_common_text_of_textline_ocr_without_common_section(textline_image):
    width = np.shape(textline_image)[1]
    height = np.shape(textline_image)[0]
    common_window = int(0.06*width)

    width1 = int ( width/2. - common_window )
    width2 = int ( width/2. + common_window )

    img_sum = np.sum(textline_image[:,:,0], axis=0)
    sum_smoothed = gaussian_filter1d(img_sum, 3)

    peaks_real, _ = find_peaks(sum_smoothed, height=0)
    if len(peaks_real)>70:

        peaks_real = peaks_real[(peaks_real<width2) & (peaks_real>width1)]

        arg_max = np.argmax(sum_smoothed[peaks_real])
        peaks_final = peaks_real[arg_max]
        return peaks_final
    else:
        return None

# Function to fit text inside the given area
def fit_text_single_line(draw, text, font_path, max_width, max_height):
    initial_font_size = 50
    font_size = initial_font_size
    while font_size > 10:  # Minimum font size
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)  # Get text bounding box
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        if text_width <= max_width and text_height <= max_height:
            return font  # Return the best-fitting font

        font_size -= 2  # Reduce font size and retry

    return ImageFont.truetype(font_path, 10)  # Smallest font fallback

def return_textlines_split_if_needed(textline_image, textline_image_bin=None):

    split_point = return_start_and_end_of_common_text_of_textline_ocr_without_common_section(textline_image)
    if split_point:
        image1 = textline_image[:, :split_point,:]# image.crop((0, 0, width2, height))
        image2 = textline_image[:, split_point:,:]#image.crop((width1, 0, width, height))
        if textline_image_bin is not None:
            image1_bin = textline_image_bin[:, :split_point,:]# image.crop((0, 0, width2, height))
            image2_bin = textline_image_bin[:, split_point:,:]#image.crop((width1, 0, width, height))
            return [image1, image2], [image1_bin, image2_bin]
        else:
            return [image1, image2], None
    else:
        return None, None

def preprocess_and_resize_image_for_ocrcnn_model(img, image_height, image_width):
    if img.shape[0]==0 or img.shape[1]==0:
        img_fin = np.ones((image_height, image_width, 3))
    else:
        ratio = image_height /float(img.shape[0])
        w_ratio = int(ratio * img.shape[1])
        
        if w_ratio <= image_width:
            width_new = w_ratio
        else:
            width_new = image_width
            
        if width_new == 0:
            width_new = img.shape[1]
            
        
        img = resize_image(img, image_height, width_new)
        img_fin = np.ones((image_height, image_width, 3))*255

        img_fin[:,:width_new,:] = img[:,:,:]
        img_fin = img_fin / 255.
    return img_fin

def get_deskewed_contour_and_bb_and_image(contour, image, deskew_angle):
    (h_in, w_in) = image.shape[:2]
    center = (w_in // 2, h_in // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, deskew_angle, 1.0)
    
    cos_angle = abs(rotation_matrix[0, 0])
    sin_angle = abs(rotation_matrix[0, 1])
    new_w = int((h_in * sin_angle) + (w_in * cos_angle))
    new_h = int((h_in * cos_angle) + (w_in * sin_angle))
    
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    deskewed_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    
    contour_points = np.array(contour, dtype=np.float32)
    transformed_points = cv2.transform(np.array([contour_points]), rotation_matrix)[0]
    
    x, y, w, h = cv2.boundingRect(np.array(transformed_points, dtype=np.int32))
    cropped_textline = deskewed_image[y:y+h, x:x+w]
    
    return cropped_textline

def rotate_image_with_padding(image, angle, border_value=(0,0,0)):
    # Get image dimensions
    (h, w) = image.shape[:2]
    
    # Calculate the center of the image
    center = (w // 2, h // 2)
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Compute the new bounding dimensions
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Perform the rotation
    try:
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), borderValue=border_value)
    except:
        rotated_image = np.copy(image)
    
    return rotated_image

def get_orientation_moments(contour):
    moments = cv2.moments(contour)
    if moments["mu20"] - moments["mu02"] == 0:  # Avoid division by zero
        return 90 if moments["mu11"] > 0 else -90
    else:
        angle = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])
        return np.degrees(angle)  # Convert radians to degrees
    
    
def get_orientation_moments_of_mask(mask):
    mask=mask.astype('uint8')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not len(contours):
        return 0
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    if moments["mu20"] - moments["mu02"] == 0:  # Avoid division by zero
        return 90 if moments["mu11"] > 0 else -90
    else:
        angle = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])
        return np.degrees(angle)  # Convert radians to degrees

def get_contours_and_bounding_boxes(mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    # Get the bounding rectangle for the contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    #bounding_boxes.append((x, y, w, h))
    
    return x, y, w, h

def return_splitting_point_of_image(image_to_spliited):
    width = np.shape(image_to_spliited)[1]
    height = np.shape(image_to_spliited)[0]
    common_window = int(0.03*width)

    width1 = int ( common_window)
    width2 = int ( width - common_window )

    img_sum = np.sum(image_to_spliited[:,:,0], axis=0)
    sum_smoothed = gaussian_filter1d(img_sum, 1)

    peaks_real, _ = find_peaks(sum_smoothed, height=0)
    peaks_real = peaks_real[(peaks_real<width2) & (peaks_real>width1)]
    
    arg_sort = np.argsort(sum_smoothed[peaks_real])
    peaks_sort_4 = peaks_real[arg_sort][::-1][:3]
    
    return np.sort(peaks_sort_4)
    
def break_curved_line_into_small_pieces_and_then_merge(img_rgb_curved, img_bin_curved, mask_curved):
    peaks_4 = return_splitting_point_of_image(img_rgb_curved)
    if len(peaks_4):
        imgs_tot = []
        for left, right in pairwise([None] + list(peaks_4) + [None]):
            img_rgb = img_rgb_curved[:, left: right]
            img_bin = img_bin_curved[:, left: right]
            mask = mask_curved[:, left: right]
            or_ma = get_orientation_moments_of_mask(mask)
            imgs_tot.append([img_rgb, img_bin, mask, or_ma])
        
        w_tot_des_list = []
        imgs_rgb_deskewed_list = []
        imgs_bin_deskewed_list = []
        
        for img_rgb_in, img_bin_in, mask_in, ori_in in imgs_tot:
            if abs(ori_in) < 45:
                img_rgb_in_des = rotate_image_with_padding(img_rgb_in, ori_in, border_value=(255,255,255) )
                img_bin_in_des = rotate_image_with_padding(img_bin_in, ori_in, border_value=(255,255,255) )
                mask_in_des = rotate_image_with_padding(mask_in, ori_in)
                # get new bounding box
                x_n, y_n, w_n, h_n = get_contours_and_bounding_boxes(mask_in_des)
                if w_n and h_n:
                    img_rgb_in_des = img_rgb_in_des[y_n: y_n + h_n, x_n: x_n + w_n]
                    img_bin_in_des = img_bin_in_des[y_n: y_n + h_n, x_n: x_n + w_n]
                else:
                    img_rgb_in_des = np.copy(img_rgb_in)
                    img_bin_in_des = np.copy(img_bin_in)
            else:
                img_rgb_in_des = np.copy(img_rgb_in)
                img_bin_in_des = np.copy(img_bin_in)

            h, w = img_rgb_in_des.shape[:2]
            new_h = 32
            new_w = 32 * w // h
            new_w = new_w or w
            img_rgb_in_des = resize_image(img_rgb_in_des, new_h, new_w)
            img_bin_in_des = resize_image(img_bin_in_des, new_h, new_w)
                
            w_tot_des_list.append(new_w)
            imgs_rgb_deskewed_list.append(img_rgb_in_des)
            imgs_bin_deskewed_list.append(img_bin_in_des)

        img_rgb_final_deskewed = np.ones((new_h, sum(w_tot_des_list), 3)) * 255
        img_bin_final_deskewed = np.ones((new_h, sum(w_tot_des_list), 3)) * 255
        
        w_indexer = 0
        for ind in range(len(w_tot_des_list)):
            w_indexer2 = w_indexer + w_tot_des_list[ind]
            img_rgb_final_deskewed[:, w_indexer: w_indexer2] = imgs_rgb_deskewed_list[ind]
            img_bin_final_deskewed[:, w_indexer: w_indexer2] = imgs_bin_deskewed_list[ind]
            w_indexer = w_indexer2
        return img_rgb_final_deskewed, img_bin_final_deskewed
    else:
        return img_rgb_curved, img_bin_curved
