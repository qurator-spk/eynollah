import math
import copy

import numpy as np
import cv2
import tensorflow as tf
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFont
from Bio import pairwise2

from .resize import resize_image


def decode_batch_predictions(pred, num_to_char, max_len = 128):
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
    
    
def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, (1, 0, 2))
    image = tf.image.flip_left_right(image)
    return image

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
    contours, _ = cv2.findContours(mask[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = max(contours, key=cv2.contourArea) if contours else None
    
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
    
def break_curved_line_into_small_pieces_and_then_merge(img_curved, mask_curved, img_bin_curved=None):
    peaks_4 = return_splitting_point_of_image(img_curved)
    if len(peaks_4)>0:
        imgs_tot = []
        
        for ind in range(len(peaks_4)+1):
            if ind==0:
                img = img_curved[:, :peaks_4[ind], :]
                if img_bin_curved is not None:
                    img_bin = img_bin_curved[:, :peaks_4[ind], :]
                mask = mask_curved[:, :peaks_4[ind], :]
            elif ind==len(peaks_4):
                img = img_curved[:, peaks_4[ind-1]:, :]
                if img_bin_curved is not None:
                    img_bin = img_bin_curved[:, peaks_4[ind-1]:, :]
                mask = mask_curved[:, peaks_4[ind-1]:, :]
            else:
                img = img_curved[:, peaks_4[ind-1]:peaks_4[ind], :]
                if img_bin_curved is not None:
                    img_bin = img_bin_curved[:, peaks_4[ind-1]:peaks_4[ind], :]
                mask = mask_curved[:, peaks_4[ind-1]:peaks_4[ind], :]
                
            or_ma = get_orientation_moments_of_mask(mask)
            
            if img_bin_curved is not None:
                imgs_tot.append([img, mask, or_ma, img_bin] )
            else:
                imgs_tot.append([img, mask, or_ma] )
        
        
        w_tot_des_list = []
        w_tot_des = 0
        imgs_deskewed_list = []
        imgs_bin_deskewed_list = []
        
        for ind in range(len(imgs_tot)):
            img_in = imgs_tot[ind][0]
            mask_in = imgs_tot[ind][1]
            ori_in = imgs_tot[ind][2]
            if img_bin_curved is not None:
                img_bin_in = imgs_tot[ind][3]
            
            if abs(ori_in)<45:
                img_in_des = rotate_image_with_padding(img_in, ori_in, border_value=(255,255,255) )
                if img_bin_curved is not None:
                    img_bin_in_des = rotate_image_with_padding(img_bin_in, ori_in, border_value=(255,255,255) )
                mask_in_des = rotate_image_with_padding(mask_in, ori_in)
                mask_in_des = mask_in_des.astype('uint8')
                
                #new bounding box
                x_n, y_n, w_n, h_n = get_contours_and_bounding_boxes(mask_in_des[:,:,0])
                
                if w_n==0 or h_n==0:
                    img_in_des = np.copy(img_in)
                    if img_bin_curved is not None:
                        img_bin_in_des = np.copy(img_bin_in)
                    w_relative = int(32 * img_in_des.shape[1]/float(img_in_des.shape[0]) )
                    if w_relative==0:
                        w_relative = img_in_des.shape[1]
                    img_in_des = resize_image(img_in_des, 32, w_relative)
                    if img_bin_curved is not None:
                        img_bin_in_des = resize_image(img_bin_in_des, 32, w_relative)
                else:
                    mask_in_des = mask_in_des[y_n:y_n+h_n, x_n:x_n+w_n, :]
                    img_in_des = img_in_des[y_n:y_n+h_n, x_n:x_n+w_n, :]
                    if img_bin_curved is not None:
                        img_bin_in_des = img_bin_in_des[y_n:y_n+h_n, x_n:x_n+w_n, :]
                    
                    w_relative = int(32 * img_in_des.shape[1]/float(img_in_des.shape[0]) )
                    if w_relative==0:
                        w_relative = img_in_des.shape[1]
                    img_in_des = resize_image(img_in_des, 32, w_relative)
                    if img_bin_curved is not None:
                        img_bin_in_des = resize_image(img_bin_in_des, 32, w_relative)
                

            else:
                img_in_des = np.copy(img_in)
                if img_bin_curved is not None:
                    img_bin_in_des = np.copy(img_bin_in)
                w_relative = int(32 * img_in_des.shape[1]/float(img_in_des.shape[0]) )
                if w_relative==0:
                    w_relative = img_in_des.shape[1]
                img_in_des = resize_image(img_in_des, 32, w_relative)
                if img_bin_curved is not None:
                    img_bin_in_des = resize_image(img_bin_in_des, 32, w_relative)
                
            w_tot_des+=img_in_des.shape[1]
            w_tot_des_list.append(img_in_des.shape[1])
            imgs_deskewed_list.append(img_in_des)
            if img_bin_curved is not None:
                imgs_bin_deskewed_list.append(img_bin_in_des)
            
            
            

        img_final_deskewed = np.zeros((32, w_tot_des, 3))+255
        if img_bin_curved is not None:
            img_bin_final_deskewed = np.zeros((32, w_tot_des, 3))+255
        else:
            img_bin_final_deskewed = None
        
        w_indexer = 0
        for ind in range(len(w_tot_des_list)):
            img_final_deskewed[:,w_indexer:w_indexer+w_tot_des_list[ind],:] = imgs_deskewed_list[ind][:,:,:]
            if img_bin_curved is not None:
                img_bin_final_deskewed[:,w_indexer:w_indexer+w_tot_des_list[ind],:] = imgs_bin_deskewed_list[ind][:,:,:]
            w_indexer = w_indexer+w_tot_des_list[ind]
        return img_final_deskewed, img_bin_final_deskewed
    else:
        return img_curved, img_bin_curved
    
def return_textline_contour_with_added_box_coordinate(textline_contour,  box_ind):
    textline_contour[:,0] = textline_contour[:,0] + box_ind[2]
    textline_contour[:,1] = textline_contour[:,1] + box_ind[0]
    return textline_contour


def return_rnn_cnn_ocr_of_given_textlines(image,
                                          all_found_textline_polygons,
                                          all_box_coord,
                                          prediction_model,
                                          b_s_ocr, num_to_char,
                                          curved_line=False):
    max_len = 512
    padding_token = 299
    image_width = 512#max_len * 4
    image_height = 32
    ind_tot = 0
    #cv2.imwrite('./img_out.png', image_page)
    ocr_all_textlines = []
    cropped_lines_region_indexer = []
    cropped_lines_meging_indexing = []
    cropped_lines = []
    indexer_text_region = 0
    
    for indexing, ind_poly_first in enumerate(all_found_textline_polygons):
        #ocr_textline_in_textregion = []
        if len(ind_poly_first)==0:
            cropped_lines_region_indexer.append(indexer_text_region)
            cropped_lines_meging_indexing.append(0)
            img_fin = np.ones((image_height, image_width, 3))*1
            cropped_lines.append(img_fin)

        else:
            for indexing2, ind_poly in enumerate(ind_poly_first):
                cropped_lines_region_indexer.append(indexer_text_region)
                if not curved_line:
                    ind_poly = copy.deepcopy(ind_poly)
                    box_ind = all_box_coord[indexing]

                    ind_poly = return_textline_contour_with_added_box_coordinate(ind_poly, box_ind)
                    #print(ind_poly_copy)
                    ind_poly[ind_poly<0] = 0
                x, y, w, h = cv2.boundingRect(ind_poly)
                
                w_scaled = w *  image_height/float(h)

                mask_poly = np.zeros(image.shape)

                img_poly_on_img = np.copy(image)
                
                mask_poly = cv2.fillPoly(mask_poly, pts=[ind_poly], color=(1, 1, 1))


                
                mask_poly = mask_poly[y:y+h, x:x+w, :]
                img_crop = img_poly_on_img[y:y+h, x:x+w, :]
                
                img_crop[mask_poly==0] = 255
                
                if w_scaled < 640:#1.5*image_width:
                    img_fin = preprocess_and_resize_image_for_ocrcnn_model(img_crop, image_height, image_width)
                    cropped_lines.append(img_fin)
                    cropped_lines_meging_indexing.append(0)
                else:
                    splited_images, splited_images_bin = return_textlines_split_if_needed(img_crop, None)
                    
                    if splited_images:
                        img_fin = preprocess_and_resize_image_for_ocrcnn_model(splited_images[0],
                                                                               image_height,
                                                                               image_width)
                        cropped_lines.append(img_fin)
                        cropped_lines_meging_indexing.append(1)
                        
                        img_fin = preprocess_and_resize_image_for_ocrcnn_model(splited_images[1],
                                                                               image_height,
                                                                               image_width)
                        
                        cropped_lines.append(img_fin)
                        cropped_lines_meging_indexing.append(-1)
                        
                    else:
                        img_fin = preprocess_and_resize_image_for_ocrcnn_model(img_crop,
                                                                               image_height,
                                                                               image_width)
                        cropped_lines.append(img_fin)
                        cropped_lines_meging_indexing.append(0)
            
        indexer_text_region+=1
        
    extracted_texts = []

    n_iterations  = math.ceil(len(cropped_lines) / b_s_ocr) 

    for i in range(n_iterations):
        if i==(n_iterations-1):
            n_start = i*b_s_ocr
            imgs = cropped_lines[n_start:]
            imgs = np.array(imgs)
            imgs = imgs.reshape(imgs.shape[0], image_height, image_width, 3)
            
            
        else:
            n_start = i*b_s_ocr
            n_end = (i+1)*b_s_ocr
            imgs = cropped_lines[n_start:n_end]
            imgs = np.array(imgs).reshape(b_s_ocr, image_height, image_width, 3)
            

        preds = prediction_model.predict(imgs, verbose=0)
        
        pred_texts = decode_batch_predictions(preds, num_to_char)

        for ib in range(imgs.shape[0]):
            pred_texts_ib = pred_texts[ib].replace("[UNK]", "")
            extracted_texts.append(pred_texts_ib)
            
    extracted_texts_merged = [extracted_texts[ind]
                              if cropped_lines_meging_indexing[ind]==0
                              else extracted_texts[ind]+" "+extracted_texts[ind+1]
                              if cropped_lines_meging_indexing[ind]==1
                              else None
                              for ind in range(len(cropped_lines_meging_indexing))]

    extracted_texts_merged = [ind for ind in extracted_texts_merged if ind is not None]
    unique_cropped_lines_region_indexer = np.unique(cropped_lines_region_indexer)
    
    ocr_all_textlines = []
    for ind in unique_cropped_lines_region_indexer:
        ocr_textline_in_textregion = []
        extracted_texts_merged_un = np.array(extracted_texts_merged)[np.array(cropped_lines_region_indexer)==ind]
        for  it_ind, text_textline in enumerate(extracted_texts_merged_un):
            ocr_textline_in_textregion.append(text_textline)
        ocr_all_textlines.append(ocr_textline_in_textregion)
    return ocr_all_textlines

def biopython_align(str1, str2):
    alignments = pairwise2.align.globalms(str1, str2, 2, -1, -2, -2)
    best_alignment = alignments[0]  # Get the best alignment
    return best_alignment.seqA, best_alignment.seqB
