import os
from logging import getLogger
from functools import partial
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from .rotate import rotate_image
from .resize import resize_image
from .contour import (
    return_parent_contours,
    filter_contours_area_of_image_tables,
    return_contours_of_image,
    filter_contours_area_of_image,
    return_contours_of_interested_textline,
    find_contours_mean_y_diff,
)
from .shm import share_ndarray, wrap_ndarray_shared
from . import (
    find_num_col_deskew,
    box2rect,
)

def dedup_separate_lines(img_patch, contour_text_interest, thetha, axis):
    (h, w) = img_patch.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
    x_d = M[0, 2]
    y_d = M[1, 2]

    thetha = thetha / 180.0 * np.pi
    rotation_matrix = np.array([[np.cos(thetha), -np.sin(thetha)], [np.sin(thetha), np.cos(thetha)]])

    x_cont = contour_text_interest[:, 0, 0]
    y_cont = contour_text_interest[:, 0, 1]
    x_cont = x_cont - np.min(x_cont)
    y_cont = y_cont - np.min(y_cont)

    x_min_cont = 0
    x_max_cont = img_patch.shape[1]
    y_min_cont = 0
    y_max_cont = img_patch.shape[0]

    xv = np.linspace(x_min_cont, x_max_cont, 1000)
    textline_patch_sum_along_width = img_patch.sum(axis=axis)
    first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

    y = textline_patch_sum_along_width[:]  # [first_nonzero:last_nonzero]
    y_padded = np.zeros(len(y) + 40)
    y_padded[20 : len(y) + 20] = y
    x = np.array(range(len(y)))

    peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
    if 1 > 0:
        try:
            y_padded_smoothed_e = gaussian_filter1d(y_padded, 2)
            y_padded_up_to_down_e = -y_padded + np.max(y_padded)
            y_padded_up_to_down_padded_e = np.zeros(len(y_padded_up_to_down_e) + 40)
            y_padded_up_to_down_padded_e[20 : len(y_padded_up_to_down_e) + 20] = y_padded_up_to_down_e
            y_padded_up_to_down_padded_e = gaussian_filter1d(y_padded_up_to_down_padded_e, 2)

            peaks_e, _ = find_peaks(y_padded_smoothed_e, height=0)
            peaks_neg_e, _ = find_peaks(y_padded_up_to_down_padded_e, height=0)
            neg_peaks_max = np.max(y_padded_up_to_down_padded_e[peaks_neg_e])

            arg_neg_must_be_deleted = np.arange(len(peaks_neg_e))[
                y_padded_up_to_down_padded_e[peaks_neg_e] / float(neg_peaks_max) < 0.3]
            diff_arg_neg_must_be_deleted = np.diff(arg_neg_must_be_deleted)

            arg_diff = np.array(range(len(diff_arg_neg_must_be_deleted)))
            arg_diff_cluster = arg_diff[diff_arg_neg_must_be_deleted > 1]

            peaks_new = peaks_e[:]
            peaks_neg_new = peaks_neg_e[:]

            clusters_to_be_deleted = []
            if len(arg_diff_cluster) > 0:
                clusters_to_be_deleted.append(
                    arg_neg_must_be_deleted[0 : arg_diff_cluster[0] + 1])
                for i in range(len(arg_diff_cluster) - 1):
                    clusters_to_be_deleted.append(
                        arg_neg_must_be_deleted[arg_diff_cluster[i] + 1 :
                                                arg_diff_cluster[i + 1] + 1])
                clusters_to_be_deleted.append(
                    arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster) - 1] + 1 :])
            if len(clusters_to_be_deleted) > 0:
                peaks_new_extra = []
                for m in range(len(clusters_to_be_deleted)):
                    min_cluster = np.min(peaks_e[clusters_to_be_deleted[m]])
                    max_cluster = np.max(peaks_e[clusters_to_be_deleted[m]])
                    peaks_new_extra.append(int((min_cluster + max_cluster) / 2.0))
                    for m1 in range(len(clusters_to_be_deleted[m])):
                        peaks_new = peaks_new[peaks_new != peaks_e[clusters_to_be_deleted[m][m1] - 1]]
                        peaks_new = peaks_new[peaks_new != peaks_e[clusters_to_be_deleted[m][m1]]]
                        peaks_neg_new = peaks_neg_new[peaks_neg_new != peaks_neg_e[clusters_to_be_deleted[m][m1]]]
                peaks_new_tot = []
                for i1 in peaks_new:
                    peaks_new_tot.append(i1)
                for i1 in peaks_new_extra:
                    peaks_new_tot.append(i1)
                peaks_new_tot = np.sort(peaks_new_tot)

            else:
                peaks_new_tot = peaks_e[:]

            textline_con, hierarchy = return_contours_of_image(img_patch)
            textline_con_fil = filter_contours_area_of_image(img_patch,
                                                             textline_con, hierarchy,
                                                             max_area=1, min_area=0.0008)
            if len(np.diff(peaks_new_tot))>1:
                y_diff_mean = np.mean(np.diff(peaks_new_tot))  # self.find_contours_mean_y_diff(textline_con_fil)
                sigma_gaus = int(y_diff_mean * (7.0 / 40.0))
            else:
                sigma_gaus = 12
        except:
            sigma_gaus = 12
        if sigma_gaus < 3:
            sigma_gaus = 3

    y_padded_smoothed = gaussian_filter1d(y_padded, sigma_gaus)
    y_padded_up_to_down = -y_padded + np.max(y_padded)
    y_padded_up_to_down_padded = np.zeros(len(y_padded_up_to_down) + 40)
    y_padded_up_to_down_padded[20 : len(y_padded_up_to_down) + 20] = y_padded_up_to_down
    y_padded_up_to_down_padded = gaussian_filter1d(y_padded_up_to_down_padded, sigma_gaus)

    peaks, _ = find_peaks(y_padded_smoothed, height=0)
    peaks_neg, _ = find_peaks(y_padded_up_to_down_padded, height=0)

    return (x, y,
            x_d, y_d,
            xv,
            x_min_cont, y_min_cont,
            x_max_cont, y_max_cont,
            first_nonzero,
            y_padded_up_to_down_padded,
            y_padded_smoothed,
            peaks, peaks_neg,
            rotation_matrix)

def separate_lines(img_patch, contour_text_interest, thetha, x_help, y_help):
    h, w = img_patch.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
    x_d = M[0, 2]
    y_d = M[1, 2]
    rotation_matrix = M[:2, :2]
    contour_text_interest_copy = contour_text_interest.copy()

    x_cont = contour_text_interest[:, 0, 0]
    y_cont = contour_text_interest[:, 0, 1]
    x_cont = x_cont - np.min(x_cont)
    y_cont = y_cont - np.min(y_cont)

    x_min_cont = 0
    x_max_cont = img_patch.shape[1]
    y_min_cont = 0
    y_max_cont = img_patch.shape[0]

    xv = np.linspace(x_min_cont, x_max_cont, 1000)
    textline_patch_sum_along_width = img_patch.sum(axis=1)
    first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

    y = textline_patch_sum_along_width[:]  # [first_nonzero:last_nonzero]
    y_padded = np.zeros(len(y) + 40)
    y_padded[20:len(y) + 20] = y
    x = np.array(range(len(y)))

    peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
    
    try:
        y_padded_smoothed_e= gaussian_filter1d(y_padded, 2)
        y_padded_up_to_down_e=-y_padded+np.max(y_padded)
        y_padded_up_to_down_padded_e=np.zeros(len(y_padded_up_to_down_e)+40)
        y_padded_up_to_down_padded_e[20:len(y_padded_up_to_down_e)+20]=y_padded_up_to_down_e
        y_padded_up_to_down_padded_e= gaussian_filter1d(y_padded_up_to_down_padded_e, 2)
        
        peaks_e, _ = find_peaks(y_padded_smoothed_e, height=0)
        peaks_neg_e, _ = find_peaks(y_padded_up_to_down_padded_e, height=0)
        neg_peaks_max=np.max(y_padded_up_to_down_padded_e[peaks_neg_e])

        arg_neg_must_be_deleted = np.arange(len(peaks_neg_e))[
            y_padded_up_to_down_padded_e[peaks_neg_e]/float(neg_peaks_max)<0.3]
        diff_arg_neg_must_be_deleted=np.diff(arg_neg_must_be_deleted)
        
        arg_diff=np.array(range(len(diff_arg_neg_must_be_deleted)))
        arg_diff_cluster=arg_diff[diff_arg_neg_must_be_deleted>1]
        peaks_new=peaks_e[:]
        peaks_neg_new=peaks_neg_e[:]

        clusters_to_be_deleted=[]
        if len(arg_diff_cluster)>0:
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[0:arg_diff_cluster[0]+1])
            for i in range(len(arg_diff_cluster)-1):
                clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[i]+1:
                                                                        arg_diff_cluster[i+1]+1])
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster)-1]+1:])
        if len(clusters_to_be_deleted)>0:
            peaks_new_extra=[]
            for m in range(len(clusters_to_be_deleted)):
                min_cluster=np.min(peaks_e[clusters_to_be_deleted[m]])
                max_cluster=np.max(peaks_e[clusters_to_be_deleted[m]])
                peaks_new_extra.append( int( (min_cluster+max_cluster)/2.0) )
                for m1 in range(len(clusters_to_be_deleted[m])):
                    peaks_new=peaks_new[peaks_new!=peaks_e[clusters_to_be_deleted[m][m1]-1]]
                    peaks_new=peaks_new[peaks_new!=peaks_e[clusters_to_be_deleted[m][m1]]]
                    peaks_neg_new=peaks_neg_new[peaks_neg_new!=peaks_neg_e[clusters_to_be_deleted[m][m1]]]
            peaks_new_tot=[]
            for i1 in peaks_new:
                peaks_new_tot.append(i1)
            for i1 in peaks_new_extra:
                peaks_new_tot.append(i1)
            peaks_new_tot=np.sort(peaks_new_tot)
        else:
            peaks_new_tot=peaks_e[:]

        textline_con,hierarchy=return_contours_of_image(img_patch)
        textline_con_fil=filter_contours_area_of_image(img_patch,
                                                        textline_con, hierarchy,
                                                        max_area=1, min_area=0.0008)

        if len(np.diff(peaks_new_tot))>0:
            y_diff_mean=np.mean(np.diff(peaks_new_tot))#self.find_contours_mean_y_diff(textline_con_fil)
            sigma_gaus=int(  y_diff_mean * (7./40.0) )
        else:
            sigma_gaus=12
            
    except:
        sigma_gaus=12
    if sigma_gaus<3:
        sigma_gaus=3

    y_padded_smoothed= gaussian_filter1d(y_padded, sigma_gaus)
    y_padded_up_to_down=-y_padded+np.max(y_padded)
    y_padded_up_to_down_padded=np.zeros(len(y_padded_up_to_down)+40)
    y_padded_up_to_down_padded[20:len(y_padded_up_to_down)+20]=y_padded_up_to_down
    y_padded_up_to_down_padded= gaussian_filter1d(y_padded_up_to_down_padded, sigma_gaus)
    peaks, _ = find_peaks(y_padded_smoothed, height=0)
    peaks_neg, _ = find_peaks(y_padded_up_to_down_padded, height=0)
        
    try:
        neg_peaks_max=np.max(y_padded_smoothed[peaks])
        arg_neg_must_be_deleted = np.arange(len(peaks_neg))[
            y_padded_up_to_down_padded[peaks_neg]/float(neg_peaks_max)<0.42]
        diff_arg_neg_must_be_deleted=np.diff(arg_neg_must_be_deleted)
        
        arg_diff=np.array(range(len(diff_arg_neg_must_be_deleted)))
        arg_diff_cluster=arg_diff[diff_arg_neg_must_be_deleted>1]
        
    except:
        arg_neg_must_be_deleted=[]
        arg_diff_cluster=[]
    try:
        peaks_new=peaks[:]
        peaks_neg_new=peaks_neg[:]
        clusters_to_be_deleted=[]
        if len(arg_diff_cluster)>=2 and len(arg_diff_cluster)>0:
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[0:arg_diff_cluster[0]+1])
            for i in range(len(arg_diff_cluster)-1):
                clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[i]+1:
                                                                      arg_diff_cluster[i+1]+1])
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster)-1]+1:])
        elif len(arg_neg_must_be_deleted)>=2 and len(arg_diff_cluster)==0:
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[:])
    
        if  len(arg_neg_must_be_deleted)==1:
            clusters_to_be_deleted.append(arg_neg_must_be_deleted)
        if len(clusters_to_be_deleted)>0:
            peaks_new_extra=[]
            for m in range(len(clusters_to_be_deleted)):
                min_cluster=np.min(peaks[clusters_to_be_deleted[m]])
                max_cluster=np.max(peaks[clusters_to_be_deleted[m]])
                peaks_new_extra.append( int( (min_cluster+max_cluster)/2.0) )
                for m1 in range(len(clusters_to_be_deleted[m])):
                    peaks_new=peaks_new[peaks_new!=peaks[clusters_to_be_deleted[m][m1]-1]]
                    peaks_new=peaks_new[peaks_new!=peaks[clusters_to_be_deleted[m][m1]]]
                    peaks_neg_new=peaks_neg_new[peaks_neg_new!=peaks_neg[clusters_to_be_deleted[m][m1]]]
            peaks_new_tot=[]
            for i1 in peaks_new:
                peaks_new_tot.append(i1)
            for i1 in peaks_new_extra:
                peaks_new_tot.append(i1)
            peaks_new_tot=np.sort(peaks_new_tot)
            
            peaks=peaks_new_tot[:]
            peaks_neg=peaks_neg_new[:]
        else:
            peaks_new_tot=peaks[:]
            peaks=peaks_new_tot[:]
            peaks_neg=peaks_neg_new[:]
    except:
        pass
    if len(y_padded_smoothed[peaks]) > 1:
        mean_value_of_peaks=np.mean(y_padded_smoothed[peaks])
        std_value_of_peaks=np.std(y_padded_smoothed[peaks])
    else:
        mean_value_of_peaks = np.nan
        std_value_of_peaks = np.nan
    peaks_values=y_padded_smoothed[peaks]
    peaks_neg = peaks_neg - 20 - 20
    peaks = peaks - 20
    for jj in range(len(peaks_neg)):
        if peaks_neg[jj] > len(x) - 1:
            peaks_neg[jj] = len(x) - 1
    for jj in range(len(peaks)):
        if peaks[jj] > len(x) - 1:
            peaks[jj] = len(x) - 1

    textline_boxes = []
    textline_boxes_rot = []
    
    if len(peaks_neg) == len(peaks) + 1 and len(peaks) >= 3:
        for jj in range(len(peaks)):
            
            if jj==(len(peaks)-1):
                dis_to_next_up = abs(peaks[jj] - peaks_neg[jj])
                dis_to_next_down = abs(peaks[jj] - peaks_neg[jj + 1])
                
                if peaks_values[jj]>mean_value_of_peaks-std_value_of_peaks/2.:
                    point_up = peaks[jj] + first_nonzero - int(1.3 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                    point_down =y_max_cont-1
                    ##peaks[jj] + first_nonzero + int(1.3 * dis_to_next_down)
                    #point_up
                    # np.max(y_cont)#peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)
                    ###-int(dis_to_next_down*1./4.0)
                else:
                    point_up = peaks[jj] + first_nonzero - int(1.4 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                    point_down =y_max_cont-1
                    ##peaks[jj] + first_nonzero + int(1.6 * dis_to_next_down)
                    #point_up
                    # np.max(y_cont)#peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)
                    ###-int(dis_to_next_down*1./4.0)

                point_down_narrow = peaks[jj] + first_nonzero + int(
                    1.4 * dis_to_next_down)
                ###-int(dis_to_next_down*1./2)
            else:
                dis_to_next_up = abs(peaks[jj] - peaks_neg[jj])
                dis_to_next_down = abs(peaks[jj] - peaks_neg[jj + 1])
                
                if peaks_values[jj]>mean_value_of_peaks-std_value_of_peaks/2.:
                    point_up = peaks[jj] + first_nonzero - int(1.1 * dis_to_next_up)
                    ##+int(dis_to_next_up*1./4.0)
                    point_down = peaks[jj] + first_nonzero + int(1.1 * dis_to_next_down)
                    ###-int(dis_to_next_down*1./4.0)
                else:
                    point_up = peaks[jj] + first_nonzero - int(1.23 * dis_to_next_up)
                    ##+int(dis_to_next_up*1./4.0)
                    point_down = peaks[jj] + first_nonzero + int(1.33 * dis_to_next_down)
                    ###-int(dis_to_next_down*1./4.0)

                point_down_narrow = peaks[jj] + first_nonzero + int(
                    1.1 * dis_to_next_down)  ###-int(dis_to_next_down*1./2)

            if point_down_narrow >= img_patch.shape[0]:
                point_down_narrow = img_patch.shape[0] - 2
            

            distances = [cv2.pointPolygonTest(contour_text_interest_copy,
                                              tuple(int(x) for x in np.array([xv[mj], peaks[jj] + first_nonzero])),
                                              True)
                            for mj in range(len(xv))]
            distances = np.array(distances)

            xvinside = xv[distances >= 0]

            if len(xvinside) == 0:
                x_min = x_min_cont
                x_max = x_max_cont
            else:
                x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
                x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)

            p1 = np.dot(rotation_matrix, [int(x_min), int(point_up)])
            p2 = np.dot(rotation_matrix, [int(x_max), int(point_up)])
            p3 = np.dot(rotation_matrix, [int(x_max), int(point_down)])
            p4 = np.dot(rotation_matrix, [int(x_min), int(point_down)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d
            
            if x_min_rot1<0:
                x_min_rot1=0
            if x_min_rot4<0:
                x_min_rot4=0
            if point_up_rot1<0:
                point_up_rot1=0
            if point_up_rot2<0:
                point_up_rot2=0

            x_min_rot1=x_min_rot1-x_help
            x_max_rot2=x_max_rot2-x_help
            x_max_rot3=x_max_rot3-x_help
            x_min_rot4=x_min_rot4-x_help
            
            point_up_rot1=point_up_rot1-y_help
            point_up_rot2=point_up_rot2-y_help
            point_down_rot3=point_down_rot3-y_help
            point_down_rot4=point_down_rot4-y_help

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                [int(x_max_rot2), int(point_up_rot2)],
                                                [int(x_max_rot3), int(point_down_rot3)],
                                                [int(x_min_rot4), int(point_down_rot4)]]))
            textline_boxes.append(np.array([[int(x_min), int(point_up)],
                                            [int(x_max), int(point_up)],
                                            [int(x_max), int(point_down)],
                                            [int(x_min), int(point_down)]]))
    elif len(peaks) < 1:
        pass

    elif len(peaks) == 1:
        distances = [cv2.pointPolygonTest(contour_text_interest_copy,
                                          tuple(int(x) for x in np.array([xv[mj], peaks[0] + first_nonzero])), True)
                     for mj in range(len(xv))]
        distances = np.array(distances)

        xvinside = xv[distances >= 0]
        if len(xvinside) == 0:
            x_min = x_min_cont
            x_max = x_max_cont
        else:
            x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
            x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)
        #x_min = x_min_cont
        #x_max = x_max_cont

        y_min = y_min_cont
        y_max = y_max_cont

        p1 = np.dot(rotation_matrix, [int(x_min), int(y_min)])
        p2 = np.dot(rotation_matrix, [int(x_max), int(y_min)])
        p3 = np.dot(rotation_matrix, [int(x_max), int(y_max)])
        p4 = np.dot(rotation_matrix, [int(x_min), int(y_max)])

        x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
        x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
        x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
        x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d
        
        if x_min_rot1<0:
            x_min_rot1=0
        if x_min_rot4<0:
            x_min_rot4=0
        if point_up_rot1<0:
            point_up_rot1=0
        if point_up_rot2<0:
            point_up_rot2=0
        
        x_min_rot1=x_min_rot1-x_help
        x_max_rot2=x_max_rot2-x_help
        x_max_rot3=x_max_rot3-x_help
        x_min_rot4=x_min_rot4-x_help
        
        point_up_rot1=point_up_rot1-y_help
        point_up_rot2=point_up_rot2-y_help
        point_down_rot3=point_down_rot3-y_help
        point_down_rot4=point_down_rot4-y_help

        textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                            [int(x_max_rot2), int(point_up_rot2)],
                                            [int(x_max_rot3), int(point_down_rot3)],
                                            [int(x_min_rot4), int(point_down_rot4)]]))
        textline_boxes.append(np.array([[int(x_min), int(y_min)],
                                        [int(x_max), int(y_min)],
                                        [int(x_max), int(y_max)],
                                        [int(x_min), int(y_max)]]))
    elif len(peaks) == 2:
        dis_to_next = np.abs(peaks[1] - peaks[0])
        for jj in range(len(peaks)):
            if jj == 0:
                point_up = 0#peaks[jj] + first_nonzero - int(1. / 1.7 * dis_to_next)
                if point_up < 0:
                    point_up = 1
                point_down = peaks_neg[1] + first_nonzero# peaks[jj] + first_nonzero + int(1. / 1.8 * dis_to_next)
            elif jj == 1:
                point_down =peaks_neg[1] + first_nonzero# peaks[jj] + first_nonzero + int(1. / 1.8 * dis_to_next)
                if point_down >= img_patch.shape[0]:
                    point_down = img_patch.shape[0] - 2
                try:
                    point_up = peaks_neg[2] + first_nonzero#peaks[jj] + first_nonzero - int(1. / 1.8 * dis_to_next)
                except:
                    point_up =peaks[jj] + first_nonzero - int(1. / 1.8 * dis_to_next)
                    
            distances = [cv2.pointPolygonTest(contour_text_interest_copy,
                                              tuple(int(x) for x in np.array([xv[mj], peaks[jj] + first_nonzero])),
                                              True)
                         for mj in range(len(xv))]
            distances = np.array(distances)

            xvinside = xv[distances >= 0]
            if len(xvinside) == 0:
                x_min = x_min_cont
                x_max = x_max_cont
            else:
                x_min = np.min(xvinside)
                x_max = np.max(xvinside)

            p1 = np.dot(rotation_matrix, [int(x_min), int(point_up)])
            p2 = np.dot(rotation_matrix, [int(x_max), int(point_up)])
            p3 = np.dot(rotation_matrix, [int(x_max), int(point_down)])
            p4 = np.dot(rotation_matrix, [int(x_min), int(point_down)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d
            
            if x_min_rot1<0:
                x_min_rot1=0
            if x_min_rot4<0:
                x_min_rot4=0
            if point_up_rot1<0:
                point_up_rot1=0
            if point_up_rot2<0:
                point_up_rot2=0                   
                
            x_min_rot1=x_min_rot1-x_help
            x_max_rot2=x_max_rot2-x_help
            x_max_rot3=x_max_rot3-x_help
            x_min_rot4=x_min_rot4-x_help
            
            point_up_rot1=point_up_rot1-y_help
            point_up_rot2=point_up_rot2-y_help
            point_down_rot3=point_down_rot3-y_help
            point_down_rot4=point_down_rot4-y_help

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                [int(x_max_rot2), int(point_up_rot2)],
                                                [int(x_max_rot3), int(point_down_rot3)],
                                                [int(x_min_rot4), int(point_down_rot4)]]))
            textline_boxes.append(np.array([[int(x_min), int(point_up)],
                                            [int(x_max), int(point_up)],
                                            [int(x_max), int(point_down)],
                                            [int(x_min), int(point_down)]]))
    else:
        for jj in range(len(peaks)):
            if jj == 0:
                dis_to_next = peaks[jj + 1] - peaks[jj]
                # point_up=peaks[jj]+first_nonzero-int(1./3*dis_to_next)
                point_up = peaks[jj] + first_nonzero - int(1. / 1.9 * dis_to_next)
                if point_up < 0:
                    point_up = 1
                # point_down=peaks[jj]+first_nonzero+int(1./3*dis_to_next)
                point_down = peaks[jj] + first_nonzero + int(1. / 1.9 * dis_to_next)
            elif jj == len(peaks) - 1:
                dis_to_next = peaks[jj] - peaks[jj - 1]
                # point_down=peaks[jj]+first_nonzero+int(1./3*dis_to_next)
                point_down = peaks[jj] + first_nonzero + int(1. / 1.7 * dis_to_next)
                if point_down >= img_patch.shape[0]:
                    point_down = img_patch.shape[0] - 2
                # point_up=peaks[jj]+first_nonzero-int(1./3*dis_to_next)
                point_up = peaks[jj] + first_nonzero - int(1. / 1.9 * dis_to_next)
            else:
                dis_to_next_down = peaks[jj + 1] - peaks[jj]
                dis_to_next_up = peaks[jj] - peaks[jj - 1]

                point_up = peaks[jj] + first_nonzero - int(1. / 1.9 * dis_to_next_up)
                point_down = peaks[jj] + first_nonzero + int(1. / 1.9 * dis_to_next_down)
                
            distances = [cv2.pointPolygonTest(contour_text_interest_copy,
                                              tuple(int(x) for x in np.array([xv[mj], peaks[jj] + first_nonzero])),
                                              True)
                         for mj in range(len(xv))]
            distances = np.array(distances)

            xvinside = xv[distances >= 0]
            if len(xvinside) == 0:
                x_min = x_min_cont
                x_max = x_max_cont
            else:
                x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
                x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)

            p1 = np.dot(rotation_matrix, [int(x_min), int(point_up)])
            p2 = np.dot(rotation_matrix, [int(x_max), int(point_up)])
            p3 = np.dot(rotation_matrix, [int(x_max), int(point_down)])
            p4 = np.dot(rotation_matrix, [int(x_min), int(point_down)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d
            
            if x_min_rot1<0:
                x_min_rot1=0
            if x_min_rot4<0:
                x_min_rot4=0
            if point_up_rot1<0:
                point_up_rot1=0
            if point_up_rot2<0:
                point_up_rot2=0                
                
            x_min_rot1=x_min_rot1-x_help
            x_max_rot2=x_max_rot2-x_help
            x_max_rot3=x_max_rot3-x_help
            x_min_rot4=x_min_rot4-x_help
            
            point_up_rot1=point_up_rot1-y_help
            point_up_rot2=point_up_rot2-y_help
            point_down_rot3=point_down_rot3-y_help
            point_down_rot4=point_down_rot4-y_help

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                [int(x_max_rot2), int(point_up_rot2)],
                                                [int(x_max_rot3), int(point_down_rot3)],
                                                [int(x_min_rot4), int(point_down_rot4)]]))
            textline_boxes.append(np.array([[int(x_min), int(point_up)],
                                            [int(x_max), int(point_up)],
                                            [int(x_max), int(point_down)],
                                            [int(x_min), int(point_down)]]))
    return peaks, textline_boxes_rot

def separate_lines_vertical(img_patch, contour_text_interest, thetha):
    thetha = thetha + 90
    contour_text_interest_copy = contour_text_interest.copy()
    x, y, x_d, y_d, xv, x_min_cont, y_min_cont, x_max_cont, y_max_cont, \
        first_nonzero, y_padded_up_to_down_padded, y_padded_smoothed, \
        peaks, peaks_neg, rotation_matrix = dedup_separate_lines(img_patch, contour_text_interest, thetha, 0)

    # plt.plot(y_padded_up_to_down_padded)
    # plt.plot(peaks_neg,y_padded_up_to_down_padded[peaks_neg],'*')
    # plt.title('negs')
    # plt.show()

    # plt.plot(y_padded_smoothed)
    # plt.plot(peaks,y_padded_smoothed[peaks],'*')
    # plt.title('poss')
    # plt.show()

    neg_peaks_max = np.max(y_padded_up_to_down_padded[peaks_neg])

    arg_neg_must_be_deleted = np.arange(len(peaks_neg))[
        y_padded_up_to_down_padded[peaks_neg] / float(neg_peaks_max) < 0.42]
    diff_arg_neg_must_be_deleted = np.diff(arg_neg_must_be_deleted)

    arg_diff = np.array(range(len(diff_arg_neg_must_be_deleted)))
    arg_diff_cluster = arg_diff[diff_arg_neg_must_be_deleted > 1]

    peaks_new = peaks[:]
    peaks_neg_new = peaks_neg[:]
    clusters_to_be_deleted = []

    if len(arg_neg_must_be_deleted) >= 2 and len(arg_diff_cluster) >= 2:
        clusters_to_be_deleted.append(arg_neg_must_be_deleted[0 : arg_diff_cluster[0] + 1])
        for i in range(len(arg_diff_cluster) - 1):
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[i] + 1 :
                                                                  arg_diff_cluster[i + 1] + 1])
        clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster) - 1] + 1 :])
    elif len(arg_neg_must_be_deleted) >= 2 and len(arg_diff_cluster) == 0:
        clusters_to_be_deleted.append(arg_neg_must_be_deleted[:])
    else:
        clusters_to_be_deleted.append(arg_neg_must_be_deleted)
    if len(clusters_to_be_deleted) > 0:
        peaks_new_extra = []
        for m in range(len(clusters_to_be_deleted)):
            min_cluster = np.min(peaks[clusters_to_be_deleted[m]])
            max_cluster = np.max(peaks[clusters_to_be_deleted[m]])
            peaks_new_extra.append(int((min_cluster + max_cluster) / 2.0))
            for m1 in range(len(clusters_to_be_deleted[m])):
                peaks_new = peaks_new[peaks_new != peaks[clusters_to_be_deleted[m][m1] - 1]]
                peaks_new = peaks_new[peaks_new != peaks[clusters_to_be_deleted[m][m1]]]
                peaks_neg_new = peaks_neg_new[peaks_neg_new != peaks_neg[clusters_to_be_deleted[m][m1]]]
        peaks_new_tot = []
        for i1 in peaks_new:
            peaks_new_tot.append(i1)
        for i1 in peaks_new_extra:
            peaks_new_tot.append(i1)
        peaks_new_tot = np.sort(peaks_new_tot)

        peaks = peaks_new_tot[:]
        peaks_neg = peaks_neg_new[:]

    else:
        peaks_new_tot = peaks[:]
        peaks = peaks_new_tot[:]
        peaks_neg = peaks_neg_new[:]
    
    if len(y_padded_smoothed[peaks])>1:
        mean_value_of_peaks = np.mean(y_padded_smoothed[peaks])
        std_value_of_peaks = np.std(y_padded_smoothed[peaks])
    else:
        mean_value_of_peaks = np.nan
        std_value_of_peaks = np.nan
        
    peaks_values = y_padded_smoothed[peaks]

    peaks_neg = peaks_neg - 20 - 20
    peaks = peaks - 20

    for jj in range(len(peaks_neg)):
        if peaks_neg[jj] > len(x) - 1:
            peaks_neg[jj] = len(x) - 1

    for jj in range(len(peaks)):
        if peaks[jj] > len(x) - 1:
            peaks[jj] = len(x) - 1

    textline_boxes = []
    textline_boxes_rot = []

    if len(peaks_neg) == len(peaks) + 1 and len(peaks) >= 3:
        for jj in range(len(peaks)):

            if jj == (len(peaks) - 1):
                dis_to_next_up = abs(peaks[jj] - peaks_neg[jj])
                dis_to_next_down = abs(peaks[jj] - peaks_neg[jj + 1])

                if peaks_values[jj] > mean_value_of_peaks - std_value_of_peaks / 2.0:
                    point_up = peaks[jj] + first_nonzero - int(1.3 * dis_to_next_up)
                    ##+int(dis_to_next_up*1./4.0)
                    point_down = x_max_cont - 1
                    ##peaks[jj] + first_nonzero + int(1.3 * dis_to_next_down)
                    #point_up
                    # np.max(y_cont)#peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)
                    ###-int(dis_to_next_down*1./4.0)
                else:
                    point_up = peaks[jj] + first_nonzero - int(1.4 * dis_to_next_up)
                    ##+int(dis_to_next_up*1./4.0)
                    point_down = x_max_cont - 1
                    ##peaks[jj] + first_nonzero + int(1.6 * dis_to_next_down)
                    #point_up
                    # np.max(y_cont)
                    #peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)
                    ###-int(dis_to_next_down*1./4.0)

                point_down_narrow = peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)
                ###-int(dis_to_next_down*1./2)
            else:
                dis_to_next_up = abs(peaks[jj] - peaks_neg[jj])
                dis_to_next_down = abs(peaks[jj] - peaks_neg[jj + 1])

                if peaks_values[jj] > mean_value_of_peaks - std_value_of_peaks / 2.0:
                    point_up = peaks[jj] + first_nonzero - int(1.1 * dis_to_next_up)
                    ##+int(dis_to_next_up*1./4.0)
                    point_down = peaks[jj] + first_nonzero + int(1.1 * dis_to_next_down)
                    ###-int(dis_to_next_down*1./4.0)
                else:
                    point_up = peaks[jj] + first_nonzero - int(1.23 * dis_to_next_up)
                    ##+int(dis_to_next_up*1./4.0)
                    point_down = peaks[jj] + first_nonzero + int(1.33 * dis_to_next_down)
                    ###-int(dis_to_next_down*1./4.0)

                point_down_narrow = peaks[jj] + first_nonzero + int(1.1 * dis_to_next_down)
                ###-int(dis_to_next_down*1./2)

            if point_down_narrow >= img_patch.shape[0]:
                point_down_narrow = img_patch.shape[0] - 2
            
            distances = [cv2.pointPolygonTest(contour_text_interest_copy,
                                              tuple(int(x) for x in np.array([xv[mj], peaks[jj] + first_nonzero])),
                                              True)
                         for mj in range(len(xv))]
            distances = np.array(distances)

            xvinside = xv[distances >= 0]
            if len(xvinside) == 0:
                x_min = x_min_cont
                x_max = x_max_cont
            else:
                x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
                x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)

            p1 = np.dot(rotation_matrix, [int(point_up), int(y_min_cont)])
            p2 = np.dot(rotation_matrix, [int(point_down), int(y_min_cont)])
            p3 = np.dot(rotation_matrix, [int(point_down), int(y_max_cont)])
            p4 = np.dot(rotation_matrix, [int(point_up), int(y_max_cont)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d

            if x_min_rot1 < 0:
                x_min_rot1 = 0
            if x_min_rot4 < 0:
                x_min_rot4 = 0
            if point_up_rot1 < 0:
                point_up_rot1 = 0
            if point_up_rot2 < 0:
                point_up_rot2 = 0

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                [int(x_max_rot2), int(point_up_rot2)],
                                                [int(x_max_rot3), int(point_down_rot3)],
                                                [int(x_min_rot4), int(point_down_rot4)]]))
            textline_boxes.append(np.array([[int(x_min), int(point_up)],
                                            [int(x_max), int(point_up)],
                                            [int(x_max), int(point_down)],
                                            [int(x_min), int(point_down)]]))
    elif len(peaks) < 1:
        pass
    elif len(peaks) == 1:
        x_min = x_min_cont
        x_max = x_max_cont

        y_min = y_min_cont
        y_max = y_max_cont

        p1 = np.dot(rotation_matrix, [int(point_up), int(y_min_cont)])
        p2 = np.dot(rotation_matrix, [int(point_down), int(y_min_cont)])
        p3 = np.dot(rotation_matrix, [int(point_down), int(y_max_cont)])
        p4 = np.dot(rotation_matrix, [int(point_up), int(y_max_cont)])

        x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
        x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
        x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
        x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d

        if x_min_rot1 < 0:
            x_min_rot1 = 0
        if x_min_rot4 < 0:
            x_min_rot4 = 0
        if point_up_rot1 < 0:
            point_up_rot1 = 0
        if point_up_rot2 < 0:
            point_up_rot2 = 0

        textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                            [int(x_max_rot2), int(point_up_rot2)],
                                            [int(x_max_rot3), int(point_down_rot3)],
                                            [int(x_min_rot4), int(point_down_rot4)]]))
        textline_boxes.append(np.array([[int(x_min), int(y_min)],
                                        [int(x_max), int(y_min)],
                                        [int(x_max), int(y_max)],
                                        [int(x_min), int(y_max)]]))
    elif len(peaks) == 2:
        dis_to_next = np.abs(peaks[1] - peaks[0])
        for jj in range(len(peaks)):
            if jj == 0:
                point_up = 0  # peaks[jj] + first_nonzero - int(1. / 1.7 * dis_to_next)
                if point_up < 0:
                    point_up = 1
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.8 * dis_to_next)
            elif jj == 1:
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.8 * dis_to_next)
                if point_down >= img_patch.shape[0]:
                    point_down = img_patch.shape[0] - 2
                point_up = peaks[jj] + first_nonzero - int(1.0 / 1.8 * dis_to_next)
            
            distances = [cv2.pointPolygonTest(contour_text_interest_copy,
                                              tuple(int(x) for x in np.array([xv[mj], peaks[jj] + first_nonzero])),
                                              True)
                         for mj in range(len(xv))]
            distances = np.array(distances)

            xvinside = xv[distances >= 0]
            if len(xvinside) == 0:
                x_min = x_min_cont
                x_max = x_max_cont
            else:
                x_min = np.min(xvinside)
                x_max = np.max(xvinside)

            p1 = np.dot(rotation_matrix, [int(point_up), int(y_min_cont)])
            p2 = np.dot(rotation_matrix, [int(point_down), int(y_min_cont)])
            p3 = np.dot(rotation_matrix, [int(point_down), int(y_max_cont)])
            p4 = np.dot(rotation_matrix, [int(point_up), int(y_max_cont)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d

            if x_min_rot1 < 0:
                x_min_rot1 = 0
            if x_min_rot4 < 0:
                x_min_rot4 = 0
            if point_up_rot1 < 0:
                point_up_rot1 = 0
            if point_up_rot2 < 0:
                point_up_rot2 = 0

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                [int(x_max_rot2), int(point_up_rot2)],
                                                [int(x_max_rot3), int(point_down_rot3)],
                                                [int(x_min_rot4), int(point_down_rot4)]]))
            textline_boxes.append(np.array([[int(x_min), int(point_up)],
                                            [int(x_max), int(point_up)],
                                            [int(x_max), int(point_down)],
                                            [int(x_min), int(point_down)]]))
    else:
        for jj in range(len(peaks)):
            if jj == 0:
                dis_to_next = peaks[jj + 1] - peaks[jj]
                # point_up=peaks[jj]+first_nonzero-int(1./3*dis_to_next)
                point_up = peaks[jj] + first_nonzero - int(1.0 / 1.9 * dis_to_next)
                if point_up < 0:
                    point_up = 1
                # point_down=peaks[jj]+first_nonzero+int(1./3*dis_to_next)
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.9 * dis_to_next)
            elif jj == len(peaks) - 1:
                dis_to_next = peaks[jj] - peaks[jj - 1]
                # point_down=peaks[jj]+first_nonzero+int(1./3*dis_to_next)
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.7 * dis_to_next)
                if point_down >= img_patch.shape[0]:
                    point_down = img_patch.shape[0] - 2
                # point_up=peaks[jj]+first_nonzero-int(1./3*dis_to_next)
                point_up = peaks[jj] + first_nonzero - int(1.0 / 1.9 * dis_to_next)
            else:
                dis_to_next_down = peaks[jj + 1] - peaks[jj]
                dis_to_next_up = peaks[jj] - peaks[jj - 1]

                point_up = peaks[jj] + first_nonzero - int(1.0 / 1.9 * dis_to_next_up)
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.9 * dis_to_next_down)
            
            distances = [cv2.pointPolygonTest(contour_text_interest_copy,
                                              tuple(int(x) for x in np.array([xv[mj], peaks[jj] + first_nonzero])),
                                              True)
                         for mj in range(len(xv))]
            distances = np.array(distances)

            xvinside = xv[distances >= 0]
            if len(xvinside) == 0:
                x_min = x_min_cont
                x_max = x_max_cont
            else:
                x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
                x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)

            p1 = np.dot(rotation_matrix, [int(point_up), int(y_min_cont)])
            p2 = np.dot(rotation_matrix, [int(point_down), int(y_min_cont)])
            p3 = np.dot(rotation_matrix, [int(point_down), int(y_max_cont)])
            p4 = np.dot(rotation_matrix, [int(point_up), int(y_max_cont)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d

            if x_min_rot1 < 0:
                x_min_rot1 = 0
            if x_min_rot4 < 0:
                x_min_rot4 = 0
            if point_up_rot1 < 0:
                point_up_rot1 = 0
            if point_up_rot2 < 0:
                point_up_rot2 = 0

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)],
                                                [int(x_max_rot2), int(point_up_rot2)],
                                                [int(x_max_rot3), int(point_down_rot3)],
                                                [int(x_min_rot4), int(point_down_rot4)]]))
            textline_boxes.append(np.array([[int(x_min), int(point_up)],
                                            [int(x_max), int(point_up)],
                                            [int(x_max), int(point_down)],
                                            [int(x_min), int(point_down)]]))
    return peaks, textline_boxes_rot

def separate_lines_new_inside_tiles2(img_patch, thetha):
    (h, w) = img_patch.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
    x_d = M[0, 2]
    y_d = M[1, 2]

    thetha = thetha / 180.0 * np.pi
    rotation_matrix = np.array([[np.cos(thetha), -np.sin(thetha)], [np.sin(thetha), np.cos(thetha)]])
    # contour_text_interest_copy = contour_text_interest.copy()

    # x_cont = contour_text_interest[:, 0, 0]
    # y_cont = contour_text_interest[:, 0, 1]
    # x_cont = x_cont - np.min(x_cont)
    # y_cont = y_cont - np.min(y_cont)

    x_min_cont = 0
    x_max_cont = img_patch.shape[1]
    y_min_cont = 0
    y_max_cont = img_patch.shape[0]

    xv = np.linspace(x_min_cont, x_max_cont, 1000)
    textline_patch_sum_along_width = img_patch.sum(axis=1)
    first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

    y = textline_patch_sum_along_width[:]  # [first_nonzero:last_nonzero]
    y_padded = np.zeros(len(y) + 40)
    y_padded[20 : len(y) + 20] = y
    x = np.array(range(len(y)))

    peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
    if 1 > 0:
        try:
            y_padded_smoothed_e = gaussian_filter1d(y_padded, 2)
            y_padded_up_to_down_e = -y_padded + np.max(y_padded)
            y_padded_up_to_down_padded_e = np.zeros(len(y_padded_up_to_down_e) + 40)
            y_padded_up_to_down_padded_e[20 : len(y_padded_up_to_down_e) + 20] = y_padded_up_to_down_e
            y_padded_up_to_down_padded_e = gaussian_filter1d(y_padded_up_to_down_padded_e, 2)

            peaks_e, _ = find_peaks(y_padded_smoothed_e, height=0)
            peaks_neg_e, _ = find_peaks(y_padded_up_to_down_padded_e, height=0)
            neg_peaks_max = np.max(y_padded_up_to_down_padded_e[peaks_neg_e])

            arg_neg_must_be_deleted = np.arange(len(peaks_neg_e))[
                y_padded_up_to_down_padded_e[peaks_neg_e] / float(neg_peaks_max) < 0.3]
            diff_arg_neg_must_be_deleted = np.diff(arg_neg_must_be_deleted)

            arg_diff = np.array(range(len(diff_arg_neg_must_be_deleted)))
            arg_diff_cluster = arg_diff[diff_arg_neg_must_be_deleted > 1]

            peaks_new = peaks_e[:]
            peaks_neg_new = peaks_neg_e[:]

            clusters_to_be_deleted = []
            if len(arg_diff_cluster) > 0:
                clusters_to_be_deleted.append(arg_neg_must_be_deleted[0 : arg_diff_cluster[0] + 1])
                for i in range(len(arg_diff_cluster) - 1):
                    clusters_to_be_deleted.append(
                        arg_neg_must_be_deleted[arg_diff_cluster[i] + 1:
                                                arg_diff_cluster[i + 1] + 1])
                clusters_to_be_deleted.append(
                    arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster) - 1] + 1 :])
            if len(clusters_to_be_deleted) > 0:
                peaks_new_extra = []
                for m in range(len(clusters_to_be_deleted)):
                    min_cluster = np.min(peaks_e[clusters_to_be_deleted[m]])
                    max_cluster = np.max(peaks_e[clusters_to_be_deleted[m]])
                    peaks_new_extra.append(int((min_cluster + max_cluster) / 2.0))
                    for m1 in range(len(clusters_to_be_deleted[m])):
                        peaks_new = peaks_new[peaks_new != peaks_e[clusters_to_be_deleted[m][m1] - 1]]
                        peaks_new = peaks_new[peaks_new != peaks_e[clusters_to_be_deleted[m][m1]]]
                        peaks_neg_new = peaks_neg_new[peaks_neg_new != peaks_neg_e[clusters_to_be_deleted[m][m1]]]
                peaks_new_tot = []
                for i1 in peaks_new:
                    peaks_new_tot.append(i1)
                for i1 in peaks_new_extra:
                    peaks_new_tot.append(i1)
                peaks_new_tot = np.sort(peaks_new_tot)
            else:
                peaks_new_tot = peaks_e[:]

            textline_con, hierarchy = return_contours_of_image(img_patch)
            textline_con_fil = filter_contours_area_of_image(img_patch,
                                                             textline_con, hierarchy,
                                                             max_area=1, min_area=0.0008)
            if len(np.diff(peaks_new_tot)):
                y_diff_mean = np.mean(np.diff(peaks_new_tot))  # self.find_contours_mean_y_diff(textline_con_fil)
                sigma_gaus = int(y_diff_mean * (7.0 / 40.0))
            else:
                sigma_gaus = 12

        except:
            sigma_gaus = 12
        if sigma_gaus < 3:
            sigma_gaus = 3

    y_padded_smoothed = gaussian_filter1d(y_padded, sigma_gaus)
    y_padded_up_to_down = -y_padded + np.max(y_padded)
    y_padded_up_to_down_padded = np.zeros(len(y_padded_up_to_down) + 40)
    y_padded_up_to_down_padded[20 : len(y_padded_up_to_down) + 20] = y_padded_up_to_down
    y_padded_up_to_down_padded = gaussian_filter1d(y_padded_up_to_down_padded, sigma_gaus)

    peaks, _ = find_peaks(y_padded_smoothed, height=0)
    peaks_neg, _ = find_peaks(y_padded_up_to_down_padded, height=0)

    peaks_new = peaks[:]
    peaks_neg_new = peaks_neg[:]

    try:
        neg_peaks_max = np.max(y_padded_smoothed[peaks])

        arg_neg_must_be_deleted = np.arange(len(peaks_neg))[
            y_padded_up_to_down_padded[peaks_neg] / float(neg_peaks_max) < 0.24]
        diff_arg_neg_must_be_deleted = np.diff(arg_neg_must_be_deleted)

        arg_diff = np.array(range(len(diff_arg_neg_must_be_deleted)))
        arg_diff_cluster = arg_diff[diff_arg_neg_must_be_deleted > 1]

        clusters_to_be_deleted = []
        if len(arg_neg_must_be_deleted) >= 2 and len(arg_diff_cluster) >= 2:
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[0 : arg_diff_cluster[0] + 1])
            for i in range(len(arg_diff_cluster) - 1):
                clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[i] + 1 :
                                                                      arg_diff_cluster[i + 1] + 1])
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster) - 1] + 1 :])
        elif len(arg_neg_must_be_deleted) >= 2 and len(arg_diff_cluster) == 0:
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[:])
        else:
            clusters_to_be_deleted.append(arg_neg_must_be_deleted)
        if len(clusters_to_be_deleted) > 0:
            peaks_new_extra = []
            for m in range(len(clusters_to_be_deleted)):
                min_cluster = np.min(peaks[clusters_to_be_deleted[m]])
                max_cluster = np.max(peaks[clusters_to_be_deleted[m]])
                peaks_new_extra.append(int((min_cluster + max_cluster) / 2.0))
                for m1 in range(len(clusters_to_be_deleted[m])):
                    peaks_new = peaks_new[peaks_new != peaks[clusters_to_be_deleted[m][m1] - 1]]
                    peaks_new = peaks_new[peaks_new != peaks[clusters_to_be_deleted[m][m1]]]
                    peaks_neg_new = peaks_neg_new[peaks_neg_new != peaks_neg[clusters_to_be_deleted[m][m1]]]
            peaks_new_tot = []
            for i1 in peaks_new:
                peaks_new_tot.append(i1)
            for i1 in peaks_new_extra:
                peaks_new_tot.append(i1)
            peaks_new_tot = np.sort(peaks_new_tot)

            # plt.plot(y_padded_up_to_down_padded)
            # plt.plot(peaks_neg,y_padded_up_to_down_padded[peaks_neg],'*')
            # plt.show()

            # plt.plot(y_padded_up_to_down_padded)
            # plt.plot(peaks_neg_new,y_padded_up_to_down_padded[peaks_neg_new],'*')
            # plt.show()

            # plt.plot(y_padded_smoothed)
            # plt.plot(peaks,y_padded_smoothed[peaks],'*')
            # plt.show()

            # plt.plot(y_padded_smoothed)
            # plt.plot(peaks_new_tot,y_padded_smoothed[peaks_new_tot],'*')
            # plt.show()
            peaks = peaks_new_tot[:]
            peaks_neg = peaks_neg_new[:]
    except:
        pass

    else:
        peaks_new_tot = peaks[:]
        peaks = peaks_new_tot[:]
        peaks_neg = peaks_neg_new[:]
    
    if len(y_padded_smoothed[peaks]) > 1:
        mean_value_of_peaks = np.mean(y_padded_smoothed[peaks])
        std_value_of_peaks = np.std(y_padded_smoothed[peaks])
    else:
        mean_value_of_peaks = np.nan
        std_value_of_peaks = np.nan
        
    peaks_values = y_padded_smoothed[peaks]

    ###peaks_neg = peaks_neg - 20 - 20
    ###peaks = peaks - 20
    peaks_neg_true = peaks_neg[:]
    peaks_pos_true = peaks[:]

    if len(peaks_neg_true) > 0:
        peaks_neg_true = np.array(peaks_neg_true)
        peaks_neg_true = peaks_neg_true - 20 - 20

        for i in range(len(peaks_neg_true)):
            img_patch[peaks_neg_true[i] - 6 : peaks_neg_true[i] + 6, :] = 0
    else:
        pass

    if len(peaks_pos_true) > 0:
        peaks_pos_true = np.array(peaks_pos_true)
        peaks_pos_true = peaks_pos_true - 20

        for i in range(len(peaks_pos_true)):
            ##img_patch[peaks_pos_true[i]-8:peaks_pos_true[i]+8,:]=1
            img_patch[peaks_pos_true[i] - 6 : peaks_pos_true[i] + 6, :] = 1
    else:
        pass
    kernel = np.ones((5, 5), np.uint8)

    # img_patch = cv2.erode(img_patch,kernel,iterations = 3)
    #######################img_patch = cv2.erode(img_patch,kernel,iterations = 2)
    img_patch = cv2.erode(img_patch, kernel, iterations=1)
    return img_patch

def separate_lines_new_inside_tiles(img_path, thetha):
    (h, w) = img_path.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
    x_d = M[0, 2]
    y_d = M[1, 2]

    thetha = thetha / 180.0 * np.pi
    rotation_matrix = np.array([[np.cos(thetha), -np.sin(thetha)], [np.sin(thetha), np.cos(thetha)]])

    x_min_cont = 0
    x_max_cont = img_path.shape[1]
    y_min_cont = 0
    y_max_cont = img_path.shape[0]

    xv = np.linspace(x_min_cont, x_max_cont, 1000)

    mada_n = img_path.sum(axis=1)

    ##plt.plot(mada_n)
    ##plt.show()

    first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

    y = mada_n[:]  # [first_nonzero:last_nonzero]
    y_help = np.zeros(len(y) + 40)
    y_help[20 : len(y) + 20] = y
    x = np.array(range(len(y)))

    peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
    if len(peaks_real) <= 2 and len(peaks_real) > 1:
        sigma_gaus = 10
    else:
        sigma_gaus = 5

    z = gaussian_filter1d(y_help, sigma_gaus)
    zneg_rev = -y_help + np.max(y_help)
    zneg = np.zeros(len(zneg_rev) + 40)
    zneg[20 : len(zneg_rev) + 20] = zneg_rev
    zneg = gaussian_filter1d(zneg, sigma_gaus)

    peaks, _ = find_peaks(z, height=0)
    peaks_neg, _ = find_peaks(zneg, height=0)

    for nn in range(len(peaks_neg)):
        if peaks_neg[nn] > len(z) - 1:
            peaks_neg[nn] = len(z) - 1
        if peaks_neg[nn] < 0:
            peaks_neg[nn] = 0

    diff_peaks = np.abs(np.diff(peaks_neg))

    cut_off = 20
    peaks_neg_true = []
    forest = []

    for i in range(len(peaks_neg)):
        if i == 0:
            forest.append(peaks_neg[i])
        if i < (len(peaks_neg) - 1):
            if diff_peaks[i] <= cut_off:
                forest.append(peaks_neg[i + 1])
            if diff_peaks[i] > cut_off:
                if not np.isnan(forest[np.argmin(z[forest])]):
                    peaks_neg_true.append(forest[np.argmin(z[forest])])
                forest = []
                forest.append(peaks_neg[i + 1])
        if i == (len(peaks_neg) - 1):
            if not np.isnan(forest[np.argmin(z[forest])]):
                peaks_neg_true.append(forest[np.argmin(z[forest])])

    diff_peaks_pos = np.abs(np.diff(peaks))

    cut_off = 20
    peaks_pos_true = []
    forest = []

    for i in range(len(peaks)):
        if i == 0:
            forest.append(peaks[i])
        if i < (len(peaks) - 1):
            if diff_peaks_pos[i] <= cut_off:
                forest.append(peaks[i + 1])
            if diff_peaks_pos[i] > cut_off:
                if not np.isnan(forest[np.argmax(z[forest])]):
                    peaks_pos_true.append(forest[np.argmax(z[forest])])
                forest = []
                forest.append(peaks[i + 1])
        if i == (len(peaks) - 1):
            if not np.isnan(forest[np.argmax(z[forest])]):
                peaks_pos_true.append(forest[np.argmax(z[forest])])


    if len(peaks_neg_true) > 0:
        peaks_neg_true = np.array(peaks_neg_true)
        """
        #plt.figure(figsize=(40,40))
        #plt.subplot(1,2,1)
        #plt.title('Textline segmentation von Textregion')
        #plt.imshow(img_path)
        #plt.xlabel('X')
        #plt.ylabel('Y')
        #plt.subplot(1,2,2)
        #plt.title('Dichte entlang X')
        #base = pyplot.gca().transData
        #rot = transforms.Affine2D().rotate_deg(90)
        #plt.plot(zneg,np.array(range(len(zneg))))
        #plt.plot(zneg[peaks_neg_true],peaks_neg_true,'*')
        #plt.gca().invert_yaxis()

        #plt.xlabel('Dichte')
        #plt.ylabel('Y')
        ##plt.plot([0,len(y)], [grenze,grenze])
        #plt.show()
        """
        peaks_neg_true = peaks_neg_true - 20 - 20

        for i in range(len(peaks_neg_true)):
            img_path[peaks_neg_true[i] - 6 : peaks_neg_true[i] + 6, :] = 0

    else:
        pass

    if len(peaks_pos_true) > 0:
        peaks_pos_true = np.array(peaks_pos_true)
        peaks_pos_true = peaks_pos_true - 20

        for i in range(len(peaks_pos_true)):
            img_path[peaks_pos_true[i] - 8 : peaks_pos_true[i] + 8, :] = 1
    else:
        pass
    kernel = np.ones((5, 5), np.uint8)

    # img_path = cv2.erode(img_path,kernel,iterations = 3)
    img_path = cv2.erode(img_path, kernel, iterations=2)
    return img_path

def separate_lines_vertical_cont(img_patch, contour_text_interest, thetha, box_ind, add_boxes_coor_into_textlines):
    kernel = np.ones((5, 5), np.uint8)
    label = 255
    min_area = 0
    max_area = 1

    if img_patch.ndim == 3:
        cnts_images = (img_patch[:, :, 0] == label) * 1
    else:
        cnts_images = (img_patch[:, :] == label) * 1
    _, thresh = cv2.threshold(cnts_images.astype(np.uint8), 0, 255, 0)
    contours_imgs, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hierarchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh,
                                                         contours_imgs, hierarchy,
                                                         max_area=max_area, min_area=min_area)
    cont_final = []
    for i in range(len(contours_imgs)):
        img_contour = np.zeros(cnts_images.shape[:2], dtype=np.uint8)
        img_contour = cv2.fillPoly(img_contour, pts=[contours_imgs[i]], color=255)

        img_contour = cv2.dilate(img_contour, kernel, iterations=4)
        _, threshrot = cv2.threshold(img_contour, 0, 255, 0)
        contours_text_rot, _ = cv2.findContours(threshrot.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ##contour_text_copy[:, 0, 0] = contour_text_copy[:, 0, 0] - box_ind[
        ##0]
        ##contour_text_copy[:, 0, 1] = contour_text_copy[:, 0, 1] - box_ind[1]
        ##if add_boxes_coor_into_textlines:
        ##contours_text_rot[0][:, 0, 0]=contours_text_rot[0][:, 0, 0] + box_ind[0]
        ##contours_text_rot[0][:, 0, 1]=contours_text_rot[0][:, 0, 1] + box_ind[1]
        cont_final.append(contours_text_rot[0])

    return None, cont_final

def textline_contours_postprocessing(textline_mask, slope,
                                     contour_text_interest, box_ind,
                                     add_boxes_coor_into_textlines=False):
    textline_mask = textline_mask * 255
    kernel = np.ones((5, 5), np.uint8)
    textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_OPEN, kernel)
    textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_CLOSE, kernel)
    textline_mask = cv2.erode(textline_mask, kernel, iterations=2)
    # textline_mask = cv2.erode(textline_mask, kernel, iterations=1)

    x_help = 30
    y_help = 2

    textline_mask_help = np.zeros((textline_mask.shape[0] + int(2 * y_help),
                                   textline_mask.shape[1] + int(2 * x_help)))
    textline_mask_help[y_help : y_help + textline_mask.shape[0],
                       x_help : x_help + textline_mask.shape[1]] = np.copy(textline_mask[:, :])

    dst = rotate_image(textline_mask_help, slope)
    dst[dst != 0] = 1

    # if np.abs(slope)>.5 and textline_mask.shape[0]/float(textline_mask.shape[1])>3:
    # plt.imshow(dst)
    # plt.show()

    contour_text_copy = contour_text_interest.copy()
    contour_text_copy[:, 0, 0] = contour_text_copy[:, 0, 0] - box_ind[0]
    contour_text_copy[:, 0, 1] = contour_text_copy[:, 0, 1] - box_ind[1]

    img_contour = np.zeros((box_ind[3], box_ind[2]))
    img_contour = cv2.fillPoly(img_contour, pts=[contour_text_copy], color=255)

    img_contour_help = np.zeros((img_contour.shape[0] + int(2 * y_help),
                                 img_contour.shape[1] + int(2 * x_help)))
    img_contour_help[y_help : y_help + img_contour.shape[0],
                     x_help : x_help + img_contour.shape[1]] = np.copy(img_contour[:, :])

    img_contour_rot = rotate_image(img_contour_help, slope)

    _, threshrot = cv2.threshold(img_contour_rot, 0, 255, 0)
    contours_text_rot, _ = cv2.findContours(threshrot.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    len_con_text_rot = [len(contours_text_rot[ib]) for ib in range(len(contours_text_rot))]
    ind_big_con = np.argmax(len_con_text_rot)

    if abs(slope) > 45:
        _, contours_rotated_clean = separate_lines_vertical_cont(
            textline_mask, contours_text_rot[ind_big_con], box_ind, slope,
            add_boxes_coor_into_textlines=add_boxes_coor_into_textlines)
    else:
        _, contours_rotated_clean = separate_lines(
            dst, contours_text_rot[ind_big_con], slope, x_help, y_help)

    return contours_rotated_clean

def separate_lines_new2(img_crop, thetha, num_col, slope_region, logger=None, plotter=None):
    if logger is None:
        logger = getLogger(__package__)
    if not np.prod(img_crop.shape):
        return img_crop

    if num_col == 1:
        num_patches = int(img_crop.shape[1] / 200.0)
    else:
        num_patches = int(img_crop.shape[1] / 140.0)
    # num_patches=int(img_crop.shape[1]/200.)
    if num_patches == 0:
        num_patches = 1

    img_patch_interest = img_crop[:, :]  # [peaks_neg_true[14]-dis_up:peaks_neg_true[15]+dis_down ,:]

    # plt.imshow(img_patch_interest)
    # plt.show()

    length_x = int(img_crop.shape[1] / float(num_patches))
    # margin = int(0.04 * length_x) just recently this was changed because it break lines into 2
    margin = int(0.04 * length_x)
    # if margin<=4:
    # margin = int(0.08 * length_x)
    # margin=0

    width_mid = length_x - 2 * margin
    nxf = img_crop.shape[1] / float(width_mid)

    if nxf > int(nxf):
        nxf = int(nxf) + 1
    else:
        nxf = int(nxf)

    slopes_tile_wise = []
    for i in range(nxf):
        if i == 0:
            index_x_d = i * width_mid
            index_x_u = index_x_d + length_x
        elif i > 0:
            index_x_d = i * width_mid
            index_x_u = index_x_d + length_x

        if index_x_u > img_crop.shape[1]:
            index_x_u = img_crop.shape[1]
            index_x_d = img_crop.shape[1] - length_x

        # img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
        img_xline = img_patch_interest[:, index_x_d:index_x_u]

        try:
            assert img_xline.any()
            slope_xline = return_deskew_slop(img_xline, 2, logger=logger, plotter=plotter)
        except:
            slope_xline = 0

        if abs(slope_region) < 25 and abs(slope_xline) > 25:
            slope_xline = [slope_region][0]
        # if abs(slope_region)>70 and abs(slope_xline)<25:
        # slope_xline=[slope_region][0]
        slopes_tile_wise.append(slope_xline)
        img_line_rotated = rotate_image(img_xline, slope_xline)
        img_line_rotated[:, :][img_line_rotated[:, :] != 0] = 1
        
    img_patch_interest = img_crop[:, :]  # [peaks_neg_true[14]-dis_up:peaks_neg_true[14]+dis_down ,:]

    img_patch_interest_revised = np.zeros(img_patch_interest.shape)

    for i in range(nxf):
        if i == 0:
            index_x_d = i * width_mid
            index_x_u = index_x_d + length_x
        elif i > 0:
            index_x_d = i * width_mid
            index_x_u = index_x_d + length_x

        if index_x_u > img_crop.shape[1]:
            index_x_u = img_crop.shape[1]
            index_x_d = img_crop.shape[1] - length_x

        img_xline = img_patch_interest[:, index_x_d:index_x_u]

        img_int = np.zeros((img_xline.shape[0], img_xline.shape[1]))
        img_int[:, :] = img_xline[:, :]  # img_patch_org[:,:,0]

        img_resized = np.zeros((int(img_int.shape[0] * (1.2)), int(img_int.shape[1] * (3))))
        img_resized[int(img_int.shape[0] * (0.1)) : int(img_int.shape[0] * (0.1)) + img_int.shape[0],
                    int(img_int.shape[1] * (1.0)) : int(img_int.shape[1] * (1.0)) + img_int.shape[1]] = img_int[:, :]
        # plt.imshow(img_xline)
        # plt.show()
        img_line_rotated = rotate_image(img_resized, slopes_tile_wise[i])
        img_line_rotated[:, :][img_line_rotated[:, :] != 0] = 1

        img_patch_separated = separate_lines_new_inside_tiles2(img_line_rotated, 0)

        img_patch_separated_returned = rotate_image(img_patch_separated, -slopes_tile_wise[i])
        img_patch_separated_returned[:, :][img_patch_separated_returned[:, :] != 0] = 1

        img_patch_separated_returned_true_size = img_patch_separated_returned[
            int(img_int.shape[0] * (0.1)) : int(img_int.shape[0] * (0.1)) + img_int.shape[0],
            int(img_int.shape[1] * (1.0)) : int(img_int.shape[1] * (1.0)) + img_int.shape[1]]

        img_patch_separated_returned_true_size = img_patch_separated_returned_true_size[:, margin : length_x - margin]
        img_patch_interest_revised[:, index_x_d + margin : index_x_u - margin] = img_patch_separated_returned_true_size

    return img_patch_interest_revised

@wrap_ndarray_shared(kw='img')
def do_image_rotation(angle, img=None, sigma_des=1.0, logger=None):
    if logger is None:
        logger = getLogger(__package__)
    img_rot = rotate_image(img, angle)
    img_rot[img_rot!=0] = 1
    try:
        var = find_num_col_deskew(img_rot, sigma_des, 20.3)
    except:
        logger.exception("cannot determine variance for angle %.2f°", angle)
        var = 0
    return var

def return_deskew_slop(img_patch_org, sigma_des,n_tot_angles=100,
                       main_page=False, logger=None, plotter=None, map=None):
    if main_page and plotter:
        plotter.save_plot_of_textline_density(img_patch_org)
    
    img_int=np.zeros((img_patch_org.shape[0],img_patch_org.shape[1]))
    img_int[:,:]=img_patch_org[:,:]#img_patch_org[:,:,0]

    max_shape=np.max(img_int.shape)
    img_resized=np.zeros((int( max_shape*(1.1) ) , int( max_shape*(1.1) ) ))

    onset_x=int((img_resized.shape[1]-img_int.shape[1])/2.)
    onset_y=int((img_resized.shape[0]-img_int.shape[0])/2.)

    #img_resized=np.zeros((int( img_int.shape[0]*(1.8) ) , int( img_int.shape[1]*(2.6) ) ))
    #img_resized[ int( img_int.shape[0]*(.4)):int( img_int.shape[0]*(.4))+img_int.shape[0],
    #             int( img_int.shape[1]*(.8)):int( img_int.shape[1]*(.8))+img_int.shape[1] ]=img_int[:,:]
    img_resized[ onset_y:onset_y+img_int.shape[0] , onset_x:onset_x+img_int.shape[1] ]=img_int[:,:]

    if main_page and img_patch_org.shape[1] > img_patch_org.shape[0]:
        angles = np.array([-45, 0, 45, 90,])
        angle, _ = get_smallest_skew(img_resized, sigma_des, angles, map=map, logger=logger, plotter=plotter)

        angles = np.linspace(angle - 22.5, angle + 22.5, n_tot_angles)
        angle, _ = get_smallest_skew(img_resized, sigma_des, angles, map=map, logger=logger, plotter=plotter)
    elif main_page:
        #angles = np.linspace(-12, 12, n_tot_angles)#np.array([0 , 45 , 90 , -45])
        angles = np.concatenate((np.linspace(-12, -7, n_tot_angles // 4),
                                 np.linspace(-6, 6, n_tot_angles // 2),
                                 np.linspace(7, 12, n_tot_angles // 4)))
        angle, var = get_smallest_skew(img_resized, sigma_des, angles, map=map, logger=logger, plotter=plotter)

        early_slope_edge=11
        if abs(angle) > early_slope_edge:
            if angle < 0:
                angles2 = np.linspace(-90, -12, n_tot_angles)
            else:
                angles2 = np.linspace(90, 12, n_tot_angles)
            angle2, var2 = get_smallest_skew(img_resized, sigma_des, angles2, map=map, logger=logger, plotter=plotter)
            if var2 > var:
                angle = angle2
    else:
        angles = np.linspace(-25, 25, int(0.5 * n_tot_angles) + 10)
        angle, var = get_smallest_skew(img_resized, sigma_des, angles, map=map, logger=logger, plotter=plotter)

        early_slope_edge=22
        if abs(angle) > early_slope_edge:
            if angle < 0:
                angles2 = np.linspace(-90, -25, int(0.5 * n_tot_angles) + 10)
            else:
                angles2 = np.linspace(90, 25, int(0.5 * n_tot_angles) + 10)
            angle2, var2 = get_smallest_skew(img_resized, sigma_des, angles2, map=map, logger=logger, plotter=plotter)
            if var2 > var:
                angle = angle2
    return angle

def get_smallest_skew(img, sigma_des, angles, logger=None, plotter=None, map=map):
    if logger is None:
        logger = getLogger(__package__)
    if map is None:
        results = [do_image_rotation.__wrapped__(angle, img=img, sigma_des=sigma_des, logger=logger)
                   for angle in angles]
    else:
        with share_ndarray(img) as img_shared:
            results = list(map(partial(do_image_rotation, img=img_shared, sigma_des=sigma_des, logger=None),
                               angles))
    if plotter:
        plotter.save_plot_of_rotation_angle(angles, results)
    try:
        var_res = np.array(results)
        assert var_res.any()
        idx = np.argmax(var_res)
        angle = angles[idx]
        var = var_res[idx]
    except:
        logger.exception("cannot determine best angle among %s", str(angles))
        angle = 0
        var = 0
    return angle, var

@wrap_ndarray_shared(kw='textline_mask_tot_ea')
@wrap_ndarray_shared(kw='mask_texts_only')
def do_work_of_slopes_new_curved(
        box_text, contour_par,
        textline_mask_tot_ea=None, mask_texts_only=None,
        num_col=1, scale_par=1.0, slope_deskew=0.0,
        logger=None, MAX_SLOPE=999, KERNEL=None, plotter=None
):
    if KERNEL is None:
        KERNEL = np.ones((5, 5), np.uint8)
    if logger is None:
        logger = getLogger(__package__)
    logger.debug("enter do_work_of_slopes_new_curved")

    x, y, w, h = box_text
    all_text_region_raw = textline_mask_tot_ea[y: y + h, x: x + w].astype(np.uint8)
    img_int_p = all_text_region_raw[:, :]

    # img_int_p=cv2.erode(img_int_p,KERNEL,iterations = 2)
    # plt.imshow(img_int_p)
    # plt.show()

    if not np.prod(img_int_p.shape) or img_int_p.shape[0] / img_int_p.shape[1] < 0.1:
        slope = 0
        slope_for_all = slope_deskew
    else:
        try:
            textline_con, hierarchy = return_contours_of_image(img_int_p)
            textline_con_fil = filter_contours_area_of_image(img_int_p, textline_con,
                                                             hierarchy,
                                                             max_area=1, min_area=0.0008)
            y_diff_mean = find_contours_mean_y_diff(textline_con_fil) if len(textline_con_fil) > 1 else np.NaN
            if np.isnan(y_diff_mean):
                slope_for_all = MAX_SLOPE
            else:
                sigma_des = max(1, int(y_diff_mean * (4.0 / 40.0)))
                img_int_p[img_int_p > 0] = 1
                slope_for_all = return_deskew_slop(img_int_p, sigma_des, logger=logger, plotter=plotter)
                if abs(slope_for_all) < 0.5:
                    slope_for_all = slope_deskew
        except:
            logger.exception("cannot determine angle of contours")
            slope_for_all = MAX_SLOPE

        if slope_for_all == MAX_SLOPE:
            slope_for_all = slope_deskew
        slope = slope_for_all

    crop_coor = box2rect(box_text)

    if abs(slope_for_all) < 45:
        textline_region_in_image = np.zeros(textline_mask_tot_ea.shape)
        x, y, w, h = cv2.boundingRect(contour_par)
        mask_biggest = np.zeros(mask_texts_only.shape)
        mask_biggest = cv2.fillPoly(mask_biggest, pts=[contour_par], color=(1, 1, 1))
        mask_region_in_patch_region = mask_biggest[y : y + h, x : x + w]
        textline_biggest_region = mask_biggest * textline_mask_tot_ea

        textline_rotated_separated = separate_lines_new2(textline_biggest_region[y: y+h, x: x+w], 0,
                                                         num_col, slope_for_all,
                                                         logger=logger, plotter=plotter)


        textline_rotated_separated[mask_region_in_patch_region[:, :] != 1] = 0

        textline_region_in_image[y : y + h, x : x + w] = textline_rotated_separated


        pixel_img = 1
        cnt_textlines_in_image = return_contours_of_interested_textline(textline_region_in_image, pixel_img)

        textlines_cnt_per_region = []
        for jjjj in range(len(cnt_textlines_in_image)):
            mask_biggest2 = np.zeros(mask_texts_only.shape)
            mask_biggest2 = cv2.fillPoly(mask_biggest2, pts=[cnt_textlines_in_image[jjjj]], color=(1, 1, 1))
            if num_col + 1 == 1:
                mask_biggest2 = cv2.dilate(mask_biggest2, KERNEL, iterations=5)
            else:
                mask_biggest2 = cv2.dilate(mask_biggest2, KERNEL, iterations=4)

            pixel_img = 1
            mask_biggest2 = resize_image(mask_biggest2,
                                         int(mask_biggest2.shape[0] * scale_par),
                                         int(mask_biggest2.shape[1] * scale_par))
            cnt_textlines_in_image_ind = return_contours_of_interested_textline(mask_biggest2, pixel_img)
            try:
                textlines_cnt_per_region.append(cnt_textlines_in_image_ind[0])
            except Exception as why:
                logger.error(why)
    else:
        textlines_cnt_per_region = textline_contours_postprocessing(all_text_region_raw,
                                                                    slope_for_all, contour_par,
                                                                    box_text, True)

    return textlines_cnt_per_region[::-1], crop_coor, slope

@wrap_ndarray_shared(kw='textline_mask_tot_ea')
def do_work_of_slopes_new_light(
        box_text, contour, contour_par,
        textline_mask_tot_ea=None, slope_deskew=0,
        logger=None
):
    if logger is None:
        logger = getLogger(__package__)
    logger.debug('enter do_work_of_slopes_new_light')

    x, y, w, h = box_text
    crop_coor = box2rect(box_text)
    mask_textline = np.zeros(textline_mask_tot_ea.shape)
    mask_textline = cv2.fillPoly(mask_textline, pts=[contour], color=(1,1,1))
    all_text_region_raw = textline_mask_tot_ea * mask_textline
    all_text_region_raw = all_text_region_raw[y: y + h, x: x + w].astype(np.uint8)

    mask_only_con_region = np.zeros(textline_mask_tot_ea.shape)
    mask_only_con_region = cv2.fillPoly(mask_only_con_region, pts=[contour_par], color=(1, 1, 1))

    all_text_region_raw = np.copy(textline_mask_tot_ea)
    all_text_region_raw[mask_only_con_region == 0] = 0
    cnt_clean_rot_raw, hir_on_cnt_clean_rot = return_contours_of_image(all_text_region_raw)
    cnt_clean_rot = filter_contours_area_of_image(all_text_region_raw, cnt_clean_rot_raw, hir_on_cnt_clean_rot,
                                                  max_area=1, min_area=0.00001)

    return cnt_clean_rot, crop_coor, slope_deskew
