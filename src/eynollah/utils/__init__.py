from typing import Tuple
from logging import getLogger
import time
import math

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
from shapely import geometry
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from .is_nan import isNaN
from .contour import (contours_in_same_horizon,
                      find_center_of_contours,
                      find_new_features_of_contours,
                      return_contours_of_image,
                      return_parent_contours)
def pairwise(iterable):
    # pairwise('ABCDEFG') → AB BC CD DE EF FG

    iterator = iter(iterable)
    a = next(iterator, None)

    for b in iterator:
        yield a, b
        a = b

def return_x_start_end_mothers_childs_and_type_of_reading_order(
        x_min_hor_some, x_max_hor_some, cy_hor_some, peak_points, cy_hor_diff):

    x_start=[]
    x_end=[]
    kind=[]#if covers 2 and more than 2 columns set it to 1 otherwise 0
    len_sep=[]
    y_sep=[]
    y_diff=[]
    new_main_sep_y=[]

    indexer=0
    for i in range(len(x_min_hor_some)):
        starting=x_min_hor_some[i]-peak_points
        starting=starting[starting>=0]
        min_start=np.argmin(starting)
        ending=peak_points-x_max_hor_some[i]
        len_ending_neg=len(ending[ending<=0])

        ending=ending[ending>0]
        max_end=np.argmin(ending)+len_ending_neg

        if (max_end-min_start)>=2:
            if (max_end-min_start)==(len(peak_points)-1):
                new_main_sep_y.append(indexer)

            #print((max_end-min_start),len(peak_points),'(max_end-min_start)')
            y_sep.append(cy_hor_some[i])
            y_diff.append(cy_hor_diff[i])
            x_end.append(max_end)

            x_start.append( min_start)

            len_sep.append(max_end-min_start)
            if max_end==min_start+1:
                kind.append(0)
            else:
                kind.append(1)

            indexer+=1

    x_start_returned = np.array(x_start, dtype=int)
    x_end_returned = np.array(x_end, dtype=int)
    y_sep_returned = np.array(y_sep, dtype=int)
    y_diff_returned = np.array(y_diff, dtype=int)

    all_args_uniq = contours_in_same_horizon(y_sep_returned)
    args_to_be_unified=[]
    y_unified=[]
    y_diff_unified=[]
    x_s_unified=[]
    x_e_unified=[]
    if len(all_args_uniq)>0:
        #print('burda')
        if type(all_args_uniq[0]) is list:
            for dd in range(len(all_args_uniq)):
                if len(all_args_uniq[dd])==2:
                    x_s_same_hor=np.array(x_start_returned)[all_args_uniq[dd]]
                    x_e_same_hor=np.array(x_end_returned)[all_args_uniq[dd]]
                    y_sep_same_hor=np.array(y_sep_returned)[all_args_uniq[dd]]
                    y_diff_same_hor=np.array(y_diff_returned)[all_args_uniq[dd]]
                    #print('burda2')
                    if (x_s_same_hor[0]==x_e_same_hor[1]-1 or
                        x_s_same_hor[1]==x_e_same_hor[0]-1 and
                        x_s_same_hor[0]!=x_s_same_hor[1] and
                        x_e_same_hor[0]!=x_e_same_hor[1]):
                        #print('burda3')
                        for arg_in in all_args_uniq[dd]:
                            #print(arg_in,'arg_in')
                            args_to_be_unified.append(arg_in)
                        y_selected=np.min(y_sep_same_hor)
                        y_diff_selected=np.max(y_diff_same_hor)
                        x_s_selected=np.min(x_s_same_hor)
                        x_e_selected=np.max(x_e_same_hor)

                        x_s_unified.append(x_s_selected)
                        x_e_unified.append(x_e_selected)
                        y_unified.append(y_selected)
                        y_diff_unified.append(y_diff_selected)
                    #print(x_s_same_hor,'x_s_same_hor')
                    #print(x_e_same_hor[:]-1,'x_e_same_hor')
                    #print('#############################')
    #print(x_s_unified,'y_selected')
    #print(x_e_unified,'x_s_selected')
    #print(y_unified,'x_e_same_hor')

    args_lines_not_unified=list( set(range(len(y_sep_returned)))-set(args_to_be_unified) )
    #print(args_lines_not_unified,'args_lines_not_unified')

    x_start_returned_not_unified=list( np.array(x_start_returned)[args_lines_not_unified] )
    x_end_returned_not_unified=list( np.array(x_end_returned)[args_lines_not_unified] )
    y_sep_returned_not_unified=list (np.array(y_sep_returned)[args_lines_not_unified] )
    y_diff_returned_not_unified=list (np.array(y_diff_returned)[args_lines_not_unified] )

    for dv in range(len(y_unified)):
        y_sep_returned_not_unified.append(y_unified[dv])
        y_diff_returned_not_unified.append(y_diff_unified[dv])
        x_start_returned_not_unified.append(x_s_unified[dv])
        x_end_returned_not_unified.append(x_e_unified[dv])

    #print(y_sep_returned,'y_sep_returned')
    #print(x_start_returned,'x_start_returned')
    #print(x_end_returned,'x_end_returned')

    x_start_returned = np.array(x_start_returned_not_unified, dtype=int)
    x_end_returned = np.array(x_end_returned_not_unified, dtype=int)
    y_sep_returned = np.array(y_sep_returned_not_unified, dtype=int)
    y_diff_returned = np.array(y_diff_returned_not_unified, dtype=int)

    #print(y_sep_returned,'y_sep_returned2')
    #print(x_start_returned,'x_start_returned2')
    #print(x_end_returned,'x_end_returned2')
    #print(new_main_sep_y,'new_main_sep_y')

    #print(x_start,'x_start')
    #print(x_end,'x_end')
    if len(new_main_sep_y)>0:

        min_ys=np.min(y_sep)
        max_ys=np.max(y_sep)

        y_mains=[]
        y_mains.append(min_ys)
        y_mains_sep_ohne_grenzen=[]

        for ii in range(len(new_main_sep_y)):
            y_mains.append(y_sep[new_main_sep_y[ii]])
            y_mains_sep_ohne_grenzen.append(y_sep[new_main_sep_y[ii]])

        y_mains.append(max_ys)

        y_mains_sorted=np.sort(y_mains)
        diff=np.diff(y_mains_sorted)
        argm=np.argmax(diff)

        y_min_new=y_mains_sorted[argm]
        y_max_new=y_mains_sorted[argm+1]

        #print(y_min_new,'y_min_new')
        #print(y_max_new,'y_max_new')
        #print(y_sep[new_main_sep_y[0]],y_sep,'yseps')
        x_start=np.array(x_start)
        x_end=np.array(x_end)
        kind=np.array(kind)
        y_sep=np.array(y_sep)
        if (y_min_new in y_mains_sep_ohne_grenzen and
            y_max_new in y_mains_sep_ohne_grenzen):
            x_start=x_start[(y_sep>y_min_new) & (y_sep<y_max_new)]
            x_end=x_end[(y_sep>y_min_new) & (y_sep<y_max_new)]
            kind=kind[(y_sep>y_min_new) & (y_sep<y_max_new)]
            y_sep=y_sep[(y_sep>y_min_new) & (y_sep<y_max_new)]
        elif (y_min_new in y_mains_sep_ohne_grenzen and
              y_max_new not in y_mains_sep_ohne_grenzen):
            #print('burda')
            x_start=x_start[(y_sep>y_min_new) & (y_sep<=y_max_new)]
            #print('burda1')
            x_end=x_end[(y_sep>y_min_new) & (y_sep<=y_max_new)]
            #print('burda2')
            kind=kind[(y_sep>y_min_new) & (y_sep<=y_max_new)]
            y_sep=y_sep[(y_sep>y_min_new) & (y_sep<=y_max_new)]
        elif (y_min_new not in y_mains_sep_ohne_grenzen and
              y_max_new in y_mains_sep_ohne_grenzen):
            x_start=x_start[(y_sep>=y_min_new) & (y_sep<y_max_new)]
            x_end=x_end[(y_sep>=y_min_new) & (y_sep<y_max_new)]
            kind=kind[(y_sep>=y_min_new) & (y_sep<y_max_new)]
            y_sep=y_sep[(y_sep>=y_min_new) & (y_sep<y_max_new)]
        else:
            x_start=x_start[(y_sep>=y_min_new) & (y_sep<=y_max_new)]
            x_end=x_end[(y_sep>=y_min_new) & (y_sep<=y_max_new)]
            kind=kind[(y_sep>=y_min_new) & (y_sep<=y_max_new)]
            y_sep=y_sep[(y_sep>=y_min_new) & (y_sep<=y_max_new)]
    #print(x_start,'x_start')
    #print(x_end,'x_end')
    #print(len_sep)

    deleted=[]
    for i in range(len(x_start)-1):
        nodes_i=set(range(x_start[i],x_end[i]+1))
        for j in range(i+1,len(x_start)):
            if nodes_i==set(range(x_start[j],x_end[j]+1)):
                    deleted.append(j)
    #print(np.unique(deleted))

    remained_sep_indexes=set(range(len(x_start)))-set(np.unique(deleted) )
    #print(remained_sep_indexes,'remained_sep_indexes')
    mother=[]#if it has mother
    child=[]
    for index_i in remained_sep_indexes:
        have_mother=0
        have_child=0
        nodes_ind=set(range(x_start[index_i],x_end[index_i]+1))
        for index_j in remained_sep_indexes:
            nodes_ind_j=set(range(x_start[index_j],x_end[index_j]+1))
            if nodes_ind<nodes_ind_j:
                have_mother=1
            if nodes_ind>nodes_ind_j:
                have_child=1
        mother.append(have_mother)
        child.append(have_child)

    #print(mother,'mother')
    #print(len(remained_sep_indexes))
    #print(len(remained_sep_indexes),len(x_start),len(x_end),len(y_sep),'lens')
    y_lines_without_mother=[]
    x_start_without_mother=[]
    x_end_without_mother=[]

    y_lines_with_child_without_mother=[]
    x_start_with_child_without_mother=[]
    x_end_with_child_without_mother=[]

    mother = np.array(mother)
    child = np.array(child)
    #print(mother,'mother')
    #print(child,'child')
    remained_sep_indexes = np.array(list(remained_sep_indexes))
    x_start = np.array(x_start)
    x_end = np.array(x_end)
    y_sep = np.array(y_sep)

    if len(remained_sep_indexes)>1:
        #print(np.array(remained_sep_indexes),'np.array(remained_sep_indexes)')
        #print(np.array(mother),'mother')
        remained_sep_indexes_without_mother = remained_sep_indexes[mother==0]
        remained_sep_indexes_with_child_without_mother = remained_sep_indexes[(mother==0) & (child==1)]
        #print(remained_sep_indexes_without_mother,'remained_sep_indexes_without_mother')
        #print(remained_sep_indexes_without_mother,'remained_sep_indexes_without_mother')

        x_end_with_child_without_mother = x_end[remained_sep_indexes_with_child_without_mother]
        x_start_with_child_without_mother = x_start[remained_sep_indexes_with_child_without_mother]
        y_lines_with_child_without_mother = y_sep[remained_sep_indexes_with_child_without_mother]

        reading_orther_type=0
        x_end_without_mother = x_end[remained_sep_indexes_without_mother]
        x_start_without_mother = x_start[remained_sep_indexes_without_mother]
        y_lines_without_mother = y_sep[remained_sep_indexes_without_mother]

        if len(remained_sep_indexes_without_mother)>=2:
            for i in range(len(remained_sep_indexes_without_mother)-1):
                nodes_i=set(range(x_start[remained_sep_indexes_without_mother[i]],
                                  x_end[remained_sep_indexes_without_mother[i]]
                                  # + 1
                                  ))
                for j in range(i+1,len(remained_sep_indexes_without_mother)):
                    nodes_j=set(range(x_start[remained_sep_indexes_without_mother[j]],
                                      x_end[remained_sep_indexes_without_mother[j]]
                                      # + 1
                                      ))
                    set_diff = nodes_i - nodes_j
                    if set_diff != nodes_i:
                        reading_orther_type = 1
    else:
        reading_orther_type = 0
    #print(reading_orther_type,'javab')
    #print(y_lines_with_child_without_mother,'y_lines_with_child_without_mother')
    #print(x_start_with_child_without_mother,'x_start_with_child_without_mother')
    #print(x_end_with_child_without_mother,'x_end_with_hild_without_mother')

    len_sep_with_child = len(child[child==1])

    #print(len_sep_with_child,'len_sep_with_child')
    there_is_sep_with_child = 0
    if len_sep_with_child >= 1:
        there_is_sep_with_child = 1
    #print(all_args_uniq,'all_args_uniq')
    #print(args_to_be_unified,'args_to_be_unified')

    return (reading_orther_type,
            x_start_returned,
            x_end_returned,
            y_sep_returned,
            y_diff_returned,
            y_lines_without_mother,
            x_start_without_mother,
            x_end_without_mother,
            there_is_sep_with_child,
            y_lines_with_child_without_mother,
            x_start_with_child_without_mother,
            x_end_with_child_without_mother,
            new_main_sep_y)

def box2rect(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return (box[1], box[1] + box[3],
            box[0], box[0] + box[2])

def box2slice(box: Tuple[int, int, int, int]) -> Tuple[slice, slice]:
    return (slice(box[1], box[1] + box[3]),
            slice(box[0], box[0] + box[2]))

def crop_image_inside_box(box, img_org_copy):
    image_box = img_org_copy[box2slice(box)]
    return image_box, box2rect(box)

def otsu_copy_binary(img):
    img_r = np.zeros((img.shape[0], img.shape[1], 3))
    img1 = img[:, :, 0]

    retval1, threshold1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_r[:, :, 0] = threshold1
    img_r[:, :, 1] = threshold1
    img_r[:, :, 2] = threshold1

    img_r = img_r / float(np.max(img_r)) * 255
    return img_r

def find_features_of_lines(contours_main):
    areas_main = np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
    M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
    cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
    cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
    x_min_main = np.array([np.min(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])
    x_max_main = np.array([np.max(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

    y_min_main = np.array([np.min(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
    y_max_main = np.array([np.max(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])

    slope_lines = []
    for kk in range(len(contours_main)):
        [vx, vy, x, y] = cv2.fitLine(contours_main[kk], cv2.DIST_L2, 0, 0.01, 0.01)
        slope_lines.append(((vy / vx) / np.pi * 180)[0])

    slope_lines_org = slope_lines
    slope_lines = np.array(slope_lines)
    slope_lines[(slope_lines < 10) & (slope_lines > -10)] = 0

    slope_lines[(slope_lines < -200) | (slope_lines > 200)] = 1
    slope_lines[(slope_lines != 0) & (slope_lines != 1)] = 2

    dis_x = np.abs(x_max_main - x_min_main)
    return (slope_lines,
            dis_x,
            x_min_main,
            x_max_main,
            np.array(cy_main),
            np.array(slope_lines_org),
            y_min_main,
            y_max_main,
            np.array(cx_main))

def boosting_headers_by_longshot_region_segmentation(textregion_pre_p, textregion_pre_np, img_only_text):
    textregion_pre_p_org = np.copy(textregion_pre_p)
    # 4 is drop capitals
    headers_in_longshot = textregion_pre_np[:, :, 0] == 2
    #headers_in_longshot = ((textregion_pre_np[:,:,0]==2) |
    #                       (textregion_pre_np[:,:,0]==1))
    textregion_pre_p[:, :, 0][headers_in_longshot &
                              (textregion_pre_p[:, :, 0] != 4)] = 2
    textregion_pre_p[:, :, 0][textregion_pre_p[:, :, 0] == 1] = 0
    # earlier it was so, but by this manner the drop capitals are also deleted
    # textregion_pre_p[:,:,0][(img_only_text[:,:]==1) &
    #                         (textregion_pre_p[:,:,0]!=7) &
    #                         (textregion_pre_p[:,:,0]!=2)] = 1
    textregion_pre_p[:, :, 0][(img_only_text[:, :] == 1) &
                              (textregion_pre_p[:, :, 0] != 7) &
                              (textregion_pre_p[:, :, 0] != 4) &
                              (textregion_pre_p[:, :, 0] != 2)] = 1
    return textregion_pre_p

def find_num_col_deskew(regions_without_separators, sigma_, multiplier=3.8):
    regions_without_separators_0 = regions_without_separators.sum(axis=1)
    z = gaussian_filter1d(regions_without_separators_0, sigma_)
    return np.std(z)

def find_num_col(
    regions_without_separators,
    num_col_classifier,
    tables,
    multiplier=3.8,
):
    if not regions_without_separators.any():
        return 0, []
    #plt.imshow(regions_without_separators)
    #plt.show()
    regions_without_separators_0 = regions_without_separators.sum(axis=0)
    ##plt.plot(regions_without_separators_0)
    ##plt.show()
    sigma_ = 35  # 70#35
    meda_n_updown = regions_without_separators_0[len(regions_without_separators_0) :: -1]
    first_nonzero = next((i for i, x in enumerate(regions_without_separators_0) if x), 0)
    last_nonzero = next((i for i, x in enumerate(meda_n_updown) if x), 0)
    last_nonzero = len(regions_without_separators_0) - last_nonzero
    y = regions_without_separators_0  # [first_nonzero:last_nonzero]
    y_help = np.zeros(len(y) + 20)
    y_help[10 : len(y) + 10] = y
    x = np.arange(len(y))
    zneg_rev = -y_help + np.max(y_help)
    zneg = np.zeros(len(zneg_rev) + 20)
    zneg[10 : len(zneg_rev) + 10] = zneg_rev
    z = gaussian_filter1d(y, sigma_)
    zneg = gaussian_filter1d(zneg, sigma_)

    peaks_neg, _ = find_peaks(zneg, height=0)
    #plt.plot(zneg)
    #plt.plot(peaks_neg, zneg[peaks_neg], 'rx')
    #plt.show()
    peaks, _ = find_peaks(z, height=0)
    peaks_neg = peaks_neg - 10 - 10

    last_nonzero = last_nonzero - 100
    first_nonzero = first_nonzero + 200

    peaks_neg = peaks_neg[(peaks_neg > first_nonzero) &
                          (peaks_neg < last_nonzero)]
    peaks = peaks[(peaks > 0.06 * regions_without_separators.shape[1]) &
                  (peaks < 0.94 * regions_without_separators.shape[1])]
    peaks_neg = peaks_neg[(peaks_neg > 370) &
                          (peaks_neg < (regions_without_separators.shape[1] - 370))]
    interest_pos = z[peaks]
    interest_pos = interest_pos[interest_pos > 10]
    if not interest_pos.any():
        return 0, []
    # plt.plot(z)
    # plt.show()
    interest_neg = z[peaks_neg]
    if not interest_neg.any():
        return 0, []

    min_peaks_pos = np.min(interest_pos)
    max_peaks_pos = np.max(interest_pos)

    if max_peaks_pos / min_peaks_pos >= 35:
        min_peaks_pos = np.mean(interest_pos)

    min_peaks_neg = 0  # np.min(interest_neg)

    # print(np.min(interest_pos),np.max(interest_pos),np.max(interest_pos)/np.min(interest_pos),'minmax')
    dis_talaei = (min_peaks_pos - min_peaks_neg) / multiplier
    grenze = min_peaks_pos - dis_talaei
    # np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

    # print(interest_neg,'interest_neg')
    # print(grenze,'grenze')
    # print(min_peaks_pos,'min_peaks_pos')
    # print(dis_talaei,'dis_talaei')
    # print(peaks_neg,'peaks_neg')

    interest_neg_fin = interest_neg[(interest_neg < grenze)]
    peaks_neg_fin = peaks_neg[(interest_neg < grenze)]
    # interest_neg_fin=interest_neg[(interest_neg<grenze)]

    if not tables:
        if ( num_col_classifier - ( (len(interest_neg_fin))+1 ) ) >= 3:
            index_sort_interest_neg_fin= np.argsort(interest_neg_fin)
            peaks_neg_sorted = np.array(peaks_neg)[index_sort_interest_neg_fin]
            interest_neg_fin_sorted = np.array(interest_neg_fin)[index_sort_interest_neg_fin]

            if len(index_sort_interest_neg_fin)>=num_col_classifier:
                peaks_neg_fin = list( peaks_neg_sorted[:num_col_classifier] )
                interest_neg_fin = list( interest_neg_fin_sorted[:num_col_classifier] )
            else:
                peaks_neg_fin = peaks_neg[:]
                interest_neg_fin = interest_neg[:]

    num_col = (len(interest_neg_fin)) + 1

    # print(peaks_neg_fin,'peaks_neg_fin')
    # print(num_col,'diz')
    p_l = 0
    p_u = len(y) - 1
    p_m = int(len(y) / 2.0)
    p_g_l = int(len(y) / 4.0)
    p_g_u = len(y) - int(len(y) / 4.0)

    if num_col == 3:
        if ((peaks_neg_fin[0] > p_g_u and
             peaks_neg_fin[1] > p_g_u) or
            (peaks_neg_fin[0] < p_g_l and
             peaks_neg_fin[1] < p_g_l) or
            (peaks_neg_fin[0] + 200 < p_m and
             peaks_neg_fin[1] < p_m) or
            (peaks_neg_fin[0] - 200 > p_m and
             peaks_neg_fin[1] > p_m)):
            num_col = 1
            peaks_neg_fin = []

    if num_col == 2:
        if (peaks_neg_fin[0] > p_g_u or
            peaks_neg_fin[0] < p_g_l):
            num_col = 1
            peaks_neg_fin = []

    ##print(len(peaks_neg_fin))

    diff_peaks = np.abs(np.diff(peaks_neg_fin))
    cut_off = 400
    peaks_neg_true = []
    forest = []

    # print(len(peaks_neg_fin),'len_')

    for i in range(len(peaks_neg_fin)):
        if i == 0:
            forest.append(peaks_neg_fin[i])
        if i < len(peaks_neg_fin) - 1:
            if diff_peaks[i] <= cut_off:
                forest.append(peaks_neg_fin[i + 1])
            if diff_peaks[i] > cut_off:
                # print(forest[np.argmin(z[forest]) ] )
                if not isNaN(forest[np.argmin(z[forest])]):
                    peaks_neg_true.append(forest[np.argmin(z[forest])])
                forest = []
                forest.append(peaks_neg_fin[i + 1])
        if i == (len(peaks_neg_fin) - 1):
            # print(print(forest[np.argmin(z[forest]) ] ))
            if not isNaN(forest[np.argmin(z[forest])]):
                peaks_neg_true.append(forest[np.argmin(z[forest])])

    num_col = len(peaks_neg_true) + 1
    p_l = 0
    p_u = len(y) - 1
    p_m = int(len(y) / 2.0)
    p_quarter = int(len(y) / 5.0)
    p_g_l = int(len(y) / 4.0)
    p_g_u = len(y) - int(len(y) / 4.0)

    p_u_quarter = len(y) - p_quarter

    ##print(num_col,'early')
    if num_col == 3:
        if ((peaks_neg_true[0] > p_g_u and
             peaks_neg_true[1] > p_g_u) or
            (peaks_neg_true[0] < p_g_l and
             peaks_neg_true[1] < p_g_l) or
            (peaks_neg_true[0] < p_m and
             peaks_neg_true[1] + 200 < p_m) or
            (peaks_neg_true[0] - 200 > p_m and
             peaks_neg_true[1] > p_m)):
            num_col = 1
            peaks_neg_true = []
        elif (peaks_neg_true[0] < p_g_u and
              peaks_neg_true[0] > p_g_l and
              peaks_neg_true[1] > p_u_quarter):
            peaks_neg_true = [peaks_neg_true[0]]
        elif (peaks_neg_true[1] < p_g_u and
              peaks_neg_true[1] > p_g_l and
              peaks_neg_true[0] < p_quarter):
            peaks_neg_true = [peaks_neg_true[1]]

    if num_col == 2:
        if (peaks_neg_true[0] > p_g_u or
            peaks_neg_true[0] < p_g_l):
            num_col = 1
            peaks_neg_true = []

    diff_peaks_abnormal = diff_peaks[diff_peaks < 360]

    if len(diff_peaks_abnormal) > 0:
        arg_help = np.arange(len(diff_peaks))
        arg_help_ann = arg_help[diff_peaks < 360]

        peaks_neg_fin_new = []

        for ii in range(len(peaks_neg_fin)):
            if ii in arg_help_ann:
                arg_min = np.argmin([interest_neg_fin[ii], interest_neg_fin[ii + 1]])
                if arg_min == 0:
                    peaks_neg_fin_new.append(peaks_neg_fin[ii])
                else:
                    peaks_neg_fin_new.append(peaks_neg_fin[ii + 1])

            elif (ii - 1) not in arg_help_ann:
                peaks_neg_fin_new.append(peaks_neg_fin[ii])
    else:
        peaks_neg_fin_new = peaks_neg_fin

    # plt.plot(gaussian_filter1d(y, sigma_))
    # plt.plot(peaks_neg_true,z[peaks_neg_true],'*')
    # plt.plot([0,len(y)], [grenze,grenze])
    # plt.show()
    ##print(len(peaks_neg_true))
    return len(peaks_neg_true), peaks_neg_true

def find_num_col_only_image(regions_without_separators, multiplier=3.8):
    regions_without_separators_0 = regions_without_separators[:, :].sum(axis=0)

    ##plt.plot(regions_without_separators_0)
    ##plt.show()
    sigma_ = 15

    meda_n_updown = regions_without_separators_0[len(regions_without_separators_0) :: -1]

    first_nonzero = next((i for i, x in enumerate(regions_without_separators_0) if x), 0)
    last_nonzero = next((i for i, x in enumerate(meda_n_updown) if x), 0)

    last_nonzero = len(regions_without_separators_0) - last_nonzero

    y = regions_without_separators_0  # [first_nonzero:last_nonzero]
    y_help = np.zeros(len(y) + 20)
    y_help[10 : len(y) + 10] = y
    x = np.arange(len(y))

    zneg_rev = -y_help + np.max(y_help)
    zneg = np.zeros(len(zneg_rev) + 20)
    zneg[10 : len(zneg_rev) + 10] = zneg_rev
    z = gaussian_filter1d(y, sigma_)
    zneg = gaussian_filter1d(zneg, sigma_)

    peaks_neg, _ = find_peaks(zneg, height=0)
    peaks, _ = find_peaks(z, height=0)
    peaks_neg = peaks_neg - 10 - 10
    peaks_neg_org = np.copy(peaks_neg)
    peaks_neg = peaks_neg[(peaks_neg > first_nonzero) &
                          (peaks_neg < last_nonzero)]
    peaks = peaks[(peaks > 0.09 * regions_without_separators.shape[1]) &
                  (peaks < 0.91 * regions_without_separators.shape[1])]

    peaks_neg = peaks_neg[(peaks_neg > 500) & (peaks_neg < (regions_without_separators.shape[1] - 500))]
    # print(peaks)
    interest_pos = z[peaks]

    interest_pos = interest_pos[interest_pos > 10]

    interest_neg = z[peaks_neg]
    min_peaks_pos = np.mean(interest_pos)  # np.min(interest_pos)
    min_peaks_neg = 0  # np.min(interest_neg)

    # $print(min_peaks_pos)
    dis_talaei = (min_peaks_pos - min_peaks_neg) / multiplier
    # print(interest_pos)
    grenze = min_peaks_pos - dis_talaei
    # np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

    interest_neg_fin = interest_neg[(interest_neg < grenze)]
    peaks_neg_fin = peaks_neg[(interest_neg < grenze)]

    num_col = (len(interest_neg_fin)) + 1

    p_l = 0
    p_u = len(y) - 1
    p_m = int(len(y) / 2.0)
    p_g_l = int(len(y) / 3.0)
    p_g_u = len(y) - int(len(y) / 3.0)

    if num_col == 3:
        if ((peaks_neg_fin[0] > p_g_u and
             peaks_neg_fin[1] > p_g_u) or
            (peaks_neg_fin[0] < p_g_l and
             peaks_neg_fin[1] < p_g_l) or
            (peaks_neg_fin[0] < p_m and
             peaks_neg_fin[1] < p_m) or
            (peaks_neg_fin[0] > p_m and
             peaks_neg_fin[1] > p_m)):
            num_col = 1
        else:
            pass

    if num_col == 2:
        if (peaks_neg_fin[0] > p_g_u or
            peaks_neg_fin[0] < p_g_l):
            num_col = 1
        else:
            pass

    diff_peaks = np.abs(np.diff(peaks_neg_fin))

    cut_off = 400
    peaks_neg_true = []
    forest = []

    for i in range(len(peaks_neg_fin)):
        if i == 0:
            forest.append(peaks_neg_fin[i])
        if i < (len(peaks_neg_fin) - 1):
            if diff_peaks[i] <= cut_off:
                forest.append(peaks_neg_fin[i + 1])
            if diff_peaks[i] > cut_off:
                # print(forest[np.argmin(z[forest]) ] )
                if not isNaN(forest[np.argmin(z[forest])]):
                    peaks_neg_true.append(forest[np.argmin(z[forest])])
                forest = []
                forest.append(peaks_neg_fin[i + 1])
        if i == (len(peaks_neg_fin) - 1):
            # print(print(forest[np.argmin(z[forest]) ] ))
            if not isNaN(forest[np.argmin(z[forest])]):
                peaks_neg_true.append(forest[np.argmin(z[forest])])

    num_col = (len(peaks_neg_true)) + 1
    p_l = 0
    p_u = len(y) - 1
    p_m = int(len(y) / 2.0)
    p_quarter = int(len(y) / 4.0)
    p_g_l = int(len(y) / 3.0)
    p_g_u = len(y) - int(len(y) / 3.0)

    p_u_quarter = len(y) - p_quarter

    if num_col == 3:
        if ((peaks_neg_true[0] > p_g_u and
             peaks_neg_true[1] > p_g_u) or
            (peaks_neg_true[0] < p_g_l and
             peaks_neg_true[1] < p_g_l) or
            (peaks_neg_true[0] < p_m and
             peaks_neg_true[1] < p_m) or
            (peaks_neg_true[0] > p_m and
             peaks_neg_true[1] > p_m)):
            num_col = 1
            peaks_neg_true = []
        elif (peaks_neg_true[0] < p_g_u and
              peaks_neg_true[0] > p_g_l and
              peaks_neg_true[1] > p_u_quarter):
            peaks_neg_true = [peaks_neg_true[0]]
        elif (peaks_neg_true[1] < p_g_u and
              peaks_neg_true[1] > p_g_l and
              peaks_neg_true[0] < p_quarter):
            peaks_neg_true = [peaks_neg_true[1]]
        else:
            pass

    if num_col == 2:
        if (peaks_neg_true[0] > p_g_u or
            peaks_neg_true[0] < p_g_l):
            num_col = 1
            peaks_neg_true = []

    if num_col == 4:
        if (len(np.array(peaks_neg_true)[np.array(peaks_neg_true) < p_g_l]) == 2 or
            len(np.array(peaks_neg_true)[np.array(peaks_neg_true) > (len(y) - p_g_l)]) == 2):
            num_col = 1
            peaks_neg_true = []
        else:
            pass

    # no deeper hill around found hills

    peaks_fin_true = []
    for i in range(len(peaks_neg_true)):
        hill_main = peaks_neg_true[i]
        # deep_depth=z[peaks_neg]
        hills_around = peaks_neg_org[((peaks_neg_org > hill_main) &
                                      (peaks_neg_org <= hill_main + 400)) |
                                     ((peaks_neg_org < hill_main) &
                                      (peaks_neg_org >= hill_main - 400))]
        deep_depth_around = z[hills_around]

        # print(hill_main,z[hill_main],hills_around,deep_depth_around,'manoooo')
        try:
            if np.min(deep_depth_around) < z[hill_main]:
                pass
            else:
                peaks_fin_true.append(hill_main)
        except:
            pass

    diff_peaks_annormal = diff_peaks[diff_peaks < 360]
    if len(diff_peaks_annormal) > 0:
        arg_help = np.arange(len(diff_peaks))
        arg_help_ann = arg_help[diff_peaks < 360]

        peaks_neg_fin_new = []
        for ii in range(len(peaks_neg_fin)):
            if ii in arg_help_ann:
                arg_min = np.argmin([interest_neg_fin[ii], interest_neg_fin[ii + 1]])
                if arg_min == 0:
                    peaks_neg_fin_new.append(peaks_neg_fin[ii])
                else:
                    peaks_neg_fin_new.append(peaks_neg_fin[ii + 1])
            elif (ii - 1) in arg_help_ann:
                pass
            else:
                peaks_neg_fin_new.append(peaks_neg_fin[ii])
    else:
        peaks_neg_fin_new = peaks_neg_fin

    # sometime pages with one columns gives also some negative peaks. delete those peaks
    param = z[peaks_neg_true] / float(min_peaks_pos) * 100
    if len(param[param <= 41]) == 0:
        peaks_neg_true = []

    return len(peaks_fin_true), peaks_fin_true

def find_num_col_by_vertical_lines(regions_without_separators, multiplier=3.8):
    regions_without_separators_0 = regions_without_separators.sum(axis=0)

    ##plt.plot(regions_without_separators_0)
    ##plt.show()
    sigma_ = 35  # 70#35

    z = gaussian_filter1d(regions_without_separators_0, sigma_)
    peaks, _ = find_peaks(z, height=0)

    # print(peaks,'peaksnew')
    return peaks

def return_regions_without_separators(regions_pre):
    kernel = np.ones((5, 5), np.uint8)
    regions_without_separators = ((regions_pre[:, :] != 6) &
                                  (regions_pre[:, :] != 0))
    # regions_without_separators=( (image_regions_eraly_p[:,:,:]!=6) &
    #                              (image_regions_eraly_p[:,:,:]!=0) &
    #                              (image_regions_eraly_p[:,:,:]!=5) &
    #                              (image_regions_eraly_p[:,:,:]!=8) &
    #                              (image_regions_eraly_p[:,:,:]!=7))

    regions_without_separators = cv2.erode(regions_without_separators.astype(np.uint8), kernel, iterations=6)

    return regions_without_separators

def put_drop_out_from_only_drop_model(layout_no_patch, layout1):
    if layout_no_patch.ndim == 3:
        layout_no_patch = layout_no_patch[:, :, 0]

    drop_only = (layout_no_patch[:, :] == 4) * 1
    contours_drop, hir_on_drop = return_contours_of_image(drop_only)
    contours_drop_parent = return_parent_contours(contours_drop, hir_on_drop)

    areas_cnt_text = np.array([cv2.contourArea(contours_drop_parent[j])
                               for j in range(len(contours_drop_parent))])
    areas_cnt_text = areas_cnt_text / float(drop_only.shape[0] * drop_only.shape[1])
    contours_drop_parent = [contours_drop_parent[jz]
                            for jz in range(len(contours_drop_parent))
                            if areas_cnt_text[jz] > 0.00001]
    areas_cnt_text = [areas_cnt_text[jz]
                      for jz in range(len(areas_cnt_text))
                      if areas_cnt_text[jz] > 0.00001]

    contours_drop_parent_final = []
    for jj in range(len(contours_drop_parent)):
        x, y, w, h = cv2.boundingRect(contours_drop_parent[jj])
        # boxes.append([int(x), int(y), int(w), int(h)])

        map_of_drop_contour_bb = np.zeros((layout1.shape[0], layout1.shape[1]))
        map_of_drop_contour_bb[y : y + h, x : x + w] = layout1[y : y + h, x : x + w]
        if (100. *
            (map_of_drop_contour_bb == 1).sum() /
            (map_of_drop_contour_bb == 5).sum()) >= 15:
            contours_drop_parent_final.append(contours_drop_parent[jj])

    layout_no_patch[:, :][layout_no_patch[:, :] == 4] = 0
    layout_no_patch = cv2.fillPoly(layout_no_patch, pts=contours_drop_parent_final, color=4)

    return layout_no_patch

def putt_bb_of_drop_capitals_of_model_in_patches_in_layout(layout_in_patch, drop_capital_label, text_regions_p):
    drop_only = (layout_in_patch[:, :, 0] == drop_capital_label) * 1
    contours_drop, hir_on_drop = return_contours_of_image(drop_only)
    contours_drop_parent = return_parent_contours(contours_drop, hir_on_drop)

    areas_cnt_text = np.array([cv2.contourArea(contours_drop_parent[j])
                               for j in range(len(contours_drop_parent))])
    areas_cnt_text = areas_cnt_text / float(drop_only.shape[0] * drop_only.shape[1])
    contours_drop_parent = [contours_drop_parent[jz]
                            for jz in range(len(contours_drop_parent))
                            if areas_cnt_text[jz] > 0.00001]
    areas_cnt_text = [areas_cnt_text[jz]
                      for jz in range(len(areas_cnt_text))
                      if areas_cnt_text[jz] > 0.00001]

    contours_drop_parent_final = []
    for jj in range(len(contours_drop_parent)):
        x, y, w, h = cv2.boundingRect(contours_drop_parent[jj])
        box = slice(y, y + h), slice(x, x + w)
        box0 = box + (0,)
        mask_of_drop_cpaital_in_early_layout = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1]))
        mask_of_drop_cpaital_in_early_layout[box] = text_regions_p[box]

        all_drop_capital_pixels_which_is_text_in_early_lo = np.sum(mask_of_drop_cpaital_in_early_layout[box]==1)
        mask_of_drop_cpaital_in_early_layout[box] = 1
        all_drop_capital_pixels = np.sum(mask_of_drop_cpaital_in_early_layout==1)

        percent_text_to_all_in_drop = all_drop_capital_pixels_which_is_text_in_early_lo / float(all_drop_capital_pixels)
        if (areas_cnt_text[jj] * float(drop_only.shape[0] * drop_only.shape[1]) / float(w * h) > 0.6 and
            percent_text_to_all_in_drop >= 0.3):
            layout_in_patch[box0] = drop_capital_label
        else:
            layout_in_patch[box0][layout_in_patch[box0] == drop_capital_label] = drop_capital_label
            layout_in_patch[box0][layout_in_patch[box0] == 0] = drop_capital_label
            layout_in_patch[box0][layout_in_patch[box0] == 4] = drop_capital_label# images
            #layout_in_patch[box0][layout_in_patch[box0] == drop_capital_label] = 1#drop_capital_label

    return layout_in_patch

def check_any_text_region_in_model_one_is_main_or_header(
        regions_model_1, regions_model_full,
        contours_only_text_parent,
        all_box_coord, all_found_textline_polygons,
        slopes,
        contours_only_text_parent_d_ordered, conf_contours):

    cx_main, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, y_corr_x_min_from_argmin = \
        find_new_features_of_contours(contours_only_text_parent)

    length_con=x_max_main-x_min_main
    height_con=y_max_main-y_min_main

    all_found_textline_polygons_main=[]
    all_found_textline_polygons_head=[]

    all_box_coord_main=[]
    all_box_coord_head=[]

    slopes_main=[]
    slopes_head=[]

    contours_only_text_parent_main=[]
    contours_only_text_parent_head=[]

    conf_contours_main=[]
    conf_contours_head=[]

    contours_only_text_parent_main_d=[]
    contours_only_text_parent_head_d=[]

    for ii, con in enumerate(contours_only_text_parent):
        img = np.zeros(regions_model_1.shape[:2])
        img = cv2.fillPoly(img, pts=[con], color=255)

        all_pixels=((img == 255)*1).sum()
        pixels_header=( ( (img == 255) & (regions_model_full[:,:,0]==2) )*1 ).sum()
        pixels_main=all_pixels-pixels_header

        if (pixels_header>=pixels_main) and ( (length_con[ii]/float(height_con[ii]) )>=1.3 ):
            regions_model_1[:,:][(regions_model_1[:,:]==1) & (img == 255) ]=2
            contours_only_text_parent_head.append(con)
            if len(contours_only_text_parent_d_ordered):
                contours_only_text_parent_head_d.append(contours_only_text_parent_d_ordered[ii])
            all_box_coord_head.append(all_box_coord[ii])
            slopes_head.append(slopes[ii])
            all_found_textline_polygons_head.append(all_found_textline_polygons[ii])
            conf_contours_head.append(None)
        else:
            regions_model_1[:,:][(regions_model_1[:,:]==1) & (img == 255) ]=1
            contours_only_text_parent_main.append(con)
            conf_contours_main.append(conf_contours[ii])
            if len(contours_only_text_parent_d_ordered):
                contours_only_text_parent_main_d.append(contours_only_text_parent_d_ordered[ii])
            all_box_coord_main.append(all_box_coord[ii])
            slopes_main.append(slopes[ii])
            all_found_textline_polygons_main.append(all_found_textline_polygons[ii])

        #print(all_pixels,pixels_main,pixels_header)

    return (regions_model_1,
            contours_only_text_parent_main,
            contours_only_text_parent_head,
            all_box_coord_main,
            all_box_coord_head,
            all_found_textline_polygons_main,
            all_found_textline_polygons_head,
            slopes_main,
            slopes_head,
            contours_only_text_parent_main_d,
            contours_only_text_parent_head_d,
            conf_contours_main,
            conf_contours_head)

def check_any_text_region_in_model_one_is_main_or_header_light(
        regions_model_1, regions_model_full,
        contours_only_text_parent,
        all_box_coord, all_found_textline_polygons,
        slopes,
        contours_only_text_parent_d_ordered,
        conf_contours):

    ### to make it faster
    h_o = regions_model_1.shape[0]
    w_o = regions_model_1.shape[1]
    zoom = 3
    regions_model_1 = cv2.resize(regions_model_1, (regions_model_1.shape[1] // zoom,
                                                   regions_model_1.shape[0] // zoom),
                                 interpolation=cv2.INTER_NEAREST)
    regions_model_full = cv2.resize(regions_model_full, (regions_model_full.shape[1] // zoom,
                                                         regions_model_full.shape[0] // zoom),
                                    interpolation=cv2.INTER_NEAREST)
    contours_only_text_parent_z = [(cnt / zoom).astype(int) for cnt in contours_only_text_parent]

    ###
    cx_main, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, y_corr_x_min_from_argmin = \
        find_new_features_of_contours(contours_only_text_parent_z)

    length_con=x_max_main-x_min_main
    height_con=y_max_main-y_min_main

    all_found_textline_polygons_main=[]
    all_found_textline_polygons_head=[]

    all_box_coord_main=[]
    all_box_coord_head=[]

    slopes_main=[]
    slopes_head=[]

    contours_only_text_parent_main=[]
    contours_only_text_parent_head=[]

    conf_contours_main=[]
    conf_contours_head=[]

    contours_only_text_parent_main_d=[]
    contours_only_text_parent_head_d=[]

    for ii, con in enumerate(contours_only_text_parent_z):
        img = np.zeros(regions_model_1.shape[:2])
        img = cv2.fillPoly(img, pts=[con], color=255)

        all_pixels = (img == 255).sum()
        pixels_header=((img == 255) &
                       (regions_model_full[:,:,0]==2)).sum()
        pixels_main = all_pixels - pixels_header

        if (( pixels_header / float(pixels_main) >= 0.6 and
              length_con[ii] / float(height_con[ii]) >= 1.3 and
              length_con[ii] / float(height_con[ii]) <= 3 ) or
            ( pixels_header / float(pixels_main) >= 0.3 and
              length_con[ii] / float(height_con[ii]) >=3 )):

            regions_model_1[:,:][(regions_model_1[:,:]==1) & (img == 255) ] = 2
            contours_only_text_parent_head.append(contours_only_text_parent[ii])
            conf_contours_head.append(None) # why not conf_contours[ii], too?
            if len(contours_only_text_parent_d_ordered):
                contours_only_text_parent_head_d.append(contours_only_text_parent_d_ordered[ii])
            all_box_coord_head.append(all_box_coord[ii])
            slopes_head.append(slopes[ii])
            all_found_textline_polygons_head.append(all_found_textline_polygons[ii])

        else:
            regions_model_1[:,:][(regions_model_1[:,:]==1) & (img == 255) ] = 1
            contours_only_text_parent_main.append(contours_only_text_parent[ii])
            conf_contours_main.append(conf_contours[ii])
            if len(contours_only_text_parent_d_ordered):
                contours_only_text_parent_main_d.append(contours_only_text_parent_d_ordered[ii])
            all_box_coord_main.append(all_box_coord[ii])
            slopes_main.append(slopes[ii])
            all_found_textline_polygons_main.append(all_found_textline_polygons[ii])
        #print(all_pixels,pixels_main,pixels_header)

    ### to make it faster
    regions_model_1 = cv2.resize(regions_model_1, (w_o, h_o), interpolation=cv2.INTER_NEAREST)
    # regions_model_full = cv2.resize(img, (regions_model_full.shape[1] // zoom,
    #                                       regions_model_full.shape[0] // zoom),
    #                                 interpolation=cv2.INTER_NEAREST)
    ###

    return (regions_model_1,
            contours_only_text_parent_main,
            contours_only_text_parent_head,
            all_box_coord_main,
            all_box_coord_head,
            all_found_textline_polygons_main,
            all_found_textline_polygons_head,
            slopes_main,
            slopes_head,
            contours_only_text_parent_main_d,
            contours_only_text_parent_head_d,
            conf_contours_main,
            conf_contours_head)

def small_textlines_to_parent_adherence2(textlines_con, textline_iamge, num_col):
    # print(textlines_con)
    # textlines_con=textlines_con.astype(np.uint32)
    textlines_con_changed = []
    for m1 in range(len(textlines_con)):

        # textlines_tot=textlines_con[m1]
        # textlines_tot=textlines_tot.astype()
        textlines_tot = []
        textlines_tot_org_form = []
        # print(textlines_tot)

        for nn in range(len(textlines_con[m1])):
            textlines_tot.append(np.array(textlines_con[m1][nn], dtype=np.int32))
            textlines_tot_org_form.append(textlines_con[m1][nn])

        ##img_text_all=np.zeros((textline_iamge.shape[0],textline_iamge.shape[1]))
        ##img_text_all=cv2.fillPoly(img_text_all, pts =textlines_tot , color=(1,1,1))

        ##plt.imshow(img_text_all)
        ##plt.show()
        areas_cnt_text = np.array([cv2.contourArea(textlines_tot[j])
                                   for j in range(len(textlines_tot))])
        areas_cnt_text = areas_cnt_text / float(textline_iamge.shape[0] * textline_iamge.shape[1])
        indexes_textlines = np.arange(len(textlines_tot))

        # print(areas_cnt_text,np.min(areas_cnt_text),np.max(areas_cnt_text))
        if num_col == 0:
            min_area = 0.0004
        elif num_col == 1:
            min_area = 0.0003
        else:
            min_area = 0.0001
        indexes_textlines_small = indexes_textlines[areas_cnt_text < min_area]

        # print(indexes_textlines)

        textlines_small = []
        textlines_small_org_form = []
        for i in indexes_textlines_small:
            textlines_small.append(textlines_tot[i])
            textlines_small_org_form.append(textlines_tot_org_form[i])

        textlines_big = []
        textlines_big_org_form = []
        for i in list(set(indexes_textlines) - set(indexes_textlines_small)):
            textlines_big.append(textlines_tot[i])
            textlines_big_org_form.append(textlines_tot_org_form[i])

        img_textline_s = np.zeros(textline_iamge.shape[:2])
        img_textline_s = cv2.fillPoly(img_textline_s, pts=textlines_small, color=1)

        img_textline_b = np.zeros(textline_iamge.shape[:2])
        img_textline_b = cv2.fillPoly(img_textline_b, pts=textlines_big, color=1)

        sum_small_big_all = img_textline_s + img_textline_b
        sum_small_big_all2 = (sum_small_big_all[:, :] == 2) * 1

        sum_intersection_sb = sum_small_big_all2.sum(axis=1).sum()
        if sum_intersection_sb > 0:
            dis_small_from_bigs_tot = []
            for z1 in range(len(textlines_small)):
                # print(len(textlines_small),'small')
                intersections = []
                for z2 in range(len(textlines_big)):
                    img_text = np.zeros(textline_iamge.shape[:2])
                    img_text = cv2.fillPoly(img_text, pts=[textlines_small[z1]], color=1)

                    img_text2 = np.zeros(textline_iamge.shape[:2])
                    img_text2 = cv2.fillPoly(img_text2, pts=[textlines_big[z2]], color=1)

                    sum_small_big = img_text2 + img_text
                    sum_small_big_2 = (sum_small_big[:, :] == 2) * 1

                    sum_intersection = sum_small_big_2.sum(axis=1).sum()
                    # print(sum_intersection)
                    intersections.append(sum_intersection)

                if len(np.array(intersections)[np.array(intersections) > 0]) == 0:
                    intersections = []
                try:
                    dis_small_from_bigs_tot.append(np.argmax(intersections))
                except:
                    dis_small_from_bigs_tot.append(-1)

            smalls_list = np.array(dis_small_from_bigs_tot)[np.array(dis_small_from_bigs_tot) >= 0]
            # index_small_textlines_rest=list( set(indexes_textlines_small)-set(smalls_list) )

            textlines_big_with_change = []
            textlines_big_with_change_con = []
            textlines_small_with_change = []
            for z in list(set(smalls_list)):
                index_small_textlines = list(np.where(np.array(dis_small_from_bigs_tot) == z)[0])
                # print(z,index_small_textlines)

                img_text2 = np.zeros(textline_iamge.shape[:2], dtype=np.uint8)
                img_text2 = cv2.fillPoly(img_text2, pts=[textlines_big[z]], color=255)

                textlines_big_with_change.append(z)

                for k in index_small_textlines:
                    img_text2 = cv2.fillPoly(img_text2, pts=[textlines_small[k]], color=255)
                    textlines_small_with_change.append(k)

                _, thresh = cv2.threshold(img_text2, 0, 255, 0)
                cont, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # print(cont[0],type(cont))
                textlines_big_with_change_con.append(cont)
                textlines_big_org_form[z] = cont[0]

                # plt.imshow(img_text2)
                # plt.show()

            # print(textlines_big_with_change,'textlines_big_with_change')
            # print(textlines_small_with_change,'textlines_small_with_change')
            # print(textlines_big)

        textlines_con_changed.append(textlines_big_org_form)
    return textlines_con_changed

def order_of_regions(textline_mask, contours_main, contours_head, y_ref):
    ##plt.imshow(textline_mask)
    ##plt.show()
    y = textline_mask.sum(axis=1) # horizontal projection profile
    y_padded = np.zeros(len(y) + 40)
    y_padded[20 : len(y) + 20] = y

    sigma_gaus = 8
    #z = gaussian_filter1d(y_padded, sigma_gaus)
    #peaks, _ = find_peaks(z, height=0)
    #peaks = peaks - 20
    zneg_rev = np.max(y_padded) - y_padded
    zneg = np.zeros(len(zneg_rev) + 40)
    zneg[20 : len(zneg_rev) + 20] = zneg_rev
    zneg = gaussian_filter1d(zneg, sigma_gaus)

    peaks_neg, _ = find_peaks(zneg, height=0)
    peaks_neg = peaks_neg - 20 - 20

    ##plt.plot(z)
    ##plt.show()
    cx_main, cy_main = find_center_of_contours(contours_main)
    cx_head, cy_head = find_center_of_contours(contours_head)

    peaks_neg_new = np.append(np.insert(peaks_neg, 0, 0), textline_mask.shape[0])
    # offset from bbox of mask
    peaks_neg_new += y_ref

    # assert not len(cy_main) or np.min(peaks_neg_new) <= np.min(cy_main) and np.max(cy_main) <= np.max(peaks_neg_new)
    # assert not len(cy_head) or np.min(peaks_neg_new) <= np.min(cy_head) and np.max(cy_head) <= np.max(peaks_neg_new)

    matrix_of_orders = np.zeros((len(contours_main) + len(contours_head), 5), dtype=int)
    matrix_of_orders[:, 0] = np.arange(len(contours_main) + len(contours_head))
    matrix_of_orders[: len(contours_main), 1] = 1
    matrix_of_orders[len(contours_main) :, 1] = 2
    matrix_of_orders[: len(contours_main), 2] = cx_main
    matrix_of_orders[len(contours_main) :, 2] = cx_head
    matrix_of_orders[: len(contours_main), 3] = cy_main
    matrix_of_orders[len(contours_main) :, 3] = cy_head
    matrix_of_orders[: len(contours_main), 4] = np.arange(len(contours_main))
    matrix_of_orders[len(contours_main) :, 4] = np.arange(len(contours_head))

    # print(peaks_neg_new,'peaks_neg_new')
    # print(matrix_of_orders,'matrix_of_orders')
    # print(peaks_neg_new,np.max(peaks_neg_new))
    final_indexers_sorted = []
    final_types = []
    final_index_type = []
    for top, bot in pairwise(peaks_neg_new):
        indexes_in, types_in, cxs_in, cys_in, typed_indexes_in = \
             matrix_of_orders[(matrix_of_orders[:, 3] >= top) &
                              (matrix_of_orders[:, 3] < bot)].T
        sorted_inside = np.argsort(cxs_in)
        final_indexers_sorted.extend(indexes_in[sorted_inside])
        final_types.extend(types_in[sorted_inside])
        final_index_type.extend(typed_indexes_in[sorted_inside])

    ##matrix_of_orders[:len_main,4]=final_indexers_sorted[:]

    # assert len(final_indexers_sorted) == len(contours_main) + len(contours_head)
    # assert not len(final_indexers_sorted) or max(final_index_type) == max(len(contours_main)

    return np.array(final_indexers_sorted), np.array(final_types), np.array(final_index_type)

def combine_hor_lines_and_delete_cross_points_and_get_lines_features_back_new(
        img_p_in_ver, img_in_hor,num_col_classifier):

    #img_p_in_ver = cv2.erode(img_p_in_ver, self.kernel, iterations=2)
    _, thresh = cv2.threshold(img_p_in_ver, 0, 255, 0)
    contours_lines_ver, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slope_lines_ver, _, x_min_main_ver, _, _, _, y_min_main_ver, y_max_main_ver, cx_main_ver = \
        find_features_of_lines(contours_lines_ver)
    for i in range(len(x_min_main_ver)):
        img_p_in_ver[int(y_min_main_ver[i]):
                     int(y_min_main_ver[i])+30,
                     int(cx_main_ver[i])-25:
                     int(cx_main_ver[i])+25] = 0
        img_p_in_ver[int(y_max_main_ver[i])-30:
                     int(y_max_main_ver[i]),
                     int(cx_main_ver[i])-25:
                     int(cx_main_ver[i])+25] = 0

    _, thresh = cv2.threshold(img_in_hor, 0, 255, 0)
    contours_lines_hor, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    slope_lines_hor, dist_x_hor, x_min_main_hor, x_max_main_hor, cy_main_hor, _, _, _, _ = \
        find_features_of_lines(contours_lines_hor)
    x_width_smaller_than_acolumn_width=img_in_hor.shape[1]/float(num_col_classifier+1.)

    len_lines_bigger_than_x_width_smaller_than_acolumn_width=len( dist_x_hor[dist_x_hor>=x_width_smaller_than_acolumn_width] )
    len_lines_bigger_than_x_width_smaller_than_acolumn_width_per_column=int(len_lines_bigger_than_x_width_smaller_than_acolumn_width /
                                                                            float(num_col_classifier))
    if len_lines_bigger_than_x_width_smaller_than_acolumn_width_per_column < 10:
        args_hor=np.arange(len(slope_lines_hor))
        all_args_uniq=contours_in_same_horizon(cy_main_hor)
        #print(all_args_uniq,'all_args_uniq')
        if len(all_args_uniq)>0:
            if type(all_args_uniq[0]) is list:
                special_separators=[]
                contours_new=[]
                for dd in range(len(all_args_uniq)):
                    merged_all=None
                    some_args=args_hor[all_args_uniq[dd]]
                    some_cy=cy_main_hor[all_args_uniq[dd]]
                    some_x_min=x_min_main_hor[all_args_uniq[dd]]
                    some_x_max=x_max_main_hor[all_args_uniq[dd]]

                    #img_in=np.zeros(separators_closeup_n[:,:,2].shape)
                    #print(img_p_in_ver.shape[1],some_x_max-some_x_min,'xdiff')
                    diff_x_some=some_x_max-some_x_min
                    for jv in range(len(some_args)):
                        img_p_in=cv2.fillPoly(img_in_hor, pts=[contours_lines_hor[some_args[jv]]], color=(1,1,1))
                        if any(i_diff>(img_p_in_ver.shape[1]/float(3.3)) for i_diff in diff_x_some):
                            img_p_in[int(np.mean(some_cy))-5:
                                     int(np.mean(some_cy))+5,
                                     int(np.min(some_x_min)):
                                     int(np.max(some_x_max)) ]=1
                    sum_dis=dist_x_hor[some_args].sum()
                    diff_max_min_uniques=np.max(x_max_main_hor[some_args])-np.min(x_min_main_hor[some_args])

                    if (diff_max_min_uniques > sum_dis and
                        sum_dis / float(diff_max_min_uniques) > 0.85 and
                        diff_max_min_uniques / float(img_p_in_ver.shape[1]) > 0.85 and
                        np.std(dist_x_hor[some_args]) < 0.55 * np.mean(dist_x_hor[some_args])):
                        # print(dist_x_hor[some_args],
                        #       dist_x_hor[some_args].sum(),
                        #       np.min(x_min_main_hor[some_args]),
                        #       np.max(x_max_main_hor[some_args]),'jalibdi')
                        # print(np.mean( dist_x_hor[some_args] ),
                        #       np.std( dist_x_hor[some_args] ),
                        #       np.var( dist_x_hor[some_args] ),'jalibdiha')
                        special_separators.append(np.mean(cy_main_hor[some_args]))
            else:
                img_p_in=img_in_hor
                special_separators=[]
        else:
            img_p_in=img_in_hor
            special_separators=[]

        img_p_in_ver[img_p_in_ver == 255] = 1
        sep_ver_hor = img_p_in + img_p_in_ver
        sep_ver_hor_cross = (sep_ver_hor == 2) * 1
        _, thresh = cv2.threshold(sep_ver_hor_cross.astype(np.uint8), 0, 255, 0)
        contours_cross, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        center_cross = np.array(find_center_of_contours(contours_cross), dtype=int)
        for cx, cy in center_cross.T:
            img_p_in[cy - 30: cy + 30, cx + 5: cx + 40] = 0
            img_p_in[cy - 30: cy + 30, cx - 40: cx - 4] = 0
    else:
        img_p_in=np.copy(img_in_hor)
        special_separators=[]
    return img_p_in, special_separators

def return_points_with_boundies(peaks_neg_fin, first_point, last_point):
    peaks_neg_tot = []
    peaks_neg_tot.append(first_point)
    for ii in range(len(peaks_neg_fin)):
        peaks_neg_tot.append(peaks_neg_fin[ii])
    peaks_neg_tot.append(last_point)
    return peaks_neg_tot

def find_number_of_columns_in_document(region_pre_p, num_col_classifier, tables, label_lines, contours_h=None):
    t_ins_c0 = time.time()
    separators_closeup=( (region_pre_p[:,:]==label_lines))*1
    separators_closeup[0:110,:]=0
    separators_closeup[separators_closeup.shape[0]-150:,:]=0

    kernel = np.ones((5,5),np.uint8)
    separators_closeup=separators_closeup.astype(np.uint8)
    separators_closeup = cv2.dilate(separators_closeup,kernel,iterations = 1)
    separators_closeup = cv2.erode(separators_closeup,kernel,iterations = 1)

    separators_closeup_new=np.zeros((separators_closeup.shape[0] ,separators_closeup.shape[1] ))
    separators_closeup_n=np.copy(separators_closeup)
    separators_closeup_n=separators_closeup_n.astype(np.uint8)

    separators_closeup_n_binary=np.zeros(( separators_closeup_n.shape[0],separators_closeup_n.shape[1]) )
    separators_closeup_n_binary[:,:]=separators_closeup_n[:,:]
    separators_closeup_n_binary[:,:][separators_closeup_n_binary[:,:]!=0]=1

    _, thresh_e = cv2.threshold(separators_closeup_n_binary, 0, 255, 0)
    contours_line_e, _ = cv2.findContours(thresh_e.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, dist_xe, _, _, _, _, y_min_main, y_max_main, _ = \
        find_features_of_lines(contours_line_e)
    dist_ye = y_max_main - y_min_main
    args_e=np.arange(len(contours_line_e))
    args_hor_e=args_e[(dist_ye<=50) &
                      (dist_xe>=3*dist_ye)]
    cnts_hor_e=[]
    for ce in args_hor_e:
        cnts_hor_e.append(contours_line_e[ce])

    separators_closeup_n_binary=cv2.fillPoly(separators_closeup_n_binary, pts=cnts_hor_e, color=0)
    gray = cv2.bitwise_not(separators_closeup_n_binary)
    gray=gray.astype(np.uint8)

    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                               cv2.THRESH_BINARY, 15, -2)
    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    kernel = np.ones((5,5),np.uint8)
    horizontal = cv2.dilate(horizontal,kernel,iterations = 2)
    horizontal = cv2.erode(horizontal,kernel,iterations = 2)
    horizontal = cv2.fillPoly(horizontal, pts=cnts_hor_e, color=255)

    rows = vertical.shape[0]
    verticalsize = rows // 30
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    vertical = cv2.dilate(vertical,kernel,iterations = 1)

    horizontal, special_separators = \
        combine_hor_lines_and_delete_cross_points_and_get_lines_features_back_new(
            vertical, horizontal, num_col_classifier)

    separators_closeup_new[:,:][vertical[:,:]!=0]=1
    separators_closeup_new[:,:][horizontal[:,:]!=0]=1

    _, thresh = cv2.threshold(vertical, 0, 255, 0)
    contours_line_vers, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slope_lines, dist_x, x_min_main, x_max_main, cy_main, slope_lines_org, y_min_main, y_max_main, cx_main = \
        find_features_of_lines(contours_line_vers)

    args=np.arange(len(slope_lines))
    args_ver=args[slope_lines==1]
    dist_x_ver=dist_x[slope_lines==1]
    y_min_main_ver=y_min_main[slope_lines==1]
    y_max_main_ver=y_max_main[slope_lines==1]
    x_min_main_ver=x_min_main[slope_lines==1]
    x_max_main_ver=x_max_main[slope_lines==1]
    cx_main_ver=cx_main[slope_lines==1]
    dist_y_ver=y_max_main_ver-y_min_main_ver
    len_y=separators_closeup.shape[0]/3.0

    _, thresh = cv2.threshold(horizontal, 0, 255, 0)
    contours_line_hors, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slope_lines, dist_x, x_min_main, x_max_main, cy_main, slope_lines_org, y_min_main, y_max_main, cx_main = \
        find_features_of_lines(contours_line_hors)

    slope_lines_org_hor=slope_lines_org[slope_lines==0]
    args=np.arange(len(slope_lines))
    len_x=separators_closeup.shape[1]/5.0
    dist_y=np.abs(y_max_main-y_min_main)

    args_hor=args[slope_lines==0]
    dist_x_hor=dist_x[slope_lines==0]
    y_min_main_hor=y_min_main[slope_lines==0]
    y_max_main_hor=y_max_main[slope_lines==0]
    x_min_main_hor=x_min_main[slope_lines==0]
    x_max_main_hor=x_max_main[slope_lines==0]
    dist_y_hor=dist_y[slope_lines==0]
    cy_main_hor=cy_main[slope_lines==0]

    args_hor=args_hor[dist_x_hor>=len_x/2.0]
    x_max_main_hor=x_max_main_hor[dist_x_hor>=len_x/2.0]
    x_min_main_hor=x_min_main_hor[dist_x_hor>=len_x/2.0]
    cy_main_hor=cy_main_hor[dist_x_hor>=len_x/2.0]
    y_min_main_hor=y_min_main_hor[dist_x_hor>=len_x/2.0]
    y_max_main_hor=y_max_main_hor[dist_x_hor>=len_x/2.0]
    dist_y_hor=dist_y_hor[dist_x_hor>=len_x/2.0]
    slope_lines_org_hor=slope_lines_org_hor[dist_x_hor>=len_x/2.0]
    dist_x_hor=dist_x_hor[dist_x_hor>=len_x/2.0]

    matrix_of_lines_ch=np.zeros((len(cy_main_hor)+len(cx_main_ver),10))
    matrix_of_lines_ch[:len(cy_main_hor),0]=args_hor
    matrix_of_lines_ch[len(cy_main_hor):,0]=args_ver
    matrix_of_lines_ch[len(cy_main_hor):,1]=cx_main_ver
    matrix_of_lines_ch[:len(cy_main_hor),2]=x_min_main_hor+50#x_min_main_hor+150
    matrix_of_lines_ch[len(cy_main_hor):,2]=x_min_main_ver
    matrix_of_lines_ch[:len(cy_main_hor),3]=x_max_main_hor-50#x_max_main_hor-150
    matrix_of_lines_ch[len(cy_main_hor):,3]=x_max_main_ver
    matrix_of_lines_ch[:len(cy_main_hor),4]=dist_x_hor
    matrix_of_lines_ch[len(cy_main_hor):,4]=dist_x_ver
    matrix_of_lines_ch[:len(cy_main_hor),5]=cy_main_hor
    matrix_of_lines_ch[:len(cy_main_hor),6]=y_min_main_hor
    matrix_of_lines_ch[len(cy_main_hor):,6]=y_min_main_ver
    matrix_of_lines_ch[:len(cy_main_hor),7]=y_max_main_hor
    matrix_of_lines_ch[len(cy_main_hor):,7]=y_max_main_ver
    matrix_of_lines_ch[:len(cy_main_hor),8]=dist_y_hor
    matrix_of_lines_ch[len(cy_main_hor):,8]=dist_y_ver
    matrix_of_lines_ch[len(cy_main_hor):,9]=1

    if contours_h is not None:
        _, dist_x_head, x_min_main_head, x_max_main_head, cy_main_head, _, y_min_main_head, y_max_main_head, _ = \
            find_features_of_lines(contours_h)
        matrix_l_n=np.zeros((matrix_of_lines_ch.shape[0]+len(cy_main_head),matrix_of_lines_ch.shape[1]))
        matrix_l_n[:matrix_of_lines_ch.shape[0],:]=np.copy(matrix_of_lines_ch[:,:])
        args_head=np.arange(len(cy_main_head)) + len(cy_main_hor)

        matrix_l_n[matrix_of_lines_ch.shape[0]:,0]=args_head
        matrix_l_n[matrix_of_lines_ch.shape[0]:,2]=x_min_main_head+30
        matrix_l_n[matrix_of_lines_ch.shape[0]:,3]=x_max_main_head-30
        matrix_l_n[matrix_of_lines_ch.shape[0]:,4]=dist_x_head
        matrix_l_n[matrix_of_lines_ch.shape[0]:,5]=y_min_main_head-3-8
        matrix_l_n[matrix_of_lines_ch.shape[0]:,6]=y_min_main_head-5-8
        matrix_l_n[matrix_of_lines_ch.shape[0]:,7]=y_max_main_head#y_min_main_head+1-8
        matrix_l_n[matrix_of_lines_ch.shape[0]:,8]=4
        matrix_of_lines_ch=np.copy(matrix_l_n)

    cy_main_splitters=cy_main_hor[(x_min_main_hor<=.16*region_pre_p.shape[1]) &
                                  (x_max_main_hor>=.84*region_pre_p.shape[1])]
    cy_main_splitters=np.array( list(cy_main_splitters)+list(special_separators))
    if contours_h is not None:
        try:
            cy_main_splitters_head=cy_main_head[(x_min_main_head<=.16*region_pre_p.shape[1]) &
                                                (x_max_main_head>=.84*region_pre_p.shape[1])]
            cy_main_splitters=np.array( list(cy_main_splitters)+list(cy_main_splitters_head))
        except:
            pass
    args_cy_splitter=np.argsort(cy_main_splitters)
    cy_main_splitters_sort=cy_main_splitters[args_cy_splitter]

    splitter_y_new=[]
    splitter_y_new.append(0)
    for i in range(len(cy_main_splitters_sort)):
        splitter_y_new.append(  cy_main_splitters_sort[i] )
    splitter_y_new.append(region_pre_p.shape[0])
    splitter_y_new_diff=np.diff(splitter_y_new)/float(region_pre_p.shape[0])*100

    args_big_parts=np.arange(len(splitter_y_new_diff))[ splitter_y_new_diff>22 ]

    regions_without_separators=return_regions_without_separators(region_pre_p)
    length_y_threshold=regions_without_separators.shape[0]/4.0

    num_col_fin=0
    peaks_neg_fin_fin=[]
    for itiles in args_big_parts:
        regions_without_separators_tile=regions_without_separators[int(splitter_y_new[itiles]):
                                                                   int(splitter_y_new[itiles+1]),:]
        try:
            num_col, peaks_neg_fin = find_num_col(regions_without_separators_tile,
                                                  num_col_classifier, tables, multiplier=7.0)
        except:
            num_col = 0
            peaks_neg_fin = []
        if num_col>num_col_fin:
            num_col_fin=num_col
            peaks_neg_fin_fin=peaks_neg_fin

    if len(args_big_parts)==1 and (len(peaks_neg_fin_fin)+1)<num_col_classifier:
        peaks_neg_fin=find_num_col_by_vertical_lines(vertical)
        peaks_neg_fin=peaks_neg_fin[peaks_neg_fin>=500]
        peaks_neg_fin=peaks_neg_fin[peaks_neg_fin<=(vertical.shape[1]-500)]
        peaks_neg_fin_fin=peaks_neg_fin[:]

    return num_col_fin, peaks_neg_fin_fin,matrix_of_lines_ch,splitter_y_new,separators_closeup_n

def return_boxes_of_images_by_order_of_reading_new(
        splitter_y_new, regions_without_separators,
        matrix_of_lines_ch,
        num_col_classifier, erosion_hurts, tables,
        right2left_readingorder,
        logger=None):

    if right2left_readingorder:
        regions_without_separators = cv2.flip(regions_without_separators,1)
    if logger is None:
        logger = getLogger(__package__)
    logger.debug('enter return_boxes_of_images_by_order_of_reading_new')

    boxes=[]
    peaks_neg_tot_tables = []
    splitter_y_new = np.array(splitter_y_new, dtype=int)
    for i in range(len(splitter_y_new)-1):
        #print(splitter_y_new[i],splitter_y_new[i+1])
        matrix_new = matrix_of_lines_ch[:,:][(matrix_of_lines_ch[:,6]> splitter_y_new[i] ) &
                                             (matrix_of_lines_ch[:,7]< splitter_y_new[i+1] )]
        #print(len( matrix_new[:,9][matrix_new[:,9]==1] ))
        #print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')
        # check to see is there any vertical separator to find holes.
        #if (len(matrix_new[:,9][matrix_new[:,9]==1]) > 0 and
        #    np.max(matrix_new[:,8][matrix_new[:,9]==1]) >=
        #    0.1 * (np.abs(splitter_y_new[i+1]-splitter_y_new[i]))):
        if True:
            try:
                num_col, peaks_neg_fin = find_num_col(
                    regions_without_separators[splitter_y_new[i]:splitter_y_new[i+1], :],
                    num_col_classifier, tables, multiplier=6. if erosion_hurts else 7.)
            except:
                peaks_neg_fin=[]
                num_col = 0
            try:
                if (len(peaks_neg_fin)+1)<num_col_classifier or num_col_classifier==6:
                    #print('burda')
                    peaks_neg_fin_org = np.copy(peaks_neg_fin)
                    if len(peaks_neg_fin)==0:
                        num_col, peaks_neg_fin = find_num_col(
                            regions_without_separators[splitter_y_new[i]:splitter_y_new[i+1], :],
                            num_col_classifier, tables, multiplier=3.)
                    peaks_neg_fin_early=[]
                    peaks_neg_fin_early.append(0)
                    #print(peaks_neg_fin,'peaks_neg_fin')
                    for p_n in peaks_neg_fin:
                        peaks_neg_fin_early.append(p_n)
                    peaks_neg_fin_early.append(regions_without_separators.shape[1]-1)

                    #print(peaks_neg_fin_early,'burda2')
                    peaks_neg_fin_rev=[]
                    for i_n in range(len(peaks_neg_fin_early)-1):
                        #print(i_n,'i_n')
                        #plt.plot(regions_without_separators[splitter_y_new[i]:
                        #                                    splitter_y_new[i+1],
                        #                                    peaks_neg_fin_early[i_n]:
                        #                                    peaks_neg_fin_early[i_n+1]].sum(axis=0) )
                        #plt.show()
                        try:
                            num_col, peaks_neg_fin1 = find_num_col(
                                regions_without_separators[splitter_y_new[i]:splitter_y_new[i+1],
                                                           peaks_neg_fin_early[i_n]:peaks_neg_fin_early[i_n+1]],
                                num_col_classifier,tables, multiplier=7.)
                        except:
                            peaks_neg_fin1=[]
                        try:
                            num_col, peaks_neg_fin2 = find_num_col(
                                regions_without_separators[splitter_y_new[i]:splitter_y_new[i+1],
                                                           peaks_neg_fin_early[i_n]:peaks_neg_fin_early[i_n+1]],
                                num_col_classifier,tables, multiplier=5.)
                        except:
                            peaks_neg_fin2=[]

                        if len(peaks_neg_fin1)>=len(peaks_neg_fin2):
                            peaks_neg_fin=list(np.copy(peaks_neg_fin1))
                        else:
                            peaks_neg_fin=list(np.copy(peaks_neg_fin2))
                        peaks_neg_fin=list(np.array(peaks_neg_fin)+peaks_neg_fin_early[i_n])

                        if i_n!=(len(peaks_neg_fin_early)-2):
                            peaks_neg_fin_rev.append(peaks_neg_fin_early[i_n+1])
                        #print(peaks_neg_fin,'peaks_neg_fin')
                        peaks_neg_fin_rev=peaks_neg_fin_rev+peaks_neg_fin

                    if len(peaks_neg_fin_rev)>=len(peaks_neg_fin_org):
                        peaks_neg_fin=list(np.sort(peaks_neg_fin_rev))
                        num_col=len(peaks_neg_fin)
                    else:
                        peaks_neg_fin=list(np.copy(peaks_neg_fin_org))
                        num_col=len(peaks_neg_fin)

                    #print(peaks_neg_fin,'peaks_neg_fin')
            except:
                logger.exception("cannot find peaks consistent with columns")
            #num_col, peaks_neg_fin = find_num_col(
            #    regions_without_separators[splitter_y_new[i]:splitter_y_new[i+1],:],
            #    multiplier=7.0)
            x_min_hor_some=matrix_new[:,2][ (matrix_new[:,9]==0) ]
            x_max_hor_some=matrix_new[:,3][ (matrix_new[:,9]==0) ]
            cy_hor_some=matrix_new[:,5][ (matrix_new[:,9]==0) ]
            cy_hor_diff=matrix_new[:,7][ (matrix_new[:,9]==0) ]
            arg_org_hor_some=matrix_new[:,0][ (matrix_new[:,9]==0) ]

            if right2left_readingorder:
                x_max_hor_some_new = regions_without_separators.shape[1] - x_min_hor_some
                x_min_hor_some_new = regions_without_separators.shape[1] - x_max_hor_some
                x_min_hor_some =list(np.copy(x_min_hor_some_new))
                x_max_hor_some =list(np.copy(x_max_hor_some_new))

            peaks_neg_tot=return_points_with_boundies(peaks_neg_fin,0, regions_without_separators[:,:].shape[1])
            peaks_neg_tot_tables.append(peaks_neg_tot)

            reading_order_type, x_starting, x_ending, y_type_2, y_diff_type_2, \
                y_lines_without_mother, x_start_without_mother, x_end_without_mother, there_is_sep_with_child, \
                y_lines_with_child_without_mother, x_start_with_child_without_mother, x_end_with_child_without_mother, \
                new_main_sep_y = return_x_start_end_mothers_childs_and_type_of_reading_order(
                    x_min_hor_some, x_max_hor_some, cy_hor_some, peaks_neg_tot, cy_hor_diff)

            all_columns = set(range(len(peaks_neg_tot) - 1))
            if ((reading_order_type==1) or
                (reading_order_type==0 and
                 (len(y_lines_without_mother)>=2 or there_is_sep_with_child==1))):
                try:
                    y_grenze = splitter_y_new[i] + 300
                    #check if there is a big separator in this y_mains_sep_ohne_grenzen

                    args_early_ys=np.arange(len(y_type_2))
                    #print(args_early_ys,'args_early_ys')
                    #print(splitter_y_new[i], splitter_y_new[i+1])

                    x_starting_up = x_starting[(y_type_2 > splitter_y_new[i]) &
                                               (y_type_2 <= y_grenze)]
                    x_ending_up = x_ending[(y_type_2 > splitter_y_new[i]) &
                                           (y_type_2 <= y_grenze)]
                    y_type_2_up = y_type_2[(y_type_2 > splitter_y_new[i]) &
                                           (y_type_2 <= y_grenze)]
                    y_diff_type_2_up = y_diff_type_2[(y_type_2 > splitter_y_new[i]) &
                                                     (y_type_2 <= y_grenze)]
                    args_up = args_early_ys[(y_type_2 > splitter_y_new[i]) &
                                            (y_type_2 <= y_grenze)]
                    if len(y_type_2_up) > 0:
                        y_main_separator_up = y_type_2_up [(x_starting_up==0) &
                                                           (x_ending_up==(len(peaks_neg_tot)-1) )]
                        y_diff_main_separator_up = y_diff_type_2_up[(x_starting_up==0) &
                                                                    (x_ending_up==(len(peaks_neg_tot)-1) )]
                        args_main_to_deleted = args_up[(x_starting_up==0) &
                                                       (x_ending_up==(len(peaks_neg_tot)-1) )]
                        #print(y_main_separator_up,y_diff_main_separator_up,args_main_to_deleted,'fffffjammmm')
                        if len(y_diff_main_separator_up) > 0:
                            args_to_be_kept = np.array(list( set(args_early_ys) - set(args_main_to_deleted) ))
                            #print(args_to_be_kept,'args_to_be_kept')
                            boxes.append([0, peaks_neg_tot[len(peaks_neg_tot)-1],
                                          splitter_y_new[i], y_diff_main_separator_up.max()])
                            splitter_y_new[i] = y_diff_main_separator_up.max()

                            #print(splitter_y_new[i],'splitter_y_new[i]')
                            y_type_2 = y_type_2[args_to_be_kept]
                            x_starting = x_starting[args_to_be_kept]
                            x_ending = x_ending[args_to_be_kept]
                            y_diff_type_2 = y_diff_type_2[args_to_be_kept]

                            #print('galdiha')
                            y_grenze = splitter_y_new[i] + 200
                            args_early_ys2=np.arange(len(y_type_2))
                            y_type_2_up=y_type_2[(y_type_2 > splitter_y_new[i]) &
                                                 (y_type_2 <= y_grenze)]
                            x_starting_up=x_starting[(y_type_2 > splitter_y_new[i]) &
                                                     (y_type_2 <= y_grenze)]
                            x_ending_up=x_ending[(y_type_2 > splitter_y_new[i]) &
                                                 (y_type_2 <= y_grenze)]
                            y_diff_type_2_up=y_diff_type_2[(y_type_2 > splitter_y_new[i]) &
                                                           (y_type_2 <= y_grenze)]
                            args_up2=args_early_ys2[(y_type_2 > splitter_y_new[i]) &
                                                    (y_type_2 <= y_grenze)]
                            #print(y_type_2_up,x_starting_up,x_ending_up,'didid')
                            nodes_in = set()
                            for ij in range(len(x_starting_up)):
                                nodes_in.update(range(x_starting_up[ij],
                                                      x_ending_up[ij]))
                            #print(nodes_in,'nodes_in')

                            if nodes_in == set(range(len(peaks_neg_tot)-1)):
                                pass
                            elif nodes_in == set(range(1, len(peaks_neg_tot)-1)):
                                pass
                            else:
                                #print('burdaydikh')
                                args_to_be_kept2=np.array(list( set(args_early_ys2)-set(args_up2) ))

                                if len(args_to_be_kept2)>0:
                                    y_type_2 = y_type_2[args_to_be_kept2]
                                    x_starting = x_starting[args_to_be_kept2]
                                    x_ending = x_ending[args_to_be_kept2]
                                    y_diff_type_2 = y_diff_type_2[args_to_be_kept2]
                                else:
                                    pass
                                #print('burdaydikh2')
                        elif len(y_diff_main_separator_up)==0:
                            nodes_in = set()
                            for ij in range(len(x_starting_up)):
                                nodes_in.update(range(x_starting_up[ij],
                                                      x_ending_up[ij]))
                            #print(nodes_in,'nodes_in2')
                            #print(np.array(range(len(peaks_neg_tot)-1)),'np.array(range(len(peaks_neg_tot)-1))')

                            if nodes_in == set(range(len(peaks_neg_tot)-1)):
                                pass
                            elif nodes_in == set(range(1,len(peaks_neg_tot)-1)):
                                pass
                            else:
                                #print('burdaydikh')
                                #print(args_early_ys,'args_early_ys')
                                #print(args_up,'args_up')
                                args_to_be_kept2=np.array(list( set(args_early_ys) - set(args_up) ))

                                #print(args_to_be_kept2,'args_to_be_kept2')
                                #print(len(y_type_2),len(x_starting),len(x_ending),len(y_diff_type_2))
                                if len(args_to_be_kept2)>0:
                                    y_type_2 = y_type_2[args_to_be_kept2]
                                    x_starting = x_starting[args_to_be_kept2]
                                    x_ending = x_ending[args_to_be_kept2]
                                    y_diff_type_2 = y_diff_type_2[args_to_be_kept2]
                                else:
                                    pass
                                #print('burdaydikh2')

                    #int(splitter_y_new[i])
                    y_lines_by_order=[]
                    x_start_by_order=[]
                    x_end_by_order=[]
                    if (len(x_end_with_child_without_mother)==0 and reading_order_type==0) or reading_order_type==1:
                        if reading_order_type==1:
                            y_lines_by_order.append(splitter_y_new[i])
                            x_start_by_order.append(0)
                            x_end_by_order.append(len(peaks_neg_tot)-2)
                        else:
                            #print(x_start_without_mother,x_end_without_mother,peaks_neg_tot,'dodo')
                            columns_covered_by_mothers = set()
                            for dj in range(len(x_start_without_mother)):
                                columns_covered_by_mothers.update(
                                    range(x_start_without_mother[dj],
                                          x_end_without_mother[dj]))
                            columns_not_covered = list(all_columns - columns_covered_by_mothers)
                            y_type_2 = np.append(y_type_2, np.ones(len(columns_not_covered) +
                                                                   len(x_start_without_mother),
                                                                   dtype=int) * splitter_y_new[i])
                            ##y_lines_by_order = np.append(y_lines_by_order, [splitter_y_new[i]] * len(columns_not_covered))
                            ##x_start_by_order = np.append(x_start_by_order, [0] * len(columns_not_covered))
                            x_starting = np.append(x_starting, np.array(columns_not_covered, int))
                            x_starting = np.append(x_starting, x_start_without_mother)
                            x_ending = np.append(x_ending, np.array(columns_not_covered, int) + 1)
                            x_ending = np.append(x_ending, x_end_without_mother)

                        ind_args=np.arange(len(y_type_2))
                        #ind_args=np.array(ind_args)
                        #print(ind_args,'ind_args')
                        for column in range(len(peaks_neg_tot)-1):
                            #print(column,'column')
                            ind_args_in_col=ind_args[x_starting==column]
                            #print('babali2')
                            #print(ind_args_in_col,'ind_args_in_col')
                            ind_args_in_col=np.array(ind_args_in_col)
                            #print(len(y_type_2))
                            y_column=y_type_2[ind_args_in_col]
                            x_start_column=x_starting[ind_args_in_col]
                            x_end_column=x_ending[ind_args_in_col]
                            #print('babali3')
                            ind_args_col_sorted=np.argsort(y_column)
                            y_col_sort=y_column[ind_args_col_sorted]
                            x_start_column_sort=x_start_column[ind_args_col_sorted]
                            x_end_column_sort=x_end_column[ind_args_col_sorted]
                            #print('babali4')
                            for ii in range(len(y_col_sort)):
                                #print('babali5')
                                y_lines_by_order.append(y_col_sort[ii])
                                x_start_by_order.append(x_start_column_sort[ii])
                                x_end_by_order.append(x_end_column_sort[ii]-1)
                    else:
                        #print(x_start_without_mother,x_end_without_mother,peaks_neg_tot,'dodo')
                        columns_covered_by_mothers = set()
                        for dj in range(len(x_start_without_mother)):
                            columns_covered_by_mothers.update(
                                range(x_start_without_mother[dj],
                                      x_end_without_mother[dj]))
                        columns_not_covered = list(all_columns - columns_covered_by_mothers)
                        y_type_2 = np.append(y_type_2, np.ones(len(columns_not_covered) + len(x_start_without_mother),
                                                               dtype=int) * splitter_y_new[i])
                        ##y_lines_by_order = np.append(y_lines_by_order, [splitter_y_new[i]] * len(columns_not_covered))
                        ##x_start_by_order = np.append(x_start_by_order, [0] * len(columns_not_covered))
                        x_starting = np.append(x_starting, np.array(columns_not_covered, int))
                        x_starting = np.append(x_starting, x_start_without_mother)
                        x_ending = np.append(x_ending, np.array(columns_not_covered, int) + 1)
                        x_ending = np.append(x_ending, x_end_without_mother)

                        columns_covered_by_with_child_no_mothers = set()
                        for dj in range(len(x_end_with_child_without_mother)):
                            columns_covered_by_with_child_no_mothers.update(
                                range(x_start_with_child_without_mother[dj],
                                      x_end_with_child_without_mother[dj]))
                        columns_not_covered_child_no_mother = list(
                            all_columns - columns_covered_by_with_child_no_mothers)
                        #indexes_to_be_spanned=[]
                        for i_s in range(len(x_end_with_child_without_mother)):
                            columns_not_covered_child_no_mother.append(x_start_with_child_without_mother[i_s])
                        columns_not_covered_child_no_mother = np.sort(columns_not_covered_child_no_mother)
                        ind_args = np.arange(len(y_type_2))
                        x_end_with_child_without_mother = np.array(x_end_with_child_without_mother, int)
                        x_start_with_child_without_mother = np.array(x_start_with_child_without_mother, int)
                        for i_s_nc in columns_not_covered_child_no_mother:
                            if i_s_nc in x_start_with_child_without_mother:
                                x_end_biggest_column = \
                                    x_end_with_child_without_mother[x_start_with_child_without_mother==i_s_nc][0]
                                args_all_biggest_lines = ind_args[(x_starting==i_s_nc) &
                                                                  (x_ending==x_end_biggest_column)]
                                y_column_nc = y_type_2[args_all_biggest_lines]
                                x_start_column_nc = x_starting[args_all_biggest_lines]
                                x_end_column_nc = x_ending[args_all_biggest_lines]
                                y_column_nc = np.sort(y_column_nc)
                                for i_c in range(len(y_column_nc)):
                                    if i_c==(len(y_column_nc)-1):
                                        ind_all_lines_between_nm_wc=ind_args[(y_type_2>y_column_nc[i_c]) &
                                                                              (y_type_2<splitter_y_new[i+1]) &
                                                                              (x_starting>=i_s_nc) &
                                                                              (x_ending<=x_end_biggest_column)]
                                    else:
                                        ind_all_lines_between_nm_wc=ind_args[(y_type_2>y_column_nc[i_c]) &
                                                                              (y_type_2<y_column_nc[i_c+1]) &
                                                                              (x_starting>=i_s_nc) &
                                                                              (x_ending<=x_end_biggest_column)]
                                    y_all_between_nm_wc = y_type_2[ind_all_lines_between_nm_wc]
                                    x_starting_all_between_nm_wc = x_starting[ind_all_lines_between_nm_wc]
                                    x_ending_all_between_nm_wc = x_ending[ind_all_lines_between_nm_wc]

                                    x_diff_all_between_nm_wc = x_ending_all_between_nm_wc - x_starting_all_between_nm_wc
                                    if len(x_diff_all_between_nm_wc)>0:
                                        biggest=np.argmax(x_diff_all_between_nm_wc)

                                    columns_covered_by_mothers = set()
                                    for dj in range(len(x_starting_all_between_nm_wc)):
                                        columns_covered_by_mothers.update(
                                            range(x_starting_all_between_nm_wc[dj],
                                                  x_ending_all_between_nm_wc[dj]))
                                    child_columns = set(range(i_s_nc, x_end_biggest_column))
                                    columns_not_covered = list(child_columns - columns_covered_by_mothers)

                                    should_longest_line_be_extended=0
                                    if (len(x_diff_all_between_nm_wc) > 0 and
                                        set(list(range(x_starting_all_between_nm_wc[biggest],
                                                        x_ending_all_between_nm_wc[biggest])) +
                                            list(columns_not_covered)) != child_columns):
                                        should_longest_line_be_extended=1
                                        index_lines_so_close_to_top_separator = \
                                            np.arange(len(y_all_between_nm_wc))[(y_all_between_nm_wc>y_column_nc[i_c]) &
                                                                                (y_all_between_nm_wc<=(y_column_nc[i_c]+500))]
                                        if len(index_lines_so_close_to_top_separator) > 0:
                                            indexes_remained_after_deleting_closed_lines= \
                                                np.array(list(set(list(range(len(y_all_between_nm_wc)))) -
                                                              set(list(index_lines_so_close_to_top_separator))))
                                            if len(indexes_remained_after_deleting_closed_lines) > 0:
                                                y_all_between_nm_wc = \
                                                    y_all_between_nm_wc[indexes_remained_after_deleting_closed_lines]
                                                x_starting_all_between_nm_wc = \
                                                    x_starting_all_between_nm_wc[indexes_remained_after_deleting_closed_lines]
                                                x_ending_all_between_nm_wc = \
                                                    x_ending_all_between_nm_wc[indexes_remained_after_deleting_closed_lines]

                                        y_all_between_nm_wc = np.append(y_all_between_nm_wc, y_column_nc[i_c])
                                        x_starting_all_between_nm_wc = np.append(x_starting_all_between_nm_wc, i_s_nc)
                                        x_ending_all_between_nm_wc = np.append(x_ending_all_between_nm_wc, x_end_biggest_column)

                                    if len(x_diff_all_between_nm_wc) > 0:
                                        try:
                                            y_all_between_nm_wc = np.append(y_all_between_nm_wc, y_column_nc[i_c])
                                            x_starting_all_between_nm_wc = np.append(x_starting_all_between_nm_wc, x_starting_all_between_nm_wc[biggest])
                                            x_ending_all_between_nm_wc = np.append(x_ending_all_between_nm_wc, x_ending_all_between_nm_wc[biggest])
                                        except:
                                            logger.exception("cannot append")

                                    y_all_between_nm_wc = np.append(y_all_between_nm_wc, [y_column_nc[i_c]] * len(columns_not_covered))
                                    x_starting_all_between_nm_wc = np.append(x_starting_all_between_nm_wc, np.array(columns_not_covered, int))
                                    x_ending_all_between_nm_wc = np.append(x_ending_all_between_nm_wc, np.array(columns_not_covered, int) + 1)

                                    ind_args_between=np.arange(len(x_ending_all_between_nm_wc))
                                    for column in range(int(i_s_nc), int(x_end_biggest_column)):
                                        ind_args_in_col=ind_args_between[x_starting_all_between_nm_wc==column]
                                        #print('babali2')
                                        #print(ind_args_in_col,'ind_args_in_col')
                                        ind_args_in_col=np.array(ind_args_in_col)
                                        #print(len(y_type_2))
                                        y_column=y_all_between_nm_wc[ind_args_in_col]
                                        x_start_column=x_starting_all_between_nm_wc[ind_args_in_col]
                                        x_end_column=x_ending_all_between_nm_wc[ind_args_in_col]
                                        #print('babali3')
                                        ind_args_col_sorted=np.argsort(y_column)
                                        y_col_sort=y_column[ind_args_col_sorted]
                                        x_start_column_sort=x_start_column[ind_args_col_sorted]
                                        x_end_column_sort=x_end_column[ind_args_col_sorted]
                                        #print('babali4')
                                        for ii in range(len(y_col_sort)):
                                            #print('babali5')
                                            y_lines_by_order.append(y_col_sort[ii])
                                            x_start_by_order.append(x_start_column_sort[ii])
                                            x_end_by_order.append(x_end_column_sort[ii]-1)
                            else:
                                #print(column,'column')
                                ind_args_in_col=ind_args[x_starting==i_s_nc]
                                #print('babali2')
                                #print(ind_args_in_col,'ind_args_in_col')
                                ind_args_in_col=np.array(ind_args_in_col)
                                #print(len(y_type_2))
                                y_column=y_type_2[ind_args_in_col]
                                x_start_column=x_starting[ind_args_in_col]
                                x_end_column=x_ending[ind_args_in_col]
                                #print('babali3')
                                ind_args_col_sorted=np.argsort(y_column)
                                y_col_sort=y_column[ind_args_col_sorted]
                                x_start_column_sort=x_start_column[ind_args_col_sorted]
                                x_end_column_sort=x_end_column[ind_args_col_sorted]
                                #print('babali4')
                                for ii in range(len(y_col_sort)):
                                    y_lines_by_order.append(y_col_sort[ii])
                                    x_start_by_order.append(x_start_column_sort[ii])
                                    x_end_by_order.append(x_end_column_sort[ii]-1)

                    for il in range(len(y_lines_by_order)):
                        y_copy = list(y_lines_by_order)
                        x_start_copy = list(x_start_by_order)
                        x_end_copy = list(x_end_by_order)

                        #print(y_copy,'y_copy')
                        y_itself=y_copy.pop(il)
                        x_start_itself=x_start_copy.pop(il)
                        x_end_itself=x_end_copy.pop(il)

                        #print(y_copy,'y_copy2')
                        for column in range(int(x_start_itself), int(x_end_itself)+1):
                            #print(column,'cols')
                            y_in_cols=[]
                            for yic in range(len(y_copy)):
                                #print('burda')
                                if (y_copy[yic]>y_itself and
                                    column>=x_start_copy[yic] and
                                    column<=x_end_copy[yic]):
                                    y_in_cols.append(y_copy[yic])
                            #print('burda2')
                            #print(y_in_cols,'y_in_cols')
                            if len(y_in_cols)>0:
                                y_down=np.min(y_in_cols)
                            else:
                                y_down=splitter_y_new[i+1]
                            #print(y_itself,'y_itself')
                            boxes.append([peaks_neg_tot[column],
                                          peaks_neg_tot[column+1],
                                          y_itself,
                                          y_down])
                except:
                    logger.exception("cannot assign boxes")
                    boxes.append([0, peaks_neg_tot[len(peaks_neg_tot)-1],
                                  splitter_y_new[i], splitter_y_new[i+1]])
            else:
                y_lines_by_order=[]
                x_start_by_order=[]
                x_end_by_order=[]
                if len(x_starting)>0:
                    columns_covered_by_lines_covered_more_than_2col = set()
                    for dj in range(len(x_starting)):
                        if set(range(x_starting[dj], x_ending[dj])) != all_columns:
                            columns_covered_by_lines_covered_more_than_2col.update(
                                range(x_starting[dj], x_ending[dj]))
                    columns_not_covered = list(all_columns - columns_covered_by_lines_covered_more_than_2col)

                    y_type_2 = np.append(y_type_2, np.ones(len(columns_not_covered) + 1,
                                                           dtype=int) * splitter_y_new[i])
                    ##y_lines_by_order = np.append(y_lines_by_order, [splitter_y_new[i]] * len(columns_not_covered))
                    ##x_start_by_order = np.append(x_start_by_order, [0] * len(columns_not_covered))
                    x_starting = np.append(x_starting, np.array(columns_not_covered, x_starting.dtype))
                    x_ending = np.append(x_ending, np.array(columns_not_covered, x_ending.dtype) + 1)
                    if len(new_main_sep_y) > 0:
                        x_starting = np.append(x_starting, 0)
                        x_ending = np.append(x_ending, len(peaks_neg_tot) - 1)
                    else:
                        x_starting = np.append(x_starting, x_starting[0])
                        x_ending = np.append(x_ending, x_ending[0])
                else:
                    columns_not_covered = list(all_columns)
                    y_type_2 = np.append(y_type_2, np.ones(len(columns_not_covered),
                                                           dtype=int) * splitter_y_new[i])
                    ##y_lines_by_order = np.append(y_lines_by_order, [splitter_y_new[i]] * len(columns_not_covered))
                    ##x_start_by_order = np.append(x_start_by_order, [0] * len(columns_not_covered))
                    x_starting = np.append(x_starting, np.array(columns_not_covered, x_starting.dtype))
                    x_ending = np.append(x_ending, np.array(columns_not_covered, x_ending.dtype) + 1)

                ind_args = np.arange(len(y_type_2))
                
                for column in range(len(peaks_neg_tot)-1):
                    #print(column,'column')
                    ind_args_in_col=ind_args[x_starting==column]
                    ind_args_in_col=np.array(ind_args_in_col)
                    #print(len(y_type_2))
                    y_column=y_type_2[ind_args_in_col]
                    x_start_column=x_starting[ind_args_in_col]
                    x_end_column=x_ending[ind_args_in_col]

                    ind_args_col_sorted=np.argsort(y_column)
                    y_col_sort=y_column[ind_args_col_sorted]
                    x_start_column_sort=x_start_column[ind_args_col_sorted]
                    x_end_column_sort=x_end_column[ind_args_col_sorted]
                    #print('babali4')
                    for ii in range(len(y_col_sort)):
                        #print('babali5')
                        y_lines_by_order.append(y_col_sort[ii])
                        x_start_by_order.append(x_start_column_sort[ii])
                        x_end_by_order.append(x_end_column_sort[ii]-1)

                for il in range(len(y_lines_by_order)):
                    y_copy = list(y_lines_by_order)
                    x_start_copy = list(x_start_by_order)
                    x_end_copy = list(x_end_by_order)

                    #print(y_copy,'y_copy')
                    y_itself=y_copy.pop(il)
                    x_start_itself=x_start_copy.pop(il)
                    x_end_itself=x_end_copy.pop(il)

                    for column in range(x_start_itself, x_end_itself+1):
                        #print(column,'cols')
                        y_in_cols=[]
                        for yic in range(len(y_copy)):
                            #print('burda')
                            if (y_copy[yic]>y_itself and
                                column>=x_start_copy[yic] and
                                column<=x_end_copy[yic]):
                                y_in_cols.append(y_copy[yic])
                        #print('burda2')
                        #print(y_in_cols,'y_in_cols')
                        if len(y_in_cols)>0:
                            y_down=np.min(y_in_cols)
                        else:
                            y_down=splitter_y_new[i+1]
                        #print(y_itself,'y_itself')
                        boxes.append([peaks_neg_tot[column],
                                      peaks_neg_tot[column+1],
                                      y_itself,
                                      y_down])
        #else:
            #boxes.append([ 0, regions_without_separators[:,:].shape[1] ,splitter_y_new[i],splitter_y_new[i+1]])

    if right2left_readingorder:
        peaks_neg_tot_tables_new = []
        if len(peaks_neg_tot_tables)>=1:
            for peaks_tab_ind in peaks_neg_tot_tables:
                peaks_neg_tot_tables_ind = regions_without_separators.shape[1] - np.array(peaks_tab_ind)
                peaks_neg_tot_tables_ind = list(peaks_neg_tot_tables_ind[::-1])
                peaks_neg_tot_tables_new.append(peaks_neg_tot_tables_ind)

        for i in range(len(boxes)):
            x_start_new = regions_without_separators.shape[1] - boxes[i][1]
            x_end_new = regions_without_separators.shape[1] - boxes[i][0]
            boxes[i][0] = x_start_new
            boxes[i][1] = x_end_new
        peaks_neg_tot_tables = peaks_neg_tot_tables_new

    logger.debug('exit return_boxes_of_images_by_order_of_reading_new')
    return boxes, peaks_neg_tot_tables

def is_image_filename(fname: str) -> bool:
    return fname.lower().endswith(('.jpg',
                                   '.jpeg',
                                   '.png',
                                   '.tif',
                                   '.tiff',
    ))

def is_xml_filename(fname: str) -> bool:
    return fname.lower().endswith('.xml')
