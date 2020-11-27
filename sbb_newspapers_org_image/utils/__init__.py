import math

import matplotlib.pyplot as plt
import numpy as np
from shapely import geometry
import cv2
import imutils
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from .is_nan import isNaN


def crop_image_inside_box(box, img_org_copy):
    image_box = img_org_copy[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
    return image_box, [box[1], box[1] + box[3], box[0], box[0] + box[2]]

def otsu_copy(img):
    img_r = np.zeros(img.shape)
    img1 = img[:, :, 0]
    img2 = img[:, :, 1]
    img3 = img[:, :, 2]
    # print(img.min())
    # print(img[:,:,0].min())
    # blur = cv2.GaussianBlur(img,(5,5))
    # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    retval1, threshold1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    retval2, threshold2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    retval3, threshold3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_r[:, :, 0] = threshold1
    img_r[:, :, 1] = threshold1
    img_r[:, :, 2] = threshold1
    return img_r

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
    return slope_lines, dis_x, x_min_main, x_max_main, np.array(cy_main), np.array(slope_lines_org), y_min_main, y_max_main, np.array(cx_main)

def boosting_headers_by_longshot_region_segmentation(textregion_pre_p, textregion_pre_np, img_only_text):
    textregion_pre_p_org = np.copy(textregion_pre_p)
    # 4 is drop capitals
    headers_in_longshot = (textregion_pre_np[:, :, 0] == 2) * 1
    # headers_in_longshot= ( (textregion_pre_np[:,:,0]==2) | (textregion_pre_np[:,:,0]==1) )*1
    textregion_pre_p[:, :, 0][(headers_in_longshot[:, :] == 1) & (textregion_pre_p[:, :, 0] != 4)] = 2
    textregion_pre_p[:, :, 0][textregion_pre_p[:, :, 0] == 1] = 0
    # textregion_pre_p[:,:,0][( img_only_text[:,:]==1) & (textregion_pre_p[:,:,0]!=7)  & (textregion_pre_p[:,:,0]!=2)]=1 # eralier it was so, but by this manner the drop capitals are alse deleted
    textregion_pre_p[:, :, 0][(img_only_text[:, :] == 1) & (textregion_pre_p[:, :, 0] != 7) & (textregion_pre_p[:, :, 0] != 4) & (textregion_pre_p[:, :, 0] != 2)] = 1
    return textregion_pre_p


def find_num_col_deskew(regions_without_seperators, sigma_, multiplier=3.8):
    regions_without_seperators_0=regions_without_seperators[:,:].sum(axis=1)

    ##meda_n_updown=regions_without_seperators_0[len(regions_without_seperators_0)::-1]

    ##first_nonzero=(next((i for i, x in enumerate(regions_without_seperators_0) if x), 0))
    ##last_nonzero=(next((i for i, x in enumerate(meda_n_updown) if x), 0))

    ##last_nonzero=len(regions_without_seperators_0)-last_nonzero


    y=regions_without_seperators_0#[first_nonzero:last_nonzero]

    ##y_help=np.zeros(len(y)+20)

    ##y_help[10:len(y)+10]=y

    ##x=np.array( range(len(y)) )




    ##zneg_rev=-y_help+np.max(y_help)

    ##zneg=np.zeros(len(zneg_rev)+20)

    ##zneg[10:len(zneg_rev)+10]=zneg_rev

    z=gaussian_filter1d(y, sigma_)
    ###zneg= gaussian_filter1d(zneg, sigma_)


    ###peaks_neg, _ = find_peaks(zneg, height=0)
    ###peaks, _ = find_peaks(z, height=0)

    ###peaks_neg=peaks_neg-10-10

    ####print(np.std(z),'np.std(z)np.std(z)np.std(z)')

    #####plt.plot(z)
    #####plt.show()

    #####plt.imshow(regions_without_seperators)
    #####plt.show()
    ###"""
    ###last_nonzero=last_nonzero-0#100
    ###first_nonzero=first_nonzero+0#+100

    ###peaks_neg=peaks_neg[(peaks_neg>first_nonzero) & (peaks_neg<last_nonzero)]

    ###peaks=peaks[(peaks>.06*regions_without_seperators.shape[1]) & (peaks<0.94*regions_without_seperators.shape[1])]
    ###"""
    ###interest_pos=z[peaks]

    ###interest_pos=interest_pos[interest_pos>10]

    ###interest_neg=z[peaks_neg]

    ###min_peaks_pos=np.mean(interest_pos)
    ###min_peaks_neg=0#np.min(interest_neg)

    ###dis_talaei=(min_peaks_pos-min_peaks_neg)/multiplier
    ####print(interest_pos)
    ###grenze=min_peaks_pos-dis_talaei#np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

    ###interest_neg_fin=interest_neg[(interest_neg<grenze)]
    ###peaks_neg_fin=peaks_neg[(interest_neg<grenze)]
    ###interest_neg_fin=interest_neg[(interest_neg<grenze)]

    ###"""
    ###if interest_neg[0]<0.1:
        ###interest_neg=interest_neg[1:]
    ###if interest_neg[len(interest_neg)-1]<0.1:
        ###interest_neg=interest_neg[:len(interest_neg)-1]



    ###min_peaks_pos=np.min(interest_pos)
    ###min_peaks_neg=0#np.min(interest_neg)


    ###dis_talaei=(min_peaks_pos-min_peaks_neg)/multiplier
    ###grenze=min_peaks_pos-dis_talaei#np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0
    ###"""
    ####interest_neg_fin=interest_neg#[(interest_neg<grenze)]
    ####peaks_neg_fin=peaks_neg#[(interest_neg<grenze)]
    ####interest_neg_fin=interest_neg#[(interest_neg<grenze)]

    ###num_col=(len(interest_neg_fin))+1


    ###p_l=0
    ###p_u=len(y)-1
    ###p_m=int(len(y)/2.)
    ###p_g_l=int(len(y)/3.)
    ###p_g_u=len(y)-int(len(y)/3.)


    ###diff_peaks=np.abs( np.diff(peaks_neg_fin) )
    ###diff_peaks_annormal=diff_peaks[diff_peaks<30]

    #print(len(interest_neg_fin),np.mean(interest_neg_fin))
    return np.std(z)#interest_neg_fin,np.std(z)

def return_hor_spliter_by_index_for_without_verticals(peaks_neg_fin_t, x_min_hor_some, x_max_hor_some):
    # print(peaks_neg_fin_t,x_min_hor_some,x_max_hor_some)
    arg_min_hor_sort = np.argsort(x_min_hor_some)
    x_min_hor_some_sort = np.sort(x_min_hor_some)
    x_max_hor_some_sort = x_max_hor_some[arg_min_hor_sort]

    arg_minmax = np.array(range(len(peaks_neg_fin_t)))
    indexer_lines = []
    indexes_to_delete = []
    indexer_lines_deletions_len = []
    indexr_uniq_ind = []
    for i in range(len(x_min_hor_some_sort)):
        min_h = peaks_neg_fin_t - x_min_hor_some_sort[i]

        max_h = peaks_neg_fin_t - x_max_hor_some_sort[i]

        min_h[0] = min_h[0]  # +20
        max_h[len(max_h) - 1] = max_h[len(max_h) - 1] - 20

        min_h_neg = arg_minmax[(min_h < 0)]
        min_h_neg_n = min_h[min_h < 0]

        try:
            min_h_neg = [min_h_neg[np.argmax(min_h_neg_n)]]
        except:
            min_h_neg = []

        max_h_neg = arg_minmax[(max_h > 0)]
        max_h_neg_n = max_h[max_h > 0]

        if len(max_h_neg_n) > 0:
            max_h_neg = [max_h_neg[np.argmin(max_h_neg_n)]]
        else:
            max_h_neg = []

        if len(min_h_neg) > 0 and len(max_h_neg) > 0:
            deletions = list(range(min_h_neg[0] + 1, max_h_neg[0]))
            unique_delets_int = []
            # print(deletions,len(deletions),'delii')
            if len(deletions) > 0:

                for j in range(len(deletions)):
                    indexes_to_delete.append(deletions[j])
                    # print(deletions,indexes_to_delete,'badiii')
                    unique_delets = np.unique(indexes_to_delete)
                    # print(min_h_neg[0],unique_delets)
                    unique_delets_int = unique_delets[unique_delets < min_h_neg[0]]

                indexer_lines_deletions_len.append(len(deletions))
                indexr_uniq_ind.append([deletions])

            else:
                indexer_lines_deletions_len.append(0)
                indexr_uniq_ind.append(-999)

            index_line_true = min_h_neg[0] - len(unique_delets_int)
            # print(index_line_true)
            if index_line_true > 0 and min_h_neg[0] >= 2:
                index_line_true = index_line_true
            else:
                index_line_true = min_h_neg[0]

            indexer_lines.append(index_line_true)

            if len(unique_delets_int) > 0:
                for dd in range(len(unique_delets_int)):
                    indexes_to_delete.append(unique_delets_int[dd])
        else:
            indexer_lines.append(-999)
            indexer_lines_deletions_len.append(-999)
            indexr_uniq_ind.append(-999)

    peaks_true = []
    for m in range(len(peaks_neg_fin_t)):
        if m in indexes_to_delete:
            pass
        else:
            peaks_true.append(peaks_neg_fin_t[m])
    return indexer_lines, peaks_true, arg_min_hor_sort, indexer_lines_deletions_len, indexr_uniq_ind

def find_num_col(regions_without_seperators, multiplier=3.8):
    regions_without_seperators_0 = regions_without_seperators[:, :].sum(axis=0)

    ##plt.plot(regions_without_seperators_0)
    ##plt.show()

    sigma_ = 35  # 70#35

    meda_n_updown = regions_without_seperators_0[len(regions_without_seperators_0) :: -1]

    first_nonzero = next((i for i, x in enumerate(regions_without_seperators_0) if x), 0)
    last_nonzero = next((i for i, x in enumerate(meda_n_updown) if x), 0)

    # print(last_nonzero)
    # print(isNaN(last_nonzero))
    # last_nonzero=0#halalikh
    last_nonzero = len(regions_without_seperators_0) - last_nonzero

    y = regions_without_seperators_0  # [first_nonzero:last_nonzero]

    y_help = np.zeros(len(y) + 20)

    y_help[10 : len(y) + 10] = y

    x = np.array(range(len(y)))

    zneg_rev = -y_help + np.max(y_help)

    zneg = np.zeros(len(zneg_rev) + 20)

    zneg[10 : len(zneg_rev) + 10] = zneg_rev

    z = gaussian_filter1d(y, sigma_)
    zneg = gaussian_filter1d(zneg, sigma_)

    peaks_neg, _ = find_peaks(zneg, height=0)
    peaks, _ = find_peaks(z, height=0)

    peaks_neg = peaks_neg - 10 - 10

    last_nonzero = last_nonzero - 100
    first_nonzero = first_nonzero + 200

    peaks_neg = peaks_neg[(peaks_neg > first_nonzero) & (peaks_neg < last_nonzero)]

    peaks = peaks[(peaks > 0.06 * regions_without_seperators.shape[1]) & (peaks < 0.94 * regions_without_seperators.shape[1])]
    peaks_neg = peaks_neg[(peaks_neg > 370) & (peaks_neg < (regions_without_seperators.shape[1] - 370))]

    # print(peaks)
    interest_pos = z[peaks]

    interest_pos = interest_pos[interest_pos > 10]

    # plt.plot(z)
    # plt.show()
    interest_neg = z[peaks_neg]

    min_peaks_pos = np.min(interest_pos)
    max_peaks_pos = np.max(interest_pos)

    if max_peaks_pos / min_peaks_pos >= 35:
        min_peaks_pos = np.mean(interest_pos)

    min_peaks_neg = 0  # np.min(interest_neg)

    # print(np.min(interest_pos),np.max(interest_pos),np.max(interest_pos)/np.min(interest_pos),'minmax')
    # $print(min_peaks_pos)
    dis_talaei = (min_peaks_pos - min_peaks_neg) / multiplier
    # print(interest_pos)
    grenze = min_peaks_pos - dis_talaei  # np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

    # print(interest_neg,'interest_neg')
    # print(grenze,'grenze')
    # print(min_peaks_pos,'min_peaks_pos')
    # print(dis_talaei,'dis_talaei')
    # print(peaks_neg,'peaks_neg')

    interest_neg_fin = interest_neg[(interest_neg < grenze)]
    peaks_neg_fin = peaks_neg[(interest_neg < grenze)]
    # interest_neg_fin=interest_neg[(interest_neg<grenze)]

    num_col = (len(interest_neg_fin)) + 1

    # print(peaks_neg_fin,'peaks_neg_fin')
    # print(num_col,'diz')
    p_l = 0
    p_u = len(y) - 1
    p_m = int(len(y) / 2.0)
    p_g_l = int(len(y) / 4.0)
    p_g_u = len(y) - int(len(y) / 4.0)

    if num_col == 3:
        if (peaks_neg_fin[0] > p_g_u and peaks_neg_fin[1] > p_g_u) or (peaks_neg_fin[0] < p_g_l and peaks_neg_fin[1] < p_g_l) or ((peaks_neg_fin[0] + 200) < p_m and peaks_neg_fin[1] < p_m) or ((peaks_neg_fin[0] - 200) > p_m and peaks_neg_fin[1] > p_m):
            num_col = 1
            peaks_neg_fin = []
        else:
            pass

    if num_col == 2:
        if (peaks_neg_fin[0] > p_g_u) or (peaks_neg_fin[0] < p_g_l):
            num_col = 1
            peaks_neg_fin = []
        else:
            pass

    ##print(len(peaks_neg_fin))

    diff_peaks = np.abs(np.diff(peaks_neg_fin))

    cut_off = 400
    peaks_neg_true = []
    forest = []

    # print(len(peaks_neg_fin),'len_')

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
    p_quarter = int(len(y) / 5.0)
    p_g_l = int(len(y) / 4.0)
    p_g_u = len(y) - int(len(y) / 4.0)

    p_u_quarter = len(y) - p_quarter

    ##print(num_col,'early')
    if num_col == 3:
        if (peaks_neg_true[0] > p_g_u and peaks_neg_true[1] > p_g_u) or (peaks_neg_true[0] < p_g_l and peaks_neg_true[1] < p_g_l) or (peaks_neg_true[0] < p_m and (peaks_neg_true[1] + 200) < p_m) or ((peaks_neg_true[0] - 200) > p_m and peaks_neg_true[1] > p_m):
            num_col = 1
            peaks_neg_true = []
        elif (peaks_neg_true[0] < p_g_u and peaks_neg_true[0] > p_g_l) and (peaks_neg_true[1] > p_u_quarter):
            peaks_neg_true = [peaks_neg_true[0]]
        elif (peaks_neg_true[1] < p_g_u and peaks_neg_true[1] > p_g_l) and (peaks_neg_true[0] < p_quarter):
            peaks_neg_true = [peaks_neg_true[1]]
        else:
            pass

    if num_col == 2:
        if (peaks_neg_true[0] > p_g_u) or (peaks_neg_true[0] < p_g_l):
            num_col = 1
            peaks_neg_true = []
        else:
            pass

    diff_peaks_annormal = diff_peaks[diff_peaks < 360]

    if len(diff_peaks_annormal) > 0:
        arg_help = np.array(range(len(diff_peaks)))
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

    # plt.plot(gaussian_filter1d(y, sigma_))
    # plt.plot(peaks_neg_true,z[peaks_neg_true],'*')
    # plt.plot([0,len(y)], [grenze,grenze])
    # plt.show()

    ##print(len(peaks_neg_true))
    return len(peaks_neg_true), peaks_neg_true

def find_num_col_only_image(regions_without_seperators, multiplier=3.8):
    regions_without_seperators_0 = regions_without_seperators[:, :].sum(axis=0)

    ##plt.plot(regions_without_seperators_0)
    ##plt.show()

    sigma_ = 15

    meda_n_updown = regions_without_seperators_0[len(regions_without_seperators_0) :: -1]

    first_nonzero = next((i for i, x in enumerate(regions_without_seperators_0) if x), 0)
    last_nonzero = next((i for i, x in enumerate(meda_n_updown) if x), 0)

    last_nonzero = len(regions_without_seperators_0) - last_nonzero

    y = regions_without_seperators_0  # [first_nonzero:last_nonzero]

    y_help = np.zeros(len(y) + 20)

    y_help[10 : len(y) + 10] = y

    x = np.array(range(len(y)))

    zneg_rev = -y_help + np.max(y_help)

    zneg = np.zeros(len(zneg_rev) + 20)

    zneg[10 : len(zneg_rev) + 10] = zneg_rev

    z = gaussian_filter1d(y, sigma_)
    zneg = gaussian_filter1d(zneg, sigma_)

    peaks_neg, _ = find_peaks(zneg, height=0)
    peaks, _ = find_peaks(z, height=0)

    peaks_neg = peaks_neg - 10 - 10

    peaks_neg_org = np.copy(peaks_neg)

    peaks_neg = peaks_neg[(peaks_neg > first_nonzero) & (peaks_neg < last_nonzero)]

    peaks = peaks[(peaks > 0.09 * regions_without_seperators.shape[1]) & (peaks < 0.91 * regions_without_seperators.shape[1])]

    peaks_neg = peaks_neg[(peaks_neg > 500) & (peaks_neg < (regions_without_seperators.shape[1] - 500))]
    # print(peaks)
    interest_pos = z[peaks]

    interest_pos = interest_pos[interest_pos > 10]

    interest_neg = z[peaks_neg]
    min_peaks_pos = np.mean(interest_pos)  # np.min(interest_pos)
    min_peaks_neg = 0  # np.min(interest_neg)

    # $print(min_peaks_pos)
    dis_talaei = (min_peaks_pos - min_peaks_neg) / multiplier
    # print(interest_pos)
    grenze = min_peaks_pos - dis_talaei  # np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

    interest_neg_fin = interest_neg[(interest_neg < grenze)]
    peaks_neg_fin = peaks_neg[(interest_neg < grenze)]

    num_col = (len(interest_neg_fin)) + 1

    p_l = 0
    p_u = len(y) - 1
    p_m = int(len(y) / 2.0)
    p_g_l = int(len(y) / 3.0)
    p_g_u = len(y) - int(len(y) / 3.0)

    if num_col == 3:
        if (peaks_neg_fin[0] > p_g_u and peaks_neg_fin[1] > p_g_u) or (peaks_neg_fin[0] < p_g_l and peaks_neg_fin[1] < p_g_l) or (peaks_neg_fin[0] < p_m and peaks_neg_fin[1] < p_m) or (peaks_neg_fin[0] > p_m and peaks_neg_fin[1] > p_m):
            num_col = 1
        else:
            pass

    if num_col == 2:
        if (peaks_neg_fin[0] > p_g_u) or (peaks_neg_fin[0] < p_g_l):
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
        if (peaks_neg_true[0] > p_g_u and peaks_neg_true[1] > p_g_u) or (peaks_neg_true[0] < p_g_l and peaks_neg_true[1] < p_g_l) or (peaks_neg_true[0] < p_m and peaks_neg_true[1] < p_m) or (peaks_neg_true[0] > p_m and peaks_neg_true[1] > p_m):
            num_col = 1
            peaks_neg_true = []
        elif (peaks_neg_true[0] < p_g_u and peaks_neg_true[0] > p_g_l) and (peaks_neg_true[1] > p_u_quarter):
            peaks_neg_true = [peaks_neg_true[0]]
        elif (peaks_neg_true[1] < p_g_u and peaks_neg_true[1] > p_g_l) and (peaks_neg_true[0] < p_quarter):
            peaks_neg_true = [peaks_neg_true[1]]
        else:
            pass

    if num_col == 2:
        if (peaks_neg_true[0] > p_g_u) or (peaks_neg_true[0] < p_g_l):
            num_col = 1
            peaks_neg_true = []

    if num_col == 4:
        if len(np.array(peaks_neg_true)[np.array(peaks_neg_true) < p_g_l]) == 2 or len(np.array(peaks_neg_true)[np.array(peaks_neg_true) > (len(y) - p_g_l)]) == 2:
            num_col = 1
            peaks_neg_true = []
        else:
            pass

    # no deeper hill around found hills

    peaks_fin_true = []
    for i in range(len(peaks_neg_true)):
        hill_main = peaks_neg_true[i]
        # deep_depth=z[peaks_neg]
        hills_around = peaks_neg_org[((peaks_neg_org > hill_main) & (peaks_neg_org <= hill_main + 400)) | ((peaks_neg_org < hill_main) & (peaks_neg_org >= hill_main - 400))]
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
        arg_help = np.array(range(len(diff_peaks)))
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

def find_num_col_by_vertical_lines(regions_without_seperators, multiplier=3.8):
    regions_without_seperators_0 = regions_without_seperators[:, :, 0].sum(axis=0)

    ##plt.plot(regions_without_seperators_0)
    ##plt.show()

    sigma_ = 35  # 70#35

    z = gaussian_filter1d(regions_without_seperators_0, sigma_)

    peaks, _ = find_peaks(z, height=0)

    # print(peaks,'peaksnew')
    return peaks


def delete_seperator_around(spliter_y, peaks_neg, image_by_region):
    # format of subboxes box=[x1, x2 , y1, y2]

    if len(image_by_region.shape) == 3:
        for i in range(len(spliter_y) - 1):
            for j in range(1, len(peaks_neg[i]) - 1):
                image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0] == 6] = 0
                image_by_region[spliter_y[i] : spliter_y[i + 1], peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 1] == 6] = 0
                image_by_region[spliter_y[i] : spliter_y[i + 1], peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 2] == 6] = 0

                image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0] == 7] = 0
                image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 1] == 7] = 0
                image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 0][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j]), 2] == 7] = 0
    else:
        for i in range(len(spliter_y) - 1):
            for j in range(1, len(peaks_neg[i]) - 1):
                image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j])][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j])] == 6] = 0

                image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j])][image_by_region[int(spliter_y[i]) : int(spliter_y[i + 1]), peaks_neg[i][j] - int(1.0 / 20.0 * peaks_neg[i][j]) : peaks_neg[i][j] + int(1.0 / 20.0 * peaks_neg[i][j])] == 7] = 0
    return image_by_region

def return_regions_without_seperators(regions_pre):
    kernel = np.ones((5, 5), np.uint8)
    regions_without_seperators = ((regions_pre[:, :] != 6) & (regions_pre[:, :] != 0)) * 1
    # regions_without_seperators=( (image_regions_eraly_p[:,:,:]!=6) & (image_regions_eraly_p[:,:,:]!=0) & (image_regions_eraly_p[:,:,:]!=5) & (image_regions_eraly_p[:,:,:]!=8) & (image_regions_eraly_p[:,:,:]!=7))*1

    regions_without_seperators = regions_without_seperators.astype(np.uint8)

    regions_without_seperators = cv2.erode(regions_without_seperators, kernel, iterations=6)

    return regions_without_seperators


def put_drop_out_from_only_drop_model(layout_no_patch, layout1):

    drop_only = (layout_no_patch[:, :, 0] == 4) * 1
    contours_drop, hir_on_drop = return_contours_of_image(drop_only)
    contours_drop_parent = return_parent_contours(contours_drop, hir_on_drop)

    areas_cnt_text = np.array([cv2.contourArea(contours_drop_parent[j]) for j in range(len(contours_drop_parent))])
    areas_cnt_text = areas_cnt_text / float(drop_only.shape[0] * drop_only.shape[1])

    contours_drop_parent = [contours_drop_parent[jz] for jz in range(len(contours_drop_parent)) if areas_cnt_text[jz] > 0.00001]

    areas_cnt_text = [areas_cnt_text[jz] for jz in range(len(areas_cnt_text)) if areas_cnt_text[jz] > 0.00001]

    contours_drop_parent_final = []

    for jj in range(len(contours_drop_parent)):
        x, y, w, h = cv2.boundingRect(contours_drop_parent[jj])
        # boxes.append([int(x), int(y), int(w), int(h)])

        map_of_drop_contour_bb = np.zeros((layout1.shape[0], layout1.shape[1]))
        map_of_drop_contour_bb[y : y + h, x : x + w] = layout1[y : y + h, x : x + w]

        if (((map_of_drop_contour_bb == 1) * 1).sum() / float(((map_of_drop_contour_bb == 5) * 1).sum()) * 100) >= 15:
            contours_drop_parent_final.append(contours_drop_parent[jj])

    layout_no_patch[:, :, 0][layout_no_patch[:, :, 0] == 4] = 0

    layout_no_patch = cv2.fillPoly(layout_no_patch, pts=contours_drop_parent_final, color=(4, 4, 4))

    return layout_no_patch

def putt_bb_of_drop_capitals_of_model_in_patches_in_layout(layout_in_patch):

    drop_only = (layout_in_patch[:, :, 0] == 4) * 1
    contours_drop, hir_on_drop = return_contours_of_image(drop_only)
    contours_drop_parent = return_parent_contours(contours_drop, hir_on_drop)

    areas_cnt_text = np.array([cv2.contourArea(contours_drop_parent[j]) for j in range(len(contours_drop_parent))])
    areas_cnt_text = areas_cnt_text / float(drop_only.shape[0] * drop_only.shape[1])

    contours_drop_parent = [contours_drop_parent[jz] for jz in range(len(contours_drop_parent)) if areas_cnt_text[jz] > 0.00001]

    areas_cnt_text = [areas_cnt_text[jz] for jz in range(len(areas_cnt_text)) if areas_cnt_text[jz] > 0.001]

    contours_drop_parent_final = []

    for jj in range(len(contours_drop_parent)):
        x, y, w, h = cv2.boundingRect(contours_drop_parent[jj])
        layout_in_patch[y : y + h, x : x + w, 0] = 4

    return layout_in_patch

def check_any_text_region_in_model_one_is_main_or_header(regions_model_1,regions_model_full,contours_only_text_parent,all_box_coord,all_found_texline_polygons,slopes,contours_only_text_parent_d_ordered):
    #text_only=(regions_model_1[:,:]==1)*1
    #contours_only_text,hir_on_text=self.return_contours_of_image(text_only)

    """
    contours_only_text_parent=self.return_parent_contours( contours_only_text,hir_on_text)

    areas_cnt_text=np.array([cv2.contourArea(contours_only_text_parent[j]) for j in range(len(contours_only_text_parent))])
    areas_cnt_text=areas_cnt_text/float(text_only.shape[0]*text_only.shape[1])

    ###areas_cnt_text_h=np.array([cv2.contourArea(contours_only_text_parent_h[j]) for j in range(len(contours_only_text_parent_h))])
    ###areas_cnt_text_h=areas_cnt_text_h/float(text_only_h.shape[0]*text_only_h.shape[1])

    ###contours_only_text_parent=[contours_only_text_parent[jz] for jz in range(len(contours_only_text_parent)) if areas_cnt_text[jz]>0.0002]
    contours_only_text_parent=[contours_only_text_parent[jz] for jz in range(len(contours_only_text_parent)) if areas_cnt_text[jz]>0.00001]
    """

    cx_main,cy_main ,x_min_main , x_max_main, y_min_main ,y_max_main,y_corr_x_min_from_argmin=self.find_new_features_of_contoures(contours_only_text_parent)

    length_con=x_max_main-x_min_main
    height_con=y_max_main-y_min_main



    all_found_texline_polygons_main=[]
    all_found_texline_polygons_head=[]

    all_box_coord_main=[]
    all_box_coord_head=[]

    slopes_main=[]
    slopes_head=[]

    contours_only_text_parent_main=[]
    contours_only_text_parent_head=[]

    contours_only_text_parent_main_d=[]
    contours_only_text_parent_head_d=[]

    for ii in range(len(contours_only_text_parent)):
        con=contours_only_text_parent[ii]
        img=np.zeros((regions_model_1.shape[0],regions_model_1.shape[1],3))
        img = cv2.fillPoly(img, pts=[con], color=(255, 255, 255))



        all_pixels=((img[:,:,0]==255)*1).sum()

        pixels_header=( ( (img[:,:,0]==255) & (regions_model_full[:,:,0]==2) )*1 ).sum()
        pixels_main=all_pixels-pixels_header


        if (pixels_header>=pixels_main) and ( (length_con[ii]/float(height_con[ii]) )>=1.3 ):
            regions_model_1[:,:][(regions_model_1[:,:]==1) & (img[:,:,0]==255) ]=2
            contours_only_text_parent_head.append(con)
            if contours_only_text_parent_d_ordered is not None:
                contours_only_text_parent_head_d.append(contours_only_text_parent_d_ordered[ii])
            all_box_coord_head.append(all_box_coord[ii])
            slopes_head.append(slopes[ii])
            all_found_texline_polygons_head.append(all_found_texline_polygons[ii])
        else:
            regions_model_1[:,:][(regions_model_1[:,:]==1) & (img[:,:,0]==255) ]=1
            contours_only_text_parent_main.append(con)
            if contours_only_text_parent_d_ordered is not None:
                contours_only_text_parent_main_d.append(contours_only_text_parent_d_ordered[ii])
            all_box_coord_main.append(all_box_coord[ii])
            slopes_main.append(slopes[ii])
            all_found_texline_polygons_main.append(all_found_texline_polygons[ii])

        #print(all_pixels,pixels_main,pixels_header)



        #plt.imshow(img[:,:,0])
        #plt.show()
    return regions_model_1,contours_only_text_parent_main,contours_only_text_parent_head,all_box_coord_main,all_box_coord_head,all_found_texline_polygons_main,all_found_texline_polygons_head,slopes_main,slopes_head,contours_only_text_parent_main_d,contours_only_text_parent_head_d

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
        areas_cnt_text = np.array([cv2.contourArea(textlines_tot[j]) for j in range(len(textlines_tot))])
        areas_cnt_text = areas_cnt_text / float(textline_iamge.shape[0] * textline_iamge.shape[1])
        indexes_textlines = np.array(range(len(textlines_tot)))

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

        img_textline_s = np.zeros((textline_iamge.shape[0], textline_iamge.shape[1]))
        img_textline_s = cv2.fillPoly(img_textline_s, pts=textlines_small, color=(1, 1, 1))

        img_textline_b = np.zeros((textline_iamge.shape[0], textline_iamge.shape[1]))
        img_textline_b = cv2.fillPoly(img_textline_b, pts=textlines_big, color=(1, 1, 1))

        sum_small_big_all = img_textline_s + img_textline_b
        sum_small_big_all2 = (sum_small_big_all[:, :] == 2) * 1

        sum_intersection_sb = sum_small_big_all2.sum(axis=1).sum()

        if sum_intersection_sb > 0:

            dis_small_from_bigs_tot = []
            for z1 in range(len(textlines_small)):
                # print(len(textlines_small),'small')
                intersections = []
                for z2 in range(len(textlines_big)):
                    img_text = np.zeros((textline_iamge.shape[0], textline_iamge.shape[1]))
                    img_text = cv2.fillPoly(img_text, pts=[textlines_small[z1]], color=(1, 1, 1))

                    img_text2 = np.zeros((textline_iamge.shape[0], textline_iamge.shape[1]))
                    img_text2 = cv2.fillPoly(img_text2, pts=[textlines_big[z2]], color=(1, 1, 1))

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

                img_text2 = np.zeros((textline_iamge.shape[0], textline_iamge.shape[1], 3))
                img_text2 = cv2.fillPoly(img_text2, pts=[textlines_big[z]], color=(255, 255, 255))

                textlines_big_with_change.append(z)

                for k in index_small_textlines:
                    img_text2 = cv2.fillPoly(img_text2, pts=[textlines_small[k]], color=(255, 255, 255))
                    textlines_small_with_change.append(k)

                img_text2 = img_text2.astype(np.uint8)
                imgray = cv2.cvtColor(img_text2, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(imgray, 0, 255, 0)
                cont, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # print(cont[0],type(cont))

                textlines_big_with_change_con.append(cont)
                textlines_big_org_form[z] = cont[0]

                # plt.imshow(img_text2)
                # plt.show()

            # print(textlines_big_with_change,'textlines_big_with_change')
            # print(textlines_small_with_change,'textlines_small_with_change')
            # print(textlines_big)
            textlines_con_changed.append(textlines_big_org_form)

        else:
            textlines_con_changed.append(textlines_big_org_form)
    return textlines_con_changed

def order_and_id_of_texts(found_polygons_text_region, found_polygons_text_region_h, matrix_of_orders, indexes_sorted, index_of_types, kind_of_texts, ref_point):
    indexes_sorted = np.array(indexes_sorted)
    index_of_types = np.array(index_of_types)
    kind_of_texts = np.array(kind_of_texts)

    id_of_texts = []
    order_of_texts = []

    index_of_types_1 = index_of_types[kind_of_texts == 1]
    indexes_sorted_1 = indexes_sorted[kind_of_texts == 1]

    index_of_types_2 = index_of_types[kind_of_texts == 2]
    indexes_sorted_2 = indexes_sorted[kind_of_texts == 2]

    ##print(index_of_types,'index_of_types')
    ##print(kind_of_texts,'kind_of_texts')
    ##print(len(found_polygons_text_region),'found_polygons_text_region')
    ##print(index_of_types_1,'index_of_types_1')
    ##print(indexes_sorted_1,'indexes_sorted_1')
    index_b = 0 + ref_point
    for mm in range(len(found_polygons_text_region)):

        id_of_texts.append("r" + str(index_b))
        interest = indexes_sorted_1[indexes_sorted_1 == index_of_types_1[mm]]

        if len(interest) > 0:
            order_of_texts.append(interest[0])
            index_b += 1
        else:
            pass

    for mm in range(len(found_polygons_text_region_h)):
        id_of_texts.append("r" + str(index_b))
        interest = indexes_sorted_2[index_of_types_2[mm]]
        order_of_texts.append(interest)
        index_b += 1

    return order_of_texts, id_of_texts

def order_of_regions(textline_mask, contours_main, contours_header, y_ref):

    ##plt.imshow(textline_mask)
    ##plt.show()
    """
    print(len(contours_main),'contours_main')
    mada_n=textline_mask.sum(axis=1)
    y=mada_n[:]

    y_help=np.zeros(len(y)+40)
    y_help[20:len(y)+20]=y
    x=np.array( range(len(y)) )


    peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)

    ##plt.imshow(textline_mask[:,:])
    ##plt.show()


    sigma_gaus=8

    z= gaussian_filter1d(y_help, sigma_gaus)
    zneg_rev=-y_help+np.max(y_help)

    zneg=np.zeros(len(zneg_rev)+40)
    zneg[20:len(zneg_rev)+20]=zneg_rev
    zneg= gaussian_filter1d(zneg, sigma_gaus)


    peaks, _ = find_peaks(z, height=0)
    peaks_neg, _ = find_peaks(zneg, height=0)

    peaks_neg=peaks_neg-20-20
    peaks=peaks-20
    """

    textline_sum_along_width = textline_mask.sum(axis=1)

    y = textline_sum_along_width[:]
    y_padded = np.zeros(len(y) + 40)
    y_padded[20 : len(y) + 20] = y
    x = np.array(range(len(y)))

    peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)

    sigma_gaus = 8

    z = gaussian_filter1d(y_padded, sigma_gaus)
    zneg_rev = -y_padded + np.max(y_padded)

    zneg = np.zeros(len(zneg_rev) + 40)
    zneg[20 : len(zneg_rev) + 20] = zneg_rev
    zneg = gaussian_filter1d(zneg, sigma_gaus)

    peaks, _ = find_peaks(z, height=0)
    peaks_neg, _ = find_peaks(zneg, height=0)

    peaks_neg = peaks_neg - 20 - 20
    peaks = peaks - 20

    ##plt.plot(z)
    ##plt.show()

    if contours_main != None:
        areas_main = np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
        M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
        cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
        cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
        x_min_main = np.array([np.min(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])
        x_max_main = np.array([np.max(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

        y_min_main = np.array([np.min(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
        y_max_main = np.array([np.max(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])

    if len(contours_header) != None:
        areas_header = np.array([cv2.contourArea(contours_header[j]) for j in range(len(contours_header))])
        M_header = [cv2.moments(contours_header[j]) for j in range(len(contours_header))]
        cx_header = [(M_header[j]["m10"] / (M_header[j]["m00"] + 1e-32)) for j in range(len(M_header))]
        cy_header = [(M_header[j]["m01"] / (M_header[j]["m00"] + 1e-32)) for j in range(len(M_header))]

        x_min_header = np.array([np.min(contours_header[j][:, 0, 0]) for j in range(len(contours_header))])
        x_max_header = np.array([np.max(contours_header[j][:, 0, 0]) for j in range(len(contours_header))])

        y_min_header = np.array([np.min(contours_header[j][:, 0, 1]) for j in range(len(contours_header))])
        y_max_header = np.array([np.max(contours_header[j][:, 0, 1]) for j in range(len(contours_header))])
        # print(cy_main,'mainy')

    peaks_neg_new = []

    peaks_neg_new.append(0 + y_ref)
    for iii in range(len(peaks_neg)):
        peaks_neg_new.append(peaks_neg[iii] + y_ref)

    peaks_neg_new.append(textline_mask.shape[0] + y_ref)

    if len(cy_main) > 0 and np.max(cy_main) > np.max(peaks_neg_new):
        cy_main = np.array(cy_main) * (np.max(peaks_neg_new) / np.max(cy_main)) - 10

    if contours_main != None:
        indexer_main = np.array(range(len(contours_main)))

    if contours_main != None:
        len_main = len(contours_main)
    else:
        len_main = 0

    matrix_of_orders = np.zeros((len(contours_main) + len(contours_header), 5))

    matrix_of_orders[:, 0] = np.array(range(len(contours_main) + len(contours_header)))

    matrix_of_orders[: len(contours_main), 1] = 1
    matrix_of_orders[len(contours_main) :, 1] = 2

    matrix_of_orders[: len(contours_main), 2] = cx_main
    matrix_of_orders[len(contours_main) :, 2] = cx_header

    matrix_of_orders[: len(contours_main), 3] = cy_main
    matrix_of_orders[len(contours_main) :, 3] = cy_header

    matrix_of_orders[: len(contours_main), 4] = np.array(range(len(contours_main)))
    matrix_of_orders[len(contours_main) :, 4] = np.array(range(len(contours_header)))

    # print(peaks_neg_new,'peaks_neg_new')

    # print(matrix_of_orders,'matrix_of_orders')
    # print(peaks_neg_new,np.max(peaks_neg_new))
    final_indexers_sorted = []
    final_types = []
    final_index_type = []
    for i in range(len(peaks_neg_new) - 1):
        top = peaks_neg_new[i]
        down = peaks_neg_new[i + 1]

        # print(top,down,'topdown')

        indexes_in = matrix_of_orders[:, 0][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]
        cxs_in = matrix_of_orders[:, 2][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]
        cys_in = matrix_of_orders[:, 3][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]
        types_of_text = matrix_of_orders[:, 1][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]
        index_types_of_text = matrix_of_orders[:, 4][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]

        # print(top,down)
        # print(cys_in,'cyyyins')
        # print(indexes_in,'indexes')
        sorted_inside = np.argsort(cxs_in)

        ind_in_int = indexes_in[sorted_inside]
        ind_in_type = types_of_text[sorted_inside]
        ind_ind_type = index_types_of_text[sorted_inside]

        for j in range(len(ind_in_int)):
            final_indexers_sorted.append(int(ind_in_int[j]))
            final_types.append(int(ind_in_type[j]))
            final_index_type.append(int(ind_ind_type[j]))

    ##matrix_of_orders[:len_main,4]=final_indexers_sorted[:]

    # print(peaks_neg_new,'peaks')
    # print(final_indexers_sorted,'indexsorted')
    # print(final_types,'types')
    # print(final_index_type,'final_index_type')

    return final_indexers_sorted, matrix_of_orders, final_types, final_index_type

def implent_law_head_main_not_parallel(text_regions):
    # print(text_regions.shape)
    text_indexes = [1, 2]  # 1: main text , 2: header , 3: comments

    for t_i in text_indexes:
        textline_mask = text_regions[:, :] == t_i
        textline_mask = textline_mask * 255.0

        textline_mask = textline_mask.astype(np.uint8)
        textline_mask = np.repeat(textline_mask[:, :, np.newaxis], 3, axis=2)
        kernel = np.ones((5, 5), np.uint8)

        # print(type(textline_mask),np.unique(textline_mask),textline_mask.shape)
        imgray = cv2.cvtColor(textline_mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        if t_i == 1:
            contours_main, hirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # print(type(contours_main))
            areas_main = np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
            M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
            cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
            cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
            x_min_main = np.array([np.min(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])
            x_max_main = np.array([np.max(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

            y_min_main = np.array([np.min(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
            y_max_main = np.array([np.max(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
            # print(contours_main[0],np.shape(contours_main[0]),contours_main[0][:,0,0])
        elif t_i == 2:
            contours_header, hirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # print(type(contours_header))
            areas_header = np.array([cv2.contourArea(contours_header[j]) for j in range(len(contours_header))])
            M_header = [cv2.moments(contours_header[j]) for j in range(len(contours_header))]
            cx_header = [(M_header[j]["m10"] / (M_header[j]["m00"] + 1e-32)) for j in range(len(M_header))]
            cy_header = [(M_header[j]["m01"] / (M_header[j]["m00"] + 1e-32)) for j in range(len(M_header))]

            x_min_header = np.array([np.min(contours_header[j][:, 0, 0]) for j in range(len(contours_header))])
            x_max_header = np.array([np.max(contours_header[j][:, 0, 0]) for j in range(len(contours_header))])

            y_min_header = np.array([np.min(contours_header[j][:, 0, 1]) for j in range(len(contours_header))])
            y_max_header = np.array([np.max(contours_header[j][:, 0, 1]) for j in range(len(contours_header))])

    args = np.array(range(1, len(cy_header) + 1))
    args_main = np.array(range(1, len(cy_main) + 1))
    for jj in range(len(contours_main)):
        headers_in_main = [(cy_header > y_min_main[jj]) & ((cy_header < y_max_main[jj]))]
        mains_in_main = [(cy_main > y_min_main[jj]) & ((cy_main < y_max_main[jj]))]
        args_log = args * headers_in_main
        res = args_log[args_log > 0]
        res_true = res - 1

        args_log_main = args_main * mains_in_main
        res_main = args_log_main[args_log_main > 0]
        res_true_main = res_main - 1

        if len(res_true) > 0:
            sum_header = np.sum(areas_header[res_true])
            sum_main = np.sum(areas_main[res_true_main])
            if sum_main > sum_header:
                cnt_int = [contours_header[j] for j in res_true]
                text_regions = cv2.fillPoly(text_regions, pts=cnt_int, color=(1, 1, 1))
            else:
                cnt_int = [contours_main[j] for j in res_true_main]
                text_regions = cv2.fillPoly(text_regions, pts=cnt_int, color=(2, 2, 2))

    for jj in range(len(contours_header)):
        main_in_header = [(cy_main > y_min_header[jj]) & ((cy_main < y_max_header[jj]))]
        header_in_header = [(cy_header > y_min_header[jj]) & ((cy_header < y_max_header[jj]))]
        args_log = args_main * main_in_header
        res = args_log[args_log > 0]
        res_true = res - 1

        args_log_header = args * header_in_header
        res_header = args_log_header[args_log_header > 0]
        res_true_header = res_header - 1

        if len(res_true) > 0:

            sum_header = np.sum(areas_header[res_true_header])
            sum_main = np.sum(areas_main[res_true])

            if sum_main > sum_header:

                cnt_int = [contours_header[j] for j in res_true_header]
                text_regions = cv2.fillPoly(text_regions, pts=cnt_int, color=(1, 1, 1))
            else:
                cnt_int = [contours_main[j] for j in res_true]
                text_regions = cv2.fillPoly(text_regions, pts=cnt_int, color=(2, 2, 2))

    return text_regions


def return_hor_spliter_by_index(peaks_neg_fin_t, x_min_hor_some, x_max_hor_some):

    arg_min_hor_sort = np.argsort(x_min_hor_some)
    x_min_hor_some_sort = np.sort(x_min_hor_some)
    x_max_hor_some_sort = x_max_hor_some[arg_min_hor_sort]

    arg_minmax = np.array(range(len(peaks_neg_fin_t)))
    indexer_lines = []
    indexes_to_delete = []
    indexer_lines_deletions_len = []
    indexr_uniq_ind = []
    for i in range(len(x_min_hor_some_sort)):
        min_h = peaks_neg_fin_t - x_min_hor_some_sort[i]
        max_h = peaks_neg_fin_t - x_max_hor_some_sort[i]

        min_h[0] = min_h[0]  # +20
        max_h[len(max_h) - 1] = max_h[len(max_h) - 1]  ##-20

        min_h_neg = arg_minmax[(min_h < 0) & (np.abs(min_h) < 360)]
        max_h_neg = arg_minmax[(max_h >= 0) & (np.abs(max_h) < 360)]

        if len(min_h_neg) > 0 and len(max_h_neg) > 0:
            deletions = list(range(min_h_neg[0] + 1, max_h_neg[0]))
            unique_delets_int = []
            # print(deletions,len(deletions),'delii')
            if len(deletions) > 0:
                # print(deletions,len(deletions),'delii2')

                for j in range(len(deletions)):
                    indexes_to_delete.append(deletions[j])
                    # print(deletions,indexes_to_delete,'badiii')
                    unique_delets = np.unique(indexes_to_delete)
                    # print(min_h_neg[0],unique_delets)
                    unique_delets_int = unique_delets[unique_delets < min_h_neg[0]]

                indexer_lines_deletions_len.append(len(deletions))
                indexr_uniq_ind.append([deletions])

            else:
                indexer_lines_deletions_len.append(0)
                indexr_uniq_ind.append(-999)

            index_line_true = min_h_neg[0] - len(unique_delets_int)
            # print(index_line_true)
            if index_line_true > 0 and min_h_neg[0] >= 2:
                index_line_true = index_line_true
            else:
                index_line_true = min_h_neg[0]

            indexer_lines.append(index_line_true)

            if len(unique_delets_int) > 0:
                for dd in range(len(unique_delets_int)):
                    indexes_to_delete.append(unique_delets_int[dd])
        else:
            indexer_lines.append(-999)
            indexer_lines_deletions_len.append(-999)
            indexr_uniq_ind.append(-999)

    peaks_true = []
    for m in range(len(peaks_neg_fin_t)):
        if m in indexes_to_delete:
            pass
        else:
            peaks_true.append(peaks_neg_fin_t[m])
    return indexer_lines, peaks_true, arg_min_hor_sort, indexer_lines_deletions_len, indexr_uniq_ind

def combine_hor_lines_and_delete_cross_points_and_get_lines_features_back_new(img_p_in_ver, img_in_hor):

    # plt.imshow(img_in_hor)
    # plt.show()

    # img_p_in_ver = cv2.erode(img_p_in_ver, self.kernel, iterations=2)
    img_p_in_ver = img_p_in_ver.astype(np.uint8)
    img_p_in_ver = np.repeat(img_p_in_ver[:, :, np.newaxis], 3, axis=2)
    imgray = cv2.cvtColor(img_p_in_ver, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    contours_lines_ver, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    slope_lines_ver, dist_x_ver, x_min_main_ver, x_max_main_ver, cy_main_ver, slope_lines_org_ver, y_min_main_ver, y_max_main_ver, cx_main_ver = find_features_of_lines(contours_lines_ver)

    for i in range(len(x_min_main_ver)):
        img_p_in_ver[int(y_min_main_ver[i]) : int(y_min_main_ver[i]) + 30, int(cx_main_ver[i]) - 25 : int(cx_main_ver[i]) + 25, 0] = 0
        img_p_in_ver[int(y_max_main_ver[i]) - 30 : int(y_max_main_ver[i]), int(cx_main_ver[i]) - 25 : int(cx_main_ver[i]) + 25, 0] = 0

    # plt.imshow(img_p_in_ver[:,:,0])
    # plt.show()
    img_in_hor = img_in_hor.astype(np.uint8)
    img_in_hor = np.repeat(img_in_hor[:, :, np.newaxis], 3, axis=2)
    imgray = cv2.cvtColor(img_in_hor, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    contours_lines_hor, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    slope_lines_hor, dist_x_hor, x_min_main_hor, x_max_main_hor, cy_main_hor, slope_lines_org_hor, y_min_main_hor, y_max_main_hor, cx_main_hor = find_features_of_lines(contours_lines_hor)

    args_hor = np.array(range(len(slope_lines_hor)))
    all_args_uniq = contours_in_same_horizon(cy_main_hor)
    # print(all_args_uniq,'all_args_uniq')
    if len(all_args_uniq) > 0:
        if type(all_args_uniq[0]) is list:
            special_seperators = []
            contours_new = []
            for dd in range(len(all_args_uniq)):
                merged_all = None
                some_args = args_hor[all_args_uniq[dd]]
                some_cy = cy_main_hor[all_args_uniq[dd]]
                some_x_min = x_min_main_hor[all_args_uniq[dd]]
                some_x_max = x_max_main_hor[all_args_uniq[dd]]

                # img_in=np.zeros(seperators_closeup_n[:,:,2].shape)
                for jv in range(len(some_args)):

                    img_p_in = cv2.fillPoly(img_in_hor, pts=[contours_lines_hor[some_args[jv]]], color=(1, 1, 1))
                    img_p_in[int(np.mean(some_cy)) - 5 : int(np.mean(some_cy)) + 5, int(np.min(some_x_min)) : int(np.max(some_x_max))] = 1

                sum_dis = dist_x_hor[some_args].sum()
                diff_max_min_uniques = np.max(x_max_main_hor[some_args]) - np.min(x_min_main_hor[some_args])

                # print( sum_dis/float(diff_max_min_uniques) ,diff_max_min_uniques/float(img_p_in_ver.shape[1]),dist_x_hor[some_args].sum(),diff_max_min_uniques,np.mean( dist_x_hor[some_args]),np.std( dist_x_hor[some_args]) )

                if diff_max_min_uniques > sum_dis and ((sum_dis / float(diff_max_min_uniques)) > 0.85) and ((diff_max_min_uniques / float(img_p_in_ver.shape[1])) > 0.85) and np.std(dist_x_hor[some_args]) < (0.55 * np.mean(dist_x_hor[some_args])):
                    # print(dist_x_hor[some_args],dist_x_hor[some_args].sum(),np.min(x_min_main_hor[some_args]) ,np.max(x_max_main_hor[some_args]),'jalibdi')
                    # print(np.mean( dist_x_hor[some_args] ),np.std( dist_x_hor[some_args] ),np.var( dist_x_hor[some_args] ),'jalibdiha')
                    special_seperators.append(np.mean(cy_main_hor[some_args]))

        else:
            img_p_in = img_in_hor
            special_seperators = []
    else:
        img_p_in = img_in_hor
        special_seperators = []

    img_p_in_ver[:, :, 0][img_p_in_ver[:, :, 0] == 255] = 1
    # print(img_p_in_ver.shape,np.unique(img_p_in_ver[:,:,0]))

    # plt.imshow(img_p_in[:,:,0])
    # plt.show()

    # plt.imshow(img_p_in_ver[:,:,0])
    # plt.show()
    sep_ver_hor = img_p_in + img_p_in_ver
    # print(sep_ver_hor.shape,np.unique(sep_ver_hor[:,:,0]),'sep_ver_horsep_ver_horsep_ver_hor')
    # plt.imshow(sep_ver_hor[:,:,0])
    # plt.show()

    sep_ver_hor_cross = (sep_ver_hor[:, :, 0] == 2) * 1

    sep_ver_hor_cross = np.repeat(sep_ver_hor_cross[:, :, np.newaxis], 3, axis=2)
    sep_ver_hor_cross = sep_ver_hor_cross.astype(np.uint8)
    imgray = cv2.cvtColor(sep_ver_hor_cross, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours_cross, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cx_cross, cy_cross, _, _, _, _, _ = find_new_features_of_contoures(contours_cross)

    for ii in range(len(cx_cross)):
        img_p_in[int(cy_cross[ii]) - 30 : int(cy_cross[ii]) + 30, int(cx_cross[ii]) + 5 : int(cx_cross[ii]) + 40, 0] = 0
        img_p_in[int(cy_cross[ii]) - 30 : int(cy_cross[ii]) + 30, int(cx_cross[ii]) - 40 : int(cx_cross[ii]) - 4, 0] = 0

    # plt.imshow(img_p_in[:,:,0])
    # plt.show()

    return img_p_in[:, :, 0], special_seperators

def return_points_with_boundies(peaks_neg_fin, first_point, last_point):
    peaks_neg_tot = []
    peaks_neg_tot.append(first_point)
    for ii in range(len(peaks_neg_fin)):
        peaks_neg_tot.append(peaks_neg_fin[ii])
    peaks_neg_tot.append(last_point)
    return peaks_neg_tot

def find_number_of_columns_in_document(region_pre_p, num_col_classifier, pixel_lines, contours_h=None):

    seperators_closeup = ((region_pre_p[:, :, :] == pixel_lines)) * 1

    seperators_closeup[0:110, :, :] = 0
    seperators_closeup[seperators_closeup.shape[0] - 150 :, :, :] = 0

    kernel = np.ones((5, 5), np.uint8)

    seperators_closeup = seperators_closeup.astype(np.uint8)
    seperators_closeup = cv2.dilate(seperators_closeup, kernel, iterations=1)
    seperators_closeup = cv2.erode(seperators_closeup, kernel, iterations=1)

    ##plt.imshow(seperators_closeup[:,:,0])
    ##plt.show()
    seperators_closeup_new = np.zeros((seperators_closeup.shape[0], seperators_closeup.shape[1]))

    ##_,seperators_closeup_n=self.combine_hor_lines_and_delete_cross_points_and_get_lines_features_back(region_pre_p[:,:,0])
    seperators_closeup_n = np.copy(seperators_closeup)

    seperators_closeup_n = seperators_closeup_n.astype(np.uint8)
    ##plt.imshow(seperators_closeup_n[:,:,0])
    ##plt.show()

    seperators_closeup_n_binary = np.zeros((seperators_closeup_n.shape[0], seperators_closeup_n.shape[1]))
    seperators_closeup_n_binary[:, :] = seperators_closeup_n[:, :, 0]

    seperators_closeup_n_binary[:, :][seperators_closeup_n_binary[:, :] != 0] = 1
    # seperators_closeup_n_binary[:,:][seperators_closeup_n_binary[:,:]==0]=255
    # seperators_closeup_n_binary[:,:][seperators_closeup_n_binary[:,:]==-255]=0

    # seperators_closeup_n_binary=(seperators_closeup_n_binary[:,:]==2)*1

    # gray = cv2.cvtColor(seperators_closeup_n, cv2.COLOR_BGR2GRAY)

    # print(np.unique(seperators_closeup_n_binary))

    ##plt.imshow(seperators_closeup_n_binary)
    ##plt.show()

    # print( np.unique(gray),np.unique(seperators_closeup_n[:,:,1]) )

    gray = cv2.bitwise_not(seperators_closeup_n_binary)
    gray = gray.astype(np.uint8)

    ##plt.imshow(gray)
    ##plt.show()
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    ##plt.imshow(bw[:,:])
    ##plt.show()

    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    kernel = np.ones((5, 5), np.uint8)

    horizontal = cv2.dilate(horizontal, kernel, iterations=2)
    horizontal = cv2.erode(horizontal, kernel, iterations=2)
    # plt.imshow(horizontal)
    # plt.show()

    rows = vertical.shape[0]
    verticalsize = rows // 30
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    vertical = cv2.dilate(vertical, kernel, iterations=1)
    # Show extracted vertical lines

    horizontal, special_seperators = combine_hor_lines_and_delete_cross_points_and_get_lines_features_back_new(vertical, horizontal)

    ##plt.imshow(vertical)
    ##plt.show()
    # print(vertical.shape,np.unique(vertical),'verticalvertical')
    seperators_closeup_new[:, :][vertical[:, :] != 0] = 1
    seperators_closeup_new[:, :][horizontal[:, :] != 0] = 1

    ##plt.imshow(seperators_closeup_new)
    ##plt.show()
    ##seperators_closeup_n
    vertical = np.repeat(vertical[:, :, np.newaxis], 3, axis=2)
    vertical = vertical.astype(np.uint8)

    ##plt.plot(vertical[:,:,0].sum(axis=0))
    ##plt.show()

    # plt.plot(vertical[:,:,0].sum(axis=1))
    # plt.show()

    imgray = cv2.cvtColor(vertical, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    contours_line_vers, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slope_lines, dist_x, x_min_main, x_max_main, cy_main, slope_lines_org, y_min_main, y_max_main, cx_main = find_features_of_lines(contours_line_vers)
    # print(slope_lines,'vertical')
    args = np.array(range(len(slope_lines)))
    args_ver = args[slope_lines == 1]
    dist_x_ver = dist_x[slope_lines == 1]
    y_min_main_ver = y_min_main[slope_lines == 1]
    y_max_main_ver = y_max_main[slope_lines == 1]
    x_min_main_ver = x_min_main[slope_lines == 1]
    x_max_main_ver = x_max_main[slope_lines == 1]
    cx_main_ver = cx_main[slope_lines == 1]
    dist_y_ver = y_max_main_ver - y_min_main_ver
    len_y = seperators_closeup.shape[0] / 3.0

    # plt.imshow(horizontal)
    # plt.show()

    horizontal = np.repeat(horizontal[:, :, np.newaxis], 3, axis=2)
    horizontal = horizontal.astype(np.uint8)
    imgray = cv2.cvtColor(horizontal, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    contours_line_hors, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slope_lines, dist_x, x_min_main, x_max_main, cy_main, slope_lines_org, y_min_main, y_max_main, cx_main = find_features_of_lines(contours_line_hors)

    slope_lines_org_hor = slope_lines_org[slope_lines == 0]
    args = np.array(range(len(slope_lines)))
    len_x = seperators_closeup.shape[1] / 5.0

    dist_y = np.abs(y_max_main - y_min_main)

    args_hor = args[slope_lines == 0]
    dist_x_hor = dist_x[slope_lines == 0]
    y_min_main_hor = y_min_main[slope_lines == 0]
    y_max_main_hor = y_max_main[slope_lines == 0]
    x_min_main_hor = x_min_main[slope_lines == 0]
    x_max_main_hor = x_max_main[slope_lines == 0]
    dist_y_hor = dist_y[slope_lines == 0]
    cy_main_hor = cy_main[slope_lines == 0]

    args_hor = args_hor[dist_x_hor >= len_x / 2.0]
    x_max_main_hor = x_max_main_hor[dist_x_hor >= len_x / 2.0]
    x_min_main_hor = x_min_main_hor[dist_x_hor >= len_x / 2.0]
    cy_main_hor = cy_main_hor[dist_x_hor >= len_x / 2.0]
    y_min_main_hor = y_min_main_hor[dist_x_hor >= len_x / 2.0]
    y_max_main_hor = y_max_main_hor[dist_x_hor >= len_x / 2.0]
    dist_y_hor = dist_y_hor[dist_x_hor >= len_x / 2.0]

    slope_lines_org_hor = slope_lines_org_hor[dist_x_hor >= len_x / 2.0]
    dist_x_hor = dist_x_hor[dist_x_hor >= len_x / 2.0]

    matrix_of_lines_ch = np.zeros((len(cy_main_hor) + len(cx_main_ver), 10))

    matrix_of_lines_ch[: len(cy_main_hor), 0] = args_hor
    matrix_of_lines_ch[len(cy_main_hor) :, 0] = args_ver

    matrix_of_lines_ch[len(cy_main_hor) :, 1] = cx_main_ver

    matrix_of_lines_ch[: len(cy_main_hor), 2] = x_min_main_hor + 50  # x_min_main_hor+150
    matrix_of_lines_ch[len(cy_main_hor) :, 2] = x_min_main_ver

    matrix_of_lines_ch[: len(cy_main_hor), 3] = x_max_main_hor - 50  # x_max_main_hor-150
    matrix_of_lines_ch[len(cy_main_hor) :, 3] = x_max_main_ver

    matrix_of_lines_ch[: len(cy_main_hor), 4] = dist_x_hor
    matrix_of_lines_ch[len(cy_main_hor) :, 4] = dist_x_ver

    matrix_of_lines_ch[: len(cy_main_hor), 5] = cy_main_hor

    matrix_of_lines_ch[: len(cy_main_hor), 6] = y_min_main_hor
    matrix_of_lines_ch[len(cy_main_hor) :, 6] = y_min_main_ver

    matrix_of_lines_ch[: len(cy_main_hor), 7] = y_max_main_hor
    matrix_of_lines_ch[len(cy_main_hor) :, 7] = y_max_main_ver

    matrix_of_lines_ch[: len(cy_main_hor), 8] = dist_y_hor
    matrix_of_lines_ch[len(cy_main_hor) :, 8] = dist_y_ver

    matrix_of_lines_ch[len(cy_main_hor) :, 9] = 1

    if contours_h is not None:
        slope_lines_head, dist_x_head, x_min_main_head, x_max_main_head, cy_main_head, slope_lines_org_head, y_min_main_head, y_max_main_head, cx_main_head = find_features_of_lines(contours_h)
        matrix_l_n = np.zeros((matrix_of_lines_ch.shape[0] + len(cy_main_head), matrix_of_lines_ch.shape[1]))
        matrix_l_n[: matrix_of_lines_ch.shape[0], :] = np.copy(matrix_of_lines_ch[:, :])
        args_head = np.array(range(len(cy_main_head))) + len(cy_main_hor)

        matrix_l_n[matrix_of_lines_ch.shape[0] :, 0] = args_head
        matrix_l_n[matrix_of_lines_ch.shape[0] :, 2] = x_min_main_head + 30
        matrix_l_n[matrix_of_lines_ch.shape[0] :, 3] = x_max_main_head - 30

        matrix_l_n[matrix_of_lines_ch.shape[0] :, 4] = dist_x_head

        matrix_l_n[matrix_of_lines_ch.shape[0] :, 5] = y_min_main_head - 3 - 8
        matrix_l_n[matrix_of_lines_ch.shape[0] :, 6] = y_min_main_head - 5 - 8
        matrix_l_n[matrix_of_lines_ch.shape[0] :, 7] = y_min_main_head + 1 - 8
        matrix_l_n[matrix_of_lines_ch.shape[0] :, 8] = 4

        matrix_of_lines_ch = np.copy(matrix_l_n)

    # print(matrix_of_lines_ch)

    """



    seperators_closeup=seperators_closeup.astype(np.uint8)
    imgray = cv2.cvtColor(seperators_closeup, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    contours_lines,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    slope_lines,dist_x, x_min_main ,x_max_main ,cy_main,slope_lines_org,y_min_main, y_max_main, cx_main=find_features_of_lines(contours_lines)

    slope_lines_org_hor=slope_lines_org[slope_lines==0]
    args=np.array( range(len(slope_lines) ))
    len_x=seperators_closeup.shape[1]/4.0

    args_hor=args[slope_lines==0]
    dist_x_hor=dist_x[slope_lines==0]
    x_min_main_hor=x_min_main[slope_lines==0]
    x_max_main_hor=x_max_main[slope_lines==0]
    cy_main_hor=cy_main[slope_lines==0]

    args_hor=args_hor[dist_x_hor>=len_x/2.0]
    x_max_main_hor=x_max_main_hor[dist_x_hor>=len_x/2.0]
    x_min_main_hor=x_min_main_hor[dist_x_hor>=len_x/2.0]
    cy_main_hor=cy_main_hor[dist_x_hor>=len_x/2.0]
    slope_lines_org_hor=slope_lines_org_hor[dist_x_hor>=len_x/2.0]


    slope_lines_org_hor=slope_lines_org_hor[np.abs(slope_lines_org_hor)<1.2]
    slope_mean_hor=np.mean(slope_lines_org_hor)



    args_ver=args[slope_lines==1]
    y_min_main_ver=y_min_main[slope_lines==1]
    y_max_main_ver=y_max_main[slope_lines==1]
    x_min_main_ver=x_min_main[slope_lines==1]
    x_max_main_ver=x_max_main[slope_lines==1]
    cx_main_ver=cx_main[slope_lines==1]
    dist_y_ver=y_max_main_ver-y_min_main_ver
    len_y=seperators_closeup.shape[0]/3.0



    print(matrix_of_lines_ch[:,8][matrix_of_lines_ch[:,9]==0],'khatlarrrr')
    args_main_spliters=matrix_of_lines_ch[:,0][ (matrix_of_lines_ch[:,9]==0) & ((matrix_of_lines_ch[:,8]<=290)) & ((matrix_of_lines_ch[:,2]<=.16*region_pre_p.shape[1])) & ((matrix_of_lines_ch[:,3]>=.84*region_pre_p.shape[1]))]

    cy_main_spliters=matrix_of_lines_ch[:,5][ (matrix_of_lines_ch[:,9]==0) & ((matrix_of_lines_ch[:,8]<=290)) & ((matrix_of_lines_ch[:,2]<=.16*region_pre_p.shape[1])) & ((matrix_of_lines_ch[:,3]>=.84*region_pre_p.shape[1]))]
    """

    cy_main_spliters = cy_main_hor[(x_min_main_hor <= 0.16 * region_pre_p.shape[1]) & (x_max_main_hor >= 0.84 * region_pre_p.shape[1])]

    cy_main_spliters = np.array(list(cy_main_spliters) + list(special_seperators))

    if contours_h is not None:
        try:
            cy_main_spliters_head = cy_main_head[(x_min_main_head <= 0.16 * region_pre_p.shape[1]) & (x_max_main_head >= 0.84 * region_pre_p.shape[1])]
            cy_main_spliters = np.array(list(cy_main_spliters) + list(cy_main_spliters_head))
        except:
            pass
    args_cy_spliter = np.argsort(cy_main_spliters)

    cy_main_spliters_sort = cy_main_spliters[args_cy_spliter]

    spliter_y_new = []
    spliter_y_new.append(0)
    for i in range(len(cy_main_spliters_sort)):
        spliter_y_new.append(cy_main_spliters_sort[i])

    spliter_y_new.append(region_pre_p.shape[0])

    spliter_y_new_diff = np.diff(spliter_y_new) / float(region_pre_p.shape[0]) * 100

    args_big_parts = np.array(range(len(spliter_y_new_diff)))[spliter_y_new_diff > 22]

    regions_without_seperators = return_regions_without_seperators(region_pre_p)

    ##print(args_big_parts,'args_big_parts')
    # image_page_otsu=otsu_copy(image_page_deskewd)
    # print(np.unique(image_page_otsu[:,:,0]))
    # image_page_background_zero=self.image_change_background_pixels_to_zero(image_page_otsu)

    length_y_threshold = regions_without_seperators.shape[0] / 4.0

    num_col_fin = 0
    peaks_neg_fin_fin = []

    for iteils in args_big_parts:

        regions_without_seperators_teil = regions_without_seperators[int(spliter_y_new[iteils]) : int(spliter_y_new[iteils + 1]), :, 0]
        # image_page_background_zero_teil=image_page_background_zero[int(spliter_y_new[iteils]):int(spliter_y_new[iteils+1]),:]

        # print(regions_without_seperators_teil.shape)
        ##plt.imshow(regions_without_seperators_teil)
        ##plt.show()

        # num_col, peaks_neg_fin=find_num_col(regions_without_seperators_teil,multiplier=6.0)

        # regions_without_seperators_teil=cv2.erode(regions_without_seperators_teil,kernel,iterations = 3)
        #
        num_col, peaks_neg_fin = find_num_col(regions_without_seperators_teil, multiplier=7.0)

        if num_col > num_col_fin:
            num_col_fin = num_col
            peaks_neg_fin_fin = peaks_neg_fin
        """
        #print(length_y_vertical_lines,length_y_threshold,'x_center_of_ver_linesx_center_of_ver_linesx_center_of_ver_lines')
        if len(cx_main_ver)>0 and len( dist_y_ver[dist_y_ver>=length_y_threshold] ) >=1:
            num_col, peaks_neg_fin=find_num_col(regions_without_seperators_teil,multiplier=6.0)
        else:
            #plt.imshow(image_page_background_zero_teil)
            #plt.show()
            #num_col, peaks_neg_fin=find_num_col_only_image(image_page_background_zero,multiplier=2.4)#2.3)
            num_col, peaks_neg_fin=find_num_col_only_image(image_page_background_zero_teil,multiplier=3.4)#2.3)

            print(num_col,'birda')
            if num_col>0:
                pass
            elif num_col==0:
                print(num_col,'birda2222')
                num_col_regions, peaks_neg_fin_regions=find_num_col(regions_without_seperators_teil,multiplier=10.0)
                if num_col_regions==0:
                    pass
                else:

                    num_col=num_col_regions
                    peaks_neg_fin=peaks_neg_fin_regions[:]
        """

        # print(num_col+1,'num colmsssssssss')

    if len(args_big_parts) == 1 and (len(peaks_neg_fin_fin) + 1) < num_col_classifier:
        peaks_neg_fin = find_num_col_by_vertical_lines(vertical)
        peaks_neg_fin = peaks_neg_fin[peaks_neg_fin >= 500]
        peaks_neg_fin = peaks_neg_fin[peaks_neg_fin <= (vertical.shape[1] - 500)]
        peaks_neg_fin_fin = peaks_neg_fin[:]

        # print(peaks_neg_fin_fin,'peaks_neg_fin_fintaza')

    return num_col_fin, peaks_neg_fin_fin, matrix_of_lines_ch, spliter_y_new, seperators_closeup_n

def return_boxes_of_images_by_order_of_reading_new(spliter_y_new, regions_without_seperators, matrix_of_lines_ch):
    boxes = []

    # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
    # holes in the text and also finding spliter which covers more than one columns.
    for i in range(len(spliter_y_new) - 1):
        # print(spliter_y_new[i],spliter_y_new[i+1])
        matrix_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 6] > spliter_y_new[i]) & (matrix_of_lines_ch[:, 7] < spliter_y_new[i + 1])]
        # print(len( matrix_new[:,9][matrix_new[:,9]==1] ))

        # print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')

        # check to see is there any vertical seperator to find holes.
        if 1 > 0:  # len( matrix_new[:,9][matrix_new[:,9]==1] )>0 and np.max(matrix_new[:,8][matrix_new[:,9]==1])>=0.1*(np.abs(spliter_y_new[i+1]-spliter_y_new[i] )):

            # org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
            # org_img_dichte=org_img_dichte-np.min(org_img_dichte)
            ##plt.figure(figsize=(20,20))
            ##plt.plot(org_img_dichte)
            ##plt.show()
            ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)

            # print(int(spliter_y_new[i]),int(spliter_y_new[i+1]),'firssst')

            # plt.imshow(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:])
            # plt.show()
            try:
                num_col, peaks_neg_fin = find_num_col(regions_without_seperators[int(spliter_y_new[i]) : int(spliter_y_new[i + 1]), :], multiplier=7.0)
            except:
                peaks_neg_fin = []

            # print(peaks_neg_fin,'peaks_neg_fin')
            # num_col, peaks_neg_fin=find_num_col(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=7.0)
            x_min_hor_some = matrix_new[:, 2][(matrix_new[:, 9] == 0)]
            x_max_hor_some = matrix_new[:, 3][(matrix_new[:, 9] == 0)]
            cy_hor_some = matrix_new[:, 5][(matrix_new[:, 9] == 0)]
            arg_org_hor_some = matrix_new[:, 0][(matrix_new[:, 9] == 0)]

            peaks_neg_tot = return_points_with_boundies(peaks_neg_fin, 0, regions_without_seperators[:, :].shape[1])

            start_index_of_hor, newest_peaks, arg_min_hor_sort, lines_length_dels, lines_indexes_deleted = return_hor_spliter_by_index_for_without_verticals(peaks_neg_tot, x_min_hor_some, x_max_hor_some)

            arg_org_hor_some_sort = arg_org_hor_some[arg_min_hor_sort]

            start_index_of_hor_with_subset = [start_index_of_hor[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]  # start_index_of_hor[lines_length_dels>0]
            arg_min_hor_sort_with_subset = [arg_min_hor_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]
            lines_indexes_deleted_with_subset = [lines_indexes_deleted[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]
            lines_length_dels_with_subset = [lines_length_dels[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]

            arg_org_hor_some_sort_subset = [arg_org_hor_some_sort[vij] for vij in range(len(start_index_of_hor)) if lines_length_dels[vij] > 0]

            # arg_min_hor_sort_with_subset=arg_min_hor_sort[lines_length_dels>0]
            # lines_indexes_deleted_with_subset=lines_indexes_deleted[lines_length_dels>0]
            # lines_length_dels_with_subset=lines_length_dels[lines_length_dels>0]

            vahid_subset = np.zeros((len(start_index_of_hor_with_subset), len(start_index_of_hor_with_subset))) - 1
            for kkk1 in range(len(start_index_of_hor_with_subset)):

                index_del_sub = np.unique(lines_indexes_deleted_with_subset[kkk1])

                for kkk2 in range(len(start_index_of_hor_with_subset)):

                    if set(lines_indexes_deleted_with_subset[kkk2][0]) < set(lines_indexes_deleted_with_subset[kkk1][0]):
                        vahid_subset[kkk1, kkk2] = kkk1
                    else:
                        pass
                # print(set(lines_indexes_deleted[kkk2][0]), set(lines_indexes_deleted[kkk1][0]))

            # check the len of matrix if it has no length means that there is no spliter at all

            if len(vahid_subset > 0):
                # print('hihoo')

                # find parenets args
                line_int = np.zeros(vahid_subset.shape[0])

                childs_id = []
                arg_child = []
                for li in range(vahid_subset.shape[0]):
                    # print(vahid_subset[:,li])
                    if np.all(vahid_subset[:, li] == -1):
                        line_int[li] = -1
                    else:
                        line_int[li] = 1

                        # childs_args_in=[ idd for idd in range(vahid_subset.shape[0]) if vahid_subset[idd,li]!=-1]
                        # helpi=[]
                        # for nad in range(len(childs_args_in)):
                        #    helpi.append(arg_min_hor_sort_with_subset[childs_args_in[nad]])

                        arg_child.append(arg_min_hor_sort_with_subset[li])

                # line_int=vahid_subset[0,:]

                arg_parent = [arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] == -1]
                start_index_of_hor_parent = [start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] == -1]
                # arg_parent=[lines_indexes_deleted_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                # arg_parent=[lines_length_dels_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]

                # arg_child=[arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]!=-1]
                start_index_of_hor_child = [start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] != -1]

                cy_hor_some_sort = cy_hor_some[arg_parent]

                # print(start_index_of_hor, lines_length_dels ,lines_indexes_deleted,'zartt')

                # args_indexes=np.array(range(len(start_index_of_hor) ))

                newest_y_spliter_tot = []

                for tj in range(len(newest_peaks) - 1):
                    newest_y_spliter = []
                    newest_y_spliter.append(spliter_y_new[i])
                    if tj in np.unique(start_index_of_hor_parent):
                        # print(cy_hor_some_sort)
                        cy_help = np.array(cy_hor_some_sort)[np.array(start_index_of_hor_parent) == tj]
                        cy_help_sort = np.sort(cy_help)

                        # print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                        for mj in range(len(cy_help_sort)):
                            newest_y_spliter.append(cy_help_sort[mj])
                    newest_y_spliter.append(spliter_y_new[i + 1])

                    newest_y_spliter_tot.append(newest_y_spliter)

            else:
                line_int = []
                newest_y_spliter_tot = []

                for tj in range(len(newest_peaks) - 1):
                    newest_y_spliter = []
                    newest_y_spliter.append(spliter_y_new[i])

                    newest_y_spliter.append(spliter_y_new[i + 1])

                    newest_y_spliter_tot.append(newest_y_spliter)

            # if line_int is all -1 means that big spliters have no child and we can easily go through
            if np.all(np.array(line_int) == -1):
                for j in range(len(newest_peaks) - 1):
                    newest_y_spliter = newest_y_spliter_tot[j]

                    for n in range(len(newest_y_spliter) - 1):
                        # print(j,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'maaaa')
                        ##plt.imshow(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]])
                        ##plt.show()

                        # print(matrix_new[:,0][ (matrix_new[:,9]==1 )])
                        for jvt in matrix_new[:, 0][(matrix_new[:, 9] == 1) & (matrix_new[:, 6] > newest_y_spliter[n]) & (matrix_new[:, 7] < newest_y_spliter[n + 1]) & ((matrix_new[:, 1]) < newest_peaks[j + 1]) & ((matrix_new[:, 1]) > newest_peaks[j])]:
                            pass

                            ###plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                        # print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                        matrix_new_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter[n]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter[n + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < newest_peaks[j + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > newest_peaks[j])]
                        # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                        if 1 > 0:  # len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                            # print( int(newest_y_spliter[n]),int(newest_y_spliter[n+1]),newest_peaks[j],newest_peaks[j+1] )
                            try:
                                num_col_sub, peaks_neg_fin_sub = find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=7.0)
                            except:
                                peaks_neg_fin_sub = []
                        else:
                            peaks_neg_fin_sub = []

                        peaks_sub = []
                        peaks_sub.append(newest_peaks[j])

                        for kj in range(len(peaks_neg_fin_sub)):
                            peaks_sub.append(peaks_neg_fin_sub[kj] + newest_peaks[j])

                        peaks_sub.append(newest_peaks[j + 1])

                        # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                        for kh in range(len(peaks_sub) - 1):
                            boxes.append([peaks_sub[kh], peaks_sub[kh + 1], newest_y_spliter[n], newest_y_spliter[n + 1]])

            else:
                for j in range(len(newest_peaks) - 1):

                    newest_y_spliter = newest_y_spliter_tot[j]

                    if j in start_index_of_hor_parent:

                        x_min_ch = x_min_hor_some[arg_child]
                        x_max_ch = x_max_hor_some[arg_child]
                        cy_hor_some_sort_child = cy_hor_some[arg_child]
                        cy_hor_some_sort_child = np.sort(cy_hor_some_sort_child)

                        for n in range(len(newest_y_spliter) - 1):

                            cy_child_in = cy_hor_some_sort_child[(cy_hor_some_sort_child > newest_y_spliter[n]) & (cy_hor_some_sort_child < newest_y_spliter[n + 1])]

                            if len(cy_child_in) > 0:
                                try:
                                    num_col_ch, peaks_neg_ch = find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=7.0)
                                except:
                                    peaks_neg_ch = []
                                # print(peaks_neg_ch,'mizzzz')
                                # peaks_neg_ch=[]
                                # for djh in range(len(peaks_neg_ch)):
                                #    peaks_neg_ch.append( peaks_neg_ch[djh]+newest_peaks[j] )

                                peaks_neg_ch_tot = return_points_with_boundies(peaks_neg_ch, newest_peaks[j], newest_peaks[j + 1])

                                ss_in_ch, nst_p_ch, arg_n_ch, lines_l_del_ch, lines_in_del_ch = return_hor_spliter_by_index_for_without_verticals(peaks_neg_ch_tot, x_min_ch, x_max_ch)

                                newest_y_spliter_ch_tot = []

                                for tjj in range(len(nst_p_ch) - 1):
                                    newest_y_spliter_new = []
                                    newest_y_spliter_new.append(newest_y_spliter[n])
                                    if tjj in np.unique(ss_in_ch):

                                        # print(tj,cy_hor_some_sort,start_index_of_hor,cy_help,'maashhaha')
                                        for mjj in range(len(cy_child_in)):
                                            newest_y_spliter_new.append(cy_child_in[mjj])
                                    newest_y_spliter_new.append(newest_y_spliter[n + 1])

                                    newest_y_spliter_ch_tot.append(newest_y_spliter_new)

                                for jn in range(len(nst_p_ch) - 1):
                                    newest_y_spliter_h = newest_y_spliter_ch_tot[jn]

                                    for nd in range(len(newest_y_spliter_h) - 1):

                                        matrix_new_new2 = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter_h[nd]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter_h[nd + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < nst_p_ch[jn + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > nst_p_ch[jn])]
                                        # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                        if 1 > 0:  # len( matrix_new_new2[:,9][matrix_new_new2[:,9]==1] )>0 and np.max(matrix_new_new2[:,8][matrix_new_new2[:,9]==1])>=0.2*(np.abs(newest_y_spliter_h[nd+1]-newest_y_spliter_h[nd] )):
                                            try:
                                                num_col_sub_ch, peaks_neg_fin_sub_ch = find_num_col(regions_without_seperators[int(newest_y_spliter_h[nd]) : int(newest_y_spliter_h[nd + 1]), nst_p_ch[jn] : nst_p_ch[jn + 1]], multiplier=7.0)
                                            except:
                                                peaks_neg_fin_sub_ch = []

                                        else:
                                            peaks_neg_fin_sub_ch = []

                                        peaks_sub_ch = []
                                        peaks_sub_ch.append(nst_p_ch[jn])

                                        for kjj in range(len(peaks_neg_fin_sub_ch)):
                                            peaks_sub_ch.append(peaks_neg_fin_sub_ch[kjj] + nst_p_ch[jn])

                                        peaks_sub_ch.append(nst_p_ch[jn + 1])

                                        # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                        for khh in range(len(peaks_sub_ch) - 1):
                                            boxes.append([peaks_sub_ch[khh], peaks_sub_ch[khh + 1], newest_y_spliter_h[nd], newest_y_spliter_h[nd + 1]])

                            else:

                                matrix_new_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter[n]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter[n + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < newest_peaks[j + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > newest_peaks[j])]
                                # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                                if 1 > 0:  # len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                    try:
                                        num_col_sub, peaks_neg_fin_sub = find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=7.0)
                                    except:
                                        peaks_neg_fin_sub = []
                                else:
                                    peaks_neg_fin_sub = []

                                peaks_sub = []
                                peaks_sub.append(newest_peaks[j])

                                for kj in range(len(peaks_neg_fin_sub)):
                                    peaks_sub.append(peaks_neg_fin_sub[kj] + newest_peaks[j])

                                peaks_sub.append(newest_peaks[j + 1])

                                # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                                for kh in range(len(peaks_sub) - 1):
                                    boxes.append([peaks_sub[kh], peaks_sub[kh + 1], newest_y_spliter[n], newest_y_spliter[n + 1]])

                    else:
                        for n in range(len(newest_y_spliter) - 1):

                            # plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                            # print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                            matrix_new_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter[n]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter[n + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < newest_peaks[j + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > newest_peaks[j])]
                            # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                            if 1 > 0:  # len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                try:
                                    num_col_sub, peaks_neg_fin_sub = find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=5.0)
                                except:
                                    peaks_neg_fin_sub = []
                            else:
                                peaks_neg_fin_sub = []

                            peaks_sub = []
                            peaks_sub.append(newest_peaks[j])

                            for kj in range(len(peaks_neg_fin_sub)):
                                peaks_sub.append(peaks_neg_fin_sub[kj] + newest_peaks[j])

                            peaks_sub.append(newest_peaks[j + 1])

                            # peaks_sub=return_points_with_boundies(peaks_neg_fin_sub+newest_peaks[j],newest_peaks[j], newest_peaks[j+1])

                            for kh in range(len(peaks_sub) - 1):
                                boxes.append([peaks_sub[kh], peaks_sub[kh + 1], newest_y_spliter[n], newest_y_spliter[n + 1]])

        else:
            boxes.append([0, regions_without_seperators[:, :].shape[1], spliter_y_new[i], spliter_y_new[i + 1]])

    return boxes

