from typing import Tuple
from logging import getLogger
import time
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
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
        peak_points, x_min_hor_some, x_max_hor_some, cy_hor_some, y_max_hor_some):
    """
    Analyse which separators overlap multiple column candidates,
    and how they overlap each other.

    Ignore separators not spanning multiple columns.

    For the separators to be returned, try to join them when they are directly
    adjacent horizontally but nearby vertically (and thus mutually compatible).
    Also, mark any separators that already span the full width.

    Furthermore, identify which pairs of (unjoined) separators span subsets of columns
    of each other (disregarding vertical positions). Referring, respectively, to the
    superset separators as  "mothers" and to the subset separators as "children",
    retrieve information on which columns are spanned by separators with no mother,
    and which columns are spanned by their children (if any).

    Moreover, determine if there is any (column) overlap among the multi-span separators
    with no mother, specifically (and thus, no simple box separation is possible).

    Arguments:
        * the x column coordinates
        * the x start column index of the raw separators
        * the x end column index of the raw separators
        * the y center coordinate of the raw separators
        * the y end coordinate of the raw separators

    Returns:
        a tuple of:
        * whether any top-level (no-mother) multi-span separators overlap each other
        * the x start column index of the resulting multi-span separators
        * the x end column index of the resulting multi-span separators
        * the y center coordinate of the resulting multi-span separators
        * the y end coordinate of the resulting multi-span separators
        * the y center (for 1 representative) of the top-level (no-mother) multi-span separators
        * the x start column index of the top-level (no-mother) multi-span separators
        * the x end column index of the top-level (no-mother) multi-span separators
        * whether any multi-span separators have super-spans of other (child) multi-span separators
        * the y center (for 1 representative) of the top-level (no-mother) multi-span separators
          which have super-spans of other (child) multi-span separators
        * the x start column index of the top-level multi-span separators
          which have super-spans of other (child) multi-span separators
        * the x end column index of the top-level multi-span separators
          which have super-spans of other (child) multi-span separators
        * indexes of multi-span separators with full-width span
    """

    x_start=[]
    x_end=[]
    len_sep=[]
    y_mid=[]
    y_max=[]
    new_main_sep_y=[]
    indexer=0
    for i in range(len(x_min_hor_some)):
        #print(indexer, "%d:%d" % (x_min_hor_some[i], x_max_hor_some[i]), cy_hor_some[i])
        starting = x_min_hor_some[i] - peak_points
        min_start = np.flatnonzero(starting >= 0)[-1] # last left-of
        ending = x_max_hor_some[i] - peak_points
        max_end = np.flatnonzero(ending < 0)[0] # first right-of
        #print(indexer, "%d:%d" % (min_start, max_end))

        if (max_end-min_start)>=2:
            # column range of separator spans more than one column candidate
            if (max_end-min_start)==(len(peak_points)-1):
                # all columns (i.e. could be true new y splitter)
                new_main_sep_y.append(indexer)

            #print((max_end-min_start),len(peak_points),'(max_end-min_start)')
            y_mid.append(cy_hor_some[i])
            y_max.append(y_max_hor_some[i])
            x_end.append(max_end)
            x_start.append(min_start)
            len_sep.append(max_end-min_start)
            indexer+=1
    #print(x_start,'x_start')
    #print(x_end,'x_end')

    x_start_returned = np.array(x_start, dtype=int)
    x_end_returned = np.array(x_end, dtype=int)
    y_mid_returned = np.array(y_mid, dtype=int)
    y_max_returned = np.array(y_max, dtype=int)
    #print(y_mid_returned,'y_mid_returned')
    #print(x_start_returned,'x_start_returned')
    #print(x_end_returned,'x_end_returned')

    # join/elongate separators if follow-up x and similar y
    sep_pairs = contours_in_same_horizon(y_mid_returned)
    if len(sep_pairs):
        #print('burda')
        args_to_be_unified = set()
        y_mid_unified = []
        y_max_unified = []
        x_start_unified = []
        x_end_unified = []
        for pair in sep_pairs:
            if (not np.array_equal(*x_start_returned[pair]) and
                not np.array_equal(*x_end_returned[pair]) and
                # immediately adjacent columns?
                np.diff(x_end_returned[pair] -
                        x_start_returned[pair])[0] in [1, -1]):

                args_to_be_unified.union(set(pair))
                y_mid_unified.append(np.min(y_mid_returned[pair]))
                y_max_unified.append(np.max(y_max_returned[pair]))
                x_start_unified.append(np.min(x_start_returned[pair]))
                x_end_unified.append(np.max(x_end_returned[pair]))
                #print(pair,'pair')
                #print(x_start_returned[pair],'x_s_same_hor')
                #print(x_end_returned[pair],'x_e_same_hor')
        #print(y_mid_unified,'y_mid_unified')
        #print(y_max_unified,'y_max_unified')
        #print(x_start_unified,'x_s_unified')
        #print(x_end_unified,'x_e_selected')
        #print('#############################')

        if len(y_mid_unified):
            args_lines_not_unified = np.setdiff1d(np.arange(len(y_mid_returned)),
                                                  list(args_to_be_unified), assume_unique=True)
            #print(args_lines_not_unified,'args_lines_not_unified')
            x_start_returned = np.append(x_start_returned[args_lines_not_unified],
                                         x_start_unified, axis=0)
            x_end_returned = np.append(x_end_returned[args_lines_not_unified],
                                       x_end_unified, axis=0)
            y_mid_returned = np.append(y_mid_returned[args_lines_not_unified],
                                       y_mid_unified, axis=0)
            y_max_returned = np.append(y_max_returned[args_lines_not_unified],
                                        y_max_unified, axis=0)
    #print(y_mid_returned,'y_mid_returned2')
    #print(x_start_returned,'x_start_returned2')
    #print(x_end_returned,'x_end_returned2')

    #print(new_main_sep_y,'new_main_sep_y')
    #print(x_start,'x_start')
    #print(x_end,'x_end')
    x_start = np.array(x_start)
    x_end = np.array(x_end)
    y_mid = np.array(y_mid)
    if len(new_main_sep_y):
        # some full-width multi-span separators exist, so
        # restrict the y range of separators to search for
        # mutual overlaps to only those within the largest
        # y strip between adjacent multi-span separators
        # that involve at least one such full-width seps.
        # (does not affect the separators to be returned)
        min_ys=np.min(y_mid)
        max_ys=np.max(y_mid)
        #print(min_ys,'min_ys')
        #print(max_ys,'max_ys')

        y_mains0 = list(y_mid[new_main_sep_y])
        y_mains = [min_ys] + y_mains0 + [max_ys]

        y_mains = np.sort(y_mains)
        argm = np.argmax(np.diff(y_mains))
        y_mid_new = y_mains[argm]
        y_mid_next_new = y_mains[argm + 1]

        #print(y_mid_new,argm,'y_mid_new')
        #print(y_mid_next_new,argm+1,'y_mid_next_new')
        #print(y_mid[new_main_sep_y],new_main_sep_y,'yseps')
        x_start=np.array(x_start)
        x_end=np.array(x_end)
        y_mid=np.array(y_mid)
        # iff either boundary is itself not a full-width separator,
        # then include it in the range of separators to be kept
        if y_mid_new in y_mains0:
            where = y_mid > y_mid_new
        else:
            where = y_mid >= y_mid_new
        if y_mid_next_new in y_mains0:
            where &= y_mid < y_mid_next_new
        else:
            where &= y_mid <= y_mid_next_new
        x_start = x_start[where]
        x_end = x_end[where]
        y_mid = y_mid[where]
    #print(x_start,'x_start')
    #print(x_end,'x_end')

    # remove redundant separators that span the same columns
    # (keeping only 1 representative each)
    deleted = set()
    for index_i in range(len(x_start) - 1):
        nodes_i = set(range(x_start[index_i], x_end[index_i] + 1))
        #print(nodes_i, "nodes_i")
        for index_j in range(index_i + 1, len(x_start)):
            nodes_j = set(range(x_start[index_j], x_end[index_j] + 1))
            #print(nodes_j, "nodes_j")
            if nodes_i == nodes_j:
                deleted.add(index_j)
    #print(deleted,"deleted")
    remained_sep_indexes = set(range(len(x_start))) - deleted
    #print(remained_sep_indexes,'remained_sep_indexes')

    # determine which separators span which columns
    mother = [] # whether the respective separator has a mother separator
    child = [] # whether the respective separator has a child separator
    for index_i in remained_sep_indexes:
        have_mother=0
        have_child=0
        nodes_i = set(range(x_start[index_i], x_end[index_i] + 1))
        for index_j in remained_sep_indexes:
            nodes_j = set(range(x_start[index_j], x_end[index_j] + 1))
            if nodes_i < nodes_j:
                have_mother=1
            if nodes_i > nodes_j:
                have_child=1
        mother.append(have_mother)
        child.append(have_child)
    #print(mother, "mother")
    #print(child, "child")

    mother = np.array(mother)
    child = np.array(child)
    #print(mother,'mother')
    #print(child,'child')
    remained_sep_indexes = np.array(list(remained_sep_indexes))
    #print(len(remained_sep_indexes))
    #print(len(remained_sep_indexes),len(x_start),len(x_end),len(y_mid),'lens')

    reading_order_type = 0
    if len(remained_sep_indexes):
        #print(np.array(remained_sep_indexes),'np.array(remained_sep_indexes)')
        #print(np.array(mother),'mother')
        remained_sep_indexes_without_mother = remained_sep_indexes[mother==0]
        remained_sep_indexes_with_child_without_mother = remained_sep_indexes[(mother==0) & (child==1)]
        #print(remained_sep_indexes_without_mother,'remained_sep_indexes_without_mother')
        #print(remained_sep_indexes_without_mother,'remained_sep_indexes_without_mother')

        x_end_with_child_without_mother = x_end[remained_sep_indexes_with_child_without_mother]
        x_start_with_child_without_mother = x_start[remained_sep_indexes_with_child_without_mother]
        y_mid_with_child_without_mother = y_mid[remained_sep_indexes_with_child_without_mother]

        x_end_without_mother = x_end[remained_sep_indexes_without_mother]
        x_start_without_mother = x_start[remained_sep_indexes_without_mother]
        y_mid_without_mother = y_mid[remained_sep_indexes_without_mother]

        if len(remained_sep_indexes_without_mother)>=2:
            for i in range(len(remained_sep_indexes_without_mother)-1):
                index_i = remained_sep_indexes_without_mother[i]
                nodes_i = set(range(x_start[index_i], x_end[index_i] + 1))
                #print(index_i, nodes_i, "nodes_i without mother")
                for j in range(i + 1, len(remained_sep_indexes_without_mother)):
                    index_j = remained_sep_indexes_without_mother[j]
                    nodes_j = set(range(x_start[index_j], x_end[index_j] + 1))
                    #print(index_j, nodes_j, "nodes_j without mother")
                    if nodes_i - nodes_j != nodes_i:
                        #print("type=1")
                        reading_order_type = 1
    else:
        y_mid_without_mother = np.zeros(0, int)
        x_start_without_mother = np.zeros(0, int)
        x_end_without_mother = np.zeros(0, int)
        y_mid_with_child_without_mother = np.zeros(0, int)
        x_start_with_child_without_mother = np.zeros(0, int)
        x_end_with_child_without_mother = np.zeros(0, int)

    #print(reading_order_type,'reading_order_type')
    #print(y_mid_with_child_without_mother,'y_mid_with_child_without_mother')
    #print(x_start_with_child_without_mother,'x_start_with_child_without_mother')
    #print(x_end_with_child_without_mother,'x_end_with_hild_without_mother')

    len_sep_with_child = len(child[child==1])
    #print(len_sep_with_child,'len_sep_with_child')
    there_is_sep_with_child = 0
    if len_sep_with_child >= 1:
        there_is_sep_with_child = 1

    return (reading_order_type,
            x_start_returned,
            x_end_returned,
            y_mid_returned,
            y_max_returned,
            y_mid_without_mother,
            x_start_without_mother,
            x_end_without_mother,
            there_is_sep_with_child,
            y_mid_with_child_without_mother,
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

def find_num_col(regions_without_separators, num_col_classifier, tables, multiplier=3.8):
    if not regions_without_separators.any():
        return 0, []
    regions_without_separators_0 = regions_without_separators.sum(axis=0)
    # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # ax1.imshow(regions_without_separators, aspect="auto")
    # ax2.plot(regions_without_separators_0)
    # plt.show()
    sigma_ = 35  # 70#35
    meda_n_updown = regions_without_separators_0[::-1]
    first_nonzero = next((i for i, x in enumerate(regions_without_separators_0) if x), 0)
    last_nonzero = next((i for i, x in enumerate(meda_n_updown) if x), 0)
    last_nonzero = len(regions_without_separators_0) - last_nonzero
    last_nonzero = last_nonzero - 100
    first_nonzero = first_nonzero + 200
    y = regions_without_separators_0  # [first_nonzero:last_nonzero]
    y_help = np.zeros(len(y) + 20)
    y_help[10 : len(y) + 10] = y
    x = np.arange(len(y))
    zneg_rev = -y_help + np.max(y_help)
    zneg = np.zeros(len(zneg_rev) + 20)
    zneg[10 : len(zneg_rev) + 10] = zneg_rev
    z = gaussian_filter1d(y, sigma_)
    zneg = gaussian_filter1d(zneg, sigma_)

    peaks, _ = find_peaks(z, height=0)
    peaks_neg, _ = find_peaks(zneg, height=0)
    # _, (ax1, ax2) = plt.subplots(2, sharex=True)
    # ax1.set_title("z")
    # ax1.plot(z)
    # ax1.scatter(peaks, z[peaks])
    # ax1.axvline(0.06 * len(y), label="first")
    # ax1.axvline(0.94 * len(y), label="last")
    # ax1.text(0.06 * len(y), 0, "first", rotation=90)
    # ax1.text(0.94 * len(y), 0, "last", rotation=90)
    # ax1.axhline(10, label="minimum")
    # ax1.text(0, 10, "minimum")
    # ax2.set_title("zneg")
    # ax2.plot(zneg)
    # ax2.scatter(peaks_neg, zneg[peaks_neg])
    # ax2.axvline(first_nonzero, label="first nonzero")
    # ax2.axvline(last_nonzero, label="last nonzero")
    # ax2.text(first_nonzero, 0, "first nonzero", rotation=90)
    # ax2.text(last_nonzero, 0, "last nonzero", rotation=90)
    # ax2.axvline(370, label="first")
    # ax2.axvline(len(y) - 370, label="last")
    # ax2.text(370, 0, "first", rotation=90)
    # ax2.text(len(y) - 370, 0, "last", rotation=90)
    # plt.show()
    peaks_neg = peaks_neg - 10 - 10

    peaks = peaks[(peaks > 0.06 * len(y)) &
                  (peaks < 0.94 * len(y))]
    interest_pos = z[peaks]
    interest_pos = interest_pos[interest_pos > 10]
    if not interest_pos.any():
        return 0, []
    # plt.plot(z)
    # plt.show()
    peaks_neg = peaks_neg[(peaks_neg > first_nonzero) &
                          (peaks_neg < last_nonzero)]
    peaks_neg = peaks_neg[(peaks_neg > 370) &
                          (peaks_neg < len(y) - 370)]
    interest_neg = z[peaks_neg]
    if not interest_neg.any():
        return 0, []

    min_peaks_pos = np.min(interest_pos)
    max_peaks_pos = np.max(interest_pos)

    #print(min_peaks_pos, max_peaks_pos, max_peaks_pos / min_peaks_pos, 'minmax')
    if max_peaks_pos / (min_peaks_pos or 1e-9) >= 35:
        min_peaks_pos = np.mean(interest_pos)

    min_peaks_neg = 0  # np.min(interest_neg)

    dis_talaei = (min_peaks_pos - min_peaks_neg) / multiplier
    grenze = min_peaks_pos - dis_talaei
    #np.mean(y[peaks_neg[0]:peaks_neg[-1]])-np.std(y[peaks_neg[0]:peaks_neg[-1]])/2.0

    # print(interest_neg,'interest_neg')
    # print(grenze,'grenze')
    # print(min_peaks_pos,'min_peaks_pos')
    # print(dis_talaei,'dis_talaei')
    # print(peaks_neg,'peaks_neg')
    # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # ax1.imshow(regions_without_separators, aspect="auto")
    # ax2.plot(z, color='red', label='z')
    # ax2.plot(zneg[20:], color='blue', label='zneg')
    # ax2.scatter(peaks_neg, z[peaks_neg], color='red')
    # ax2.scatter(peaks_neg, zneg[20:][peaks_neg], color='blue')
    # ax2.axhline(min_peaks_pos, color='red', label="min_peaks_pos")
    # ax2.axhline(grenze, color='blue', label="grenze")
    # ax2.text(0, grenze, "grenze")
    # plt.show()

    interest_neg_fin = interest_neg[(interest_neg < grenze)]
    peaks_neg_fin = peaks_neg[(interest_neg < grenze)]

    if not tables:
        if ( num_col_classifier - ( (len(interest_neg_fin))+1 ) ) >= 3:
            # found too few columns here: ignore 'grenze' and take the deepest N peaks
            sort_by_height = np.argsort(interest_neg)[:num_col_classifier]
            peaks_neg_fin = peaks_neg[sort_by_height]
            interest_neg_fin = interest_neg[sort_by_height]
            # print(peaks_neg_fin, "peaks_neg[sorted_by_height]")
            sort_by_pos = np.argsort(peaks_neg_fin)
            peaks_neg_fin = peaks_neg_fin[sort_by_pos]
            interest_neg_fin = interest_neg_fin[sort_by_pos]

    num_col = len(interest_neg_fin) + 1

    # print(peaks_neg_fin,'peaks_neg_fin')
    # print(num_col,'diz')
    # cancel if resulting split is highly unbalanced across available width
    if ((num_col == 3 and
        ((peaks_neg_fin[0] > 0.75 * len(y) and
          peaks_neg_fin[1] > 0.75 * len(y)) or
         (peaks_neg_fin[0] < 0.25 * len(y) and
          peaks_neg_fin[1] < 0.25 * len(y)) or
         (peaks_neg_fin[0] < 0.5 * len(y) - 200 and
          peaks_neg_fin[1] < 0.5 * len(y)) or
         (peaks_neg_fin[0] > 0.5 * len(y) + 200 and
          peaks_neg_fin[1] > 0.5 * len(y)))) or
        (num_col == 2 and
         (peaks_neg_fin[0] > 0.75 * len(y) or
          peaks_neg_fin[0] < 0.25 * len(y)))):
        num_col = 1
        peaks_neg_fin = []

    ##print(len(peaks_neg_fin))

    # filter out peaks that are too close (<400px) to each other:
    # among each group, pick the position with smallest amount of text
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
            else:
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
    #print(peaks_neg_true, "peaks_neg_true")
    ##print(num_col,'early')
    # cancel if resulting split is highly unbalanced across available width
    if ((num_col == 3 and
        ((peaks_neg_true[0] > 0.75 * len(y) and
          peaks_neg_true[1] > 0.75 * len(y)) or
         (peaks_neg_true[0] < 0.25 * len(y) and
          peaks_neg_true[1] < 0.25 * len(y)) or
         (peaks_neg_true[0] < 0.5 * len(y) - 200 and
          peaks_neg_true[1] < 0.5 * len(y)) or
         (peaks_neg_true[0] > 0.5 * len(y) + 200 and
          peaks_neg_true[1] > 0.5 * len(y)))) or
        (num_col == 2 and
         (peaks_neg_true[0] > 0.75 * len(y) or
          peaks_neg_true[0] < 0.25 * len(y)))):
        num_col = 1
        peaks_neg_true = []
    if (num_col == 3 and
        (peaks_neg_true[0] < 0.75 * len(y) and
         peaks_neg_true[0] > 0.25 * len(y) and
         peaks_neg_true[1] > 0.80 * len(y))):
        num_col = 2
        peaks_neg_true = [peaks_neg_true[0]]
    if (num_col == 3 and
        (peaks_neg_true[1] < 0.75 * len(y) and
         peaks_neg_true[1] > 0.25 * len(y) and
         peaks_neg_true[0] < 0.20 * len(y))):
        num_col = 2
        peaks_neg_true = [peaks_neg_true[1]]

    # get rid of too narrow columns (not used)
    # if np.count_nonzero(diff_peaks < 360):
    #     arg_help = np.arange(len(diff_peaks))
    #     arg_help_ann = arg_help[diff_peaks < 360]
    #     peaks_neg_fin_new = []
    #     for ii in range(len(peaks_neg_fin)):
    #         if ii in arg_help_ann:
    #             if interest_neg_fin[ii] < interest_neg_fin[ii + 1]:
    #                 peaks_neg_fin_new.append(peaks_neg_fin[ii])
    #             else:
    #                 peaks_neg_fin_new.append(peaks_neg_fin[ii + 1])

    #         elif (ii - 1) not in arg_help_ann:
    #             peaks_neg_fin_new.append(peaks_neg_fin[ii])
    # else:
    #     peaks_neg_fin_new = peaks_neg_fin

    # plt.plot(gaussian_filter1d(y, sigma_))
    # plt.plot(peaks_neg_true,z[peaks_neg_true],'*')
    # plt.plot([0,len(y)], [grenze,grenze])
    # plt.show()
    ##print(len(peaks_neg_true))
    #print(peaks_neg_true, "peaks_neg_true")
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
    # fig, (ax1, ax2) = plt.subplots(2, sharex=True, suptitle='find_num_col_by_vertical_lines')
    # ax1.imshow(regions_without_separators, aspect="auto")
    # ax2.plot(z)
    # ax2.scatter(peaks, z[peaks])
    # ax2.set_title('find_peaks(regions_without_separators.sum(axis=0), height=0)')
    # plt.show()
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

def order_of_regions(textline_mask, contours_main, contours_head, y_ref, x_ref):
    ##plt.imshow(textline_mask)
    ##plt.show()
    y = textline_mask.sum(axis=1) # horizontal projection profile
    y_padded = np.zeros(len(y) + 40)
    y_padded[20 : len(y) + 20] = y

    sigma_gaus = 8
    #z = gaussian_filter1d(y_padded, sigma_gaus)
    #peaks, _ = find_peaks(z, height=0)
    #peaks = peaks - 20
    ##plt.plot(z)
    ##plt.show()
    zneg_rev = np.max(y_padded) - y_padded
    zneg = np.zeros(len(zneg_rev) + 40)
    zneg[20 : len(zneg_rev) + 20] = zneg_rev
    zneg = gaussian_filter1d(zneg, sigma_gaus)

    peaks_neg, _ = find_peaks(zneg, height=0)
    peaks_neg = peaks_neg - 20 - 20

    peaks_neg_new = np.array([0] +
                             # peaks can be beyond box due to padding and smoothing
                             [peak for peak in peaks_neg
                              if 0 < peak and peak < textline_mask.shape[0]] +
                             [textline_mask.shape[0]])
    # offset from bbox of mask
    peaks_neg_new += y_ref

    cx_main, cy_main = find_center_of_contours(contours_main)
    cx_head, cy_head = find_center_of_contours(contours_head)
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
        # if indexes_in.size:
        #     img = textline_mask.copy()
        #     plt.imshow(img)
        #     plt.gca().add_patch(patches.Rectangle((0, top-y_ref), img.shape[1], bot-top, alpha=0.5, color='gray'))
        #     xrange = np.arange(0, img.shape[1], 50)
        #     yrange = np.arange(0, img.shape[0], 50)
        #     plt.gca().set_xticks(xrange, xrange + x_ref)
        #     plt.gca().set_yticks(yrange, yrange + y_ref)
        #     for idx, type_, cx, cy in zip(typed_indexes_in, types_in, cxs_in, cys_in):
        #         cnt = (contours_main if type_ == 1 else contours_head)[idx]
        #         col = 'red' if type_ == 1 else 'blue'
        #         plt.scatter(cx - x_ref, cy - y_ref, 20, c=col, marker='o')
        #         plt.gca().add_patch(patches.Polygon(cnt[:, 0] - [[x_ref, y_ref]], closed=False, fill=False, color=col))
        #     plt.title("box contours centered in %d:%d (red=main / blue=heading)" % (top, bot))
        #     plt.show()

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
        sep_pairs=contours_in_same_horizon(cy_main_hor)
        if len(sep_pairs):
            special_separators=[]
            contours_new=[]
            for pair in sep_pairs:
                merged_all=None
                some_args=args_hor[pair]
                some_cy=cy_main_hor[pair]
                some_x_min=x_min_main_hor[pair]
                some_x_max=x_max_main_hor[pair]

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

def find_number_of_columns_in_document(region_pre_p, num_col_classifier, tables, label_seps, contours_h=None):
    separators_closeup = 1 * (region_pre_p == label_seps)
    separators_closeup[0:110] = 0
    separators_closeup[-150:] = 0

    kernel = np.ones((5,5),np.uint8)
    separators_closeup = separators_closeup.astype(np.uint8)
    separators_closeup = cv2.morphologyEx(separators_closeup, cv2.MORPH_CLOSE, kernel, iterations=1)

    separators_closeup_n = separators_closeup.astype(np.uint8) # to be returned

    separators_closeup_n_binary = separators_closeup_n.copy()

    # find horizontal lines by contour properties
    contours_sep_e, _ = cv2.findContours(separators_closeup_n_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts_hor_e = []
    for cnt in contours_sep_e:
        max_xe = cnt[:, 0, 0].max()
        min_xe = cnt[:, 0, 0].min()
        max_ye = cnt[:, 0, 1].max()
        min_ye = cnt[:, 0, 1].min()
        med_ye = int(np.median(cnt[:, 0, 1]))
        dist_xe = max_xe - min_xe
        dist_ye = max_ye - min_ye
        if dist_ye <= 50 and dist_xe >= 3 * dist_ye:
            cnts_hor_e.append(cnt)

    # delete horizontal contours (leaving only the edges)
    separators_closeup_n_binary = cv2.fillPoly(separators_closeup_n_binary, pts=cnts_hor_e, color=0)
    edges = cv2.adaptiveThreshold(separators_closeup_n_binary * 255, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    horizontal = np.copy(edges)
    vertical = np.copy(edges)

    horizontal_size = horizontal.shape[1] // 30
    # find horizontal lines by morphology
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontalStructure)
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, kernel, iterations=2)
    # re-insert deleted horizontal contours
    horizontal = cv2.fillPoly(horizontal, pts=cnts_hor_e, color=255)

    vertical_size = vertical.shape[0] // 30
    # find vertical lines by morphology
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, verticalStructure)
    vertical = cv2.dilate(vertical, kernel, iterations=1)

    horizontal, special_separators = \
        combine_hor_lines_and_delete_cross_points_and_get_lines_features_back_new(
            vertical, horizontal, num_col_classifier)

    _, thresh = cv2.threshold(vertical, 0, 255, 0)
    contours_sep_vers, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slope_seps, dist_x, x_min_seps, x_max_seps, cy_seps, slope_seps_org, y_min_seps, y_max_seps, cx_seps = \
        find_features_of_lines(contours_sep_vers)

    args=np.arange(len(slope_seps))
    args_ver=args[slope_seps==1]
    dist_x_ver=dist_x[slope_seps==1]
    y_min_seps_ver=y_min_seps[slope_seps==1]
    y_max_seps_ver=y_max_seps[slope_seps==1]
    x_min_seps_ver=x_min_seps[slope_seps==1]
    x_max_seps_ver=x_max_seps[slope_seps==1]
    cx_seps_ver=cx_seps[slope_seps==1]
    dist_y_ver=y_max_seps_ver-y_min_seps_ver
    len_y=separators_closeup.shape[0]/3.0

    _, thresh = cv2.threshold(horizontal, 0, 255, 0)
    contours_sep_hors, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slope_seps, dist_x, x_min_seps, x_max_seps, cy_seps, slope_seps_org, y_min_seps, y_max_seps, cx_seps = \
        find_features_of_lines(contours_sep_hors)

    slope_seps_org_hor=slope_seps_org[slope_seps==0]
    args=np.arange(len(slope_seps))
    len_x=separators_closeup.shape[1]/5.0
    dist_y=np.abs(y_max_seps-y_min_seps)

    args_hor=args[slope_seps==0]
    dist_x_hor=dist_x[slope_seps==0]
    y_min_seps_hor=y_min_seps[slope_seps==0]
    y_max_seps_hor=y_max_seps[slope_seps==0]
    x_min_seps_hor=x_min_seps[slope_seps==0]
    x_max_seps_hor=x_max_seps[slope_seps==0]
    dist_y_hor=dist_y[slope_seps==0]
    cy_seps_hor=cy_seps[slope_seps==0]

    args_hor=args_hor[dist_x_hor>=len_x/2.0]
    x_max_seps_hor=x_max_seps_hor[dist_x_hor>=len_x/2.0]
    x_min_seps_hor=x_min_seps_hor[dist_x_hor>=len_x/2.0]
    cy_seps_hor=cy_seps_hor[dist_x_hor>=len_x/2.0]
    y_min_seps_hor=y_min_seps_hor[dist_x_hor>=len_x/2.0]
    y_max_seps_hor=y_max_seps_hor[dist_x_hor>=len_x/2.0]
    dist_y_hor=dist_y_hor[dist_x_hor>=len_x/2.0]
    slope_seps_org_hor=slope_seps_org_hor[dist_x_hor>=len_x/2.0]
    dist_x_hor=dist_x_hor[dist_x_hor>=len_x/2.0]

    matrix_of_seps_ch = np.zeros((len(cy_seps_hor)+len(cx_seps_ver), 10), dtype=int)
    matrix_of_seps_ch[:len(cy_seps_hor),0]=args_hor
    matrix_of_seps_ch[len(cy_seps_hor):,0]=args_ver
    matrix_of_seps_ch[len(cy_seps_hor):,1]=cx_seps_ver
    matrix_of_seps_ch[:len(cy_seps_hor),2]=x_min_seps_hor+50#x_min_seps_hor+150
    matrix_of_seps_ch[len(cy_seps_hor):,2]=x_min_seps_ver
    matrix_of_seps_ch[:len(cy_seps_hor),3]=x_max_seps_hor-50#x_max_seps_hor-150
    matrix_of_seps_ch[len(cy_seps_hor):,3]=x_max_seps_ver
    matrix_of_seps_ch[:len(cy_seps_hor),4]=dist_x_hor
    matrix_of_seps_ch[len(cy_seps_hor):,4]=dist_x_ver
    matrix_of_seps_ch[:len(cy_seps_hor),5]=cy_seps_hor
    matrix_of_seps_ch[:len(cy_seps_hor),6]=y_min_seps_hor
    matrix_of_seps_ch[len(cy_seps_hor):,6]=y_min_seps_ver
    matrix_of_seps_ch[:len(cy_seps_hor),7]=y_max_seps_hor
    matrix_of_seps_ch[len(cy_seps_hor):,7]=y_max_seps_ver
    matrix_of_seps_ch[:len(cy_seps_hor),8]=dist_y_hor
    matrix_of_seps_ch[len(cy_seps_hor):,8]=dist_y_ver
    matrix_of_seps_ch[len(cy_seps_hor):,9]=1

    if contours_h is not None:
        _, dist_x_head, x_min_head, x_max_head, cy_head, _, y_min_head, y_max_head, _ = \
            find_features_of_lines(contours_h)
        matrix_l_n = np.zeros((len(cy_head), matrix_of_seps_ch.shape[1]), dtype=int)
        args_head = np.arange(len(cy_head))
        matrix_l_n[:, 0] = args_head
        matrix_l_n[:, 2] = x_min_head
        matrix_l_n[:, 3] = x_max_head
        matrix_l_n[:, 4] = dist_x_head
        matrix_l_n[:, 5] = cy_head
        matrix_l_n[:, 6] = y_min_head
        matrix_l_n[:, 7] = y_max_head
        matrix_l_n[:, 8] = y_max_head - y_min_head
        matrix_l_n[:, 9] = 2 # mark as heading (so it can be split into 2 horizontal separators as needed)
        matrix_of_seps_ch = np.append(
            matrix_of_seps_ch, matrix_l_n, axis=0)

    cy_seps_splitters=cy_seps_hor[(x_min_seps_hor<=.16*region_pre_p.shape[1]) &
                                  (x_max_seps_hor>=.84*region_pre_p.shape[1])]
    cy_seps_splitters = np.append(cy_seps_splitters, special_separators)

    if contours_h is not None:
        y_min_splitters_head = y_min_head[(x_min_head<=.16*region_pre_p.shape[1]) &
                                          (x_max_head>=.84*region_pre_p.shape[1])]
        y_max_splitters_head = y_max_head[(x_min_head<=.16*region_pre_p.shape[1]) &
                                          (x_max_head>=.84*region_pre_p.shape[1])]
        cy_seps_splitters = np.append(cy_seps_splitters, y_min_splitters_head)
        cy_seps_splitters = np.append(cy_seps_splitters, y_max_splitters_head)

    cy_seps_splitters = np.sort(cy_seps_splitters).astype(int)
    splitter_y_new = [0] + list(cy_seps_splitters) + [region_pre_p.shape[0]]
    big_part = 22 * region_pre_p.shape[0] // 100 # percent height

    regions_without_separators=return_regions_without_separators(region_pre_p)

    num_col_fin=0
    peaks_neg_fin_fin=[]
    num_big_parts = 0
    for top, bot in pairwise(splitter_y_new):
        if bot - top < big_part:
            continue
        num_big_parts += 1
        try:
            num_col, peaks_neg_fin = find_num_col(regions_without_separators[top: bot],
                                                  num_col_classifier, tables, multiplier=7.0)
            #print("big part %d:%d has %d columns" % (top, bot, num_col), peaks_neg_fin)
        except:
            num_col = 0
            peaks_neg_fin = []
        if num_col>num_col_fin:
            num_col_fin=num_col
            peaks_neg_fin_fin=peaks_neg_fin

    if num_big_parts == 1 and len(peaks_neg_fin_fin) + 1 < num_col_classifier:
        peaks_neg_fin=find_num_col_by_vertical_lines(vertical)
        peaks_neg_fin=peaks_neg_fin[peaks_neg_fin>=500]
        peaks_neg_fin=peaks_neg_fin[peaks_neg_fin<=(vertical.shape[1]-500)]
        peaks_neg_fin_fin=peaks_neg_fin[:]

    return num_col_fin, peaks_neg_fin_fin, matrix_of_seps_ch, splitter_y_new, separators_closeup_n

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

    # def dbg_plt(box=None, title=None, rectangles=None, rectangles_showidx=False):
    #     minx, maxx, miny, maxy = box or (0, None, 0, None)
    #     img = regions_without_separators[miny:maxy, minx:maxx]
    #     plt.imshow(img)
    #     xrange = np.arange(0, img.shape[1], 100)
    #     yrange = np.arange(0, img.shape[0], 100)
    #     ax = plt.gca()
    #     ax.set_xticks(xrange)
    #     ax.set_yticks(yrange)
    #     ax.set_xticklabels(xrange + minx)
    #     ax.set_yticklabels(yrange + miny)
    #     def format_coord(x, y):
    #         return 'x={:g}, y={:g}'.format(x + minx, y + miny)
    #     ax.format_coord = format_coord
    #     if title:
    #         plt.title(title)
    #     if rectangles:
    #         for i, (xmin, xmax, ymin, ymax) in enumerate(rectangles):
    #             ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                            fill=False, linewidth=1, edgecolor='r'))
    #             if rectangles_showidx:
    #                 ax.text((xmin+xmax)/2, (ymin+ymax)/2, str(i + 1), c='r')
    #     plt.show()
    # dbg_plt(title="return_boxes_of_images_by_order_of_reading_new")

    boxes=[]
    peaks_neg_tot_tables = []
    splitter_y_new = np.array(splitter_y_new, dtype=int)
    height_tot, width_tot = regions_without_separators.shape
    big_part = 22 * height_tot // 100 # percent height
    for top, bot in pairwise(splitter_y_new):
        # print("%d:%d" % (top, bot), 'i')
        # dbg_plt([0, None, top, bot], "image cut for y split %d:%d" % (top, bot))
        matrix_new = matrix_of_lines_ch[(matrix_of_lines_ch[:,6] > top) &
                                        (matrix_of_lines_ch[:,7] < bot)]
        #print(len( matrix_new[:,9][matrix_new[:,9]==1] ))
        #print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')
        # check to see is there any vertical separator to find holes.
        #if (len(matrix_new[:,9][matrix_new[:,9]==1]) > 0 and
        #    np.max(matrix_new[:,8][matrix_new[:,9]==1]) >=
        #    0.1 * (np.abs(bot-top))):
        try:
            num_col, peaks_neg_fin = find_num_col(
                regions_without_separators[top:bot],
                # we do not expect to get all columns in small parts (headings etc.):
                num_col_classifier if bot - top >= big_part else 1,
                tables, multiplier=6. if erosion_hurts else 7.)
        except:
            peaks_neg_fin=[]
            num_col = 0
        try:
            if ((len(peaks_neg_fin) + 1 < num_col_classifier or
                num_col_classifier == 6) and
                # we do not expect to get all columns in small parts (headings etc.):
                bot - top >= big_part):
                # found too few columns here
                #print('burda')
                peaks_neg_fin_org = np.copy(peaks_neg_fin)
                #print("peaks_neg_fin_org", peaks_neg_fin_org)
                if len(peaks_neg_fin)==0:
                    num_col, peaks_neg_fin = find_num_col(
                        regions_without_separators[top:bot],
                        num_col_classifier, tables, multiplier=3.)
                #print(peaks_neg_fin,'peaks_neg_fin')
                peaks_neg_fin_early = [0] + peaks_neg_fin + [width_tot-1]

                #print(peaks_neg_fin_early,'burda2')
                peaks_neg_fin_rev=[]
                for left, right in pairwise(peaks_neg_fin_early):
                    # print("%d:%d" % (left, right), 'i_n')
                    # dbg_plt([left, right, top, bot],
                    #         "image cut for y split %d:%d / x gap %d:%d" % (
                    #             top, bot, left, right))
                    # plt.plot(regions_without_separators[top:bot, left:right].sum(axis=0))
                    # plt.title("vertical projection (sum over y)")
                    # plt.show()
                    try:
                        _, peaks_neg_fin1 = find_num_col(
                            regions_without_separators[top:bot, left:right],
                            num_col_classifier, tables, multiplier=7.)
                    except:
                        peaks_neg_fin1 = []
                    try:
                        _, peaks_neg_fin2 = find_num_col(
                            regions_without_separators[top:bot, left:right],
                            num_col_classifier, tables, multiplier=5.)
                    except:
                        peaks_neg_fin2 = []
                    if len(peaks_neg_fin1) >= len(peaks_neg_fin2):
                        peaks_neg_fin = peaks_neg_fin1
                    else:
                        peaks_neg_fin = peaks_neg_fin2
                    # add offset to local result
                    peaks_neg_fin = list(np.array(peaks_neg_fin) + left)
                    #print(peaks_neg_fin,'peaks_neg_fin')

                    peaks_neg_fin_rev.extend(peaks_neg_fin)
                    if right < peaks_neg_fin_early[-1]:
                        # all but the last column: interject the preexisting boundary
                        peaks_neg_fin_rev.append(right)
                    #print(peaks_neg_fin_rev,'peaks_neg_fin_rev')

                if len(peaks_neg_fin_rev) >= len(peaks_neg_fin_org):
                    peaks_neg_fin = peaks_neg_fin_rev
                else:
                    peaks_neg_fin = peaks_neg_fin_org
                num_col = len(peaks_neg_fin)
                #print(peaks_neg_fin,'peaks_neg_fin')
        except:
            logger.exception("cannot find peaks consistent with columns")
        #num_col, peaks_neg_fin = find_num_col(
        #    regions_without_separators[top:bot,:],
        #    multiplier=7.0)
        peaks_neg_tot = np.array([0] + peaks_neg_fin + [width_tot])
        #print(peaks_neg_tot,'peaks_neg_tot')
        peaks_neg_tot_tables.append(peaks_neg_tot)

        all_columns = set(range(len(peaks_neg_tot) - 1))
        #print("all_columns", all_columns)

        # elongate horizontal separators+headings as much as possible without overlap
        args_nonver = matrix_new[:, 9] != 1
        regions_with_separators = np.copy(regions_without_separators[top:bot])
        for xmin, xmax, ymin, ymax in matrix_new[:, [2, 3, 6, 7]]:
            regions_with_separators[ymin - top: ymax - top, xmin: xmax] = 6
        # def dbg_imshow(box, title):
        #     xmin, xmax, ymin, ymax = box
        #     plt.imshow(regions_with_separators, extent=[0, width_tot, bot, top])
        #     plt.gca().add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                                           fill=False, linewidth=1, edgecolor='r'))
        #     plt.title(title + " at %d:%d, %d:%d" % (ymin, ymax, xmin, xmax))
        #     plt.show()
        for i in np.flatnonzero(args_nonver):
            xmin, xmax, ymin, ymax, typ = matrix_new[i, [2, 3, 6, 7, 9]]
            cut = regions_with_separators[ymin - top: ymax - top]
            # dbg_imshow([xmin, xmax, ymin, ymax], "separator %d (%s)" % (i, "heading" if typ else "horizontal"))
            starting = xmin - peaks_neg_tot
            min_start = np.flatnonzero(starting >= 0)[-1] # last left-of
            ending = xmax - peaks_neg_tot
            max_end = np.flatnonzero(ending < 0)[0] # first right-of
            # skip elongation unless this is already a multi-column separator/heading:
            if not max_end - min_start > 1:
                continue
            # is there anything left of min_start?
            for j in range(min_start):
                # dbg_imshow([peaks_neg_tot[j], xmin, ymin, ymax], "start of %d candidate %d" % (i, j))
                if not np.any(cut[:, peaks_neg_tot[j]: xmin]):
                    # print("elongated sep", i, "typ", typ, "start", xmin, "to", j, peaks_neg_tot[j])
                    matrix_new[i, 2] = peaks_neg_tot[j] + 1 # elongate to start of this column
                    break
            # is there anything right of max_end?
            for j in range(len(peaks_neg_tot) - 1, max_end, -1):
                # dbg_imshow([xmax, peaks_neg_tot[j], ymin, ymax], "end of %d candidate %d" % (i, j))
                if not np.any(cut[:, xmax: peaks_neg_tot[j]]):
                    # print("elongated sep", i, "typ", typ, "end", xmax, "to", j, peaks_neg_tot[j])
                    matrix_new[i, 3] = peaks_neg_tot[j] - 1 # elongate to end of this column
                    break

        args_hor = matrix_new[:, 9] == 0
        x_min_hor_some = matrix_new[:, 2][args_hor]
        x_max_hor_some = matrix_new[:, 3][args_hor]
        y_max_hor_some = matrix_new[:, 7][args_hor]
        cy_hor_some = matrix_new[:, 5][args_hor]

        args_head = matrix_new[:, 9] == 2
        x_min_hor_head = matrix_new[:, 2][args_head]
        x_max_hor_head = matrix_new[:, 3][args_head]
        y_min_hor_head = matrix_new[:, 6][args_head]
        y_max_hor_head = matrix_new[:, 7][args_head]
        cy_hor_head = matrix_new[:, 5][args_head]

        # split headings at toplines (y_min_head) and baselines (y_max_head)
        # instead of merely adding their center (cy_head) as horizontal separator
        # (x +/- 30px to avoid crossing col peaks by accident)
        x_min_hor_some = np.append(x_min_hor_some, np.tile(x_min_hor_head + 30, 2))
        x_max_hor_some = np.append(x_max_hor_some, np.tile(x_max_hor_head - 30, 2))
        y_max_hor_some = np.append(y_max_hor_some, # baselines
                                   np.concatenate((y_min_hor_head + 2,
                                                   y_max_hor_head + 2)))
        cy_hor_some = np.append(cy_hor_some, # toplines
                                np.concatenate((y_min_hor_head - 2,
                                                y_max_hor_head - 2)))

        if right2left_readingorder:
            x_max_hor_some = width_tot - x_min_hor_some
            x_min_hor_some = width_tot - x_max_hor_some


        reading_order_type, x_starting, x_ending, y_mid, y_max, \
            y_mid_without_mother, x_start_without_mother, x_end_without_mother, \
            there_is_sep_with_child, \
            y_mid_with_child_without_mother, x_start_with_child_without_mother, x_end_with_child_without_mother, \
            new_main_sep_y = return_x_start_end_mothers_childs_and_type_of_reading_order(
                peaks_neg_tot, x_min_hor_some, x_max_hor_some, cy_hor_some, y_max_hor_some)

        # show multi-column separators
        # dbg_plt([0, None, top, bot], "multi-column separators in current split", 
        #         list(zip(peaks_neg_tot[x_starting], peaks_neg_tot[x_ending],
        #                  y_mid - top, y_max - top)), True)

        if (reading_order_type == 1 or
            len(y_mid_without_mother) >= 2 or
            there_is_sep_with_child == 1):
            # there are top-level multi-colspan horizontal separators which overlap each other
            # or multiple top-level multi-colspan horizontal separators
            # or multi-colspan horizontal separators shorter than their respective top-level:
            # todo: explain how this is dealt with
            try:
                y_grenze = top + 300
                up = (y_mid > top) & (y_mid <= y_grenze)

                args_early_ys=np.arange(len(y_mid))
                #print(args_early_ys,'args_early_ys')
                #print(y_mid,'y_mid')

                x_starting_up = x_starting[up]
                x_ending_up = x_ending[up]
                y_mid_up = y_mid[up]
                y_max_up = y_max[up]
                args_up = args_early_ys[up]
                #print(args_up,'args_up')
                #print(y_mid_up,'y_mid_up')
                #check if there is a big separator in this y_mains0
                if len(y_mid_up) > 0:
                    # is there a separator with full-width span?
                    main_separator = (x_starting_up == 0) & (x_ending_up == len(peaks_neg_tot) - 1)
                    y_mid_main_separator_up = y_mid_up[main_separator]
                    y_max_main_separator_up = y_max_up[main_separator]
                    args_main_to_deleted = args_up[main_separator]
                    #print(y_mid_main_separator_up,y_max_main_separator_up,args_main_to_deleted,'fffffjammmm')
                    if len(y_max_main_separator_up):
                        args_to_be_kept = np.array(list( set(args_early_ys) - set(args_main_to_deleted) ))
                        #print(args_to_be_kept,'args_to_be_kept')
                        boxes.append([0, peaks_neg_tot[-1],
                                      top, y_max_main_separator_up.max()])
                        # dbg_plt(boxes[-1], "near top main separator box")
                        top = y_max_main_separator_up.max()

                        #print(top,'top')
                        y_mid = y_mid[args_to_be_kept]
                        x_starting = x_starting[args_to_be_kept]
                        x_ending = x_ending[args_to_be_kept]
                        y_max = y_max[args_to_be_kept]

                        #print('galdiha')
                        y_grenze = top + 200
                        up = (y_mid > top) & (y_mid <= y_grenze)
                        args_early_ys2 = np.arange(len(y_mid))
                        x_starting_up = x_starting[up]
                        x_ending_up = x_ending[up]
                        y_mid_up = y_mid[up]
                        y_max_up = y_max[up]
                        args_up2 = args_early_ys2[up]
                        #print(y_mid_up,x_starting_up,x_ending_up,'didid')
                    else:
                        args_early_ys2 = args_early_ys
                        args_up2 = args_up

                    nodes_in = set()
                    for ij in range(len(x_starting_up)):
                        nodes_in.update(range(x_starting_up[ij],
                                              x_ending_up[ij]))
                    #print(nodes_in,'nodes_in')
                    #print(np.array(range(len(peaks_neg_tot)-1)),'np.array(range(len(peaks_neg_tot)-1))')

                    if nodes_in == set(range(len(peaks_neg_tot)-1)):
                        pass
                    elif nodes_in == set(range(1, len(peaks_neg_tot)-1)):
                        pass
                    else:
                        #print('burdaydikh')
                        args_to_be_kept2 = np.array(list( set(args_early_ys2) - set(args_up2) ))

                        if len(args_to_be_kept2):
                            #print(args_to_be_kept2, "args_to_be_kept2")
                            y_mid = y_mid[args_to_be_kept2]
                            x_starting = x_starting[args_to_be_kept2]
                            x_ending = x_ending[args_to_be_kept2]
                            y_max = y_max[args_to_be_kept2]

                #int(top)
                # order multi-column separators
                y_mid_by_order=[]
                x_start_by_order=[]
                x_end_by_order=[]
                if (reading_order_type == 1 or
                    len(x_end_with_child_without_mother) == 0):
                    if reading_order_type == 1:
                        # there are top-level multi-colspan horizontal separators which overlap each other
                        #print("adding all columns at top because of multiple overlapping mothers")
                        y_mid_by_order.append(top)
                        x_start_by_order.append(0)
                        x_end_by_order.append(len(peaks_neg_tot)-2)
                    else:
                        # there are no top-level multi-colspan horizontal separators which themselves
                        # contain shorter multi-colspan separators
                        #print(x_start_without_mother,x_end_without_mother,peaks_neg_tot,'dodo')
                        columns_covered_by_mothers = set()
                        for dj in range(len(x_start_without_mother)):
                            columns_covered_by_mothers.update(
                                range(x_start_without_mother[dj],
                                      x_end_without_mother[dj]))
                        columns_not_covered = list(all_columns - columns_covered_by_mothers)
                        #print(columns_covered_by_mothers, "columns_covered_by_mothers")
                        #print(columns_not_covered, "columns_not_covered")
                        y_mid = np.append(y_mid, np.ones(len(columns_not_covered) +
                                                         len(x_start_without_mother),
                                                         dtype=int) * top)
                        ##y_mid_by_order = np.append(y_mid_by_order, [top] * len(columns_not_covered))
                        ##x_start_by_order = np.append(x_start_by_order, [0] * len(columns_not_covered))
                        x_starting = np.append(x_starting, np.array(columns_not_covered, int))
                        x_starting = np.append(x_starting, x_start_without_mother)
                        x_ending = np.append(x_ending, np.array(columns_not_covered, int) + 1)
                        x_ending = np.append(x_ending, x_end_without_mother)

                    ind_args=np.arange(len(y_mid))
                    #print(ind_args,'ind_args')
                    for column in range(len(peaks_neg_tot)-1):
                        #print(column,'column')
                        ind_args_in_col=ind_args[x_starting==column]
                        #print('babali2')
                        #print(ind_args_in_col,'ind_args_in_col')
                        #print(len(y_mid))
                        y_mid_column=y_mid[ind_args_in_col]
                        x_start_column=x_starting[ind_args_in_col]
                        x_end_column=x_ending[ind_args_in_col]
                        #print('babali3')
                        ind_args_col_sorted=np.argsort(y_mid_column)
                        y_mid_by_order.extend(y_mid_column[ind_args_col_sorted])
                        x_start_by_order.extend(x_start_column[ind_args_col_sorted])
                        x_end_by_order.extend(x_end_column[ind_args_col_sorted] - 1)
                else:
                    #print(x_start_without_mother,x_end_without_mother,peaks_neg_tot,'dodo')
                    columns_covered_by_mothers = set()
                    for dj in range(len(x_start_without_mother)):
                        columns_covered_by_mothers.update(
                            range(x_start_without_mother[dj],
                                  x_end_without_mother[dj]))
                    columns_not_covered = list(all_columns - columns_covered_by_mothers)
                    #print(columns_covered_by_mothers, "columns_covered_by_mothers")
                    #print(columns_not_covered, "columns_not_covered")
                    y_mid = np.append(y_mid, np.ones(len(columns_not_covered) +
                                                     len(x_start_without_mother),
                                                     dtype=int) * top)
                    ##y_mid_by_order = np.append(y_mid_by_order, [top] * len(columns_not_covered))
                    ##x_start_by_order = np.append(x_start_by_order, [0] * len(columns_not_covered))
                    x_starting = np.append(x_starting, np.array(columns_not_covered, int))
                    x_starting = np.append(x_starting, x_start_without_mother)
                    x_ending = np.append(x_ending, np.array(columns_not_covered, int) + 1)
                    x_ending = np.append(x_ending, x_end_without_mother)

                    columns_covered_by_mothers_with_child = set()
                    for dj in range(len(x_end_with_child_without_mother)):
                        columns_covered_by_mothers_with_child.update(
                            range(x_start_with_child_without_mother[dj],
                                  x_end_with_child_without_mother[dj]))
                    #print(columns_covered_by_mothers_with_child, "columns_covered_by_mothers_with_child")
                    columns_not_covered_by_mothers_with_child = list(
                        all_columns - columns_covered_by_mothers_with_child)
                    #indexes_to_be_spanned=[]
                    for i_s in range(len(x_end_with_child_without_mother)):
                        columns_not_covered_by_mothers_with_child.append(x_start_with_child_without_mother[i_s])
                    columns_not_covered_by_mothers_with_child = np.sort(columns_not_covered_by_mothers_with_child)
                    #print(columns_not_covered_by_mothers_with_child, "columns_not_covered_by_mothers_with_child")
                    ind_args = np.arange(len(y_mid))
                    for i_s_nc in columns_not_covered_by_mothers_with_child:
                        if i_s_nc in x_start_with_child_without_mother:
                            # use only seps with mother's span ("biggest")
                            #print("i_s_nc", i_s_nc)
                            x_end_biggest_column = \
                                x_end_with_child_without_mother[
                                    x_start_with_child_without_mother == i_s_nc][0]
                            args_all_biggest_seps = \
                                ind_args[(x_starting == i_s_nc) &
                                         (x_ending == x_end_biggest_column)]
                            y_mid_column_nc = y_mid[args_all_biggest_seps]
                            #print("%d:%d" % (i_s_nc, x_end_biggest_column), "columns covered by mother with child")
                            #x_start_column_nc = x_starting[args_all_biggest_seps]
                            #x_end_column_nc = x_ending[args_all_biggest_seps]
                            y_mid_column_nc = np.sort(y_mid_column_nc)
                            #print(y_mid_column_nc, "y_mid_column_nc (sorted)")
                            for nc_top, nc_bot in pairwise(np.append(y_mid_column_nc, bot)):
                                #print("i_c", i_c)
                                #print("%d:%d" % (nc_top, nc_bot), "y_mid_column_nc")
                                ind_all_seps_between_nm_wc = \
                                    ind_args[(y_mid > nc_top) &
                                             (y_mid < nc_bot) &
                                             (x_starting >= i_s_nc) &
                                             (x_ending <= x_end_biggest_column)]
                                y_mid_all_between_nm_wc = y_mid[ind_all_seps_between_nm_wc]
                                x_starting_all_between_nm_wc = x_starting[ind_all_seps_between_nm_wc]
                                x_ending_all_between_nm_wc = x_ending[ind_all_seps_between_nm_wc]

                                columns_covered_by_mothers = set()
                                for dj in range(len(ind_all_seps_between_nm_wc)):
                                    columns_covered_by_mothers.update(
                                        range(x_starting_all_between_nm_wc[dj],
                                              x_ending_all_between_nm_wc[dj]))
                                #print(columns_covered_by_mothers, "columns_covered_by_mothers")
                                child_columns = set(range(i_s_nc, x_end_biggest_column))
                                columns_not_covered = list(child_columns - columns_covered_by_mothers)
                                #print(child_columns, "child_columns")
                                #print(columns_not_covered, "columns_not_covered")

                                if len(ind_all_seps_between_nm_wc):
                                    biggest = np.argmax(x_ending_all_between_nm_wc -
                                                        x_starting_all_between_nm_wc)
                                    #print(ind_all_seps_between_nm_wc, "ind_all_seps_between_nm_wc")
                                    #print(biggest, "%d:%d" % (x_starting_all_between_nm_wc[biggest],
                                                              x_ending_all_between_nm_wc[biggest]), "biggest")
                                    if columns_covered_by_mothers == set(
                                            range(x_starting_all_between_nm_wc[biggest],
                                                  x_ending_all_between_nm_wc[biggest])):
                                        # single biggest accounts for all covered columns alone,
                                        # this separator should be extended to cover all
                                        seps_too_close_to_top_separator = \
                                            ((y_mid_all_between_nm_wc > nc_top) &
                                             (y_mid_all_between_nm_wc <= nc_top + 500))
                                        if (np.count_nonzero(seps_too_close_to_top_separator) and
                                            np.count_nonzero(seps_too_close_to_top_separator) <
                                            len(ind_all_seps_between_nm_wc)):
                                            #print(seps_too_close_to_top_separator, "seps_too_close_to_top_separator")
                                            y_mid_all_between_nm_wc = \
                                                y_mid_all_between_nm_wc[~seps_too_close_to_top_separator]
                                            x_starting_all_between_nm_wc = \
                                                x_starting_all_between_nm_wc[~seps_too_close_to_top_separator]
                                            x_ending_all_between_nm_wc = \
                                                x_ending_all_between_nm_wc[~seps_too_close_to_top_separator]

                                        y_mid_all_between_nm_wc = np.append(
                                            y_mid_all_between_nm_wc, nc_top)
                                        x_starting_all_between_nm_wc = np.append(
                                            x_starting_all_between_nm_wc, i_s_nc)
                                        x_ending_all_between_nm_wc = np.append(
                                            x_ending_all_between_nm_wc, x_end_biggest_column)
                                    else:
                                        y_mid_all_between_nm_wc = np.append(
                                            y_mid_all_between_nm_wc, nc_top)
                                        x_starting_all_between_nm_wc = np.append(
                                            x_starting_all_between_nm_wc, x_starting_all_between_nm_wc[biggest])
                                        x_ending_all_between_nm_wc = np.append(
                                            x_ending_all_between_nm_wc, x_ending_all_between_nm_wc[biggest])

                                if len(columns_not_covered):
                                    y_mid_all_between_nm_wc = np.append(
                                        y_mid_all_between_nm_wc, [nc_top] * len(columns_not_covered))
                                    x_starting_all_between_nm_wc = np.append(
                                        x_starting_all_between_nm_wc, np.array(columns_not_covered, int))
                                    x_ending_all_between_nm_wc = np.append(
                                        x_ending_all_between_nm_wc, np.array(columns_not_covered, int) + 1)

                                ind_args_between=np.arange(len(x_ending_all_between_nm_wc))
                                for column in range(int(i_s_nc), int(x_end_biggest_column)):
                                    ind_args_in_col=ind_args_between[x_starting_all_between_nm_wc==column]
                                    #print('babali2')
                                    #print(ind_args_in_col,'ind_args_in_col')
                                    #print(len(y_mid))
                                    y_mid_column=y_mid_all_between_nm_wc[ind_args_in_col]
                                    x_start_column=x_starting_all_between_nm_wc[ind_args_in_col]
                                    x_end_column=x_ending_all_between_nm_wc[ind_args_in_col]
                                    #print('babali3')
                                    ind_args_col_sorted=np.argsort(y_mid_column)
                                    y_mid_by_order.extend(y_mid_column[ind_args_col_sorted])
                                    x_start_by_order.extend(x_start_column[ind_args_col_sorted])
                                    x_end_by_order.extend(x_end_column[ind_args_col_sorted] - 1)
                        else:
                            #print(i_s_nc,'column not covered by mothers with child')
                            ind_args_in_col=ind_args[x_starting==i_s_nc]
                            #print('babali2')
                            #print(ind_args_in_col,'ind_args_in_col')
                            #print(len(y_mid))
                            y_mid_column=y_mid[ind_args_in_col]
                            x_start_column=x_starting[ind_args_in_col]
                            x_end_column=x_ending[ind_args_in_col]
                            #print('babali3')
                            ind_args_col_sorted = np.argsort(y_mid_column)
                            y_mid_by_order.extend(y_mid_column[ind_args_col_sorted])
                            x_start_by_order.extend(x_start_column[ind_args_col_sorted])
                            x_end_by_order.extend(x_end_column[ind_args_col_sorted] - 1)

                # create single-column boxes from multi-column separators
                y_mid_by_order = np.array(y_mid_by_order)
                x_start_by_order = np.array(x_start_by_order)
                x_end_by_order = np.array(x_end_by_order)
                for il in range(len(y_mid_by_order)):
                    #print(il, "il")
                    y_mid_itself = y_mid_by_order[il]
                    x_start_itself = x_start_by_order[il]
                    x_end_itself = x_end_by_order[il]
                    for column in range(int(x_start_itself), int(x_end_itself)+1):
                        #print(column,'cols')
                        #print('burda')
                        #print('burda2')
                        y_mid_next = y_mid_by_order[(y_mid_itself < y_mid_by_order) &
                                                    (column >= x_start_by_order) &
                                                    (column <= x_end_by_order)]
                        y_mid_next = y_mid_next.min(initial=bot)
                        #print(y_mid_next,'y_mid_next')
                        #print(y_mid_itself,'y_mid_itself')
                        boxes.append([peaks_neg_tot[column],
                                      peaks_neg_tot[column+1],
                                      y_mid_itself,
                                      y_mid_next])
                        # dbg_plt(boxes[-1], "A column %d box" % (column + 1))
            except:
                logger.exception("cannot assign boxes")
                boxes.append([0, peaks_neg_tot[len(peaks_neg_tot)-1],
                              top, bot])
                # dbg_plt(boxes[-1], "fallback box")
        else:
            # order multi-column separators
            y_mid_by_order=[]
            x_start_by_order=[]
            x_end_by_order=[]
            if len(x_starting)>0:
                columns_covered_by_seps_covered_more_than_2col = set()
                for dj in range(len(x_starting)):
                    if set(range(x_starting[dj], x_ending[dj])) != all_columns:
                        columns_covered_by_seps_covered_more_than_2col.update(
                            range(x_starting[dj], x_ending[dj]))
                columns_not_covered = list(all_columns - columns_covered_by_seps_covered_more_than_2col)

                y_mid = np.append(y_mid, np.ones(len(columns_not_covered) + 1,
                                                 dtype=int) * top)
                ##y_mid_by_order = np.append(y_mid_by_order, [top] * len(columns_not_covered))
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
                y_mid = np.append(y_mid, np.ones(len(columns_not_covered),
                                                 dtype=int) * top)
                ##y_mid_by_order = np.append(y_mid_by_order, [top] * len(columns_not_covered))
                ##x_start_by_order = np.append(x_start_by_order, [0] * len(columns_not_covered))
                x_starting = np.append(x_starting, np.array(columns_not_covered, x_starting.dtype))
                x_ending = np.append(x_ending, np.array(columns_not_covered, x_ending.dtype) + 1)

            ind_args = np.arange(len(y_mid))

            for column in range(len(peaks_neg_tot)-1):
                #print(column,'column')
                ind_args_in_col=ind_args[x_starting==column]
                #print(len(y_mid))
                y_mid_column=y_mid[ind_args_in_col]
                x_start_column=x_starting[ind_args_in_col]
                x_end_column=x_ending[ind_args_in_col]

                ind_args_col_sorted = np.argsort(y_mid_column)
                y_mid_by_order.extend(y_mid_column[ind_args_col_sorted])
                x_start_by_order.extend(x_start_column[ind_args_col_sorted])
                x_end_by_order.extend(x_end_column[ind_args_col_sorted] - 1)

            # create single-column boxes from multi-column separators
            y_mid_by_order = np.array(y_mid_by_order)
            x_start_by_order = np.array(x_start_by_order)
            x_end_by_order = np.array(x_end_by_order)
            for il in range(len(y_mid_by_order)):
                #print(il, "il")
                y_mid_itself = y_mid_by_order[il]
                #print(y_mid_itself,'y_mid_itself')
                x_start_itself = x_start_by_order[il]
                x_end_itself = x_end_by_order[il]
                for column in range(x_start_itself, x_end_itself+1):
                    #print(column,'cols')
                    #print('burda2')
                    y_mid_next = y_mid_by_order[(y_mid_itself < y_mid_by_order) &
                                                (column >= x_start_by_order) &
                                                (column <= x_end_by_order)]
                    #print(y_mid_next,'y_mid_next')
                    y_mid_next = y_mid_next.min(initial=bot)
                    #print(y_mid_next,'y_mid_next')
                    boxes.append([peaks_neg_tot[column],
                                  peaks_neg_tot[column+1],
                                  y_mid_itself,
                                  y_mid_next])
                    # dbg_plt(boxes[-1], "B column %d box" % (column + 1))

    if right2left_readingorder:
        peaks_neg_tot_tables_new = []
        if len(peaks_neg_tot_tables)>=1:
            for peaks_tab_ind in peaks_neg_tot_tables:
                peaks_neg_tot_tables_ind = width_tot - np.array(peaks_tab_ind)
                peaks_neg_tot_tables_ind = list(peaks_neg_tot_tables_ind[::-1])
                peaks_neg_tot_tables_new.append(peaks_neg_tot_tables_ind)

        for i in range(len(boxes)):
            x_start_new = width_tot - boxes[i][1]
            x_end_new = width_tot - boxes[i][0]
            boxes[i][0] = x_start_new
            boxes[i][1] = x_end_new
        peaks_neg_tot_tables = peaks_neg_tot_tables_new

    # show final xy-cut
    # dbg_plt(None, "final XY-Cut", boxes, True)

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
