from __future__ import annotations
from collections.abc import Iterable
from logging import getLogger
import time
import math
from dataclasses import dataclass, field
from itertools import islice, compress

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = mpatches = None
import numpy as np
from shapely import geometry, prepared, ops
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from skimage import morphology

from .is_nan import isNaN
from .contour import (contour2polygon,
                      contours_in_same_horizon,
                      find_center_of_contours,
                      find_features_of_contours,
                      find_new_features_of_contours,
                      polygon2contour,
                      return_contours_of_class)


def pairwise(iterable):
    # pairwise('ABCDEFG') → AB BC CD DE EF FG

    iterator = iter(iterable)
    a = next(iterator, None)

    for b in iterator:
        yield a, b
        a = b

def batched(iterable, n):
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch

def itemgetter(seq):
    # replacement for operator.itemgetter
    # (which is ambiguous about its return type:
    #  single-item if 1 argument, tuple otherwise)
    def fun(obj):
        return [obj[idx] for idx in seq]
    return fun

@dataclass
class Region:
    contour: np.ndarray
    area: int = 0
    cx: float = 0
    cy: float = 0
    skew: float = 0
    conf: float = 0
    def __post_init__(self):
        self.area = cv2.contourArea(self.contour)
        moments = cv2.moments(self.contour)
        self.cx = moments['m10'] / (moments['m00'] or 1e-32)
        self.cy = moments['m01'] / (moments['m00'] or 1e-32)

@dataclass
class TextRegion(Region):
    lines: list[Region] = field(default_factory=list)

def return_multicol_separators_x_start_end(
        regions_without_separators, peak_points, top, bot,
        x_min_hor_some, x_max_hor_some, cy_hor_some, y_min_hor_some, y_max_hor_some):
    """
    Analyse which separators overlap multiple column candidates,
    and how they overlap each other.

    Ignore separators not spanning multiple columns.

    For the separators to be returned, try to remove or unify them when there
    is no region between them (vertically) and their neighbours.

    Arguments:
        * the text mask (with all separators suppressed)
        * the x column coordinates
        * the y start coordinate to consider in total
        * the y end coordinate to consider in total
        * the x start coordinate of the horizontal separators
        * the x end coordinate of the horizontal separators
        * the y start coordinate of the horizontal separators
        * the y center coordinate of the horizontal separators
        * the y end coordinate of the horizontal separators

    Returns:
        a tuple of:
        * the x start column index of the resulting multi-span separators
        * the x end column index of the resulting multi-span separators
        * the y start coordinate of the resulting multi-span separators
        * the y center coordinate of the resulting multi-span separators
        * the y end coordinate of the resulting multi-span separators
    """

    x_start = [0]
    x_end = [len(peak_points) - 1]
    y_min = [top]
    y_mid = [top]
    y_max = [top + 2]
    indexer = 1
    for i in range(len(x_min_hor_some)):
        #print(indexer, "%d:%d" % (x_min_hor_some[i], x_max_hor_some[i]), cy_hor_some[i])
        starting = x_min_hor_some[i] - peak_points
        min_start = np.flatnonzero(starting >= 0)[-1] # last left-of
        ending = x_max_hor_some[i] - peak_points
        max_end = np.flatnonzero(ending <= 0)[0] # first right-of
        #print(indexer, "%d:%d" % (min_start, max_end))

        if (max_end-min_start)>=2:
            # column range of separator spans more than one column candidate
            #print((max_end-min_start),len(peak_points),'(max_end-min_start)')
            y_min.append(y_min_hor_some[i])
            y_mid.append(cy_hor_some[i])
            y_max.append(y_max_hor_some[i])
            x_end.append(max_end)
            x_start.append(min_start)
            indexer+=1
    #print(x_start,'x_start')
    #print(x_end,'x_end')

    x_start = np.array(x_start, dtype=int)
    x_end = np.array(x_end, dtype=int)
    y_min = np.array(y_min, dtype=int)
    y_mid = np.array(y_mid, dtype=int)
    y_max = np.array(y_max, dtype=int)
    #print(y_mid,'y_mid')
    #print(x_start,'x_start')
    #print(x_end,'x_end')

    # remove redundant separators (with nothing in between)
    args_emptysep = set()
    args_ysorted = np.argsort(y_mid)
    for i in range(len(y_mid)):
        # find nearest neighbours above with nothing in between
        prev = (~np.eye(len(y_mid), dtype=bool)[i] &
                (y_mid[i] >= y_mid) &
                # complete subsumption:
                # (x_start[i] >= x_start) &
                # (x_end[i] <= x_end)
                # partial overlap
                (x_start[i] < x_end) &
                (x_end[i] > x_start)
        )
        prev[list(args_emptysep)] = False # but no pair we already saw
        if not prev.any():
            continue
        prev = np.flatnonzero(prev[args_ysorted])
        j = args_ysorted[prev[-1]]
        if not np.any(regions_without_separators[y_max[j]: y_min[i],
                                                 peak_points[min(x_start[i], x_start[j])]:
                                                 peak_points[max(x_end[i], x_end[j])]]):
            args_emptysep.add(i)
            if x_start[j] > x_start[i]: # noqa: PLR1730
                # print(j, "now starts at", x_start[i])
                x_start[j] = x_start[i]
            if x_end[j] < x_end[i]: # noqa: PLR1730
                # print(j, "now ends at", x_end[i])
                x_end[j] = x_end[i]
            # print(j, i, "%d:%d" % (y_mid[j], y_mid[i]), "%d:%d" % (x_start[i], x_end[i]), "empty prev sep")
            continue
        # find nearest neighbours below with nothing in between
        nExt = (~np.eye(len(y_mid), dtype=bool)[i] &
                (y_mid[i] <= y_mid) &
                (x_start[i] >= x_start) &
                (x_end[i] <= x_end))
        nExt[list(args_emptysep)] = False # but no pair we already saw
        if not nExt.any():
            continue
        nExt = np.flatnonzero(nExt[args_ysorted])
        j = args_ysorted[nExt[0]]
        if not np.any(regions_without_separators[y_max[i]: y_min[j],
                                                 peak_points[x_start[i]]:
                                                 peak_points[x_end[i]]]):
            args_emptysep.add(i)
            # print(j, i, "%d:%d" % (y_mid[j], y_mid[i]), "%d:%d" % (x_start[i], x_end[i]), "empty next sep")
    args_to_be_kept = [arg for arg in args_ysorted
                       if arg not in args_emptysep]
    x_start = x_start[args_to_be_kept]
    x_end = x_end[args_to_be_kept]
    y_min = y_min[args_to_be_kept]
    y_mid = y_mid[args_to_be_kept]
    y_max = y_max[args_to_be_kept]

    return (x_start,
            x_end,
            y_min,
            y_mid,
            y_max)

def box2rect(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (box[1], box[1] + box[3],
            box[0], box[0] + box[2])

def box2slice(box: tuple[int, int, int, int]) -> tuple[slice, slice]:
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

def find_features_of_lines(contours):
    """analyse contours

    Returns a tuple of arrays:
    - slope class (0 for nearly horizontal, 1 for nearly vertical, 2 otherwise)
    - the horizontal distance
    - the left boundary
    - the right boundary
    - the vertical centers
    - angles (in degrees, from -180 to 180)
    - the upper boundary
    - the lower boundary
    - the horizontal centers
    """
    cx, cy, x_min, x_max, y_min, y_max, _ = find_new_features_of_contours(contours)

    angles = np.empty(len(contours))
    for i, cont in enumerate(contours):
        vx, vy, _, _ = cv2.fitLine(cont, cv2.DIST_L2, 0, 0.01, 0.01)[:, 0]
        angles[i] = np.degrees(np.arctan2(vy, vx)) # [-180,180]

    slopes = np.empty(len(contours), dtype=np.uint8)
    slopes[:] = 2
    slopes[np.abs(angles) < 9.9] = 0
    slopes[np.abs(angles) > 74.] = 1

    dist_x = np.abs(x_max - x_min)
    return (slopes,
            dist_x,
            x_min,
            x_max,
            np.array(cy),
            angles,
            y_min,
            y_max,
            np.array(cx))

def boosting_headers_by_longshot_region_segmentation(textregion_pre_p, textregion_pre_np, img_only_text):
    textregion_pre_p_org = np.copy(textregion_pre_p)
    # 4 is drop capitals
    headers_in_longshot = textregion_pre_np == 2
    #headers_in_longshot = ((textregion_pre_np==2) |
    #                       (textregion_pre_np==1))
    textregion_pre_p[headers_in_longshot &
                     (textregion_pre_p != 4)] = 2
    textregion_pre_p[textregion_pre_p == 1] = 0
    # earlier it was so, but by this manner the drop capitals are also deleted
    # textregion_pre_p[(img_only_text[:,:]==1) &
    #                  (textregion_pre_p!=7) &
    #                  (textregion_pre_p!=2)] = 1
    textregion_pre_p[(img_only_text[:, :] == 1) &
                     (textregion_pre_p != 7) &
                     (textregion_pre_p != 4) &
                     (textregion_pre_p != 2)] = 1
    return textregion_pre_p

def get_projection_var(regions_without_separators, sigma, axis=1):
    z = regions_without_separators.sum(axis=axis)
    z = gaussian_filter1d(z, sigma)
    return np.std(z)

def find_num_col(
        regions_without_separators,
        num_col_classifier,
        tables,
        multiplier=3.8,
        unbalanced=False,
        vertical_separators=None
):
    if not regions_without_separators.any():
        return 0, []
    if vertical_separators is None:
        vertical_separators = np.zeros_like(regions_without_separators)
    regions_without_separators_0 = regions_without_separators.sum(axis=0)
    vertical_separators_0 = vertical_separators.sum(axis=0)
    # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # ax1.imshow(regions_without_separators, aspect="auto")
    # ax2.plot(regions_without_separators_0)
    # plt.show()
    sigma_ = 25  # 70#35
    meda_n_updown = regions_without_separators_0[::-1]
    first_nonzero = next((i for i, x in enumerate(regions_without_separators_0) if x), 0)
    last_nonzero = next((i for i, x in enumerate(meda_n_updown) if x), 0)
    last_nonzero = len(regions_without_separators_0) - last_nonzero
    last_nonzero = last_nonzero - 50 #- 100
    first_nonzero = first_nonzero + 50 #+ 200
    last_offmargin = len(regions_without_separators_0) - 170 #370
    first_offmargin = 170 #370
    x = vertical_separators_0
    y = regions_without_separators_0  # [first_nonzero:last_nonzero]
    y_help = np.pad(y, (10, 10), constant_values=(0, 0))
    zneg_rev = y.max() - y_help
    zneg = np.pad(zneg_rev, (10, 10), constant_values=(0, 0))
    x = gaussian_filter1d(x, sigma_)
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
    # ax2.axvline(first_offmargin, label="first offmargin")
    # ax2.axvline(last_offmargin, label="last offmargin")
    # ax2.text(first_offmargin, 0, "first offmargin", rotation=90)
    # ax2.text(last_offmargin, 0, "last offmargin", rotation=90)
    # plt.show()
    peaks_neg = peaks_neg - 10 - 10

    # print("raw peaks", peaks)
    peaks = peaks[(peaks > 0.06 * len(y)) &
                  (peaks < 0.94 * len(y))]
    # print("non-marginal peaks", peaks)
    interest_pos = z[peaks]
    # print("interest_pos", interest_pos)
    interest_pos = interest_pos[interest_pos > 10]
    if not interest_pos.any():
        return 0, []

    # plt.plot(z)
    # plt.show()
    #print("raw peaks_neg", peaks_neg)
    peaks_neg = peaks_neg[(peaks_neg > first_nonzero) &
                          (peaks_neg < last_nonzero)]
    #print("non-zero peaks_neg", peaks_neg)
    peaks_neg = peaks_neg[(peaks_neg > first_offmargin) &
                          (peaks_neg < last_offmargin)]
    #print("non-marginal peaks_neg", peaks_neg)
    interest_neg = z[peaks_neg]
    #print("interest_neg", interest_neg)
    if not interest_neg.any():
        return 0, []

    min_peaks_pos = np.min(interest_pos)
    max_peaks_pos = np.max(interest_pos)

    #print(min_peaks_pos, max_peaks_pos, max_peaks_pos / min_peaks_pos, 'minmax')
    if max_peaks_pos / (min_peaks_pos or 1e-9) >= 35:
        min_peaks_pos = np.mean(interest_pos)

    min_peaks_neg = 0  # np.min(interest_neg)

    # cutoff criterion: fixed fraction of lowest column height
    dis_talaei = (min_peaks_pos - min_peaks_neg) / multiplier
    grenze = min_peaks_pos - dis_talaei
    #np.mean(y[peaks_neg[0]:peaks_neg[-1]])-np.std(y[peaks_neg[0]:peaks_neg[-1]])/2.0

    # extra criterion: fixed multiple of lowest gap height
    # print("grenze", grenze, multiplier * (5 + np.min(interest_neg)))
    grenze = min(grenze, multiplier * (5 + np.min(interest_neg)))

    # print(interest_neg,'interest_neg')
    # print(grenze,'grenze')
    # print(min_peaks_pos,'min_peaks_pos')
    # print(dis_talaei,'dis_talaei')
    # print(peaks_neg,'peaks_neg')
    # fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # ax1.imshow(regions_without_separators + 5 * vertical_separators, aspect="auto")
    # ax2.plot(z, color='red', label='z')
    # ax2.plot(zneg[20:], color='blue', label='zneg')
    # ax2.plot(x, color='green', label='vsep')
    # ax2.scatter(peaks_neg, z[peaks_neg], color='red')
    # ax2.scatter(peaks_neg, zneg[20:][peaks_neg], color='blue')
    # ax2.axhline(min_peaks_pos, color='red')
    # ax2.axhline(grenze, color='blue')
    # ax2.annotate("min_peaks_pos", xy=(0, min_peaks_pos), color='red')
    # ax2.annotate("grenze", xy=(0, grenze), color='blue')
    # ax2.text(0, grenze, "grenze")
    # ax2.legend()
    # plt.show()

    # print("vsep", x[peaks_neg])
    interest_neg = interest_neg - x[peaks_neg]
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
    if unbalanced:
        pass
    elif ((num_col == 3 and
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
    cut_off = 300 #400
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
    if unbalanced:
        pass
    elif ((num_col == 3 and
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
    elif (num_col == 3 and
          (peaks_neg_true[0] < 0.75 * len(y) and
           peaks_neg_true[0] > 0.25 * len(y) and
           peaks_neg_true[1] > 0.80 * len(y))):
        num_col = 2
        peaks_neg_true = [peaks_neg_true[0]]
    elif (num_col == 3 and
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

def fill_bb_of_drop_capitals(
        full_prediction, early_prediction,
        label_bg=0,
        label_text=1,
        label_imgs=5,
        label_drop_fl_model=3,
        label_imgs_fl_model=4):
    """
    Given segmentation maps from full layout model (including drop-capital)
    and early layout model (after post-processing), re-assign regions which
    are (large enough and) majority classified as drop-capital to that label.
    """
    drop_mask = (full_prediction == label_drop_fl_model).astype(np.uint8)
    drop_cont = return_contours_of_class(drop_mask, 1, 1e-5)
    text_mask = ((early_prediction == label_text) |
                 (early_prediction == label_imgs))
    _, text_segs, text_bbox, _ = cv2.connectedComponentsWithStats(early_prediction * text_mask)

    for contour in drop_cont:
        area_drop = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        box = slice(y, y + h), slice(x, x + w)
        area_box = w * h
        area_text_in_early_layout = np.sum(text_mask[box] == label_text)

        if (area_drop > 0.6 * area_box and
            area_text_in_early_layout >= 0.3 * area_box):
            mask = np.ones((h, w), dtype=bool)
        else:
            mask = ((full_prediction[box] == label_drop_fl_model) |
                    (full_prediction[box] == label_imgs_fl_model) |
                    (full_prediction[box] == label_bg))
        full_prediction[box][mask] = label_drop_fl_model

        # also try to enlarge to corresponding labels in early_prediction
        for label in range(1, len(text_bbox)):
            x0, y0, w0, h0, area0 = text_bbox[label]
            x1 = max(0, x0 - x)
            y1 = max(0, y0 - y)
            w1 = min(w0, w - x1) if x0 >= x else min(w, w0 - x + x0)
            h1 = min(h0, h - y1) if y0 >= y else min(h, h0 - y + y0)
            if w1 < 0 or h1 < 0:
                continue
            area1 = np.count_nonzero(mask[y1: y1 + h1, x1: x1 + w1])
            if area1 and area1 >= 0.8 * area0:
                full_prediction[text_segs == label] = label_drop_fl_model

    return full_prediction == label_drop_fl_model

def split_textregion_main_vs_head(
        regions_model_1: np.ndarray,
        regions_model_full: np.ndarray,
        textregions: list[Region],
        textregions_d: list[Region],
        label_text=1,
        label_head_full=2,
        label_head_final=2,
        label_main_final=1,
) -> tuple[np.ndarray, list[Region], list[Region], list[Region], list[Region]]:

    ### to make it faster
    h_o = regions_model_1.shape[0]
    w_o = regions_model_1.shape[1]
    zoom = 3
    regions_model_1 = cv2.resize(regions_model_1,
                                 (regions_model_1.shape[1] // zoom,
                                  regions_model_1.shape[0] // zoom),
                                 interpolation=cv2.INTER_NEAREST)
    regions_model_full = cv2.resize(regions_model_full,
                                    (regions_model_full.shape[1] // zoom,
                                     regions_model_full.shape[0] // zoom),
                                    interpolation=cv2.INTER_NEAREST)
    contours_z = [textregion.contour // zoom
                  for textregion in textregions]

    main = np.ones(len(contours_z), dtype=bool)
    for ii, con in enumerate(contours_z):
        width, height = cv2.boundingRect(con)[2:]
        parent = np.zeros_like(regions_model_1)
        parent = cv2.fillPoly(parent, pts=[con], color=1)

        pixels_head = ((parent > 0) & (regions_model_full == label_head_full)).sum()
        pixels_main = parent.sum() - pixels_head

        if (( pixels_head >= 0.6 * pixels_main and
              width >= 1.3 * height and
              width <= 3 * height ) or
            ( pixels_head >= 0.3 * pixels_main and
              width >= 3 * height )):

            main[ii] = False
            label = label_head_final
        else:
            label = label_main_final

        regions_model_1[(regions_model_1 == label_text) & (parent > 0)] = label

    ### to make it faster
    regions_model_1 = cv2.resize(regions_model_1, (w_o, h_o), interpolation=cv2.INTER_NEAREST)
    # regions_model_full = cv2.resize(parent, (regions_model_full.shape[1] // zoom,
    #                                          regions_model_full.shape[0] // zoom),
    #                                 interpolation=cv2.INTER_NEAREST)
    ###

    return (regions_model_1,
            list(compress(textregions, main)),
            list(compress(textregions, ~main)),
            list(compress(textregions_d, main)),
            list(compress(textregions_d, ~main)),
    )

def small_textlines_to_parent_adherence2(
        textregions: list[TextRegion],
        area_factor: float,
        num_col: int,
) -> None:
    """
    for each region, split up textlines into small and large;
    keep only the ones with large area, but expanded (by merging
    contours) by all intersecting lines with small area
    """
    textlines_con_new = []
    for textregion in textregions:
        areas = np.array([line.area for line in textregion.lines]) * area_factor

        if num_col == 0:
            min_area = 0.0004
        elif num_col == 1:
            min_area = 0.0003
        else:
            min_area = 0.0001
        textlines_small = list(compress(textregion.lines, areas < min_area))
        textlines_large = list(compress(textregion.lines, areas >= min_area))
        textlines_small_poly = [contour2polygon(line.contour) for line in textlines_small]
        textlines_large_poly = [contour2polygon(line.contour) for line in textlines_large]
        textregion.lines = textlines_large

        if geometry.MultiPolygon(textlines_small_poly).intersects(
                geometry.MultiPolygon(textlines_large_poly)):
            # FIXME: also consider confidence (less certain lines replaced by better ones)...
            textlines_large_prep = [prepared.prep(poly) for poly in textlines_large_poly]
            textlines_small_indexes_interlarge = []
            for small_poly in textlines_small_poly:
                intersections = [small_poly.intersection(prep.context).area
                                 if prep.intersects(small_poly) else 0
                                 for prep in textlines_large_prep]
                idx_large = np.argmax(intersections)
                if intersections[idx_large] == 0:
                    idx_large = -1
                textlines_small_indexes_interlarge.append(idx_large)
            textlines_small_indexes_interlarge = np.array(textlines_small_indexes_interlarge)
            for idx_large in set(textlines_small_indexes_interlarge):
                if idx_large < 0:
                    continue
                large_poly = textlines_large_poly[idx_large]
                indexes_small = np.flatnonzero(textlines_small_indexes_interlarge == idx_large)
                for idx_small in indexes_small:
                    large_poly = large_poly.union(textlines_small_poly[idx_small])
                if large_poly.geom_type == 'GeometryCollection':
                    # hetergeneous: filter lines and points
                    large_poly = ops.unary_union([geom for geom in large_poly.geoms
                                                  if geom.area > 0])
                if large_poly.geom_type == 'MultiPolygon':
                    # disjoint: pick largest part
                    idx_large = np.argmax([geom.area for geom in large_poly.geoms])
                    large_poly = large_poly.geoms[idx_large]
                # replace original
                textregion.lines[idx_large].contour = polygon2contour(large_poly)

def order_of_regions(contours_main, contours_head, contours_drop, r2l=False):
    """
    Order text region contours within a single column bbox in a top-down-left-right way.

    \b
    First, pre-sort by vertical centers. Then, from the adjacent contours, 
    form groups to be sorted by horizontal centers: contours belong
    into the same group, iff
    - they overlap each other vertically centrally and
    - they do not overlap each other horizontally significantly.

    Arguments:
      * contours_main: paragraph text region contours to be sorted
      * contours_head: the heading text region contours to be sorted
      * contours_drop: the drop-capital region contours to be sorted

    Keyword Args:
      * r2l: whether contours within groups should be ordered
        right-to-left (instead of left-to-right)

    Returns: a tuple of
      * the list of contour indexes overall within this box
            (i.e. into main+head+drop)
      * the list of types
            (1 for paragraph, 2 for heading, 3 for drop-capital)
      * the list of contour indexes for the respective type
            (i.e. into contours_main or contours_head or contours_drop)
    """
    total = len(contours_main) + len(contours_head) + len(contours_drop)
    if not total:
        return [], [], []

    contours = np.concatenate((contours_main, contours_head, contours_drop))
    index = np.arange(len(contours))
    types = np.array([1] * len(contours_main) +
                     [2] * len(contours_head) +
                     [3] * len(contours_drop))
    local_index = np.array(list(range(len(contours_main))) +
                           list(range(len(contours_head))) +
                           list(range(len(contours_drop))))
    cx, cy = find_center_of_contours(contours)
    y_min = [contour[:, 0, 1].min() for contour in contours]
    y_max = [contour[:, 0, 1].max() for contour in contours]
    x_min = [contour[:, 0, 0].min() for contour in contours]
    x_max = [contour[:, 0, 0].max() for contour in contours]
    yorder = np.argsort(cy)
    groups = [[yorder[0]]]
    for i, j in pairwise(yorder):
        if (
                # i/j overlap vertically:
                (y_max[i] >= y_min[j] and y_max[j] >= y_min[i])
                and
                # i/j are centered vertically:
                (y_min[j] <= cy[i] < y_max[j] or
                 y_min[i] <= cy[j] < y_max[i])
                and
                # there is no k (=i or any contour already in that group)
                # horizontally in j's vicinity (overlapping more than 10% of their width):
                not any(max(0, min(x_max[j], x_max[k]) - max(x_min[j], x_min[k])) >
                        min(x_max[j] - x_min[j], x_max[k] - x_min[k]) * 0.1
                        for k in groups[-1])
                ):
            groups[-1].append(j)
        else:
            groups.append([j])
    cx = np.array(cx)
    cy = np.array(cy)
    rorder = []
    for group in groups:
        group = np.array(group)
        xorder = np.argsort(cx[group])[::-1 if r2l else 1]
        rorder.extend(group[xorder])

    assert len(set(rorder)) == total

    return rorder, types[rorder], local_index[rorder]

def combine_hor_lines(
        img_p_in_ver: np.ndarray,
        img_p_in_hor: np.ndarray,
        num_col_classifier: int,
) -> tuple[np.ndarray, list[float]]:
    """
    Given a horizontal and vertical separator mask, combine horizontal separators
    (where possible) and identify those spanning the entire page

    Arguments:
      * img_p_in_ver: mask of vertical separators
      * img_p_in_hor: mask of horizontal separators
      * num_col_classifier: predicted (expected) number of columns

    Returns: a tuple of
      * the final horizontal separators
      * the y coordinates with horizontal separators spanning the full width
    """
    height, width = img_p_in_ver.shape

    # cut horizontal seps by vertical seps
    # rs: does more harm than good
    # (continuous horizontal seps are needed as page splitters)
    #img_p_in_hor[img_p_in_ver > 0] = 0

    #img_p_in_ver = cv2.erode(img_p_in_ver, self.kernel, iterations=2)
    contours_seps_ver, _ = cv2.findContours(img_p_in_ver.astype(np.uint8),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    y_min_ver, y_max_ver = find_features_of_contours(contours_seps_ver)
    cx_ver, _ = find_center_of_contours(contours_seps_ver)
    # shorten vertical seps at both ends
    # (so they will not be considered crossing
    #  any horizontal seps below):
    for i in range(len(contours_seps_ver)):
        img_p_in_ver[int(y_min_ver[i]):
                     int(y_min_ver[i]) + 30,
                     int(cx_ver[i]) - 25:
                     int(cx_ver[i]) + 25] = 0
        img_p_in_ver[int(y_max_ver[i]) - 30:
                     int(y_max_ver[i] + 1),
                     int(cx_ver[i]) - 25:
                     int(cx_ver[i]) + 25] = 0

    contours_seps_hor, _ = cv2.findContours(img_p_in_hor.astype(np.uint8),
                                             cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
    (_,
     dist_x_hor,
     x_min_hor,
     x_max_hor,
     cy_hor, _,
     y_min_hor,
     y_max_hor,
     _) = find_features_of_lines(contours_seps_hor)

    avg_col_width = width / float(num_col_classifier + 1)
    nseps_wider_than_than_avg_col_width = np.count_nonzero(dist_x_hor>=avg_col_width)
    if nseps_wider_than_than_avg_col_width < 10 * num_col_classifier:
        args_hor = np.arange(len(contours_seps_hor))
        sep_pairs = contours_in_same_horizon(cy_hor)
        img_p_in = np.copy(img_p_in_hor)
        if len(sep_pairs):
            special_separators=[]
            contours_new=[]
            for pair in sep_pairs:
                merged_all=None
                some_args=args_hor[pair]
                some_cy=cy_hor[pair]
                some_x_min=x_min_hor[pair]
                some_x_max=x_max_hor[pair]
                some_y_min=y_min_hor[pair]
                some_y_max=y_max_hor[pair]
                if np.any(img_p_in_ver[some_y_min.min() - 0: some_y_max.max() + 0,
                                       some_x_max.min() - 3: some_x_min.max() + 3]):
                    # print("horizontal pair cut by vertical sep", pair, some_args, some_cy,
                    #       "%d:%d" % (some_x_min[0], some_x_max[0]),
                    #       "%d:%d" % (some_x_min[1], some_x_max[1]))
                    continue

                #img_in=np.zeros(separators_closeup_n[:,:,2].shape)
                #print(img_p_in_ver.shape[1],some_x_max-some_x_min,'xdiff')
                sum_xspan = dist_x_hor[some_args].sum()
                tot_xspan = (np.max(x_max_hor[some_args]) -
                             np.min(x_min_hor[some_args]))
                dev_xspan = (np.std(dist_x_hor[some_args]) /
                             np.mean(dist_x_hor[some_args])) if sum_xspan else 1
                if (tot_xspan > sum_xspan and # no x overlap
                    sum_xspan > 0.85 * tot_xspan): # x close to each other
                    # print("merging horizontal pair", pair, some_args, some_cy,
                    #       "%d:%d" % (some_x_min[0], some_x_max[0]),
                    #       "%d:%d" % (some_x_min[1], some_x_max[1]))
                    img_p_in[int(np.mean(some_cy)) - 5:
                             int(np.mean(some_cy)) + 5,
                             np.min(some_x_min):
                             np.max(some_x_max)] = 1

                if (tot_xspan > sum_xspan and # no x overlap
                    sum_xspan > 0.85 * tot_xspan and # x close to each other
                    tot_xspan > 0.85 * width and # nearly full width
                    dev_xspan < 0.55): # similar x span
                    # print(dist_x_hor[some_args],
                    #       dist_x_hor[some_args].sum(),
                    #       np.min(x_min_hor[some_args]),
                    #       np.max(x_max_hor[some_args]),'jalibdi')
                    # print(np.mean( dist_x_hor[some_args] ),
                    #       np.std( dist_x_hor[some_args] ),
                    #       np.var( dist_x_hor[some_args] ),'jalibdiha')
                    special_separators.append(np.mean(cy_hor[some_args]))
                    # print("special separator for midline", special_separators[-1])
            # plt.subplot(1, 2, 1, title='original horizontal (1) / vertical (2) seps')
            # plt.imshow(1 * (img_p_in_hor > 0) + 2 * (img_p_in_ver > 0))
            # plt.subplot(1, 2, 2, title='extended horizontal seps')
            # plt.imshow(img_p_in)
            # plt.show()
        else:
            img_p_in = img_p_in_hor
            special_separators = []

        # rs: removing seps around crossings: unnecessary
        #     and causes small isolated seps at crosspoints
        # sep_ver_hor_cross = 1 * ((img_p_in > 0) & (img_p_in_ver > 0))
        # contours_cross, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # center_cross = np.array(find_center_of_contours(contours_cross), dtype=int)
        # for cx, cy in center_cross.T:
        #     img_p_in[cy - 30: cy + 30, cx + 5: cx + 40] = 0
        #     img_p_in[cy - 30: cy + 30, cx - 40: cx - 4] = 0
    else:
        img_p_in = np.copy(img_p_in_hor)
        special_separators = []
    return img_p_in, special_separators

def return_points_with_boundies(peaks_neg_fin, first_point, last_point):
    peaks_neg_tot = []
    peaks_neg_tot.append(first_point)
    for ii in range(len(peaks_neg_fin)):
        peaks_neg_tot.append(peaks_neg_fin[ii])
    peaks_neg_tot.append(last_point)
    return peaks_neg_tot

def find_number_of_columns_in_document(
        regions_without_separators: np.ndarray,
        separator_mask: np.ndarray,
        num_col_classifier: int,
        tables: bool,
        contours_h: list[np.ndarray] = [], # ruff: ignore[B006] (not modified)
        logger=None
) -> tuple[int, list[int], np.ndarray, list[int], np.ndarray]:
    """
    Extract vertical and horizontal separators and vertical splits on page.

    Arguments:
      * regions_without_separators: mask of (non-separator) region labels
      * separator_mask: mask of (separator-only) region labels
      * num_col_classifier: predicted (expected) number of columns of the page
      * tables: whether tables may be present
      * contours_h: contours of potential headings (serving as additional horizontal separators)
      * logger

    Returns: a tuple of
      * an array of the separators (bounding boxes and types)
      * the y coordinates of the page splits
    """
    if logger is None:
        logger = getLogger(__package__)

    height, width = separator_mask.shape
    separators_closeup = separator_mask.astype(np.uint8)
    separators_closeup[0:110] = 0
    separators_closeup[-150:] = 0

    kernel = np.ones((5,5),np.uint8)
    separators_closeup = cv2.morphologyEx(separators_closeup,
                                          cv2.MORPH_CLOSE, kernel, iterations=1)
    # find horizontal lines by contour properties
    contours_sep_e, _ = cv2.findContours(separators_closeup,
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
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
    separators_closeup = cv2.fillPoly(separators_closeup, pts=cnts_hor_e, color=0)
    horizontal = np.copy(separators_closeup)
    vertical = np.copy(separators_closeup)

    horizontal_size = horizontal.shape[1] // 30
    # find horizontal lines by morphology
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontalStructure)
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, kernel, iterations=2)
    # re-insert deleted horizontal contours
    horizontal = cv2.fillPoly(horizontal, pts=cnts_hor_e, color=1)

    vertical_size = vertical.shape[0] // 30
    # find vertical lines by morphology
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, verticalStructure)
    vertical = cv2.dilate(vertical, kernel, iterations=1)

    horizontal, special_separators = combine_hor_lines(
            vertical, horizontal, num_col_classifier)

    contours_seps_ver, _ = cv2.findContours(vertical.astype(np.uint8),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    (slope_seps,
     dist_x,
     x_min_seps,
     x_max_seps,
     cy_seps,
     _,
     y_min_seps,
     y_max_seps,
     cx_seps) = find_features_of_lines(contours_seps_ver)
    args = np.arange(len(contours_seps_ver))
    is_vertical = slope_seps == 1
    args_ver = args[is_vertical]
    dist_x_ver = dist_x[is_vertical]
    y_min_ver = y_min_seps[is_vertical]
    y_max_ver = y_max_seps[is_vertical]
    x_min_ver = x_min_seps[is_vertical]
    x_max_ver = x_max_seps[is_vertical]
    cx_ver = cx_seps[is_vertical]
    dist_y_ver = y_max_ver - y_min_ver

    contours_seps_hor, _ = cv2.findContours(horizontal.astype(np.uint8),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    (slope_seps,
     dist_x,
     x_min_seps,
     x_max_seps,
     cy_seps,
     _,
     y_min_seps,
     y_max_seps,
     cx_seps) = find_features_of_lines(contours_seps_hor)
    args = np.arange(len(contours_seps_hor))
    is_horizontal = (slope_seps == 0) & (dist_x >= width / 10.)
    args_hor = args[is_horizontal]
    dist_x_hor = dist_x[is_horizontal]
    y_min_hor = y_min_seps[is_horizontal]
    y_max_hor = y_max_seps[is_horizontal]
    x_min_hor = x_min_seps[is_horizontal]
    x_max_hor = x_max_seps[is_horizontal]
    cy_hor = cy_seps[is_horizontal]
    dist_y_hor = y_max_hor - y_min_hor

    matrix_of_seps_ch = np.zeros((len(cy_hor) + len(cx_ver), 10), dtype=int)
    matrix_of_seps_ch[:len(cy_hor), 0] = args_hor
    matrix_of_seps_ch[len(cy_hor):, 0] = args_ver
    matrix_of_seps_ch[len(cy_hor):, 1] = cx_ver
    matrix_of_seps_ch[:len(cy_hor), 2] = x_min_hor + 50 # x_min_hor+150
    matrix_of_seps_ch[len(cy_hor):, 2] = x_min_ver
    matrix_of_seps_ch[:len(cy_hor), 3] = x_max_hor - 50 # x_max_hor-150
    matrix_of_seps_ch[len(cy_hor):, 3] = x_max_ver
    matrix_of_seps_ch[:len(cy_hor), 4] = dist_x_hor
    matrix_of_seps_ch[len(cy_hor):, 4] = dist_x_ver
    matrix_of_seps_ch[:len(cy_hor), 5] = cy_hor
    matrix_of_seps_ch[:len(cy_hor), 6] = y_min_hor
    matrix_of_seps_ch[len(cy_hor):, 6] = y_min_ver
    matrix_of_seps_ch[:len(cy_hor), 7] = y_max_hor
    matrix_of_seps_ch[len(cy_hor):, 7] = y_max_ver
    matrix_of_seps_ch[:len(cy_hor), 8] = dist_y_hor
    matrix_of_seps_ch[len(cy_hor):, 8] = dist_y_ver
    matrix_of_seps_ch[len(cy_hor):, 9] = 1

    if len(contours_h):
        (_,
         dist_x_head,
         x_min_head,
         x_max_head,
         cy_head,
         _,
         y_min_head,
         y_max_head,
         _) = find_features_of_lines(contours_h)
        matrix_l_n = np.zeros((len(contours_h), matrix_of_seps_ch.shape[1]), dtype=int)
        args_head = np.arange(len(contours_h))
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

    # ensure no seps are out of bounds
    matrix_of_seps_ch[:, 1] = np.maximum(np.minimum(matrix_of_seps_ch[:, 1], width), 0)
    matrix_of_seps_ch[:, 2] = np.maximum(matrix_of_seps_ch[:, 2], 0)
    matrix_of_seps_ch[:, 3] = np.minimum(matrix_of_seps_ch[:, 3], width)
    matrix_of_seps_ch[:, 5] = np.maximum(np.minimum(matrix_of_seps_ch[:, 5], height), 0)
    matrix_of_seps_ch[:, 6] = np.maximum(matrix_of_seps_ch[:, 6], 0)
    matrix_of_seps_ch[:, 7] = np.minimum(matrix_of_seps_ch[:, 7], height)

    cy_splitters = cy_hor[(x_min_hor <= .16 * width) &
                          (x_max_hor >= .84 * width)]
    cy_splitters = np.append(cy_splitters, special_separators)

    if len(contours_h):
        y_min_splitters_head = y_min_head[(x_min_head <= .16 * width) &
                                          (x_max_head >= .84 * width)]
        y_max_splitters_head = y_max_head[(x_min_head <= .16 * width) &
                                          (x_max_head >= .84 * width)]
        cy_splitters = np.append(cy_splitters, y_min_splitters_head)
        cy_splitters = np.append(cy_splitters, y_max_splitters_head)

    regions_y = gaussian_filter1d(regions_without_separators.sum(axis=1), 10)
    peaks_y, _ = find_peaks(regions_y.max() - regions_y, distance=20)
    for peak_y in peaks_y:
        if regions_y[peak_y] > 0.02 * width or vertical[peak_y].any():
            continue
        if not regions_without_separators[:peak_y].any():
            continue
        if not regions_without_separators[peak_y:].any():
            continue
        cy_splitters = np.append(cy_splitters, peak_y)

    cy_splitters = np.sort(cy_splitters).astype(int)
    splitter_y_new = [0] + list(cy_splitters) + [height]

    return matrix_of_seps_ch, splitter_y_new

def return_boxes_of_images_by_order_of_reading_new(
        splitter_y_new,
        text_mask,
        sep_mask,
        matrix_of_seps_ch,
        num_col_classifier, erosion_hurts, tables,
        right2left_readingorder,
        logger=None):
    """
    Iterate through the vertical parts of a page, each with its own set of columns,
    and from the matrix of horizontal separators for that part, find an ordered
    list of bounding boxes through all columns and regions.

    Arguments:
       * splitter_y_new: the y coordinates separating the parts
       * text_mask: binary text region mask
             (needed to find per-part columns and to combine separators if possible)
       * sep_mask: binary separator region mask
             (needed to elongate separators if possible)
       * matrix_of_seps: type and coordinates of horizontal and vertical separators,
             as well as headings
       * num_col_classifier: predicted number of columns for the entire page
       * erosion_hurts: whether region masks have already been eroded
                        (and thus gaps can be expected to be wider)
       * tables: bool
       * right2left_readingorder: whether to invert the default left-to-right order

    Returns: a tuple of
       * the ordered list of bounding boxes
       * a list of arrays: the x coordinates delimiting the columns for every page part
             (according to splitter)
    """

    if right2left_readingorder:
        text_mask = cv2.flip(text_mask,1)
        sep_mask = cv2.flip(sep_mask,1)
    if logger is None:
        logger = getLogger(__package__)
    logger.debug('enter return_boxes_of_images_by_order_of_reading_new')

    # def dbg_imshow(box, title):
    #     xmin, xmax, ymin, ymax = box
    #     plt.imshow(1 * text_mask + 3 * sep_mask) #, extent=[0, width_tot, bot, top])
    #     plt.gca().add_patch(mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                            fill=False, linewidth=1, edgecolor='r'))
    #     plt.title(title + " at %d:%d, %d:%d" % (ymin, ymax, xmin, xmax))
    #     plt.show()
    # def dbg_plt(box=None, title=None, rectangles=None, rectangles_showidx=False):
    #     minx, maxx, miny, maxy = box or (0, None, 0, None)
    #     img = text_mask[miny:maxy, minx:maxx]
    #     plt.imshow(img)
    #     step = max(img.shape) // 10
    #     xrange = np.arange(0, img.shape[1], step)
    #     yrange = np.arange(0, img.shape[0], step)
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
    #             ax.add_patch(mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                             fill=False, linewidth=1, edgecolor='r'))
    #             if rectangles_showidx:
    #                 ax.text((xmin+xmax)/2, (ymin+ymax)/2, str(i), c='r')
    #     plt.show()
    # dbg_plt(title="return_boxes_of_images_by_order_of_reading_new")

    boxes=[]
    peaks_neg_tot_tables = []
    splitter_y_new = np.array(splitter_y_new, dtype=int)
    height_tot, width_tot = text_mask.shape
    big_part = 22 * height_tot // 100 # percent height
    _, ccomps, cstats, _ = cv2.connectedComponentsWithStats(text_mask.astype(np.uint8))
    args_ver = matrix_of_seps_ch[:, 9] == 1
    mask_ver = np.zeros_like(sep_mask, dtype=bool)
    for i in np.flatnonzero(args_ver):
        mask_ver[matrix_of_seps_ch[i, 6]: matrix_of_seps_ch[i, 7],
                 matrix_of_seps_ch[i, 2]: matrix_of_seps_ch[i, 3]] = True
    vertical_seps = 1 * (sep_mask & mask_ver)
    for top, bot in pairwise(splitter_y_new):
        # print("%d:%d" % (top, bot), 'i')
        # dbg_plt([0, None, top, bot], "image cut for y split %d:%d" % (top, bot))
        matrix_new = matrix_of_seps_ch[(matrix_of_seps_ch[:,6] >= top) &
                                       (matrix_of_seps_ch[:,7] < bot)]
        #print(len( matrix_new[:,9][matrix_new[:,9]==1] ))
        #print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')
        # check to see is there any vertical separator to find holes.
        #if (len(matrix_new[:,9][matrix_new[:,9]==1]) > 0 and
        #    np.max(matrix_new[:,8][matrix_new[:,9]==1]) >=
        #    0.1 * (np.abs(bot-top))):
        num_col, peaks_neg_fin = find_num_col(
            text_mask[top:bot],
            # we do not expect to get all columns in small parts (headings etc.):
            num_col_classifier if bot - top >= big_part else 1,
            tables, vertical_separators=vertical_seps[top: bot],
            multiplier=6. if erosion_hurts else 7.,
            unbalanced=True)
        try:
            if ((len(peaks_neg_fin) + 1 < num_col_classifier or
                num_col_classifier == 6) and
                # we do not expect to get all columns in small parts (headings etc.):
                bot - top >= big_part):
                # found too few columns here
                #print('burda')
                logger.debug("searching for more than %d columns in big part %d:%d",
                             len(peaks_neg_fin) + 1, top, bot)
                peaks_neg_fin_org = np.copy(peaks_neg_fin)
                #print("peaks_neg_fin_org", peaks_neg_fin_org)
                if len(peaks_neg_fin) == 0:
                    num_col, peaks_neg_fin = find_num_col(
                        text_mask[top:bot],
                        num_col_classifier, tables,
                        vertical_separators=vertical_seps[top: bot],
                        # try to be less strict (lower threshold than above)
                        multiplier=7. if erosion_hurts else 8.,
                        unbalanced=True)
                #print(peaks_neg_fin,'peaks_neg_fin')
                peaks_neg_fin_early = [0] + peaks_neg_fin + [width_tot-1]

                #print(peaks_neg_fin_early,'burda2')
                peaks_neg_fin_rev=[]
                for left, right in pairwise(peaks_neg_fin_early):
                    # print("%d:%d" % (left, right), 'i_n')
                    # dbg_plt([left, right, top, bot],
                    #         "image cut for y split %d:%d / x gap %d:%d" % (
                    #             top, bot, left, right))
                    # plt.plot(text_mask[top:bot, left:right].sum(axis=0))
                    # plt.title("vertical projection (sum over y)")
                    # plt.show()
                    # try to get more peaks with different multipliers
                    num_col_expected = round((right - left) / width_tot * num_col_classifier)
                    args = text_mask[top:bot, left:right], num_col_expected, tables
                    kwargs = dict(vertical_separators=vertical_seps[top: bot, left:right])
                    _, peaks_neg_fin1 = find_num_col(*args, **kwargs, multiplier=7.)
                    _, peaks_neg_fin2 = find_num_col(*args, **kwargs, multiplier=5.)
                    if len(peaks_neg_fin1) >= len(peaks_neg_fin2):
                        peaks_neg_fin = peaks_neg_fin1
                    else:
                        peaks_neg_fin = peaks_neg_fin2
                    # print(peaks_neg_fin)
                    logger.debug("found %d additional column boundaries in %d:%d",
                                 len(peaks_neg_fin), left, right)
                    # add offset to local result
                    peaks_neg_fin = list(np.array(peaks_neg_fin) + left)
                    #print(peaks_neg_fin,'peaks_neg_fin')

                    peaks_neg_fin_rev.extend(peaks_neg_fin)
                    if right < peaks_neg_fin_early[-1]:
                        # all but the last column: interject the preexisting boundary
                        peaks_neg_fin_rev.append(right)
                    #print(peaks_neg_fin_rev,'peaks_neg_fin_rev')

                if len(peaks_neg_fin_rev) >= len(peaks_neg_fin_org):
                    #print("found more peaks than at first glance", peaks_neg_fin_rev, peaks_neg_fin_org)
                    peaks_neg_fin = peaks_neg_fin_rev
                else:
                    peaks_neg_fin = peaks_neg_fin_org
                num_col = len(peaks_neg_fin)
                #print(peaks_neg_fin,'peaks_neg_fin')
        except:
            logger.exception("cannot find peaks consistent with columns")
        #num_col, peaks_neg_fin = find_num_col(
        #    text_mask[top:bot,:],
        #    multiplier=7.0)
        peaks_neg_tot = np.array([0] + peaks_neg_fin + [width_tot - 1])
        #print(peaks_neg_tot,'peaks_neg_tot')
        peaks_neg_tot_tables.append(peaks_neg_tot)

        args_hor = matrix_new[:, 9] == 0
        x_min_hor_some = matrix_new[:, 2][args_hor]
        x_max_hor_some = matrix_new[:, 3][args_hor]
        y_min_hor_some = matrix_new[:, 6][args_hor]
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
        y_min_hor_some = np.append(y_min_hor_some, # toplines
                                   np.concatenate((y_min_hor_head - 2,
                                                   y_max_hor_head - 0)))
        y_max_hor_some = np.append(y_max_hor_some, # baselines
                                   np.concatenate((y_min_hor_head + 0,
                                                   y_max_hor_head + 2)))
        cy_hor_some = np.append(cy_hor_some, # centerlines
                                np.concatenate((y_min_hor_head - 1,
                                                y_max_hor_head + 1)))

        # analyse connected components of regions to gain additional separators
        # and prepare a map for cross-column boxes
        ccounts = np.bincount(ccomps[top: bot].flatten())
        ccounts_median = np.median(ccounts) if len(ccounts) else 0
        col_ccounts = np.stack([np.bincount(ccomps[top: bot, left: right].flatten(),
                                            minlength=ccounts.size)
                                for left, right in pairwise(peaks_neg_tot)])
        labelcolmap = dict()
        for label, label_count in enumerate(ccounts):
            if not label:
                continue
            # ignore small labels for the purpose of finding multicol seps
            if label_count < 0.5 * ccounts_median:
                continue
            label_left, label_top, label_width, label_height, label_area = cstats[label]
            label_right = label_left + label_width
            label_bot = label_top + label_height
            # enforce boundaries
            label_left = max(0, label_left)
            label_right = min(width_tot - 1, label_right)
            label_top = max(0, label_top)
            label_bot = min(height_tot - 1, label_bot)
            # disregard vertical portions of the cross-column label
            # if they are split by vertical separators already
            # (thus connected across columns only above or below
            #  those portions in a T- or H- or ⟂-shape):
            vsep_overlap_y, _ = vertical_seps[label_top: label_bot,
                                              label_left: label_right].nonzero()
            if len(vsep_overlap_y):
                # keep only largest portion not overlapped by any vseps
                non_overlap_y = np.setdiff1d(np.arange(label_top, label_bot),
                                             vsep_overlap_y + label_top)
                if not len(non_overlap_y):
                    # print("not keeping ccomp %d completely along vseps" % label)
                    continue
                # print("reducing ccomp %d vertically from %d:%d to %d:%d due to vseps" %
                #       (label, label_top, label_bot, non_overlap_y.min(), non_overlap_y.max()))
                label_top = non_overlap_y.min()
                label_bot = non_overlap_y.max()
            # if label_count < 0.9 * label_area:
            #     # mostly not in this part of the page
            #     continue
            if label_count < 0.01 * (top - bot) * width_tot:
                continue
            #assert np.sum(col_ccounts[:, label]) == label_count
            label_start = np.flatnonzero(peaks_neg_tot > label_left)[0] - 1
            label_end = np.flatnonzero(peaks_neg_tot >= label_right)[0]
            if label_end - label_start < 2:
                continue
            if np.count_nonzero(col_ccounts[:, label] > 0.1 * label_count) < 2:
                continue
            # ignore tiny passages with less 10% of height
            # (i.e. label is already almost separated by columns)
            for start in range(label_start, label_end):
                if text_mask[label_top: label_bot,
                             peaks_neg_tot[start + 1]].sum() < 0.1 * (label_bot - label_top):
                    label_start = start + 1
                else:
                    break
            for end in reversed(range(label_start, label_end)):
                if text_mask[label_top: label_bot,
                             peaks_neg_tot[end]].sum() < 0.1 * (label_bot - label_top):
                    label_end = end
                else:
                    break
            if label_end - label_start < 2:
                continue
            # store as dict for multi-column boxes:
            for start in range(label_start, label_end):
                labelcolmap.setdefault(start, list()).append(
                    (label_end, label_top, label_bot, sum(col_ccounts[start: label_end, label])))
            # make additional separators:
            x_min_hor_some = np.append(x_min_hor_some, [label_left + 25] * 2)
            x_max_hor_some = np.append(x_max_hor_some, [label_right - 25] * 2)
            y_min_hor_some = np.append(y_min_hor_some, [label_top - 2, label_bot])
            y_max_hor_some = np.append(y_max_hor_some, [label_top, label_bot + 2])
            cy_hor_some = np.append(cy_hor_some, [label_top - 1, label_bot + 1])

        # ensure no seps are out of bounds
        x_min_hor_some = np.maximum(0, np.minimum(width_tot, x_min_hor_some))
        x_max_hor_some = np.maximum(0, np.minimum(width_tot, x_max_hor_some))
        y_min_hor_some = np.maximum(0, np.minimum(height_tot, y_min_hor_some))
        y_max_hor_some = np.maximum(0, np.minimum(height_tot, y_max_hor_some))
        cy_hor_some = np.maximum(0, np.minimum(height_tot, cy_hor_some))

        if right2left_readingorder:
            x_max_hor_some = width_tot - x_min_hor_some
            x_min_hor_some = width_tot - x_max_hor_some

        x_starting, x_ending, y_min, y_mid, y_max = return_multicol_separators_x_start_end(
            text_mask, peaks_neg_tot, top, bot,
            x_min_hor_some, x_max_hor_some, cy_hor_some, y_min_hor_some, y_max_hor_some)
        # dbg_plt([0, None, top, bot], "non-empty multi-column separators in current split", 
        #         list(zip(peaks_neg_tot[x_starting], peaks_neg_tot[x_ending],
        #                  y_min - top, y_max - top)), True)

        # core algorithm:
        # 1. iterate through multi-column separators, pre-ordered by their y coord
        # 2. for each separator, iterate from its starting to its ending column
        # 3. in each starting column, determine the next downwards separator,
        # 4. if there is none, then fill up the column to the bottom;
        #    otherwise, fill up to that next separator
        # 5. moreover, determine the next rightward column that would not cut through
        #     any regions, advancing to that column, and storing a new in-order bbox
        #     for that down/right span
        # 6. if there was a next separator, and it ends no further than the current one,
        #    then recurse on that separator from step 1, then continue (with the next
        #    column for the current separator) at step 2, or (with the next separator
        #    in order) at step 1
        args = list(range(len(y_mid)))
        while len(args):
            cur = args[0]
            args = args[1:]
            # print("iter", cur, y_mid[cur], "%d:%d" % (x_starting[cur], x_ending[cur]))
            # ruff: disable[B023] (rs: seems like a false positive)
            def get_span(start, y_top, y_bot):
                # for last, l_top, l_bot, l_count in labelcolmap.get(start, []):
                #     if y_top < l_bot and y_bot > l_top and last > start + 1:
                #         width = (peaks_neg_tot[last] - peaks_neg_tot[start])
                #         print("span", start, last, l_top, l_bot, l_count,
                #               "box area", (y_bot - y_top) * width,
                #               "label area", (min(y_bot, l_bot) - max(y_top, l_top)) * width,
                #               "box height", (y_bot - y_top),
                #               "label height", sum(text_mask[
                #                   y_top: y_bot, peaks_neg_tot[start + 1]]))
                return max((last for last, l_top, l_bot, l_count in labelcolmap.get(start, [])
                            # yield the right-most column that does not cut through
                            # any regions in this horizontal span
                            if y_top < l_bot and y_bot > l_top
                            # Ignore if it ends here, anyway
                            and last > start + 1
                            # Ensure this is not just a tiny region near larger regions
                            and l_count > 0.1 * max(l_count2 for _, l_top2, l_bot2, l_count2 in labelcolmap[start]
                                                    if y_top < l_bot2 and y_bot > l_top2)
                            # or just a small cut of the respective region
                            # (i.e. box should cover at least 10% of the label).
                            and ((min(y_bot, l_bot) - max(y_top, l_top)) *
                                 (peaks_neg_tot[last] - peaks_neg_tot[start])) > 0.1 * l_count),
                           # Otherwise advance only 1 column.
                           default=start + 1)
            def add_sep(cur):
                column = x_starting[cur]
                while column < x_ending[cur]:
                    nxt = np.flatnonzero((y_mid[cur] < y_mid) &
                                         (column >= x_starting) &
                                         (column < x_ending))
                    if len(nxt):
                        nxt = nxt[0]
                        # print("column", column)
                        last = get_span(column, y_max[cur], y_min[nxt])
                        last = min(last, x_ending[nxt], x_ending[cur])
                        # print("nxt", nxt, y_mid[nxt], "%d:%d" % (column, last))
                        boxes.append([peaks_neg_tot[column],
                                      peaks_neg_tot[last],
                                      y_mid[cur],
                                      y_mid[nxt]])
                        # dbg_plt(boxes[-1], "recursive column %d:%d box [%d]" % (column, last, len(boxes)))
                        column = last
                        if (last == x_ending[nxt] and
                            x_ending[nxt] <= x_ending[cur] and
                            x_starting[nxt] >= x_starting[cur] and
                            nxt in args):
                            # child – recur
                            # print("recur", nxt, y_mid[nxt], "%d:%d" % (x_starting[nxt], x_ending[nxt]))
                            args.remove(nxt)
                            add_sep(nxt)
                    else:
                        # print("column", column)
                        last = get_span(column, y_max[cur], bot)
                        # print("bot", bot, "%d:%d" % (column, last))
                        boxes.append([peaks_neg_tot[column],
                                      peaks_neg_tot[last],
                                      y_mid[cur],
                                      bot])
                        # dbg_plt(boxes[-1], "non-recursive column %d box [%d]" % (column, len(boxes)))
                        column = last
            # ruff: enable[B023]
            add_sep(cur)

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

def ensure_array(obj: Iterable) -> np.ndarray:
    """convert sequence to array of type `object` so items can be of heterogeneous shape
    (but ensure not to convert inner arrays to `object` if len=1)
    """
    if not isinstance(obj, np.ndarray):
        return np.fromiter(obj, object)
    return obj

def seg_mask_label(segmap:np.ndarray,
                   mask:np.ndarray,
                   only:bool=False,
                   label:int=2,
                   skeletonize:bool=False,
                   dilate:int=0,
                   keep:int=0,
) -> None:
    """
    overwrite an existing segmentation map from a binary mask with a given label

    Args:
        segmap: integer array of existing segmentation labels ([H, W] or [B, H, W] shape)
        mask: boolean array for specific label
    Keyword Args:
        label: the class label to be written
        only: whether to suppress the `label` outside `mask`
        skeletonize: whether to transform the mask to its skeleton
        dilate: whether to also apply dilatation after this (convolution with square kernel of given size)
        keep: if nonzero, a clas label to be kept untouched

    Use this to enforce specific confidence thresholds or rules after segmentation.
    """
    if not mask.any():
        return
    # plt.subplot(2, 2, 1, title="segmap orig")
    # plt.imshow(segmap)
    # plt.subplot(2, 2, 2, title="mask")
    # plt.imshow(mask)
    if keep:
        keepmask = segmap == keep
    if only:
        segmap[segmap == label] = 0
    if skeletonize:
        if mask.ndim == 3:
            mask = np.stack(morphology.skeletonize(m) for m in mask)
        else:
            mask = morphology.skeletonize(mask)
        if dilate:
            kernel = np.ones((dilate, dilate), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
    segmap[mask] = label
    # plt.subplot(2, 2, 3, title="segmap masked")
    # plt.imshow(segmap)
    if keep:
        segmap[keepmask] = keep
    # plt.subplot(2, 2, 4, title="segmap final")
    # plt.imshow(segmap)
    # plt.show()
