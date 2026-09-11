from logging import getLogger

import numpy as np
import cv2
from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate as rotate_polygon
from shapely.ops import split as split_polygon
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from .contour import (
    contour2polygon,
    polygon2contour,
    return_contours_of_class,
)
from .resize import resize_image
from .rotate import rotate_image

# from matplotlib import pyplot as plt
# from shapely.plotting import plot_polygon

MIN_THICKNESS_MAINTEXT_PRCT_HEIGHT = 14
"""minimum vertical extent of the text mask (in percent of image height)

(pages with smaller share of text will not get marginalia)"""

MIN_DIST_GAPS = 40 # 20
"""minimum horizontal distance between neighbouring gaps to be candidates for margin points

(Gaps in the text mask projection are primary candidates.)"""

MAX_THICKNESS_GAPS_FRACT_MAIN = 17. # 20. # 30.
"""maximum vertical height of gaps (relative to the maximum height) to be candidates for margin points

(Gaps in the text mask projection are primary candidates.)"""

MIN_THICKNESS_STEP_FRACT_MAIN = 0.05
"""minimum vertical height of margin plateaus (relative to the main average) to be candidates for margin points

(Steps in the text mask projection are fallback candidates.)"""

MAX_THICKNESS_STEP_FRACT_MAIN = 0.5
"""maximum vertical height of margin plateaus (relative to the main average) to be candidates for margin points

(Steps in the text mask projection are fallback candidates.)"""


def get_marginals(num_col, slope_deskew,
                  early_layout,
                  textline_mask,
                  allow_l=True,
                  allow_r=True,
                  kernel=None,
                  logger=None,
                  label_text=1,
                  label_seps=3,
                  label_marg=4,
                  label_tabs=10,
):
    """
    Detect left and right margins, apply to given segmentation labelling.

    Find horizontal gaps in the (deskewed) text mask (as minima in its
    vertical projection curve).

    If there are multiple candidates on one side, then pick the one
    with highest prominence (drop towards neighbour) and deepest gap
    (lowest minimum).

    If there are no candidates on one side, then try finding steps
    in the text mask (as jumps in its vertical projection curve).
    Pick the steepest jumps that introduce a suitable plateau.
    (But suppress jumps merely caused by left / right indentation.)

    Constrain the ranges of left and right points depending on the
    given number of columns:
    1. for one-column pages, the left / right positions can be between
       the first / last occurrance of text and the middle of both
    2. for two-column pages, similarly, but also one third towards
       the left / right from the middle.

    (Even one-column pages are allowed to have marginalia on both sides.)

    Use the retrieved points on the left and right side to define margins,
    respectively. Rotate the margins and main masks back to the original
    (skewed) text mask.

    Then extract all contours in the text mask of the region segmentation
    and of the line segmentation. For each contour, if it falls completely
    within either margin, then mark it for marginalia. If it intersects
    either margin, then split it: For each resulting section, if it falls
    completely in the margin, then mark it. But ignore any parts which are
    mostly vertical, when the contour itself was not (to counter imprecision
    of the straight cut).

    Finally, fill all marked contours and partial contours with the marginalia
    label.

    Also, for the text mask of the line segmentation, cut along the left / right
    margin lines (by drawing a line with the background label), to ensure that
    text lines will also be assigned separately.

    Return without changes if
    - there is not enough text (i.e. the maximum of the vertical projection
      of the text mask is too small)
    - there are no sufficient peaks
    - there would be no remaining main text
    """
    if logger is None:
        logger = getLogger(__package__)
    kernel = np.ones((2, 2), dtype=np.uint8)
    kernel_hor = np.ones((1, 2), dtype=np.uint8)

    text_mask = ((early_layout == label_text) |
                 (early_layout == label_tabs)).astype(np.uint8)
    seps_mask = (early_layout == label_seps).astype(np.uint8)
    text_mask_d = rotate_image(text_mask, slope_deskew)
    seps_mask_d = rotate_image(seps_mask, slope_deskew)
    height, width = text_mask_d.shape

    # plt.figure("text mask")
    # plt.subplot(1, 3, 1, title="original text mask")
    # plt.imshow(text_mask_d)
    if height <= 1500:
        pass
    elif 1500 < height <= 1800:
        text_mask_d = resize_image(text_mask_d, int(height / 1.5), width)
        text_mask_d = cv2.erode(text_mask_d, kernel, iterations=3)
        # rs: and back to original size
        text_mask_d = resize_image(text_mask_d, height, width)
    else:
        text_mask_d = resize_image(text_mask_d, int(height / 1.8), width)
        text_mask_d = cv2.erode(text_mask_d, kernel, iterations=5)
        # rs: and back to original size
        text_mask_d = resize_image(text_mask_d, height, width)
    # plt.subplot(1, 3, 2, title="eroded text mask")
    # plt.imshow(text_mask_d)

    text_mask_d = cv2.erode(text_mask_d, kernel_hor, iterations=4)
    # plt.subplot(1, 3, 3, title="horizontally eroded")
    # plt.imshow(text_mask_d)
    # plt.show()
    text_mask_d_y = text_mask_d.sum(axis=0) # len = width
    #text_mask_d_x = text_mask_d.sum(axis=1) # len = height

    max_text_thickness = text_mask_d_y.max()
    max_text_thickness_percent = 100. * max_text_thickness / height
    min_text_thickness = max_text_thickness / MAX_THICKNESS_GAPS_FRACT_MAIN

    if max_text_thickness_percent < MIN_THICKNESS_MAINTEXT_PRCT_HEIGHT:
        # plt.figure("not enough text")
        # ax1 = plt.subplot(2, 2, 1, title="text_mask_d")
        # ax1.imshow(text_mask_d, aspect='auto')
        # ax2 = plt.subplot(2, 2, 3, title="text_mask_d_y", sharex=ax1)
        # ax2.plot(list(range(width)), text_mask_d_y)
        # ax2.hlines(int(0.14 * height), 0, width,
        #            label='max_text_thickness=14%', colors='r')
        # ax2.hlines([min_text_thickness], 0, width,
        #            label='min_text_thickness', colors='g')
        # ax2.scatter([np.argmax(text_mask_d_y)],
        #             [text_mask_d_y.max()], color='r',
        #             label='max = %d%%' % max_text_thickness_percent)
        # ax2.legend()
        # ax1 = plt.subplot(2, 2, 4, title="early layout")
        # ax1.imshow(early_layout, aspect='auto')
        # plt.legend()
        # plt.show()
        logger.debug("marginalia: insufficient max vertical thickness (%d%%)",
                     max_text_thickness_percent)
        return

    text_mask_d_ys = gaussian_filter1d(text_mask_d_y, 1)
    #text_mask_d_xs = gaussian_filter1d(text_mask_d_x, 3)
    first_nonzero = text_mask_d_ys.nonzero()[0][0] # outer left
    last_nonzero = text_mask_d_ys.nonzero()[0][-1] # outer right
    mid_point = (last_nonzero + first_nonzero) // 2
    mid_point_l = mid_point - (mid_point - first_nonzero) // 3
    mid_point_r = mid_point + (last_nonzero - mid_point) // 3
    
    # find minima (horizontal gaps)
    gaps_y, props_y = find_peaks(
        max_text_thickness - text_mask_d_ys,
        # constrain thickness lower bound (min_text_thickness)
        #height=0.5 * max_text_thickness,
        height=max_text_thickness - min_text_thickness,
        # constrain the prominence towards neighbours
        prominence=0.5 * min_text_thickness,
        # constrain the distance at least 2 characters at 12pt
        distance=MIN_DIST_GAPS)

    # plt.figure("gaps")
    # ax1 = plt.subplot(2, 2, 1, title="text_mask_d")
    # ax1.imshow(text_mask_d, aspect='auto')
    # ax3 = plt.subplot(2, 2, 2, title="text_mask_d_x", sharey=ax1)
    # ax2 = plt.subplot(2, 2, 3, title="text_mask_d_y", sharex=ax1)
    # ax2.plot(list(range(width)), text_mask_d_y, label='unsmoothed', color='b')
    # ax2.plot(list(range(width)), text_mask_d_ys, label='smoothed', color='m')
    # ax2.hlines(int(0.14 * height), 0, width,
    #            label='max_text_thickness=14%', colors='r')
    # ax2.hlines([min_text_thickness], 0, width,
    #            label='min_text_thickness', colors='m')
    # ax2.vlines([first_nonzero], 0, height, label='first_nonzero', colors='r')
    # ax2.vlines([last_nonzero], 0, height, label='last_nonzero', colors='r')
    # if num_col == 2:
    #     ax2.vlines([mid_point], 0, height, label='mid_point', colors='r')
    # ax2.scatter([np.argmax(text_mask_d_y)],
    #             [text_mask_d_y.max()], color='r',
    #             label='max = %d%%' % max_text_thickness_percent)
    # ax2.scatter(gaps_y, text_mask_d_ys[gaps_y], label='gaps_y', color='m')
    # for i in range(len(peaks_y)):
    #     ax2.text(gaps_y[i], text_mask_d_ys[gaps_y[i]], str(props_y['prominences'][i]))
    # ax1 = plt.subplot(2, 2, 4, title="early layout")
    # ax1.imshow(early_layout, aspect='auto')
    # plt.legend()

    # also calculate the product of prominence and height (for final selection)
    scores = np.zeros(gaps_y.max(initial=width) + 1)
    scores[gaps_y] = props_y['prominences'] * props_y['peak_heights']

    gaps_y = gaps_y[(gaps_y > first_nonzero) & (gaps_y < last_nonzero)]

    if num_col == 1:
        gaps_r = gaps_y[gaps_y > mid_point]
        gaps_l = gaps_y[gaps_y < mid_point]
    elif num_col == 2:
        gaps_r = gaps_y[gaps_y > mid_point_r]
        gaps_l = gaps_y[gaps_y < mid_point_l]
    else:
        # should not happen, anyway
        return

    if len(gaps_r) and allow_r:
        best_r = np.argmax(scores[gaps_r])
        point_r = gaps_r[best_r]
    else:
        point_r = width - 1
        #point_r = last_nonzero
    if len(gaps_l) and allow_l:
        best_l = np.argmax(scores[gaps_l])
        point_l = gaps_l[best_l]
    else:
        point_l = 0
        #point_l = first_nonzero
    if len(gaps_r) and len(gaps_l):
        if scores[best_l] < 0.1 * scores[best_r]:
            point_l = 0
            #point_l = first_nonzero
        if scores[best_r] < 0.1 * scores[best_l]:
            point_r = 0
            #point_r = last_nonzero

    # if there are no valid peaks (i.e. gaps) on either side,
    # because marginalia are very close to main,
    # then look at positive/negative peaks of derivative on left/right side,
    # to identify smaller plateaus of sufficient thickness:
    if len(gaps_l) == 0 and allow_l:
        logger.debug("marginalia: trying to find jumps instead of gaps on the left")
        text_mask_d_ys2 = np.diff(text_mask_d_ys[:mid_point+1].astype(int))
        jumps, _ = find_peaks(text_mask_d_ys2,
                              # constrain slope lower bound
                              height=0.04 * max_text_thickness,
                              # constrain the distance at least 2 characters at 12pt
                              distance=MIN_DIST_GAPS)
        # ax2.plot(list(range(mid_point)), text_mask_d_ys2, label='derivative', color='g')
        # ax2.scatter(jumps, text_mask_d_ys2[jumps], label='jumps', color='g')
        # ax2.hlines([0.04 * max_text_thickness], 0, width,
        #            label='min2', colors='lightgreen')
        if num_col == 1:
            jumps_l = jumps[jumps < mid_point]
        else:
            jumps_l = jumps[jumps < mid_point_l]
        # suppress these if there are horizontal separators spanning across
        seps_cont_l, _ = cv2.findContours(seps_mask_d[:, :mid_point_l],
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
        seps_x_min_l = np.array([contour[:, 0, 0].min() + 15 for contour in seps_cont_l])
        seps_x_max_l = np.array([contour[:, 0, 0].max() - 15 for contour in seps_cont_l])
        jumps_l = jumps_l[~((seps_x_min_l[:, np.newaxis] <= jumps_l[np.newaxis]) &
                            (seps_x_max_l[:, np.newaxis] > jumps_l[np.newaxis])).any(axis=0)]
        # to discern between left marginalia and left indentation,
        # analyse horizontal projection of (left half) of text mask:
        # if there are subpeaks left of (=above) the main peaks (=paragraphs)
        # and the average difference between them is approximately the jump point
        # then that jump point cannot be from genuine marginalia
        text_mask_d_lx = text_mask_d[:, :mid_point_l].sum(axis=1)
        text_mask_d_lxs = gaussian_filter1d(text_mask_d_lx, 3)
        text_mask_d_lxs2 = np.diff(text_mask_d_lxs.astype(int))
        peaks_lx, props_lx = find_peaks(text_mask_d_lx,
                                        prominence=MIN_DIST_GAPS)
        bases_lx = props_lx['left_bases']
        proms_lx = text_mask_d_lx[peaks_lx] - text_mask_d_lx[bases_lx]
        med_main_width_l = np.median(text_mask_d_lx[peaks_lx])
        # ax3.plot(text_mask_d_lx, list(range(height)), label='unsmoothed', color='b')
        # ax3.plot(text_mask_d_lxs, list(range(height)), label='smoothed', color='m')
        # ax3.scatter(text_mask_d_lx[peaks_lx], peaks_lx, label='peaks_lx', color='m')
        # ax3.scatter(text_mask_d_lx[props_lx['left_bases']], props_lx['left_bases'], label='bases_l', color='g')
        # for i in range(len(peaks_lx)):
        #     ax3.text(text_mask_d_lx[peaks_lx[i]], peaks_lx[i], str(props_lx['prominences_l'][i]))
        # ax3.vlines([med_main_width_l], 0, height, colors='m')
        if np.isclose(med_main_width_l,
                      mid_point_l - first_nonzero,
                      rtol=0.1):
            flexpoints_lx, _ = find_peaks(np.abs(text_mask_d_lxs2 < 3))
            #ax3.scatter(text_mask_d_lx[flexpoints_lx], flexpoints_lx, label="flexpoints_lx", color='y')
            subpeaks_lx = flexpoints_lx[
                ~np.isclose(text_mask_d_lx[flexpoints_lx], med_main_width_l, rtol=0.05) &
                ~np.isclose(text_mask_d_lx[flexpoints_lx], 0, atol=20)]
            # consider only those subpeaks left of peaks:
            subpeaks_lx = subpeaks_lx[np.searchsorted(bases_lx, subpeaks_lx, 'right') ==
                                      np.searchsorted(peaks_lx, subpeaks_lx) + 1]
            if len(subpeaks_lx):
                med_preceding_width_l = np.median(text_mask_d_lx[subpeaks_lx])
                indent_l = np.isclose(jumps_l - first_nonzero,
                                      med_main_width_l - med_preceding_width_l,
                                      rtol=0.1, atol=10)
                logger.debug("%d out of %d step candidates for left margin are from indentation",
                             np.count_nonzero(indent_l), len(jumps_l))
                jumps_l = jumps_l[~indent_l]
                # ax3.scatter(text_mask_d_lx[subpeaks_lx], subpeaks_lx, label="subpeaks_lx", color='y')
                # ax3.vlines([med_preceding_width_l], 0, height, colors='b')
        # search from right (widest) to left
        for jump in reversed(jumps_l):
            if jump < first_nonzero + MIN_DIST_GAPS:
                continue
            # non-zero, non-max plateau
            if (MIN_THICKNESS_STEP_FRACT_MAIN
                < (np.mean(text_mask_d_ys[first_nonzero: jump]) / 
                   np.mean(text_mask_d_ys[jump: mid_point])) <
                MAX_THICKNESS_STEP_FRACT_MAIN):
                point_l = jump
                # ax2.vlines([jump], 0, height, label='jump', colors='y')
                break
            # TODO: try another criterion: deviation in average textline heights
    if len(gaps_r) == 0 and allow_r:
        logger.debug("marginalia: trying to find jumps instead of gaps on the right")
        text_mask_d_ys2 = -np.diff(text_mask_d_ys[mid_point-1:].astype(int))
        jumps, _ = find_peaks(text_mask_d_ys2,
                              # constrain slope lower bound
                              height=0.04 * max_text_thickness,
                              # constrain the distance at least 2 characters at 12pt
                              distance=MIN_DIST_GAPS)
        # ax2.plot(np.arange(mid_point, width), text_mask_d_ys2, label='derivative', color='g')
        # ax2.scatter(jumps + mid_point, text_mask_d_ys2[jumps], label='jumps', color='g')
        # ax2.hlines([0.04 * max_text_thickness], 0, width,
        #            label='min2', colors='lightgreen')
        jumps += mid_point
        if num_col == 1:
            jumps_r = jumps[jumps > mid_point]
        else:
            jumps_r = jumps[jumps > mid_point_r]
        # suppress these if there are horizontal separators spanning across
        seps_cont_r, _ = cv2.findContours(seps_mask_d[:, mid_point_r:],
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
        seps_x_min_r = np.array([contour[:, 0, 0].min() + 15 + mid_point_r for contour in seps_cont_r])
        seps_x_max_r = np.array([contour[:, 0, 0].max() - 15 + mid_point_r for contour in seps_cont_r])
        jumps_r = jumps_r[~((seps_x_min_r[:, np.newaxis] <= jumps_r[np.newaxis]) &
                            (seps_x_max_r[:, np.newaxis] > jumps_r[np.newaxis])).any(axis=0)]
        # to discern between right marginalia and right indentation
        # (i.e. unjustified final lines of a paragraph),
        # analyse horizontal projection of (right half) of text mask:
        # if there are subpeaks right of (=below) the main peaks (=paragraphs)
        # and the average difference between them is approximately the jump point
        # then that jump point cannot be from genuine marginalia
        text_mask_d_rx = text_mask_d[:, mid_point_r:].sum(axis=1)
        text_mask_d_rxs = gaussian_filter1d(text_mask_d_rx, 3)
        text_mask_d_rxs2 = np.diff(text_mask_d_rxs.astype(int))
        peaks_rx, props_rx = find_peaks(text_mask_d_rx,
                                        prominence=MIN_DIST_GAPS)
        bases_rx = props_rx['right_bases']
        proms_rx = text_mask_d_rx[peaks_rx] - text_mask_d_rx[bases_rx]
        med_main_width_r = np.median(text_mask_d_rx[peaks_rx])
        # ax3.plot(text_mask_d_rx + mid_point_r, list(range(height)), label='unsmoothed', color='b')
        # ax3.plot(text_mask_d_rxs + mid_point_r, list(range(height)), label='smoothed', color='m')
        # ax3.scatter(text_mask_d_rx[peaks_rx] + mid_point_r, peaks_rx, label='peaks_rx', color='m')
        # ax3.scatter(text_mask_d_rx[props_rx['left_bases']] + mid_point_r, props_rx['left_bases'], label='bases_r', color='g')
        # for i in range(len(peaks_rx)):
        #     ax3.text(text_mask_d_rx[peaks_rx[i]] + mid_point_r, peaks_rx[i], str(props_rx['prominences_r'][i]))
        # ax3.vlines([med_main_width_r + mid_point_r], 0, height, colors='m')
        if np.isclose(med_main_width_r,
                      last_nonzero - mid_point_r,
                      rtol=0.1):
            flexpoints_rx, _ = find_peaks(np.abs(text_mask_d_rxs2 < 3))
            #ax3.scatter(text_mask_d_rx[flexpoints_rx] + mid_point_r, flexpoints_rx, label="flexpoints_rx", color='y')
            subpeaks_rx = flexpoints_rx[
                ~np.isclose(text_mask_d_rx[flexpoints_rx], med_main_width_r, rtol=0.05) &
                ~np.isclose(text_mask_d_rx[flexpoints_rx], 0, atol=20)]
            # consider only those subpeaks right of peaks:
            subpeaks_rx = subpeaks_rx[np.searchsorted(peaks_rx, subpeaks_rx, 'right') ==
                                      np.searchsorted(bases_rx, subpeaks_rx) + 1]
            if len(subpeaks_rx):
                med_following_width_r = np.median(text_mask_d_rx[subpeaks_rx])
                indent_r = np.isclose(last_nonzero - jumps_r,
                                      med_main_width_r - med_following_width_r,
                                      rtol=0.1, atol=10)
                logger.debug("%d out of %d step candidates for right margin are from indentation",
                             np.count_nonzero(indent_r), len(jumps_r))
                jumps_r = jumps_r[~indent_r]
                # ax3.scatter(text_mask_d_rx[subpeaks_rx] + mid_point_r, subpeaks_rx, label="subpeaks_rx", color='y')
                # ax3.vlines([med_following_width_r + mid_point_r], 0, height, colors='b')
        # search from left (widest) to right
        for jump in jumps_r:
            if jump > last_nonzero - MIN_DIST_GAPS:
                continue
            # non-zero, non-max plateau
            if (MIN_THICKNESS_STEP_FRACT_MAIN
                < (np.mean(text_mask_d_ys[jump: last_nonzero]) /
                   np.mean(text_mask_d_ys[mid_point: jump])) <
                MAX_THICKNESS_STEP_FRACT_MAIN):
                point_r = jump
                # ax2.vlines([jump], 0, height, label='jump', colors='y')
                break
    # ax3.legend()
    # ax2.legend()
    # plt.show()

    # plt.figure("final points")
    # ax1 = plt.subplot(2, 2, 1)
    # ax1.title.set_text('text_mask_d (deskewed text+table mask)')
    # ax1.imshow(text_mask_d)
    # ax1.vlines([point_l], 0, height, label='point_l', colors='r')
    # ax1.vlines([point_r], 0, height, label='point_r', colors='r')
    # ax3 = plt.subplot(2, 2, 3, title='text_mask_d_y (projection for minima)', sharex=ax1)
    # ax3.plot(list(range(width)), text_mask_d_y)
    # ax3.set_aspect('auto')
    # ax3.vlines([point_l], 0, height, label='point_l', colors='r')
    # ax3.vlines([point_r], 0, height, label='point_r', colors='r')
    # ax4 = plt.subplot(2, 2, 4, title='early_layout (undeskewed labels)')
    # ax4.imshow(early_layout)
    # plt.legend()
    # plt.show()

    if point_l == 0 and point_r == width - 1:
        return

    line_l = LineString([(point_l, -0.4 * height), (point_l, 1.4 * height)])
    line_r = LineString([(point_r, -0.4 * height), (point_r, 1.4 * height)])
    # rotate back (into undeskewed/original shape as early_layout input):
    line_l = rotate_polygon(line_l, slope_deskew, origin=(0.5 * width, 0.5 * height))
    line_r = rotate_polygon(line_r, slope_deskew, origin=(0.5 * width, 0.5 * height))
    main = Polygon([(0, 0), (0, height), (width, height), (width, 0)])
    if point_l > 0:
        page_l = split_polygon(main, line_l)
        marg_l, main = page_l.geoms
    else:
        marg_l = Polygon()
    if point_r < width - 1:
        page_r = split_polygon(main, line_r)
        main, marg_r = page_r.geoms
    else:
        marg_r = Polygon()

    # plt.imshow(early_layout)
    # plot_polygon(marg_l, color='yellow')
    # plot_polygon(marg_r, color='red')
    # plot_polygon(main, color='magenta')
    # plt.show()
    marg_l3 = marg_l.buffer(3)
    marg_r3 = marg_r.buffer(3)
    line_l3 = line_l.offset_curve(5)
    line_r3 = line_r.offset_curve(-5)

    # re-assign (marg/main), cutting through existing contours if necessary
    min_area_text = 0.00001
    text_contours = return_contours_of_class(early_layout, label_text, min_area_text)
    marg_contours = []
    for cont in text_contours:
        poly = contour2polygon(cont)
        if not poly.area:
            continue # fixme: warn
        # plt.imshow(early_layout)
        minx, miny, maxx, maxy = poly.bounds
        aspect = (maxy - miny) / (maxx - minx)
        if poly.within(marg_l3) or poly.within(marg_r3):
            marg_contours.append(cont)
        elif poly.within(main):
            continue
        elif poly.intersects(marg_l) or poly.intersects(marg_r):
            # partial: split, but ignore marg parts overly vertical
            # plot_polygon(main, color='magenta')
            inter = poly.intersection(marg_l)
            iminx, iminy, imaxx, imaxy = inter.bounds
            iaspect = (imaxy - iminy) / (imaxx - iminx)
            # ignore if intersection is almost vertical
            if not (iaspect > 4 and iaspect > 8 * aspect):
                for geom in split_polygon(poly, line_l3).geoms:
                    gminx, gminy, gmaxx, gmaxy = geom.bounds
                    gaspect = (gmaxy - gminy) / (gmaxx - gminx)
                    if gaspect > 4 and gaspect > 8 * aspect:
                        continue
                    if not geom.within(marg_l3):
                        continue
                    # plot_polygon(geom, color='blue')
                    marg_contours.append(polygon2contour(geom))
            inter = poly.intersection(marg_r)
            iminx, iminy, imaxx, imaxy = inter.bounds
            iaspect = (imaxy - iminy) / (imaxx - iminx)
            # ignore if intersection is almost vertical
            if not (iaspect > 4 and iaspect > 8 * aspect):
                for geom in split_polygon(poly, line_r3).geoms:
                    gminx, gminy, gmaxx, gmaxy = geom.bounds
                    gaspect = (gmaxy - gminy) / (gmaxx - gminx)
                    if gaspect > 4 and gaspect > 8 * aspect:
                        continue
                    if not geom.within(marg_r3):
                        continue
                    # plot_polygon(geom, color='green')
                    marg_contours.append(polygon2contour(geom))
            # plt.show()
        else:
            # plot_polygon(main, color='magenta')
            # plot_polygon(poly, color='red')
            # plt.show()
            assert False

    # write marginals to segmentation
    early_layout = cv2.fillPoly(early_layout, pts=marg_contours, color=label_marg)

    # also cut textlines
    # plt.subplot(1, 2, 1, title='textline mask')
    # plt.imshow(textline_mask)
    line_l = np.stack(line_l.xy).astype(int)
    line_r = np.stack(line_r.xy).astype(int)
    textline_mask = cv2.line(textline_mask, line_l[:, 0], line_l[:, 1], 0, 1)
    textline_mask = cv2.line(textline_mask, line_r[:, 0], line_r[:, 1], 0, 1)
    # plt.subplot(1, 2, 2, title='textline mask (split)')
    # plt.imshow(textline_mask)
    # plt.show()

    # if there was no main text, then relabel marginalia as main
    if not np.any(early_layout == label_text):
        early_layout[early_layout == label_marg] = label_text
