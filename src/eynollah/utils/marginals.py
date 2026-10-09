from logging import getLogger

import numpy as np
import cv2
from shapely.geometry import Polygon, LineString
from shapely.affinity import rotate as rotate_polygon
from shapely.ops import split as split_polygon
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from . import pairwise
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
                  label_imgs=2,
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
    seps_mask = ((early_layout == label_seps) |
                 (early_layout == label_imgs)).astype(np.uint8)
    text_mask_d = rotate_image(text_mask, slope_deskew)
    seps_mask_d = rotate_image(seps_mask, slope_deskew)
    height, width = text_mask_d.shape

    # plt.figure("text mask")
    # plt.subplot(1, 3, 1, title="original text mask")
    # plt.imshow(text_mask_d)
    if height <= 1500:
        text_mask_de = text_mask_d
    elif 1500 < height <= 1800:
        text_mask_de = resize_image(text_mask_d, int(height / 1.5), width)
        text_mask_de = cv2.erode(text_mask_de, kernel, iterations=3)
        # rs: and back to original size
        text_mask_de = resize_image(text_mask_de, height, width)
    else:
        text_mask_de = resize_image(text_mask_d, int(height / 1.8), width)
        text_mask_de = cv2.erode(text_mask_de, kernel, iterations=5)
        # rs: and back to original size
        text_mask_de = resize_image(text_mask_de, height, width)
    # plt.subplot(1, 3, 2, title="eroded text mask")
    # plt.imshow(text_mask_d)

    text_mask_de = cv2.erode(text_mask_de, kernel_hor, iterations=4)
    # plt.subplot(1, 3, 3, title="horizontally eroded")
    # plt.imshow(text_mask_d)
    # plt.show()
    text_mask_d_y = text_mask_de.sum(axis=0) # len = width
    #text_mask_d_x = text_mask_de.sum(axis=1) # len = height

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
    qrt_point_l = mid_point - (mid_point - first_nonzero) // 2
    qrt_point_r = mid_point + (last_nonzero - mid_point) // 2
    
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

    # def extend_ticks(axis, locations, labels, minor=True, **labelargs):
    #     locs = list(axis.get_ticklocs(minor=minor))
    #     labs = list(axis.get_ticklabels(minor=minor))
    #     locs += locations
    #     labs += labels
    #     axis.set_ticks(locs, labels=labs, minor=minor, **labelargs)
    # plt.figure("gaps")
    # ax1 = plt.subplot(2, 2, 1, title="text_mask_d")
    # ax1.imshow(text_mask_d, aspect='auto')
    # ax3 = plt.subplot(2, 2, 2, title="text_mask_d_x", sharey=ax1)
    # ax2 = plt.subplot(2, 2, 3, title="text_mask_d_y", sharex=ax1)
    # ax2.plot(list(range(width)), text_mask_d_y, label='unsmoothed', color='b')
    # ax2.plot(list(range(width)), text_mask_d_ys, label='smoothed', color='m')
    # ax2.hlines(int(0.14 * height), 0, width, colors='r')
    # extend_ticks(ax2.yaxis, [int(0.14 * height)], ['max=14%%'])
    # ax2.hlines([min_text_thickness], 0, width, colors='m')
    # extend_ticks(ax2.yaxis, [min_text_thickness], ['min'])
    # ax2.vlines([first_nonzero, last_nonzero], 0, height, colors='r')
    # extend_ticks(ax2.xaxis, [first_nonzero, last_nonzero], ['first', 'last'])
    # ax2.vlines([mid_point_l, qrt_point_l], 0, height, colors='b')
    # ax2.vlines([mid_point_r, qrt_point_r], 0, height, colors='b')
    # extend_ticks(ax2.xaxis, [mid_point_l, qrt_point_l], ['mid_l', 'qrt_l'])
    # if num_col == 2:
    #     ax2.vlines([mid_point], 0, height, label='mid_point', colors='r')
    # ax2.scatter([np.argmax(text_mask_d_y)],
    #             [text_mask_d_y.max()], color='r',
    #             label='max = %d%%' % max_text_thickness_percent)
    # ax2.scatter(gaps_y, text_mask_d_ys[gaps_y], label='gaps_y', color='m')
    # for i in range(len(gaps_y)):
    #     ax2.text(gaps_y[i], text_mask_d_ys[gaps_y[i]], str(props_y['prominences'][i]))
    # ax2.tick_params(top=True, labeltop=True, bottom=True, labelbottom=False,
    #                 left=True, labelleft=False, right=True, labelright=True,
    #                 which='minor')
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
    if len(gaps_r) and allow_r and len(gaps_l) and allow_l:
        if scores[best_l] < 0.1 * scores[best_r]:
            point_l = 0
            #point_l = first_nonzero
        if scores[best_r] < 0.1 * scores[best_l]:
            point_r = width - 1
            #point_r = last_nonzero

    # if there are no valid peaks (i.e. gaps) on either side,
    # because marginalia are very close to main,
    # then look at positive/negative peaks of derivative on left/right side,
    # to identify smaller plateaus of sufficient thickness:
    def detect_step_point(side):
        min_text_thickness = max_text_thickness / 25.
        logger.debug("marginalia: trying to find jumps instead of gaps on the %s", side)
        if side == 'left':
            text_mask_d_ys2 = np.diff(text_mask_d_ys[:mid_point+1].astype(int))
            offset = 0
        else:
            text_mask_d_ys2 = -np.diff(text_mask_d_ys[mid_point-1:].astype(int))
            offset = mid_point
        # ax2.hlines([0.04 * max_text_thickness], 0, width, colors='lightgreen')
        # extend_ticks(ax2.yaxis, [0.04 * max_text_thickness], ['min2'])
        # ax2.plot(np.arange(len(text_mask_d_ys2)) + offset, text_mask_d_ys2, label='derivative', color='g')
        jumps, _ = find_peaks(text_mask_d_ys2,
                              # constrain slope lower bound
                              height=min_text_thickness,
                              # constrain the distance at least 2 characters at 12pt
                              #distance=MIN_DIST_GAPS)
                              )
        if not len(jumps):
            return None
        # combine follow-up peaks if smaller MIN_DIST_GAPS
        combined = [jumps[0]]
        for jump in jumps[1:]:
            if (jump - combined[-1] < MIN_DIST_GAPS and
                text_mask_d_ys2[(new := (combined[-1] + jump) // 2)] >=
                min_text_thickness):
                combined[-1] = new
            else:
                combined.append(jump)
        jumps = np.array(combined)

        # ax2.scatter(jumps + offset, text_mask_d_ys2[jumps], label='jumps', color='g')
        jumps += offset
        if side == 'left':
            # ax2.vlines([mid_point_l, qrt_point_l], 0, height, colors='b')
            # extend_ticks(ax2.xaxis, [mid_point_l, qrt_point_l], ['mid_l', 'qrt_l'])
            jumps = jumps[(first_nonzero + MIN_DIST_GAPS < jumps) &
                          (jumps < (mid_point_l if num_col == 1 else qrt_point_l))]
        else:
            # ax2.vlines([mid_point_r, qrt_point_r], 0, height, colors='b')
            # extend_ticks(ax2.xaxis, [mid_point_r, qrt_point_r], ['mid_r', 'qrt_r'])
            jumps = jumps[(last_nonzero - MIN_DIST_GAPS > jumps) &
                          (jumps > (mid_point_r if num_col == 1 else qrt_point_r))]
        # suppress these if there are horizontal separators spanning across
        seps_cont, _ = cv2.findContours(seps_mask_d[:, :mid_point] if side == 'left' else
                                        seps_mask_d[:, mid_point:],
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        seps_x_min = np.array([contour[:, 0, 0].min() + 15
                               for contour in seps_cont]) + offset
        seps_x_max = np.array([contour[:, 0, 0].max() - 15
                               for contour in seps_cont]) + offset
        jumps = jumps[~((seps_x_min[:, np.newaxis] <= jumps[np.newaxis]) &
                        (seps_x_max[:, np.newaxis] > jumps[np.newaxis])).any(axis=0)]
        if not len(jumps):
            return None

        # to discern between left/right marginalia and left/right indentation,
        # (i.e. indented first lines / unjustified final lines of a paragraph),
        # analyse horizontal projection of (left/right half) of text mask:
        # if there are subpeaks beside (above or below) the main peaks (=paragraphs)
        # and the average difference between them is approximately the jump point
        # then that jump point cannot be from genuine marginalia
        text_mask_d_x = np.sum(text_mask_d[:, :qrt_point_l] if side == 'left' else
                               text_mask_d[:, qrt_point_r:], axis=1)
        text_mask_d_xs = gaussian_filter1d(text_mask_d_x, 2)
        text_mask_d_xs2 = np.diff(text_mask_d_xs.astype(int))
        text_mask_d_xs3 = np.diff(text_mask_d_xs2.astype(int))
        peaks_x, props_x = find_peaks(text_mask_d_x,
                                      distance=MIN_DIST_GAPS,
                                      prominence=MIN_DIST_GAPS)
        med_main_width = np.median(text_mask_d_x[peaks_x])
        offset_x = 0 if side == 'left' else qrt_point_r
        # ax3.plot(text_mask_d_x + offset_x, list(range(height)), label='unsmoothed', color='b')
        # ax3.plot(text_mask_d_xs + offset_x, list(range(height)), label='smoothed', color='m')
        # ax3.scatter(text_mask_d_x[peaks_x] + offset_x, peaks_x, label='peaks_x', color='m')
        # ax3.vlines([med_main_width + offset_x], 0, height, colors='m')
        if np.isclose(med_main_width,
                      qrt_point_l - first_nonzero if side == 'left' else
                      last_nonzero - qrt_point_r,
                      rtol=0.1):
            flexpoints_x, _ = find_peaks(np.abs(text_mask_d_xs2 < 3))
            flexpoints_x -= 1
            #ax3.scatter(text_mask_d_x[flexpoints_x] + offset_x, flexpoints_x, label="flexpoints_x", color='y')
            subpeaks_x = flexpoints_x[
                # no minima/saddles
                (text_mask_d_xs3[flexpoints_x] < 0) &
                # not near the main maxima
                ~np.isclose(text_mask_d_x[flexpoints_x], med_main_width, rtol=0.05) &
                # not too small
                ~np.isclose(text_mask_d_x[flexpoints_x], 0, atol=0.3 * med_main_width)]
            if len(subpeaks_x) > 0.5 * len(peaks_x):
                med_sub_width = np.median(text_mask_d_x[subpeaks_x])
                indent = np.isclose(jumps - first_nonzero if side == 'left' else
                                    last_nonzero - jumps,
                                    med_main_width - med_sub_width,
                                    rtol=0.1, atol=10)
                logger.debug("%d out of %d step candidates for %s margin are from indentation",
                             np.count_nonzero(indent), len(jumps), side)
                jumps = jumps[~indent]
                # ax3.scatter(text_mask_d_x[subpeaks_x] + offset_x, subpeaks_x, label="subpeaks_x", color='y')
                # ax3.vlines([med_sub_width + offset_x], 0, height, colors='b')
                if not len(jumps):
                    return None

        if side == 'left':
            text_mask_d_ys2_iszero = np.append(np.diff(np.isclose(text_mask_d_ys2, 0, atol=8) * 1), -1)
        else:
            text_mask_d_ys2_iszero = np.insert(np.diff(np.isclose(text_mask_d_ys2, 0, atol=8) * 1), 0, 1)
        text_mask_d_ys2_iszero_min = np.flatnonzero(text_mask_d_ys2_iszero > 0)
        text_mask_d_ys2_iszero_max = np.flatnonzero(text_mask_d_ys2_iszero < 0)
        if (not len(text_mask_d_ys2_iszero_min) or
            not len(text_mask_d_ys2_iszero_max)):
            return None

        if (text_mask_d_ys2_iszero_min[0] >
            text_mask_d_ys2_iszero_max[0]):
            text_mask_d_ys2_iszero_max = text_mask_d_ys2_iszero_max[1:]
        length = min(len(text_mask_d_ys2_iszero_min),
                     len(text_mask_d_ys2_iszero_max))
        if length <= 1:
            return None

        text_mask_d_ys_plateaus = np.stack((text_mask_d_ys2_iszero_min[:length],
                                            text_mask_d_ys2_iszero_max[:length]),
                                           axis=1)
        text_mask_d_ys_plateaus += offset
        indices = [0]
        for i, pair in enumerate(pairwise(text_mask_d_ys_plateaus), start=1):
            (prev_min, prev_max), (next_min, next_max) = pair
            if np.isclose(np.mean(text_mask_d_ys[prev_min: prev_max]),
                          np.mean(text_mask_d_ys[next_min: next_max]),
                          rtol=0.2):
                pass
            else:
                indices.append(i)
        indices.append(length)
        text_mask_d_ys_plateaus = np.array(
            [(prev_min, next_max)
             for previ, nexti in pairwise(indices)
             if ((next_max := text_mask_d_ys_plateaus[previ: nexti, 1][-1]) -
                 (prev_min := text_mask_d_ys_plateaus[previ: nexti, 0][0])) > 20])
        text_mask_d_ys_plateaus = text_mask_d_ys_plateaus.T

        if side == 'left':
            # search from right (widest) to left
            jumps = reversed(jumps)
        else:
            # search from left (widest) to right
            pass
        for jump in jumps:
            # non-zero, non-max plateau
            next_plateau = np.searchsorted(text_mask_d_ys_plateaus[0], jump)
            prev_plateau = np.searchsorted(text_mask_d_ys_plateaus[1], jump) - 1
            if next_plateau > text_mask_d_ys_plateaus.shape[1] - 1:
                break
            if prev_plateau < 0:
                break
            next_min = text_mask_d_ys_plateaus[0, next_plateau]
            next_max = text_mask_d_ys_plateaus[1, next_plateau]
            prev_min = text_mask_d_ys_plateaus[0, prev_plateau]
            prev_max = text_mask_d_ys_plateaus[1, prev_plateau]
            if side == 'left':
                jump_fract_main = (np.mean(text_mask_d_ys[prev_min: prev_max]) /
                                   np.mean(text_mask_d_ys[next_min: next_max]))
            else:
                jump_fract_main = (np.mean(text_mask_d_ys[next_min: next_max]) /
                                   np.mean(text_mask_d_ys[prev_min: prev_max]))

            if (MIN_THICKNESS_STEP_FRACT_MAIN < jump_fract_main < MAX_THICKNESS_STEP_FRACT_MAIN):
                # ax2.vlines([jump], 0, height, colors='y')
                # extend_ticks(ax2.xaxis, [jump], ['jump'])
                # ax2.axvspan(prev_min, prev_max, color='lightyellow')
                # ax2.axvspan(next_min, next_max, color='gold')
                return jump
            # TODO: try another criterion: deviation in average textline heights

    if len(gaps_l) == 0 and allow_l:
        if (step_l := detect_step_point("left")) is not None:
            point_l = step_l
    if len(gaps_r) == 0 and allow_r:
        if (step_r := detect_step_point("right")) is not None:
            point_r = step_r
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
    main3 = main.buffer(3)
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
        elif poly.within(main3):
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
