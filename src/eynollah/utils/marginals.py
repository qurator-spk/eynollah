import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from .contour import find_center_of_contours, return_contours_of_interested_region
from .resize import resize_image
from .rotate import rotate_image

def get_marginals(text_mask, early_layout, num_col, slope_deskew,
                  kernel=None,
                  label_text=1,
                  label_marg=4,
):
    if kernel is None:
        kernel = np.ones((5, 5), dtype=np.uint8)
    kernel_hor = np.ones((1, 5), dtype=np.uint8)

    text_mask_d = rotate_image(text_mask, slope_deskew)
    main_mask_d = np.zeros_like(text_mask_d)
    height, width = main_mask_d.shape

    text_mask_d_eroded = cv2.erode(text_mask_d, kernel, iterations=5)

    if height <= 1500:
        pass
    elif 1500 < height <= 1800:
        # rs: why not / 1.5???
        text_mask_d = resize_image(text_mask_d, int(height * 1.5), width)
        text_mask_d = cv2.erode(text_mask_d, kernel, iterations=5)
        # rs: and back to original size
        text_mask_d = resize_image(text_mask_d, height, width)
    else:
        # rs: why not / 1.8???
        text_mask_d = resize_image(text_mask_d, int(height * 1.8), width)
        text_mask_d = cv2.erode(text_mask_d, kernel, iterations=7)
        # rs: and back to original size
        text_mask_d = resize_image(text_mask_d, height, width)

    text_mask_d = cv2.erode(text_mask_d, kernel_hor, iterations=6)
    text_mask_d_y = text_mask_d.sum(axis=0)
    text_mask_d_y_eroded = text_mask_d_eroded.sum(axis=0)

    max_text_thickness_percent = 100. * text_mask_d_y_eroded.max() / height
    min_text_thickness = max_text_thickness_percent / 100. * height / 20.

    # plt.figure()
    # ax1 = plt.subplot(2, 1, 1, title="text_mask_d_eroded")
    # ax1.imshow(text_mask_d_eroded, aspect='auto')
    # ax2 = plt.subplot(2, 1, 2, title="text_mask_d_y_eroded", sharex=ax1)
    # ax2.plot(list(range(width)), text_mask_d_y_eroded)
    # ax2.hlines(int(0.14 * height), 0, width,
    #            label='max_text_thickness=14%', colors='r')
    # ax2.hlines([min_text_thickness], 0, width,
    #            label='min_text_thickness', colors='g')
    # ax2.scatter([np.argmax(text_mask_d_y_eroded)],
    #             [text_mask_d_y_eroded.max()], color='r',
    #             label='max = %d%%' % max_text_thickness_percent)
    # plt.legend()
    # plt.show()

    if max_text_thickness_percent < 14:
        return

    text_mask_d_y_rev = np.max(text_mask_d_y) - text_mask_d_y
    region_sum_0 = gaussian_filter1d(text_mask_d_y, 1)
    first_nonzero = region_sum_0.nonzero()[0][0] # outer left
    last_nonzero = region_sum_0.nonzero()[0][-1] # outer right
    mid_point = 0.5 * (last_nonzero + first_nonzero)
    one_third_right = (last_nonzero - mid_point) / 3.0
    one_third_left = (mid_point - first_nonzero) / 3.0

    # rs: constrain the distance at least 2 characters at 12pt, retrieve height and prominence
    peaks, props = find_peaks(text_mask_d_y_rev, height=0, prominence=0, distance=30)
    peaks_orig = np.copy(peaks)
    # rs: also calculate the product of prominence and height (for final selection)
    scores = np.zeros(peaks.max() + 1)
    scores[peaks] = props['prominences'] * props['peak_heights']

    peaks = peaks[(peaks > first_nonzero) & (peaks < last_nonzero)]
    peaks = peaks[region_sum_0[peaks] < min_text_thickness]

    if num_col == 1:
        peaks_right = peaks[peaks > mid_point]
        peaks_left = peaks[peaks < mid_point]
    elif num_col == 2:
        peaks_right = peaks[peaks > mid_point + one_third_right]
        peaks_left = peaks[peaks < mid_point - one_third_left]
    else:
        # should not happen, anyway
        return

    if len(peaks_left) == 0:
        if len(peaks_right) == 0:
            # plt.figure()
            # ax1 = plt.subplot(2, 1, 1, title='text_mask_d (deskewed text+sep mask)')
            # ax1.imshow(text_mask_d, aspect='auto')
            # ax1.vlines([first_nonzero], 0, height, label='first_nonzero', colors='r')
            # ax1.vlines([last_nonzero], 0, height, label='last_nonzero', colors='r')
            # ax1.vlines(peaks_left, 0, height, label='peaks_left', colors='orange')
            # ax1.vlines(peaks_right, 0, height, label='peaks_right', colors='orange')
            # ax2 = plt.subplot(2, 1, 2, title='text_mask_d_y (smoothed)', sharex=ax1)
            # ax2.plot(list(range(width)), region_sum_0)
            # ax2.hlines(min_text_thickness, 0, width, colors='g',
            #            label='min_text_thickness=%d' % min_text_thickness)
            # ax2.scatter(peaks_orig, region_sum_0[peaks_orig], label='peaks')
            # plt.legend()
            # plt.show()
            return
        point_right = peaks_right[np.argmax(scores[peaks_right])]
        #point_left = first_nonzero
        point_left = 0
    elif len(peaks_right) == 0:
        point_left = peaks_left[np.argmax(scores[peaks_left])]
        #point_right = last_nonzero
        point_right = width - 1
    else:
        best_left = np.argmax(scores[peaks_left])
        best_right = np.argmax(scores[peaks_right])
        point_left = peaks_left[best_left]
        point_right = peaks_right[best_right]
        if scores[best_left] < 0.1 * scores[best_right]:
            point_left = 0
            #point_left = first_nonzero
        if scores[best_right] < 0.1 * scores[best_left]:
            point_right = 0
            #point_right = last_nonzero

    main_mask_d[:, point_left: point_right] = 1
    if not np.any(main_mask_d):
        return

    # plt.figure()
    # ax1 = plt.subplot(2, 2, 1)
    # ax1.title.set_text('text_mask_d (deskewed text+table mask)')
    # ax1.imshow(text_mask_d)
    # ax1.vlines(peaks_left, 0, height, label='peaks_left', colors='b')
    # ax1.vlines(peaks_right, 0, height, label='peaks_right', colors='b')
    # ax1.vlines([first_nonzero], 0, height, label='first_nonzero', colors='g')
    # ax1.vlines([last_nonzero], 0, height, label='last_nonzero', colors='g')
    # ax1.vlines([point_left], 0, height, label='point_left', colors='r')
    # ax1.vlines([point_right], 0, height, label='point_right', colors='r')
    # ax2 = plt.subplot(2, 2, 2, title='main_mask_d (deskewed main mask)', sharey=ax1)
    # ax2.imshow(main_mask_d)
    # ax3 = plt.subplot(2, 2, 3, title='text_mask_d_y (projection for minima)', sharex=ax1)
    # ax3.plot(list(range(width)), text_mask_d_y)
    # ax3.set_aspect('auto')
    # ax4 = plt.subplot(2, 2, 4, title='early_layout (undeskewed labels)')
    # ax4.imshow(early_layout)
    # plt.legend()
    # plt.show()

    # rs: rotate back (into undeskewed/original shape as early_layout input):
    main_mask = rotate_image(main_mask_d, -slope_deskew)

    min_area_text = 0.00001
    main_contour = return_contours_of_interested_region(main_mask, 1, min_area_text)[0]
    text_contours = return_contours_of_interested_region(early_layout, label_text, min_area_text)
    cx_text, cy_text = find_center_of_contours(text_contours)

    marg_contours = []
    for i, contour in enumerate(text_contours):
        if -1 == cv2.pointPolygonTest(main_contour,
                                      (cx_text[i],
                                       cy_text[i]),
                                      False):
            marg_contours.append(contour)

    # early_layout_orig = np.copy(early_layout)
    early_layout = cv2.fillPoly(early_layout, pts=marg_contours, color=label_marg)

    # plt.figure()
    # ax1 = plt.subplot(2, 2, 1, title='main_mask_d (deskewed main mask)')
    # plt.imshow(main_mask_d)
    # ax2 = plt.subplot(2, 2, 2, title='main_mask (undeskewed main mask)')
    # plt.imshow(main_mask)
    # ax3 = plt.subplot(2, 2, 3, title='early_layout (undeskewed labels original)')
    # plt.imshow(early_layout_orig)
    # ax4 = plt.subplot(2, 2, 4, title='early_layout (undeskewed labels split)')
    # plt.imshow(early_layout)
    # plt.show()

    # if there was no main text, then relabel marginalia as main
    if not np.any(early_layout == label_text):
        early_layout[early_layout == label_marg] = label_text
