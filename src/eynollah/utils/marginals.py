import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from .contour import find_new_features_of_contours, return_contours_of_interested_region
from .resize import resize_image
from .rotate import rotate_image

def get_marginals(text_with_lines, text_regions, num_col, slope_deskew, kernel=None):
    # rs: text_with_lines should be called text_mask_d
    # rs: text_regions should be called early_layout (contains other classes, too)
    # rs: text_with_lines is already deskewed, while text_regions is not...
    mask_marginals = np.zeros_like(text_with_lines)
    height, width = mask_marginals.shape

    ##text_with_lines=cv2.erode(text_with_lines,self.kernel,iterations=3)
    text_with_lines_eroded = cv2.erode(text_with_lines,kernel,iterations=5)

    if height <= 1500:
        pass
    elif 1500 < height <= 1800:
        # rs: why not / 1.5???
        text_with_lines = resize_image(text_with_lines, int(height * 1.5), width)
        text_with_lines = cv2.erode(text_with_lines, kernel, iterations=5)
        # rs: and back to original size
        text_with_lines = resize_image(text_with_lines, height, width)
    else:
        # rs: why not / 1.8???
        text_with_lines = resize_image(text_with_lines, int(height * 1.8), width)
        text_with_lines = cv2.erode(text_with_lines, kernel, iterations=7)
        # rs: and back to original size
        text_with_lines = resize_image(text_with_lines, height, width)

    kernel_hor = np.ones((1, 5), dtype=np.uint8)
    text_with_lines = cv2.erode(text_with_lines, kernel_hor, iterations=6)
    text_with_lines_y = text_with_lines.sum(axis=0)
    text_with_lines_y_eroded = text_with_lines_eroded.sum(axis=0)

    max_textline_thickness_percent = 100. * text_with_lines_y_eroded.max() / height
    min_textline_thickness = max_textline_thickness_percent / 100. * height / 20.

    # plt.figure()
    # ax1 = plt.subplot(2, 1, 1, title="text_with_lines_eroded")
    # ax1.imshow(text_with_lines_eroded, aspect='auto')
    # ax2 = plt.subplot(2, 1, 2, title="text_with_lines_y_eroded", sharex=ax1)
    # ax2.plot(list(range(width)), text_with_lines_y_eroded)
    # ax2.hlines(int(0.14 * height), 0, width,
    #            label='max_textline_thickness=14%', colors='r')
    # ax2.hlines([min_textline_thickness], 0, width,
    #            label='min_textline_thickness', colors='g')
    # ax2.scatter([np.argmax(text_with_lines_y_eroded)],
    #             [text_with_lines_y_eroded.max()], color='r',
    #             label='max = %d%%' % max_textline_thickness_percent)
    # plt.legend()
    # plt.show()

    if max_textline_thickness_percent >= 14:
        text_with_lines_y_rev = np.max(text_with_lines_y) - text_with_lines_y

        region_sum_0 = gaussian_filter1d(text_with_lines_y, 1)
        first_nonzero = region_sum_0.nonzero()[0][0] # outer left
        last_nonzero = region_sum_0.nonzero()[0][-1] # outer right
        mid_point = 0.5 * (last_nonzero + first_nonzero)
        one_third_right = (last_nonzero - mid_point) / 3.0
        one_third_left = (mid_point - first_nonzero) / 3.0

        # rs: constrain the distance at least 2 characters at 12pt, retrieve height and prominence
        peaks, props = find_peaks(text_with_lines_y_rev, height=0, prominence=0, distance=30)
        peaks_orig = np.copy(peaks)
        # rs: also calculate the product of prominence and height (for final selection)
        scores = np.zeros(peaks.max() + 1)
        scores[peaks] = props['prominences'] * props['peak_heights']
        
        peaks = peaks[(peaks > first_nonzero) & (peaks < last_nonzero)]
        peaks = peaks[region_sum_0[peaks] < min_textline_thickness]

        if num_col == 1:
            peaks_right = peaks[peaks > mid_point]
            peaks_left = peaks[peaks < mid_point]
        if num_col == 2:
            peaks_right = peaks[peaks > mid_point + one_third_right]
            peaks_left = peaks[peaks < mid_point - one_third_left]

        if len(peaks_left) == 0:
            if len(peaks_right) == 0:
                # plt.figure()
                # ax1 = plt.subplot(2, 1, 1, title='text_with_lines (deskewed text+sep mask)')
                # ax1.imshow(text_with_lines, aspect='auto')
                # ax1.vlines([first_nonzero], 0, height, label='first_nonzero', colors='r')
                # ax1.vlines([last_nonzero], 0, height, label='last_nonzero', colors='r')
                # ax1.vlines(peaks_left, 0, height, label='peaks_left', colors='orange')
                # ax1.vlines(peaks_right, 0, height, label='peaks_right', colors='orange')
                # ax2 = plt.subplot(2, 1, 2, title='text_with_lines_y (smoothed)', sharex=ax1)
                # ax2.plot(list(range(width)), region_sum_0)
                # ax2.hlines(min_textline_thickness, 0, width, colors='g',
                #            label='min_textline_thickness=%d' % min_textline_thickness)
                # ax2.scatter(peaks_orig, region_sum_0[peaks_orig], label='peaks')
                # plt.legend()
                # plt.show()
                return text_regions
            point_right = peaks_right[np.argmax(scores[peaks_right])]
            #point_left = first_nonzero
            point_left = 0
        elif len(peaks_right) == 0:
            point_left = peaks_left[np.argmax(scores[peaks_left])]
            #point_right = last_nonzero
            point_right = width - 1
        elif scores[peaks_left].max() < scores[peaks_right].max():
            point_right = peaks_right[np.argmax(scores[peaks_right])]
            #point_left = first_nonzero
            point_left = 0
        else:
            point_left = peaks_left[np.argmax(scores[peaks_left])]
            #point_right = last_nonzero

        # rs: should be called mask_main (i.e. inverted semantics here)
        mask_marginals[:, point_left: point_right] = 1

        # plt.figure()
        # ax1 = plt.subplot(2, 2, 1)
        # ax1.title.set_text('text_with_lines (deskewed text+sep mask)')
        # ax1.imshow(text_with_lines)
        # ax1.vlines(peaks_left, 0, height, label='peaks_left', colors='b')
        # ax1.vlines(peaks_right, 0, height, label='peaks_right', colors='b')
        # ax1.vlines([first_nonzero], 0, height, label='first_nonzero', colors='g')
        # ax1.vlines([last_nonzero], 0, height, label='last_nonzero', colors='g')
        # ax1.vlines([point_left], 0, height, label='point_left', colors='r')
        # ax1.vlines([point_right], 0, height, label='point_right', colors='r')
        # ax2 = plt.subplot(2, 2, 2, title='mask_marginals (deskewed marginal mask)', sharey=ax1)
        # ax2.imshow(mask_marginals)
        # ax3 = plt.subplot(2, 2, 3, title='text_with_lines_y (projection for minima)', sharex=ax1)
        # ax3.plot(list(range(width)), text_with_lines_y)
        # ax3.set_aspect('auto')
        # ax4 = plt.subplot(2, 2, 4, title='text_regions (undeskewed labels)')
        # ax4.imshow(text_regions)
        # plt.legend()
        # plt.show()

        # rs: rotate back (into undeskewed/original shape as text_regions input):
        mask_marginals_rotated = rotate_image(mask_marginals, -slope_deskew)
        mask_marginals_rotated_y = mask_marginals_rotated.sum(axis=0)
        mask_marginals_rotated_y_nz = np.flatnonzero(mask_marginals_rotated_y)
        min_point_of_left_marginal = max(0, np.min(mask_marginals_rotated_y_nz) - 16)
        max_point_of_right_marginal = min(width - 1, np.max(mask_marginals_rotated_y_nz) + 16)

        min_area_text = 0.00001
        # rs: why not extract from mask_marginals_rotated???
        # rs: why not largest area instead of first?
        polygon_mask_marginals_rotated = return_contours_of_interested_region(mask_marginals, 1, min_area_text)[0]
        polygons_of_marginals = return_contours_of_interested_region(text_regions, 1, min_area_text)

        (cx_text_only,
         cy_text_only,
         x_min_text_only,
         x_max_text_only,
         y_min_text_only,
         y_max_text_only,
         y_cor_x_min_main) = find_new_features_of_contours(polygons_of_marginals)

        main_text_should_be_marginals = []
        x_min_marginals_left=[]
        x_min_marginals_right=[]

        for i, polygon in enumerate(polygons_of_marginals):
            if -1 == cv2.pointPolygonTest(polygon_mask_marginals_rotated,
                                          (cx_text_only[i],
                                           cy_text_only[i]),
                                          False):
                main_text_should_be_marginals.append(polygon)

        text_regions = cv2.fillPoly(text_regions, pts=main_text_should_be_marginals, color=4)
        # plt.figure()
        # ax1 = plt.subplot(2, 2, 1, title='mask_marginals (deskewed marginal mask)')
        # plt.imshow(mask_marginals)
        # ax2 = plt.subplot(2, 2, 2, title='mask_marginals_rotated (undeskewed marginal mask)')
        # plt.imshow(mask_marginals_rotated)
        # ax4 = plt.subplot(2, 2, 4, title='text_regions (undeskewed labels split)')
        # plt.imshow(text_regions)
        # plt.show()

        #plt.imshow(text_regions)
        #plt.show()

        #sys.exit()
    else:
        pass
    return text_regions
