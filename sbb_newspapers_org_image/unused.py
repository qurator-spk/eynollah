"""
Unused methods from eynollah
"""

import numpy as np
from shapely import geometry
import cv2

def color_images_diva(seg, n_classes):
    """
    XXX unused
    """
    ann_u = range(n_classes)
    if len(np.shape(seg)) == 3:
        seg = seg[:, :, 0]

    seg_img = np.zeros((np.shape(seg)[0], np.shape(seg)[1], 3)).astype(float)
    # colors=sns.color_palette("hls", n_classes)
    colors = [[1, 0, 0], [8, 0, 0], [2, 0, 0], [4, 0, 0]]

    for c in ann_u:
        c = int(c)
        segl = seg == c
        seg_img[:, :, 0][seg == c] = colors[c][0]  # segl*(colors[c][0])
        seg_img[:, :, 1][seg == c] = colors[c][1]  # seg_img[:,:,1]=segl*(colors[c][1])
        seg_img[:, :, 2][seg == c] = colors[c][2]  # seg_img[:,:,2]=segl*(colors[c][2])
    return seg_img

def find_polygons_size_filter(contours, median_area, scaler_up=1.2, scaler_down=0.8):
    """
    XXX unused
    """
    found_polygons_early = list()

    for c in contours:
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue

        polygon = geometry.Polygon([point[0] for point in c])
        area = polygon.area
        # Check that polygon has area greater than minimal area
        if area >= median_area * scaler_down and area <= median_area * scaler_up:
            found_polygons_early.append(np.array([point for point in polygon.exterior.coords], dtype=np.uint))
    return found_polygons_early

def resize_ann(seg_in, input_height, input_width):
    """
    XXX unused
    """
    return cv2.resize(seg_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)

def get_one_hot(seg, input_height, input_width, n_classes):
    seg = seg[:, :, 0]
    seg_f = np.zeros((input_height, input_width, n_classes))
    for j in range(n_classes):
        seg_f[:, :, j] = (seg == j).astype(int)
    return seg_f

def color_images(seg, n_classes):
    ann_u = range(n_classes)
    if len(np.shape(seg)) == 3:
        seg = seg[:, :, 0]

    seg_img = np.zeros((np.shape(seg)[0], np.shape(seg)[1], 3)).astype(np.uint8)
    colors = sns.color_palette("hls", n_classes)

    for c in ann_u:
        c = int(c)
        segl = seg == c
        seg_img[:, :, 0] = segl * c
        seg_img[:, :, 1] = segl * c
        seg_img[:, :, 2] = segl * c
    return seg_img

def cleaning_probs(probs: np.ndarray, sigma: float) -> np.ndarray:
    # Smooth
    if sigma > 0.0:
        return cv2.GaussianBlur(probs, (int(3 * sigma) * 2 + 1, int(3 * sigma) * 2 + 1), sigma)
    elif sigma == 0.0:
        return cv2.fastNlMeansDenoising((probs * 255).astype(np.uint8), h=20) / 255
    else:  # Negative sigma, do not do anything
        return probs


def early_deskewing_slope_calculation_based_on_lines(region_pre_p):
    # lines are labels by 6 in this model
    seperators_closeup = ((region_pre_p[:, :, :] == 6)) * 1

    seperators_closeup = seperators_closeup.astype(np.uint8)
    imgray = cv2.cvtColor(seperators_closeup, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    contours_lines, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slope_lines, dist_x, x_min_main, x_max_main, cy_main, slope_lines_org, y_min_main, y_max_main, cx_main = find_features_of_lines(contours_lines)

    slope_lines_org_hor = slope_lines_org[slope_lines == 0]
    args = np.array(range(len(slope_lines)))
    len_x = seperators_closeup.shape[1] / 4.0

    args_hor = args[slope_lines == 0]
    dist_x_hor = dist_x[slope_lines == 0]
    x_min_main_hor = x_min_main[slope_lines == 0]
    x_max_main_hor = x_max_main[slope_lines == 0]
    cy_main_hor = cy_main[slope_lines == 0]

    args_hor = args_hor[dist_x_hor >= len_x / 2.0]
    x_max_main_hor = x_max_main_hor[dist_x_hor >= len_x / 2.0]
    x_min_main_hor = x_min_main_hor[dist_x_hor >= len_x / 2.0]
    cy_main_hor = cy_main_hor[dist_x_hor >= len_x / 2.0]
    slope_lines_org_hor = slope_lines_org_hor[dist_x_hor >= len_x / 2.0]

    slope_lines_org_hor = slope_lines_org_hor[np.abs(slope_lines_org_hor) < 1.2]
    slope_mean_hor = np.mean(slope_lines_org_hor)

    if np.abs(slope_mean_hor) > 1.2:
        slope_mean_hor = 0

    # deskewed_new=rotate_image(image_regions_eraly_p[:,:,:],slope_mean_hor)

    args_ver = args[slope_lines == 1]
    y_min_main_ver = y_min_main[slope_lines == 1]
    y_max_main_ver = y_max_main[slope_lines == 1]
    x_min_main_ver = x_min_main[slope_lines == 1]
    x_max_main_ver = x_max_main[slope_lines == 1]
    cx_main_ver = cx_main[slope_lines == 1]
    dist_y_ver = y_max_main_ver - y_min_main_ver
    len_y = seperators_closeup.shape[0] / 3.0

    return slope_mean_hor, cx_main_ver, dist_y_ver

def boosting_text_only_regions_by_header(textregion_pre_np, img_only_text):
    result = ((img_only_text[:, :] == 1) | (textregion_pre_np[:, :, 0] == 2)) * 1
    return result

def return_rotated_contours(slope, img_patch):
    dst = rotate_image(img_patch, slope)
    dst = dst.astype(np.uint8)
    dst = dst[:, :, 0]
    dst[dst != 0] = 1

    imgray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 0, 255, 0)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_textlines_for_each_textregions(self, textline_mask_tot, boxes):
    textline_mask_tot = cv2.erode(textline_mask_tot, self.kernel, iterations=1)
    self.area_of_cropped = []
    self.all_text_region_raw = []
    for jk in range(len(boxes)):
        crop_img, crop_coor = crop_image_inside_box(boxes[jk], np.repeat(textline_mask_tot[:, :, np.newaxis], 3, axis=2))
        crop_img = crop_img.astype(np.uint8)
        self.all_text_region_raw.append(crop_img[:, :, 0])
        self.area_of_cropped.append(crop_img.shape[0] * crop_img.shape[1])

def deskew_region_prediction(regions_prediction, slope):
    image_regions_deskewd = np.zeros(regions_prediction[:, :].shape)
    for ind in np.unique(regions_prediction[:, :]):
        interest_reg = (regions_prediction[:, :] == ind) * 1
        interest_reg = interest_reg.astype(np.uint8)
        deskewed_new = rotate_image(interest_reg, slope)
        deskewed_new = deskewed_new[:, :]
        deskewed_new[deskewed_new != 0] = ind

        image_regions_deskewd = image_regions_deskewd + deskewed_new
    return image_regions_deskewd

def deskew_erarly(textline_mask):
    textline_mask_org = np.copy(textline_mask)
    # print(textline_mask.shape,np.unique(textline_mask),'hizzzzz')
    # slope_new=0#deskew_images(img_patch)

    textline_mask = np.repeat(textline_mask[:, :, np.newaxis], 3, axis=2) * 255

    textline_mask = textline_mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    imgray = cv2.cvtColor(textline_mask, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    contours, hirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(hirarchy)

    commenst_contours = filter_contours_area_of_image(thresh, contours, hirarchy, max_area=0.01, min_area=0.003)
    main_contours = filter_contours_area_of_image(thresh, contours, hirarchy, max_area=1, min_area=0.003)
    interior_contours = filter_contours_area_of_image_interiors(thresh, contours, hirarchy, max_area=1, min_area=0)

    img_comm = np.zeros(thresh.shape)
    img_comm_in = cv2.fillPoly(img_comm, pts=main_contours, color=(255, 255, 255))
    ###img_comm_in=cv2.fillPoly(img_comm, pts =interior_contours, color=(0,0,0))

    img_comm_in = np.repeat(img_comm_in[:, :, np.newaxis], 3, axis=2)
    img_comm_in = img_comm_in.astype(np.uint8)

    imgray = cv2.cvtColor(img_comm_in, cv2.COLOR_BGR2GRAY)
    ##imgray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ##mask = cv2.inRange(imgray, lower_blue, upper_blue)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    # print(np.unique(mask))
    ##ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    ##plt.imshow(thresh)
    ##plt.show()

    contours, hirarchy = cv2.findContours(thresh.copy(), cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(contours[jj]) for jj in range(len(contours))]

    median_area = np.mean(areas)
    contours_slope = contours  # self.find_polugons_size_filter(contours,median_area=median_area,scaler_up=100,scaler_down=0.5)

    if len(contours_slope) > 0:
        for jv in range(len(contours_slope)):
            new_poly = list(contours_slope[jv])
            if jv == 0:
                merged_all = new_poly
            else:
                merged_all = merged_all + new_poly

        merge = np.array(merged_all)

        img_in = np.zeros(textline_mask.shape)
        img_p_in = cv2.fillPoly(img_in, pts=[merge], color=(255, 255, 255))

        ##plt.imshow(img_p_in)
        ##plt.show()

        rect = cv2.minAreaRect(merge)

        box = cv2.boxPoints(rect)

        box = np.int0(box)

        indexes = [0, 1, 2, 3]
        x_list = box[:, 0]
        y_list = box[:, 1]

        index_y_sort = np.argsort(y_list)

        index_upper_left = index_y_sort[np.argmin(x_list[index_y_sort[0:2]])]
        index_upper_right = index_y_sort[np.argmax(x_list[index_y_sort[0:2]])]

        index_lower_left = index_y_sort[np.argmin(x_list[index_y_sort[2:]]) + 2]
        index_lower_right = index_y_sort[np.argmax(x_list[index_y_sort[2:]]) + 2]

        alpha1 = float(box[index_upper_right][1] - box[index_upper_left][1]) / (float(box[index_upper_right][0] - box[index_upper_left][0]))
        alpha2 = float(box[index_lower_right][1] - box[index_lower_left][1]) / (float(box[index_lower_right][0] - box[index_lower_left][0]))

        slope_true = (alpha1 + alpha2) / 2.0

        # slope=0#slope_true/np.pi*180

        # if abs(slope)>=1:
        # slope=0

        # dst=rotate_image(textline_mask,slope_true)
        # dst=dst[:,:,0]
        # dst[dst!=0]=1
    image_regions_deskewd = np.zeros(textline_mask_org[:, :].shape)
    for ind in np.unique(textline_mask_org[:, :]):
        interest_reg = (textline_mask_org[:, :] == ind) * 1
        interest_reg = interest_reg.astype(np.uint8)
        deskewed_new = rotate_image(interest_reg, slope_true)
        deskewed_new = deskewed_new[:, :]
        deskewed_new[deskewed_new != 0] = ind

        image_regions_deskewd = image_regions_deskewd + deskewed_new
    return image_regions_deskewd, slope_true

