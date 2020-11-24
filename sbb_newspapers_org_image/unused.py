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

def get_all_image_patches_coordination(self, image_page):
    self.all_box_coord = []
    for jk in range(len(self.boxes)):
        _, crop_coor = crop_image_inside_box(self.boxes[jk], image_page)
        self.all_box_coord.append(crop_coor)

def find_num_col_olddd(self, regions_without_seperators, sigma_, multiplier=3.8):
    regions_without_seperators_0 = regions_without_seperators[:, :].sum(axis=1)

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

    last_nonzero = last_nonzero - 0  # 100
    first_nonzero = first_nonzero + 0  # +100

    peaks_neg = peaks_neg[(peaks_neg > first_nonzero) & (peaks_neg < last_nonzero)]

    peaks = peaks[(peaks > 0.06 * regions_without_seperators.shape[1]) & (peaks < 0.94 * regions_without_seperators.shape[1])]

    interest_pos = z[peaks]

    interest_pos = interest_pos[interest_pos > 10]

    interest_neg = z[peaks_neg]

    if interest_neg[0] < 0.1:
        interest_neg = interest_neg[1:]
    if interest_neg[len(interest_neg) - 1] < 0.1:
        interest_neg = interest_neg[: len(interest_neg) - 1]

    min_peaks_pos = np.min(interest_pos)
    min_peaks_neg = 0  # np.min(interest_neg)

    dis_talaei = (min_peaks_pos - min_peaks_neg) / multiplier
    grenze = min_peaks_pos - dis_talaei  # np.mean(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])-np.std(y[peaks_neg[0]:peaks_neg[len(peaks_neg)-1]])/2.0

    interest_neg_fin = interest_neg  # [(interest_neg<grenze)]
    peaks_neg_fin = peaks_neg  # [(interest_neg<grenze)]
    interest_neg_fin = interest_neg  # [(interest_neg<grenze)]

    num_col = (len(interest_neg_fin)) + 1

    p_l = 0
    p_u = len(y) - 1
    p_m = int(len(y) / 2.0)
    p_g_l = int(len(y) / 3.0)
    p_g_u = len(y) - int(len(y) / 3.0)

    diff_peaks = np.abs(np.diff(peaks_neg_fin))
    diff_peaks_annormal = diff_peaks[diff_peaks < 30]

    return interest_neg_fin

def return_regions_without_seperators_new(self, regions_pre, regions_only_text):
    kernel = np.ones((5, 5), np.uint8)

    regions_without_seperators = ((regions_pre[:, :] != 6) & (regions_pre[:, :] != 0) & (regions_pre[:, :] != 1) & (regions_pre[:, :] != 2)) * 1

    # plt.imshow(regions_without_seperators)
    # plt.show()

    regions_without_seperators_n = ((regions_without_seperators[:, :] == 1) | (regions_only_text[:, :] == 1)) * 1

    # regions_without_seperators=( (image_regions_eraly_p[:,:,:]!=6) & (image_regions_eraly_p[:,:,:]!=0) & (image_regions_eraly_p[:,:,:]!=5) & (image_regions_eraly_p[:,:,:]!=8) & (image_regions_eraly_p[:,:,:]!=7))*1

    regions_without_seperators_n = regions_without_seperators_n.astype(np.uint8)

    regions_without_seperators_n = cv2.erode(regions_without_seperators_n, kernel, iterations=6)

    return regions_without_seperators_n

def find_images_contours_and_replace_table_and_graphic_pixels_by_image(region_pre_p):

    # pixels of images are identified by 5
    cnts_images = (region_pre_p[:, :, 0] == 5) * 1
    cnts_images = cnts_images.astype(np.uint8)
    cnts_images = np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
    imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hiearchy)
    # print(len(contours_imgs),'contours_imgs')
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=1, min_area=0.0003)

    # print(len(contours_imgs),'contours_imgs')

    boxes_imgs = return_bonding_box_of_contours(contours_imgs)

    for i in range(len(boxes_imgs)):
        x1 = int(boxes_imgs[i][0])
        x2 = int(boxes_imgs[i][0] + boxes_imgs[i][2])
        y1 = int(boxes_imgs[i][1])
        y2 = int(boxes_imgs[i][1] + boxes_imgs[i][3])
        region_pre_p[y1:y2, x1:x2, 0][region_pre_p[y1:y2, x1:x2, 0] == 8] = 5
        region_pre_p[y1:y2, x1:x2, 0][region_pre_p[y1:y2, x1:x2, 0] == 7] = 5
    return region_pre_p

def order_and_id_of_texts_old(found_polygons_text_region, matrix_of_orders, indexes_sorted):
    id_of_texts = []
    order_of_texts = []
    index_b = 0
    for mm in range(len(found_polygons_text_region)):
        id_of_texts.append("r" + str(index_b))
        index_matrix = matrix_of_orders[:, 0][(matrix_of_orders[:, 1] == 1) & (matrix_of_orders[:, 4] == mm)]
        order_of_texts.append(np.where(indexes_sorted == index_matrix)[0][0])

        index_b += 1

    order_of_texts
    return order_of_texts, id_of_texts

def order_of_regions_old(textline_mask, contours_main):
    mada_n = textline_mask.sum(axis=1)
    y = mada_n[:]

    y_help = np.zeros(len(y) + 40)
    y_help[20 : len(y) + 20] = y
    x = np.array(range(len(y)))

    peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)

    sigma_gaus = 8

    z = gaussian_filter1d(y_help, sigma_gaus)
    zneg_rev = -y_help + np.max(y_help)

    zneg = np.zeros(len(zneg_rev) + 40)
    zneg[20 : len(zneg_rev) + 20] = zneg_rev
    zneg = gaussian_filter1d(zneg, sigma_gaus)

    peaks, _ = find_peaks(z, height=0)
    peaks_neg, _ = find_peaks(zneg, height=0)

    peaks_neg = peaks_neg - 20 - 20
    peaks = peaks - 20

    if contours_main != None:
        areas_main = np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
        M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
        cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
        cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
        x_min_main = np.array([np.min(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])
        x_max_main = np.array([np.max(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

        y_min_main = np.array([np.min(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
        y_max_main = np.array([np.max(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])

    if contours_main != None:
        indexer_main = np.array(range(len(contours_main)))

    if contours_main != None:
        len_main = len(contours_main)
    else:
        len_main = 0

    matrix_of_orders = np.zeros((len_main, 5))

    matrix_of_orders[:, 0] = np.array(range(len_main))

    matrix_of_orders[:len_main, 1] = 1
    matrix_of_orders[len_main:, 1] = 2

    matrix_of_orders[:len_main, 2] = cx_main
    matrix_of_orders[:len_main, 3] = cy_main

    matrix_of_orders[:len_main, 4] = np.array(range(len_main))

    peaks_neg_new = []
    peaks_neg_new.append(0)
    for iii in range(len(peaks_neg)):
        peaks_neg_new.append(peaks_neg[iii])
    peaks_neg_new.append(textline_mask.shape[0])

    final_indexers_sorted = []
    for i in range(len(peaks_neg_new) - 1):
        top = peaks_neg_new[i]
        down = peaks_neg_new[i + 1]

        indexes_in = matrix_of_orders[:, 0][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]
        cxs_in = matrix_of_orders[:, 2][(matrix_of_orders[:, 3] >= top) & ((matrix_of_orders[:, 3] < down))]

        sorted_inside = np.argsort(cxs_in)

        ind_in_int = indexes_in[sorted_inside]

        for j in range(len(ind_in_int)):
            final_indexers_sorted.append(int(ind_in_int[j]))

    return final_indexers_sorted, matrix_of_orders
