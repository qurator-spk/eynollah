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

def cleaning_probs(probs, sigma):
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

def remove_headers_and_mains_intersection(seperators_closeup_n, img_revised_tab, boxes):
    for ind in range(len(boxes)):
        asp = np.zeros((img_revised_tab[:, :, 0].shape[0], seperators_closeup_n[:, :, 0].shape[1]))
        asp[int(boxes[ind][2]) : int(boxes[ind][3]), int(boxes[ind][0]) : int(boxes[ind][1])] = img_revised_tab[int(boxes[ind][2]) : int(boxes[ind][3]), int(boxes[ind][0]) : int(boxes[ind][1]), 0]

        head_patch_con = (asp[:, :] == 2) * 1
        main_patch_con = (asp[:, :] == 1) * 1
        # print(head_patch_con)
        head_patch_con = head_patch_con.astype(np.uint8)
        main_patch_con = main_patch_con.astype(np.uint8)

        head_patch_con = np.repeat(head_patch_con[:, :, np.newaxis], 3, axis=2)
        main_patch_con = np.repeat(main_patch_con[:, :, np.newaxis], 3, axis=2)

        imgray = cv2.cvtColor(head_patch_con, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours_head_patch_con, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_head_patch_con = return_parent_contours(contours_head_patch_con, hiearchy)

        imgray = cv2.cvtColor(main_patch_con, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours_main_patch_con, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_main_patch_con = return_parent_contours(contours_main_patch_con, hiearchy)

        y_patch_head_min, y_patch_head_max, _ = find_features_of_contours(contours_head_patch_con)
        y_patch_main_min, y_patch_main_max, _ = find_features_of_contours(contours_main_patch_con)

        for i in range(len(y_patch_head_min)):
            for j in range(len(y_patch_main_min)):
                if y_patch_head_max[i] > y_patch_main_min[j] and y_patch_head_min[i] < y_patch_main_min[j]:
                    y_down = y_patch_head_max[i]
                    y_up = y_patch_main_min[j]

                    patch_intersection = np.zeros(asp.shape)
                    patch_intersection[y_up:y_down, :] = asp[y_up:y_down, :]

                    head_patch_con = (patch_intersection[:, :] == 2) * 1
                    main_patch_con = (patch_intersection[:, :] == 1) * 1
                    head_patch_con = head_patch_con.astype(np.uint8)
                    main_patch_con = main_patch_con.astype(np.uint8)

                    head_patch_con = np.repeat(head_patch_con[:, :, np.newaxis], 3, axis=2)
                    main_patch_con = np.repeat(main_patch_con[:, :, np.newaxis], 3, axis=2)

                    imgray = cv2.cvtColor(head_patch_con, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                    contours_head_patch_con, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours_head_patch_con = return_parent_contours(contours_head_patch_con, hiearchy)

                    imgray = cv2.cvtColor(main_patch_con, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                    contours_main_patch_con, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours_main_patch_con = return_parent_contours(contours_main_patch_con, hiearchy)

                    _, _, areas_head = find_features_of_contours(contours_head_patch_con)
                    _, _, areas_main = find_features_of_contours(contours_main_patch_con)

                    if np.sum(areas_head) > np.sum(areas_main):
                        img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0][img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0] == 1] = 2
                    else:
                        img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0][img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0] == 2] = 1

                elif y_patch_head_min[i] < y_patch_main_max[j] and y_patch_head_max[i] > y_patch_main_max[j]:
                    y_down = y_patch_main_max[j]
                    y_up = y_patch_head_min[i]

                    patch_intersection = np.zeros(asp.shape)
                    patch_intersection[y_up:y_down, :] = asp[y_up:y_down, :]

                    head_patch_con = (patch_intersection[:, :] == 2) * 1
                    main_patch_con = (patch_intersection[:, :] == 1) * 1
                    head_patch_con = head_patch_con.astype(np.uint8)
                    main_patch_con = main_patch_con.astype(np.uint8)

                    head_patch_con = np.repeat(head_patch_con[:, :, np.newaxis], 3, axis=2)
                    main_patch_con = np.repeat(main_patch_con[:, :, np.newaxis], 3, axis=2)

                    imgray = cv2.cvtColor(head_patch_con, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                    contours_head_patch_con, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours_head_patch_con = return_parent_contours(contours_head_patch_con, hiearchy)

                    imgray = cv2.cvtColor(main_patch_con, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                    contours_main_patch_con, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours_main_patch_con = return_parent_contours(contours_main_patch_con, hiearchy)

                    _, _, areas_head = find_features_of_contours(contours_head_patch_con)
                    _, _, areas_main = find_features_of_contours(contours_main_patch_con)

                    if np.sum(areas_head) > np.sum(areas_main):
                        img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0][img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0] == 1] = 2
                    else:
                        img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0][img_revised_tab[y_up:y_down, int(boxes[ind][0]) : int(boxes[ind][1]), 0] == 2] = 1

                    # print(np.unique(patch_intersection) )
                    ##plt.figure(figsize=(20,20))
                    ##plt.imshow(patch_intersection)
                    ##plt.show()
                else:
                    pass

    return img_revised_tab

def tear_main_texts_on_the_boundaries_of_boxes(img_revised_tab, boxes):
    for i in range(len(boxes)):
        img_revised_tab[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][1] - 10) : int(boxes[i][1]), 0][img_revised_tab[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][1] - 10) : int(boxes[i][1]), 0] == 1] = 0
        img_revised_tab[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][1] - 10) : int(boxes[i][1]), 1][img_revised_tab[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][1] - 10) : int(boxes[i][1]), 1] == 1] = 0
        img_revised_tab[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][1] - 10) : int(boxes[i][1]), 2][img_revised_tab[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][1] - 10) : int(boxes[i][1]), 2] == 1] = 0
    return img_revised_tab

def combine_hor_lines_and_delete_cross_points_and_get_lines_features_back(self, regions_pre_p):
    seperators_closeup = ((regions_pre_p[:, :] == 6)) * 1

    seperators_closeup = seperators_closeup.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    seperators_closeup = cv2.dilate(seperators_closeup, kernel, iterations=1)
    seperators_closeup = cv2.erode(seperators_closeup, kernel, iterations=1)

    seperators_closeup = cv2.erode(seperators_closeup, kernel, iterations=1)
    seperators_closeup = cv2.dilate(seperators_closeup, kernel, iterations=1)

    if len(seperators_closeup.shape) == 2:
        seperators_closeup_n = np.zeros((seperators_closeup.shape[0], seperators_closeup.shape[1], 3))
        seperators_closeup_n[:, :, 0] = seperators_closeup
        seperators_closeup_n[:, :, 1] = seperators_closeup
        seperators_closeup_n[:, :, 2] = seperators_closeup
    else:
        seperators_closeup_n = seperators_closeup[:, :, :]
    # seperators_closeup=seperators_closeup.astype(np.uint8)
    seperators_closeup_n = seperators_closeup_n.astype(np.uint8)
    imgray = cv2.cvtColor(seperators_closeup_n, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours_lines, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    slope_lines, dist_x, x_min_main, x_max_main, cy_main, slope_lines_org, y_min_main, y_max_main, cx_main = find_features_of_lines(contours_lines)

    dist_y = np.abs(y_max_main - y_min_main)

    slope_lines_org_hor = slope_lines_org[slope_lines == 0]
    args = np.array(range(len(slope_lines)))
    len_x = seperators_closeup.shape[1] * 0
    len_y = seperators_closeup.shape[0] * 0.01

    args_hor = args[slope_lines == 0]
    dist_x_hor = dist_x[slope_lines == 0]
    dist_y_hor = dist_y[slope_lines == 0]
    x_min_main_hor = x_min_main[slope_lines == 0]
    x_max_main_hor = x_max_main[slope_lines == 0]
    cy_main_hor = cy_main[slope_lines == 0]
    y_min_main_hor = y_min_main[slope_lines == 0]
    y_max_main_hor = y_max_main[slope_lines == 0]

    args_hor = args_hor[dist_x_hor >= len_x]
    x_max_main_hor = x_max_main_hor[dist_x_hor >= len_x]
    x_min_main_hor = x_min_main_hor[dist_x_hor >= len_x]
    cy_main_hor = cy_main_hor[dist_x_hor >= len_x]
    y_min_main_hor = y_min_main_hor[dist_x_hor >= len_x]
    y_max_main_hor = y_max_main_hor[dist_x_hor >= len_x]
    slope_lines_org_hor = slope_lines_org_hor[dist_x_hor >= len_x]
    dist_y_hor = dist_y_hor[dist_x_hor >= len_x]
    dist_x_hor = dist_x_hor[dist_x_hor >= len_x]

    args_ver = args[slope_lines == 1]
    dist_y_ver = dist_y[slope_lines == 1]
    dist_x_ver = dist_x[slope_lines == 1]
    x_min_main_ver = x_min_main[slope_lines == 1]
    x_max_main_ver = x_max_main[slope_lines == 1]
    y_min_main_ver = y_min_main[slope_lines == 1]
    y_max_main_ver = y_max_main[slope_lines == 1]
    cx_main_ver = cx_main[slope_lines == 1]

    args_ver = args_ver[dist_y_ver >= len_y]
    x_max_main_ver = x_max_main_ver[dist_y_ver >= len_y]
    x_min_main_ver = x_min_main_ver[dist_y_ver >= len_y]
    cx_main_ver = cx_main_ver[dist_y_ver >= len_y]
    y_min_main_ver = y_min_main_ver[dist_y_ver >= len_y]
    y_max_main_ver = y_max_main_ver[dist_y_ver >= len_y]
    dist_x_ver = dist_x_ver[dist_y_ver >= len_y]
    dist_y_ver = dist_y_ver[dist_y_ver >= len_y]

    img_p_in_ver = np.zeros(seperators_closeup_n[:, :, 2].shape)
    for jv in range(len(args_ver)):
        img_p_in_ver = cv2.fillPoly(img_p_in_ver, pts=[contours_lines[args_ver[jv]]], color=(1, 1, 1))

    img_in_hor = np.zeros(seperators_closeup_n[:, :, 2].shape)
    for jv in range(len(args_hor)):
        img_p_in_hor = cv2.fillPoly(img_in_hor, pts=[contours_lines[args_hor[jv]]], color=(1, 1, 1))

    all_args_uniq = contours_in_same_horizon(cy_main_hor)
    # print(all_args_uniq,'all_args_uniq')
    if len(all_args_uniq) > 0:
        if type(all_args_uniq[0]) is list:
            contours_new = []
            for dd in range(len(all_args_uniq)):
                merged_all = None
                some_args = args_hor[all_args_uniq[dd]]
                some_cy = cy_main_hor[all_args_uniq[dd]]
                some_x_min = x_min_main_hor[all_args_uniq[dd]]
                some_x_max = x_max_main_hor[all_args_uniq[dd]]

                img_in = np.zeros(seperators_closeup_n[:, :, 2].shape)
                for jv in range(len(some_args)):

                    img_p_in = cv2.fillPoly(img_p_in_hor, pts=[contours_lines[some_args[jv]]], color=(1, 1, 1))
                    img_p_in[int(np.mean(some_cy)) - 5 : int(np.mean(some_cy)) + 5, int(np.min(some_x_min)) : int(np.max(some_x_max))] = 1

        else:
            img_p_in = seperators_closeup
    else:
        img_p_in = seperators_closeup

    sep_ver_hor = img_p_in + img_p_in_ver
    sep_ver_hor_cross = (sep_ver_hor == 2) * 1

    sep_ver_hor_cross = np.repeat(sep_ver_hor_cross[:, :, np.newaxis], 3, axis=2)
    sep_ver_hor_cross = sep_ver_hor_cross.astype(np.uint8)
    imgray = cv2.cvtColor(sep_ver_hor_cross, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours_cross, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cx_cross, cy_cross, _, _, _, _, _ = find_new_features_of_contoures(contours_cross)

    for ii in range(len(cx_cross)):
        sep_ver_hor[int(cy_cross[ii]) - 15 : int(cy_cross[ii]) + 15, int(cx_cross[ii]) + 5 : int(cx_cross[ii]) + 40] = 0
        sep_ver_hor[int(cy_cross[ii]) - 15 : int(cy_cross[ii]) + 15, int(cx_cross[ii]) - 40 : int(cx_cross[ii]) - 4] = 0

    img_p_in[:, :] = sep_ver_hor[:, :]

    if len(img_p_in.shape) == 2:
        seperators_closeup_n = np.zeros((img_p_in.shape[0], img_p_in.shape[1], 3))
        seperators_closeup_n[:, :, 0] = img_p_in
        seperators_closeup_n[:, :, 1] = img_p_in
        seperators_closeup_n[:, :, 2] = img_p_in
    else:
        seperators_closeup_n = img_p_in[:, :, :]
    # seperators_closeup=seperators_closeup.astype(np.uint8)
    seperators_closeup_n = seperators_closeup_n.astype(np.uint8)
    imgray = cv2.cvtColor(seperators_closeup_n, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    contours_lines, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    slope_lines, dist_x, x_min_main, x_max_main, cy_main, slope_lines_org, y_min_main, y_max_main, cx_main = find_features_of_lines(contours_lines)

    dist_y = np.abs(y_max_main - y_min_main)

    slope_lines_org_hor = slope_lines_org[slope_lines == 0]
    args = np.array(range(len(slope_lines)))
    len_x = seperators_closeup.shape[1] * 0.04
    len_y = seperators_closeup.shape[0] * 0.08

    args_hor = args[slope_lines == 0]
    dist_x_hor = dist_x[slope_lines == 0]
    dist_y_hor = dist_y[slope_lines == 0]
    x_min_main_hor = x_min_main[slope_lines == 0]
    x_max_main_hor = x_max_main[slope_lines == 0]
    cy_main_hor = cy_main[slope_lines == 0]
    y_min_main_hor = y_min_main[slope_lines == 0]
    y_max_main_hor = y_max_main[slope_lines == 0]

    args_hor = args_hor[dist_x_hor >= len_x]
    x_max_main_hor = x_max_main_hor[dist_x_hor >= len_x]
    x_min_main_hor = x_min_main_hor[dist_x_hor >= len_x]
    cy_main_hor = cy_main_hor[dist_x_hor >= len_x]
    y_min_main_hor = y_min_main_hor[dist_x_hor >= len_x]
    y_max_main_hor = y_max_main_hor[dist_x_hor >= len_x]
    slope_lines_org_hor = slope_lines_org_hor[dist_x_hor >= len_x]
    dist_y_hor = dist_y_hor[dist_x_hor >= len_x]
    dist_x_hor = dist_x_hor[dist_x_hor >= len_x]

    args_ver = args[slope_lines == 1]
    dist_y_ver = dist_y[slope_lines == 1]
    dist_x_ver = dist_x[slope_lines == 1]
    x_min_main_ver = x_min_main[slope_lines == 1]
    x_max_main_ver = x_max_main[slope_lines == 1]
    y_min_main_ver = y_min_main[slope_lines == 1]
    y_max_main_ver = y_max_main[slope_lines == 1]
    cx_main_ver = cx_main[slope_lines == 1]

    args_ver = args_ver[dist_y_ver >= len_y]
    x_max_main_ver = x_max_main_ver[dist_y_ver >= len_y]
    x_min_main_ver = x_min_main_ver[dist_y_ver >= len_y]
    cx_main_ver = cx_main_ver[dist_y_ver >= len_y]
    y_min_main_ver = y_min_main_ver[dist_y_ver >= len_y]
    y_max_main_ver = y_max_main_ver[dist_y_ver >= len_y]
    dist_x_ver = dist_x_ver[dist_y_ver >= len_y]
    dist_y_ver = dist_y_ver[dist_y_ver >= len_y]

    matrix_of_lines_ch = np.zeros((len(cy_main_hor) + len(cx_main_ver), 10))

    matrix_of_lines_ch[: len(cy_main_hor), 0] = args_hor
    matrix_of_lines_ch[len(cy_main_hor) :, 0] = args_ver

    matrix_of_lines_ch[len(cy_main_hor) :, 1] = cx_main_ver

    matrix_of_lines_ch[: len(cy_main_hor), 2] = x_min_main_hor
    matrix_of_lines_ch[len(cy_main_hor) :, 2] = x_min_main_ver

    matrix_of_lines_ch[: len(cy_main_hor), 3] = x_max_main_hor
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

    return matrix_of_lines_ch, seperators_closeup_n

def image_change_background_pixels_to_zero(self, image_page):
    image_back_zero = np.zeros((image_page.shape[0], image_page.shape[1]))
    image_back_zero[:, :] = image_page[:, :, 0]
    image_back_zero[:, :][image_back_zero[:, :] == 0] = -255
    image_back_zero[:, :][image_back_zero[:, :] == 255] = 0
    image_back_zero[:, :][image_back_zero[:, :] == -255] = 255
    return image_back_zero

def return_boxes_of_images_by_order_of_reading_without_seperator(spliter_y_new, image_p_rev, regions_without_seperators, matrix_of_lines_ch, seperators_closeup_n):

    boxes = []

    # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
    # holes in the text and also finding spliter which covers more than one columns.
    for i in range(len(spliter_y_new) - 1):
        # print(spliter_y_new[i],spliter_y_new[i+1])
        matrix_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 6] > spliter_y_new[i]) & (matrix_of_lines_ch[:, 7] < spliter_y_new[i + 1])]
        # print(len( matrix_new[:,9][matrix_new[:,9]==1] ))

        # print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')

        # check to see is there any vertical seperator to find holes.
        if np.abs(spliter_y_new[i + 1] - spliter_y_new[i]) > 1.0 / 3.0 * regions_without_seperators.shape[0]:  # len( matrix_new[:,9][matrix_new[:,9]==1] )>0 and np.max(matrix_new[:,8][matrix_new[:,9]==1])>=0.1*(np.abs(spliter_y_new[i+1]-spliter_y_new[i] )):

            # org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
            # org_img_dichte=org_img_dichte-np.min(org_img_dichte)
            ##plt.figure(figsize=(20,20))
            ##plt.plot(org_img_dichte)
            ##plt.show()
            ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)

            num_col, peaks_neg_fin = find_num_col_only_image(image_p_rev[int(spliter_y_new[i]) : int(spliter_y_new[i + 1]), :], multiplier=2.4)

            # num_col, peaks_neg_fin=find_num_col(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=7.0)
            x_min_hor_some = matrix_new[:, 2][(matrix_new[:, 9] == 0)]
            x_max_hor_some = matrix_new[:, 3][(matrix_new[:, 9] == 0)]
            cy_hor_some = matrix_new[:, 5][(matrix_new[:, 9] == 0)]
            arg_org_hor_some = matrix_new[:, 0][(matrix_new[:, 9] == 0)]

            peaks_neg_tot = return_points_with_boundies(peaks_neg_fin, 0, seperators_closeup_n[:, :, 0].shape[1])

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

            # print(len(arg_min_hor_sort),len(arg_org_hor_some_sort),'vizzzzzz')

            vahid_subset = np.zeros((len(start_index_of_hor_with_subset), len(start_index_of_hor_with_subset))) - 1
            for kkk1 in range(len(start_index_of_hor_with_subset)):

                # print(lines_indexes_deleted,'hiii')
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
                    if np.all(vahid_subset[:, li] == -1):
                        line_int[li] = -1
                    else:
                        line_int[li] = 1

                        # childs_args_in=[ idd for idd in range(vahid_subset.shape[0]) if vahid_subset[idd,li]!=-1]
                        # helpi=[]
                        # for nad in range(len(childs_args_in)):
                        #    helpi.append(arg_min_hor_sort_with_subset[childs_args_in[nad]])

                        arg_child.append(arg_min_hor_sort_with_subset[li])

                arg_parent = [arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] == -1]
                start_index_of_hor_parent = [start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] == -1]
                # arg_parent=[lines_indexes_deleted_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]
                # arg_parent=[lines_length_dels_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]==-1]

                # arg_child=[arg_min_hor_sort_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij]!=-1]
                start_index_of_hor_child = [start_index_of_hor_with_subset[vij] for vij in range(len(arg_min_hor_sort_with_subset)) if line_int[vij] != -1]

                cy_hor_some_sort = cy_hor_some[arg_parent]

                newest_y_spliter_tot = []

                for tj in range(len(newest_peaks) - 1):
                    newest_y_spliter = []
                    newest_y_spliter.append(spliter_y_new[i])
                    if tj in np.unique(start_index_of_hor_parent):
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
                            # num_col_sub, peaks_neg_fin_sub=find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=2.3)
                            num_col_sub, peaks_neg_fin_sub = find_num_col_only_image(image_p_rev[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=2.4)
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
                                ###num_col_ch, peaks_neg_ch=find_num_col( regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=2.3)

                                num_col_ch, peaks_neg_ch = find_num_col_only_image(image_p_rev[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=2.3)

                                peaks_neg_ch = peaks_neg_ch[:] + newest_peaks[j]

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
                                            # num_col_sub_ch, peaks_neg_fin_sub_ch=find_num_col(regions_without_seperators[int(newest_y_spliter_h[nd]):int(newest_y_spliter_h[nd+1]),nst_p_ch[jn]:nst_p_ch[jn+1]],multiplier=2.3)

                                            num_col_sub_ch, peaks_neg_fin_sub_ch = find_num_col_only_image(image_p_rev[int(newest_y_spliter_h[nd]) : int(newest_y_spliter_h[nd + 1]), nst_p_ch[jn] : nst_p_ch[jn + 1]], multiplier=2.3)
                                            # print(peaks_neg_fin_sub_ch,'gada kutullllllll')
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
                                    ###num_col_sub, peaks_neg_fin_sub=find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=2.3)
                                    num_col_sub, peaks_neg_fin_sub = find_num_col_only_image(image_p_rev[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=2.3)
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

                            for jvt in matrix_new[:, 0][(matrix_new[:, 9] == 1) & (matrix_new[:, 6] > newest_y_spliter[n]) & (matrix_new[:, 7] < newest_y_spliter[n + 1]) & ((matrix_new[:, 1]) < newest_peaks[j + 1]) & ((matrix_new[:, 1]) > newest_peaks[j])]:
                                pass

                                # plot_contour(regions_without_seperators.shape[0],regions_without_seperators.shape[1], contours_lines[int(jvt)])
                            # print(matrix_of_lines_ch[matrix_of_lines_ch[:,9]==1])
                            matrix_new_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 9] == 1) & (matrix_of_lines_ch[:, 6] > newest_y_spliter[n]) & (matrix_of_lines_ch[:, 7] < newest_y_spliter[n + 1]) & ((matrix_of_lines_ch[:, 1] + 500) < newest_peaks[j + 1]) & ((matrix_of_lines_ch[:, 1] - 500) > newest_peaks[j])]
                            # print(matrix_new_new,newest_y_spliter[n],newest_y_spliter[n+1],newest_peaks[j],newest_peaks[j+1],'gada')
                            if 1 > 0:  # len( matrix_new_new[:,9][matrix_new_new[:,9]==1] )>0 and np.max(matrix_new_new[:,8][matrix_new_new[:,9]==1])>=0.2*(np.abs(newest_y_spliter[n+1]-newest_y_spliter[n] )):
                                ###num_col_sub, peaks_neg_fin_sub=find_num_col(regions_without_seperators[int(newest_y_spliter[n]):int(newest_y_spliter[n+1]),newest_peaks[j]:newest_peaks[j+1]],multiplier=5.0)
                                num_col_sub, peaks_neg_fin_sub = find_num_col_only_image(image_p_rev[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=2.3)
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
            boxes.append([0, seperators_closeup_n[:, :, 0].shape[1], spliter_y_new[i], spliter_y_new[i + 1]])
    return boxes

def return_region_segmentation_after_implementing_not_head_maintext_parallel(image_regions_eraly_p, boxes):
    image_revised = np.zeros((image_regions_eraly_p.shape[0], image_regions_eraly_p.shape[1]))
    for i in range(len(boxes)):

        image_box = image_regions_eraly_p[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][0]) : int(boxes[i][1])]
        image_box = np.array(image_box)
        # plt.imshow(image_box)
        # plt.show()

        # print(int(boxes[i][2]),int(boxes[i][3]),int(boxes[i][0]),int(boxes[i][1]),'addaa')
        image_box = implent_law_head_main_not_parallel(image_box)
        image_box = implent_law_head_main_not_parallel(image_box)
        image_box = implent_law_head_main_not_parallel(image_box)

        image_revised[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][0]) : int(boxes[i][1])] = image_box[:, :]
    return image_revised

def return_boxes_of_images_by_order_of_reading_2cols(spliter_y_new, regions_without_seperators, matrix_of_lines_ch, seperators_closeup_n):
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
            # print(int(spliter_y_new[i]),int(spliter_y_new[i+1]),'burayaaaa galimiirrrrrrrrrrrrrrrrrrrrrrrrrrr')
            # org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
            # org_img_dichte=org_img_dichte-np.min(org_img_dichte)
            ##plt.figure(figsize=(20,20))
            ##plt.plot(org_img_dichte)
            ##plt.show()
            ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)

            try:
                num_col, peaks_neg_fin = find_num_col(regions_without_seperators[int(spliter_y_new[i]) : int(spliter_y_new[i + 1]), :], multiplier=7.0)

            except:
                peaks_neg_fin = []
                num_col = 0

            peaks_neg_tot = return_points_with_boundies(peaks_neg_fin, 0, seperators_closeup_n[:, :, 0].shape[1])

            for kh in range(len(peaks_neg_tot) - 1):
                boxes.append([peaks_neg_tot[kh], peaks_neg_tot[kh + 1], spliter_y_new[i], spliter_y_new[i + 1]])

        else:
            boxes.append([0, seperators_closeup_n[:, :, 0].shape[1], spliter_y_new[i], spliter_y_new[i + 1]])

    return boxes

def return_boxes_of_images_by_order_of_reading(spliter_y_new, regions_without_seperators, matrix_of_lines_ch, seperators_closeup_n):
    boxes = []

    # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
    # holes in the text and also finding spliter which covers more than one columns.
    for i in range(len(spliter_y_new) - 1):
        # print(spliter_y_new[i],spliter_y_new[i+1])
        matrix_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 6] > spliter_y_new[i]) & (matrix_of_lines_ch[:, 7] < spliter_y_new[i + 1])]
        # print(len( matrix_new[:,9][matrix_new[:,9]==1] ))

        # print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')

        # check to see is there any vertical seperator to find holes.
        if len(matrix_new[:, 9][matrix_new[:, 9] == 1]) > 0 and np.max(matrix_new[:, 8][matrix_new[:, 9] == 1]) >= 0.1 * (np.abs(spliter_y_new[i + 1] - spliter_y_new[i])):

            # org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
            # org_img_dichte=org_img_dichte-np.min(org_img_dichte)
            ##plt.figure(figsize=(20,20))
            ##plt.plot(org_img_dichte)
            ##plt.show()
            ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)

            num_col, peaks_neg_fin = find_num_col(regions_without_seperators[int(spliter_y_new[i]) : int(spliter_y_new[i + 1]), :], multiplier=7.0)

            # num_col, peaks_neg_fin=find_num_col(regions_without_seperators[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:],multiplier=7.0)
            x_min_hor_some = matrix_new[:, 2][(matrix_new[:, 9] == 0)]
            x_max_hor_some = matrix_new[:, 3][(matrix_new[:, 9] == 0)]
            cy_hor_some = matrix_new[:, 5][(matrix_new[:, 9] == 0)]
            arg_org_hor_some = matrix_new[:, 0][(matrix_new[:, 9] == 0)]

            peaks_neg_tot = return_points_with_boundies(peaks_neg_fin, 0, seperators_closeup_n[:, :, 0].shape[1])

            start_index_of_hor, newest_peaks, arg_min_hor_sort, lines_length_dels, lines_indexes_deleted = return_hor_spliter_by_index(peaks_neg_tot, x_min_hor_some, x_max_hor_some)

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

            # print(vahid_subset,'zartt222')

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

                # print(arg_child,line_int[0],'zartt33333')
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
                        ##print(cy_hor_some_sort)
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
                        if len(matrix_new_new[:, 9][matrix_new_new[:, 9] == 1]) > 0 and np.max(matrix_new_new[:, 8][matrix_new_new[:, 9] == 1]) >= 0.2 * (np.abs(newest_y_spliter[n + 1] - newest_y_spliter[n])):
                            num_col_sub, peaks_neg_fin_sub = find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=5.0)
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

                        # print(cy_hor_some_sort_child,'ychilds')

                        for n in range(len(newest_y_spliter) - 1):

                            cy_child_in = cy_hor_some_sort_child[(cy_hor_some_sort_child > newest_y_spliter[n]) & (cy_hor_some_sort_child < newest_y_spliter[n + 1])]

                            if len(cy_child_in) > 0:
                                num_col_ch, peaks_neg_ch = find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=5.0)
                                # print(peaks_neg_ch,'mizzzz')
                                # peaks_neg_ch=[]
                                # for djh in range(len(peaks_neg_ch)):
                                #    peaks_neg_ch.append( peaks_neg_ch[djh]+newest_peaks[j] )

                                peaks_neg_ch_tot = return_points_with_boundies(peaks_neg_ch, newest_peaks[j], newest_peaks[j + 1])

                                ss_in_ch, nst_p_ch, arg_n_ch, lines_l_del_ch, lines_in_del_ch = return_hor_spliter_by_index(peaks_neg_ch_tot, x_min_ch, x_max_ch)

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
                                        if len(matrix_new_new2[:, 9][matrix_new_new2[:, 9] == 1]) > 0 and np.max(matrix_new_new2[:, 8][matrix_new_new2[:, 9] == 1]) >= 0.2 * (np.abs(newest_y_spliter_h[nd + 1] - newest_y_spliter_h[nd])):
                                            num_col_sub_ch, peaks_neg_fin_sub_ch = find_num_col(regions_without_seperators[int(newest_y_spliter_h[nd]) : int(newest_y_spliter_h[nd + 1]), nst_p_ch[jn] : nst_p_ch[jn + 1]], multiplier=5.0)

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
                                if len(matrix_new_new[:, 9][matrix_new_new[:, 9] == 1]) > 0 and np.max(matrix_new_new[:, 8][matrix_new_new[:, 9] == 1]) >= 0.2 * (np.abs(newest_y_spliter[n + 1] - newest_y_spliter[n])):
                                    num_col_sub, peaks_neg_fin_sub = find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=5.0)
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
                            if len(matrix_new_new[:, 9][matrix_new_new[:, 9] == 1]) > 0 and np.max(matrix_new_new[:, 8][matrix_new_new[:, 9] == 1]) >= 0.2 * (np.abs(newest_y_spliter[n + 1] - newest_y_spliter[n])):
                                num_col_sub, peaks_neg_fin_sub = find_num_col(regions_without_seperators[int(newest_y_spliter[n]) : int(newest_y_spliter[n + 1]), newest_peaks[j] : newest_peaks[j + 1]], multiplier=5.0)
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
            boxes.append([0, seperators_closeup_n[:, :, 0].shape[1], spliter_y_new[i], spliter_y_new[i + 1]])

    return boxes

def return_boxes_of_images_by_order_of_reading_without_seperators_2cols(spliter_y_new, image_p_rev, regions_without_seperators, matrix_of_lines_ch, seperators_closeup_n):

    boxes = []

    # here I go through main spliters and i do check whether a vertical seperator there is. If so i am searching for \
    # holes in the text and also finding spliter which covers more than one columns.
    for i in range(len(spliter_y_new) - 1):
        # print(spliter_y_new[i],spliter_y_new[i+1])
        matrix_new = matrix_of_lines_ch[:, :][(matrix_of_lines_ch[:, 6] > spliter_y_new[i]) & (matrix_of_lines_ch[:, 7] < spliter_y_new[i + 1])]
        # print(len( matrix_new[:,9][matrix_new[:,9]==1] ))

        # print(matrix_new[:,8][matrix_new[:,9]==1],'gaddaaa')

        # check to see is there any vertical seperator to find holes.
        if np.abs(spliter_y_new[i + 1] - spliter_y_new[i]) > 1.0 / 3.0 * regions_without_seperators.shape[0]:  # len( matrix_new[:,9][matrix_new[:,9]==1] )>0 and np.max(matrix_new[:,8][matrix_new[:,9]==1])>=0.1*(np.abs(spliter_y_new[i+1]-spliter_y_new[i] )):

            # org_img_dichte=-gaussian_filter1d(( image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,0]/255.).sum(axis=0) ,30)
            # org_img_dichte=org_img_dichte-np.min(org_img_dichte)
            ##plt.figure(figsize=(20,20))
            ##plt.plot(org_img_dichte)
            ##plt.show()
            ###find_num_col_both_layout_and_org(regions_without_seperators,image_page[int(spliter_y_new[i]):int(spliter_y_new[i+1]),:,:],7.)

            try:
                num_col, peaks_neg_fin = find_num_col_only_image(image_p_rev[int(spliter_y_new[i]) : int(spliter_y_new[i + 1]), :], multiplier=2.4)
            except:
                peaks_neg_fin = []
                num_col = 0

            peaks_neg_tot = return_points_with_boundies(peaks_neg_fin, 0, seperators_closeup_n[:, :, 0].shape[1])

            for kh in range(len(peaks_neg_tot) - 1):
                boxes.append([peaks_neg_tot[kh], peaks_neg_tot[kh + 1], spliter_y_new[i], spliter_y_new[i + 1]])
        else:
            boxes.append([0, seperators_closeup_n[:, :, 0].shape[1], spliter_y_new[i], spliter_y_new[i + 1]])

    return boxes

def add_tables_heuristic_to_layout(image_regions_eraly_p, boxes, slope_mean_hor, spliter_y, peaks_neg_tot, image_revised):

    image_revised_1 = delete_seperator_around(spliter_y, peaks_neg_tot, image_revised)
    img_comm_e = np.zeros(image_revised_1.shape)
    img_comm = np.repeat(img_comm_e[:, :, np.newaxis], 3, axis=2)

    for indiv in np.unique(image_revised_1):

        # print(indiv,'indd')
        image_col = (image_revised_1 == indiv) * 255
        img_comm_in = np.repeat(image_col[:, :, np.newaxis], 3, axis=2)
        img_comm_in = img_comm_in.astype(np.uint8)

        imgray = cv2.cvtColor(img_comm_in, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        contours, hirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        main_contours = filter_contours_area_of_image_tables(thresh, contours, hirarchy, max_area=1, min_area=0.0001)

        img_comm = cv2.fillPoly(img_comm, pts=main_contours, color=(indiv, indiv, indiv))
        ###img_comm_in=cv2.fillPoly(img_comm, pts =interior_contours, color=(0,0,0))

        # img_comm=np.repeat(img_comm[:, :, np.newaxis], 3, axis=2)
        img_comm = img_comm.astype(np.uint8)

    if not isNaN(slope_mean_hor):
        image_revised_last = np.zeros((image_regions_eraly_p.shape[0], image_regions_eraly_p.shape[1], 3))
        for i in range(len(boxes)):

            image_box = img_comm[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][0]) : int(boxes[i][1]), :]

            image_box_tabels_1 = (image_box[:, :, 0] == 7) * 1

            contours_tab, _ = return_contours_of_image(image_box_tabels_1)

            contours_tab = filter_contours_area_of_image_tables(image_box_tabels_1, contours_tab, _, 1, 0.001)

            image_box_tabels_1 = (image_box[:, :, 0] == 6) * 1

            image_box_tabels_and_m_text = ((image_box[:, :, 0] == 7) | (image_box[:, :, 0] == 1)) * 1
            image_box_tabels_and_m_text = image_box_tabels_and_m_text.astype(np.uint8)

            image_box_tabels_1 = image_box_tabels_1.astype(np.uint8)
            image_box_tabels_1 = cv2.dilate(image_box_tabels_1, self.kernel, iterations=5)

            contours_table_m_text, _ = return_contours_of_image(image_box_tabels_and_m_text)

            image_box_tabels = np.repeat(image_box_tabels_1[:, :, np.newaxis], 3, axis=2)

            image_box_tabels = image_box_tabels.astype(np.uint8)
            imgray = cv2.cvtColor(image_box_tabels, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 0, 255, 0)

            contours_line, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            y_min_main_line, y_max_main_line, _ = find_features_of_contours(contours_line)
            # _,_,y_min_main_line ,y_max_main_line,x_min_main_line,x_max_main_line=find_new_features_of_contoures(contours_line)
            y_min_main_tab, y_max_main_tab, _ = find_features_of_contours(contours_tab)

            cx_tab_m_text, cy_tab_m_text, x_min_tab_m_text, x_max_tab_m_text, y_min_tab_m_text, y_max_tab_m_text = find_new_features_of_contoures(contours_table_m_text)
            cx_tabl, cy_tabl, x_min_tabl, x_max_tabl, y_min_tabl, y_max_tabl, _ = find_new_features_of_contoures(contours_tab)

            if len(y_min_main_tab) > 0:
                y_down_tabs = []
                y_up_tabs = []

                for i_t in range(len(y_min_main_tab)):
                    y_down_tab = []
                    y_up_tab = []
                    for i_l in range(len(y_min_main_line)):
                        if y_min_main_tab[i_t] > y_min_main_line[i_l] and y_max_main_tab[i_t] > y_min_main_line[i_l] and y_min_main_tab[i_t] > y_max_main_line[i_l] and y_max_main_tab[i_t] > y_min_main_line[i_l]:
                            pass
                        elif y_min_main_tab[i_t] < y_max_main_line[i_l] and y_max_main_tab[i_t] < y_max_main_line[i_l] and y_max_main_tab[i_t] < y_min_main_line[i_l] and y_min_main_tab[i_t] < y_min_main_line[i_l]:
                            pass
                        elif np.abs(y_max_main_line[i_l] - y_min_main_line[i_l]) < 100:
                            pass

                        else:
                            y_up_tab.append(np.min([y_min_main_line[i_l], y_min_main_tab[i_t]]))
                            y_down_tab.append(np.max([y_max_main_line[i_l], y_max_main_tab[i_t]]))

                    if len(y_up_tab) == 0:
                        for v_n in range(len(cx_tab_m_text)):
                            if cx_tabl[i_t] <= x_max_tab_m_text[v_n] and cx_tabl[i_t] >= x_min_tab_m_text[v_n] and cy_tabl[i_t] <= y_max_tab_m_text[v_n] and cy_tabl[i_t] >= y_min_tab_m_text[v_n] and cx_tabl[i_t] != cx_tab_m_text[v_n] and cy_tabl[i_t] != cy_tab_m_text[v_n]:
                                y_up_tabs.append(y_min_tab_m_text[v_n])
                                y_down_tabs.append(y_max_tab_m_text[v_n])
                        # y_up_tabs.append(y_min_main_tab[i_t])
                        # y_down_tabs.append(y_max_main_tab[i_t])
                    else:
                        y_up_tabs.append(np.min(y_up_tab))
                        y_down_tabs.append(np.max(y_down_tab))

            else:
                y_down_tabs = []
                y_up_tabs = []
                pass

            for ii in range(len(y_up_tabs)):
                image_box[y_up_tabs[ii] : y_down_tabs[ii], :, 0] = 7

            image_revised_last[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][0]) : int(boxes[i][1]), :] = image_box[:, :, :]

    else:
        for i in range(len(boxes)):

            image_box = img_comm[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][0]) : int(boxes[i][1]), :]
            image_revised_last[int(boxes[i][2]) : int(boxes[i][3]), int(boxes[i][0]) : int(boxes[i][1]), :] = image_box[:, :, :]

            ##plt.figure(figsize=(20,20))
            ##plt.imshow(image_box[:,:,0])
            ##plt.show()
    return image_revised_last

def get_regions_from_xy_2models_ens(self, img):
    img_org = np.copy(img)

    img_height_h = img_org.shape[0]
    img_width_h = img_org.shape[1]

    model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p_ens)

    gaussian_filter = False
    patches = False
    binary = False

    ratio_x = 1
    ratio_y = 1
    img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

    prediction_regions_long = self.do_prediction(patches, img, model_region)

    prediction_regions_long = resize_image(prediction_regions_long, img_height_h, img_width_h)

    gaussian_filter = False
    patches = True
    binary = False

    ratio_x = 1
    ratio_y = 1.2
    median_blur = False

    img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

    if binary:
        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint16)

    if median_blur:
        img = cv2.medianBlur(img, 5)
    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.uint16)
    prediction_regions_org_y = self.do_prediction(patches, img, model_region)

    prediction_regions_org_y = resize_image(prediction_regions_org_y, img_height_h, img_width_h)

    # plt.imshow(prediction_regions_org[:,:,0])
    # plt.show()
    # sys.exit()
    prediction_regions_org_y = prediction_regions_org_y[:, :, 0]

    mask_zeros_y = (prediction_regions_org_y[:, :] == 0) * 1

    ratio_x = 1.2
    ratio_y = 1
    median_blur = False

    img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

    if binary:
        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint16)

    if median_blur:
        img = cv2.medianBlur(img, 5)
    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.uint16)
    prediction_regions_org = self.do_prediction(patches, img, model_region)

    prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h)

    # plt.imshow(prediction_regions_org[:,:,0])
    # plt.show()
    # sys.exit()
    prediction_regions_org = prediction_regions_org[:, :, 0]

    prediction_regions_org[(prediction_regions_org[:, :] == 1) & (mask_zeros_y[:, :] == 1)] = 0

    prediction_regions_org[(prediction_regions_long[:, :, 0] == 1) & (prediction_regions_org[:, :] == 2)] = 1

    session_region.close()
    del model_region
    del session_region
    gc.collect()

    return prediction_regions_org

def resize_and_enhance_image(self, is_image_enhanced):
    dpi = self.check_dpi()
    img = cv2.imread(self.image_dir)
    img = img.astype(np.uint8)
    # sys.exit()

    print(dpi)

    if dpi < 298:
        if img.shape[0] < 1000:
            img_h_new = int(img.shape[0] * 3)
            img_w_new = int(img.shape[1] * 3)
            if img_h_new < 2800:
                img_h_new = 3000
                img_w_new = int(img.shape[1] / float(img.shape[0]) * 3000)
        elif img.shape[0] >= 1000 and img.shape[0] < 2000:
            img_h_new = int(img.shape[0] * 2)
            img_w_new = int(img.shape[1] * 2)
            if img_h_new < 2800:
                img_h_new = 3000
                img_w_new = int(img.shape[1] / float(img.shape[0]) * 3000)
        else:
            img_h_new = int(img.shape[0] * 1.5)
            img_w_new = int(img.shape[1] * 1.5)
        img_new = resize_image(img, img_h_new, img_w_new)
        image_res = self.predict_enhancement(img_new)
        # cv2.imwrite(os.path.join(self.dir_out, self.f_name) + ".tif",self.image)
        # self.image=self.image.astype(np.uint16)

        # self.scale_x=1
        # self.scale_y=1
        # self.height_org = self.image.shape[0]
        # self.width_org = self.image.shape[1]
        is_image_enhanced = True
    else:
        is_image_enhanced = False
        image_res = np.copy(img)

    return is_image_enhanced, img, image_res

def resize_and_enhance_image_new(self, is_image_enhanced):
    # self.check_dpi()
    img = cv2.imread(self.image_dir)
    img = img.astype(np.uint8)
    # sys.exit()

    image_res = np.copy(img)

    return is_image_enhanced, img, image_res

def get_image_and_scales_deskewd(self, img_deskewd):

    self.image = img_deskewd
    self.image_org = np.copy(self.image)
    self.height_org = self.image.shape[0]
    self.width_org = self.image.shape[1]

    self.img_hight_int = int(self.image.shape[0] * 1)
    self.img_width_int = int(self.image.shape[1] * 1)
    self.scale_y = self.img_hight_int / float(self.image.shape[0])
    self.scale_x = self.img_width_int / float(self.image.shape[1])

    self.image = resize_image(self.image, self.img_hight_int, self.img_width_int)

def extract_drop_capital_13(self, img, patches, cols):

    img_height_h = img.shape[0]
    img_width_h = img.shape[1]
    patches = False

    img = otsu_copy_binary(img)  # otsu_copy(img)
    img = img.astype(np.uint16)

    model_region, session_region = self.start_new_session_and_model(self.model_region_dir_fully_np)

    img_1 = img[: int(img.shape[0] / 3.0), :, :]
    img_2 = img[int(img.shape[0] / 3.0) : int(2 * img.shape[0] / 3.0), :, :]
    img_3 = img[int(2 * img.shape[0] / 3.0) :, :, :]

    # img_1 = otsu_copy_binary(img_1)#otsu_copy(img)
    # img_1 = img_1.astype(np.uint16)

    plt.imshow(img_1)
    plt.show()
    # img_2 = otsu_copy_binary(img_2)#otsu_copy(img)
    # img_2 = img_2.astype(np.uint16)

    plt.imshow(img_2)
    plt.show()
    # img_3 = otsu_copy_binary(img_3)#otsu_copy(img)
    # img_3 = img_3.astype(np.uint16)

    plt.imshow(img_3)
    plt.show()

    prediction_regions_1 = self.do_prediction(patches, img_1, model_region)

    plt.imshow(prediction_regions_1)
    plt.show()

    prediction_regions_2 = self.do_prediction(patches, img_2, model_region)

    plt.imshow(prediction_regions_2)
    plt.show()
    prediction_regions_3 = self.do_prediction(patches, img_3, model_region)

    plt.imshow(prediction_regions_3)
    plt.show()
    prediction_regions = np.zeros((img_height_h, img_width_h))

    prediction_regions[: int(img.shape[0] / 3.0), :] = prediction_regions_1[:, :, 0]
    prediction_regions[int(img.shape[0] / 3.0) : int(2 * img.shape[0] / 3.0), :] = prediction_regions_2[:, :, 0]
    prediction_regions[int(2 * img.shape[0] / 3.0) :, :] = prediction_regions_3[:, :, 0]

    session_region.close()
    del img_1
    del img_2
    del img_3
    del prediction_regions_1
    del prediction_regions_2
    del prediction_regions_3
    del model_region
    del session_region
    del img
    gc.collect()
    return prediction_regions

def extract_only_text_regions(self, img, patches):

    model_region, session_region = self.start_new_session_and_model(self.model_only_text)
    img = otsu_copy_binary(img)  # otsu_copy(img)
    img = img.astype(np.uint8)
    img_org = np.copy(img)

    img_h = img_org.shape[0]
    img_w = img_org.shape[1]

    img = resize_image(img_org, int(img_org.shape[0] * 1), int(img_org.shape[1] * 1))

    prediction_regions1 = self.do_prediction(patches, img, model_region)

    prediction_regions1 = resize_image(prediction_regions1, img_h, img_w)

    # prediction_regions1 = cv2.dilate(prediction_regions1, self.kernel, iterations=4)
    # prediction_regions1 = cv2.erode(prediction_regions1, self.kernel, iterations=7)
    # prediction_regions1 = cv2.dilate(prediction_regions1, self.kernel, iterations=2)

    img = resize_image(img_org, int(img_org.shape[0] * 1), int(img_org.shape[1] * 1))

    prediction_regions2 = self.do_prediction(patches, img, model_region)

    prediction_regions2 = resize_image(prediction_regions2, img_h, img_w)

    # prediction_regions2 = cv2.dilate(prediction_regions2, self.kernel, iterations=2)
    prediction_regions2 = cv2.erode(prediction_regions2, self.kernel, iterations=2)
    prediction_regions2 = cv2.dilate(prediction_regions2, self.kernel, iterations=2)

    # prediction_regions=(  (prediction_regions2[:,:,0]==1) & (prediction_regions1[:,:,0]==1) )
    # prediction_regions=(prediction_regions1[:,:,0]==1)

    session_region.close()
    del model_region
    del session_region
    gc.collect()
    return prediction_regions1[:, :, 0]

def extract_binarization(self, img, patches):

    model_bin, session_bin = self.start_new_session_and_model(self.model_binafrization)

    img_h = img.shape[0]
    img_w = img.shape[1]

    img = resize_image(img, int(img.shape[0] * 1), int(img.shape[1] * 1))

    prediction_regions = self.do_prediction(patches, img, model_bin)

    res = (prediction_regions[:, :, 0] != 0) * 1

    img_fin = np.zeros((res.shape[0], res.shape[1], 3))
    res[:, :][res[:, :] == 0] = 2
    res = res - 1
    res = res * 255
    img_fin[:, :, 0] = res
    img_fin[:, :, 1] = res
    img_fin[:, :, 2] = res

    session_bin.close()
    del model_bin
    del session_bin
    gc.collect()
    # plt.imshow(img_fin[:,:,0])
    # plt.show()
    return img_fin

def get_text_region_contours_and_boxes(self, image):
    rgb_class_of_texts = (1, 1, 1)
    mask_texts = np.all(image == rgb_class_of_texts, axis=-1)

    image = np.repeat(mask_texts[:, :, np.newaxis], 3, axis=2) * 255
    image = image.astype(np.uint8)

    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.kernel)

    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(imgray, 0, 255, 0)

    contours, hirarchy = cv2.findContours(thresh.copy(), cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    main_contours = filter_contours_area_of_image(thresh, contours, hirarchy, max_area=1, min_area=0.00001)
    self.boxes = []

    for jj in range(len(main_contours)):
        x, y, w, h = cv2.boundingRect(main_contours[jj])
        self.boxes.append([x, y, w, h])

    return main_contours

def textline_contours_to_get_slope_correctly(self, textline_mask, img_patch, contour_interest):

    slope_new = 0  # deskew_images(img_patch)

    textline_mask = np.repeat(textline_mask[:, :, np.newaxis], 3, axis=2) * 255

    textline_mask = textline_mask.astype(np.uint8)
    textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_OPEN, self.kernel)
    textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_CLOSE, self.kernel)
    textline_mask = cv2.erode(textline_mask, self.kernel, iterations=1)
    imgray = cv2.cvtColor(textline_mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 0, 255, 0)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel)

    contours, hirarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    main_contours = filter_contours_area_of_image_tables(thresh, contours, hirarchy, max_area=1, min_area=0.003)

    textline_maskt = textline_mask[:, :, 0]
    textline_maskt[textline_maskt != 0] = 1

    peaks_point, _ = seperate_lines(textline_maskt, contour_interest, slope_new)

    mean_dis = np.mean(np.diff(peaks_point))

    len_x = thresh.shape[1]

    slope_lines = []
    contours_slope_new = []

    for kk in range(len(main_contours)):

        if len(main_contours[kk].shape) == 2:
            xminh = np.min(main_contours[kk][:, 0])
            xmaxh = np.max(main_contours[kk][:, 0])

            yminh = np.min(main_contours[kk][:, 1])
            ymaxh = np.max(main_contours[kk][:, 1])
        elif len(main_contours[kk].shape) == 3:
            xminh = np.min(main_contours[kk][:, 0, 0])
            xmaxh = np.max(main_contours[kk][:, 0, 0])

            yminh = np.min(main_contours[kk][:, 0, 1])
            ymaxh = np.max(main_contours[kk][:, 0, 1])

        if ymaxh - yminh <= mean_dis and (xmaxh - xminh) >= 0.3 * len_x:  # xminh>=0.05*len_x and xminh<=0.4*len_x and xmaxh<=0.95*len_x and xmaxh>=0.6*len_x:
            contours_slope_new.append(main_contours[kk])

            rows, cols = thresh.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(main_contours[kk], cv2.DIST_L2, 0, 0.01, 0.01)

            slope_lines.append((vy / vx) / np.pi * 180)

        if len(slope_lines) >= 2:

            slope = np.mean(slope_lines)  # slope_true/np.pi*180
        else:
            slope = 999

    else:
        slope = 0

    return slope


def return_deskew_slope_new(self, img_patch, sigma_des):
    max_x_y = max(img_patch.shape[0], img_patch.shape[1])

    ##img_patch=resize_image(img_patch,max_x_y,max_x_y)

    img_patch_copy = np.zeros((img_patch.shape[0], img_patch.shape[1]))
    img_patch_copy[:, :] = img_patch[:, :]  # img_patch_org[:,:,0]

    img_patch_padded = np.zeros((int(max_x_y * (1.4)), int(max_x_y * (1.4))))

    img_patch_padded_center_p = int(img_patch_padded.shape[0] / 2.0)
    len_x_org_patch_half = int(img_patch_copy.shape[1] / 2.0)
    len_y_org_patch_half = int(img_patch_copy.shape[0] / 2.0)

    img_patch_padded[img_patch_padded_center_p - len_y_org_patch_half : img_patch_padded_center_p - len_y_org_patch_half + img_patch_copy.shape[0], img_patch_padded_center_p - len_x_org_patch_half : img_patch_padded_center_p - len_x_org_patch_half + img_patch_copy.shape[1]] = img_patch_copy[:, :]
    # img_patch_padded[ int( img_patch_copy.shape[0]*(.1)):int( img_patch_copy.shape[0]*(.1))+img_patch_copy.shape[0] , int( img_patch_copy.shape[1]*(.8)):int( img_patch_copy.shape[1]*(.8))+img_patch_copy.shape[1] ]=img_patch_copy[:,:]
    angles = np.linspace(-25, 25, 80)

    res = []
    num_of_peaks = []
    index_cor = []
    var_res = []

    # plt.imshow(img_patch)
    # plt.show()
    indexer = 0
    for rot in angles:
        # print(rot,'rot')
        img_rotated = rotate_image(img_patch_padded, rot)
        img_rotated[img_rotated != 0] = 1

        # plt.imshow(img_rotated)
        # plt.show()

        try:
            neg_peaks, var_spectrum = self.get_standard_deviation_of_summed_textline_patch_along_width(img_rotated, sigma_des, 20.3)
            res_me = np.mean(neg_peaks)
            if res_me == 0:
                res_me = VERY_LARGE_NUMBER
            else:
                pass

            res_num = len(neg_peaks)
        except:
            res_me = VERY_LARGE_NUMBER
            res_num = 0
            var_spectrum = 0
        if isNaN(res_me):
            pass
        else:
            res.append(res_me)
            var_res.append(var_spectrum)
            num_of_peaks.append(res_num)
            index_cor.append(indexer)
        indexer = indexer + 1

    try:
        var_res = np.array(var_res)
        # print(var_res)

        ang_int = angles[np.argmax(var_res)]  # angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
    except:
        ang_int = 0

    if abs(ang_int) > 15:
        angles = np.linspace(-90, -50, 30)
        res = []
        num_of_peaks = []
        index_cor = []
        var_res = []

        # plt.imshow(img_patch)
        # plt.show()
        indexer = 0
        for rot in angles:
            # print(rot,'rot')
            img_rotated = rotate_image(img_patch_padded, rot)
            img_rotated[img_rotated != 0] = 1

            # plt.imshow(img_rotated)
            # plt.show()

            try:
                neg_peaks, var_spectrum = self.get_standard_deviation_of_summed_textline_patch_along_width(img_rotated, sigma_des, 20.3)
                res_me = np.mean(neg_peaks)
                if res_me == 0:
                    res_me = VERY_LARGE_NUMBER
                else:
                    pass

                res_num = len(neg_peaks)
            except:
                res_me = VERY_LARGE_NUMBER
                res_num = 0
                var_spectrum = 0
            if isNaN(res_me):
                pass
            else:
                res.append(res_me)
                var_res.append(var_spectrum)
                num_of_peaks.append(res_num)
                index_cor.append(indexer)
            indexer = indexer + 1

        try:
            var_res = np.array(var_res)
            # print(var_res)

            ang_int = angles[np.argmax(var_res)]  # angels_sorted[arg_final]#angels[arg_sort_early[arg_sort[arg_final]]]#angels[arg_fin]
        except:
            ang_int = 0

    return ang_int

def get_slopes_and_deskew(self, contours, textline_mask_tot):

    slope_biggest = 0  # return_deskew_slop(img_int_p,sigma_des, dir_of_all=self.dir_of_all, f_name=self.f_name)

    num_cores = cpu_count()
    q = Queue()
    poly = Queue()
    box_sub = Queue()

    processes = []
    nh = np.linspace(0, len(self.boxes), num_cores + 1)

    for i in range(num_cores):
        boxes_per_process = self.boxes[int(nh[i]) : int(nh[i + 1])]
        contours_per_process = contours[int(nh[i]) : int(nh[i + 1])]
        processes.append(Process(target=self.do_work_of_slopes, args=(q, poly, box_sub, boxes_per_process, textline_mask_tot, contours_per_process)))

    for i in range(num_cores):
        processes[i].start()

    self.slopes = []
    self.all_found_texline_polygons = []
    self.boxes = []

    for i in range(num_cores):
        slopes_for_sub_process = q.get(True)
        boxes_for_sub_process = box_sub.get(True)
        polys_for_sub_process = poly.get(True)

        for j in range(len(slopes_for_sub_process)):
            self.slopes.append(slopes_for_sub_process[j])
            self.all_found_texline_polygons.append(polys_for_sub_process[j])
            self.boxes.append(boxes_for_sub_process[j])

    for i in range(num_cores):
        processes[i].join()


def write_into_page_xml_only_textlines(self, contours, page_coord, all_found_texline_polygons, all_box_coord, dir_of_image):

    found_polygons_text_region = contours

    # create the file structure
    data = ET.Element("PcGts")

    data.set("xmlns", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")
    data.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    data.set("xsi:schemaLocation", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")

    metadata = ET.SubElement(data, "Metadata")

    author = ET.SubElement(metadata, "Creator")
    author.text = "SBB_QURATOR"

    created = ET.SubElement(metadata, "Created")
    created.text = "2019-06-17T18:15:12"

    changetime = ET.SubElement(metadata, "LastChange")
    changetime.text = "2019-06-17T18:15:12"

    page = ET.SubElement(data, "Page")

    page.set("imageFilename", self.image_dir)
    page.set("imageHeight", str(self.height_org))
    page.set("imageWidth", str(self.width_org))
    page.set("type", "content")
    page.set("readingDirection", "left-to-right")
    page.set("textLineOrder", "top-to-bottom")

    page_print_sub = ET.SubElement(page, "PrintSpace")
    coord_page = ET.SubElement(page_print_sub, "Coords")
    points_page_print = ""

    for lmm in range(len(self.cont_page[0])):
        if len(self.cont_page[0][lmm]) == 2:
            points_page_print = points_page_print + str(int((self.cont_page[0][lmm][0]) / self.scale_x))
            points_page_print = points_page_print + ","
            points_page_print = points_page_print + str(int((self.cont_page[0][lmm][1]) / self.scale_y))
        else:
            points_page_print = points_page_print + str(int((self.cont_page[0][lmm][0][0]) / self.scale_x))
            points_page_print = points_page_print + ","
            points_page_print = points_page_print + str(int((self.cont_page[0][lmm][0][1]) / self.scale_y))

        if lmm < (len(self.cont_page[0]) - 1):
            points_page_print = points_page_print + " "
    coord_page.set("points", points_page_print)

    if len(contours) > 0:

        id_indexer = 0
        id_indexer_l = 0

        for mm in range(len(found_polygons_text_region)):
            textregion = ET.SubElement(page, "TextRegion")

            textregion.set("id", "r" + str(id_indexer))
            id_indexer += 1

            textregion.set("type", "paragraph")
            # if mm==0:
            #    textregion.set('type','header')
            # else:
            #    textregion.set('type','paragraph')
            coord_text = ET.SubElement(textregion, "Coords")

            points_co = ""
            for lmm in range(len(found_polygons_text_region[mm])):
                if len(found_polygons_text_region[mm][lmm]) == 2:
                    points_co = points_co + str(int((found_polygons_text_region[mm][lmm][0] + page_coord[2]) / self.scale_x))
                    points_co = points_co + ","
                    points_co = points_co + str(int((found_polygons_text_region[mm][lmm][1] + page_coord[0]) / self.scale_y))
                else:
                    points_co = points_co + str(int((found_polygons_text_region[mm][lmm][0][0] + page_coord[2]) / self.scale_x))
                    points_co = points_co + ","
                    points_co = points_co + str(int((found_polygons_text_region[mm][lmm][0][1] + page_coord[0]) / self.scale_y))

                if lmm < (len(found_polygons_text_region[mm]) - 1):
                    points_co = points_co + " "
            # print(points_co)
            coord_text.set("points", points_co)

            for j in range(len(all_found_texline_polygons[mm])):

                textline = ET.SubElement(textregion, "TextLine")

                textline.set("id", "l" + str(id_indexer_l))

                id_indexer_l += 1

                coord = ET.SubElement(textline, "Coords")

                texteq = ET.SubElement(textline, "TextEquiv")

                uni = ET.SubElement(texteq, "Unicode")
                uni.text = " "

                # points = ET.SubElement(coord, 'Points')

                points_co = ""
                for l in range(len(all_found_texline_polygons[mm][j])):
                    # point = ET.SubElement(coord, 'Point')

                    # point.set('x',str(found_polygons[j][l][0]))
                    # point.set('y',str(found_polygons[j][l][1]))
                    if len(all_found_texline_polygons[mm][j][l]) == 2:
                        points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][0] + page_coord[2]) / self.scale_x))
                        points_co = points_co + ","
                        points_co = points_co + str(int((all_found_texline_polygons[mm][j][l][0][1] + page_coord[0]) / self.scale_y))

                    if l < (len(all_found_texline_polygons[mm][j]) - 1):
                        points_co = points_co + " "
                # print(points_co)
                coord.set("points", points_co)

            texteqreg = ET.SubElement(textregion, "TextEquiv")

            unireg = ET.SubElement(texteqreg, "Unicode")
            unireg.text = " "

    # print(dir_of_image)
    print(self.f_name)
    # print(os.path.join(dir_of_image, self.f_name) + ".xml")
    tree = ET.ElementTree(data)
    tree.write(os.path.join(dir_of_image, self.f_name) + ".xml")

def return_teilwiese_deskewed_lines(self, text_regions_p, textline_rotated):

    kernel = np.ones((5, 5), np.uint8)
    textline_rotated = cv2.erode(textline_rotated, kernel, iterations=1)

    textline_rotated_new = np.zeros(textline_rotated.shape)
    rgb_m = 1
    rgb_h = 2

    cnt_m, boxes_m = return_contours_of_interested_region_and_bounding_box(text_regions_p, rgb_m)
    cnt_h, boxes_h = return_contours_of_interested_region_and_bounding_box(text_regions_p, rgb_h)

    areas_cnt_m = np.array([cv2.contourArea(cnt_m[j]) for j in range(len(cnt_m))])

    argmax = np.argmax(areas_cnt_m)

    # plt.imshow(textline_rotated[ boxes_m[argmax][1]:boxes_m[argmax][1]+boxes_m[argmax][3] ,boxes_m[argmax][0]:boxes_m[argmax][0]+boxes_m[argmax][2]])
    # plt.show()

    for argmax in range(len(boxes_m)):

        textline_text_region = textline_rotated[boxes_m[argmax][1] : boxes_m[argmax][1] + boxes_m[argmax][3], boxes_m[argmax][0] : boxes_m[argmax][0] + boxes_m[argmax][2]]

        textline_text_region_revised = self.seperate_lines_new(textline_text_region, 0)
        # except:
        #    textline_text_region_revised=textline_rotated[ boxes_m[argmax][1]:boxes_m[argmax][1]+boxes_m[argmax][3] ,boxes_m[argmax][0]:boxes_m[argmax][0]+boxes_m[argmax][2]  ]
        textline_rotated_new[boxes_m[argmax][1] : boxes_m[argmax][1] + boxes_m[argmax][3], boxes_m[argmax][0] : boxes_m[argmax][0] + boxes_m[argmax][2]] = textline_text_region_revised[:, :]

    # textline_rotated_new[textline_rotated_new>0]=1
    # textline_rotated_new[textline_rotated_new<0]=0
    # plt.imshow(textline_rotated_new)
    # plt.show()

def get_regions_from_xy_neu(self, img):
    img_org = np.copy(img)

    img_height_h = img_org.shape[0]
    img_width_h = img_org.shape[1]

    model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p)

    gaussian_filter = False
    patches = True
    binary = True

    ratio_x = 1
    ratio_y = 1
    median_blur = False

    img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

    if binary:
        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint16)

    if median_blur:
        img = cv2.medianBlur(img, 5)
    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.uint16)
    prediction_regions_org = self.do_prediction(patches, img, model_region)

    prediction_regions_org = resize_image(prediction_regions_org, img_height_h, img_width_h)

    # plt.imshow(prediction_regions_org[:,:,0])
    # plt.show()
    # sys.exit()
    prediction_regions_org = prediction_regions_org[:, :, 0]

    gaussian_filter = False
    patches = False
    binary = False

    ratio_x = 1
    ratio_y = 1
    median_blur = False

    img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

    if binary:
        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint16)

    if median_blur:
        img = cv2.medianBlur(img, 5)
        img = cv2.medianBlur(img, 5)
    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.uint16)
    prediction_regions_orgt = self.do_prediction(patches, img, model_region)

    prediction_regions_orgt = resize_image(prediction_regions_orgt, img_height_h, img_width_h)

    # plt.imshow(prediction_regions_orgt[:,:,0])
    # plt.show()
    # sys.exit()
    prediction_regions_orgt = prediction_regions_orgt[:, :, 0]

    mask_texts_longshot = (prediction_regions_orgt[:, :] == 1) * 1

    mask_texts_longshot = np.uint8(mask_texts_longshot)
    # mask_texts_longshot = cv2.dilate(mask_texts_longshot[:,:], self.kernel, iterations=2)

    pixel_img = 1
    polygons_of_only_texts_longshot = return_contours_of_interested_region(mask_texts_longshot, pixel_img)

    longshot_true = np.zeros(mask_texts_longshot.shape)
    # text_regions_p_true[:,:]=text_regions_p_1[:,:]

    longshot_true = cv2.fillPoly(longshot_true, pts=polygons_of_only_texts_longshot, color=(1, 1, 1))

    # plt.imshow(longshot_true)
    # plt.show()

    gaussian_filter = False
    patches = False
    binary = False

    ratio_x = 1
    ratio_y = 1
    median_blur = False

    img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

    one_third_upper_ny = int(img.shape[0] / 3.0)

    img = img[0:one_third_upper_ny, :, :]

    if binary:
        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint16)

    if median_blur:
        img = cv2.medianBlur(img, 5)

    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.uint16)
    prediction_regions_longshot_one_third = self.do_prediction(patches, img, model_region)

    prediction_regions_longshot_one_third = resize_image(prediction_regions_longshot_one_third, one_third_upper_ny, img_width_h)

    img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))
    img = img[one_third_upper_ny : int(2 * one_third_upper_ny), :, :]

    if binary:
        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint16)

    if median_blur:
        img = cv2.medianBlur(img, 5)

    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.uint16)
    prediction_regions_longshot_one_third_middle = self.do_prediction(patches, img, model_region)

    prediction_regions_longshot_one_third_middle = resize_image(prediction_regions_longshot_one_third_middle, one_third_upper_ny, img_width_h)

    img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))
    img = img[int(2 * one_third_upper_ny) :, :, :]

    if binary:
        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint16)

    if median_blur:
        img = cv2.medianBlur(img, 5)

    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.uint16)
    prediction_regions_longshot_one_third_down = self.do_prediction(patches, img, model_region)

    prediction_regions_longshot_one_third_down = resize_image(prediction_regions_longshot_one_third_down, img_height_h - int(2 * one_third_upper_ny), img_width_h)

    # plt.imshow(prediction_regions_org[:,:,0])
    # plt.show()
    # sys.exit()
    prediction_regions_longshot = np.zeros((img_height_h, img_width_h))

    # prediction_regions_longshot=prediction_regions_longshot[:,:,0]

    # prediction_regions_longshot[0:one_third_upper_ny,:]=prediction_regions_longshot_one_third[:,:,0]
    # prediction_regions_longshot[one_third_upper_ny:int(2*one_third_upper_ny):,:]=prediction_regions_longshot_one_third_middle[:,:,0]
    # prediction_regions_longshot[int(2*one_third_upper_ny):,:]=prediction_regions_longshot_one_third_down[:,:,0]

    prediction_regions_longshot = longshot_true[:, :]
    # plt.imshow(prediction_regions_longshot)
    # plt.show()

    gaussian_filter = False
    patches = True
    binary = False

    ratio_x = 1  # 1.1
    ratio_y = 1
    median_blur = False

    # img= resize_image(img_org, int(img_org.shape[0]*0.8), int(img_org.shape[1]*1.6))
    img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

    if binary:
        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint16)

    if median_blur:
        img = cv2.medianBlur(img, 5)
    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.uint16)

    prediction_regions = self.do_prediction(patches, img, model_region)
    text_region1 = resize_image(prediction_regions, img_height_h, img_width_h)

    # plt.imshow(text_region1[:,:,0])
    # plt.show()
    ratio_x = 1
    ratio_y = 1.2  # 1.3
    binary = False
    median_blur = False

    img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

    if binary:
        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint16)

    if median_blur:
        img = cv2.medianBlur(img, 5)
    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.uint16)

    prediction_regions = self.do_prediction(patches, img, model_region)
    text_region2 = resize_image(prediction_regions, img_height_h, img_width_h)

    # plt.imshow(text_region2[:,:,0])
    # plt.show()
    session_region.close()
    del model_region
    del session_region
    gc.collect()

    # text_region1=text_region1[:,:,0]
    # text_region2=text_region2[:,:,0]

    # text_region1[(text_region1[:,:]==2) & (text_region2[:,:]==1)]=1

    mask_zeros_from_1 = (text_region2[:, :, 0] == 0) * 1
    # mask_text_from_1=(text_region1[:,:,0]==1)*1

    mask_img_text_region1 = (text_region1[:, :, 0] == 2) * 1
    text_region2_1st_channel = text_region1[:, :, 0]

    text_region2_1st_channel[mask_zeros_from_1 == 1] = 0

    ##text_region2_1st_channel[mask_img_text_region1[:,:]==1]=2
    # text_region2_1st_channel[(mask_text_from_1==1) & (text_region2_1st_channel==2)]=1

    mask_lines1 = (text_region1[:, :, 0] == 3) * 1
    mask_lines2 = (text_region2[:, :, 0] == 3) * 1

    mask_lines2[mask_lines1[:, :] == 1] = 1

    # plt.imshow(text_region2_1st_channel)
    # plt.show()

    text_region2_1st_channel = cv2.erode(text_region2_1st_channel[:, :], self.kernel, iterations=4)

    # plt.imshow(text_region2_1st_channel)
    # plt.show()

    text_region2_1st_channel = cv2.dilate(text_region2_1st_channel[:, :], self.kernel, iterations=4)

    text_region2_1st_channel[mask_lines2[:, :] == 1] = 3

    # text_region2_1st_channel[ (prediction_regions_org[:,:]==1) & (text_region2_1st_channel[:,:]==2)]=1

    # only in the case of model 3

    text_region2_1st_channel[(prediction_regions_longshot[:, :] == 1) & (text_region2_1st_channel[:, :] == 2)] = 1

    text_region2_1st_channel[(prediction_regions_org[:, :] == 2) & (text_region2_1st_channel[:, :] == 0)] = 2

    # text_region2_1st_channel[prediction_regions_org[:,:]==0]=0

    # plt.imshow(text_region2_1st_channel)
    # plt.show()

    # text_region2_1st_channel[:,:400]=0

    mask_texts_only = (text_region2_1st_channel[:, :] == 1) * 1

    mask_images_only = (text_region2_1st_channel[:, :] == 2) * 1

    mask_lines_only = (text_region2_1st_channel[:, :] == 3) * 1

    pixel_img = 1
    polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only, pixel_img)

    polygons_of_only_images = return_contours_of_interested_region(mask_images_only, pixel_img)

    polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only, pixel_img)

    text_regions_p_true = np.zeros(text_region2_1st_channel.shape)
    # text_regions_p_true[:,:]=text_regions_p_1[:,:]

    text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts=polygons_of_only_lines, color=(3, 3, 3))

    text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts=polygons_of_only_images, color=(2, 2, 2))

    text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts=polygons_of_only_texts, color=(1, 1, 1))

    ##print(np.unique(text_regions_p_true))

    # text_regions_p_true_3d=np.repeat(text_regions_p_1[:, :, np.newaxis], 3, axis=2)
    # text_regions_p_true_3d=text_regions_p_true_3d.astype(np.uint8)

    return text_regions_p_true  # text_region2_1st_channel

def get_regions_from_xy(self, img):
    img_org = np.copy(img)

    img_height_h = img_org.shape[0]
    img_width_h = img_org.shape[1]

    model_region, session_region = self.start_new_session_and_model(self.model_region_dir_p)

    gaussian_filter = False
    patches = True
    binary = True

    ratio_x = 1
    ratio_y = 1
    median_blur = False

    if binary:
        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint16)

    if median_blur:
        img = cv2.medianBlur(img, 5)

    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.uint16)
    prediction_regions_org = self.do_prediction(patches, img, model_region)

    ###plt.imshow(prediction_regions_org[:,:,0])
    ###plt.show()
    ##sys.exit()
    prediction_regions_org = prediction_regions_org[:, :, 0]

    gaussian_filter = False
    patches = True
    binary = False

    ratio_x = 1.1
    ratio_y = 1
    median_blur = False

    # img= resize_image(img_org, int(img_org.shape[0]*0.8), int(img_org.shape[1]*1.6))
    img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

    if binary:
        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint16)

    if median_blur:
        img = cv2.medianBlur(img, 5)
    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.uint16)

    prediction_regions = self.do_prediction(patches, img, model_region)
    text_region1 = resize_image(prediction_regions, img_height_h, img_width_h)

    ratio_x = 1
    ratio_y = 1.1
    binary = False
    median_blur = False

    img = resize_image(img_org, int(img_org.shape[0] * ratio_y), int(img_org.shape[1] * ratio_x))

    if binary:
        img = otsu_copy_binary(img)  # otsu_copy(img)
        img = img.astype(np.uint16)

    if median_blur:
        img = cv2.medianBlur(img, 5)
    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.uint16)

    prediction_regions = self.do_prediction(patches, img, model_region)
    text_region2 = resize_image(prediction_regions, img_height_h, img_width_h)

    session_region.close()
    del model_region
    del session_region
    gc.collect()

    mask_zeros_from_1 = (text_region1[:, :, 0] == 0) * 1
    # mask_text_from_1=(text_region1[:,:,0]==1)*1

    mask_img_text_region1 = (text_region1[:, :, 0] == 2) * 1
    text_region2_1st_channel = text_region2[:, :, 0]

    text_region2_1st_channel[mask_zeros_from_1 == 1] = 0

    text_region2_1st_channel[mask_img_text_region1[:, :] == 1] = 2
    # text_region2_1st_channel[(mask_text_from_1==1) & (text_region2_1st_channel==2)]=1

    mask_lines1 = (text_region1[:, :, 0] == 3) * 1
    mask_lines2 = (text_region2[:, :, 0] == 3) * 1

    mask_lines2[mask_lines1[:, :] == 1] = 1

    ##plt.imshow(text_region2_1st_channel)
    ##plt.show()

    text_region2_1st_channel = cv2.erode(text_region2_1st_channel[:, :], self.kernel, iterations=5)

    ##plt.imshow(text_region2_1st_channel)
    ##plt.show()

    text_region2_1st_channel = cv2.dilate(text_region2_1st_channel[:, :], self.kernel, iterations=5)

    text_region2_1st_channel[mask_lines2[:, :] == 1] = 3

    text_region2_1st_channel[(prediction_regions_org[:, :] == 1) & (text_region2_1st_channel[:, :] == 2)] = 1
    text_region2_1st_channel[prediction_regions_org[:, :] == 3] = 3

    ##plt.imshow(text_region2_1st_channel)
    ##plt.show()
    return text_region2_1st_channel

def do_work_of_textline_seperation(self, queue_of_all_params, polygons_per_process, index_polygons_per_process, con_par_org, textline_mask_tot, mask_texts_only, num_col, scale_par, boxes_text):

    textregions_cnt_tot_per_process = []
    textlines_cnt_tot_per_process = []
    index_polygons_per_process_per_process = []
    polygons_per_par_process_per_process = []
    textline_cnt_seperated = np.zeros(textline_mask_tot.shape)
    for iiii in range(len(polygons_per_process)):
        # crop_img,crop_coor=crop_image_inside_box(boxes_text[mv],image_page_rotated)
        # arg_max=np.argmax(areas_cnt_only_text)
        textregions_cnt_tot_per_process.append(polygons_per_process[iiii] / scale_par)
        textline_region_in_image = np.zeros(textline_mask_tot.shape)
        cnt_o_t_max = polygons_per_process[iiii]

        x, y, w, h = cv2.boundingRect(cnt_o_t_max)

        mask_biggest = np.zeros(mask_texts_only.shape)
        mask_biggest = cv2.fillPoly(mask_biggest, pts=[cnt_o_t_max], color=(1, 1, 1))

        mask_region_in_patch_region = mask_biggest[y : y + h, x : x + w]

        textline_biggest_region = mask_biggest * textline_mask_tot

        textline_rotated_seperated = self.seperate_lines_new2(textline_biggest_region[y : y + h, x : x + w], 0, num_col)

        # new line added
        ##print(np.shape(textline_rotated_seperated),np.shape(mask_biggest))
        textline_rotated_seperated[mask_region_in_patch_region[:, :] != 1] = 0
        # till here

        textline_cnt_seperated[y : y + h, x : x + w] = textline_rotated_seperated
        textline_region_in_image[y : y + h, x : x + w] = textline_rotated_seperated

        # plt.imshow(textline_region_in_image)
        # plt.show()

        # plt.imshow(textline_cnt_seperated)
        # plt.show()

        pixel_img = 1
        cnt_textlines_in_image = return_contours_of_interested_textline(textline_region_in_image, pixel_img)

        textlines_cnt_per_region = []
        for jjjj in range(len(cnt_textlines_in_image)):
            mask_biggest2 = np.zeros(mask_texts_only.shape)
            mask_biggest2 = cv2.fillPoly(mask_biggest2, pts=[cnt_textlines_in_image[jjjj]], color=(1, 1, 1))
            if num_col + 1 == 1:
                mask_biggest2 = cv2.dilate(mask_biggest2, self.kernel, iterations=5)
            else:

                mask_biggest2 = cv2.dilate(mask_biggest2, self.kernel, iterations=4)

            pixel_img = 1
            cnt_textlines_in_image_ind = return_contours_of_interested_textline(mask_biggest2, pixel_img)

            try:
                textlines_cnt_per_region.append(cnt_textlines_in_image_ind[0] / scale_par)
            except:
                pass
            # print(len(cnt_textlines_in_image_ind))

            # plt.imshow(mask_biggest2)
            # plt.show()
        textlines_cnt_tot_per_process.append(textlines_cnt_per_region)
        index_polygons_per_process_per_process.append(index_polygons_per_process[iiii])
        polygons_per_par_process_per_process.append(con_par_org[iiii])

    queue_of_all_params.put([index_polygons_per_process_per_process, polygons_per_par_process_per_process, textregions_cnt_tot_per_process, textlines_cnt_tot_per_process])


def seperate_lines_new(img_path, thetha, num_col, dir_of_all, f_name):

    if num_col == 1:
        num_patches = int(img_path.shape[1] / 200.0)
    else:
        num_patches = int(img_path.shape[1] / 100.0)
    # num_patches=int(img_path.shape[1]/200.)
    if num_patches == 0:
        num_patches = 1
    (h, w) = img_path.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
    x_d = M[0, 2]
    y_d = M[1, 2]

    thetha = thetha / 180.0 * np.pi
    rotation_matrix = np.array([[np.cos(thetha), -np.sin(thetha)], [np.sin(thetha), np.cos(thetha)]])

    x_min_cont = 0
    x_max_cont = img_path.shape[1]
    y_min_cont = 0
    y_max_cont = img_path.shape[0]

    xv = np.linspace(x_min_cont, x_max_cont, 1000)

    mada_n = img_path.sum(axis=1)

    ##plt.plot(mada_n)
    ##plt.show()
    first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

    y = mada_n[:]  # [first_nonzero:last_nonzero]
    y_help = np.zeros(len(y) + 40)
    y_help[20 : len(y) + 20] = y
    x = np.array(range(len(y)))

    peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
    if len(peaks_real) <= 2 and len(peaks_real) > 1:
        sigma_gaus = 10
    else:
        sigma_gaus = 6

    z = gaussian_filter1d(y_help, sigma_gaus)
    zneg_rev = -y_help + np.max(y_help)
    zneg = np.zeros(len(zneg_rev) + 40)
    zneg[20 : len(zneg_rev) + 20] = zneg_rev
    zneg = gaussian_filter1d(zneg, sigma_gaus)

    peaks, _ = find_peaks(z, height=0)
    peaks_neg, _ = find_peaks(zneg, height=0)

    for nn in range(len(peaks_neg)):
        if peaks_neg[nn] > len(z) - 1:
            peaks_neg[nn] = len(z) - 1
        if peaks_neg[nn] < 0:
            peaks_neg[nn] = 0

    diff_peaks = np.abs(np.diff(peaks_neg))
    cut_off = 20
    peaks_neg_true = []
    forest = []

    for i in range(len(peaks_neg)):
        if i == 0:
            forest.append(peaks_neg[i])
        if i < (len(peaks_neg) - 1):
            if diff_peaks[i] <= cut_off:
                forest.append(peaks_neg[i + 1])
            if diff_peaks[i] > cut_off:
                # print(forest[np.argmin(z[forest]) ] )
                if not isNaN(forest[np.argmin(z[forest])]):
                    # print(len(z),forest)
                    peaks_neg_true.append(forest[np.argmin(z[forest])])
                forest = []
                forest.append(peaks_neg[i + 1])
        if i == (len(peaks_neg) - 1):
            # print(print(forest[np.argmin(z[forest]) ] ))
            if not isNaN(forest[np.argmin(z[forest])]):

                peaks_neg_true.append(forest[np.argmin(z[forest])])

    peaks_neg_true = np.array(peaks_neg_true)

    """
    #plt.figure(figsize=(40,40))
    #plt.subplot(1,2,1)
    #plt.title('Textline segmentation von Textregion')
    #plt.imshow(img_path)
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.subplot(1,2,2)
    #plt.title('Dichte entlang X')
    #base = pyplot.gca().transData
    #rot = transforms.Affine2D().rotate_deg(90)
    #plt.plot(zneg,np.array(range(len(zneg))))
    #plt.plot(zneg[peaks_neg_true],peaks_neg_true,'*')
    #plt.gca().invert_yaxis()

    #plt.xlabel('Dichte')
    #plt.ylabel('Y')
    ##plt.plot([0,len(y)], [grenze,grenze])
    #plt.show()
    """

    peaks_neg_true = peaks_neg_true - 20 - 20
    peaks = peaks - 20

    # dis_up=peaks_neg_true[14]-peaks_neg_true[0]
    # dis_down=peaks_neg_true[18]-peaks_neg_true[14]

    img_patch_ineterst = img_path[:, :]  # [peaks_neg_true[14]-dis_up:peaks_neg_true[15]+dis_down ,:]

    ##plt.imshow(img_patch_ineterst)
    ##plt.show()

    length_x = int(img_path.shape[1] / float(num_patches))
    margin = int(0.04 * length_x)

    width_mid = length_x - 2 * margin

    nxf = img_path.shape[1] / float(width_mid)

    if nxf > int(nxf):
        nxf = int(nxf) + 1
    else:
        nxf = int(nxf)

    slopes_tile_wise = []
    for i in range(nxf):
        if i == 0:
            index_x_d = i * width_mid
            index_x_u = index_x_d + length_x
        elif i > 0:
            index_x_d = i * width_mid
            index_x_u = index_x_d + length_x

        if index_x_u > img_path.shape[1]:
            index_x_u = img_path.shape[1]
            index_x_d = img_path.shape[1] - length_x

        # img_patch = img[index_y_d:index_y_u, index_x_d:index_x_u, :]
        img_xline = img_patch_ineterst[:, index_x_d:index_x_u]

        sigma = 2
        try:
            slope_xline = return_deskew_slop(img_xline, sigma, dir_of_all=dir_of_all, f_name=f_name)
        except:
            slope_xline = 0
        slopes_tile_wise.append(slope_xline)
        # print(slope_xline,'xlineeee')
        img_line_rotated = rotate_image(img_xline, slope_xline)
        img_line_rotated[:, :][img_line_rotated[:, :] != 0] = 1

    """

    xline=np.linspace(0,img_path.shape[1],nx)
    slopes_tile_wise=[]

    for ui in range( nx-1 ):
        img_xline=img_patch_ineterst[:,int(xline[ui]):int(xline[ui+1])]


        ##plt.imshow(img_xline)
        ##plt.show()

        sigma=3
        try:
            slope_xline=return_deskew_slop(img_xline,sigma, dir_of_all=self.dir_of_all, f_name=self.f_name)
        except:
            slope_xline=0
        slopes_tile_wise.append(slope_xline)
        print(slope_xline,'xlineeee')
        img_line_rotated=rotate_image(img_xline,slope_xline)

        ##plt.imshow(img_line_rotated)
        ##plt.show()
    """

    # dis_up=peaks_neg_true[14]-peaks_neg_true[0]
    # dis_down=peaks_neg_true[18]-peaks_neg_true[14]

    img_patch_ineterst = img_path[:, :]  # [peaks_neg_true[14]-dis_up:peaks_neg_true[14]+dis_down ,:]

    img_patch_ineterst_revised = np.zeros(img_patch_ineterst.shape)

    for i in range(nxf):
        if i == 0:
            index_x_d = i * width_mid
            index_x_u = index_x_d + length_x
        elif i > 0:
            index_x_d = i * width_mid
            index_x_u = index_x_d + length_x

        if index_x_u > img_path.shape[1]:
            index_x_u = img_path.shape[1]
            index_x_d = img_path.shape[1] - length_x

        img_xline = img_patch_ineterst[:, index_x_d:index_x_u]

        img_int = np.zeros((img_xline.shape[0], img_xline.shape[1]))
        img_int[:, :] = img_xline[:, :]  # img_patch_org[:,:,0]

        img_resized = np.zeros((int(img_int.shape[0] * (1.2)), int(img_int.shape[1] * (3))))

        img_resized[int(img_int.shape[0] * (0.1)) : int(img_int.shape[0] * (0.1)) + img_int.shape[0], int(img_int.shape[1] * (1)) : int(img_int.shape[1] * (1)) + img_int.shape[1]] = img_int[:, :]
        ##plt.imshow(img_xline)
        ##plt.show()
        img_line_rotated = rotate_image(img_resized, slopes_tile_wise[i])
        img_line_rotated[:, :][img_line_rotated[:, :] != 0] = 1

        img_patch_seperated = seperate_lines_new_inside_teils(img_line_rotated, 0)

        ##plt.imshow(img_patch_seperated)
        ##plt.show()
        img_patch_seperated_returned = rotate_image(img_patch_seperated, -slopes_tile_wise[i])
        img_patch_seperated_returned[:, :][img_patch_seperated_returned[:, :] != 0] = 1

        img_patch_seperated_returned_true_size = img_patch_seperated_returned[int(img_int.shape[0] * (0.1)) : int(img_int.shape[0] * (0.1)) + img_int.shape[0], int(img_int.shape[1] * (1)) : int(img_int.shape[1] * (1)) + img_int.shape[1]]

        img_patch_seperated_returned_true_size = img_patch_seperated_returned_true_size[:, margin : length_x - margin]
        img_patch_ineterst_revised[:, index_x_d + margin : index_x_u - margin] = img_patch_seperated_returned_true_size

    """
    for ui in range( nx-1 ):
        img_xline=img_patch_ineterst[:,int(xline[ui]):int(xline[ui+1])]


        img_int=np.zeros((img_xline.shape[0],img_xline.shape[1]))
        img_int[:,:]=img_xline[:,:]#img_patch_org[:,:,0]

        img_resized=np.zeros((int( img_int.shape[0]*(1.2) ) , int( img_int.shape[1]*(3) ) ))

        img_resized[ int( img_int.shape[0]*(.1)):int( img_int.shape[0]*(.1))+img_int.shape[0] , int( img_int.shape[1]*(1)):int( img_int.shape[1]*(1))+img_int.shape[1] ]=img_int[:,:]
        ##plt.imshow(img_xline)
        ##plt.show()
        img_line_rotated=rotate_image(img_resized,slopes_tile_wise[ui])


        #img_patch_seperated = seperate_lines_new_inside_teils(img_line_rotated,0)

        img_patch_seperated = seperate_lines_new_inside_teils(img_line_rotated,0)

        img_patch_seperated_returned=rotate_image(img_patch_seperated,-slopes_tile_wise[ui])
        ##plt.imshow(img_patch_seperated)
        ##plt.show()
        print(img_patch_seperated_returned.shape)
        #plt.imshow(img_patch_seperated_returned[ int( img_int.shape[0]*(.1)):int( img_int.shape[0]*(.1))+img_int.shape[0] , int( img_int.shape[1]*(1)):int( img_int.shape[1]*(1))+img_int.shape[1] ])
        #plt.show()

        img_patch_ineterst_revised[:,int(xline[ui]):int(xline[ui+1])]=img_patch_seperated_returned[ int( img_int.shape[0]*(.1)):int( img_int.shape[0]*(.1))+img_int.shape[0] , int( img_int.shape[1]*(1)):int( img_int.shape[1]*(1))+img_int.shape[1] ]


    """

    # print(img_patch_ineterst_revised.shape,np.unique(img_patch_ineterst_revised))
    ##plt.imshow(img_patch_ineterst_revised)
    ##plt.show()
    return img_patch_ineterst_revised

def return_contours_of_interested_region_and_bounding_box(region_pre_p, pixel):

    # pixels of images are identified by 5
    cnts_images = (region_pre_p[:, :, 0] == pixel) * 1
    cnts_images = cnts_images.astype(np.uint8)
    cnts_images = np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
    imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hiearchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=1, min_area=0.0003)

    boxes = []

    for jj in range(len(contours_imgs)):
        x, y, w, h = cv2.boundingRect(contours_imgs[jj])
        boxes.append([int(x), int(y), int(w), int(h)])
    return contours_imgs, boxes

def return_bonding_box_of_contours(cnts):
    boxes_tot = []
    for i in range(len(cnts)):
        x, y, w, h = cv2.boundingRect(cnts[i])

        box = [x, y, w, h]
        boxes_tot.append(box)
    return boxes_tot

def find_features_of_contours(contours_main):

    areas_main = np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
    M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
    cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
    cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
    x_min_main = np.array([np.min(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])
    x_max_main = np.array([np.max(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

    y_min_main = np.array([np.min(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
    y_max_main = np.array([np.max(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])

    return y_min_main, y_max_main, areas_main

def filter_contours_area_of_image_interiors(image, contours, hirarchy, max_area, min_area):
    found_polygons_early = list()

    jv = 0
    for c in contours:
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue

        polygon = geometry.Polygon([point[0] for point in c])
        area = polygon.area
        if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]) and hirarchy[0][jv][3] != -1:
            # print(c[0][0][1])
            found_polygons_early.append(np.array([point for point in polygon.exterior.coords], dtype=np.uint))
        jv += 1
    return found_polygons_early

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

