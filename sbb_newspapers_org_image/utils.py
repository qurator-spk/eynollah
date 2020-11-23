import numpy as np
from shapely import geometry
import cv2
import imutils

def filter_contours_area_of_image(image, contours, hirarchy, max_area, min_area):
    found_polygons_early = list()

    jv = 0
    for c in contours:
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue

        polygon = geometry.Polygon([point[0] for point in c])
        area = polygon.area
        if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]) and hirarchy[0][jv][3] == -1:  # and hirarchy[0][jv][3]==-1 :
            found_polygons_early.append(np.array([[point] for point in polygon.exterior.coords], dtype=np.uint))
        jv += 1
    return found_polygons_early

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


def filter_contours_area_of_image_tables(image, contours, hirarchy, max_area, min_area):
    found_polygons_early = list()

    jv = 0
    for c in contours:
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue

        polygon = geometry.Polygon([point[0] for point in c])
        # area = cv2.contourArea(c)
        area = polygon.area
        ##print(np.prod(thresh.shape[:2]))
        # Check that polygon has area greater than minimal area
        # print(hirarchy[0][jv][3],hirarchy )
        if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]):  # and hirarchy[0][jv][3]==-1 :
            # print(c[0][0][1])
            found_polygons_early.append(np.array([[point] for point in polygon.exterior.coords], dtype=np.int32))
        jv += 1
    return found_polygons_early

def resize_image(img_in, input_height, input_width):
    return cv2.resize(img_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)

def rotatedRectWithMaxArea(w, h, angle):
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr

def rotate_max_area_new(image, rotated, angle):
    wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0], math.radians(angle))
    h, w, _ = rotated.shape
    y1 = h // 2 - int(hr / 2)
    y2 = y1 + int(hr)
    x1 = w // 2 - int(wr / 2)
    x2 = x1 + int(wr)
    return rotated[y1:y2, x1:x2]

def rotation_image_new(img, thetha):
    rotated = imutils.rotate(img, thetha)
    return rotate_max_area_new(img, rotated, thetha)

def rotate_image(img_patch, slope):
    (h, w) = img_patch.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, slope, 1.0)
    return cv2.warpAffine(img_patch, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def rotyate_image_different( img, slope):
    # img = cv2.imread('images/input.jpg')
    num_rows, num_cols = img.shape[:2]

    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), slope, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    return img_rotation

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


def return_bonding_box_of_contours(cnts):
    boxes_tot = []
    for i in range(len(cnts)):
        x, y, w, h = cv2.boundingRect(cnts[i])

        box = [x, y, w, h]
        boxes_tot.append(box)
    return boxes_tot

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

def isNaN(num):
    return num != num

def return_parent_contours(contours, hierarchy):
    contours_parent = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]
    return contours_parent

def return_contours_of_interested_region(region_pre_p, pixel, min_area=0.0002):

    # pixels of images are identified by 5
    if len(region_pre_p.shape) == 3:
        cnts_images = (region_pre_p[:, :, 0] == pixel) * 1
    else:
        cnts_images = (region_pre_p[:, :] == pixel) * 1
    cnts_images = cnts_images.astype(np.uint8)
    cnts_images = np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
    imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hiearchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=1, min_area=min_area)

    return contours_imgs

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

def return_contours_of_image(image):

    if len(image.shape) == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image = image.astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierachy

def return_contours_of_interested_region_by_min_size(region_pre_p, pixel, min_size=0.00003):

    # pixels of images are identified by 5
    if len(region_pre_p.shape) == 3:
        cnts_images = (region_pre_p[:, :, 0] == pixel) * 1
    else:
        cnts_images = (region_pre_p[:, :] == pixel) * 1
    cnts_images = cnts_images.astype(np.uint8)
    cnts_images = np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
    imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hiearchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=1, min_area=min_size)

    return contours_imgs

def get_textregion_contours_in_org_image(cnts, img, slope_first):

    cnts_org = []
    # print(cnts,'cnts')
    for i in range(len(cnts)):
        img_copy = np.zeros(img.shape)
        img_copy = cv2.fillPoly(img_copy, pts=[cnts[i]], color=(1, 1, 1))

        # plt.imshow(img_copy)
        # plt.show()

        # print(img.shape,'img')
        img_copy = rotation_image_new(img_copy, -slope_first)
        ##print(img_copy.shape,'img_copy')
        # plt.imshow(img_copy)
        # plt.show()

        img_copy = img_copy.astype(np.uint8)
        imgray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        cont_int, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cont_int[0][:, 0, 0] = cont_int[0][:, 0, 0] + np.abs(img_copy.shape[1] - img.shape[1])
        cont_int[0][:, 0, 1] = cont_int[0][:, 0, 1] + np.abs(img_copy.shape[0] - img.shape[0])
        # print(np.shape(cont_int[0]))
        cnts_org.append(cont_int[0])

    # print(cnts_org,'cnts_org')

    # sys.exit()
    # self.y_shift = np.abs(img_copy.shape[0] - img.shape[0])
    # self.x_shift = np.abs(img_copy.shape[1] - img.shape[1])
    return cnts_org

def return_contours_of_interested_textline(region_pre_p, pixel):

    # pixels of images are identified by 5
    if len(region_pre_p.shape) == 3:
        cnts_images = (region_pre_p[:, :, 0] == pixel) * 1
    else:
        cnts_images = (region_pre_p[:, :] == pixel) * 1
    cnts_images = cnts_images.astype(np.uint8)
    cnts_images = np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
    imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hiearchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=1, min_area=0.000000003)
    return contours_imgs

def seperate_lines_vertical_cont(img_patch, contour_text_interest, thetha, box_ind, add_boxes_coor_into_textlines):
    kernel = np.ones((5, 5), np.uint8)
    pixel = 255
    min_area = 0
    max_area = 1

    if len(img_patch.shape) == 3:
        cnts_images = (img_patch[:, :, 0] == pixel) * 1
    else:
        cnts_images = (img_patch[:, :] == pixel) * 1
    cnts_images = cnts_images.astype(np.uint8)
    cnts_images = np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
    imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hiearchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=max_area, min_area=min_area)

    cont_final = []
    ###print(add_boxes_coor_into_textlines,'ikki')
    for i in range(len(contours_imgs)):
        img_contour = np.zeros((cnts_images.shape[0], cnts_images.shape[1], 3))
        img_contour = cv2.fillPoly(img_contour, pts=[contours_imgs[i]], color=(255, 255, 255))

        img_contour = img_contour.astype(np.uint8)

        img_contour = cv2.dilate(img_contour, kernel, iterations=4)
        imgrayrot = cv2.cvtColor(img_contour, cv2.COLOR_BGR2GRAY)
        _, threshrot = cv2.threshold(imgrayrot, 0, 255, 0)
        contours_text_rot, _ = cv2.findContours(threshrot.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ##contour_text_copy[:, 0, 0] = contour_text_copy[:, 0, 0] - box_ind[
        ##0]
        ##contour_text_copy[:, 0, 1] = contour_text_copy[:, 0, 1] - box_ind[1]
        ##if add_boxes_coor_into_textlines:
        ##print(np.shape(contours_text_rot[0]),'sjppo')
        ##contours_text_rot[0][:, 0, 0]=contours_text_rot[0][:, 0, 0] + box_ind[0]
        ##contours_text_rot[0][:, 0, 1]=contours_text_rot[0][:, 0, 1] + box_ind[1]
        cont_final.append(contours_text_rot[0])

    ##print(cont_final,'nadizzzz')
    return None, cont_final


def seperate_lines(img_patch, contour_text_interest, thetha, x_help, y_help):

    (h, w) = img_patch.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
    x_d = M[0, 2]
    y_d = M[1, 2]

    thetha = thetha / 180.0 * np.pi
    rotation_matrix = np.array([[np.cos(thetha), -np.sin(thetha)], [np.sin(thetha), np.cos(thetha)]])
    contour_text_interest_copy = contour_text_interest.copy()

    x_cont = contour_text_interest[:, 0, 0]
    y_cont = contour_text_interest[:, 0, 1]
    x_cont = x_cont - np.min(x_cont)
    y_cont = y_cont - np.min(y_cont)

    x_min_cont = 0
    x_max_cont = img_patch.shape[1]
    y_min_cont = 0
    y_max_cont = img_patch.shape[0]

    xv = np.linspace(x_min_cont, x_max_cont, 1000)

    textline_patch_sum_along_width = img_patch.sum(axis=1)

    first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

    y = textline_patch_sum_along_width[:]  # [first_nonzero:last_nonzero]
    y_padded = np.zeros(len(y) + 40)
    y_padded[20 : len(y) + 20] = y
    x = np.array(range(len(y)))

    peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)

    if 1 > 0:

        try:

            y_padded_smoothed_e = gaussian_filter1d(y_padded, 2)
            y_padded_up_to_down_e = -y_padded + np.max(y_padded)
            y_padded_up_to_down_padded_e = np.zeros(len(y_padded_up_to_down_e) + 40)
            y_padded_up_to_down_padded_e[20 : len(y_padded_up_to_down_e) + 20] = y_padded_up_to_down_e
            y_padded_up_to_down_padded_e = gaussian_filter1d(y_padded_up_to_down_padded_e, 2)

            peaks_e, _ = find_peaks(y_padded_smoothed_e, height=0)
            peaks_neg_e, _ = find_peaks(y_padded_up_to_down_padded_e, height=0)
            neg_peaks_max = np.max(y_padded_up_to_down_padded_e[peaks_neg_e])

            arg_neg_must_be_deleted = np.array(range(len(peaks_neg_e)))[y_padded_up_to_down_padded_e[peaks_neg_e] / float(neg_peaks_max) < 0.3]
            diff_arg_neg_must_be_deleted = np.diff(arg_neg_must_be_deleted)

            arg_diff = np.array(range(len(diff_arg_neg_must_be_deleted)))
            arg_diff_cluster = arg_diff[diff_arg_neg_must_be_deleted > 1]

            peaks_new = peaks_e[:]
            peaks_neg_new = peaks_neg_e[:]

            clusters_to_be_deleted = []
            if len(arg_diff_cluster) > 0:

                clusters_to_be_deleted.append(arg_neg_must_be_deleted[0 : arg_diff_cluster[0] + 1])
                for i in range(len(arg_diff_cluster) - 1):
                    clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[i] + 1 : arg_diff_cluster[i + 1] + 1])
                clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster) - 1] + 1 :])

            if len(clusters_to_be_deleted) > 0:
                peaks_new_extra = []
                for m in range(len(clusters_to_be_deleted)):
                    min_cluster = np.min(peaks_e[clusters_to_be_deleted[m]])
                    max_cluster = np.max(peaks_e[clusters_to_be_deleted[m]])
                    peaks_new_extra.append(int((min_cluster + max_cluster) / 2.0))
                    for m1 in range(len(clusters_to_be_deleted[m])):
                        peaks_new = peaks_new[peaks_new != peaks_e[clusters_to_be_deleted[m][m1] - 1]]
                        peaks_new = peaks_new[peaks_new != peaks_e[clusters_to_be_deleted[m][m1]]]

                        peaks_neg_new = peaks_neg_new[peaks_neg_new != peaks_neg_e[clusters_to_be_deleted[m][m1]]]
                peaks_new_tot = []
                for i1 in peaks_new:
                    peaks_new_tot.append(i1)
                for i1 in peaks_new_extra:
                    peaks_new_tot.append(i1)
                peaks_new_tot = np.sort(peaks_new_tot)

            else:
                peaks_new_tot = peaks_e[:]

            textline_con, hierachy = return_contours_of_image(img_patch)
            textline_con_fil = filter_contours_area_of_image(img_patch, textline_con, hierachy, max_area=1, min_area=0.0008)
            y_diff_mean = np.mean(np.diff(peaks_new_tot))  # self.find_contours_mean_y_diff(textline_con_fil)

            sigma_gaus = int(y_diff_mean * (7.0 / 40.0))
            # print(sigma_gaus,'sigma_gaus')
        except:
            sigma_gaus = 12
        if sigma_gaus < 3:
            sigma_gaus = 3
        # print(sigma_gaus,'sigma')

    y_padded_smoothed = gaussian_filter1d(y_padded, sigma_gaus)
    y_padded_up_to_down = -y_padded + np.max(y_padded)
    y_padded_up_to_down_padded = np.zeros(len(y_padded_up_to_down) + 40)
    y_padded_up_to_down_padded[20 : len(y_padded_up_to_down) + 20] = y_padded_up_to_down
    y_padded_up_to_down_padded = gaussian_filter1d(y_padded_up_to_down_padded, sigma_gaus)

    peaks, _ = find_peaks(y_padded_smoothed, height=0)
    peaks_neg, _ = find_peaks(y_padded_up_to_down_padded, height=0)

    try:
        neg_peaks_max = np.max(y_padded_smoothed[peaks])

        arg_neg_must_be_deleted = np.array(range(len(peaks_neg)))[y_padded_up_to_down_padded[peaks_neg] / float(neg_peaks_max) < 0.42]

        diff_arg_neg_must_be_deleted = np.diff(arg_neg_must_be_deleted)

        arg_diff = np.array(range(len(diff_arg_neg_must_be_deleted)))
        arg_diff_cluster = arg_diff[diff_arg_neg_must_be_deleted > 1]
    except:
        arg_neg_must_be_deleted = []
        arg_diff_cluster = []

    try:
        peaks_new = peaks[:]
        peaks_neg_new = peaks_neg[:]
        clusters_to_be_deleted = []

        if len(arg_diff_cluster) >= 2 and len(arg_diff_cluster) > 0:

            clusters_to_be_deleted.append(arg_neg_must_be_deleted[0 : arg_diff_cluster[0] + 1])
            for i in range(len(arg_diff_cluster) - 1):
                clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[i] + 1 : arg_diff_cluster[i + 1] + 1])
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster) - 1] + 1 :])
        elif len(arg_neg_must_be_deleted) >= 2 and len(arg_diff_cluster) == 0:
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[:])

        if len(arg_neg_must_be_deleted) == 1:
            clusters_to_be_deleted.append(arg_neg_must_be_deleted)

        if len(clusters_to_be_deleted) > 0:
            peaks_new_extra = []
            for m in range(len(clusters_to_be_deleted)):
                min_cluster = np.min(peaks[clusters_to_be_deleted[m]])
                max_cluster = np.max(peaks[clusters_to_be_deleted[m]])
                peaks_new_extra.append(int((min_cluster + max_cluster) / 2.0))
                for m1 in range(len(clusters_to_be_deleted[m])):
                    peaks_new = peaks_new[peaks_new != peaks[clusters_to_be_deleted[m][m1] - 1]]
                    peaks_new = peaks_new[peaks_new != peaks[clusters_to_be_deleted[m][m1]]]

                    peaks_neg_new = peaks_neg_new[peaks_neg_new != peaks_neg[clusters_to_be_deleted[m][m1]]]
            peaks_new_tot = []
            for i1 in peaks_new:
                peaks_new_tot.append(i1)
            for i1 in peaks_new_extra:
                peaks_new_tot.append(i1)
            peaks_new_tot = np.sort(peaks_new_tot)

            ##plt.plot(y_padded_up_to_down_padded)
            ##plt.plot(peaks_neg,y_padded_up_to_down_padded[peaks_neg],'*')
            ##plt.show()

            ##plt.plot(y_padded_up_to_down_padded)
            ##plt.plot(peaks_neg_new,y_padded_up_to_down_padded[peaks_neg_new],'*')
            ##plt.show()

            ##plt.plot(y_padded_smoothed)
            ##plt.plot(peaks,y_padded_smoothed[peaks],'*')
            ##plt.show()

            ##plt.plot(y_padded_smoothed)
            ##plt.plot(peaks_new_tot,y_padded_smoothed[peaks_new_tot],'*')
            ##plt.show()

            peaks = peaks_new_tot[:]
            peaks_neg = peaks_neg_new[:]

        else:
            peaks_new_tot = peaks[:]
            peaks = peaks_new_tot[:]
            peaks_neg = peaks_neg_new[:]
    except:
        pass

    mean_value_of_peaks = np.mean(y_padded_smoothed[peaks])
    std_value_of_peaks = np.std(y_padded_smoothed[peaks])
    peaks_values = y_padded_smoothed[peaks]

    peaks_neg = peaks_neg - 20 - 20
    peaks = peaks - 20

    for jj in range(len(peaks_neg)):
        if peaks_neg[jj] > len(x) - 1:
            peaks_neg[jj] = len(x) - 1

    for jj in range(len(peaks)):
        if peaks[jj] > len(x) - 1:
            peaks[jj] = len(x) - 1

    textline_boxes = []
    textline_boxes_rot = []

    if len(peaks_neg) == len(peaks) + 1 and len(peaks) >= 3:
        for jj in range(len(peaks)):

            if jj == (len(peaks) - 1):
                dis_to_next_up = abs(peaks[jj] - peaks_neg[jj])
                dis_to_next_down = abs(peaks[jj] - peaks_neg[jj + 1])

                if peaks_values[jj] > mean_value_of_peaks - std_value_of_peaks / 2.0:
                    point_up = peaks[jj] + first_nonzero - int(1.3 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                    point_down = y_max_cont - 1  ##peaks[jj] + first_nonzero + int(1.3 * dis_to_next_down) #point_up# np.max(y_cont)#peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)
                else:
                    point_up = peaks[jj] + first_nonzero - int(1.4 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                    point_down = y_max_cont - 1  ##peaks[jj] + first_nonzero + int(1.6 * dis_to_next_down) #point_up# np.max(y_cont)#peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)

                point_down_narrow = peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)  ###-int(dis_to_next_down*1./2)
            else:
                dis_to_next_up = abs(peaks[jj] - peaks_neg[jj])
                dis_to_next_down = abs(peaks[jj] - peaks_neg[jj + 1])

                if peaks_values[jj] > mean_value_of_peaks - std_value_of_peaks / 2.0:
                    point_up = peaks[jj] + first_nonzero - int(1.1 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                    point_down = peaks[jj] + first_nonzero + int(1.1 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)
                else:
                    point_up = peaks[jj] + first_nonzero - int(1.23 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                    point_down = peaks[jj] + first_nonzero + int(1.33 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)

                point_down_narrow = peaks[jj] + first_nonzero + int(1.1 * dis_to_next_down)  ###-int(dis_to_next_down*1./2)

            if point_down_narrow >= img_patch.shape[0]:
                point_down_narrow = img_patch.shape[0] - 2

            distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[jj] + first_nonzero), True) for mj in range(len(xv))]
            distances = np.array(distances)

            xvinside = xv[distances >= 0]

            if len(xvinside) == 0:
                x_min = x_min_cont
                x_max = x_max_cont
            else:
                x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
                x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)

            p1 = np.dot(rotation_matrix, [int(x_min), int(point_up)])
            p2 = np.dot(rotation_matrix, [int(x_max), int(point_up)])
            p3 = np.dot(rotation_matrix, [int(x_max), int(point_down)])
            p4 = np.dot(rotation_matrix, [int(x_min), int(point_down)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d

            if x_min_rot1 < 0:
                x_min_rot1 = 0
            if x_min_rot4 < 0:
                x_min_rot4 = 0
            if point_up_rot1 < 0:
                point_up_rot1 = 0
            if point_up_rot2 < 0:
                point_up_rot2 = 0

            x_min_rot1 = x_min_rot1 - x_help
            x_max_rot2 = x_max_rot2 - x_help
            x_max_rot3 = x_max_rot3 - x_help
            x_min_rot4 = x_min_rot4 - x_help

            point_up_rot1 = point_up_rot1 - y_help
            point_up_rot2 = point_up_rot2 - y_help
            point_down_rot3 = point_down_rot3 - y_help
            point_down_rot4 = point_down_rot4 - y_help

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)], [int(x_max_rot2), int(point_up_rot2)], [int(x_max_rot3), int(point_down_rot3)], [int(x_min_rot4), int(point_down_rot4)]]))

            textline_boxes.append(np.array([[int(x_min), int(point_up)], [int(x_max), int(point_up)], [int(x_max), int(point_down)], [int(x_min), int(point_down)]]))

    elif len(peaks) < 1:
        pass

    elif len(peaks) == 1:

        distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[0] + first_nonzero), True) for mj in range(len(xv))]
        distances = np.array(distances)

        xvinside = xv[distances >= 0]

        if len(xvinside) == 0:
            x_min = x_min_cont
            x_max = x_max_cont
        else:
            x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
            x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)
        # x_min = x_min_cont
        # x_max = x_max_cont

        y_min = y_min_cont
        y_max = y_max_cont

        p1 = np.dot(rotation_matrix, [int(x_min), int(y_min)])
        p2 = np.dot(rotation_matrix, [int(x_max), int(y_min)])
        p3 = np.dot(rotation_matrix, [int(x_max), int(y_max)])
        p4 = np.dot(rotation_matrix, [int(x_min), int(y_max)])

        x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
        x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
        x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
        x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d

        if x_min_rot1 < 0:
            x_min_rot1 = 0
        if x_min_rot4 < 0:
            x_min_rot4 = 0
        if point_up_rot1 < 0:
            point_up_rot1 = 0
        if point_up_rot2 < 0:
            point_up_rot2 = 0

        x_min_rot1 = x_min_rot1 - x_help
        x_max_rot2 = x_max_rot2 - x_help
        x_max_rot3 = x_max_rot3 - x_help
        x_min_rot4 = x_min_rot4 - x_help

        point_up_rot1 = point_up_rot1 - y_help
        point_up_rot2 = point_up_rot2 - y_help
        point_down_rot3 = point_down_rot3 - y_help
        point_down_rot4 = point_down_rot4 - y_help

        textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)], [int(x_max_rot2), int(point_up_rot2)], [int(x_max_rot3), int(point_down_rot3)], [int(x_min_rot4), int(point_down_rot4)]]))

        textline_boxes.append(np.array([[int(x_min), int(y_min)], [int(x_max), int(y_min)], [int(x_max), int(y_max)], [int(x_min), int(y_max)]]))

    elif len(peaks) == 2:
        dis_to_next = np.abs(peaks[1] - peaks[0])
        for jj in range(len(peaks)):
            if jj == 0:
                point_up = 0  # peaks[jj] + first_nonzero - int(1. / 1.7 * dis_to_next)
                if point_up < 0:
                    point_up = 1
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.8 * dis_to_next)
            elif jj == 1:
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.8 * dis_to_next)
                if point_down >= img_patch.shape[0]:
                    point_down = img_patch.shape[0] - 2
                point_up = peaks[jj] + first_nonzero - int(1.0 / 1.8 * dis_to_next)

            distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[jj] + first_nonzero), True) for mj in range(len(xv))]
            distances = np.array(distances)

            xvinside = xv[distances >= 0]

            if len(xvinside) == 0:
                x_min = x_min_cont
                x_max = x_max_cont
            else:
                x_min = np.min(xvinside)
                x_max = np.max(xvinside)

            p1 = np.dot(rotation_matrix, [int(x_min), int(point_up)])
            p2 = np.dot(rotation_matrix, [int(x_max), int(point_up)])
            p3 = np.dot(rotation_matrix, [int(x_max), int(point_down)])
            p4 = np.dot(rotation_matrix, [int(x_min), int(point_down)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d

            if x_min_rot1 < 0:
                x_min_rot1 = 0
            if x_min_rot4 < 0:
                x_min_rot4 = 0
            if point_up_rot1 < 0:
                point_up_rot1 = 0
            if point_up_rot2 < 0:
                point_up_rot2 = 0

            x_min_rot1 = x_min_rot1 - x_help
            x_max_rot2 = x_max_rot2 - x_help
            x_max_rot3 = x_max_rot3 - x_help
            x_min_rot4 = x_min_rot4 - x_help

            point_up_rot1 = point_up_rot1 - y_help
            point_up_rot2 = point_up_rot2 - y_help
            point_down_rot3 = point_down_rot3 - y_help
            point_down_rot4 = point_down_rot4 - y_help

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)], [int(x_max_rot2), int(point_up_rot2)], [int(x_max_rot3), int(point_down_rot3)], [int(x_min_rot4), int(point_down_rot4)]]))

            textline_boxes.append(np.array([[int(x_min), int(point_up)], [int(x_max), int(point_up)], [int(x_max), int(point_down)], [int(x_min), int(point_down)]]))
    else:
        for jj in range(len(peaks)):

            if jj == 0:
                dis_to_next = peaks[jj + 1] - peaks[jj]
                # point_up=peaks[jj]+first_nonzero-int(1./3*dis_to_next)
                point_up = peaks[jj] + first_nonzero - int(1.0 / 1.9 * dis_to_next)
                if point_up < 0:
                    point_up = 1
                # point_down=peaks[jj]+first_nonzero+int(1./3*dis_to_next)
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.9 * dis_to_next)
            elif jj == len(peaks) - 1:
                dis_to_next = peaks[jj] - peaks[jj - 1]
                # point_down=peaks[jj]+first_nonzero+int(1./3*dis_to_next)
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.7 * dis_to_next)
                if point_down >= img_patch.shape[0]:
                    point_down = img_patch.shape[0] - 2
                # point_up=peaks[jj]+first_nonzero-int(1./3*dis_to_next)
                point_up = peaks[jj] + first_nonzero - int(1.0 / 1.9 * dis_to_next)
            else:
                dis_to_next_down = peaks[jj + 1] - peaks[jj]
                dis_to_next_up = peaks[jj] - peaks[jj - 1]

                point_up = peaks[jj] + first_nonzero - int(1.0 / 1.9 * dis_to_next_up)
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.9 * dis_to_next_down)

            distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[jj] + first_nonzero), True) for mj in range(len(xv))]
            distances = np.array(distances)

            xvinside = xv[distances >= 0]

            if len(xvinside) == 0:
                x_min = x_min_cont
                x_max = x_max_cont
            else:
                x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
                x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)

            p1 = np.dot(rotation_matrix, [int(x_min), int(point_up)])
            p2 = np.dot(rotation_matrix, [int(x_max), int(point_up)])
            p3 = np.dot(rotation_matrix, [int(x_max), int(point_down)])
            p4 = np.dot(rotation_matrix, [int(x_min), int(point_down)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d

            if x_min_rot1 < 0:
                x_min_rot1 = 0
            if x_min_rot4 < 0:
                x_min_rot4 = 0
            if point_up_rot1 < 0:
                point_up_rot1 = 0
            if point_up_rot2 < 0:
                point_up_rot2 = 0

            x_min_rot1 = x_min_rot1 - x_help
            x_max_rot2 = x_max_rot2 - x_help
            x_max_rot3 = x_max_rot3 - x_help
            x_min_rot4 = x_min_rot4 - x_help

            point_up_rot1 = point_up_rot1 - y_help
            point_up_rot2 = point_up_rot2 - y_help
            point_down_rot3 = point_down_rot3 - y_help
            point_down_rot4 = point_down_rot4 - y_help

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)], [int(x_max_rot2), int(point_up_rot2)], [int(x_max_rot3), int(point_down_rot3)], [int(x_min_rot4), int(point_down_rot4)]]))

            textline_boxes.append(np.array([[int(x_min), int(point_up)], [int(x_max), int(point_up)], [int(x_max), int(point_down)], [int(x_min), int(point_down)]]))

    return peaks, textline_boxes_rot

def seperate_lines_vertical(img_patch, contour_text_interest, thetha):

    thetha = thetha + 90

    (h, w) = img_patch.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
    x_d = M[0, 2]
    y_d = M[1, 2]

    thetha = thetha / 180.0 * np.pi
    rotation_matrix = np.array([[np.cos(thetha), -np.sin(thetha)], [np.sin(thetha), np.cos(thetha)]])
    contour_text_interest_copy = contour_text_interest.copy()

    x_cont = contour_text_interest[:, 0, 0]
    y_cont = contour_text_interest[:, 0, 1]
    x_cont = x_cont - np.min(x_cont)
    y_cont = y_cont - np.min(y_cont)

    x_min_cont = 0
    x_max_cont = img_patch.shape[1]
    y_min_cont = 0
    y_max_cont = img_patch.shape[0]

    xv = np.linspace(x_min_cont, x_max_cont, 1000)

    textline_patch_sum_along_width = img_patch.sum(axis=0)

    first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

    y = textline_patch_sum_along_width[:]  # [first_nonzero:last_nonzero]
    y_padded = np.zeros(len(y) + 40)
    y_padded[20 : len(y) + 20] = y
    x = np.array(range(len(y)))

    peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
    if 1 > 0:

        try:

            y_padded_smoothed_e = gaussian_filter1d(y_padded, 2)
            y_padded_up_to_down_e = -y_padded + np.max(y_padded)
            y_padded_up_to_down_padded_e = np.zeros(len(y_padded_up_to_down_e) + 40)
            y_padded_up_to_down_padded_e[20 : len(y_padded_up_to_down_e) + 20] = y_padded_up_to_down_e
            y_padded_up_to_down_padded_e = gaussian_filter1d(y_padded_up_to_down_padded_e, 2)

            peaks_e, _ = find_peaks(y_padded_smoothed_e, height=0)
            peaks_neg_e, _ = find_peaks(y_padded_up_to_down_padded_e, height=0)
            neg_peaks_max = np.max(y_padded_up_to_down_padded_e[peaks_neg_e])

            arg_neg_must_be_deleted = np.array(range(len(peaks_neg_e)))[y_padded_up_to_down_padded_e[peaks_neg_e] / float(neg_peaks_max) < 0.3]
            diff_arg_neg_must_be_deleted = np.diff(arg_neg_must_be_deleted)

            arg_diff = np.array(range(len(diff_arg_neg_must_be_deleted)))
            arg_diff_cluster = arg_diff[diff_arg_neg_must_be_deleted > 1]

            peaks_new = peaks_e[:]
            peaks_neg_new = peaks_neg_e[:]

            clusters_to_be_deleted = []
            if len(arg_diff_cluster) > 0:

                clusters_to_be_deleted.append(arg_neg_must_be_deleted[0 : arg_diff_cluster[0] + 1])
                for i in range(len(arg_diff_cluster) - 1):
                    clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[i] + 1 : arg_diff_cluster[i + 1] + 1])
                clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster) - 1] + 1 :])

            if len(clusters_to_be_deleted) > 0:
                peaks_new_extra = []
                for m in range(len(clusters_to_be_deleted)):
                    min_cluster = np.min(peaks_e[clusters_to_be_deleted[m]])
                    max_cluster = np.max(peaks_e[clusters_to_be_deleted[m]])
                    peaks_new_extra.append(int((min_cluster + max_cluster) / 2.0))
                    for m1 in range(len(clusters_to_be_deleted[m])):
                        peaks_new = peaks_new[peaks_new != peaks_e[clusters_to_be_deleted[m][m1] - 1]]
                        peaks_new = peaks_new[peaks_new != peaks_e[clusters_to_be_deleted[m][m1]]]

                        peaks_neg_new = peaks_neg_new[peaks_neg_new != peaks_neg_e[clusters_to_be_deleted[m][m1]]]
                peaks_new_tot = []
                for i1 in peaks_new:
                    peaks_new_tot.append(i1)
                for i1 in peaks_new_extra:
                    peaks_new_tot.append(i1)
                peaks_new_tot = np.sort(peaks_new_tot)

            else:
                peaks_new_tot = peaks_e[:]

            textline_con, hierachy = return_contours_of_image(img_patch)
            textline_con_fil = filter_contours_area_of_image(img_patch, textline_con, hierachy, max_area=1, min_area=0.0008)
            y_diff_mean = np.mean(np.diff(peaks_new_tot))  # self.find_contours_mean_y_diff(textline_con_fil)

            sigma_gaus = int(y_diff_mean * (7.0 / 40.0))
            # print(sigma_gaus,'sigma_gaus')
        except:
            sigma_gaus = 12
        if sigma_gaus < 3:
            sigma_gaus = 3
        # print(sigma_gaus,'sigma')

    y_padded_smoothed = gaussian_filter1d(y_padded, sigma_gaus)
    y_padded_up_to_down = -y_padded + np.max(y_padded)
    y_padded_up_to_down_padded = np.zeros(len(y_padded_up_to_down) + 40)
    y_padded_up_to_down_padded[20 : len(y_padded_up_to_down) + 20] = y_padded_up_to_down
    y_padded_up_to_down_padded = gaussian_filter1d(y_padded_up_to_down_padded, sigma_gaus)

    peaks, _ = find_peaks(y_padded_smoothed, height=0)
    peaks_neg, _ = find_peaks(y_padded_up_to_down_padded, height=0)

    # plt.plot(y_padded_up_to_down_padded)
    # plt.plot(peaks_neg,y_padded_up_to_down_padded[peaks_neg],'*')
    # plt.title('negs')
    # plt.show()

    # plt.plot(y_padded_smoothed)
    # plt.plot(peaks,y_padded_smoothed[peaks],'*')
    # plt.title('poss')
    # plt.show()

    neg_peaks_max = np.max(y_padded_up_to_down_padded[peaks_neg])

    arg_neg_must_be_deleted = np.array(range(len(peaks_neg)))[y_padded_up_to_down_padded[peaks_neg] / float(neg_peaks_max) < 0.42]

    diff_arg_neg_must_be_deleted = np.diff(arg_neg_must_be_deleted)

    arg_diff = np.array(range(len(diff_arg_neg_must_be_deleted)))
    arg_diff_cluster = arg_diff[diff_arg_neg_must_be_deleted > 1]

    peaks_new = peaks[:]
    peaks_neg_new = peaks_neg[:]
    clusters_to_be_deleted = []

    if len(arg_diff_cluster) >= 2 and len(arg_diff_cluster) > 0:

        clusters_to_be_deleted.append(arg_neg_must_be_deleted[0 : arg_diff_cluster[0] + 1])
        for i in range(len(arg_diff_cluster) - 1):
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[i] + 1 : arg_diff_cluster[i + 1] + 1])
        clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster) - 1] + 1 :])
    elif len(arg_neg_must_be_deleted) >= 2 and len(arg_diff_cluster) == 0:
        clusters_to_be_deleted.append(arg_neg_must_be_deleted[:])

    if len(arg_neg_must_be_deleted) == 1:
        clusters_to_be_deleted.append(arg_neg_must_be_deleted)

    if len(clusters_to_be_deleted) > 0:
        peaks_new_extra = []
        for m in range(len(clusters_to_be_deleted)):
            min_cluster = np.min(peaks[clusters_to_be_deleted[m]])
            max_cluster = np.max(peaks[clusters_to_be_deleted[m]])
            peaks_new_extra.append(int((min_cluster + max_cluster) / 2.0))
            for m1 in range(len(clusters_to_be_deleted[m])):
                peaks_new = peaks_new[peaks_new != peaks[clusters_to_be_deleted[m][m1] - 1]]
                peaks_new = peaks_new[peaks_new != peaks[clusters_to_be_deleted[m][m1]]]

                peaks_neg_new = peaks_neg_new[peaks_neg_new != peaks_neg[clusters_to_be_deleted[m][m1]]]
        peaks_new_tot = []
        for i1 in peaks_new:
            peaks_new_tot.append(i1)
        for i1 in peaks_new_extra:
            peaks_new_tot.append(i1)
        peaks_new_tot = np.sort(peaks_new_tot)

        peaks = peaks_new_tot[:]
        peaks_neg = peaks_neg_new[:]

    else:
        peaks_new_tot = peaks[:]
        peaks = peaks_new_tot[:]
        peaks_neg = peaks_neg_new[:]

    mean_value_of_peaks = np.mean(y_padded_smoothed[peaks])
    std_value_of_peaks = np.std(y_padded_smoothed[peaks])
    peaks_values = y_padded_smoothed[peaks]

    peaks_neg = peaks_neg - 20 - 20
    peaks = peaks - 20

    for jj in range(len(peaks_neg)):
        if peaks_neg[jj] > len(x) - 1:
            peaks_neg[jj] = len(x) - 1

    for jj in range(len(peaks)):
        if peaks[jj] > len(x) - 1:
            peaks[jj] = len(x) - 1

    textline_boxes = []
    textline_boxes_rot = []

    if len(peaks_neg) == len(peaks) + 1 and len(peaks) >= 3:
        # print('11')
        for jj in range(len(peaks)):

            if jj == (len(peaks) - 1):
                dis_to_next_up = abs(peaks[jj] - peaks_neg[jj])
                dis_to_next_down = abs(peaks[jj] - peaks_neg[jj + 1])

                if peaks_values[jj] > mean_value_of_peaks - std_value_of_peaks / 2.0:
                    point_up = peaks[jj] + first_nonzero - int(1.3 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                    point_down = x_max_cont - 1  ##peaks[jj] + first_nonzero + int(1.3 * dis_to_next_down) #point_up# np.max(y_cont)#peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)
                else:
                    point_up = peaks[jj] + first_nonzero - int(1.4 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                    point_down = x_max_cont - 1  ##peaks[jj] + first_nonzero + int(1.6 * dis_to_next_down) #point_up# np.max(y_cont)#peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)

                point_down_narrow = peaks[jj] + first_nonzero + int(1.4 * dis_to_next_down)  ###-int(dis_to_next_down*1./2)
            else:
                dis_to_next_up = abs(peaks[jj] - peaks_neg[jj])
                dis_to_next_down = abs(peaks[jj] - peaks_neg[jj + 1])

                if peaks_values[jj] > mean_value_of_peaks - std_value_of_peaks / 2.0:
                    point_up = peaks[jj] + first_nonzero - int(1.1 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                    point_down = peaks[jj] + first_nonzero + int(1.1 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)
                else:
                    point_up = peaks[jj] + first_nonzero - int(1.23 * dis_to_next_up)  ##+int(dis_to_next_up*1./4.0)
                    point_down = peaks[jj] + first_nonzero + int(1.33 * dis_to_next_down)  ###-int(dis_to_next_down*1./4.0)

                point_down_narrow = peaks[jj] + first_nonzero + int(1.1 * dis_to_next_down)  ###-int(dis_to_next_down*1./2)

            if point_down_narrow >= img_patch.shape[0]:
                point_down_narrow = img_patch.shape[0] - 2

            distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[jj] + first_nonzero), True) for mj in range(len(xv))]
            distances = np.array(distances)

            xvinside = xv[distances >= 0]

            if len(xvinside) == 0:
                x_min = x_min_cont
                x_max = x_max_cont
            else:
                x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
                x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)

            p1 = np.dot(rotation_matrix, [int(point_up), int(y_min_cont)])
            p2 = np.dot(rotation_matrix, [int(point_down), int(y_min_cont)])
            p3 = np.dot(rotation_matrix, [int(point_down), int(y_max_cont)])
            p4 = np.dot(rotation_matrix, [int(point_up), int(y_max_cont)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d

            if x_min_rot1 < 0:
                x_min_rot1 = 0
            if x_min_rot4 < 0:
                x_min_rot4 = 0
            if point_up_rot1 < 0:
                point_up_rot1 = 0
            if point_up_rot2 < 0:
                point_up_rot2 = 0

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)], [int(x_max_rot2), int(point_up_rot2)], [int(x_max_rot3), int(point_down_rot3)], [int(x_min_rot4), int(point_down_rot4)]]))

            textline_boxes.append(np.array([[int(x_min), int(point_up)], [int(x_max), int(point_up)], [int(x_max), int(point_down)], [int(x_min), int(point_down)]]))

    elif len(peaks) < 1:
        pass

    elif len(peaks) == 1:
        x_min = x_min_cont
        x_max = x_max_cont

        y_min = y_min_cont
        y_max = y_max_cont

        p1 = np.dot(rotation_matrix, [int(point_up), int(y_min_cont)])
        p2 = np.dot(rotation_matrix, [int(point_down), int(y_min_cont)])
        p3 = np.dot(rotation_matrix, [int(point_down), int(y_max_cont)])
        p4 = np.dot(rotation_matrix, [int(point_up), int(y_max_cont)])

        x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
        x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
        x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
        x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d

        if x_min_rot1 < 0:
            x_min_rot1 = 0
        if x_min_rot4 < 0:
            x_min_rot4 = 0
        if point_up_rot1 < 0:
            point_up_rot1 = 0
        if point_up_rot2 < 0:
            point_up_rot2 = 0

        textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)], [int(x_max_rot2), int(point_up_rot2)], [int(x_max_rot3), int(point_down_rot3)], [int(x_min_rot4), int(point_down_rot4)]]))

        textline_boxes.append(np.array([[int(x_min), int(y_min)], [int(x_max), int(y_min)], [int(x_max), int(y_max)], [int(x_min), int(y_max)]]))

    elif len(peaks) == 2:
        dis_to_next = np.abs(peaks[1] - peaks[0])
        for jj in range(len(peaks)):
            if jj == 0:
                point_up = 0  # peaks[jj] + first_nonzero - int(1. / 1.7 * dis_to_next)
                if point_up < 0:
                    point_up = 1
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.8 * dis_to_next)
            elif jj == 1:
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.8 * dis_to_next)
                if point_down >= img_patch.shape[0]:
                    point_down = img_patch.shape[0] - 2
                point_up = peaks[jj] + first_nonzero - int(1.0 / 1.8 * dis_to_next)

            distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[jj] + first_nonzero), True) for mj in range(len(xv))]
            distances = np.array(distances)

            xvinside = xv[distances >= 0]

            if len(xvinside) == 0:
                x_min = x_min_cont
                x_max = x_max_cont
            else:
                x_min = np.min(xvinside)
                x_max = np.max(xvinside)

            p1 = np.dot(rotation_matrix, [int(point_up), int(y_min_cont)])
            p2 = np.dot(rotation_matrix, [int(point_down), int(y_min_cont)])
            p3 = np.dot(rotation_matrix, [int(point_down), int(y_max_cont)])
            p4 = np.dot(rotation_matrix, [int(point_up), int(y_max_cont)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d

            if x_min_rot1 < 0:
                x_min_rot1 = 0
            if x_min_rot4 < 0:
                x_min_rot4 = 0
            if point_up_rot1 < 0:
                point_up_rot1 = 0
            if point_up_rot2 < 0:
                point_up_rot2 = 0

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)], [int(x_max_rot2), int(point_up_rot2)], [int(x_max_rot3), int(point_down_rot3)], [int(x_min_rot4), int(point_down_rot4)]]))

            textline_boxes.append(np.array([[int(x_min), int(point_up)], [int(x_max), int(point_up)], [int(x_max), int(point_down)], [int(x_min), int(point_down)]]))
    else:
        for jj in range(len(peaks)):

            if jj == 0:
                dis_to_next = peaks[jj + 1] - peaks[jj]
                # point_up=peaks[jj]+first_nonzero-int(1./3*dis_to_next)
                point_up = peaks[jj] + first_nonzero - int(1.0 / 1.9 * dis_to_next)
                if point_up < 0:
                    point_up = 1
                # point_down=peaks[jj]+first_nonzero+int(1./3*dis_to_next)
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.9 * dis_to_next)
            elif jj == len(peaks) - 1:
                dis_to_next = peaks[jj] - peaks[jj - 1]
                # point_down=peaks[jj]+first_nonzero+int(1./3*dis_to_next)
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.7 * dis_to_next)
                if point_down >= img_patch.shape[0]:
                    point_down = img_patch.shape[0] - 2
                # point_up=peaks[jj]+first_nonzero-int(1./3*dis_to_next)
                point_up = peaks[jj] + first_nonzero - int(1.0 / 1.9 * dis_to_next)
            else:
                dis_to_next_down = peaks[jj + 1] - peaks[jj]
                dis_to_next_up = peaks[jj] - peaks[jj - 1]

                point_up = peaks[jj] + first_nonzero - int(1.0 / 1.9 * dis_to_next_up)
                point_down = peaks[jj] + first_nonzero + int(1.0 / 1.9 * dis_to_next_down)

            distances = [cv2.pointPolygonTest(contour_text_interest_copy, (xv[mj], peaks[jj] + first_nonzero), True) for mj in range(len(xv))]
            distances = np.array(distances)

            xvinside = xv[distances >= 0]

            if len(xvinside) == 0:
                x_min = x_min_cont
                x_max = x_max_cont
            else:
                x_min = np.min(xvinside)  # max(x_min_interest,x_min_cont)
                x_max = np.max(xvinside)  # min(x_max_interest,x_max_cont)

            p1 = np.dot(rotation_matrix, [int(point_up), int(y_min_cont)])
            p2 = np.dot(rotation_matrix, [int(point_down), int(y_min_cont)])
            p3 = np.dot(rotation_matrix, [int(point_down), int(y_max_cont)])
            p4 = np.dot(rotation_matrix, [int(point_up), int(y_max_cont)])

            x_min_rot1, point_up_rot1 = p1[0] + x_d, p1[1] + y_d
            x_max_rot2, point_up_rot2 = p2[0] + x_d, p2[1] + y_d
            x_max_rot3, point_down_rot3 = p3[0] + x_d, p3[1] + y_d
            x_min_rot4, point_down_rot4 = p4[0] + x_d, p4[1] + y_d

            if x_min_rot1 < 0:
                x_min_rot1 = 0
            if x_min_rot4 < 0:
                x_min_rot4 = 0
            if point_up_rot1 < 0:
                point_up_rot1 = 0
            if point_up_rot2 < 0:
                point_up_rot2 = 0

            textline_boxes_rot.append(np.array([[int(x_min_rot1), int(point_up_rot1)], [int(x_max_rot2), int(point_up_rot2)], [int(x_max_rot3), int(point_down_rot3)], [int(x_min_rot4), int(point_down_rot4)]]))

            textline_boxes.append(np.array([[int(x_min), int(point_up)], [int(x_max), int(point_up)], [int(x_max), int(point_down)], [int(x_min), int(point_down)]]))

    return peaks, textline_boxes_rot

def seperate_lines_new_inside_teils2(img_patch, thetha):

    (h, w) = img_patch.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -thetha, 1.0)
    x_d = M[0, 2]
    y_d = M[1, 2]

    thetha = thetha / 180.0 * np.pi
    rotation_matrix = np.array([[np.cos(thetha), -np.sin(thetha)], [np.sin(thetha), np.cos(thetha)]])
    # contour_text_interest_copy = contour_text_interest.copy()

    # x_cont = contour_text_interest[:, 0, 0]
    # y_cont = contour_text_interest[:, 0, 1]
    # x_cont = x_cont - np.min(x_cont)
    # y_cont = y_cont - np.min(y_cont)

    x_min_cont = 0
    x_max_cont = img_patch.shape[1]
    y_min_cont = 0
    y_max_cont = img_patch.shape[0]

    xv = np.linspace(x_min_cont, x_max_cont, 1000)

    textline_patch_sum_along_width = img_patch.sum(axis=1)

    first_nonzero = 0  # (next((i for i, x in enumerate(mada_n) if x), None))

    y = textline_patch_sum_along_width[:]  # [first_nonzero:last_nonzero]
    y_padded = np.zeros(len(y) + 40)
    y_padded[20 : len(y) + 20] = y
    x = np.array(range(len(y)))

    peaks_real, _ = find_peaks(gaussian_filter1d(y, 3), height=0)
    if 1 > 0:

        try:

            y_padded_smoothed_e = gaussian_filter1d(y_padded, 2)
            y_padded_up_to_down_e = -y_padded + np.max(y_padded)
            y_padded_up_to_down_padded_e = np.zeros(len(y_padded_up_to_down_e) + 40)
            y_padded_up_to_down_padded_e[20 : len(y_padded_up_to_down_e) + 20] = y_padded_up_to_down_e
            y_padded_up_to_down_padded_e = gaussian_filter1d(y_padded_up_to_down_padded_e, 2)

            peaks_e, _ = find_peaks(y_padded_smoothed_e, height=0)
            peaks_neg_e, _ = find_peaks(y_padded_up_to_down_padded_e, height=0)
            neg_peaks_max = np.max(y_padded_up_to_down_padded_e[peaks_neg_e])

            arg_neg_must_be_deleted = np.array(range(len(peaks_neg_e)))[y_padded_up_to_down_padded_e[peaks_neg_e] / float(neg_peaks_max) < 0.3]
            diff_arg_neg_must_be_deleted = np.diff(arg_neg_must_be_deleted)

            arg_diff = np.array(range(len(diff_arg_neg_must_be_deleted)))
            arg_diff_cluster = arg_diff[diff_arg_neg_must_be_deleted > 1]

            peaks_new = peaks_e[:]
            peaks_neg_new = peaks_neg_e[:]

            clusters_to_be_deleted = []
            if len(arg_diff_cluster) > 0:

                clusters_to_be_deleted.append(arg_neg_must_be_deleted[0 : arg_diff_cluster[0] + 1])
                for i in range(len(arg_diff_cluster) - 1):
                    clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[i] + 1 : arg_diff_cluster[i + 1] + 1])
                clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster) - 1] + 1 :])

            if len(clusters_to_be_deleted) > 0:
                peaks_new_extra = []
                for m in range(len(clusters_to_be_deleted)):
                    min_cluster = np.min(peaks_e[clusters_to_be_deleted[m]])
                    max_cluster = np.max(peaks_e[clusters_to_be_deleted[m]])
                    peaks_new_extra.append(int((min_cluster + max_cluster) / 2.0))
                    for m1 in range(len(clusters_to_be_deleted[m])):
                        peaks_new = peaks_new[peaks_new != peaks_e[clusters_to_be_deleted[m][m1] - 1]]
                        peaks_new = peaks_new[peaks_new != peaks_e[clusters_to_be_deleted[m][m1]]]

                        peaks_neg_new = peaks_neg_new[peaks_neg_new != peaks_neg_e[clusters_to_be_deleted[m][m1]]]
                peaks_new_tot = []
                for i1 in peaks_new:
                    peaks_new_tot.append(i1)
                for i1 in peaks_new_extra:
                    peaks_new_tot.append(i1)
                peaks_new_tot = np.sort(peaks_new_tot)

            else:
                peaks_new_tot = peaks_e[:]

            textline_con, hierachy = return_contours_of_image(img_patch)
            textline_con_fil = filter_contours_area_of_image(img_patch, textline_con, hierachy, max_area=1, min_area=0.0008)
            y_diff_mean = np.mean(np.diff(peaks_new_tot))  # self.find_contours_mean_y_diff(textline_con_fil)

            sigma_gaus = int(y_diff_mean * (7.0 / 40.0))
            # print(sigma_gaus,'sigma_gaus')
        except:
            sigma_gaus = 12
        if sigma_gaus < 3:
            sigma_gaus = 3
        # print(sigma_gaus,'sigma')

    y_padded_smoothed = gaussian_filter1d(y_padded, sigma_gaus)
    y_padded_up_to_down = -y_padded + np.max(y_padded)
    y_padded_up_to_down_padded = np.zeros(len(y_padded_up_to_down) + 40)
    y_padded_up_to_down_padded[20 : len(y_padded_up_to_down) + 20] = y_padded_up_to_down
    y_padded_up_to_down_padded = gaussian_filter1d(y_padded_up_to_down_padded, sigma_gaus)

    peaks, _ = find_peaks(y_padded_smoothed, height=0)
    peaks_neg, _ = find_peaks(y_padded_up_to_down_padded, height=0)

    peaks_new = peaks[:]
    peaks_neg_new = peaks_neg[:]

    try:
        neg_peaks_max = np.max(y_padded_smoothed[peaks])

        arg_neg_must_be_deleted = np.array(range(len(peaks_neg)))[y_padded_up_to_down_padded[peaks_neg] / float(neg_peaks_max) < 0.24]

        diff_arg_neg_must_be_deleted = np.diff(arg_neg_must_be_deleted)

        arg_diff = np.array(range(len(diff_arg_neg_must_be_deleted)))
        arg_diff_cluster = arg_diff[diff_arg_neg_must_be_deleted > 1]

        clusters_to_be_deleted = []

        if len(arg_diff_cluster) >= 2 and len(arg_diff_cluster) > 0:

            clusters_to_be_deleted.append(arg_neg_must_be_deleted[0 : arg_diff_cluster[0] + 1])
            for i in range(len(arg_diff_cluster) - 1):
                clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[i] + 1 : arg_diff_cluster[i + 1] + 1])
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[arg_diff_cluster[len(arg_diff_cluster) - 1] + 1 :])
        elif len(arg_neg_must_be_deleted) >= 2 and len(arg_diff_cluster) == 0:
            clusters_to_be_deleted.append(arg_neg_must_be_deleted[:])

        if len(arg_neg_must_be_deleted) == 1:
            clusters_to_be_deleted.append(arg_neg_must_be_deleted)

        if len(clusters_to_be_deleted) > 0:
            peaks_new_extra = []
            for m in range(len(clusters_to_be_deleted)):
                min_cluster = np.min(peaks[clusters_to_be_deleted[m]])
                max_cluster = np.max(peaks[clusters_to_be_deleted[m]])
                peaks_new_extra.append(int((min_cluster + max_cluster) / 2.0))
                for m1 in range(len(clusters_to_be_deleted[m])):
                    peaks_new = peaks_new[peaks_new != peaks[clusters_to_be_deleted[m][m1] - 1]]
                    peaks_new = peaks_new[peaks_new != peaks[clusters_to_be_deleted[m][m1]]]

                    peaks_neg_new = peaks_neg_new[peaks_neg_new != peaks_neg[clusters_to_be_deleted[m][m1]]]
            peaks_new_tot = []
            for i1 in peaks_new:
                peaks_new_tot.append(i1)
            for i1 in peaks_new_extra:
                peaks_new_tot.append(i1)
            peaks_new_tot = np.sort(peaks_new_tot)

            # plt.plot(y_padded_up_to_down_padded)
            # plt.plot(peaks_neg,y_padded_up_to_down_padded[peaks_neg],'*')
            # plt.show()

            # plt.plot(y_padded_up_to_down_padded)
            # plt.plot(peaks_neg_new,y_padded_up_to_down_padded[peaks_neg_new],'*')
            # plt.show()

            # plt.plot(y_padded_smoothed)
            # plt.plot(peaks,y_padded_smoothed[peaks],'*')
            # plt.show()

            # plt.plot(y_padded_smoothed)
            # plt.plot(peaks_new_tot,y_padded_smoothed[peaks_new_tot],'*')
            # plt.show()

            peaks = peaks_new_tot[:]
            peaks_neg = peaks_neg_new[:]
    except:
        pass

    else:
        peaks_new_tot = peaks[:]
        peaks = peaks_new_tot[:]
        peaks_neg = peaks_neg_new[:]

    mean_value_of_peaks = np.mean(y_padded_smoothed[peaks])
    std_value_of_peaks = np.std(y_padded_smoothed[peaks])
    peaks_values = y_padded_smoothed[peaks]

    ###peaks_neg = peaks_neg - 20 - 20
    ###peaks = peaks - 20
    peaks_neg_true = peaks_neg[:]
    peaks_pos_true = peaks[:]

    if len(peaks_neg_true) > 0:
        peaks_neg_true = np.array(peaks_neg_true)

        peaks_neg_true = peaks_neg_true - 20 - 20

        # print(peaks_neg_true)
        for i in range(len(peaks_neg_true)):
            img_patch[peaks_neg_true[i] - 6 : peaks_neg_true[i] + 6, :] = 0

    else:
        pass

    if len(peaks_pos_true) > 0:
        peaks_pos_true = np.array(peaks_pos_true)
        peaks_pos_true = peaks_pos_true - 20

        for i in range(len(peaks_pos_true)):
            ##img_patch[peaks_pos_true[i]-8:peaks_pos_true[i]+8,:]=1
            img_patch[peaks_pos_true[i] - 6 : peaks_pos_true[i] + 6, :] = 1
    else:
        pass
    kernel = np.ones((5, 5), np.uint8)

    # img_patch = cv2.erode(img_patch,kernel,iterations = 3)
    #######################img_patch = cv2.erode(img_patch,kernel,iterations = 2)
    img_patch = cv2.erode(img_patch, kernel, iterations=1)
    return img_patch

def filter_small_drop_capitals_from_no_patch_layout(layout_no_patch, layout1):

    drop_only = (layout_no_patch[:, :, 0] == 4) * 1
    contours_drop, hir_on_drop = return_contours_of_image(drop_only)
    contours_drop_parent = return_parent_contours(contours_drop, hir_on_drop)

    areas_cnt_text = np.array([cv2.contourArea(contours_drop_parent[j]) for j in range(len(contours_drop_parent))])
    areas_cnt_text = areas_cnt_text / float(drop_only.shape[0] * drop_only.shape[1])

    contours_drop_parent = [contours_drop_parent[jz] for jz in range(len(contours_drop_parent)) if areas_cnt_text[jz] > 0.001]

    areas_cnt_text = [areas_cnt_text[jz] for jz in range(len(areas_cnt_text)) if areas_cnt_text[jz] > 0.001]

    contours_drop_parent_final = []

    for jj in range(len(contours_drop_parent)):
        x, y, w, h = cv2.boundingRect(contours_drop_parent[jj])
        # boxes.append([int(x), int(y), int(w), int(h)])

        iou_of_box_and_contoure = float(drop_only.shape[0] * drop_only.shape[1]) * areas_cnt_text[jj] / float(w * h) * 100
        height_to_weight_ratio = h / float(w)
        weigh_to_height_ratio = w / float(h)

        if iou_of_box_and_contoure > 60 and weigh_to_height_ratio < 1.2 and height_to_weight_ratio < 2:
            map_of_drop_contour_bb = np.zeros((layout1.shape[0], layout1.shape[1]))
            map_of_drop_contour_bb[y : y + h, x : x + w] = layout1[y : y + h, x : x + w]

            if (((map_of_drop_contour_bb == 1) * 1).sum() / float(((map_of_drop_contour_bb == 5) * 1).sum()) * 100) >= 15:
                contours_drop_parent_final.append(contours_drop_parent[jj])

    layout_no_patch[:, :, 0][layout_no_patch[:, :, 0] == 4] = 0

    layout_no_patch = cv2.fillPoly(layout_no_patch, pts=contours_drop_parent_final, color=(4, 4, 4))

    return layout_no_patch

