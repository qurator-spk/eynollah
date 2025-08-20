from functools import partial
import cv2
import numpy as np
from shapely.geometry import Polygon

from .rotate import rotate_image, rotation_image_new

def contours_in_same_horizon(cy_main_hor):
    X1 = np.zeros((len(cy_main_hor), len(cy_main_hor)))
    X2 = np.zeros((len(cy_main_hor), len(cy_main_hor)))

    X1[0::1, :] = cy_main_hor[:]
    X2 = X1.T

    X_dif = np.abs(X2 - X1)
    args_help = np.array(range(len(cy_main_hor)))
    all_args = []
    for i in range(len(cy_main_hor)):
        list_h = list(args_help[X_dif[i, :] <= 20])
        list_h.append(i)
        if len(list_h) > 1:
            all_args.append(list(set(list_h)))
    return np.unique(np.array(all_args, dtype=object))

def find_contours_mean_y_diff(contours_main):
    M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
    cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
    return np.mean(np.diff(np.sort(np.array(cy_main))))

def get_text_region_boxes_by_given_contours(contours):
    boxes = []
    contours_new = []
    for jj in range(len(contours)):
        box = cv2.boundingRect(contours[jj])
        boxes.append(box)
        contours_new.append(contours[jj])

    return boxes, contours_new

def filter_contours_area_of_image(image, contours, hierarchy, max_area, min_area):
    found_polygons_early = []
    for jv,c in enumerate(contours):
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue

        polygon = Polygon([point[0] for point in c])
        area = polygon.area
        if (area >= min_area * np.prod(image.shape[:2]) and
            area <= max_area * np.prod(image.shape[:2]) and
            hierarchy[0][jv][3] == -1):
            found_polygons_early.append(np.array([[point]
                                                  for point in polygon.exterior.coords[:-1]], dtype=np.uint))
    return found_polygons_early

def filter_contours_area_of_image_tables(image, contours, hierarchy, max_area, min_area):
    found_polygons_early = []
    for jv,c in enumerate(contours):
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue

        polygon = Polygon([point[0] for point in c])
        # area = cv2.contourArea(c)
        area = polygon.area
        ##print(np.prod(thresh.shape[:2]))
        # Check that polygon has area greater than minimal area
        # print(hierarchy[0][jv][3],hierarchy )
        if (area >= min_area * np.prod(image.shape[:2]) and
            area <= max_area * np.prod(image.shape[:2]) and
            # hierarchy[0][jv][3]==-1
            True):
            # print(c[0][0][1])
            found_polygons_early.append(np.array([[point]
                                                  for point in polygon.exterior.coords[:-1]], dtype=np.int32))
    return found_polygons_early

def find_new_features_of_contours(contours_main):
    areas_main = np.array([cv2.contourArea(contours_main[j])
                           for j in range(len(contours_main))])
    M_main = [cv2.moments(contours_main[j])
              for j in range(len(contours_main))]
    cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32))
               for j in range(len(M_main))]
    cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32))
               for j in range(len(M_main))]
    try:
        x_min_main = np.array([np.min(contours_main[j][:, 0, 0])
                               for j in range(len(contours_main))])
        argmin_x_main = np.array([np.argmin(contours_main[j][:, 0, 0])
                                  for j in range(len(contours_main))])
        x_min_from_argmin = np.array([contours_main[j][argmin_x_main[j], 0, 0]
                                      for j in range(len(contours_main))])
        y_corr_x_min_from_argmin = np.array([contours_main[j][argmin_x_main[j], 0, 1]
                                             for j in range(len(contours_main))])
        x_max_main = np.array([np.max(contours_main[j][:, 0, 0])
                               for j in range(len(contours_main))])
        y_min_main = np.array([np.min(contours_main[j][:, 0, 1])
                               for j in range(len(contours_main))])
        y_max_main = np.array([np.max(contours_main[j][:, 0, 1])
                               for j in range(len(contours_main))])
    except:
        x_min_main = np.array([np.min(contours_main[j][:, 0])
                               for j in range(len(contours_main))])
        argmin_x_main = np.array([np.argmin(contours_main[j][:, 0])
                                  for j in range(len(contours_main))])
        x_min_from_argmin = np.array([contours_main[j][argmin_x_main[j], 0]
                                      for j in range(len(contours_main))])
        y_corr_x_min_from_argmin = np.array([contours_main[j][argmin_x_main[j], 1]
                                             for j in range(len(contours_main))])
        x_max_main = np.array([np.max(contours_main[j][:, 0])
                               for j in range(len(contours_main))])
        y_min_main = np.array([np.min(contours_main[j][:, 1])
                               for j in range(len(contours_main))])
        y_max_main = np.array([np.max(contours_main[j][:, 1])
                               for j in range(len(contours_main))])
    # dis_x=np.abs(x_max_main-x_min_main)

    return cx_main, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, y_corr_x_min_from_argmin

def find_features_of_contours(contours_main):
    areas_main=np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
    M_main=[cv2.moments(contours_main[j]) for j in range(len(contours_main))]
    cx_main=[(M_main[j]['m10']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
    cy_main=[(M_main[j]['m01']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
    x_min_main=np.array([np.min(contours_main[j][:,0,0]) for j in range(len(contours_main))])
    x_max_main=np.array([np.max(contours_main[j][:,0,0]) for j in range(len(contours_main))])

    y_min_main=np.array([np.min(contours_main[j][:,0,1]) for j in range(len(contours_main))])
    y_max_main=np.array([np.max(contours_main[j][:,0,1]) for j in range(len(contours_main))])

    return y_min_main, y_max_main

def return_parent_contours(contours, hierarchy):
    contours_parent = [contours[i]
                       for i in range(len(contours))
                       if hierarchy[0][i][3] == -1]
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

    contours_imgs, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_imgs = return_parent_contours(contours_imgs, hierarchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hierarchy,
                                                         max_area=1, min_area=min_area)
    return contours_imgs

def do_work_of_contours_in_image(contour, index_r_con, img, slope_first):
    img_copy = np.zeros(img.shape)
    img_copy = cv2.fillPoly(img_copy, pts=[contour], color=(1, 1, 1))

    img_copy = rotation_image_new(img_copy, -slope_first)
    img_copy = img_copy.astype(np.uint8)
    imgray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    cont_int, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cont_int[0][:, 0, 0] = cont_int[0][:, 0, 0] + np.abs(img_copy.shape[1] - img.shape[1])
    cont_int[0][:, 0, 1] = cont_int[0][:, 0, 1] + np.abs(img_copy.shape[0] - img.shape[0])

    return cont_int[0], index_r_con

def get_textregion_contours_in_org_image_multi(cnts, img, slope_first, map=map):
    if not len(cnts):
        return [], []
    results = map(partial(do_work_of_contours_in_image,
                          img=img,
                          slope_first=slope_first,
                          ),
                  cnts, range(len(cnts)))
    return tuple(zip(*results))

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

    return cnts_org

def get_textregion_contours_in_org_image_light_old(cnts, img, slope_first):
    zoom = 3
    img = cv2.resize(img, (img.shape[1] // zoom,
                           img.shape[0] // zoom),
                     interpolation=cv2.INTER_NEAREST)
    cnts_org = []
    for cnt in cnts:
        img_copy = np.zeros(img.shape)
        img_copy = cv2.fillPoly(img_copy, pts=[(cnt / zoom).astype(int)], color=(1, 1, 1))

        img_copy = rotation_image_new(img_copy, -slope_first).astype(np.uint8)
        imgray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        cont_int, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_int[0][:, 0, 0] = cont_int[0][:, 0, 0] + np.abs(img_copy.shape[1] - img.shape[1])
        cont_int[0][:, 0, 1] = cont_int[0][:, 0, 1] + np.abs(img_copy.shape[0] - img.shape[0])
        cnts_org.append(cont_int[0] * zoom)

    return cnts_org

def do_back_rotation_and_get_cnt_back(contour_par, index_r_con, img, slope_first, confidence_matrix):
    img_copy = np.zeros(img.shape)
    img_copy = cv2.fillPoly(img_copy, pts=[contour_par], color=(1, 1, 1))
    confidence_matrix_mapped_with_contour = confidence_matrix * img_copy[:,:,0]
    confidence_contour = np.sum(confidence_matrix_mapped_with_contour) / float(np.sum(img_copy[:,:,0]))

    img_copy = rotation_image_new(img_copy, -slope_first).astype(np.uint8)
    imgray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    cont_int, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cont_int)==0:
        cont_int = []
        cont_int.append(contour_par)
        confidence_contour = 0
    else:
        cont_int[0][:, 0, 0] = cont_int[0][:, 0, 0] + np.abs(img_copy.shape[1] - img.shape[1])
        cont_int[0][:, 0, 1] = cont_int[0][:, 0, 1] + np.abs(img_copy.shape[0] - img.shape[0])
    return cont_int[0], index_r_con, confidence_contour

def get_textregion_contours_in_org_image_light(cnts, img, confidence_matrix, map=map):
    if not len(cnts):
        return [], []

    confs = []
    for cnt in cnts:
        cnt_mask = np.zeros(confidence_matrix.shape)
        cnt_mask = cv2.fillPoly(cnt_mask, pts=[cnt], color=1.0)
        confs.append(np.sum(confidence_matrix * cnt_mask) / np.sum(cnt_mask))
    return cnts, confs

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
    contours_imgs, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hierarchy)
    contours_imgs = filter_contours_area_of_image_tables(
        thresh, contours_imgs, hierarchy, max_area=1, min_area=0.000000003)
    return contours_imgs

def return_contours_of_image(image):
    if len(image.shape) == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image = image.astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

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

    contours_imgs, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_imgs = return_parent_contours(contours_imgs, hierarchy)
    contours_imgs = filter_contours_area_of_image_tables(
        thresh, contours_imgs, hierarchy, max_area=1, min_area=min_size)

    return contours_imgs

def return_contours_of_interested_region_by_size(region_pre_p, pixel, min_area, max_area):
    # pixels of images are identified by 5
    if len(region_pre_p.shape) == 3:
        cnts_images = (region_pre_p[:, :, 0] == pixel) * 1
    else:
        cnts_images = (region_pre_p[:, :] == pixel) * 1
    cnts_images = cnts_images.astype(np.uint8)
    cnts_images = np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
    imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours_imgs, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hierarchy)
    contours_imgs = filter_contours_area_of_image_tables(
        thresh, contours_imgs, hierarchy, max_area=max_area, min_area=min_area)

    img_ret = np.zeros((region_pre_p.shape[0], region_pre_p.shape[1], 3))
    img_ret = cv2.fillPoly(img_ret, pts=contours_imgs, color=(1, 1, 1))

    return img_ret[:, :, 0]

def make_valid(polygon: Polygon) -> Polygon:
    """Ensures shapely.geometry.Polygon object is valid by repeated rearrangement/simplification/enlargement."""
    def isint(x):
        return isinstance(x, int) or int(x) == x
    # make sure rounding does not invalidate
    if not all(map(isint, np.array(polygon.exterior.coords).flat)) and polygon.minimum_clearance < 1.0:
        polygon = Polygon(np.round(polygon.exterior.coords))
    points = list(polygon.exterior.coords)
    # try by re-arranging points
    for split in range(1, len(points)):
        if polygon.is_valid or polygon.simplify(polygon.area).is_valid:
            break
        # simplification may not be possible (at all) due to ordering
        # in that case, try another starting point
        polygon = Polygon(points[-split:]+points[:-split])
    # try by simplification
    for tolerance in range(int(polygon.area + 1.5)):
        if polygon.is_valid:
            break
        # simplification may require a larger tolerance
        polygon = polygon.simplify(tolerance + 1)
    # try by enlarging
    for tolerance in range(1, int(polygon.area + 2.5)):
        if polygon.is_valid:
            break
        # enlargement may require a larger tolerance
        polygon = polygon.buffer(tolerance)
    assert polygon.is_valid, polygon.wkt
    return polygon
