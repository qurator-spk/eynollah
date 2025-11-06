from typing import Sequence, Union
from numbers import Number
from functools import partial
import itertools

import cv2
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import Polygon, LineString
from shapely.geometry.polygon import orient
from shapely import set_precision
from shapely.ops import unary_union, nearest_points

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
    return [cv2.boundingRect(contour)
            for contour in contours]

def filter_contours_area_of_image(image, contours, hierarchy, max_area=1.0, min_area=0.0, dilate=0):
    found_polygons_early = []
    for jv, contour in enumerate(contours):
        if len(contour) < 3:  # A polygon cannot have less than 3 points
            continue

        polygon = contour2polygon(contour, dilate=dilate)
        area = polygon.area
        if (area >= min_area * np.prod(image.shape[:2]) and
            area <= max_area * np.prod(image.shape[:2]) and
            hierarchy[0][jv][3] == -1):
            found_polygons_early.append(polygon2contour(polygon))
    return found_polygons_early

def filter_contours_area_of_image_tables(image, contours, hierarchy, max_area=1.0, min_area=0.0, dilate=0):
    found_polygons_early = []
    for jv, contour in enumerate(contours):
        if len(contour) < 3:  # A polygon cannot have less than 3 points
            continue

        polygon = contour2polygon(contour, dilate=dilate)
        # area = cv2.contourArea(contour)
        area = polygon.area
        ##print(np.prod(thresh.shape[:2]))
        # Check that polygon has area greater than minimal area
        # print(hierarchy[0][jv][3],hierarchy )
        if (area >= min_area * np.prod(image.shape[:2]) and
            area <= max_area * np.prod(image.shape[:2]) and
            # hierarchy[0][jv][3]==-1
            True):
            # print(contour[0][0][1])
            found_polygons_early.append(polygon2contour(polygon))
    return found_polygons_early

def find_center_of_contours(contours):
    moments = [cv2.moments(contour) for contour in contours]
    cx = [feat["m10"] / (feat["m00"] + 1e-32)
          for feat in moments]
    cy = [feat["m01"] / (feat["m00"] + 1e-32)
          for feat in moments]
    return cx, cy

def find_new_features_of_contours(contours):
    # areas = np.array([cv2.contourArea(contour) for contour in contours])
    cx, cy = find_center_of_contours(contours)
    slice_x = np.index_exp[:, 0, 0]
    slice_y = np.index_exp[:, 0, 1]
    if any(contour.ndim < 3 for contour in contours):
        slice_x = np.index_exp[:, 0]
        slice_y = np.index_exp[:, 1]
    x_min = np.array([np.min(contour[slice_x]) for contour in contours])
    x_max = np.array([np.max(contour[slice_x]) for contour in contours])
    y_min = np.array([np.min(contour[slice_y]) for contour in contours])
    y_max = np.array([np.max(contour[slice_y]) for contour in contours])
    # dis_x=np.abs(x_max-x_min)
    y_corr_x_min = np.array([contour[np.argmin(contour[slice_x])][slice_y[1:]]
                             for contour in contours])

    return cx, cy, x_min, x_max, y_min, y_max, y_corr_x_min

def find_features_of_contours(contours):
    y_min = np.array([np.min(contour[:,0,1]) for contour in contours])
    y_max = np.array([np.max(contour[:,0,1]) for contour in contours])

    return y_min, y_max

def return_parent_contours(contours, hierarchy):
    contours_parent = [contours[i]
                       for i in range(len(contours))
                       if hierarchy[0][i][3] == -1]
    return contours_parent

def return_contours_of_interested_region(region_pre_p, label, min_area=0.0002):
    # pixels of images are identified by 5
    if region_pre_p.ndim == 3:
        cnts_images = (region_pre_p[:, :, 0] == label) * 1
    else:
        cnts_images = (region_pre_p[:, :] == label) * 1
    _, thresh = cv2.threshold(cnts_images.astype(np.uint8), 0, 255, 0)

    contours_imgs, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_imgs = return_parent_contours(contours_imgs, hierarchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hierarchy,
                                                         max_area=1, min_area=min_area)
    return contours_imgs

def do_work_of_contours_in_image(contour, index_r_con, img, slope_first):
    img_copy = np.zeros(img.shape[:2], dtype=np.uint8)
    img_copy = cv2.fillPoly(img_copy, pts=[contour], color=1)

    img_copy = rotation_image_new(img_copy, -slope_first)
    _, thresh = cv2.threshold(img_copy, 0, 255, 0)

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
        img_copy = np.zeros(img.shape[:2], dtype=np.uint8)
        img_copy = cv2.fillPoly(img_copy, pts=[cnts[i]], color=1)

        # plt.imshow(img_copy)
        # plt.show()

        # print(img.shape,'img')
        img_copy = rotation_image_new(img_copy, -slope_first)
        ##print(img_copy.shape,'img_copy')
        # plt.imshow(img_copy)
        # plt.show()

        _, thresh = cv2.threshold(img_copy, 0, 255, 0)

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
        img_copy = np.zeros(img.shape[:2], dtype=np.uint8)
        img_copy = cv2.fillPoly(img_copy, pts=[cnt // zoom], color=1)

        img_copy = rotation_image_new(img_copy, -slope_first).astype(np.uint8)
        _, thresh = cv2.threshold(img_copy, 0, 255, 0)

        cont_int, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_int[0][:, 0, 0] = cont_int[0][:, 0, 0] + np.abs(img_copy.shape[1] - img.shape[1])
        cont_int[0][:, 0, 1] = cont_int[0][:, 0, 1] + np.abs(img_copy.shape[0] - img.shape[0])
        cnts_org.append(cont_int[0] * zoom)

    return cnts_org

def do_back_rotation_and_get_cnt_back(contour_par, index_r_con, img, slope_first, confidence_matrix):
    img_copy = np.zeros(img.shape[:2], dtype=np.uint8)
    img_copy = cv2.fillPoly(img_copy, pts=[contour_par], color=1)
    confidence_matrix_mapped_with_contour = confidence_matrix * img_copy
    confidence_contour = np.sum(confidence_matrix_mapped_with_contour) / float(np.sum(img_copy))

    img_copy = rotation_image_new(img_copy, -slope_first).astype(np.uint8)
    _, thresh = cv2.threshold(img_copy, 0, 255, 0)

    cont_int, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cont_int)==0:
        cont_int = [contour_par]
        confidence_contour = 0
    else:
        cont_int[0][:, 0, 0] = cont_int[0][:, 0, 0] + np.abs(img_copy.shape[1] - img.shape[1])
        cont_int[0][:, 0, 1] = cont_int[0][:, 0, 1] + np.abs(img_copy.shape[0] - img.shape[0])
    return cont_int[0], index_r_con, confidence_contour

def get_textregion_contours_in_org_image_light(cnts, img, confidence_matrix):
    if not len(cnts):
        return []

    confidence_matrix = cv2.resize(confidence_matrix,
                                   (img.shape[1] // 6, img.shape[0] // 6),
                                   interpolation=cv2.INTER_NEAREST)
    confs = []
    for cnt in cnts:
        cnt_mask = np.zeros(confidence_matrix.shape)
        cnt_mask = cv2.fillPoly(cnt_mask, pts=[cnt // 6], color=1.0)
        confs.append(np.sum(confidence_matrix * cnt_mask) / np.sum(cnt_mask))
    return confs

def return_contours_of_interested_textline(region_pre_p, label):
    # pixels of images are identified by 5
    if region_pre_p.ndim == 3:
        cnts_images = (region_pre_p[:, :, 0] == label) * 1
    else:
        cnts_images = (region_pre_p[:, :] == label) * 1
    _, thresh = cv2.threshold(cnts_images.astype(np.uint8), 0, 255, 0)
    contours_imgs, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hierarchy)
    contours_imgs = filter_contours_area_of_image_tables(
        thresh, contours_imgs, hierarchy, max_area=1, min_area=0.000000003)
    return contours_imgs

def return_contours_of_image(image):
    if len(image.shape) == 2:
        image = image.astype(np.uint8)
        imgray = image
    else:
        image = image.astype(np.uint8)
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def dilate_textline_contours(all_found_textline_polygons):
    return [[polygon2contour(contour2polygon(contour, dilate=6))
             for contour in region]
            for region in all_found_textline_polygons]

def dilate_textregion_contours(all_found_textline_polygons):
    return [polygon2contour(contour2polygon(contour, dilate=6))
            for contour in all_found_textline_polygons]

def contour2polygon(contour: Union[np.ndarray, Sequence[Sequence[Sequence[Number]]]], dilate=0):
    polygon = Polygon([point[0] for point in contour])
    if dilate:
        polygon = polygon.buffer(dilate)
    if polygon.geom_type == 'GeometryCollection':
        # heterogeneous result: filter zero-area shapes (LineString, Point)
        polygon = unary_union([geom for geom in polygon.geoms if geom.area > 0])
    if polygon.geom_type == 'MultiPolygon':
        # homogeneous result: construct convex hull to connect
        polygon = join_polygons(polygon.geoms)
    return make_valid(polygon)

def polygon2contour(polygon: Polygon) -> np.ndarray:
    polygon = np.array(polygon.exterior.coords[:-1], dtype=int)
    return np.maximum(0, polygon).astype(int)[:, np.newaxis]

def make_intersection(poly1, poly2):
    interp = poly1.intersection(poly2)
    # post-process
    if interp.is_empty or interp.area == 0.0:
        return None
    if interp.geom_type == 'GeometryCollection':
        # heterogeneous result: filter zero-area shapes (LineString, Point)
        interp = unary_union([geom for geom in interp.geoms if geom.area > 0])
    if interp.geom_type == 'MultiPolygon':
        # homogeneous result: construct convex hull to connect
        interp = join_polygons(interp.geoms)
    assert interp.geom_type == 'Polygon', interp.wkt
    interp = make_valid(interp)
    return interp

def make_valid(polygon: Polygon) -> Polygon:
    """Ensures shapely.geometry.Polygon object is valid by repeated rearrangement/simplification/enlargement."""
    def isint(x):
        return isinstance(x, int) or int(x) == x
    # make sure rounding does not invalidate
    if not all(map(isint, np.array(polygon.exterior.coords).flat)) and polygon.minimum_clearance < 1.0:
        polygon = Polygon(np.round(polygon.exterior.coords))
    points = list(polygon.exterior.coords[:-1])
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

def join_polygons(polygons: Sequence[Polygon], scale=20) -> Polygon:
    """construct concave hull (alpha shape) from input polygons by connecting their pairwise nearest points"""
    # ensure input polygons are simply typed and all oriented equally
    polygons = [orient(poly)
                for poly in itertools.chain.from_iterable(
                        [poly.geoms
                         if poly.geom_type in ['MultiPolygon', 'GeometryCollection']
                         else [poly]
                         for poly in polygons])]
    npoly = len(polygons)
    if npoly == 1:
        return polygons[0]
    # find min-dist path through all polygons (travelling salesman)
    pairs = itertools.combinations(range(npoly), 2)
    dists = np.zeros((npoly, npoly), dtype=float)
    for i, j in pairs:
        dist = polygons[i].distance(polygons[j])
        if dist < 1e-5:
            dist = 1e-5 # if pair merely touches, we still need to get an edge
        dists[i, j] = dist
        dists[j, i] = dist
    dists = minimum_spanning_tree(dists, overwrite=True)
    # add bridge polygons (where necessary)
    for prevp, nextp in zip(*dists.nonzero()):
        prevp = polygons[prevp]
        nextp = polygons[nextp]
        nearest = nearest_points(prevp, nextp)
        bridgep = orient(LineString(nearest).buffer(max(1, scale/5), resolution=1), -1)
        polygons.append(bridgep)
    jointp = unary_union(polygons)
    if jointp.geom_type == 'MultiPolygon':
        jointp = unary_union(jointp.geoms)
    assert jointp.geom_type == 'Polygon', jointp.wkt
    # follow-up calculations will necessarily be integer;
    # so anticipate rounding here and then ensure validity
    jointp2 = set_precision(jointp, 1.0, mode="keep_collapsed")
    if jointp2.geom_type != 'Polygon' or not jointp2.is_valid:
        jointp2 = Polygon(np.round(jointp.exterior.coords))
        jointp2 = make_valid(jointp2)
    assert jointp2.geom_type == 'Polygon', jointp2.wkt
    return jointp2
