from typing import Sequence, Union
from numbers import Number
from functools import partial
import itertools

import cv2
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import Polygon, LineString
from shapely.geometry.polygon import orient
from shapely import set_precision, affinity
from shapely.ops import unary_union, nearest_points

from .rotate import rotate_image, rotation_image_new

def contours_in_same_horizon(cy_main_hor):
    """
    Takes an array of y coords, identifies all pairs among them
    which are close to each other, and returns all such pairs
    by index into the array.
    """
    sort = np.argsort(cy_main_hor)
    same = np.diff(cy_main_hor[sort]) <= 20
    # groups = np.split(sort, np.arange(len(cy_main_hor) - 1)[~same] + 1)
    same = np.flatnonzero(same)
    return np.stack((sort[:-1][same], sort[1:][same])).T

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
        if (area >= min_area * image.size and
            area <= max_area * image.size and
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

def return_contours_of_interested_region(region_pre_p, label, min_area=0.0002, dilate=0):
    if region_pre_p.ndim == 3:
        mask = (region_pre_p[:, :, 0] == label).astype(np.uint8)
    else:
        mask = (region_pre_p[:, :] == label).astype(np.uint8)

    contours_imgs, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_imgs = return_parent_contours(contours_imgs, hierarchy)
    contours_imgs = filter_contours_area_of_image_tables(mask, contours_imgs, hierarchy,
                                                         max_area=1,
                                                         min_area=min_area,
                                                         dilate=dilate)
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

def get_textregion_confidences_old(cnts, img, slope_first):
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

def get_region_confidences(cnts, confidence_matrix):
    if not len(cnts):
        return []

    height, width = confidence_matrix.shape
    confidence_matrix = cv2.resize(confidence_matrix,
                                   (width // 6, height // 6),
                                   interpolation=cv2.INTER_NEAREST)
    confs = []
    for cnt in cnts:
        cnt_mask = np.zeros_like(confidence_matrix)
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
    from . import ensure_array
    return [ensure_array(
        [polygon2contour(contour2polygon(contour, dilate=6))
         for contour in region])
            for region in all_found_textline_polygons]

def dilate_textregion_contours(all_found_textregion_polygons):
    from . import ensure_array
    return ensure_array(
        [polygon2contour(contour2polygon(contour, dilate=6))
         for contour in all_found_textregion_polygons])

def match_deskewed_contours(slope_deskew, contours_o, contours_d, shape_o, shape_d):
    from . import ensure_array

    cntareas_o = np.array([cv2.contourArea(contour) for contour in contours_o])
    cntareas_d = np.array([cv2.contourArea(contour) for contour in contours_d])
    cntareas_o = cntareas_o / float(np.prod(shape_o[:2]))
    cntareas_d = cntareas_d / float(np.prod(shape_d[:2]))

    contours_o = ensure_array(contours_o)
    contours_d = ensure_array(contours_d)

    sort_o = np.argsort(cntareas_o)
    sort_d = np.argsort(cntareas_d)
    contours_o = contours_o[sort_o]
    contours_d = contours_d[sort_d]
    cntareas_o = cntareas_o[sort_o]
    cntareas_d = cntareas_d[sort_d]

    centers_o = np.stack(find_center_of_contours(contours_o)) # [2, N]
    centers_d = np.stack(find_center_of_contours(contours_d)) # [2, N]
    center0_o = centers_o[:, -1:] # [2, 1]
    center0_d = centers_d[:, -1:] # [2, 1]

    # find the largest among the largest 5 deskewed contours
    # that is also closest to the largest original contour
    last5_centers_d = centers_d[:, -5:]
    dists_d = np.linalg.norm(center0_o - last5_centers_d, axis=0)
    ind_largest = len(contours_d) - last5_centers_d.shape[1] + np.argmin(dists_d)
    center0_d[:, 0] = centers_d[:, ind_largest]

    # order new contours the same way as the undeskewed contours
    # (by calculating the offset of the largest contours, respectively,
    #  of the new and undeskewed image; then for each contour,
    #  finding the closest new contour, with proximity calculated
    #  as distance of their centers modulo offset vector)
    h_o, w_o = shape_o[:2]
    center_o = (w_o // 2, h_o // 2)
    M = cv2.getRotationMatrix2D(center_o, slope_deskew, 1.0)
    M_22 = np.array(M)[:2, :2]
    center0_o = np.dot(M_22, center0_o) # [2, 1]
    offset = center0_o - center0_d # [2, 1]

    centers_o = np.dot(M_22, centers_o) - offset # [2,N]
    # add dimension for area (so only contours of similar size will be considered close)
    centers_o = np.append(centers_o, cntareas_o[np.newaxis], axis=0)
    centers_d = np.append(centers_d, cntareas_d[np.newaxis], axis=0)

    dists = np.zeros((len(contours_o), len(contours_d)))
    for i in range(len(contours_o)):
        dists[i] = np.linalg.norm(centers_o[:, i: i + 1] - centers_d, axis=0)
    corresp = np.zeros(dists.shape, dtype=bool)
    # keep searching next-closest until at least one correspondence on each side
    while not np.all(corresp.sum(axis=1)) or not np.all(corresp.sum(axis=0)):
        idx = np.nanargmin(dists)
        i, j = np.unravel_index(idx, dists.shape)
        dists[i, j] = np.nan
        corresp[i, j] = True
    # print("original/deskewed adjacency", corresp.nonzero())
    contours_d_ordered = contours_d[np.argmax(corresp, axis=1)]
    # from matplotlib import pyplot as plt
    # img1 = np.zeros(shape_d[:2], dtype=np.uint8)
    # for i in range(len(contours_o)):
    #     cv2.fillPoly(img1, pts=[contours_d_ordered[i]], color=i + 1)
    # plt.subplot(1, 4, 1, title="direct corresp contours")
    # plt.imshow(img1)
    # img2 = np.zeros(shape_d[:2], dtype=np.uint8)
    # join deskewed regions mapping to single original ones
    for i in range(len(contours_o)):
        if np.count_nonzero(corresp[i]) > 1:
            indices = np.flatnonzero(corresp[i])
            # print("joining", indices)
            polygons_d = [contour2polygon(contour)
                          for contour in contours_d[indices]]
            contour_d_joined = polygon2contour(join_polygons(polygons_d))
            contours_d_ordered[i] = contour_d_joined
    #         cv2.fillPoly(img2, pts=[contour_d_joined], color=i + 1)
    # plt.subplot(1, 4, 2, title="joined contours")
    # plt.imshow(img2)
    # img3 = np.zeros(shape_d[:2], dtype=np.uint8)
    # split deskewed regions mapping to multiple original ones
    def deskew(polygon):
        polygon = affinity.rotate(polygon, -slope_deskew, origin=center_o)
        #polygon = affinity.translate(polygon, *offset.squeeze())
        return polygon
    for j in range(len(contours_d)):
        if np.count_nonzero(corresp[:, j]) > 1:
            indices = np.flatnonzero(corresp[:, j])
            # print("splitting along", indices)
            polygons_o = [deskew(contour2polygon(contour))
                          for contour in contours_o[indices]]
            polygon_d = contour2polygon(contours_d[j])
            polygons_d = [make_intersection(polygon_d, polygon)
                          for polygon in polygons_o]
            # ignore where there is no actual overlap
            indices = indices[np.flatnonzero(polygons_d)]
            contours_d_joined = [polygon2contour(polygon_d)
                                 for polygon_d in polygons_d
                                 if polygon_d]
            contours_d_ordered[indices] = contours_d_joined
    #         cv2.fillPoly(img3, pts=contours_d_joined, color=j + 1)
    # plt.subplot(1, 4, 3, title="split contours")
    # plt.imshow(img3)
    # img4 = np.zeros(shape_d[:2], dtype=np.uint8)
    # for i in range(len(contours_o)):
    #     cv2.fillPoly(img4, pts=[contours_d_ordered[i]], color=i + 1)
    # plt.subplot(1, 4, 4, title="result contours")
    # plt.imshow(img4)
    # plt.show()
    # from matplotlib import patches as ptchs
    # plt.subplot(1, 2, 1, title="undeskewed")
    # plt.imshow(mask_o)
    # centers_o = np.stack(find_center_of_contours(contours_o)) # [2, N]
    # for i in range(len(contours_o)):
    #     cnt = contours_o[i]
    #     ctr = centers_o[:, i]
    #     plt.gca().add_patch(ptchs.Polygon(cnt[:, 0], closed=False, fill=False, color='blue'))
    #     plt.gca().scatter(ctr[0], ctr[1], 20, c='blue', marker='x')
    #     plt.gca().text(ctr[0], ctr[1], str(i), c='blue')
    # plt.subplot(1, 2, 2, title="deskewed")
    # plt.imshow(mask_d)
    # centers_d = np.stack(find_center_of_contours(contours_d_ordered)) # [2, N]
    # for i in range(len(contours_o)):
    #     cnt = contours_o[i]
    #     cnt = polygon2contour(deskew(contour2polygon(cnt)))
    #     plt.gca().add_patch(ptchs.Polygon(cnt[:, 0], closed=False, fill=False, color='blue'))
    # for i in range(len(contours_d_ordered)):
    #     cnt = contours_d_ordered[i]
    #     ctr = centers_d[:, i]
    #     plt.gca().add_patch(ptchs.Polygon(cnt[:, 0], closed=False, fill=False, color='red'))
    #     plt.gca().scatter(ctr[0], ctr[1], 20, c='red', marker='x')
    #     plt.gca().text(ctr[0], ctr[1], str(i), c='red')
    # plt.show()
    invsort_o = np.argsort(sort_o)
    return contours_d_ordered[invsort_o]

def estimate_skew_contours(contours):
    if not len(contours):
        raise ValueError("not enough contours")
    _, size_in, angle_in = zip(*map(cv2.minAreaRect, contours))
    w_in, h_in = np.array(size_in).T
    angle_in = np.array(angle_in)
    transposed = h_in > w_in
    # print("transposed", transposed, angle_in)
    w_in[transposed], h_in[transposed] = h_in[transposed], w_in[transposed]
    angle_in[transposed] -= 90
    usable = w_in > 3 * h_in
    # print("usable aspect", w_in / h_in, usable, angle_in[usable])
    if not np.any(usable):
        raise ValueError("not enough contours with high aspect ratio")
    w_avg = np.median(w_in[usable])
    w_dev = w_in[usable] / w_avg
    usable[usable] = (0.67 <= w_dev) & (w_dev <= 1.33)
    # print("usable width", usable, w_in[usable], angle_in[usable])
    if not np.any(usable):
        raise ValueError("not enough contours with consistent length")
    angle_avg = np.median(angle_in[usable])
    angle_dev = np.abs(angle_in[usable] - angle_avg)
    usable[usable] = (angle_dev <= 2 * np.median(angle_dev))
    # print("usable angle", usable, angle_in[usable], np.mean(angle_in[usable]))
    if not np.any(usable):
        raise ValueError("not enough contours with consistent angle")
    return np.mean(angle_in[usable])

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
