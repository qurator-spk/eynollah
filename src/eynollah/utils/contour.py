from __future__ import annotations
from collections.abc import Sequence
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

from .rotate import rotate_image

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

def filter_contours_area_of_image(image, contours, hierarchy, max_area=1.0, min_area=0.0):
    found_polygons_early = []
    for jv, contour in enumerate(contours):
        if len(contour) < 3:  # A polygon cannot have less than 3 points
            continue

        area = cv2.contourArea(contour)
        if (area >= min_area * np.prod(image.shape[:2]) and
            area <= max_area * np.prod(image.shape[:2]) and
            hierarchy[0][jv][3] == -1):
            found_polygons_early.append(contour)
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

def return_contours_of_class(region_pre_p, label, min_area=0.0, holes=False):
    mask = (region_pre_p == label).astype(np.uint8)
    if holes:
        min_area *= region_pre_p.size
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not len(contours):
            return []
        areas = [cv2.contourArea(contour) for contour in contours]
        parents = []
        for contour, area, relations in zip(contours, areas, hierarchy[0]):
            if len(contour) < 3:  # A polygon cannot have less than 3 points
                continue
            if relations[3] == -1: # parent
                if area < min_area:
                    continue
                children = []
                child = relations[2] # next_child
                while child >= 0:
                    relations = hierarchy[0][child]
                    area -= areas[child]
                    if len(contours[child]) >= 4 and areas[child] >= min_area:
                        children.append(contours[child])
                    child = relations[0] # next
                if area < min_area:
                    continue
                parents.append((contour, children))
        # open holes
        contours = []
        for contour, children in parents:
            if len(children):
                poly = contour2polygon(contour)
                interiors = [contour2polygon(interior) for interior in children]
                # from shapely.plotting import plot_polygon
                # from matplotlib import pyplot as plt
                # plt.figure("child contours")
                # plt.subplot(2, 2, 1, title="original")
                # plot_polygon(Polygon(shell=poly, holes=interiors))
                new_interior = join_polygons(interiors)
                # plt.subplot(2, 2, 2, title="new_interior")
                # plot_polygon(poly)
                # plot_polygon(new_interior, color='r')
                bridge = bridge_polygons(poly.exterior, orient(new_interior, -1))
                # plt.subplot(2, 2, 3, title="bridge")
                # plot_polygon(poly)
                # plot_polygon(bridge)
                poly = poly.difference(bridge).difference(new_interior)
                # plt.subplot(2, 2, 4, title="new")
                # plot_polygon(poly)
                # plt.show()
                contour = polygon2contour(ensure_polygon(poly))
            contours.append(contour)
        return contours

    # filter_contours_area_of_image also allows non-children only,
    # so instead of a tree we can retrieve only the external contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours_area_of_image(mask, contours, hierarchy,
                                             max_area=1.0,
                                             min_area=min_area)
    return contours

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
        cnt_area = np.sum(cnt_mask)
        if cnt_area:
            cnt_conf = np.sum(confidence_matrix * cnt_mask) / cnt_area
        else:
            cnt_conf = 0.
        confs.append(cnt_conf)
    return confs

def rotate_contours(
        contours: np.ndarray | Sequence[np.ndarray],
        slope_deskew: float,
        shape_o: tuple[int, int],
) -> list[np.ndarray]:
    # rotate_image() does not enlarge canvas,
    # so our calculation must compensate
    h_o, w_o = shape_o
    M = cv2.getRotationMatrix2D((0.5 * w_o, 0.5 * h_o), -slope_deskew, 1.0)[:2, :2]
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    off = np.array([[0.5 * (w_o * cos + h_o * sin - w_o),
                     0.5 * (w_o * sin + h_o * cos - h_o)]],
                   dtype=int)
    # no idea why this is necessary...
    if slope_deskew > 0:
        off[0, 1] = -off[0, 1]
    else:
        off[0, 0] = -off[0, 0]
    # apply transformation
    contours = [np.dot(cont, M).astype(int) - off
                for cont in contours]
    # clip to (unchanged) canvas
    return [np.maximum(0, np.minimum([w_o, h_o], cont))
            for cont in contours]

def estimate_skew_contours(contours):
    if not len(contours):
        raise ValueError("not enough contours")
    _, size_in, angle_in = zip(*map(cv2.minAreaRect, contours))
    w_in, h_in = np.array(size_in).T
    angle_in = np.array(angle_in)
    # 1. depending on how contours are oriented,
    # and where they start, minAreaRect can present
    # either side as width or height; so we first
    # need to normalise
    transposed = h_in > w_in
    # print("transposed", transposed, angle_in)
    w_in[transposed], h_in[transposed] = h_in[transposed], w_in[transposed]
    angle_in[transposed] -= 90
    # 2. now we look at aspect ratio: too short
    # textlines do not yield reliable angles
    usable = w_in > 2.5 * h_in
    # print("usable aspect", w_in / h_in, usable, angle_in[usable])
    if not np.any(usable):
        raise ValueError("not enough contours with high aspect ratio")
    # 3. next, get rid of outliers regarding length
    w_avg = np.median(w_in[usable])
    w_dev = w_in[usable] / w_avg
    usable[usable] = (0.67 <= w_dev) & (w_dev <= 1.33)
    # print("usable length", w_in[usable] / w_avg, usable, angle_in[usable])
    if not np.any(usable):
        raise ValueError("not enough contours with consistent length")
    if np.count_nonzero(usable) == 1:
        return angle_in[usable][0]
    # 4. there is no way to distinguish between +90 and -89.9 here,
    # so map to [0,180] when calculating averages, then map back to [-90,90]
    # (we don't want -90 and +89 to average zero, or +1 and +179 to average 90)
    angles = angle_in[usable]
    if transposed := np.median(np.abs(angles)) >= 45:
        angles %= 180
    angle_avg = np.median(angles)
    angle_dev = np.abs(angles - angle_avg)
    usable[usable] = (angle_dev <= 2 * np.median(angle_dev))
    # print("usable angle", usable, angle_in[usable])
    if not np.any(usable):
        raise ValueError("not enough contours with consistent angle")
    if transposed:
        angle = 90 - (90 - np.mean(angle_in[usable] % 180)) % 180
    else:
        angle = np.mean(angle_in[usable])
    # print("mean angle", angle)
    return angle

def contour2polygon(
        contour: np.ndarray | Sequence[Sequence[Sequence[Number]]],
        dilate: int = 0,
        holes: bool = False,
):
    polygon = Polygon([point[0] for point in contour])
    if dilate:
        polygon = polygon.buffer(dilate)
        if holes and len(polygon.interiors):
            # from shapely.plotting import plot_polygon
            # from matplotlib import pyplot as plt
            # plt.figure("dilation interiors")
            # plt.subplot(2, 2, 1, title="original")
            # plot_polygon(polygon)
            new_interior = join_polygons(Polygon(poly) for poly in polygon.interiors)
            # plt.subplot(2, 2, 2, title="new_interior")
            # plot_polygon(polygon)
            # plot_polygon(new_interior, color='r')
            bridge = bridge_polygons(polygon.exterior, new_interior)
            # plt.subplot(2, 2, 3, title="bridge")
            # plot_polygon(polygon)
            # plot_polygon(bridge, color='r')
            polygon = polygon.difference(bridge).difference(new_interior)
            # plt.subplot(2, 2, 4, title="new")
            # plot_polygon(polygon)
            # plt.show()
        polygon = ensure_polygon(polygon)
    return ensure_polygon(make_valid(polygon))

def polygon2contour(polygon: Polygon) -> np.ndarray:
    polygon = np.array(polygon.exterior.coords[:-1], dtype=int)
    return np.maximum(0, polygon).astype(int)[:, np.newaxis]

def make_intersection(poly1, poly2):
    interp = poly1.intersection(poly2)
    # post-process
    if interp.is_empty or interp.area == 0.0:
        return None
    interp = ensure_polygon(interp)
    interp = make_valid(interp)
    interp = ensure_polygon(interp)
    return interp

def ensure_polygon(geometry):
    if geometry.geom_type == 'GeometryCollection':
        # heterogeneous result: filter zero-area shapes (LineString, Point)
        geometry = unary_union([geom for geom in geometry.geoms if geom.area > 0])
    if geometry.geom_type == 'MultiPolygon':
        # homogeneous result: construct convex hull to connect
        geometry = join_polygons(geometry.geoms)
    poly = Polygon(geometry)
    assert poly.geom_type == 'Polygon', poly.wkt
    return poly

def make_valid(polygon: Polygon) -> Polygon:
    """Ensures shapely.geometry.Polygon object is valid by repeated rearrangement/simplification/enlargement."""
    def isint(x):
        return isinstance(x, int) or int(x) == x
    # make sure rounding does not invalidate
    if (not all(map(isint, np.array(polygon.exterior.coords).flat)) and
        polygon.minimum_clearance < 1.0):
        polygon = Polygon(np.round(polygon.exterior.coords))
    if polygon.is_valid:
        return polygon
    points = list(polygon.exterior.coords[:-1])
    def step(split, tolerance):
        # try by re-arranging points
        poly = Polygon(points[-split:]+points[:-split])
        if poly.is_valid:
            return poly
        # try by simplification
        poly = poly.simplify(tolerance + 1.0)
        if poly.is_valid:
            return poly
        # try by enlarging
        poly = poly.buffer(tolerance)
        if poly.is_valid:
            return poly
        return None
    for split in range(1, len(points)):
        for tolerance in np.linspace(1, np.sqrt(polygon.area), 100):
            # simplification may not be possible (at all) due to ordering
            # in that case, try another starting point
            if poly := step(split, tolerance):
                return poly
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
                         for poly in polygons])
                if not poly.is_empty]
    npoly = len(polygons)
    if npoly == 1:
        return polygons[0]
    # find min-dist path through all polygons (travelling salesman)
    pairs = itertools.combinations(range(npoly), 2)
    dists = np.zeros((npoly, npoly), dtype=float)
    for i, j in pairs:
        dist = polygons[i].distance(polygons[j])
        dist = max(dist, 1e-5) # if pair merely touches, we still need to get an edge
        dists[i, j] = dist
        dists[j, i] = dist
    dists = minimum_spanning_tree(dists, overwrite=True)
    # add bridge polygons (where necessary)
    for prevp, nextp in zip(*dists.nonzero()):
        prevp = polygons[prevp]
        nextp = polygons[nextp]
        polygons.append(bridge_polygons(prevp, nextp, max(1, scale/5)))
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

def bridge_polygons(poly1, poly2, strength=1):
    nearest = nearest_points(poly1, poly2)
    bridgep = orient(LineString(nearest).buffer(strength, resolution=1), -1)
    return bridgep

