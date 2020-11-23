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

def cleaning_probs(self, probs: np.ndarray, sigma: float) -> np.ndarray:
    # Smooth
    if sigma > 0.0:
        return cv2.GaussianBlur(probs, (int(3 * sigma) * 2 + 1, int(3 * sigma) * 2 + 1), sigma)
    elif sigma == 0.0:
        return cv2.fastNlMeansDenoising((probs * 255).astype(np.uint8), h=20) / 255
    else:  # Negative sigma, do not do anything
        return probs

