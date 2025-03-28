import math
import numpy as np
import cv2

def rotatedRectWithMaxArea(w, h, angle):
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # it suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side, 
        # the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def rotate_image_opencv(image, angle):
    # Calculate the original image dimensions (h, w) and the center point (cx, cy)
    h, w = image.shape[:2]
    cx, cy = (w // 2, h // 2)

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Calculate the new bounding box
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ])

    # Apply rotation matrix to the corner points
    ones = np.ones(shape=(len(corners), 1))
    corners_ones = np.hstack([corners, ones])
    transformed_corners = M @ corners_ones.T
    transformed_corners = transformed_corners.T

    # Calculate the new bounding box dimensions
    min_x, min_y = np.min(transformed_corners, axis=0)
    max_x, max_y = np.max(transformed_corners, axis=0)

    newW = int(np.ceil(max_x - min_x))
    newH = int(np.ceil(max_y - min_y))

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (newW / 2) - cx
    M[1, 2] += (newH / 2) - cy

    # Perform the affine transformation (rotation)
    rotated_image = cv2.warpAffine(image, M, (newW, newH))

    return rotated_image

def rotate_max_area_new(image, rotated, angle):
    wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0], math.radians(angle))
    h, w, _ = rotated.shape
    y1 = h // 2 - int(hr / 2)
    y2 = y1 + int(hr)
    x1 = w // 2 - int(wr / 2)
    x2 = x1 + int(wr)
    return rotated[y1:y2, x1:x2]

def rotation_image_new(img, thetha):
    rotated = rotate_image_opencv(img, thetha)
    return rotate_max_area_new(img, rotated, thetha)

def rotate_image(img_patch, slope):
    (h, w) = img_patch.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, slope, 1.0)
    return cv2.warpAffine(img_patch, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def rotate_image_different(img, slope):
    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), slope, 1)
    return cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))

def rotate_max_area(image, rotated, rotated_textline, rotated_layout, rotated_table_prediction, angle):
    wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0], math.radians(angle))
    h, w, _ = rotated.shape
    y1 = h // 2 - int(hr / 2)
    y2 = y1 + int(hr)
    x1 = w // 2 - int(wr / 2)
    x2 = x1 + int(wr)
    return rotated[y1:y2, x1:x2], rotated_textline[y1:y2, x1:x2], rotated_layout[y1:y2, x1:x2], rotated_table_prediction[y1:y2, x1:x2]

def rotation_not_90_func(img, textline, text_regions_p_1, table_prediction, thetha):
    rotated = rotate_image_opencv(img, thetha)
    rotated_textline = rotate_image_opencv(textline, thetha)
    rotated_layout = rotate_image_opencv(text_regions_p_1, thetha)
    rotated_table_prediction = rotate_image_opencv(table_prediction, thetha)
    return rotate_max_area(img, rotated, rotated_textline, rotated_layout, rotated_table_prediction, thetha)

def rotation_not_90_func_full_layout(img, textline, text_regions_p_1, text_regions_p_fully, thetha):
    rotated = rotate_image_opencv(img, thetha)
    rotated_textline = rotate_image_opencv(textline, thetha)
    rotated_layout = rotate_image_opencv(text_regions_p_1, thetha)
    rotated_layout_full = rotate_image_opencv(text_regions_p_fully, thetha)
    return rotate_max_area_full_layout(img, rotated, rotated_textline, rotated_layout, rotated_layout_full, thetha)

def rotate_max_area_full_layout(image, rotated, rotated_textline, rotated_layout, rotated_layout_full, angle):
    wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0], math.radians(angle))
    h, w, _ = rotated.shape
    y1 = h // 2 - int(hr / 2)
    y2 = y1 + int(hr)
    x1 = w // 2 - int(wr / 2)
    x2 = x1 + int(wr)
    return rotated[y1:y2, x1:x2], rotated_textline[y1:y2, x1:x2], rotated_layout[y1:y2, x1:x2], rotated_layout_full[y1:y2, x1:x2]
