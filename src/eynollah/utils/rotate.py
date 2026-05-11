import math
import cv2


def rotation_image_new(img, thetha):
    rotated = rotate_image(img, thetha)
    return rotate_max_area_new(img, rotated, thetha)

def rotate_image(img_patch, slope):
    (h, w) = img_patch.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, slope, 1.0)
    return cv2.warpAffine(img_patch, M, (w, h) )

def rotate_image_different( img, slope):
    # img = cv2.imread('images/input.jpg')
    num_rows, num_cols = img.shape[:2]

    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), slope, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    return img_rotation

def rotate_image_enlarge(img, angle):
    h, w = img.shape[:2]
    cx, cy = 0.5 * w, 0.5 * h
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    radian = angle / 180 * math.pi
    cos = abs(math.cos(radian))
    sin = abs(math.sin(radian))
    new_w, new_h = (w * cos + h * sin,
                    w * sin + h * cos)
    # box is larger after resize, so instead of shifting
    # back from center, shift from new center
    matrix[0, 2] += 0.5 * new_w - cx
    matrix[1, 2] += 0.5 * new_h - cy
    return cv2.warpAffine(img, matrix, (int(new_w + 0.5),
                                        int(new_h + 0.5)),
                          flags=cv2.INTER_CUBIC)
