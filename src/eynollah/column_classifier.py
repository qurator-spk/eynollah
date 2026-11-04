"""The two model stages used to classify a page's column count.

These functions are shared with the layout pipeline so the standalone command
uses the same image preparation and page crop.
"""

import cv2
import numpy as np

from .utils import crop_image_inside_box
from .utils.resize import resize_image


PAGE_KERNEL = np.ones((5, 5), np.uint8)


def prepare_page_input(image, model_shape):
    """Prepare one color image for the page extraction model."""
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = (image / 255.).astype(np.float16)
    return resize_image(image, model_shape[1], model_shape[2])


def crop_page_from_prediction(image, prediction):
    """Return the largest detected page and its (top, bottom, left, right) box."""
    mask = np.argmax(prediction, axis=2).astype(np.uint8)
    mask = resize_image(mask, image.shape[0], image.shape[1])
    mask = cv2.dilate(mask, PAGE_KERNEL, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = np.array(list(map(cv2.contourArea, contours)))
        box = cv2.boundingRect(contours[np.argmax(areas)])
    else:
        box = (0, 0, image.shape[1], image.shape[0])
    return crop_image_inside_box(box, image)


def crop_pages(images, page_model):
    """Predict and crop a bounded batch of pages."""
    batch = np.stack([prepare_page_input(image, page_model.input_shape) for image in images])
    predictions = page_model.predict(batch, verbose=0)
    return [crop_page_from_prediction(image, prediction)
            for image, prediction in zip(images, predictions)]


def prepare_classifier_input(page_image, page_coord, grayscale_image=None, binary=False):
    """Prepare a crop exactly as the current Eynollah column path does."""
    if binary:
        image = page_image
    elif grayscale_image is None:
        grayscale_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    else:
        top, bottom, left, right = page_coord
        grayscale_image = grayscale_image[top:bottom, left:right]
    if not binary:
        image = np.repeat(grayscale_image[:, :, np.newaxis], 3, axis=2)
    image = image / 255.0
    return cv2.resize(image, (448, 448), interpolation=cv2.INTER_NEAREST).astype(np.float16)


def classify_pages(cropped_pages, classifier_model):
    """Return one six-class score vector for each crop."""
    batch = np.stack([prepare_classifier_input(page, coord) for page, coord in cropped_pages])
    return classifier_model.predict(batch, verbose=0)
