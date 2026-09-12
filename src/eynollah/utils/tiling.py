from logging import getLogger
import math
import gc

import numpy as np

from . import seg_mask_label
from .resize import resize_image
from ..predictor import Predictor

def do_prediction(
        img: np.ndarray,
        model: Predictor,
        logger=None,
        patches=False,
        n_batch_inference=1,
        marginal_of_patch_percent=0.1,
        thresholding_for_some_classes=False,
        thresholding_for_heading=False,
        heading_class=2,
        thresholding_for_artificial_class=False,
        threshold_art_class=0.1,
        artificial_class=2,
        is_enhancement=False,
) -> np.ndarray:
    if logger is None:
        logger = getLogger('eynollah')
        
    logger.debug("enter do_prediction (patches=%d)", patches)
    _, img_height_model, img_width_model, _ = model.input_shape
    img_h_page = img.shape[0]
    img_w_page = img.shape[1]

    img = img / 255.
    img = img.astype(np.float16)

    if not patches:
        img = resize_image(img, img_height_model, img_width_model)

        label_p_pred = model.predict(img[np.newaxis], verbose=0)[0]
        if is_enhancement:
            seg = np.round(label_p_pred * 255).astype(np.uint8)
        else:
            seg = np.argmax(label_p_pred, axis=2).astype(np.uint8)

        if thresholding_for_artificial_class:
            seg_mask_label(
                seg, label_p_pred[:, :, artificial_class] >= threshold_art_class,
                label=artificial_class,
                skeletonize=True)

        if thresholding_for_heading:
            seg_mask_label(
                seg, label_p_pred[:, :, heading_class] >= 0.2,
                label=heading_class)

        return resize_image(seg, img_h_page, img_w_page)

    if img_h_page < img_height_model:
        img = resize_image(img, img_height_model, img.shape[1])
    if img_w_page < img_width_model:
        img = resize_image(img, img.shape[0], img_width_model)

    logger.debug("Patch size: %sx%s", img_height_model, img_width_model)
    margin = int(marginal_of_patch_percent * img_height_model)
    if 2 * margin > 0.5 * img_width_model:
        margin = img_width_model // 4
    if 2 * margin > 0.5 * img_height_model:
        margin = img_height_model // 4
    window = 1 / (1 + np.exp(5.0 - 5 * np.arange(2 * margin) / margin))
    width_mid = img_width_model - 2 * margin
    height_mid = img_height_model - 2 * margin
    img_h = img.shape[0]
    img_w = img.shape[1]
    prediction: np.ndarray = None # type: ignore
    nxf = math.ceil((img_w - 2.0 * margin) / width_mid)
    nyf = math.ceil((img_h - 2.0 * margin) / height_mid)

    batch_i = []
    batch_j = []
    batch_x_u = []
    batch_x_d = []
    batch_x_s = []
    batch_y_u = []
    batch_y_d = []
    batch_y_s = []

    batch = 0
    img_patch = np.zeros((n_batch_inference,
                          img_height_model,
                          img_width_model,
                          3), dtype=np.float16)
    for i in range(nxf):
        for j in range(nyf):
            index_x_d = i * width_mid
            index_x_u = index_x_d + img_width_model
            if index_x_u > img_w:
                index_x_s = index_x_u - img_w
                index_x_u = img_w
                index_x_d = img_w - img_width_model
            else:
                index_x_s = 0
            index_y_d = j * height_mid
            index_y_u = index_y_d + img_height_model
            if index_y_u > img_h:
                index_y_s = index_y_u - img_h
                index_y_u = img_h
                index_y_d = img_h - img_height_model
            else:
                index_y_s = 0

            batch_i.append(i)
            batch_j.append(j)
            batch_x_u.append(index_x_u)
            batch_x_d.append(index_x_d)
            batch_x_s.append(index_x_s)
            batch_y_d.append(index_y_d)
            batch_y_u.append(index_y_u)
            batch_y_s.append(index_y_s)

            img_patch[batch] = img[index_y_d: index_y_u,
                                   index_x_d: index_x_u]
            batch += 1
            if (batch == n_batch_inference or
                # last batch
                i == nxf - 1 and j == nyf - 1):
                logger.debug("predicting patches on %s", str(img_patch.shape))
                label_p_pred = model.predict(img_patch, verbose=0)
                if prediction is None:
                    # now we know the number of classes
                    prediction = np.zeros((img_h, img_w, label_p_pred.shape[-1]), dtype=float)

                for k in range(batch):
                    where = np.index_exp[batch_y_d[k]: batch_y_u[k],
                                         batch_x_d[k]: batch_x_u[k]]
                    # shorter window on last tile
                    part = np.index_exp[batch_y_s[k]:,
                                        batch_x_s[k]:]
                    # normalize probability (where windows overlap)
                    attenuation_y = np.ones(img_height_model - batch_y_s[k])
                    attenuation_x = np.ones(img_width_model - batch_x_s[k])
                    if margin and batch_j[k] > 0:
                        attenuation_y[:2 * margin] = window
                    if margin and batch_j[k] < nyf - 1:
                        attenuation_y[-2 * margin:] = 1 - window
                    if margin and batch_i[k] > 0:
                        attenuation_x[:2 * margin] = window
                    if margin and batch_i[k] < nxf - 1:
                        attenuation_x[-2 * margin:] = 1 - window
                    label_p_pred[k][part] *= attenuation_y[:, np.newaxis, np.newaxis]
                    label_p_pred[k][part] *= attenuation_x[np.newaxis, :, np.newaxis]
                    prediction[where][part] += label_p_pred[k][part]

                batch_i = []
                batch_j = []
                batch_x_u = []
                batch_x_d = []
                batch_x_s = []
                batch_y_u = []
                batch_y_d = []
                batch_y_s = []
                batch = 0
                img_patch[:] = 0

    if is_enhancement:
        seg = np.round(prediction * 255).astype(np.uint8)
    else:
        seg = np.argmax(prediction, axis=2).astype(np.uint8)
    if thresholding_for_some_classes:
        seg_mask_label(
            seg, prediction[:, :, 4] > 0.03,
            label=4) # art
        seg_mask_label(
            seg, prediction[:, :, 0] > 0.25,
            label=0) # bg
        seg_mask_label(
            seg, prediction[:, :, 3] > 0.10 & seg == 0,
            label=3) # sep
    if thresholding_for_artificial_class:
        seg_art = prediction[:, :, artificial_class] >= threshold_art_class
        seg_mask_label(seg, seg_art,
                       label=artificial_class,
                       only=True,
                       skeletonize=True,
                       dilate=3)

    if img_h != img_h_page or img_w != img_w_page:
        seg = resize_image(seg, img_h_page, img_w_page)

    gc.collect()
    return seg

def do_prediction_new_concept(
        img: np.ndarray,
        model: Predictor,
        logger=None,
        patches=False,
        n_batch_inference=1,
        marginal_of_patch_percent=0.1,
        thresholding_for_heading=False,
        heading_class=2,
        thresholding_for_artificial_class=False,
        threshold_art_class=0.1,
        artificial_class=4,
        separator_class=0,
) -> np.ndarray:
    if logger is None:
        logger = getLogger('eynollah')
        
    logger.debug("enter do_prediction_new_concept (patches=%d)", patches)
    _, img_height_model, img_width_model, _ = model.input_shape

    img = img / 255.0
    img = img.astype(np.float16)

    if not patches:
        img_h_page = img.shape[0]
        img_w_page = img.shape[1]
        img = resize_image(img, img_height_model, img_width_model)

        label_p_pred = model.predict(img[np.newaxis], verbose=0)[0]
        seg = np.argmax(label_p_pred, axis=2).astype(np.uint8)

        prediction = resize_image(seg, img_h_page, img_w_page)

        if thresholding_for_artificial_class:
            mask = resize_image(label_p_pred[:, :, artificial_class],
                                img_h_page, img_w_page) >= threshold_art_class
            seg_mask_label(prediction, mask,
                           label=artificial_class,
                           only=True,
                           skeletonize=True,
                           dilate=3,
                           keep=separator_class)
        if thresholding_for_heading:
            mask = resize_image(label_p_pred[:, :, heading_class],
                                img_h_page, img_w_page) >= 0.2
            seg_mask_label(prediction, mask,
                           label=heading_class)

        conf = label_p_pred[tuple(np.indices(seg.shape)) + (seg,)]
        conf = resize_image(conf, img_h_page, img_w_page)
        return prediction, conf

    if img.shape[0] < img_height_model:
        img = resize_image(img, img_height_model, img.shape[1])
    if img.shape[1] < img_width_model:
        img = resize_image(img, img.shape[0], img_width_model)

    logger.debug("Patch size: %sx%s", img_height_model, img_width_model)
    margin = int(marginal_of_patch_percent * img_height_model)
    if 2 * margin > 0.5 * img_width_model:
        margin = img_width_model // 4
    if 2 * margin > 0.5 * img_height_model:
        margin = img_height_model // 4
    window = 1 / (1 + np.exp(5.0 - 5 * np.arange(2 * margin) / margin))
    width_mid = img_width_model - 2 * margin
    height_mid = img_height_model - 2 * margin
    img_h = img.shape[0]
    img_w = img.shape[1]
    prediction = None
    nxf = math.ceil((img_w - 2.0 * margin) / width_mid)
    nyf = math.ceil((img_h - 2.0 * margin) / height_mid)

    batch_i = []
    batch_j = []
    batch_x_u = []
    batch_x_d = []
    batch_x_s = []
    batch_y_u = []
    batch_y_d = []
    batch_y_s = []
    batch = 0
    img_patch = np.zeros((n_batch_inference,
                          img_height_model,
                          img_width_model,
                          3), dtype=np.float16)
    for i in range(nxf):
        for j in range(nyf):
            index_x_d = i * width_mid
            index_x_u = index_x_d + img_width_model
            if index_x_u > img_w:
                index_x_s = index_x_u - img_w
                index_x_u = img_w
                index_x_d = img_w - img_width_model
            else:
                index_x_s = 0
            index_y_d = j * height_mid
            index_y_u = index_y_d + img_height_model
            if index_y_u > img_h:
                index_y_s = index_y_u - img_h
                index_y_u = img_h
                index_y_d = img_h - img_height_model
            else:
                index_y_s = 0

            batch_i.append(i)
            batch_j.append(j)
            batch_x_u.append(index_x_u)
            batch_x_d.append(index_x_d)
            batch_x_s.append(index_x_s)
            batch_y_d.append(index_y_d)
            batch_y_u.append(index_y_u)
            batch_y_s.append(index_y_s)

            img_patch[batch] = img[index_y_d: index_y_u,
                                   index_x_d: index_x_u]
            batch += 1
            if (batch == n_batch_inference or
                # last batch
                i == nxf - 1 and j == nyf - 1):
                logger.debug("predicting patches on %s", str(img_patch.shape))
                label_p_pred = model.predict(img_patch, verbose=0)
                if prediction is None:
                    # now we know the number of classes
                    prediction = np.zeros((img_h, img_w, label_p_pred.shape[-1]), dtype=float)

                for k in range(batch):
                    where = np.index_exp[batch_y_d[k]: batch_y_u[k],
                                         batch_x_d[k]: batch_x_u[k]]
                    # shorter window on last tile
                    part = np.index_exp[batch_y_s[k]:,
                                        batch_x_s[k]:]
                    # normalize probability (where windows overlap)
                    attenuation_y = np.ones(img_height_model - batch_y_s[k])
                    attenuation_x = np.ones(img_width_model - batch_x_s[k])
                    if margin and batch_j[k] > 0:
                        attenuation_y[:2 * margin] = window
                    if margin and batch_j[k] < nyf - 1:
                        attenuation_y[-2 * margin:] = 1 - window
                    if margin and batch_i[k] > 0:
                        attenuation_x[:2 * margin] = window
                    if margin and batch_i[k] < nxf - 1:
                        attenuation_x[-2 * margin:] = 1 - window
                    label_p_pred[k][part] *= attenuation_y[:, np.newaxis, np.newaxis]
                    label_p_pred[k][part] *= attenuation_x[np.newaxis, :, np.newaxis]
                    prediction[where][part] += label_p_pred[k][part]

                batch_i = []
                batch_j = []
                batch_x_u = []
                batch_x_d = []
                batch_x_s = []
                batch_y_u = []
                batch_y_d = []
                batch_y_s = []
                batch = 0
                img_patch[:] = 0

    # decode
    seg = np.argmax(prediction, axis=2).astype(np.uint8)
    conf = prediction[tuple(np.indices(seg.shape)) + (seg,)]
    if thresholding_for_artificial_class:
        seg_art = prediction[:, :, artificial_class] >= threshold_art_class
        seg_mask_label(seg, seg_art,
                       label=artificial_class,
                       only=True,
                       skeletonize=True,
                       dilate=3,
                       keep=separator_class)
    gc.collect()
    return seg, conf

# variant of do_prediction_new_concept with no need
# for resizing or tiling into patches - done on model
# (Tensorflow/CUDA) side
# (after loading wrapped resized or patched model)
def do_prediction_new_concept_autosize(
        img: np.ndarray,
        model: Predictor,
        logger=None,
        n_batch_inference=None,
        thresholding_for_heading=False,
        thresholding_for_artificial_class=False,
        threshold_art_class=0.1,
        artificial_class=4,
) -> np.ndarray:
    if logger is None:
        logger = getLogger('eynollah')
        
    logger.debug("enter do_prediction_new_concept (%s)", model.name)
    img = img / 255.0
    img = img.astype(np.float16)

    prediction = model.predict(img[np.newaxis])[0]
    confidence = prediction[:, :, 1]
    segmentation = np.argmax(prediction, axis=2).astype(np.uint8)

    if thresholding_for_artificial_class:
        seg_mask_label(segmentation,
                       prediction[:, :, artificial_class] >= threshold_art_class,
                       label=artificial_class,
                       only=True,
                       skeletonize=True,
                       dilate=3)
    if thresholding_for_heading:
        seg_mask_label(segmentation,
                       prediction[:, :, 2] >= 0.2,
                       label=2)
    gc.collect()
    return segmentation, confidence

