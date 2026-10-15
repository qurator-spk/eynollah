import pytest
import cv2
import numpy as np
from matplotlib import pyplot as plt

from eynollah.utils.tiling import do_prediction, do_prediction_new_concept

@pytest.mark.parametrize(
    "height,width",
    [
        (448, 448),
        (672, 672),
        (1088, 832),
        (1152, 896),
    ])
def test_tiling_idem(image_resources, height, width):
    infile = image_resources[0]
    class PseudoModel:
        def predict(self, images, **kwargs):
            return images
        @property
        def input_shape(self):
            return None, height, width, None
    model = PseudoModel()
    in_img = cv2.imread(infile)
    outimg = do_prediction(in_img, model, patches=True, is_enhancement=True,
                           marginal_of_patch_percent=0)
    assert in_img.shape == outimg.shape
    assert np.all(in_img == outimg)
    outimg = do_prediction(in_img, model, patches=True, is_enhancement=True,
                           marginal_of_patch_percent=0.1)
    assert in_img.shape == outimg.shape
    assert np.all(in_img == outimg)
    outimg = do_prediction(in_img, model, patches=True, is_enhancement=True,
                           marginal_of_patch_percent=0.2)
    assert in_img.shape == outimg.shape
    assert in_img.dtype == outimg.dtype
    assert np.all(in_img == outimg)

@pytest.mark.parametrize(
    "height,width",
    [
        (448, 448),
        (672, 672),
        (1088, 832),
        (1152, 896),
    ])
def test_tiling_min(image_resources, height, width):
    infile = image_resources[0]
    class PseudoModel:
        def predict(self, images, **kwargs):
            M = images.min(axis=(1, 2, 3))
            return 1. * (images == M)
        @property
        def input_shape(self):
            return None, height, width, None
    model = PseudoModel()
    in_img = cv2.imread(infile)
    outimg = do_prediction(in_img, model, patches=True,
                           marginal_of_patch_percent=0)
    assert in_img.shape[:2] == outimg.shape
    assert np.any(outimg)
    outimg = do_prediction(in_img, model, patches=True,
                           marginal_of_patch_percent=0.1)
    assert in_img.shape[:2] == outimg.shape
    assert np.any(outimg)
    outimg = do_prediction(in_img, model, patches=True,
                           marginal_of_patch_percent=0.2)
    assert in_img.shape[:2] == outimg.shape
    assert np.any(outimg)

@pytest.mark.parametrize(
    "height,width",
    [
        (448, 448),
        (672, 672),
        (1088, 832),
        (1152, 896),
    ])
def test_tiling_min_conf(image_resources, height, width):
    infile = image_resources[0]
    class PseudoModel:
        def predict(self, images, **kwargs):
            M = images.min(axis=(1, 2, 3))
            return 1. * (images == M)
        @property
        def input_shape(self):
            return None, height, width, None
    model = PseudoModel()
    in_img = cv2.imread(infile)
    outimg, conf = do_prediction_new_concept(
        in_img, model, patches=True,
        marginal_of_patch_percent=0)
    assert in_img.shape[:2] == outimg.shape
    assert np.any(outimg)
    assert np.sum(conf) < conf.size
    outimg, conf = do_prediction_new_concept(
        in_img, model, patches=True,
        marginal_of_patch_percent=0.1)
    assert in_img.shape[:2] == outimg.shape
    assert np.any(outimg)
    assert np.sum(conf) < conf.size
    outimg, conf = do_prediction_new_concept(
        in_img, model, patches=True,
        marginal_of_patch_percent=0.2)
    assert in_img.shape[:2] == outimg.shape
    assert np.any(outimg)
    assert np.sum(conf) < conf.size
