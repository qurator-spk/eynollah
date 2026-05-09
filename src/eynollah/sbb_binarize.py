"""
Tool to load model and binarize a given image.
"""

# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportPossiblyUnboundVariable=false

import os
import logging
from pathlib import Path 
from typing import Optional

import numpy as np
import cv2

from .eynollah import Eynollah
from .model_zoo import EynollahModelZoo
from .utils.resize import resize_image
from .utils import is_image_filename

class SbbBinarizer(Eynollah):

    def __init__(
            self,
            *,
            model_zoo: EynollahModelZoo,
            logger: Optional[logging.Logger] = None,
            device: str = '',
    ):
        self.logger = logger if logger else logging.getLogger('eynollah.binarization')
        self.model_zoo = model_zoo
        self.setup_models(device=device)

    def setup_models(self, device=''):
        loadable = ['binarization']
        self.model_zoo.load_models(*loadable, device=device)
        for model in loadable:
            self.logger.debug("model %s has input shape %s", model,
                              self.model_zoo.get(model).input_shape)

    def run(self,
            image=None,
            image_filename=None,
            output=None,
            use_patches=False,
            dir_in=None,
            overwrite=False
    ):
        """
        Binarize the scanned images
        """
        if dir_in:
            ls_imgs = [(os.path.join(dir_in, image_filename),
                        os.path.join(output, Path(image_filename).stem + '.png'))
                       for image_filename in filter(is_image_filename,
                                                    os.listdir(dir_in))]
        elif image_filename:
            ls_imgs = [(image_filename, output)]
        else:
            raise ValueError("run requires either a single image filename or a directory")

        for img_filename, output_filename in ls_imgs:
            self.logger.info(img_filename)

            if os.path.exists(output_filename):
                if overwrite:
                    self.logger.warning("will overwrite existing output file '%s'", output_filename)
                else:
                    self.logger.warning("will skip input for existing output file '%s'", output_filename)
                    continue

            img_res = self.run_single(img_filename,
                                      use_patches=use_patches)

            cv2.imwrite(output_filename, img_res)
            self.logger.info("output filename: '%s'", output_filename)

    def run_single(self,
                   img_filename: str,
                   img_pil=None,
                   use_patches: bool = False,
    ):
        image = self.cache_images(image_filename=img_filename, image_pil=img_pil)
        img = self.imread(image)
        img_bin = self.do_prediction(use_patches, img, self.model_zoo.get("binarization"),
                                     n_batch_inference=5)
        img_bin = 255 * (img_bin == 0).astype(np.uint8)
        #img_bin = np.repeat(img_bin[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        return img_bin
