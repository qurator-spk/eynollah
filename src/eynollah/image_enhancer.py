"""
Image enhancer. The output can be written as same scale of input or in new predicted scale.
"""

import logging
import os
from typing import Optional
from pathlib import Path

import cv2

from .eynollah import Eynollah
from .model_zoo import EynollahModelZoo
from .utils.resize import resize_image
from .utils import is_image_filename


class Enhancer(Eynollah):
    def __init__(
            self,
            *,
            model_zoo: EynollahModelZoo,
            num_col_upper: int = 0,
            num_col_lower: int = 0,
            save_org_scale: bool = False,
            device: str = '',
    ):
        self.save_org_scale = save_org_scale
        self.num_col_upper = int(num_col_upper)
        self.num_col_lower = int(num_col_lower)
        self.input_binary = False
        self.ignore_page_extraction = False
            
        self.logger = logging.getLogger('eynollah.enhance')
        self.model_zoo = model_zoo
        self.setup_models()

    def setup_models(self, device=''):
        loadable = ['enhancement', 'col_classifier', 'page']
        self.model_zoo.load_models(*loadable, device=device)
        for model in loadable:
            self.logger.debug("model %s has input shape %s", model,
                              self.model_zoo.get(model).input_shape)

    def run_single(self,
                   img_filename: str,
                   img_pil=None,
                   dir_out: Optional[str] = None,
                   overwrite: bool = False,
    ) -> None:

        image = self.cache_images(image_filename=img_filename, image_pil=img_pil)
        output_filename = os.path.join(dir_out or "", image['name'] +'.png')
        
        if os.path.exists(output_filename):
            if overwrite:
                self.logger.warning("will overwrite existing output file '%s'", output_filename)
            else:
                self.logger.warning("will skip input for existing output file '%s'", output_filename)
                return

        self.resize_image_with_column_classifier(image)
        img_org = image['img']
        img_res = image['img_res']
        if self.save_org_scale:
            img_res = resize_image(img_res, img_org.shape[0], img_org.shape[1])

        cv2.imwrite(output_filename, img_res)
        self.logger.info("output filename: '%s'", output_filename)
        
    def run(self,
            overwrite: bool = False,
            image_filename: Optional[str] = None,
            dir_in: Optional[str] = None,
            dir_out: Optional[str] = None,
    ):
        """
        Enlarge and enhance the scanned images
        """
        if dir_in:
            ls_imgs = [os.path.join(dir_in, image_filename)
                       for image_filename in filter(is_image_filename,
                                                    os.listdir(dir_in))]
        elif image_filename:
            ls_imgs = [image_filename]
        else:
            raise ValueError("run requires either a single image filename or a directory")

        for img_filename in ls_imgs:
            self.logger.info(img_filename)

            self.run_single(img_filename,
                            dir_out=dir_out,
                            overwrite=overwrite)
