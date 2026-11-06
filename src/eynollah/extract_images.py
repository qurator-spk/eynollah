"""
extract image regions only
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import logging
from multiprocessing import cpu_count
import os
import time
from pathlib import Path
import numpy as np
import cv2

from eynollah.utils.contour import return_contours_of_class
from eynollah.utils.resize import resize_image

from .model_zoo.model_zoo import EynollahModelZoo
from .writer import EynollahXmlWriter
from .eynollah import Eynollah
from .utils import box2rect, is_image_filename
from .utils.tiling import do_prediction_new_concept
from .plot import EynollahPlotter
from .utils import Region

class EynollahImageExtractor(Eynollah):

    def __init__(
        self,
        *,
        model_zoo: EynollahModelZoo,
        enable_plotting : bool = False,
        input_binary : bool = False,
        ignore_page_extraction : bool = False,
        num_col_upper : int | None = None,
        num_col_lower : int | None = None,
        full_layout : bool = False,
        tables : bool = False,
        curved_line : bool = False,
        allow_enhancement : bool = False,
        
    ):
        self.logger = logging.getLogger('eynollah.extract_images')
        self.model_zoo = model_zoo
        self.plotter = None
        self.tables = tables
        self.curved_line = curved_line
        self.allow_enhancement = allow_enhancement
        
        self.enable_plotting = enable_plotting
        # --input-binary sensible if image is very dark, if layout is not working.
        self.input_binary = input_binary
        self.ignore_page_extraction = ignore_page_extraction
        self.full_layout = full_layout
        if num_col_upper:
            self.num_col_upper = int(num_col_upper)
        else:
            self.num_col_upper = num_col_upper
        if num_col_lower:
            self.num_col_lower = int(num_col_lower)
        else:
            self.num_col_lower = num_col_lower

        # for parallelization of CPU-intensive tasks:
        self.executor = ProcessPoolExecutor(max_workers=cpu_count())

        t_start = time.time()

        self.logger.info("Loading models...")
        self.setup_models()
        self.logger.info(f"Model initialization complete ({time.time() - t_start:.1f}s)")

    def setup_models(self, device=''):

        loadable = [
            "col_classifier",
            "page",
            "extract_images",
        ]
        if self.input_binary:
            loadable.append("binarization")
        self.model_zoo.load_models(*loadable, device=device)

    def get_early_layout(
            self,
            img,
            num_col_classifier,
            label_text=1,
            label_imgs=2,
            label_seps=3,
    ):
        self.logger.debug("enter get_regions_extract_images_only")
        erosion_hurts = False
        # already cropped
        img_height_h, img_width_h = img.shape[:2]

        if num_col_classifier == 1:
            img_w_new = 700
        elif num_col_classifier == 2:
            img_w_new = 900
        elif num_col_classifier == 3:
            img_w_new = 1500
        elif num_col_classifier == 4:
            img_w_new = 1800
        elif num_col_classifier == 5:
            img_w_new = 2200
        elif num_col_classifier == 6:
            img_w_new = 2500
        else:
            raise ValueError("num_col_classifier must be in range 1..6")
        img_h_new = img_w_new * img_height_h // img_width_h
        img_resized = resize_image(img, img_h_new, img_w_new)

        prediction_regions, _ = do_prediction_new_concept(
            img_resized, self.model_zoo.get("extract_images"),
            patches=True, logger=self.logger)
        prediction_regions = resize_image(prediction_regions, img_height_h, img_width_h)

        mask_texts_only = (prediction_regions == label_text).astype(np.uint8)
        mask_images_only = (prediction_regions == label_imgs).astype(np.uint8)
        mask_seps_only = (prediction_regions == label_seps).astype(np.uint8)

        texts_only_cont = return_contours_of_class(mask_texts_only, 1, 1e-5)
        seps_only_cont = return_contours_of_class(mask_seps_only, 1, 1e-5)

        text_regions_p = np.zeros_like(prediction_regions)
        text_regions_p = cv2.fillPoly(text_regions_p, pts=seps_only_cont, color=label_seps)
        text_regions_p[mask_images_only == 1] = label_imgs
        text_regions_p = cv2.fillPoly(text_regions_p, pts=texts_only_cont, color=label_text)

        # rs: why?
        text_regions_p[-15:] = 0
        text_regions_p[:, -15:] = 0

        images_cont = return_contours_of_class(text_regions_p, label_imgs, 1e-3)

        images_cont_fin = []
        for cont in images_cont:
            _, _, w, h = box = cv2.boundingRect(cont)
            if h < 150 or w < 150:
                pass
            else:
                y1, y2, x1, x2 = box2rect(box) # type: ignore
                images_cont_fin.append(np.array([[[x1, y1]],
                                                 [[x2, y1]],
                                                 [[x2, y2]],
                                                 [[x1, y2]]]))

        self.logger.debug("exit get_regions_extract_images_only")
        return (text_regions_p,
                erosion_hurts,
                images_cont_fin)

    def run(self,
            overwrite: bool = False,
            image_filename: str | None = None,
            dir_in: str | None = None,
            dir_out: str | None = None,
            dir_of_cropped_images: str | None = None,
            **kwargs
    ):
        """
        Get scanned image and scales, then crop, and detect image regions
        """
        self.logger.debug("enter run")
        # Log enabled features directly
        enabled_modes = []
        if self.enable_plotting:
            self.logger.info("Saving debug plots")
            if dir_of_cropped_images:
                self.logger.info(f"Saving cropped images to: {dir_of_cropped_images}")
            self.plotter = EynollahPlotter(
                dir_out=dir_out,
                dir_of_cropped_images=dir_of_cropped_images,
            )
        if dir_in:
            t0_tot = time.time()
            ls_imgs = [os.path.join(dir_in, image_filename)
                       for image_filename in filter(is_image_filename,
                                                    os.listdir(dir_in))]
        elif image_filename:
            ls_imgs = [image_filename]
        else:
            raise ValueError("run requires either a single image filename or a directory")

        for img_filename in ls_imgs:
            self.run_single(img_filename, dir_out=dir_out, overwrite=overwrite)

        if dir_in:
            self.logger.info("All jobs done in %.1fs", time.time() - t0_tot)

    def run_single(self,
                   img_filename: str,
                   dir_out: str | None = None,
                   overwrite: bool = False
    ) -> None:
        t0 = time.time()
        self.logger.info(img_filename)

        image = self.cache_images(image_filename=img_filename)
        writer = EynollahXmlWriter(
            dir_out=dir_out,
            image_filename=img_filename,
            image_width=image['img'].shape[1],
            image_height=image['img'].shape[0],
        )

        if os.path.exists(writer.output_filename):
            if overwrite:
                self.logger.warning("will overwrite existing output file '%s'", writer.output_filename)
            else:
                self.logger.warning("will skip input for existing output file '%s'", writer.output_filename)
                return

        self.logger.info(f"Processing file: {writer.image_filename}")
        self.logger.info("Step 1/5: Image Enhancement")

        num_col_classifier, _ = self.run_enhancement(image)
        writer.scale_x = image['scale_x']
        writer.scale_y = image['scale_y']
        
        self.logger.info(f"Image: {image['img_res'].shape[1]}x{image['img_res'].shape[0]}, "
                         f"scale {image['scale_x']:.1f}x{image['scale_y']:.1f}, "
                         f"{image['dpi']} DPI, {num_col_classifier} columns")
        self.logger.info(f"Enhancement complete ({time.time() - t0:.1f}s)")

        # Image Extraction Mode
        self.logger.info("Step 2/5: Image Extraction Mode")
        t1 = time.time()
        page_cont, image_page, _ = self.extract_page(image)
        page = Region(page_cont)
        
        _, _, images_cont = self.get_early_layout(
            image['img_res'], num_col_classifier)
        self.logger.debug("Found %d images", len(images_cont))

        # FIXME: post-hoc cropping (remove when models support it, and replace image['img_res'] with image_page)
        page_box = cv2.boundingRect(page.contour)
        images_cont = [cont - [page_box[:2]]
                       for cont in images_cont]
        if self.plotter:
            self.plotter.write_images_into_directory(images_cont, image_page,
                                                     name=image['name'])
        self.logger.info("Image extraction complete")

        images = [Region(cont) for cont in images_cont]
        # can be empty if above page frame
        images = [image for image in images if image.area]
        pcgts = writer.build_pagexml(
            page=page,
            img_bin=self.imread(image, binary=True) if self.input_binary else None,
            num_col=num_col_classifier,
            images=images,
        )
        writer.write_pagexml(pcgts)
        self.logger.info("Job done in %.1fs", time.time() - t0)
