"""
extract page border (i.e. crop)
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import logging
from multiprocessing import cpu_count
import os
import time
from typing import Optional
from pathlib import Path
import numpy as np
import cv2

from .model_zoo.model_zoo import EynollahModelZoo
from .writer import EynollahXmlWriter
from .eynollah import Eynollah
from .plot import EynollahPlotter
from .utils import box2rect, is_image_filename, Region
from .utils.resize import resize_image

class EynollahPageExtractor(Eynollah):

    def __init__(
        self,
        *,
        model_zoo: EynollahModelZoo,
        enable_plotting : bool = False,
        input_binary : bool = False,
        ignore_page_extraction : bool = False,
        skip_layout_and_reading_order : bool = True,
        num_col_upper : int | None = None,
        num_col_lower : int | None = None,
        full_layout : bool = False,
        tables : bool = False,
        curved_line : bool = False,
        allow_enhancement : bool = False,
        enable_deskewing : bool = False,
    ):
        self.logger = logging.getLogger('eynollah.extract_page')
        self.model_zoo = model_zoo
        self.plotter = None
        self.tables = tables
        self.curved_line = curved_line
        self.allow_enhancement = allow_enhancement
        
        self.enable_plotting = enable_plotting
        # --input-binary sensible if image is very dark, if layout is not working.
        self.input_binary = input_binary
        self.full_layout = full_layout
        self.ignore_page_extraction = ignore_page_extraction
        self.skip_layout_and_reading_order = skip_layout_and_reading_order
        if num_col_upper:
            self.num_col_upper = int(num_col_upper)
        else:
            self.num_col_upper = num_col_upper
        if num_col_lower:
            self.num_col_lower = int(num_col_lower)
        else:
            self.num_col_lower = num_col_lower
        self.enable_deskewing = enable_deskewing

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
        ]
        if self.input_binary:
            loadable.append("binarization")
        if self.enable_deskewing:
            loadable.append("textline")
        self.model_zoo.load_models(*loadable, device=device)

    def run(self,
            overwrite: bool = False,
            image_filename: str | None = None,
            dir_in: str | None = None,
            dir_out: str | None = None,
            **kwargs
    ):
        """
        Get scanned image and scales, then detect the page border
        """
        self.logger.debug("enter run")
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
        page_cont, _, _ = self.extract_page(image)
        page = Region(page_cont)

        if self.enable_deskewing:
            t2 = time.time()
            _, _, _, _, _, textline_mask, _, _ = self.get_early_layout(
                image['img_res'], num_col_classifier)
            page.skew = self.run_deskew(
                textline_mask, num_col_classifier)
            t3 = time.time()
            self.logger.info("Deskewing took %.1fs", t3 - t2)
            
        pcgts = writer.build_pagexml(
            page=page,
            img_bin=self.imread(image, binary=True) if self.input_binary else None,
            num_col=num_col_classifier,
        )
        writer.write_pagexml(pcgts)
        self.logger.info("Job done in %.1fs", time.time() - t0)
