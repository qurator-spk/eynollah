"""
extract images?
"""

from concurrent.futures import ProcessPoolExecutor
import logging
from multiprocessing import cpu_count
import os
import time
from typing import Optional
from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2

from eynollah.utils.contour import filter_contours_area_of_image, return_contours_of_image, return_contours_of_interested_region
from eynollah.utils.resize import resize_image

from .model_zoo.model_zoo import EynollahModelZoo
from .eynollah import Eynollah
from .utils import box2rect, is_image_filename
from .plot import EynollahPlotter

class EynollahImageExtractor(Eynollah):

    def __init__(
        self,
        *,
        model_zoo: EynollahModelZoo,
        enable_plotting : bool = False,
        input_binary : bool = False,
        ignore_page_extraction : bool = False,
        num_col_upper : Optional[int] = None,
        num_col_lower : Optional[int] = None,
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

        try:
            for device in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(device, True)
        except:
            self.logger.warning("no GPU device available")
            
        self.logger.info("Loading models...")
        self.setup_models()
        self.logger.info(f"Model initialization complete ({time.time() - t_start:.1f}s)")

    def setup_models(self):

        loadable = [
            "col_classifier",
            "binarization",
            "page",
            "extract_images",
        ]
        self.model_zoo.load_models(*loadable)

    def get_regions_light_v_extract_only_images(self,img, num_col_classifier):
        self.logger.debug("enter get_regions_extract_images_only")
        erosion_hurts = False
        img_org = np.copy(img)
        img_height_h = img_org.shape[0]
        img_width_h = img_org.shape[1]

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
        img_h_new = int(img.shape[0] / float(img.shape[1]) * img_w_new)
        img_resized = resize_image(img,img_h_new, img_w_new )

        prediction_regions_org, _ = self.do_prediction_new_concept(True, img_resized, self.model_zoo.get("extract_images"))

        prediction_regions_org = resize_image(prediction_regions_org,img_height_h, img_width_h )
        image_page, page_coord, cont_page = self.extract_page()

        prediction_regions_org = prediction_regions_org[page_coord[0] : page_coord[1], page_coord[2] : page_coord[3]]
        prediction_regions_org=prediction_regions_org[:,:,0]

        mask_lines_only = (prediction_regions_org[:,:] ==3)*1
        mask_texts_only = (prediction_regions_org[:,:] ==1)*1
        mask_images_only=(prediction_regions_org[:,:] ==2)*1

        polygons_seplines, hir_seplines = return_contours_of_image(mask_lines_only)
        polygons_seplines = filter_contours_area_of_image(
            mask_lines_only, polygons_seplines, hir_seplines, max_area=1, min_area=0.00001, dilate=1)

        polygons_of_only_texts = return_contours_of_interested_region(mask_texts_only,1,0.00001)
        polygons_of_only_lines = return_contours_of_interested_region(mask_lines_only,1,0.00001)

        text_regions_p_true = np.zeros(prediction_regions_org.shape)
        text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts = polygons_of_only_lines, color=(3,3,3))

        text_regions_p_true[:,:][mask_images_only[:,:] == 1] = 2
        text_regions_p_true = cv2.fillPoly(text_regions_p_true, pts=polygons_of_only_texts, color=(1,1,1))

        text_regions_p_true[text_regions_p_true.shape[0]-15:text_regions_p_true.shape[0], :] = 0
        text_regions_p_true[:, text_regions_p_true.shape[1]-15:text_regions_p_true.shape[1]] = 0

        ##polygons_of_images = return_contours_of_interested_region(text_regions_p_true, 2, 0.0001)
        polygons_of_images = return_contours_of_interested_region(text_regions_p_true, 2, 0.001)

        polygons_of_images_fin = []
        for ploy_img_ind in polygons_of_images:
            box = _, _, w, h = cv2.boundingRect(ploy_img_ind)
            if h < 150 or w < 150:
                pass
            else:
                page_coord_img = box2rect(box) # type: ignore
                polygons_of_images_fin.append(np.array([[page_coord_img[2], page_coord_img[0]],
                                                        [page_coord_img[3], page_coord_img[0]],
                                                        [page_coord_img[3], page_coord_img[1]],
                                                        [page_coord_img[2], page_coord_img[1]]]))

        self.logger.debug("exit get_regions_extract_images_only")
        return (text_regions_p_true,
                erosion_hurts,
                polygons_seplines,
                polygons_of_images_fin,
                image_page,
                page_coord,
                cont_page)

    def run(self,
            overwrite: bool = False,
            image_filename: Optional[str] = None,
            dir_in: Optional[str] = None,
            dir_out: Optional[str] = None,
            dir_of_cropped_images: Optional[str] = None,
            dir_of_layout: Optional[str] = None,
            dir_of_deskewed: Optional[str] = None,
            dir_of_all: Optional[str] = None,
            dir_save_page: Optional[str] = None,
    ):
        """
        Get image and scales, then extract the page of scanned image
        """
        self.logger.debug("enter run")
        t0_tot = time.time()
        # Log enabled features directly
        enabled_modes = []
        if self.full_layout:
            enabled_modes.append("Full layout analysis")
        if self.tables:
            enabled_modes.append("Table detection")
        if enabled_modes:
            self.logger.info("Enabled modes: " + ", ".join(enabled_modes))
        if self.enable_plotting:
            self.logger.info("Saving debug plots")
            if dir_of_cropped_images:
                self.logger.info(f"Saving cropped images to: {dir_of_cropped_images}")
            if dir_of_layout:
                self.logger.info(f"Saving layout plots to: {dir_of_layout}")
            if dir_of_deskewed:
                self.logger.info(f"Saving deskewed images to: {dir_of_deskewed}")

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
            t0 = time.time()

            self.reset_file_name_dir(img_filename, dir_out)
            if self.enable_plotting:
                self.plotter = EynollahPlotter(dir_out=dir_out,
                                               dir_of_all=dir_of_all,
                                               dir_save_page=dir_save_page,
                                               dir_of_deskewed=dir_of_deskewed,
                                               dir_of_cropped_images=dir_of_cropped_images,
                                               dir_of_layout=dir_of_layout,
                                               image_filename_stem=Path(img_filename).stem)
            #print("text region early -11 in %.1fs", time.time() - t0)
            if os.path.exists(self.writer.output_filename):
                if overwrite:
                    self.logger.warning("will overwrite existing output file '%s'", self.writer.output_filename)
                else:
                    self.logger.warning("will skip input for existing output file '%s'", self.writer.output_filename)
                    continue

            pcgts = self.run_single()
            self.logger.info("Job done in %.1fs", time.time() - t0)
            self.writer.write_pagexml(pcgts)

        if dir_in:
            self.logger.info("All jobs done in %.1fs", time.time() - t0_tot)

    def run_single(self):
        t0 = time.time()
    
        self.logger.info(f"Processing file: {self.writer.image_filename}")
        self.logger.info("Step 1/5: Image Enhancement")
        
        img_res, is_image_enhanced, num_col_classifier, _ = \
            self.run_enhancement()
        
        self.logger.info(f"Image: {self.image.shape[1]}x{self.image.shape[0]}, "
                         f"{self.dpi} DPI, {num_col_classifier} columns")
        if is_image_enhanced:
            self.logger.info("Enhancement applied")
        
        self.logger.info(f"Enhancement complete ({time.time() - t0:.1f}s)")
        

        # Image Extraction Mode
        self.logger.info("Step 2/5: Image Extraction Mode")
        
        _, _, _, polygons_of_images, \
            image_page, page_coord, cont_page = \
            self.get_regions_light_v_extract_only_images(img_res, num_col_classifier)

        pcgts = self.writer.build_pagexml_no_full_layout(
            found_polygons_text_region=[],                   
            page_coord=page_coord,                  
            order_of_texts=[],              
            all_found_textline_polygons=[],   
            all_box_coord=[],                  
            found_polygons_text_region_img=polygons_of_images,                          
            found_polygons_marginals_left=[],                           
            found_polygons_marginals_right=[],                            
            all_found_textline_polygons_marginals_left=[],                                           
            all_found_textline_polygons_marginals_right=[],                                            
            all_box_coord_marginals_left=[],                             
            all_box_coord_marginals_right=[],                              
            slopes=[],                      
            slopes_marginals_left=[],                          
            slopes_marginals_right=[],                          
            cont_page=cont_page,                   
            polygons_seplines=[],                          
            found_polygons_tables=[],                          
        )
        if self.plotter:
            self.plotter.write_images_into_directory(polygons_of_images, image_page)
            
        self.logger.info("Image extraction complete")
        return pcgts
