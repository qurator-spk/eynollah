# pyright: reportPossiblyUnboundVariable=false

from logging import Logger, getLogger
from typing import Optional
from pathlib import Path
import os
import json
import gc
import sys
import math
import time

from keras.layers import StringLookup
import cv2
import xml.etree.ElementTree as ET
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from eynollah.model_zoo import EynollahModelZoo
try:
    import torch
except ImportError:
    torch = None


from .utils import is_image_filename
from .utils.resize import resize_image
from .utils.utils_ocr import (
    break_curved_line_into_small_pieces_and_then_merge,
    decode_batch_predictions,
    fit_text_single_line,
    get_contours_and_bounding_boxes,
    get_orientation_moments,
    preprocess_and_resize_image_for_ocrcnn_model,
    return_textlines_split_if_needed,
    rotate_image_with_padding,
)

# cannot use importlib.resources until we move to 3.9+ forimportlib.resources.files
if sys.version_info < (3, 10):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    TrOCRProcessor = VisionEncoderDecoderModel = None

class Eynollah_ocr:
    def __init__(
        self,
        *,
        model_zoo: EynollahModelZoo,
        dir_xmls=None,
        tr_ocr=False,
        batch_size: Optional[int]=None,
        export_textline_images_and_text: bool=False,
        do_not_mask_with_textline_contour: bool=False,
        pref_of_dataset=None,
        min_conf_value_of_textline_text : Optional[float]=None,
        logger: Optional[Logger]=None,
    ):
        self.tr_ocr = tr_ocr
        # For generating textline-image pairs for traning, move to generate_gt_for_training
        self.export_textline_images_and_text = export_textline_images_and_text
        # masking for OCR and GT generation, relevant for skewed lines and bounding boxes
        self.do_not_mask_with_textline_contour = do_not_mask_with_textline_contour
        # prefix or dataset
        self.pref_of_dataset = pref_of_dataset
        self.logger = logger if logger else getLogger('eynollah.ocr')
        self.model_zoo = model_zoo
        
        # TODO: Properly document what 'export_textline_images_and_text' is about
        if export_textline_images_and_text:
            self.logger.info("export_textline_images_and_text was set, so no actual models are loaded")
            return

        self.min_conf_value_of_textline_text = min_conf_value_of_textline_text if min_conf_value_of_textline_text else 0.3
        self.b_s = 2 if batch_size is None and tr_ocr else 8 if batch_size is None else batch_size

        if tr_ocr:
            self.model_zoo.load_model('trocr_processor')
            self.model_zoo.load_model('ocr', 'tr', model_path_override=model_name)
            self.model_zoo.get('ocr').to(self.device)
        else:
            self.model_zoo.load_model('ocr', '', model_path_override=model_name)
            self.model_zoo.load_model('num_to_char')
            self.end_character = len(self.model_zoo.load_model('characters')) + 2

    @property
    def device(self):
        if torch.cuda.is_available():
            self.logger.info("Using GPU acceleration")
            return torch.device("cuda:0")
        else:
            self.logger.info("Using CPU processing")
            return torch.device("cpu")

    def run(self, overwrite: bool = False,
            dir_in: Optional[str] = None,
            # Prediction with RGB and binarized images for selected pages, should not be the default
            dir_in_bin: Optional[str] = None,
            image_filename: Optional[str] = None,
            dir_xmls: Optional[str] = None,
            dir_out_image_text: Optional[str] = None,
            dir_out: Optional[str] = None,
    ):
        if dir_in:
            ls_imgs = [os.path.join(dir_in, image_filename)
                       for image_filename in filter(is_image_filename,
                                                    os.listdir(dir_in))]
        else:
            assert image_filename
            ls_imgs = [image_filename]

        if self.tr_ocr:
            tr_ocr_input_height_and_width = 384
            for dir_img in ls_imgs:
                file_name = Path(dir_img).stem
                assert dir_xmls  # FIXME: check the logic
                dir_xml = os.path.join(dir_xmls, file_name+'.xml')
                assert dir_out # FIXME: check the logic
                out_file_ocr = os.path.join(dir_out, file_name+'.xml')
                
                if os.path.exists(out_file_ocr):
                    if overwrite:
                        self.logger.warning("will overwrite existing output file '%s'", out_file_ocr)
                    else:
                        self.logger.warning("will skip input for existing output file '%s'", out_file_ocr)
                        continue
                    
                img = cv2.imread(dir_img)
                
                if dir_out_image_text:
                    out_image_with_text = os.path.join(dir_out_image_text, file_name+'.png')
                    image_text = Image.new("RGB", (img.shape[1], img.shape[0]), "white")
                    draw = ImageDraw.Draw(image_text)
                    total_bb_coordinates = []

                ##file_name = Path(dir_xmls).stem
                tree1 = ET.parse(dir_xml, parser = ET.XMLParser(encoding="utf-8"))
                root1=tree1.getroot()
                alltags=[elem.tag for elem in root1.iter()]
                link=alltags[0].split('}')[0]+'}'

                name_space = alltags[0].split('}')[0]
                name_space = name_space.split('{')[1]

                region_tags=np.unique([x for x in alltags if x.endswith('TextRegion')]) 
                        
                    
                    
                cropped_lines = []
                cropped_lines_region_indexer = []
                cropped_lines_meging_indexing = []
                
                extracted_texts = []

                indexer_text_region = 0
                indexer_b_s = 0
                
                for nn in root1.iter(region_tags):
                    for child_textregion in nn:
                        if child_textregion.tag.endswith("TextLine"):
                            
                            for child_textlines in child_textregion:
                                if child_textlines.tag.endswith("Coords"):
                                    cropped_lines_region_indexer.append(indexer_text_region)
                                    p_h=child_textlines.attrib['points'].split(' ')
                                    textline_coords =  np.array( [ [int(x.split(',')[0]),
                                                                    int(x.split(',')[1]) ]
                                                                   for x in p_h] )
                                    x,y,w,h = cv2.boundingRect(textline_coords)
                                    
                                    if dir_out_image_text:
                                        total_bb_coordinates.append([x,y,w,h])
                                    
                                    h2w_ratio = h/float(w)
                                    
                                    img_poly_on_img = np.copy(img)
                                    mask_poly = np.zeros(img.shape)
                                    mask_poly = cv2.fillPoly(mask_poly, pts=[textline_coords], color=(1, 1, 1))
                                    
                                    mask_poly = mask_poly[y:y+h, x:x+w, :]
                                    img_crop = img_poly_on_img[y:y+h, x:x+w, :]
                                    img_crop[mask_poly==0] = 255
                                    
                                    self.logger.debug("processing %d lines for '%s'",
                                                      len(cropped_lines), nn.attrib['id'])
                                    if h2w_ratio > 0.1:
                                        cropped_lines.append(resize_image(img_crop,
                                                                          tr_ocr_input_height_and_width,
                                                                          tr_ocr_input_height_and_width)  )
                                        cropped_lines_meging_indexing.append(0)
                                        indexer_b_s+=1
                                        if indexer_b_s==self.b_s:
                                            imgs = cropped_lines[:]
                                            cropped_lines = []
                                            indexer_b_s = 0
                                            
                                            pixel_values_merged = self.model_zoo.get('trocr_processor')(imgs, return_tensors="pt").pixel_values
                                            generated_ids_merged = self.model_zoo.get('ocr').generate(
                                                pixel_values_merged.to(self.device))
                                            generated_text_merged = self.model_zoo.get('trocr_processor').batch_decode(
                                                generated_ids_merged, skip_special_tokens=True)
                                            
                                            extracted_texts = extracted_texts + generated_text_merged
                                            
                                    else:
                                        splited_images, _ = return_textlines_split_if_needed(img_crop, None)
                                        #print(splited_images)
                                        if splited_images:
                                            cropped_lines.append(resize_image(splited_images[0],
                                                                              tr_ocr_input_height_and_width,
                                                                              tr_ocr_input_height_and_width))
                                            cropped_lines_meging_indexing.append(1)
                                            indexer_b_s+=1
                                            
                                            if indexer_b_s==self.b_s:
                                                imgs = cropped_lines[:]
                                                cropped_lines = []
                                                indexer_b_s = 0
                                                
                                                pixel_values_merged = self.model_zoo.get('trocr_processor')(imgs, return_tensors="pt").pixel_values
                                                generated_ids_merged = self.model_zoo.get('ocr').generate(
                                                    pixel_values_merged.to(self.device))
                                                generated_text_merged = self.model_zoo.get('trocr_processor').batch_decode(
                                                    generated_ids_merged, skip_special_tokens=True)
                                                
                                                extracted_texts = extracted_texts + generated_text_merged
                                            
                                            
                                            cropped_lines.append(resize_image(splited_images[1],
                                                                              tr_ocr_input_height_and_width,
                                                                              tr_ocr_input_height_and_width))
                                            cropped_lines_meging_indexing.append(-1)
                                            indexer_b_s+=1
                                            
                                            if indexer_b_s==self.b_s:
                                                imgs = cropped_lines[:]
                                                cropped_lines = []
                                                indexer_b_s = 0
                                                
                                                pixel_values_merged = self.model_zoo.get('trocr_processor')(imgs, return_tensors="pt").pixel_values
                                                generated_ids_merged = self.model_zoo.get('ocr').generate(
                                                    pixel_values_merged.to(self.device))
                                                generated_text_merged = self.model_zoo.get('trocr_processor').batch_decode(
                                                    generated_ids_merged, skip_special_tokens=True)
                                                
                                                extracted_texts = extracted_texts + generated_text_merged
                                                
                                        else:
                                            cropped_lines.append(img_crop)
                                            cropped_lines_meging_indexing.append(0)
                                            indexer_b_s+=1
                                            
                                            if indexer_b_s==self.b_s:
                                                imgs = cropped_lines[:]
                                                cropped_lines = []
                                                indexer_b_s = 0
                                                
                                                pixel_values_merged = self.model_zoo.get('trocr_processor')(imgs, return_tensors="pt").pixel_values
                                                generated_ids_merged = self.model_zoo.get('ocr').generate(
                                                    pixel_values_merged.to(self.device))
                                                generated_text_merged = self.model_zoo.get('trocr_processor').batch_decode(
                                                    generated_ids_merged, skip_special_tokens=True)
                                                
                                                extracted_texts = extracted_texts + generated_text_merged
                                                
                    
                                            
                    indexer_text_region = indexer_text_region +1
        
                if indexer_b_s!=0:
                    imgs = cropped_lines[:]
                    cropped_lines = []
                    indexer_b_s = 0
                    
                    pixel_values_merged = self.model_zoo.get('trocr_processor')(imgs, return_tensors="pt").pixel_values
                    generated_ids_merged = self.model_zoo.get('ocr').generate(pixel_values_merged.to(self.device))
                    generated_text_merged = self.model_zoo.get('trocr_processor').batch_decode(generated_ids_merged, skip_special_tokens=True)
                    
                    extracted_texts = extracted_texts + generated_text_merged
                    
                ####extracted_texts = []
                ####n_iterations  = math.ceil(len(cropped_lines) / self.b_s) 

                ####for i in range(n_iterations):
                    ####if i==(n_iterations-1):
                        ####n_start = i*self.b_s
                        ####imgs = cropped_lines[n_start:]
                    ####else:
                        ####n_start = i*self.b_s
                        ####n_end = (i+1)*self.b_s
                        ####imgs = cropped_lines[n_start:n_end]
                    ####pixel_values_merged = self.model_zoo.get('trocr_processor')(imgs, return_tensors="pt").pixel_values
                    ####generated_ids_merged = self.model_ocr.generate(
                    ####    pixel_values_merged.to(self.device))
                    ####generated_text_merged = self.model_zoo.get('trocr_processor').batch_decode(
                    ####    generated_ids_merged, skip_special_tokens=True)
                    
                    ####extracted_texts = extracted_texts + generated_text_merged
                    
                del cropped_lines
                gc.collect()

                extracted_texts_merged = [extracted_texts[ind]
                                          if cropped_lines_meging_indexing[ind]==0
                                          else extracted_texts[ind]+" "+extracted_texts[ind+1]
                                          if cropped_lines_meging_indexing[ind]==1
                                          else None
                                          for ind in range(len(cropped_lines_meging_indexing))]

                extracted_texts_merged = [ind for ind in extracted_texts_merged if ind is not None]
                #print(extracted_texts_merged, len(extracted_texts_merged))

                unique_cropped_lines_region_indexer = np.unique(cropped_lines_region_indexer)
                
                if dir_out_image_text:
                    
                    #font_path = "Charis-7.000/Charis-Regular.ttf"  # Make sure this file exists!
                    font = importlib_resources.files(__package__) / "Charis-Regular.ttf"
                    with importlib_resources.as_file(font) as font:
                        font = ImageFont.truetype(font=font, size=40)
                    
                    for indexer_text, bb_ind in enumerate(total_bb_coordinates):
                        
                        
                        x_bb = bb_ind[0]
                        y_bb = bb_ind[1]
                        w_bb = bb_ind[2]
                        h_bb = bb_ind[3]
                        
                        font = fit_text_single_line(draw, extracted_texts_merged[indexer_text],
                                                    font.path, w_bb, int(h_bb*0.4) )
                        
                        ##draw.rectangle([x_bb, y_bb, x_bb + w_bb, y_bb + h_bb], outline="red", width=2)
                        
                        text_bbox = draw.textbbox((0, 0), extracted_texts_merged[indexer_text], font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]

                        text_x = x_bb + (w_bb - text_width) // 2  # Center horizontally
                        text_y = y_bb + (h_bb - text_height) // 2  # Center vertically

                        # Draw the text
                        draw.text((text_x, text_y), extracted_texts_merged[indexer_text], fill="black", font=font)
                    image_text.save(out_image_with_text)

                #print(len(unique_cropped_lines_region_indexer), 'unique_cropped_lines_region_indexer')
                #######text_by_textregion = []
                #######for ind in unique_cropped_lines_region_indexer:
                    #######ind = np.array(cropped_lines_region_indexer)==ind
                    #######extracted_texts_merged_un = np.array(extracted_texts_merged)[ind]
                    #######text_by_textregion.append(" ".join(extracted_texts_merged_un))
                    
                text_by_textregion = []
                for ind in unique_cropped_lines_region_indexer:
                    ind = np.array(cropped_lines_region_indexer) == ind
                    extracted_texts_merged_un = np.array(extracted_texts_merged)[ind]
                    if len(extracted_texts_merged_un)>1:
                        text_by_textregion_ind = ""
                        next_glue = ""
                        for indt in range(len(extracted_texts_merged_un)):
                            if (extracted_texts_merged_un[indt].endswith('⸗') or
                                extracted_texts_merged_un[indt].endswith('-') or
                                extracted_texts_merged_un[indt].endswith('¬')):
                                text_by_textregion_ind += next_glue + extracted_texts_merged_un[indt][:-1]
                                next_glue = ""
                            else:
                                text_by_textregion_ind += next_glue + extracted_texts_merged_un[indt]
                                next_glue = " "
                        text_by_textregion.append(text_by_textregion_ind)
                    else:
                        text_by_textregion.append(" ".join(extracted_texts_merged_un))
                        
                        
                indexer = 0
                indexer_textregion = 0
                for nn in root1.iter(region_tags):
                    #id_textregion = nn.attrib['id']
                    #id_textregions.append(id_textregion)
                    #textregions_by_existing_ids.append(text_by_textregion[indexer_textregion])
                    
                    is_textregion_text = False
                    for childtest in nn:
                        if childtest.tag.endswith("TextEquiv"):
                            is_textregion_text = True
                    
                    if not is_textregion_text:
                        text_subelement_textregion = ET.SubElement(nn, 'TextEquiv')
                        unicode_textregion = ET.SubElement(text_subelement_textregion, 'Unicode')

                    
                    has_textline = False
                    for child_textregion in nn:
                        if child_textregion.tag.endswith("TextLine"):
                            
                            is_textline_text = False
                            for childtest2 in child_textregion:
                                if childtest2.tag.endswith("TextEquiv"):
                                    is_textline_text = True
                            
                            
                            if not is_textline_text:
                                text_subelement = ET.SubElement(child_textregion, 'TextEquiv')
                                ##text_subelement.set('conf', f"{extracted_conf_value_merged[indexer]:.2f}")
                                unicode_textline = ET.SubElement(text_subelement, 'Unicode')
                                unicode_textline.text = extracted_texts_merged[indexer]
                            else:
                                for childtest3 in child_textregion:
                                    if childtest3.tag.endswith("TextEquiv"):
                                        for child_uc in childtest3:
                                            if child_uc.tag.endswith("Unicode"):
                                                ##childtest3.set('conf', f"{extracted_conf_value_merged[indexer]:.2f}")
                                                child_uc.text = extracted_texts_merged[indexer]
                                    
                            indexer = indexer + 1
                            has_textline = True
                    if has_textline:
                        if is_textregion_text:
                            for child4 in nn:
                                if child4.tag.endswith("TextEquiv"):
                                    for childtr_uc in child4:
                                        if childtr_uc.tag.endswith("Unicode"):
                                            childtr_uc.text = text_by_textregion[indexer_textregion]
                        else:
                            unicode_textregion.text = text_by_textregion[indexer_textregion]
                        indexer_textregion = indexer_textregion + 1
                        
                ###sample_order  = [(id_to_order[tid], text)
                ###                 for tid, text in zip(id_textregions, textregions_by_existing_ids)
                ###                 if tid in id_to_order]
                
                ##ordered_texts_sample = [text for _, text in sorted(sample_order)]
                ##tot_page_text = ' '.join(ordered_texts_sample)
                
                ##for page_element in root1.iter(link+'Page'):
                    ##text_page = ET.SubElement(page_element, 'TextEquiv')
                    ##unicode_textpage = ET.SubElement(text_page, 'Unicode')
                    ##unicode_textpage.text = tot_page_text
                
                ET.register_namespace("",name_space)
                tree1.write(out_file_ocr,xml_declaration=True,method='xml',encoding="utf-8",default_namespace=None)
        else:
            ###max_len = 280#512#280#512
            ###padding_token = 1500#299#1500#299
            image_width = 512#max_len * 4
            image_height = 32


            img_size=(image_width, image_height)
            
            for dir_img in ls_imgs:
                file_name = Path(dir_img).stem
                dir_xml = os.path.join(dir_xmls, file_name+'.xml')
                out_file_ocr = os.path.join(dir_out, file_name+'.xml')
                
                if os.path.exists(out_file_ocr):
                    if overwrite:
                        self.logger.warning("will overwrite existing output file '%s'", out_file_ocr)
                    else:
                        self.logger.warning("will skip input for existing output file '%s'", out_file_ocr)
                        continue
                
                img = cv2.imread(dir_img)
                if dir_in_bin is not None:
                    cropped_lines_bin = []
                    dir_img_bin = os.path.join(dir_in_bin, file_name+'.png')
                    img_bin = cv2.imread(dir_img_bin)
                
                if dir_out_image_text:
                    out_image_with_text = os.path.join(dir_out_image_text, file_name+'.png')
                    image_text = Image.new("RGB", (img.shape[1], img.shape[0]), "white")
                    draw = ImageDraw.Draw(image_text)
                    total_bb_coordinates = []

                tree1 = ET.parse(dir_xml, parser = ET.XMLParser(encoding="utf-8"))
                root1=tree1.getroot()
                alltags=[elem.tag for elem in root1.iter()]
                link=alltags[0].split('}')[0]+'}'

                name_space = alltags[0].split('}')[0]
                name_space = name_space.split('{')[1]

                region_tags=np.unique([x for x in alltags if x.endswith('TextRegion')]) 
                    
                cropped_lines = []
                cropped_lines_ver_index = []
                cropped_lines_region_indexer = []
                cropped_lines_meging_indexing = []
                
                tinl = time.time()
                indexer_text_region = 0
                indexer_textlines = 0
                for nn in root1.iter(region_tags):
                    try:
                        type_textregion = nn.attrib['type']
                    except:
                        type_textregion = 'paragraph'
                    for child_textregion in nn:
                        if child_textregion.tag.endswith("TextLine"):
                            for child_textlines in child_textregion:
                                if child_textlines.tag.endswith("Coords"):
                                    cropped_lines_region_indexer.append(indexer_text_region)
                                    p_h=child_textlines.attrib['points'].split(' ')
                                    textline_coords =  np.array( [ [int(x.split(',')[0]),
                                                                    int(x.split(',')[1]) ]
                                                                   for x in p_h] )
                                    
                                    x,y,w,h = cv2.boundingRect(textline_coords)
                                    
                                    angle_radians = math.atan2(h, w)
                                    # Convert to degrees
                                    angle_degrees = math.degrees(angle_radians)
                                    if type_textregion=='drop-capital':
                                        angle_degrees = 0
                                        
                                    if dir_out_image_text:
                                        total_bb_coordinates.append([x,y,w,h])
                                       
                                    w_scaled = w *  image_height/float(h)
                                    
                                    img_poly_on_img = np.copy(img)
                                    if dir_in_bin is not None:
                                        img_poly_on_img_bin = np.copy(img_bin)
                                        img_crop_bin = img_poly_on_img_bin[y:y+h, x:x+w, :]
                                    
                                    mask_poly = np.zeros(img.shape)
                                    mask_poly = cv2.fillPoly(mask_poly, pts=[textline_coords], color=(1, 1, 1))
                                    
                                    
                                    mask_poly = mask_poly[y:y+h, x:x+w, :]
                                    img_crop = img_poly_on_img[y:y+h, x:x+w, :]
                                    
                                    if self.export_textline_images_and_text:
                                        if not self.do_not_mask_with_textline_contour:
                                            img_crop[mask_poly==0] = 255
                                        
                                    else:
                                        # print(file_name, angle_degrees, w*h,
                                        #       mask_poly[:,:,0].sum(),
                                        #       mask_poly[:,:,0].sum() /float(w*h) ,
                                        #       'didi')
                                        
                                        if angle_degrees > 3:
                                            better_des_slope = get_orientation_moments(textline_coords)
                                            
                                            img_crop = rotate_image_with_padding(img_crop, better_des_slope)
                                            if dir_in_bin is not None:
                                                img_crop_bin = rotate_image_with_padding(img_crop_bin, better_des_slope)
                                                
                                            mask_poly = rotate_image_with_padding(mask_poly, better_des_slope)
                                            mask_poly = mask_poly.astype('uint8')
                                            
                                            #new bounding box
                                            x_n, y_n, w_n, h_n = get_contours_and_bounding_boxes(mask_poly[:,:,0])
                                            
                                            mask_poly = mask_poly[y_n:y_n+h_n, x_n:x_n+w_n, :]
                                            img_crop = img_crop[y_n:y_n+h_n, x_n:x_n+w_n, :]
                                                
                                            if not self.do_not_mask_with_textline_contour:
                                                img_crop[mask_poly==0] = 255
                                            if dir_in_bin is not None:
                                                img_crop_bin = img_crop_bin[y_n:y_n+h_n, x_n:x_n+w_n, :]
                                                if not self.do_not_mask_with_textline_contour:
                                                    img_crop_bin[mask_poly==0] = 255
                                            
                                            if mask_poly[:,:,0].sum() /float(w_n*h_n) < 0.50 and w_scaled > 90:
                                                if dir_in_bin is not None:
                                                    img_crop, img_crop_bin = \
                                                        break_curved_line_into_small_pieces_and_then_merge(
                                                            img_crop, mask_poly, img_crop_bin)
                                                else:
                                                    img_crop, _ = \
                                                        break_curved_line_into_small_pieces_and_then_merge(
                                                            img_crop, mask_poly)
        
                                        else:
                                            better_des_slope = 0
                                            if not self.do_not_mask_with_textline_contour:
                                                img_crop[mask_poly==0] = 255
                                            if dir_in_bin is not None:
                                                if not self.do_not_mask_with_textline_contour:
                                                    img_crop_bin[mask_poly==0] = 255
                                            if type_textregion=='drop-capital':
                                                pass
                                            else:
                                                if mask_poly[:,:,0].sum() /float(w*h) < 0.50 and w_scaled > 90:
                                                    if dir_in_bin is not None:
                                                        img_crop, img_crop_bin = \
                                                            break_curved_line_into_small_pieces_and_then_merge(
                                                                img_crop, mask_poly, img_crop_bin)
                                                    else:
                                                        img_crop, _ = \
                                                            break_curved_line_into_small_pieces_and_then_merge(
                                                                img_crop, mask_poly)
                                    
                                    if not self.export_textline_images_and_text:
                                        if w_scaled < 750:#1.5*image_width:
                                            img_fin = preprocess_and_resize_image_for_ocrcnn_model(
                                                img_crop, image_height, image_width)
                                            cropped_lines.append(img_fin)
                                            if abs(better_des_slope) > 45:
                                                cropped_lines_ver_index.append(1)
                                            else:
                                                cropped_lines_ver_index.append(0)
                                                
                                            cropped_lines_meging_indexing.append(0)
                                            if dir_in_bin is not None:
                                                img_fin = preprocess_and_resize_image_for_ocrcnn_model(
                                                    img_crop_bin, image_height, image_width)
                                                cropped_lines_bin.append(img_fin)
                                        else:
                                            splited_images, splited_images_bin = return_textlines_split_if_needed(
                                                img_crop, img_crop_bin if dir_in_bin is not None else None)
                                            if splited_images:
                                                img_fin = preprocess_and_resize_image_for_ocrcnn_model(
                                                    splited_images[0], image_height, image_width)
                                                cropped_lines.append(img_fin)
                                                cropped_lines_meging_indexing.append(1)
                                                
                                                if abs(better_des_slope) > 45:
                                                    cropped_lines_ver_index.append(1)
                                                else:
                                                    cropped_lines_ver_index.append(0)
                                                
                                                img_fin = preprocess_and_resize_image_for_ocrcnn_model(
                                                    splited_images[1], image_height, image_width)
                                                
                                                cropped_lines.append(img_fin)
                                                cropped_lines_meging_indexing.append(-1)
                                                
                                                if abs(better_des_slope) > 45:
                                                    cropped_lines_ver_index.append(1)
                                                else:
                                                    cropped_lines_ver_index.append(0)
                                                
                                                if dir_in_bin is not None:
                                                    img_fin = preprocess_and_resize_image_for_ocrcnn_model(
                                                        splited_images_bin[0], image_height, image_width)
                                                    cropped_lines_bin.append(img_fin)
                                                    img_fin = preprocess_and_resize_image_for_ocrcnn_model(
                                                        splited_images_bin[1], image_height, image_width)
                                                    cropped_lines_bin.append(img_fin)
                                                    
                                            else:
                                                img_fin = preprocess_and_resize_image_for_ocrcnn_model(
                                                    img_crop, image_height, image_width)
                                                cropped_lines.append(img_fin)
                                                cropped_lines_meging_indexing.append(0)
                                                
                                                if abs(better_des_slope) > 45:
                                                    cropped_lines_ver_index.append(1)
                                                else:
                                                    cropped_lines_ver_index.append(0)
                                                
                                                if dir_in_bin is not None:
                                                    img_fin = preprocess_and_resize_image_for_ocrcnn_model(
                                                        img_crop_bin, image_height, image_width)
                                                    cropped_lines_bin.append(img_fin)
                                        
                                if self.export_textline_images_and_text:
                                    if img_crop.shape[0]==0 or img_crop.shape[1]==0:
                                        pass
                                    else:
                                        if child_textlines.tag.endswith("TextEquiv"):
                                            for cheild_text in child_textlines:
                                                if cheild_text.tag.endswith("Unicode"):
                                                    textline_text = cheild_text.text
                                                    if textline_text:
                                                        base_name = os.path.join(
                                                            dir_out, file_name + '_line_' + str(indexer_textlines))
                                                        if self.pref_of_dataset:
                                                            base_name += '_' + self.pref_of_dataset
                                                        if not self.do_not_mask_with_textline_contour:
                                                            base_name += '_masked'
                                                            
                                                        with open(base_name + '.txt', 'w') as text_file:
                                                            text_file.write(textline_text)
                                                        cv2.imwrite(base_name + '.png', img_crop)
                                                    indexer_textlines+=1

                    if not self.export_textline_images_and_text:
                        indexer_text_region = indexer_text_region +1
                    
                if not self.export_textline_images_and_text:
                    extracted_texts = []
                    extracted_conf_value = []

                    n_iterations  = math.ceil(len(cropped_lines) / self.b_s) 

                    for i in range(n_iterations):
                        if i==(n_iterations-1):
                            n_start = i*self.b_s
                            imgs = cropped_lines[n_start:]
                            imgs = np.array(imgs)
                            imgs = imgs.reshape(imgs.shape[0], image_height, image_width, 3)
                            
                            ver_imgs = np.array( cropped_lines_ver_index[n_start:] )
                            indices_ver = np.where(ver_imgs == 1)[0]
                            
                            #print(indices_ver, 'indices_ver')
                            if len(indices_ver)>0:
                                imgs_ver_flipped = imgs[indices_ver, : ,: ,:]
                                imgs_ver_flipped = imgs_ver_flipped[:,::-1,::-1,:]
                                #print(imgs_ver_flipped, 'imgs_ver_flipped')
                                
                            else:
                                imgs_ver_flipped = None
                            
                            if dir_in_bin is not None:
                                imgs_bin = cropped_lines_bin[n_start:]
                                imgs_bin = np.array(imgs_bin)
                                imgs_bin = imgs_bin.reshape(imgs_bin.shape[0], image_height, image_width, 3)
                                
                                if len(indices_ver)>0:
                                    imgs_bin_ver_flipped = imgs_bin[indices_ver, : ,: ,:]
                                    imgs_bin_ver_flipped = imgs_bin_ver_flipped[:,::-1,::-1,:]
                                    #print(imgs_ver_flipped, 'imgs_ver_flipped')
                                    
                                else:
                                    imgs_bin_ver_flipped = None
                        else:
                            n_start = i*self.b_s
                            n_end = (i+1)*self.b_s
                            imgs = cropped_lines[n_start:n_end]
                            imgs = np.array(imgs).reshape(self.b_s, image_height, image_width, 3)
                            
                            ver_imgs = np.array( cropped_lines_ver_index[n_start:n_end] )
                            indices_ver = np.where(ver_imgs == 1)[0]
                            #print(indices_ver, 'indices_ver')
                            
                            if len(indices_ver)>0:
                                imgs_ver_flipped = imgs[indices_ver, : ,: ,:]
                                imgs_ver_flipped = imgs_ver_flipped[:,::-1,::-1,:]
                                #print(imgs_ver_flipped, 'imgs_ver_flipped')
                            else:
                                imgs_ver_flipped = None

                            
                            if dir_in_bin is not None:
                                imgs_bin = cropped_lines_bin[n_start:n_end]
                                imgs_bin = np.array(imgs_bin).reshape(self.b_s, image_height, image_width, 3)
                                
                                
                                if len(indices_ver)>0:
                                    imgs_bin_ver_flipped = imgs_bin[indices_ver, : ,: ,:]
                                    imgs_bin_ver_flipped = imgs_bin_ver_flipped[:,::-1,::-1,:]
                                    #print(imgs_ver_flipped, 'imgs_ver_flipped')
                                else:
                                    imgs_bin_ver_flipped = None
                            

                        self.logger.debug("processing next %d lines", len(imgs))
                        preds = self.model_zoo.get('ocr').predict(imgs, verbose=0)
                        
                        if len(indices_ver)>0:
                            preds_flipped = self.model_zoo.get('ocr').predict(imgs_ver_flipped, verbose=0)
                            preds_max_fliped = np.max(preds_flipped, axis=2 )
                            preds_max_args_flipped = np.argmax(preds_flipped, axis=2 )
                            pred_max_not_unk_mask_bool_flipped = preds_max_args_flipped[:,:]!=self.end_character
                            masked_means_flipped = \
                                np.sum(preds_max_fliped * pred_max_not_unk_mask_bool_flipped, axis=1) / \
                                np.sum(pred_max_not_unk_mask_bool_flipped, axis=1)
                            masked_means_flipped[np.isnan(masked_means_flipped)] = 0
                            
                            preds_max = np.max(preds, axis=2 )
                            preds_max_args = np.argmax(preds, axis=2 )
                            pred_max_not_unk_mask_bool = preds_max_args[:,:]!=self.end_character
                            
                            masked_means = \
                                np.sum(preds_max * pred_max_not_unk_mask_bool, axis=1) / \
                                np.sum(pred_max_not_unk_mask_bool, axis=1)
                            masked_means[np.isnan(masked_means)] = 0
                            
                            masked_means_ver = masked_means[indices_ver]
                            #print(masked_means_ver, 'pred_max_not_unk')
                            
                            indices_where_flipped_conf_value_is_higher = \
                                np.where(masked_means_flipped > masked_means_ver)[0]
                            
                            #print(indices_where_flipped_conf_value_is_higher, 'indices_where_flipped_conf_value_is_higher')
                            if len(indices_where_flipped_conf_value_is_higher)>0:
                                indices_to_be_replaced = indices_ver[indices_where_flipped_conf_value_is_higher]
                                preds[indices_to_be_replaced,:,:] = \
                                    preds_flipped[indices_where_flipped_conf_value_is_higher, :, :]
                        if dir_in_bin is not None:
                            preds_bin = self.model_zoo.get('ocr').predict(imgs_bin, verbose=0)
                            
                            if len(indices_ver)>0:
                                preds_flipped = self.model_zoo.get('ocr').predict(imgs_bin_ver_flipped, verbose=0)
                                preds_max_fliped = np.max(preds_flipped, axis=2 )
                                preds_max_args_flipped = np.argmax(preds_flipped, axis=2 )
                                pred_max_not_unk_mask_bool_flipped = preds_max_args_flipped[:,:]!=self.end_character
                                masked_means_flipped = \
                                    np.sum(preds_max_fliped * pred_max_not_unk_mask_bool_flipped, axis=1) / \
                                    np.sum(pred_max_not_unk_mask_bool_flipped, axis=1)
                                masked_means_flipped[np.isnan(masked_means_flipped)] = 0
                                
                                preds_max = np.max(preds, axis=2 )
                                preds_max_args = np.argmax(preds, axis=2 )
                                pred_max_not_unk_mask_bool = preds_max_args[:,:]!=self.end_character
                                
                                masked_means = \
                                    np.sum(preds_max * pred_max_not_unk_mask_bool, axis=1) / \
                                    np.sum(pred_max_not_unk_mask_bool, axis=1)
                                masked_means[np.isnan(masked_means)] = 0
                                
                                masked_means_ver = masked_means[indices_ver]
                                #print(masked_means_ver, 'pred_max_not_unk')
                                
                                indices_where_flipped_conf_value_is_higher = \
                                    np.where(masked_means_flipped > masked_means_ver)[0]
                                
                                #print(indices_where_flipped_conf_value_is_higher, 'indices_where_flipped_conf_value_is_higher')
                                if len(indices_where_flipped_conf_value_is_higher)>0:
                                    indices_to_be_replaced = indices_ver[indices_where_flipped_conf_value_is_higher]
                                    preds_bin[indices_to_be_replaced,:,:] = \
                                        preds_flipped[indices_where_flipped_conf_value_is_higher, :, :]
                            
                            preds = (preds + preds_bin) / 2.

                        pred_texts = decode_batch_predictions(preds, self.model_zoo.get('num_to_char'))
                        
                        preds_max = np.max(preds, axis=2 )
                        preds_max_args = np.argmax(preds, axis=2 )
                        pred_max_not_unk_mask_bool = preds_max_args[:,:]!=self.end_character
                        masked_means = \
                            np.sum(preds_max * pred_max_not_unk_mask_bool, axis=1) / \
                            np.sum(pred_max_not_unk_mask_bool, axis=1)

                        for ib in range(imgs.shape[0]):
                            pred_texts_ib = pred_texts[ib].replace("[UNK]", "")
                            if masked_means[ib] >= self.min_conf_value_of_textline_text:
                                extracted_texts.append(pred_texts_ib)
                                extracted_conf_value.append(masked_means[ib])
                            else:
                                extracted_texts.append("")
                                extracted_conf_value.append(0)
                    del cropped_lines
                    if dir_in_bin is not None:
                        del cropped_lines_bin
                    gc.collect()
                    
                    extracted_texts_merged = [extracted_texts[ind]
                                              if cropped_lines_meging_indexing[ind]==0
                                              else extracted_texts[ind]+" "+extracted_texts[ind+1]
                                              if cropped_lines_meging_indexing[ind]==1
                                              else None
                                              for ind in range(len(cropped_lines_meging_indexing))]
                    
                    extracted_conf_value_merged = [extracted_conf_value[ind]
                                                   if cropped_lines_meging_indexing[ind]==0
                                                   else (extracted_conf_value[ind]+extracted_conf_value[ind+1])/2.
                                                   if cropped_lines_meging_indexing[ind]==1
                                                   else None
                                                   for ind in range(len(cropped_lines_meging_indexing))]

                    extracted_conf_value_merged = [extracted_conf_value_merged[ind_cfm]
                                                   for ind_cfm in range(len(extracted_texts_merged))
                                                   if extracted_texts_merged[ind_cfm] is not None]
                    extracted_texts_merged = [ind for ind in extracted_texts_merged if ind is not None]
                    unique_cropped_lines_region_indexer = np.unique(cropped_lines_region_indexer)
                    
                    if dir_out_image_text:
                        #font_path = "Charis-7.000/Charis-Regular.ttf"  # Make sure this file exists!
                        font = importlib_resources.files(__package__) / "Charis-Regular.ttf"
                        with importlib_resources.as_file(font) as font:
                            font = ImageFont.truetype(font=font, size=40)
                        
                        for indexer_text, bb_ind in enumerate(total_bb_coordinates):
                            x_bb = bb_ind[0]
                            y_bb = bb_ind[1]
                            w_bb = bb_ind[2]
                            h_bb = bb_ind[3]
                            
                            font = fit_text_single_line(draw, extracted_texts_merged[indexer_text],
                                                        font.path, w_bb, int(h_bb*0.4) )
                            
                            ##draw.rectangle([x_bb, y_bb, x_bb + w_bb, y_bb + h_bb], outline="red", width=2)
                            
                            text_bbox = draw.textbbox((0, 0), extracted_texts_merged[indexer_text], font=font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]

                            text_x = x_bb + (w_bb - text_width) // 2  # Center horizontally
                            text_y = y_bb + (h_bb - text_height) // 2  # Center vertically

                            # Draw the text
                            draw.text((text_x, text_y), extracted_texts_merged[indexer_text], fill="black", font=font)
                        image_text.save(out_image_with_text)

                    text_by_textregion = []
                    for ind in unique_cropped_lines_region_indexer:
                        ind = np.array(cropped_lines_region_indexer)==ind
                        extracted_texts_merged_un = np.array(extracted_texts_merged)[ind]
                        if len(extracted_texts_merged_un)>1:
                            text_by_textregion_ind = ""
                            next_glue = ""
                            for indt in range(len(extracted_texts_merged_un)):
                                if (extracted_texts_merged_un[indt].endswith('⸗') or
                                    extracted_texts_merged_un[indt].endswith('-') or
                                    extracted_texts_merged_un[indt].endswith('¬')):
                                    text_by_textregion_ind += next_glue + extracted_texts_merged_un[indt][:-1]
                                    next_glue = ""
                                else:
                                    text_by_textregion_ind += next_glue + extracted_texts_merged_un[indt]
                                    next_glue = " "
                            text_by_textregion.append(text_by_textregion_ind)
                        else:
                            text_by_textregion.append(" ".join(extracted_texts_merged_un))
                        #print(text_by_textregion, 'text_by_textregiontext_by_textregiontext_by_textregiontext_by_textregiontext_by_textregion')

                    ###index_tot_regions = []
                    ###tot_region_ref = []

                    ###for jj in root1.iter(link+'RegionRefIndexed'):
                        ###index_tot_regions.append(jj.attrib['index'])
                        ###tot_region_ref.append(jj.attrib['regionRef'])
                        
                    ###id_to_order = {tid: ro for tid, ro in zip(tot_region_ref, index_tot_regions)}
        
                    #id_textregions = []
                    #textregions_by_existing_ids = []
                    indexer = 0
                    indexer_textregion = 0
                    for nn in root1.iter(region_tags):
                        #id_textregion = nn.attrib['id']
                        #id_textregions.append(id_textregion)
                        #textregions_by_existing_ids.append(text_by_textregion[indexer_textregion])
                        
                        is_textregion_text = False
                        for childtest in nn:
                            if childtest.tag.endswith("TextEquiv"):
                                is_textregion_text = True
                        
                        if not is_textregion_text:
                            text_subelement_textregion = ET.SubElement(nn, 'TextEquiv')
                            unicode_textregion = ET.SubElement(text_subelement_textregion, 'Unicode')

                        
                        has_textline = False
                        for child_textregion in nn:
                            if child_textregion.tag.endswith("TextLine"):
                                
                                is_textline_text = False
                                for childtest2 in child_textregion:
                                    if childtest2.tag.endswith("TextEquiv"):
                                        is_textline_text = True
                                
                                
                                if not is_textline_text:
                                    text_subelement = ET.SubElement(child_textregion, 'TextEquiv')
                                    text_subelement.set('conf', f"{extracted_conf_value_merged[indexer]:.2f}")
                                    unicode_textline = ET.SubElement(text_subelement, 'Unicode')
                                    unicode_textline.text = extracted_texts_merged[indexer]
                                else:
                                    for childtest3 in child_textregion:
                                        if childtest3.tag.endswith("TextEquiv"):
                                            for child_uc in childtest3:
                                                if child_uc.tag.endswith("Unicode"):
                                                    childtest3.set('conf',
                                                                   f"{extracted_conf_value_merged[indexer]:.2f}")
                                                    child_uc.text = extracted_texts_merged[indexer]
                                        
                                indexer = indexer + 1
                                has_textline = True
                        if has_textline:
                            if is_textregion_text:
                                for child4 in nn:
                                    if child4.tag.endswith("TextEquiv"):
                                        for childtr_uc in child4:
                                            if childtr_uc.tag.endswith("Unicode"):
                                                childtr_uc.text = text_by_textregion[indexer_textregion]
                            else:
                                unicode_textregion.text = text_by_textregion[indexer_textregion]
                            indexer_textregion = indexer_textregion + 1
                            
                    ###sample_order  = [(id_to_order[tid], text)
                    ###                 for tid, text in zip(id_textregions, textregions_by_existing_ids)
                    ###                 if tid in id_to_order]
                    
                    ##ordered_texts_sample = [text for _, text in sorted(sample_order)]
                    ##tot_page_text = ' '.join(ordered_texts_sample)
                    
                    ##for page_element in root1.iter(link+'Page'):
                        ##text_page = ET.SubElement(page_element, 'TextEquiv')
                        ##unicode_textpage = ET.SubElement(text_page, 'Unicode')
                        ##unicode_textpage.text = tot_page_text
                    
                    ET.register_namespace("",name_space)
                    tree1.write(out_file_ocr,xml_declaration=True,method='xml',encoding="utf-8",default_namespace=None)
                    #print("Job done in %.1fs", time.time() - t0)
