"""
Machine learning based reading order detection
"""

# pyright: reportCallIssue=false
# pyright: reportUnboundVariable=false
# pyright: reportArgumentType=false

import logging 
import os
import time
from typing import Optional
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
from keras.models import Model
import numpy as np
import statistics
import tensorflow as tf

from .model_zoo import EynollahModelZoo
from .utils.resize import resize_image
from .utils.contour import (
    find_new_features_of_contours,
    return_contours_of_image,
    return_parent_contours,
)
from .utils import is_xml_filename

DPI_THRESHOLD = 298
KERNEL = np.ones((5, 5), np.uint8)


class machine_based_reading_order_on_layout:
    def __init__(
        self,
        *,
        model_zoo: EynollahModelZoo,
        logger : Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger('eynollah.mbreorder')
        self.model_zoo = model_zoo
        
        try:
            for device in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(device, True)
        except:
            self.logger.warning("no GPU device available")
            
        self.model_zoo.load_model('reading_order')

    def read_xml(self, xml_file):
        tree1 = ET.parse(xml_file, parser = ET.XMLParser(encoding='utf-8'))
        root1=tree1.getroot()
        alltags=[elem.tag for elem in root1.iter()]
        link=alltags[0].split('}')[0]+'}'

        index_tot_regions = []
        tot_region_ref = []

        y_len, x_len = 0, 0
        for jj in root1.iter(link+'Page'):
            y_len=int(jj.attrib['imageHeight'])
            x_len=int(jj.attrib['imageWidth'])

        for jj in root1.iter(link+'RegionRefIndexed'):
            index_tot_regions.append(jj.attrib['index'])
            tot_region_ref.append(jj.attrib['regionRef'])
            
        if (link+'PrintSpace' in alltags) or  (link+'Border' in alltags):
            co_printspace = []
            if link+'PrintSpace' in alltags:
                region_tags_printspace = np.unique([x for x in alltags if x.endswith('PrintSpace')])
            else:
                region_tags_printspace = np.unique([x for x in alltags if x.endswith('Border')])
                
            for tag in region_tags_printspace:
                if link+'PrintSpace' in alltags:
                    tag_endings_printspace = ['}PrintSpace','}printspace']
                else:
                    tag_endings_printspace = ['}Border','}border']
                    
                if tag.endswith(tag_endings_printspace[0]) or tag.endswith(tag_endings_printspace[1]):
                    for nn in root1.iter(tag):
                        c_t_in = []
                        sumi = 0
                        for vv in nn.iter():
                            # check the format of coords
                            if vv.tag == link + 'Coords':
                                coords = bool(vv.attrib)
                                if coords:
                                    p_h = vv.attrib['points'].split(' ')
                                    c_t_in.append(
                                        np.array([[int(x.split(',')[0]), int(x.split(',')[1])] for x in p_h]))
                                    break
                                else:
                                    pass

                            if vv.tag == link + 'Point':
                                c_t_in.append([int(float(vv.attrib['x'])), int(float(vv.attrib['y']))])
                                sumi += 1
                            elif vv.tag != link + 'Point' and sumi >= 1:
                                break
                        co_printspace.append(np.array(c_t_in))
            img_printspace = np.zeros( (y_len,x_len,3) ) 
            img_printspace=cv2.fillPoly(img_printspace, pts =co_printspace, color=(1,1,1))
            img_printspace = img_printspace.astype(np.uint8)
            
            imgray = cv2.cvtColor(img_printspace, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgray, 0, 255, 0)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])
            cnt = contours[np.argmax(cnt_size)]
            x, y, w, h = cv2.boundingRect(cnt)
            
            bb_coord_printspace = [x, y, w, h]
                        
        else:
            bb_coord_printspace = None
                        

        region_tags=np.unique([x for x in alltags if x.endswith('Region')])   
        co_text_paragraph=[]
        co_text_drop=[]
        co_text_heading=[]
        co_text_header=[]
        co_text_marginalia=[]
        co_text_catch=[]
        co_text_page_number=[]
        co_text_signature_mark=[]
        co_sep=[]
        co_img=[]
        co_table=[]
        co_graphic=[]
        co_graphic_text_annotation=[]
        co_graphic_decoration=[]
        co_noise=[]

        co_text_paragraph_text=[]
        co_text_drop_text=[]
        co_text_heading_text=[]
        co_text_header_text=[]
        co_text_marginalia_text=[]
        co_text_catch_text=[]
        co_text_page_number_text=[]
        co_text_signature_mark_text=[]
        co_sep_text=[]
        co_img_text=[]
        co_table_text=[]
        co_graphic_text=[]
        co_graphic_text_annotation_text=[]
        co_graphic_decoration_text=[]
        co_noise_text=[]

        id_paragraph = []
        id_header = []
        id_heading = []
        id_marginalia = []

        for tag in region_tags:
            if tag.endswith('}TextRegion') or tag.endswith('}Textregion'):
                for nn in root1.iter(tag):
                    for child2 in nn:
                        tag2 = child2.tag
                        if tag2.endswith('}TextEquiv') or tag2.endswith('}TextEquiv'):
                            for childtext2 in child2:
                                if childtext2.tag.endswith('}Unicode') or childtext2.tag.endswith('}Unicode'):
                                    if "type" in nn.attrib and nn.attrib['type']=='drop-capital':
                                        co_text_drop_text.append(childtext2.text)
                                    elif "type" in nn.attrib and nn.attrib['type']=='heading':
                                        co_text_heading_text.append(childtext2.text)
                                    elif "type" in nn.attrib and nn.attrib['type']=='signature-mark':
                                        co_text_signature_mark_text.append(childtext2.text)
                                    elif "type" in nn.attrib and nn.attrib['type']=='header':
                                        co_text_header_text.append(childtext2.text)
                                    ###elif "type" in nn.attrib and nn.attrib['type']=='catch-word':
                                        ###co_text_catch_text.append(childtext2.text)
                                    ###elif "type" in nn.attrib and nn.attrib['type']=='page-number':
                                        ###co_text_page_number_text.append(childtext2.text)
                                    elif "type" in nn.attrib and nn.attrib['type']=='marginalia':
                                        co_text_marginalia_text.append(childtext2.text)
                                    else:
                                        co_text_paragraph_text.append(childtext2.text)
                    c_t_in_drop=[]
                    c_t_in_paragraph=[]
                    c_t_in_heading=[]
                    c_t_in_header=[]
                    c_t_in_page_number=[]
                    c_t_in_signature_mark=[]
                    c_t_in_catch=[]
                    c_t_in_marginalia=[]


                    sumi=0
                    for vv in nn.iter():
                        # check the format of coords
                        if vv.tag==link+'Coords':

                            coords=bool(vv.attrib)
                            if coords:
                                #print('birda1')
                                p_h=vv.attrib['points'].split(' ')



                                if "type" in nn.attrib and nn.attrib['type']=='drop-capital':

                                    c_t_in_drop.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )

                                elif "type" in nn.attrib and nn.attrib['type']=='heading':
                                    ##id_heading.append(nn.attrib['id'])
                                    c_t_in_heading.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )


                                elif "type" in nn.attrib and nn.attrib['type']=='signature-mark':

                                    c_t_in_signature_mark.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                    #print(c_t_in_paragraph)
                                elif "type" in nn.attrib and nn.attrib['type']=='header':
                                    #id_header.append(nn.attrib['id'])
                                    c_t_in_header.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )


                                ###elif "type" in nn.attrib and nn.attrib['type']=='catch-word':
                                    ###c_t_in_catch.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )


                                ###elif "type" in nn.attrib and nn.attrib['type']=='page-number':

                                    ###c_t_in_page_number.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )

                                elif "type" in nn.attrib and nn.attrib['type']=='marginalia':
                                    #id_marginalia.append(nn.attrib['id'])

                                    c_t_in_marginalia.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                else:
                                    #id_paragraph.append(nn.attrib['id'])

                                    c_t_in_paragraph.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )

                                break
                            else:
                                pass


                        if vv.tag==link+'Point':
                            if "type" in nn.attrib and nn.attrib['type']=='drop-capital':

                                c_t_in_drop.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1

                            elif "type" in nn.attrib and nn.attrib['type']=='heading':
                                #id_heading.append(nn.attrib['id'])
                                c_t_in_heading.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1


                            elif "type" in nn.attrib and nn.attrib['type']=='signature-mark':

                                c_t_in_signature_mark.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1
                            elif "type" in nn.attrib and nn.attrib['type']=='header':
                                #id_header.append(nn.attrib['id'])
                                c_t_in_header.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1


                            ###elif "type" in nn.attrib and nn.attrib['type']=='catch-word':
                                ###c_t_in_catch.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                ###sumi+=1

                            ###elif "type" in nn.attrib and nn.attrib['type']=='page-number':

                                ###c_t_in_page_number.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                ###sumi+=1

                            elif "type" in nn.attrib and nn.attrib['type']=='marginalia':
                                #id_marginalia.append(nn.attrib['id'])

                                c_t_in_marginalia.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1

                            else:
                                #id_paragraph.append(nn.attrib['id'])
                                c_t_in_paragraph.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1

                        elif vv.tag!=link+'Point' and sumi>=1:
                            break

                    if len(c_t_in_drop)>0:
                        co_text_drop.append(np.array(c_t_in_drop))
                    if len(c_t_in_paragraph)>0:
                        co_text_paragraph.append(np.array(c_t_in_paragraph))
                        id_paragraph.append(nn.attrib['id'])
                    if len(c_t_in_heading)>0:
                        co_text_heading.append(np.array(c_t_in_heading))
                        id_heading.append(nn.attrib['id'])

                    if len(c_t_in_header)>0:
                        co_text_header.append(np.array(c_t_in_header))
                        id_header.append(nn.attrib['id'])
                    if len(c_t_in_page_number)>0:
                        co_text_page_number.append(np.array(c_t_in_page_number))
                    if len(c_t_in_catch)>0:
                        co_text_catch.append(np.array(c_t_in_catch))

                    if len(c_t_in_signature_mark)>0:
                        co_text_signature_mark.append(np.array(c_t_in_signature_mark))

                    if len(c_t_in_marginalia)>0:
                        co_text_marginalia.append(np.array(c_t_in_marginalia))
                        id_marginalia.append(nn.attrib['id'])


            elif tag.endswith('}GraphicRegion') or tag.endswith('}graphicregion'):
                for nn in root1.iter(tag):
                    c_t_in=[]
                    c_t_in_text_annotation=[]
                    c_t_in_decoration=[]
                    sumi=0
                    for vv in nn.iter():
                        # check the format of coords
                        if vv.tag==link+'Coords':
                            coords=bool(vv.attrib)
                            if coords:
                                p_h=vv.attrib['points'].split(' ')

                                if "type" in nn.attrib and nn.attrib['type']=='handwritten-annotation':
                                    c_t_in_text_annotation.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                    
                                elif "type" in nn.attrib and nn.attrib['type']=='decoration':
                                    c_t_in_decoration.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                    
                                else:
                                    c_t_in.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )


                                break
                            else:
                                pass


                        if vv.tag==link+'Point':
                            if "type" in nn.attrib and nn.attrib['type']=='handwritten-annotation':
                                c_t_in_text_annotation.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1

                            elif "type" in nn.attrib and nn.attrib['type']=='decoration':
                                c_t_in_decoration.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1
                                
                            else:
                                c_t_in.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                                sumi+=1

                    if len(c_t_in_text_annotation)>0:
                        co_graphic_text_annotation.append(np.array(c_t_in_text_annotation))
                    if len(c_t_in_decoration)>0:
                        co_graphic_decoration.append(np.array(c_t_in_decoration))
                    if len(c_t_in)>0:
                        co_graphic.append(np.array(c_t_in))



            elif tag.endswith('}ImageRegion') or tag.endswith('}imageregion'):
                for nn in root1.iter(tag):
                    c_t_in=[]
                    sumi=0
                    for vv in nn.iter():
                        # check the format of coords
                        if vv.tag==link+'Coords':
                            coords=bool(vv.attrib)
                            if coords:
                                p_h=vv.attrib['points'].split(' ')
                                c_t_in.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                break
                            else:
                                pass


                        if vv.tag==link+'Point':
                            c_t_in.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                            sumi+=1
                        elif vv.tag!=link+'Point' and sumi>=1:
                            break
                    co_img.append(np.array(c_t_in))
                    co_img_text.append(' ')


            elif tag.endswith('}SeparatorRegion') or tag.endswith('}separatorregion'):
                for nn in root1.iter(tag):
                    c_t_in=[]
                    sumi=0
                    for vv in nn.iter():
                        # check the format of coords
                        if vv.tag==link+'Coords':
                            coords=bool(vv.attrib)
                            if coords:
                                p_h=vv.attrib['points'].split(' ')
                                c_t_in.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                break
                            else:
                                pass


                        if vv.tag==link+'Point':
                            c_t_in.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                            sumi+=1
                        elif vv.tag!=link+'Point' and sumi>=1:
                            break
                    co_sep.append(np.array(c_t_in))



            elif tag.endswith('}TableRegion') or tag.endswith('}tableregion'):
                for nn in root1.iter(tag):
                    c_t_in=[]
                    sumi=0
                    for vv in nn.iter():
                        # check the format of coords
                        if vv.tag==link+'Coords':
                            coords=bool(vv.attrib)
                            if coords:
                                p_h=vv.attrib['points'].split(' ')
                                c_t_in.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                break
                            else:
                                pass


                        if vv.tag==link+'Point':
                            c_t_in.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                            sumi+=1
                            
                        elif vv.tag!=link+'Point' and sumi>=1:
                            break
                    co_table.append(np.array(c_t_in))
                    co_table_text.append(' ')

            elif tag.endswith('}NoiseRegion') or tag.endswith('}noiseregion'):
                for nn in root1.iter(tag):
                    c_t_in=[]
                    sumi=0
                    for vv in nn.iter():
                        # check the format of coords
                        if vv.tag==link+'Coords':
                            coords=bool(vv.attrib)
                            if coords:
                                p_h=vv.attrib['points'].split(' ')
                                c_t_in.append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                break
                            else:
                                pass


                        if vv.tag==link+'Point':
                            c_t_in.append([ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ])
                            sumi+=1

                        elif vv.tag!=link+'Point' and sumi>=1:
                            break
                    co_noise.append(np.array(c_t_in))
                    co_noise_text.append(' ')

        img = np.zeros( (y_len,x_len,3) ) 
        img_poly=cv2.fillPoly(img, pts =co_text_paragraph, color=(1,1,1))

        img_poly=cv2.fillPoly(img, pts =co_text_heading, color=(2,2,2))
        img_poly=cv2.fillPoly(img, pts =co_text_header, color=(2,2,2))
        img_poly=cv2.fillPoly(img, pts =co_text_marginalia, color=(3,3,3))
        img_poly=cv2.fillPoly(img, pts =co_img, color=(4,4,4))
        img_poly=cv2.fillPoly(img, pts =co_sep, color=(5,5,5))

        return tree1, root1, bb_coord_printspace, id_paragraph, id_header+id_heading, co_text_paragraph, co_text_header+co_text_heading,\
    tot_region_ref,x_len, y_len,index_tot_regions, img_poly

    def return_indexes_of_contours_loctaed_inside_another_list_of_contours(self, contours, contours_loc, cx_main_loc, cy_main_loc, indexes_loc):
        indexes_of_located_cont = []
        center_x_coordinates_of_located = []
        center_y_coordinates_of_located = []
        #M_main_tot = [cv2.moments(contours_loc[j])
                        #for j in range(len(contours_loc))]
        #cx_main_loc = [(M_main_tot[j]["m10"] / (M_main_tot[j]["m00"] + 1e-32)) for j in range(len(M_main_tot))]
        #cy_main_loc = [(M_main_tot[j]["m01"] / (M_main_tot[j]["m00"] + 1e-32)) for j in range(len(M_main_tot))]
        
        for ij in range(len(contours)):
            results = [cv2.pointPolygonTest(contours[ij], (cx_main_loc[ind], cy_main_loc[ind]), False)
                        for ind in range(len(cy_main_loc)) ]
            results = np.array(results)
            indexes_in = np.where((results == 0) | (results == 1))
            indexes = indexes_loc[indexes_in]# [(results == 0) | (results == 1)]#np.where((results == 0) | (results == 1))

            indexes_of_located_cont.append(indexes)
            center_x_coordinates_of_located.append(np.array(cx_main_loc)[indexes_in] )
            center_y_coordinates_of_located.append(np.array(cy_main_loc)[indexes_in] )
            
        return indexes_of_located_cont, center_x_coordinates_of_located, center_y_coordinates_of_located

    def do_order_of_regions_with_model(self, contours_only_text_parent, contours_only_text_parent_h, text_regions_p):
        height1 =672#448
        width1 = 448#224

        height2 =672#448
        width2= 448#224

        height3 =672#448
        width3 = 448#224
        
        inference_bs = 3
        
        ver_kernel = np.ones((5, 1), dtype=np.uint8)
        hor_kernel = np.ones((1, 5), dtype=np.uint8)
        
        
        min_cont_size_to_be_dilated = 10
        if len(contours_only_text_parent)>min_cont_size_to_be_dilated:
            cx_conts, cy_conts, x_min_conts, x_max_conts, y_min_conts, y_max_conts, _ = find_new_features_of_contours(contours_only_text_parent)
            args_cont_located = np.array(range(len(contours_only_text_parent)))
            
            diff_y_conts = np.abs(y_max_conts[:]-y_min_conts)
            diff_x_conts = np.abs(x_max_conts[:]-x_min_conts)
            
            mean_x = statistics.mean(diff_x_conts)
            median_x = statistics.median(diff_x_conts)
            
            
            diff_x_ratio= diff_x_conts/mean_x
            
            args_cont_located_excluded = args_cont_located[diff_x_ratio>=1.3]
            args_cont_located_included = args_cont_located[diff_x_ratio<1.3]
            
            contours_only_text_parent_excluded = [contours_only_text_parent[ind] for ind in range(len(contours_only_text_parent)) if diff_x_ratio[ind]>=1.3]#contours_only_text_parent[diff_x_ratio>=1.3]
            contours_only_text_parent_included = [contours_only_text_parent[ind] for ind in range(len(contours_only_text_parent)) if diff_x_ratio[ind]<1.3]#contours_only_text_parent[diff_x_ratio<1.3]
            
            
            cx_conts_excluded = [cx_conts[ind] for ind in range(len(cx_conts)) if diff_x_ratio[ind]>=1.3]#cx_conts[diff_x_ratio>=1.3]
            cx_conts_included = [cx_conts[ind] for ind in range(len(cx_conts)) if diff_x_ratio[ind]<1.3]#cx_conts[diff_x_ratio<1.3]
            
            cy_conts_excluded = [cy_conts[ind] for ind in range(len(cy_conts)) if diff_x_ratio[ind]>=1.3]#cy_conts[diff_x_ratio>=1.3]
            cy_conts_included = [cy_conts[ind] for ind in range(len(cy_conts)) if diff_x_ratio[ind]<1.3]#cy_conts[diff_x_ratio<1.3]
            
            #print(diff_x_ratio, 'ratio')
            text_regions_p = text_regions_p.astype('uint8')
            
            if len(contours_only_text_parent_excluded)>0:
                textregion_par = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1])).astype('uint8')
                textregion_par = cv2.fillPoly(textregion_par, pts=contours_only_text_parent_included, color=(1,1))
            else:
                textregion_par = (text_regions_p[:,:]==1)*1
                textregion_par = textregion_par.astype('uint8')
                
            text_regions_p_textregions_dilated = cv2.erode(textregion_par , hor_kernel, iterations=2)
            text_regions_p_textregions_dilated = cv2.dilate(text_regions_p_textregions_dilated , ver_kernel, iterations=4)
            text_regions_p_textregions_dilated = cv2.erode(text_regions_p_textregions_dilated , hor_kernel, iterations=1)
            text_regions_p_textregions_dilated = cv2.dilate(text_regions_p_textregions_dilated , ver_kernel, iterations=5)
            text_regions_p_textregions_dilated[text_regions_p[:,:]>1] = 0
            
            
            contours_only_dilated, hir_on_text_dilated = return_contours_of_image(text_regions_p_textregions_dilated)
            contours_only_dilated = return_parent_contours(contours_only_dilated, hir_on_text_dilated)
            
            indexes_of_located_cont, center_x_coordinates_of_located, center_y_coordinates_of_located = self.return_indexes_of_contours_loctaed_inside_another_list_of_contours(contours_only_dilated, contours_only_text_parent_included, cx_conts_included, cy_conts_included, args_cont_located_included)
            
            
            if len(args_cont_located_excluded)>0:
                for ind in args_cont_located_excluded:
                    indexes_of_located_cont.append(np.array([ind]))
                    contours_only_dilated.append(contours_only_text_parent[ind])
                    center_y_coordinates_of_located.append(0)
            
            array_list = [np.array([elem]) if isinstance(elem, int) else elem for elem in indexes_of_located_cont]
            flattened_array = np.concatenate([arr.ravel() for arr in array_list])
            #print(len( np.unique(flattened_array)), 'indexes_of_located_cont uniques')
            
            missing_textregions = list( set(np.array(range(len(contours_only_text_parent))) ) - set(np.unique(flattened_array)) )
            #print(missing_textregions, 'missing_textregions')

            for ind in missing_textregions:
                indexes_of_located_cont.append(np.array([ind]))
                contours_only_dilated.append(contours_only_text_parent[ind])
                center_y_coordinates_of_located.append(0)
                
                
            if contours_only_text_parent_h:
                for vi in range(len(contours_only_text_parent_h)):
                    indexes_of_located_cont.append(int(vi+len(contours_only_text_parent)))
                    
            array_list = [np.array([elem]) if isinstance(elem, int) else elem for elem in indexes_of_located_cont]
            flattened_array = np.concatenate([arr.ravel() for arr in array_list])
        
        y_len = text_regions_p.shape[0]
        x_len = text_regions_p.shape[1]

        img_poly = np.zeros((y_len,x_len), dtype='uint8')
        ###img_poly[text_regions_p[:,:]==1] = 1
        ###img_poly[text_regions_p[:,:]==2] = 2
        ###img_poly[text_regions_p[:,:]==3] = 4
        ###img_poly[text_regions_p[:,:]==6] = 5
        
        ##img_poly[text_regions_p[:,:]==1] = 1
        ##img_poly[text_regions_p[:,:]==2] = 2
        ##img_poly[text_regions_p[:,:]==3] = 3
        ##img_poly[text_regions_p[:,:]==4] = 4
        ##img_poly[text_regions_p[:,:]==5] = 5
        
        img_poly = np.copy(text_regions_p)
        
        img_header_and_sep = np.zeros((y_len,x_len), dtype='uint8')
        if contours_only_text_parent_h:
            _, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, _ = find_new_features_of_contours(
                contours_only_text_parent_h)
            for j in range(len(cy_main)):
                img_header_and_sep[int(y_max_main[j]):int(y_max_main[j])+12,
                                   int(x_min_main[j]):int(x_max_main[j])] = 1
            co_text_all_org = contours_only_text_parent + contours_only_text_parent_h
            if len(contours_only_text_parent)>min_cont_size_to_be_dilated:
                co_text_all = contours_only_dilated + contours_only_text_parent_h
            else:
                co_text_all = contours_only_text_parent + contours_only_text_parent_h
        else:
            co_text_all_org = contours_only_text_parent
            if len(contours_only_text_parent)>min_cont_size_to_be_dilated:
                co_text_all = contours_only_dilated
            else:
                co_text_all = contours_only_text_parent

        if not len(co_text_all):
            return [], []

        labels_con = np.zeros((int(y_len /6.), int(x_len/6.), len(co_text_all)), dtype=bool)
        
        co_text_all = [(i/6).astype(int) for i in co_text_all]
        for i in range(len(co_text_all)):
            img = labels_con[:,:,i].astype(np.uint8)
            
            #img = cv2.resize(img, (int(img.shape[1]/6), int(img.shape[0]/6)), interpolation=cv2.INTER_NEAREST)
            
            cv2.fillPoly(img, pts=[co_text_all[i]], color=(1,))
            labels_con[:,:,i] = img


        labels_con = resize_image(labels_con.astype(np.uint8), height1, width1).astype(bool)
        img_header_and_sep = resize_image(img_header_and_sep, height1, width1)
        img_poly = resize_image(img_poly, height3, width3)
        

        
        input_1 = np.zeros((inference_bs, height1, width1, 3))
        ordered = [list(range(len(co_text_all)))]
        index_update = 0
        #print(labels_con.shape[2],"number of regions for reading order")
        while index_update>=0:
            ij_list = ordered.pop(index_update)
            i = ij_list.pop(0)

            ante_list = []
            post_list = []
            tot_counter = 0
            batch = []
            for j in ij_list:
                img1 = labels_con[:,:,i].astype(float)
                img2 = labels_con[:,:,j].astype(float)
                img1[img_poly==5] = 2
                img2[img_poly==5] = 2
                img1[img_header_and_sep==1] = 3
                img2[img_header_and_sep==1] = 3

                input_1[len(batch), :, :, 0] = img1 / 3.
                input_1[len(batch), :, :, 2] = img2 / 3.
                input_1[len(batch), :, :, 1] = img_poly / 5.

                tot_counter += 1
                batch.append(j)
                if tot_counter % inference_bs == 0 or tot_counter == len(ij_list):
                    y_pr = self.model_zoo.get('reading_order', Model).predict(input_1 , verbose='0')
                    for jb, j in enumerate(batch):
                        if y_pr[jb][0]>=0.5:
                            post_list.append(j)
                        else:
                            ante_list.append(j)
                    batch = []

            if len(ante_list):
                ordered.insert(index_update, ante_list)
                index_update += 1
            ordered.insert(index_update, [i])
            if len(post_list):
                ordered.insert(index_update + 1, post_list)

            index_update = -1
            for index_next, ij_list in enumerate(ordered):
                if len(ij_list) > 1:
                    index_update = index_next
                    break

        ordered = [i[0] for i in ordered]
        
        ##id_all_text = np.array(id_all_text)[index_sort]
        
        
        if len(contours_only_text_parent)>min_cont_size_to_be_dilated:
            org_contours_indexes = []
            for ind in range(len(ordered)):
                region_with_curr_order = ordered[ind]
                if region_with_curr_order < len(contours_only_dilated):
                    if np.isscalar(indexes_of_located_cont[region_with_curr_order]):
                        org_contours_indexes = org_contours_indexes + [indexes_of_located_cont[region_with_curr_order]]
                    else:
                        arg_sort_located_cont = np.argsort(center_y_coordinates_of_located[region_with_curr_order])
                        org_contours_indexes = org_contours_indexes + list(np.array(indexes_of_located_cont[region_with_curr_order])[arg_sort_located_cont]) ##org_contours_indexes + list ( 
                else:
                    org_contours_indexes = org_contours_indexes + [indexes_of_located_cont[region_with_curr_order]]
            
            region_ids = ['region_%04d' % i for i in range(len(co_text_all_org))]
            return org_contours_indexes, region_ids
        else:
            region_ids = ['region_%04d' % i for i in range(len(co_text_all_org))]
            return ordered, region_ids
    

        
        
    def run(self,
            overwrite: bool = False,
            xml_filename: Optional[str] = None,
            dir_in: Optional[str] = None,
            dir_out: Optional[str] = None,
    ):
        """
        Get image and scales, then extract the page of scanned image
        """
        self.logger.debug("enter run")
        t0_tot = time.time()

        if dir_in:
            ls_xmls  = [os.path.join(dir_in, xml_filename)
                        for xml_filename in filter(is_xml_filename,
                                                   os.listdir(dir_in))]
        elif xml_filename:
            ls_xmls = [xml_filename]
        else:
            raise ValueError("run requires either a single image filename or a directory")

        for xml_filename in ls_xmls:
            self.logger.info(xml_filename)
            t0 = time.time()

            file_name = Path(xml_filename).stem
            (tree_xml, root_xml, bb_coord_printspace, id_paragraph, id_header,
             co_text_paragraph, co_text_header, tot_region_ref,
             x_len, y_len, index_tot_regions, img_poly) = self.read_xml(xml_filename)
            
            id_all_text = id_paragraph + id_header
            
            order_text_new, id_of_texts_tot = self.do_order_of_regions_with_model(co_text_paragraph, co_text_header, img_poly[:,:,0])
            
            id_all_text = np.array(id_all_text)[order_text_new]
            
            alltags=[elem.tag for elem in root_xml.iter()]
            
            
            
            link=alltags[0].split('}')[0]+'}'
            name_space = alltags[0].split('}')[0]
            name_space = name_space.split('{')[1]
            
            page_element = root_xml.find(link+'Page')
            
            
            old_ro = root_xml.find(".//{*}ReadingOrder")
            
            if old_ro is not None:
                page_element.remove(old_ro)
            
            #print(old_ro, 'old_ro')
            ro_subelement = ET.Element('ReadingOrder')
            
            ro_subelement2 = ET.SubElement(ro_subelement, 'OrderedGroup')
            ro_subelement2.set('id', "ro357564684568544579089")
            
            for index, id_text in enumerate(id_all_text):
                new_element_2 = ET.SubElement(ro_subelement2, 'RegionRefIndexed')
                new_element_2.set('regionRef', id_all_text[index])
                new_element_2.set('index', str(index))
            
            if (link+'PrintSpace' in alltags) or  (link+'Border' in alltags):
                page_element.insert(1, ro_subelement)
            else:
                page_element.insert(0, ro_subelement)
            
            alltags=[elem.tag for elem in root_xml.iter()]
            
            ET.register_namespace("",name_space)
            assert dir_out
            tree_xml.write(os.path.join(dir_out, file_name+'.xml'),
                           xml_declaration=True,
                           method='xml',
                           encoding="utf-8",
                           default_namespace=None)
            
            #sys.exit()
            
