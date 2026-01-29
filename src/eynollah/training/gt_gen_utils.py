import os
import numpy as np
import warnings
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2
from shapely import geometry
from pathlib import Path
from PIL import ImageFont


KERNEL = np.ones((5, 5), np.uint8)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    
def visualize_image_from_contours_layout(co_par, co_header, co_drop, co_sep, co_image, co_marginal, co_table, co_map, img):
    alpha = 0.5
    
    blank_image = np.ones( (img.shape[:]), dtype=np.uint8) * 255
    
    col_header = (173, 216, 230)
    col_drop = (0, 191, 255)
    boundary_color = (143, 216, 200)#(0, 0, 255)  # Dark gray for the boundary
    col_par = (0, 0, 139)   # Lighter gray for the filled area
    col_image = (0, 100, 0)
    col_sep = (255, 0, 0)
    col_marginal =  (106, 90, 205)
    col_table =  (0, 90, 205)
    col_map =  (90, 90, 205)
    
    if len(co_image)>0:
        cv2.drawContours(blank_image, co_image, -1, col_image, thickness=cv2.FILLED)  # Fill the contour
        
    if len(co_sep)>0:
        cv2.drawContours(blank_image, co_sep, -1, col_sep, thickness=cv2.FILLED)  # Fill the contour
        
        
    if len(co_header)>0:
        cv2.drawContours(blank_image, co_header, -1, col_header, thickness=cv2.FILLED)  # Fill the contour
        
    if len(co_par)>0:
        cv2.drawContours(blank_image, co_par, -1, col_par, thickness=cv2.FILLED)  # Fill the contour
        
        cv2.drawContours(blank_image, co_par, -1, boundary_color, thickness=1)       # Draw the boundary
        
    if len(co_drop)>0:
        cv2.drawContours(blank_image, co_drop, -1, col_drop, thickness=cv2.FILLED)  # Fill the contour
        
    if len(co_marginal)>0:
        cv2.drawContours(blank_image, co_marginal, -1, col_marginal, thickness=cv2.FILLED)  # Fill the contour
        
    if len(co_table)>0:
        cv2.drawContours(blank_image, co_table, -1, col_table, thickness=cv2.FILLED)  # Fill the contour
        
    if len(co_map)>0:
        cv2.drawContours(blank_image, co_map, -1, col_map, thickness=cv2.FILLED)  # Fill the contour
    
    img_final =cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
    
    added_image = cv2.addWeighted(img,alpha,img_final,1- alpha,0)
    return added_image


def visualize_image_from_contours(contours, img):
    alpha = 0.5
    
    blank_image = np.ones( (img.shape[:]), dtype=np.uint8) * 255
    
    boundary_color = (0, 0, 255)  # Dark gray for the boundary
    fill_color = (173, 216, 230)   # Lighter gray for the filled area
    
    cv2.drawContours(blank_image, contours, -1, fill_color, thickness=cv2.FILLED)  # Fill the contour
    cv2.drawContours(blank_image, contours, -1, boundary_color, thickness=1)       # Draw the boundary
    
    img_final =cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
    
    added_image = cv2.addWeighted(img,alpha,img_final,1- alpha,0)
    return added_image

def visualize_model_output(prediction, img, task):
    if task == "binarization":
        prediction = prediction * -1
        prediction = prediction + 1
        added_image = prediction * 255
        layout_only = None
    else:
        unique_classes = np.unique(prediction[:,:,0])
        rgb_colors = {'0' : [255, 255, 255],
                    '1' : [255, 0, 0],
                    '2' : [255, 125, 0],
                    '3' : [255, 0, 125],
                    '4' : [125, 125, 125],
                    '5' : [125, 125, 0],
                    '6' : [0, 125, 255],
                    '7' : [0, 125, 0],
                    '8' : [125, 125, 125],
                    '9' : [0, 125, 255],
                    '10' : [125, 0, 125],
                    '11' : [0, 255, 0],
                    '12' : [0, 0, 255],
                    '13' : [0, 255, 255],
                    '14' : [255, 125, 125],
                    '15' : [255, 0, 255]}
    
        layout_only = np.zeros(prediction.shape)
    
        for unq_class in unique_classes:
            rgb_class_unique = rgb_colors[str(int(unq_class))]
            layout_only[:,:,0][prediction[:,:,0]==unq_class] = rgb_class_unique[0]
            layout_only[:,:,1][prediction[:,:,0]==unq_class] = rgb_class_unique[1]
            layout_only[:,:,2][prediction[:,:,0]==unq_class] = rgb_class_unique[2]
    
    
    
        img = resize_image(img, layout_only.shape[0], layout_only.shape[1])
    
        layout_only = layout_only.astype(np.int32)
        img = img.astype(np.int32)
    
        
        
        added_image = cv2.addWeighted(img,0.5,layout_only,0.1,0)
        
    return added_image, layout_only
    
def get_content_of_dir(dir_in):
    """
    Listing all ground truth page xml files. All files are needed to have xml format.
    """

    gt_all=os.listdir(dir_in)
    gt_list = [file for file in gt_all if os.path.splitext(file)[1] == '.xml']
    return gt_list
    
def return_parent_contours(contours, hierarchy):
    contours_parent = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]
    return contours_parent
def filter_contours_area_of_image_tables(image, contours, hierarchy, max_area, min_area):
    found_polygons_early = list()

    jv = 0
    for c in contours:
        if len(np.shape(c)) == 3:
            c = c[0]
        elif len(np.shape(c)) == 2:
            pass
        #c = c[0]
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue

        c_e = [point for point in c]
        polygon = geometry.Polygon(c_e)
        # area = cv2.contourArea(c)
        area = polygon.area
        # Check that polygon has area greater than minimal area
        if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]):  # and hierarchy[0][jv][3]==-1 :
            found_polygons_early.append(np.array([[point] for point in polygon.exterior.coords], dtype=np.int32))
        jv += 1
    return found_polygons_early

def filter_contours_area_of_image(image, contours, order_index, max_area, min_area, min_early=None):
    found_polygons_early = list()
    order_index_filtered = list()
    regions_ar_less_than_early_min = list()
    #jv = 0
    for jv, c in enumerate(contours):
        if len(np.shape(c)) == 3:
            c = c[0]
        elif len(np.shape(c)) == 2:
            pass
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue
        c_e = [point for point in c]
        polygon = geometry.Polygon(c_e)
        area = polygon.area
        if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]):  # and hierarchy[0][jv][3]==-1 :
            found_polygons_early.append(np.array([[point] for point in polygon.exterior.coords], dtype=np.uint))
            order_index_filtered.append(order_index[jv])
            if min_early:
                if area < min_early * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]):  # and hierarchy[0][jv][3]==-1 :
                    regions_ar_less_than_early_min.append(1)
                else:
                    regions_ar_less_than_early_min.append(0)
            else:
                regions_ar_less_than_early_min = None
                
        #jv += 1
    return found_polygons_early, order_index_filtered, regions_ar_less_than_early_min

def return_contours_of_interested_region(region_pre_p, pixel, min_area=0.0002):

    # pixels of images are identified by 5
    if len(region_pre_p.shape) == 3:
        cnts_images = (region_pre_p[:, :, 0] == pixel) * 1
    else:
        cnts_images = (region_pre_p[:, :] == pixel) * 1
    cnts_images = cnts_images.astype(np.uint8)
    cnts_images = np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
    imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    contours_imgs, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #print(len(contours_imgs), hierarchy)

    contours_imgs = return_parent_contours(contours_imgs, hierarchy)
    
    #print(len(contours_imgs), "iki")
    #contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hierarchy, max_area=1, min_area=min_area)

    return contours_imgs
def update_region_contours(co_text, img_boundary, erosion_rate, dilation_rate, y_len, x_len, dilation_early=None, erosion_early=None):
    co_text_eroded = []
    for con in co_text:
        img_boundary_in = np.zeros( (y_len,x_len) )
        img_boundary_in = cv2.fillPoly(img_boundary_in, pts=[con], color=(1, 1, 1))
        
        if dilation_early:
            img_boundary_in = cv2.dilate(img_boundary_in[:,:], KERNEL, iterations=dilation_early)
            
        if erosion_early:
            img_boundary_in = cv2.erode(img_boundary_in[:,:], KERNEL, iterations=erosion_early)
        
        #img_boundary_in = cv2.erode(img_boundary_in[:,:], KERNEL, iterations=7)#asiatica
        if erosion_rate > 0:
            img_boundary_in = cv2.erode(img_boundary_in[:,:], KERNEL, iterations=erosion_rate)
        
        pixel = 1
        min_size = 0
        
        img_boundary_in =  img_boundary_in.astype("uint8")
        
        con_eroded = return_contours_of_interested_region(img_boundary_in,pixel, min_size )
        
        try:
            if len(con_eroded)>1:
                cnt_size = np.array([cv2.contourArea(con_eroded[j]) for j in range(len(con_eroded))])
                cnt = contours[np.argmax(cnt_size)]
                co_text_eroded.append(cnt)
            else:
                co_text_eroded.append(con_eroded[0])
        except:
            co_text_eroded.append(con)
        

        img_boundary_in_dilated = cv2.dilate(img_boundary_in[:,:], KERNEL, iterations=dilation_rate)
        #img_boundary_in_dilated = cv2.dilate(img_boundary_in[:,:], KERNEL, iterations=5)
        
        boundary = img_boundary_in_dilated[:,:] - img_boundary_in[:,:]
        
        img_boundary[:,:][boundary[:,:]==1] =1
    return co_text_eroded, img_boundary

def get_textline_contours_for_visualization(xml_file):
    tree1 = ET.parse(xml_file, parser = ET.XMLParser(encoding='utf-8'))
    root1=tree1.getroot()
    alltags=[elem.tag for elem in root1.iter()]
    link=alltags[0].split('}')[0]+'}'
                            
        
                            
    x_len, y_len = 0, 0
    for jj in root1.iter(link+'Page'):
        y_len=int(jj.attrib['imageHeight'])
        x_len=int(jj.attrib['imageWidth'])
        
    region_tags = np.unique([x for x in alltags if x.endswith('TextLine')])
    tag_endings = ['}TextLine','}textline']
    co_use_case = []

    for tag in region_tags:
        if tag.endswith(tag_endings[0]) or tag.endswith(tag_endings[1]):
            for nn in root1.iter(tag):
                c_t_in = []
                sumi = 0
                for vv in nn.iter():
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
                co_use_case.append(np.array(c_t_in))
    return co_use_case, y_len, x_len


def get_textline_contours_and_ocr_text(xml_file):
    tree1 = ET.parse(xml_file, parser = ET.XMLParser(encoding='utf-8'))
    root1=tree1.getroot()
    alltags=[elem.tag for elem in root1.iter()]
    link=alltags[0].split('}')[0]+'}'
                            
        
                            
    x_len, y_len = 0, 0
    for jj in root1.iter(link+'Page'):
        y_len=int(jj.attrib['imageHeight'])
        x_len=int(jj.attrib['imageWidth'])
        
    region_tags = np.unique([x for x in alltags if x.endswith('TextLine')])
    tag_endings = ['}TextLine','}textline']
    co_use_case = []
    ocr_textlines = []

    for tag in region_tags:
        if tag.endswith(tag_endings[0]) or tag.endswith(tag_endings[1]):
            for nn in root1.iter(tag):
                c_t_in = []
                ocr_text_in = ['']
                sumi = 0
                for vv in nn.iter():
                    if vv.tag == link + 'Coords':
                        for childtest2 in nn:
                            if childtest2.tag.endswith("TextEquiv"):
                                for child_uc in childtest2:
                                    if child_uc.tag.endswith("Unicode"):
                                        text = child_uc.text
                                        ocr_text_in[0]= text
                            
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
                
                        
                co_use_case.append(np.array(c_t_in))
                ocr_textlines.append(ocr_text_in[0])
    return co_use_case, y_len, x_len, ocr_textlines

def fit_text_single_line(draw, text, font_path, max_width, max_height):
    initial_font_size = 50
    font_size = initial_font_size
    while font_size > 10:  # Minimum font size
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)  # Get text bounding box
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        if text_width <= max_width and text_height <= max_height:
            return font  # Return the best-fitting font

        font_size -= 2  # Reduce font size and retry

    return ImageFont.truetype(font_path, 10)  # Smallest font fallback

def get_layout_contours_for_visualization(xml_file):
    tree1 = ET.parse(xml_file, parser = ET.XMLParser(encoding='utf-8'))
    root1=tree1.getroot()
    alltags=[elem.tag for elem in root1.iter()]
    link=alltags[0].split('}')[0]+'}'
                            
        
    x_len, y_len = 0, 0 
    for jj in root1.iter(link+'Page'):
        y_len=int(jj.attrib['imageHeight'])
        x_len=int(jj.attrib['imageWidth'])
        
    region_tags=np.unique([x for x in alltags if x.endswith('Region')])   
    co_text = {'drop-capital':[], "footnote":[], "footnote-continued":[], "heading":[], "signature-mark":[], "header":[], "catch-word":[], "page-number":[], "marginalia":[], "paragraph":[]}
    all_defined_textregion_types = list(co_text.keys())
    co_graphic = {"handwritten-annotation":[], "decoration":[], "stamp":[], "signature":[]}
    all_defined_graphic_types = list(co_graphic.keys())
    co_sep=[]
    co_img=[]
    co_table=[]
    co_map=[]
    co_noise=[]
    
    types_text = []
    types_graphic = []
    
    for tag in region_tags:
        if tag.endswith('}TextRegion') or tag.endswith('}Textregion'):
            for nn in root1.iter(tag):
                c_t_in = {'drop-capital':[], "footnote":[], "footnote-continued":[], "heading":[], "signature-mark":[], "header":[], "catch-word":[], "page-number":[], "marginalia":[], "paragraph":[]}
                sumi=0
                for vv in nn.iter():
                    # check the format of coords
                    if vv.tag==link+'Coords':
    
                        coords=bool(vv.attrib)
                        if coords:
                            p_h=vv.attrib['points'].split(' ')
                            
                            if "rest_as_paragraph" in types_text:
                                types_text_without_paragraph = [element for element in types_text if element!='rest_as_paragraph' and element!='paragraph']
                                if len(types_text_without_paragraph) == 0:
                                    if "type" in nn.attrib:
                                        c_t_in['paragraph'].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                    else:
                                        c_t_in['paragraph'].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                        
                                elif len(types_text_without_paragraph) >= 1:
                                    if "type" in nn.attrib:
                                        if nn.attrib['type'] in types_text_without_paragraph:
                                            c_t_in[nn.attrib['type']].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                        else:
                                            c_t_in['paragraph'].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                            
                                    else:
                                        c_t_in['paragraph'].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                            
                            else:
                                if "type" in nn.attrib:
                                    if nn.attrib['type'] in all_defined_textregion_types:
                                        c_t_in[nn.attrib['type']].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                else:
                                    c_t_in['paragraph'].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
    
                            break
                        else:
                            pass
    
                    
                    if vv.tag==link+'Point':
                        if "rest_as_paragraph" in types_text:
                            types_text_without_paragraph = [element for element in types_text if element!='rest_as_paragraph' and element!='paragraph']
                            if len(types_text_without_paragraph) == 0:
                                if "type" in nn.attrib:
                                    c_t_in['paragraph'].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                    sumi+=1
                            elif len(types_text_without_paragraph) >= 1:
                                if "type" in nn.attrib:
                                    if nn.attrib['type'] in types_text_without_paragraph:
                                        c_t_in[nn.attrib['type']].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                        sumi+=1
                                    else:
                                        c_t_in['paragraph'].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                        sumi+=1
                                        
                        else:
                            if "type" in nn.attrib:
                                if nn.attrib['type'] in all_defined_textregion_types:
                                    c_t_in[nn.attrib['type']].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                    sumi+=1


                    elif vv.tag!=link+'Point' and sumi>=1:
                        break
    
                for element_text in list(c_t_in.keys()):
                    if len(c_t_in[element_text])>0:
                        co_text[element_text].append(np.array(c_t_in[element_text]))
                            
                            
        if tag.endswith('}GraphicRegion') or tag.endswith('}graphicregion'):
            #print('sth')
            for nn in root1.iter(tag):
                c_t_in_graphic = {"handwritten-annotation":[], "decoration":[], "stamp":[], "signature":[]}
                sumi=0
                for vv in nn.iter():
                    # check the format of coords
                    if vv.tag==link+'Coords':
                        coords=bool(vv.attrib)
                        if coords:
                            p_h=vv.attrib['points'].split(' ')
                            
                            if "rest_as_decoration" in types_graphic:
                                types_graphic_without_decoration = [element for element in types_graphic if element!='rest_as_decoration' and element!='decoration']
                                if len(types_graphic_without_decoration) == 0:
                                    if "type" in nn.attrib:
                                        c_t_in_graphic['decoration'].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                elif len(types_graphic_without_decoration) >= 1:
                                    if "type" in nn.attrib:
                                        if nn.attrib['type'] in types_graphic_without_decoration:
                                            c_t_in_graphic[nn.attrib['type']].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                        else:
                                            c_t_in_graphic['decoration'].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                            
                            else:
                                if "type" in nn.attrib:
                                    if nn.attrib['type'] in all_defined_graphic_types:
                                        c_t_in_graphic[nn.attrib['type']].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )        
                            
                            break
                        else:
                            pass
    
    
                    if vv.tag==link+'Point':
                        if "rest_as_decoration" in types_graphic:
                            types_graphic_without_decoration = [element for element in types_graphic if element!='rest_as_decoration' and element!='decoration']
                            if len(types_graphic_without_decoration) == 0:
                                if "type" in nn.attrib:
                                    c_t_in_graphic['decoration'].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                    sumi+=1
                            elif len(types_graphic_without_decoration) >= 1:
                                if "type" in nn.attrib:
                                    if nn.attrib['type'] in types_graphic_without_decoration:
                                        c_t_in_graphic[nn.attrib['type']].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                        sumi+=1
                                    else:
                                        c_t_in_graphic['decoration'].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                        sumi+=1
                                        
                        else:
                            if "type" in nn.attrib:
                                if nn.attrib['type'] in all_defined_graphic_types:
                                    c_t_in_graphic[nn.attrib['type']].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] ) 
                                    sumi+=1
                                
                    elif vv.tag!=link+'Point' and sumi>=1:
                        break
                    
                for element_graphic in list(c_t_in_graphic.keys()):
                    if len(c_t_in_graphic[element_graphic])>0:
                        co_graphic[element_graphic].append(np.array(c_t_in_graphic[element_graphic]))
                            
    
        if tag.endswith('}ImageRegion') or tag.endswith('}imageregion'):
            for nn in root1.iter(tag):
                c_t_in=[]
                sumi=0
                for vv in nn.iter():
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
    
        
        if tag.endswith('}SeparatorRegion') or tag.endswith('}separatorregion'):
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


        if tag.endswith('}TableRegion') or tag.endswith('}tableregion'):
            #print('sth')
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
                    #print(vv.tag,'in')
                    elif vv.tag!=link+'Point' and sumi>=1:
                        break
                co_table.append(np.array(c_t_in))
                
        if tag.endswith('}MapRegion') or tag.endswith('}mapregion'):
            #print('sth')
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
                    #print(vv.tag,'in')
                    elif vv.tag!=link+'Point' and sumi>=1:
                        break
                co_map.append(np.array(c_t_in))
    

        if tag.endswith('}NoiseRegion') or tag.endswith('}noiseregion'):
            #print('sth')
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
                    #print(vv.tag,'in')
                    elif vv.tag!=link+'Point' and sumi>=1:
                        break
                co_noise.append(np.array(c_t_in))
    return co_text, co_graphic, co_sep, co_img, co_table, co_map, co_noise, y_len, x_len
    
def get_images_of_ground_truth(gt_list, dir_in, output_dir, output_type, config_file, config_params, printspace, dir_images, dir_out_images):
    """
    Reading the page xml files and write the ground truth images into given output directory.
    """
    ## to do: add footnote to text regions
    
    if dir_images:
        ls_org_imgs = os.listdir(dir_images)
        ls_org_imgs_stem = [os.path.splitext(item)[0] for item in ls_org_imgs]
    for index in tqdm(range(len(gt_list))):
        #try:
        print(gt_list[index])
        tree1 = ET.parse(dir_in+'/'+gt_list[index], parser = ET.XMLParser(encoding='utf-8'))
        root1=tree1.getroot()
        alltags=[elem.tag for elem in root1.iter()]
        link=alltags[0].split('}')[0]+'}'
                            
        
        x_len, y_len = 0, 0
        for jj in root1.iter(link+'Page'):
            y_len=int(jj.attrib['imageHeight'])
            x_len=int(jj.attrib['imageWidth'])
            
        if 'columns_width' in list(config_params.keys()):
            columns_width_dict = config_params['columns_width']
            metadata_element = root1.find(link+'Metadata')
            num_col = None
            for child in metadata_element:
                tag2 = child.tag
                if tag2.endswith('}Comments') or tag2.endswith('}comments'):
                    text_comments = child.text
                    num_col = int(text_comments.split('num_col')[1])
                
            if num_col:
                x_new = columns_width_dict[str(num_col)]
                y_new = int ( x_new * (y_len / float(x_len)) )
            
        if printspace or "printspace_as_class_in_layout" in list(config_params.keys()):
            region_tags = np.unique([x for x in alltags if x.endswith('PrintSpace') or x.endswith('Border')])
            co_use_case = []

            for tag in region_tags:
                tag_endings = ['}PrintSpace','}Border']
                    
                if tag.endswith(tag_endings[0]) or tag.endswith(tag_endings[1]):
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
                        co_use_case.append(np.array(c_t_in))
                        
            img = np.zeros((y_len, x_len, 3))
            
            img_poly = cv2.fillPoly(img, pts=co_use_case, color=(1, 1, 1))
            
            img_poly = img_poly.astype(np.uint8)
            
            imgray = cv2.cvtColor(img_poly, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(imgray, 0, 255, 0)

            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            cnt_size = np.array([cv2.contourArea(contours[j]) for j in range(len(contours))])
            
            try:
                cnt = contours[np.argmax(cnt_size)]
                x, y, w, h = cv2.boundingRect(cnt)
            except:
                x, y , w, h = 0, 0, x_len, y_len
            
            bb_xywh = [x, y, w, h]
            
            
        if config_file and (config_params['use_case']=='textline' or config_params['use_case']=='word' or config_params['use_case']=='glyph' or config_params['use_case']=='printspace'):
            keys = list(config_params.keys())
            if "artificial_class_label" in keys:
                artificial_class_rgb_color = (255,255,0)
                artificial_class_label = config_params['artificial_class_label']
                
            textline_rgb_color = (255, 0, 0)
                
            if config_params['use_case']=='textline':
                region_tags = np.unique([x for x in alltags if x.endswith('TextLine')])
            elif config_params['use_case']=='word':
                region_tags = np.unique([x for x in alltags if x.endswith('Word')])
            elif config_params['use_case']=='glyph':
                region_tags = np.unique([x for x in alltags if x.endswith('Glyph')])
            elif config_params['use_case']=='printspace':
                region_tags = np.unique([x for x in alltags if x.endswith('PrintSpace')])
                
            co_use_case = []

            for tag in region_tags:
                if config_params['use_case']=='textline':
                    tag_endings = ['}TextLine','}textline']
                elif config_params['use_case']=='word':
                    tag_endings = ['}Word','}word']
                elif config_params['use_case']=='glyph':
                    tag_endings = ['}Glyph','}glyph']
                elif config_params['use_case']=='printspace':
                    tag_endings = ['}PrintSpace','}printspace']
                    
                if tag.endswith(tag_endings[0]) or tag.endswith(tag_endings[1]):
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
                        co_use_case.append(np.array(c_t_in))
                        
                        
            if "artificial_class_label" in keys:
                img_boundary = np.zeros((y_len, x_len))
                erosion_rate = 0#1
                dilation_rate = 2
                dilation_early = 0
                erosion_early = 2
                co_use_case, img_boundary = update_region_contours(co_use_case, img_boundary, erosion_rate, dilation_rate, y_len, x_len, dilation_early=dilation_early, erosion_early=erosion_early)
            
                
            img = np.zeros((y_len, x_len, 3))
            if output_type == '2d':
                img_poly = cv2.fillPoly(img, pts=co_use_case, color=(1, 1, 1))
                if "artificial_class_label" in keys:
                    img_mask = np.copy(img_poly)
                    ##img_poly[:,:][(img_boundary[:,:]==1) & (img_mask[:,:,0]!=1)] = artificial_class_label
                    img_poly[:,:][img_boundary[:,:]==1] = artificial_class_label
            elif output_type == '3d':
                img_poly = cv2.fillPoly(img, pts=co_use_case, color=textline_rgb_color)
                if "artificial_class_label" in keys:
                    img_mask = np.copy(img_poly)
                    img_poly[:,:,0][(img_boundary[:,:]==1) & (img_mask[:,:,0]!=255)] = artificial_class_rgb_color[0]
                    img_poly[:,:,1][(img_boundary[:,:]==1) & (img_mask[:,:,0]!=255)] = artificial_class_rgb_color[1]
                    img_poly[:,:,2][(img_boundary[:,:]==1) & (img_mask[:,:,0]!=255)] = artificial_class_rgb_color[2]
                    
                    
            if printspace and config_params['use_case']!='printspace':
                img_poly = img_poly[bb_xywh[1]:bb_xywh[1]+bb_xywh[3], bb_xywh[0]:bb_xywh[0]+bb_xywh[2], :]
                
                
            if 'columns_width' in list(config_params.keys()) and num_col and config_params['use_case']!='printspace':
                img_poly = resize_image(img_poly, y_new, x_new)

            try:
                xml_file_stem = os.path.splitext(gt_list[index])[0]
                cv2.imwrite(os.path.join(output_dir, xml_file_stem + '.png'), img_poly)
            except:
                xml_file_stem = os.path.splitext(gt_list[index])[0]
                cv2.imwrite(os.path.join(output_dir, xml_file_stem + '.png'), img_poly)
                
            if dir_images:
                org_image_name = ls_org_imgs[ls_org_imgs_stem.index(xml_file_stem)]
                img_org = cv2.imread(os.path.join(dir_images, org_image_name))
                
                if printspace and config_params['use_case']!='printspace':
                    img_org = img_org[bb_xywh[1]:bb_xywh[1]+bb_xywh[3], bb_xywh[0]:bb_xywh[0]+bb_xywh[2], :]
                    
                if 'columns_width' in list(config_params.keys()) and num_col and config_params['use_case']!='printspace':
                    img_org = resize_image(img_org, y_new, x_new)
                    
                cv2.imwrite(os.path.join(dir_out_images, org_image_name), img_org)

            
        if config_file and config_params['use_case']=='layout':
            keys = list(config_params.keys())
            
            if "artificial_class_on_boundary" in keys:
                elements_with_artificial_class = list(config_params['artificial_class_on_boundary'])
                artificial_class_rgb_color = (255,255,0)
                artificial_class_label = config_params['artificial_class_label']
            #values = config_params.values()
            
            if "printspace_as_class_in_layout" in list(config_params.keys()):
                printspace_class_rgb_color = (125,125,255)
                printspace_class_label = config_params['printspace_as_class_in_layout']

            if 'textregions' in keys:
                types_text_dict = config_params['textregions']
                types_text = list(types_text_dict.keys())
                types_text_label = list(types_text_dict.values())
            if 'graphicregions' in keys:
                types_graphic_dict = config_params['graphicregions']
                types_graphic = list(types_graphic_dict.keys())
                types_graphic_label = list(types_graphic_dict.values())

                
            labels_rgb_color = [ (0,0,0), (255,0,0), (255,125,0), (255,0,125), (125,255,125), (125,125,0), (0,125,255), (0,125,0), (125,125,125), (255,0,255), (125,0,125), (0,255,0),(0,0,255), (0,255,255), (255,125,125),  (0,125,125), (0,255,125), (255,125,255), (125,255,0), (125,255,255)]
            
            
            region_tags=np.unique([x for x in alltags if x.endswith('Region')])   
            co_text = {'drop-capital':[], "footnote":[], "footnote-continued":[], "heading":[], "signature-mark":[], "header":[], "catch-word":[], "page-number":[], "marginalia":[], "paragraph":[]}
            all_defined_textregion_types = list(co_text.keys())
            co_graphic = {"handwritten-annotation":[], "decoration":[], "stamp":[], "signature":[]}
            all_defined_graphic_types = list(co_graphic.keys())
            co_sep=[]
            co_img=[]
            co_table=[]
            co_map=[]
            co_noise=[]
            
            for tag in region_tags:
                if 'textregions' in keys:
                    if tag.endswith('}TextRegion') or tag.endswith('}Textregion'):
                        for nn in root1.iter(tag):
                            c_t_in = {'drop-capital':[], "footnote":[], "footnote-continued":[], "heading":[], "signature-mark":[], "header":[], "catch-word":[], "page-number":[], "marginalia":[], "paragraph":[]}
                            sumi=0
                            for vv in nn.iter():
                                # check the format of coords
                                if vv.tag==link+'Coords':
                
                                    coords=bool(vv.attrib)
                                    if coords:
                                        p_h=vv.attrib['points'].split(' ')
                                        
                                        if "rest_as_paragraph" in types_text:
                                            types_text_without_paragraph = [element for element in types_text if element!='rest_as_paragraph' and element!='paragraph']
                                            if len(types_text_without_paragraph) == 0:
                                                if "type" in nn.attrib:
                                                    c_t_in['paragraph'].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                            elif len(types_text_without_paragraph) >= 1:
                                                if "type" in nn.attrib:
                                                    if nn.attrib['type'] in types_text_without_paragraph:
                                                        c_t_in[nn.attrib['type']].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                                    else:
                                                        c_t_in['paragraph'].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                                        
                                        else:
                                            if "type" in nn.attrib:
                                                if nn.attrib['type'] in all_defined_textregion_types:
                                                    c_t_in[nn.attrib['type']].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                
                                        break
                                    else:
                                        pass
                
                                
                                if vv.tag==link+'Point':
                                    if "rest_as_paragraph" in types_text:
                                        types_text_without_paragraph = [element for element in types_text if element!='rest_as_paragraph' and element!='paragraph']
                                        if len(types_text_without_paragraph) == 0:
                                            if "type" in nn.attrib:
                                                c_t_in['paragraph'].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                                sumi+=1
                                        elif len(types_text_without_paragraph) >= 1:
                                            if "type" in nn.attrib:
                                                if nn.attrib['type'] in types_text_without_paragraph:
                                                    c_t_in[nn.attrib['type']].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                                    sumi+=1
                                                else:
                                                    c_t_in['paragraph'].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                                    sumi+=1
                                                    
                                    else:
                                        if "type" in nn.attrib:
                                            if nn.attrib['type'] in all_defined_textregion_types:
                                                c_t_in[nn.attrib['type']].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                                sumi+=1


                                elif vv.tag!=link+'Point' and sumi>=1:
                                    break
                
                            for element_text in list(c_t_in.keys()):
                                if len(c_t_in[element_text])>0:
                                    co_text[element_text].append(np.array(c_t_in[element_text]))
                                    
                if 'graphicregions' in keys:
                    if tag.endswith('}GraphicRegion') or tag.endswith('}graphicregion'):
                        #print('sth')
                        for nn in root1.iter(tag):
                            c_t_in_graphic = {"handwritten-annotation":[], "decoration":[], "stamp":[], "signature":[]}
                            sumi=0
                            for vv in nn.iter():
                                # check the format of coords
                                if vv.tag==link+'Coords':
                                    coords=bool(vv.attrib)
                                    if coords:
                                        p_h=vv.attrib['points'].split(' ')
                                        
                                        if "rest_as_decoration" in types_graphic:
                                            types_graphic_without_decoration = [element for element in types_graphic if element!='rest_as_decoration' and element!='decoration']
                                            if len(types_graphic_without_decoration) == 0:
                                                if "type" in nn.attrib:
                                                    c_t_in_graphic['decoration'].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                            elif len(types_graphic_without_decoration) >= 1:
                                                if "type" in nn.attrib:
                                                    if nn.attrib['type'] in types_graphic_without_decoration:
                                                        c_t_in_graphic[nn.attrib['type']].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                                    else:
                                                        c_t_in_graphic['decoration'].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )
                                                        
                                        else:
                                            if "type" in nn.attrib:
                                                if nn.attrib['type'] in all_defined_graphic_types:
                                                    c_t_in_graphic[nn.attrib['type']].append( np.array( [ [ int(x.split(',')[0]) , int(x.split(',')[1]) ]  for x in p_h] ) )        
                                        
                                        break
                                    else:
                                        pass
                
                
                                if vv.tag==link+'Point':
                                    if "rest_as_decoration" in types_graphic:
                                        types_graphic_without_decoration = [element for element in types_graphic if element!='rest_as_decoration' and element!='decoration']
                                        if len(types_graphic_without_decoration) == 0:
                                            if "type" in nn.attrib:
                                                c_t_in_graphic['decoration'].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                                sumi+=1
                                        elif len(types_graphic_without_decoration) >= 1:
                                            if "type" in nn.attrib:
                                                if nn.attrib['type'] in types_graphic_without_decoration:
                                                    c_t_in_graphic[nn.attrib['type']].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                                    sumi+=1
                                                else:
                                                    c_t_in_graphic['decoration'].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] )
                                                    sumi+=1
                                                    
                                    else:
                                        if "type" in nn.attrib:
                                            if nn.attrib['type'] in all_defined_graphic_types:
                                                c_t_in_graphic[nn.attrib['type']].append( [ int(float(vv.attrib['x'])) , int(float(vv.attrib['y'])) ] ) 
                                                sumi+=1
                                            
                                elif vv.tag!=link+'Point' and sumi>=1:
                                    break
                                
                            for element_graphic in list(c_t_in_graphic.keys()):
                                if len(c_t_in_graphic[element_graphic])>0:
                                    co_graphic[element_graphic].append(np.array(c_t_in_graphic[element_graphic]))
                                    
            
                if 'imageregion' in keys:
                    if tag.endswith('}ImageRegion') or tag.endswith('}imageregion'):
                        for nn in root1.iter(tag):
                            c_t_in=[]
                            sumi=0
                            for vv in nn.iter():
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
            
                
                if 'separatorregion' in keys:
                    if tag.endswith('}SeparatorRegion') or tag.endswith('}separatorregion'):
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
            
            
            
                if 'tableregion' in keys:
                    if tag.endswith('}TableRegion') or tag.endswith('}tableregion'):
                        #print('sth')
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
                                #print(vv.tag,'in')
                                elif vv.tag!=link+'Point' and sumi>=1:
                                    break
                            co_table.append(np.array(c_t_in))
                            
                if 'mapregion' in keys:
                    if tag.endswith('}MapRegion') or tag.endswith('}mapregion'):
                        #print('sth')
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
                                #print(vv.tag,'in')
                                elif vv.tag!=link+'Point' and sumi>=1:
                                    break
                            co_map.append(np.array(c_t_in))
            
                if 'noiseregion' in keys:
                    if tag.endswith('}NoiseRegion') or tag.endswith('}noiseregion'):
                        #print('sth')
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
                                #print(vv.tag,'in')
                                elif vv.tag!=link+'Point' and sumi>=1:
                                    break
                            co_noise.append(np.array(c_t_in))
                            
            if "artificial_class_on_boundary" in keys:
                img_boundary = np.zeros( (y_len,x_len) )
                if "paragraph" in elements_with_artificial_class:
                    erosion_rate = 2
                    dilation_rate = 4
                    co_text['paragraph'], img_boundary = update_region_contours(co_text['paragraph'], img_boundary, erosion_rate, dilation_rate, y_len, x_len )
                if "drop-capital" in elements_with_artificial_class:
                    erosion_rate = 1
                    dilation_rate = 3
                    co_text["drop-capital"], img_boundary = update_region_contours(co_text["drop-capital"], img_boundary, erosion_rate, dilation_rate, y_len, x_len )
                if "catch-word" in elements_with_artificial_class:
                    erosion_rate = 0
                    dilation_rate = 3#4
                    co_text["catch-word"], img_boundary = update_region_contours(co_text["catch-word"], img_boundary, erosion_rate, dilation_rate, y_len, x_len )
                if "page-number" in elements_with_artificial_class:
                    erosion_rate = 0
                    dilation_rate = 3#4
                    co_text["page-number"], img_boundary = update_region_contours(co_text["page-number"], img_boundary, erosion_rate, dilation_rate, y_len, x_len )
                if "header" in elements_with_artificial_class:
                    erosion_rate = 1
                    dilation_rate = 4
                    co_text["header"], img_boundary = update_region_contours(co_text["header"], img_boundary, erosion_rate, dilation_rate, y_len, x_len )
                if "heading" in elements_with_artificial_class:
                    erosion_rate = 1
                    dilation_rate = 4
                    co_text["heading"], img_boundary = update_region_contours(co_text["heading"], img_boundary, erosion_rate, dilation_rate, y_len, x_len )
                if "signature-mark" in elements_with_artificial_class:
                    erosion_rate = 1
                    dilation_rate = 4
                    co_text["signature-mark"], img_boundary = update_region_contours(co_text["signature-mark"], img_boundary, erosion_rate, dilation_rate, y_len, x_len )
                if "marginalia" in elements_with_artificial_class:
                    erosion_rate = 2
                    dilation_rate = 4
                    co_text["marginalia"], img_boundary = update_region_contours(co_text["marginalia"], img_boundary, erosion_rate, dilation_rate, y_len, x_len )
                if "footnote" in elements_with_artificial_class:
                    erosion_rate = 0#2
                    dilation_rate = 2#4
                    co_text["footnote"], img_boundary = update_region_contours(co_text["footnote"], img_boundary, erosion_rate, dilation_rate, y_len, x_len )
                if "footnote-continued" in elements_with_artificial_class:
                    erosion_rate = 0#2
                    dilation_rate = 2#4
                    co_text["footnote-continued"], img_boundary = update_region_contours(co_text["footnote-continued"], img_boundary, erosion_rate, dilation_rate, y_len, x_len )
                if "tableregion" in elements_with_artificial_class:
                    erosion_rate = 0#2
                    dilation_rate = 3#4
                    co_table, img_boundary = update_region_contours(co_table, img_boundary, erosion_rate, dilation_rate, y_len, x_len )
                if "mapregion" in elements_with_artificial_class:
                    erosion_rate = 0#2
                    dilation_rate = 3#4
                    co_map, img_boundary = update_region_contours(co_map, img_boundary, erosion_rate, dilation_rate, y_len, x_len )
                    
                    
                
            img = np.zeros( (y_len,x_len,3) )

            if output_type == '3d':
                if 'graphicregions' in keys:
                    if 'rest_as_decoration' in types_graphic:
                        types_graphic[types_graphic=='rest_as_decoration'] = 'decoration'
                        for element_graphic in types_graphic:
                            if element_graphic == 'decoration':
                                color_label = labels_rgb_color[ config_params['graphicregions']['rest_as_decoration']]
                            else:
                                color_label = labels_rgb_color[ config_params['graphicregions'][element_graphic]]
                            img_poly=cv2.fillPoly(img, pts =co_graphic[element_graphic], color=color_label)
                    else:
                        for element_graphic in types_graphic:
                            color_label = labels_rgb_color[ config_params['graphicregions'][element_graphic]]
                            img_poly=cv2.fillPoly(img, pts =co_graphic[element_graphic], color=color_label)
                    
                        
                if 'imageregion' in keys: 
                    img_poly=cv2.fillPoly(img, pts =co_img, color=labels_rgb_color[ config_params['imageregion']])
                if 'tableregion' in keys:  
                    img_poly=cv2.fillPoly(img, pts =co_table, color=labels_rgb_color[ config_params['tableregion']])
                if 'mapregion' in keys:  
                    img_poly=cv2.fillPoly(img, pts =co_map, color=labels_rgb_color[ config_params['mapregion']])
                if 'noiseregion' in keys:  
                    img_poly=cv2.fillPoly(img, pts =co_noise, color=labels_rgb_color[ config_params['noiseregion']])
                    
                if 'textregions' in keys:
                    if 'rest_as_paragraph' in types_text:
                        types_text = ['paragraph'if ttind=='rest_as_paragraph' else ttind for ttind in types_text]
                        for element_text in types_text:
                            if element_text == 'paragraph':
                                color_label = labels_rgb_color[ config_params['textregions']['rest_as_paragraph']]
                            else:
                                color_label = labels_rgb_color[ config_params['textregions'][element_text]]
                            img_poly=cv2.fillPoly(img, pts =co_text[element_text], color=color_label)
                    else:
                        for element_text in types_text:
                            color_label = labels_rgb_color[ config_params['textregions'][element_text]]
                            img_poly=cv2.fillPoly(img, pts =co_text[element_text], color=color_label)
                        
                        
                if "artificial_class_on_boundary" in keys:
                    img_poly[:,:,0][img_boundary[:,:]==1] = artificial_class_rgb_color[0]
                    img_poly[:,:,1][img_boundary[:,:]==1] = artificial_class_rgb_color[1]
                    img_poly[:,:,2][img_boundary[:,:]==1] = artificial_class_rgb_color[2]
                    
                if 'separatorregion' in keys: 
                    img_poly=cv2.fillPoly(img, pts =co_sep, color=labels_rgb_color[ config_params['separatorregion']])
                    
                    
                if "printspace_as_class_in_layout" in list(config_params.keys()):
                    printspace_mask = np.zeros((img_poly.shape[0], img_poly.shape[1]))
                    printspace_mask[bb_xywh[1]:bb_xywh[1]+bb_xywh[3], bb_xywh[0]:bb_xywh[0]+bb_xywh[2]] = 1
                    
                    img_poly[:,:,0][printspace_mask[:,:] == 0] = printspace_class_rgb_color[0]
                    img_poly[:,:,1][printspace_mask[:,:] == 0] = printspace_class_rgb_color[1]
                    img_poly[:,:,2][printspace_mask[:,:] == 0] = printspace_class_rgb_color[2]
                    

                    
                
            elif output_type == '2d':
                if 'graphicregions' in keys:
                    if 'rest_as_decoration' in types_graphic:
                        types_graphic[types_graphic=='rest_as_decoration'] = 'decoration'
                        for element_graphic in types_graphic:
                            if element_graphic == 'decoration':
                                color_label = config_params['graphicregions']['rest_as_decoration']
                            else:
                                color_label = config_params['graphicregions'][element_graphic]
                            img_poly=cv2.fillPoly(img, pts =co_graphic[element_graphic], color=color_label)
                    else:
                        for element_graphic in types_graphic:
                            color_label = config_params['graphicregions'][element_graphic]
                            img_poly=cv2.fillPoly(img, pts =co_graphic[element_graphic], color=color_label)
                            
                
                if 'imageregion' in keys:
                    color_label = config_params['imageregion']
                    img_poly=cv2.fillPoly(img, pts =co_img, color=(color_label,color_label,color_label))
                if 'tableregion' in keys:
                    color_label = config_params['tableregion']
                    img_poly=cv2.fillPoly(img, pts =co_table, color=(color_label,color_label,color_label))
                if 'mapregion' in keys:
                    color_label = config_params['mapregion']
                    img_poly=cv2.fillPoly(img, pts =co_map, color=(color_label,color_label,color_label))
                if 'noiseregion' in keys:
                    color_label = config_params['noiseregion']
                    img_poly=cv2.fillPoly(img, pts =co_noise, color=(color_label,color_label,color_label))
                    
                if 'textregions' in keys:
                    if 'rest_as_paragraph' in types_text:
                        types_text = ['paragraph'if ttind=='rest_as_paragraph' else ttind for ttind in types_text]
                        for element_text in types_text:
                            if element_text == 'paragraph':
                                color_label = config_params['textregions']['rest_as_paragraph']
                            else:
                                color_label = config_params['textregions'][element_text]
                            img_poly=cv2.fillPoly(img, pts =co_text[element_text], color=color_label)
                    else:
                        for element_text in types_text:
                            color_label = config_params['textregions'][element_text]
                            img_poly=cv2.fillPoly(img, pts =co_text[element_text], color=color_label)
                        
                if "artificial_class_on_boundary" in keys:
                    img_poly[:,:][img_boundary[:,:]==1] = artificial_class_label
                    
                if 'separatorregion' in keys: 
                    color_label = config_params['separatorregion']
                    img_poly=cv2.fillPoly(img, pts =co_sep, color=(color_label,color_label,color_label))
                    
                if "printspace_as_class_in_layout" in list(config_params.keys()):
                    printspace_mask = np.zeros((img_poly.shape[0], img_poly.shape[1]))
                    printspace_mask[bb_xywh[1]:bb_xywh[1]+bb_xywh[3], bb_xywh[0]:bb_xywh[0]+bb_xywh[2]] = 1
                    
                    img_poly[:,:,0][printspace_mask[:,:] == 0] = printspace_class_label
                    img_poly[:,:,1][printspace_mask[:,:] == 0] = printspace_class_label
                    img_poly[:,:,2][printspace_mask[:,:] == 0] = printspace_class_label
                
                
                
            if printspace:
                img_poly = img_poly[bb_xywh[1]:bb_xywh[1]+bb_xywh[3], bb_xywh[0]:bb_xywh[0]+bb_xywh[2], :]
                
            if 'columns_width' in list(config_params.keys()) and num_col:
                img_poly = resize_image(img_poly, y_new, x_new)
                
            try:
                xml_file_stem = os.path.splitext(gt_list[index])[0]
                cv2.imwrite(os.path.join(output_dir, xml_file_stem + '.png'), img_poly)
            except:
                xml_file_stem = os.path.splitext(gt_list[index])[0]
                cv2.imwrite(os.path.join(output_dir, xml_file_stem + '.png'), img_poly)
                
                
            if dir_images:
                org_image_name = ls_org_imgs[ls_org_imgs_stem.index(xml_file_stem)]
                img_org = cv2.imread(os.path.join(dir_images, org_image_name))
                
                if printspace:
                    img_org = img_org[bb_xywh[1]:bb_xywh[1]+bb_xywh[3], bb_xywh[0]:bb_xywh[0]+bb_xywh[2], :]
                    
                if 'columns_width' in list(config_params.keys()) and num_col:
                    img_org = resize_image(img_org, y_new, x_new)
                    
                cv2.imwrite(os.path.join(dir_out_images, org_image_name), img_org)
                
                
                
def find_new_features_of_contours(contours_main):

    areas_main = np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
    M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
    cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
    cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
    try:
        x_min_main = np.array([np.min(contours_main[j][0][:, 0]) for j in range(len(contours_main))])

        argmin_x_main = np.array([np.argmin(contours_main[j][0][:, 0]) for j in range(len(contours_main))])

        x_min_from_argmin = np.array([contours_main[j][0][argmin_x_main[j], 0] for j in range(len(contours_main))])
        y_corr_x_min_from_argmin = np.array([contours_main[j][0][argmin_x_main[j], 1] for j in range(len(contours_main))])

        x_max_main = np.array([np.max(contours_main[j][0][:, 0]) for j in range(len(contours_main))])

        y_min_main = np.array([np.min(contours_main[j][0][:, 1]) for j in range(len(contours_main))])
        y_max_main = np.array([np.max(contours_main[j][0][:, 1]) for j in range(len(contours_main))])
    except:
        x_min_main = np.array([np.min(contours_main[j][:, 0]) for j in range(len(contours_main))])

        argmin_x_main = np.array([np.argmin(contours_main[j][:, 0]) for j in range(len(contours_main))])

        x_min_from_argmin = np.array([contours_main[j][argmin_x_main[j], 0] for j in range(len(contours_main))])
        y_corr_x_min_from_argmin = np.array([contours_main[j][argmin_x_main[j], 1] for j in range(len(contours_main))])

        x_max_main = np.array([np.max(contours_main[j][:, 0]) for j in range(len(contours_main))])

        y_min_main = np.array([np.min(contours_main[j][:, 1]) for j in range(len(contours_main))])
        y_max_main = np.array([np.max(contours_main[j][:, 1]) for j in range(len(contours_main))])

    return cx_main, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, y_corr_x_min_from_argmin
def read_xml(xml_file):
    file_name = Path(xml_file).stem
    tree1 = ET.parse(xml_file, parser = ET.XMLParser(encoding='utf-8'))
    root1=tree1.getroot()
    alltags=[elem.tag for elem in root1.iter()]
    link=alltags[0].split('}')[0]+'}'

    index_tot_regions = []
    tot_region_ref = []

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
        elif link+'Border' in alltags:
            region_tags_printspace = np.unique([x for x in alltags if x.endswith('Border')])
            
        for tag in region_tags_printspace:
            if link+'PrintSpace' in alltags:
                tag_endings_printspace = ['}PrintSpace','}printspace']
            elif link+'Border' in alltags:
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

    return tree1, root1, bb_coord_printspace, file_name, id_paragraph, id_header+id_heading, co_text_paragraph, co_text_header+co_text_heading,\
tot_region_ref,x_len, y_len,index_tot_regions, img_poly




# def bounding_box(cnt,color, corr_order_index ):
#     x, y, w, h = cv2.boundingRect(cnt)
#     x = int(x*scale_w)
#     y = int(y*scale_h)
#     
#     w = int(w*scale_w)
#     h = int(h*scale_h)
#     
#     return [x,y,w,h,int(color), int(corr_order_index)+1]

def resize_image(seg_in,input_height,input_width):
    return cv2.resize(seg_in,(input_width,input_height),interpolation=cv2.INTER_NEAREST)

def make_image_from_bb(width_l, height_l, bb_all):
    bb_all =np.array(bb_all)
    img_remade = np.zeros((height_l,width_l ))
    
    for i in range(bb_all.shape[0]):
        img_remade[bb_all[i,1]:bb_all[i,1]+bb_all[i,3],bb_all[i,0]:bb_all[i,0]+bb_all[i,2] ] = 1
    return img_remade

def update_list_and_return_first_with_length_bigger_than_one(index_element_to_be_updated, innner_index_pr_pos, pr_list, pos_list,list_inp):
    list_inp.pop(index_element_to_be_updated)
    if len(pr_list)>0:
        list_inp.insert(index_element_to_be_updated, pr_list)
    else:
        index_element_to_be_updated = index_element_to_be_updated -1
    
    list_inp.insert(index_element_to_be_updated+1, [innner_index_pr_pos])
    if len(pos_list)>0:
        list_inp.insert(index_element_to_be_updated+2, pos_list)
    
    len_all_elements = [len(i) for i in list_inp]
    list_len_bigger_1 = np.where(np.array(len_all_elements)>1)
    list_len_bigger_1 = list_len_bigger_1[0]
    
    if len(list_len_bigger_1)>0:
        early_list_bigger_than_one = list_len_bigger_1[0]
    else:
        early_list_bigger_than_one = -20
    return list_inp, early_list_bigger_than_one

def overlay_layout_on_image(prediction, img, cx_ordered, cy_ordered, color, thickness):
    
    unique_classes = np.unique(prediction[:,:,0])
    rgb_colors = {'0' : [255, 255, 255],
                '1' : [255, 0, 0],
                '2' : [0, 0, 255],
                '3' : [255, 0, 125],
                '4' : [125, 125, 125],
                '5' : [125, 125, 0],
                '6' : [0, 125, 255],
                '7' : [0, 125, 0],
                '8' : [125, 125, 125],
                '9' : [0, 125, 255],
                '10' : [125, 0, 125],
                '11' : [0, 255, 0],
                '12' : [255, 125, 0],
                '13' : [0, 255, 255],
                '14' : [255, 125, 125],
                '15' : [255, 0, 255]}

    layout_only = np.zeros(prediction.shape)

    for unq_class in unique_classes:
        rgb_class_unique = rgb_colors[str(int(unq_class))]
        layout_only[:,:,0][prediction[:,:,0]==unq_class] = rgb_class_unique[0]
        layout_only[:,:,1][prediction[:,:,0]==unq_class] = rgb_class_unique[1]
        layout_only[:,:,2][prediction[:,:,0]==unq_class] = rgb_class_unique[2]



    #img = self.resize_image(img, layout_only.shape[0], layout_only.shape[1])

    layout_only = layout_only.astype(np.int32)
    
    for i in range(len(cx_ordered)-1):
        start_point = (int(cx_ordered[i]), int(cy_ordered[i]))
        end_point = (int(cx_ordered[i+1]), int(cy_ordered[i+1]))
        layout_only = cv2.arrowedLine(layout_only, start_point, end_point, 
                                    color, thickness, tipLength = 0.03)
                
    img = img.astype(np.int32)

    
    
    added_image = cv2.addWeighted(img,0.5,layout_only,0.1,0)
        
    return added_image

def find_format_of_given_filename_in_dir(dir_imgs, f_name):
    ls_imgs = os.listdir(dir_imgs)
    file_interested = [ind for ind in ls_imgs if ind.startswith(f_name+'.')]
    return file_interested[0]
