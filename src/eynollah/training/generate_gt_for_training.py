import click
import json
import os
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from eynollah.training.gt_gen_utils import (
    filter_contours_area_of_image,
    find_format_of_given_filename_in_dir,
    find_new_features_of_contours,
    fit_text_single_line,
    get_content_of_dir,
    get_images_of_ground_truth,
    get_layout_contours_for_visualization,
    get_textline_contours_and_ocr_text,
    get_textline_contours_for_visualization,
    overlay_layout_on_image,
    read_xml,
    resize_image,
    visualize_image_from_contours,
    visualize_image_from_contours_layout
)

@click.group()
def main():
    pass

@main.command()
@click.option(
    "--dir_xml",
    "-dx",
    help="directory of GT page-xml files",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--dir_images",
    "-di",
    help="directory of org images. If print space cropping or scaling is needed for labels it would be great to provide the original images to apply the same function on them. So if -ps is not set true or in config files no columns_width key is given this argumnet can be ignored. File stems in this directory should be the same as those in dir_xml.",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--dir_out_images",
    "-doi",
    help="directory where the output org images after undergoing a process (like print space cropping or scaling) will be written.",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--dir_out",
    "-do",
    help="directory where ground truth label images would be written",
    type=click.Path(exists=True, file_okay=False),
)

@click.option(
    "--config",
    "-cfg",
    help="config file of prefered layout or use case.",
    type=click.Path(exists=True, dir_okay=False),
)

@click.option(
    "--type_output",
    "-to",
    help="this defines how output should be. A 2d image array or a 3d image array encoded with RGB color. Just pass 2d or 3d. The file will be saved one directory up. 2D image array is 3d but only information of one channel would be enough since all channels have the same values.",
)
@click.option(
    "--printspace",
    "-ps",
    is_flag=True,
    help="if this parameter set to true, generated labels and in the case of provided org images cropping will be imposed and cropped labels and images will be written in output directories.",
)

def pagexml2label(dir_xml,dir_out,type_output,config, printspace, dir_images, dir_out_images):
    if config:
        with open(config) as f:
            config_params = json.load(f)
    else:
        print("passed")
        config_params = None
    gt_list = get_content_of_dir(dir_xml)
    get_images_of_ground_truth(gt_list,dir_xml,dir_out,type_output, config, config_params, printspace, dir_images, dir_out_images)
    
@main.command()
@click.option(
    "--dir_imgs",
    "-dis",
    help="directory of images with high resolution.",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--dir_out_images",
    "-dois",
    help="directory where degraded images will be written.",
    type=click.Path(exists=True, file_okay=False),
)

@click.option(
    "--dir_out_labels",
    "-dols",
    help="directory where original images will be written as labels.",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--scales",
    "-scs",
    help="json dictionary where the scales are written.",
    type=click.Path(exists=True, dir_okay=False),
)
def image_enhancement(dir_imgs, dir_out_images, dir_out_labels, scales):
    ls_imgs = os.listdir(dir_imgs)
    with open(scales) as f:
        scale_dict = json.load(f)
    ls_scales = scale_dict['scales']

    for img in tqdm(ls_imgs):
        img_name = img.split('.')[0]
        img_type = img.split('.')[1]
        image = cv2.imread(os.path.join(dir_imgs, img))
        for i, scale in enumerate(ls_scales):
            height_sc = int(image.shape[0]*scale)
            width_sc = int(image.shape[1]*scale)
            
            image_down_scaled = resize_image(image, height_sc, width_sc)
            image_back_to_org_scale = resize_image(image_down_scaled, image.shape[0], image.shape[1])
            
            cv2.imwrite(os.path.join(dir_out_images, img_name+'_'+str(i)+'.'+img_type), image_back_to_org_scale)
            cv2.imwrite(os.path.join(dir_out_labels, img_name+'_'+str(i)+'.'+img_type), image)
    
    
@main.command()
@click.option(
    "--dir_xml",
    "-dx",
    help="directory of GT page-xml files",
    type=click.Path(exists=True, file_okay=False),
)

@click.option(
    "--dir_out_modal_image",
    "-domi",
    help="directory where ground truth images would be written",
    type=click.Path(exists=True, file_okay=False),
)

@click.option(
    "--dir_out_classes",
    "-docl",
    help="directory where ground truth classes would be written",
    type=click.Path(exists=True, file_okay=False),
)

@click.option(
    "--input_height",
    "-ih",
    help="input height",
)
@click.option(
    "--input_width",
    "-iw",
    help="input width",
)
@click.option(
    "--min_area_size",
    "-min",
    help="min area size of regions considered for reading order training.",
)

@click.option(
    "--min_area_early",
    "-min_early",
    help="If you have already generated a training dataset using a specific minimum area value and now wish to create a dataset with a smaller minimum area value, you can avoid regenerating the previous dataset by providing the earlier minimum area value. This will ensure that only the missing data is generated.",
)

def machine_based_reading_order(dir_xml, dir_out_modal_image, dir_out_classes, input_height, input_width, min_area_size, min_area_early):
    xml_files_ind = os.listdir(dir_xml)
    xml_files_ind = [ind_xml for ind_xml in xml_files_ind if ind_xml.endswith('.xml')]
    input_height = int(input_height)
    input_width = int(input_width)
    min_area = float(min_area_size)
    if min_area_early:
        min_area_early = float(min_area_early)
    

    indexer_start= 0#55166
    max_area = 1
    #min_area = 0.0001

    for ind_xml in tqdm(xml_files_ind):
        indexer = 0
        #print(ind_xml)
        #print('########################')
        xml_file = os.path.join(dir_xml,ind_xml )
        f_name = ind_xml.split('.')[0]
        _, _, _, file_name, id_paragraph, id_header,co_text_paragraph,co_text_header,tot_region_ref,x_len, y_len,index_tot_regions,img_poly = read_xml(xml_file)
        
        id_all_text = id_paragraph + id_header
        co_text_all = co_text_paragraph + co_text_header
        
        
        _, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, _ = find_new_features_of_contours(co_text_header)
        
        img_header_and_sep = np.zeros((y_len,x_len), dtype='uint8')

        for j in range(len(cy_main)):
            img_header_and_sep[int(y_max_main[j]):int(y_max_main[j])+12,int(x_min_main[j]):int(x_max_main[j]) ] = 1


        texts_corr_order_index  = [index_tot_regions[tot_region_ref.index(i)] for i in id_all_text ]
        texts_corr_order_index_int = [int(x) for x in texts_corr_order_index]
        

        co_text_all, texts_corr_order_index_int, regions_ar_less_than_early_min = filter_contours_area_of_image(img_poly, co_text_all, texts_corr_order_index_int, max_area, min_area, min_area_early)
        
        
        arg_array = np.array(range(len(texts_corr_order_index_int)))
        
        labels_con = np.zeros((y_len,x_len,len(arg_array)),dtype='uint8')
        for i in range(len(co_text_all)):
            img_label = np.zeros((y_len,x_len,3),dtype='uint8')
            img_label=cv2.fillPoly(img_label, pts =[co_text_all[i]], color=(1,1,1))
            
            img_label[:,:,0][img_poly[:,:,0]==5] = 2
            img_label[:,:,0][img_header_and_sep[:,:]==1] = 3
            
            labels_con[:,:,i] = img_label[:,:,0]
        
        labels_con = resize_image(labels_con, input_height, input_width)
        img_poly = resize_image(img_poly, input_height, input_width)
        
        
        for i in range(len(texts_corr_order_index_int)):
            for j in range(len(texts_corr_order_index_int)):
                if i!=j:
                    if regions_ar_less_than_early_min:
                        if regions_ar_less_than_early_min[i]==1:
                            input_multi_visual_modal = np.zeros((input_height,input_width,3)).astype(np.int8)
                            final_f_name = f_name+'_'+str(indexer+indexer_start)
                            order_class_condition = texts_corr_order_index_int[i]-texts_corr_order_index_int[j]
                            if order_class_condition<0:
                                class_type = 1
                            else:
                                class_type = 0

                            input_multi_visual_modal[:,:,0] = labels_con[:,:,i]
                            input_multi_visual_modal[:,:,1] = img_poly[:,:,0]
                            input_multi_visual_modal[:,:,2] = labels_con[:,:,j]

                            np.save(os.path.join(dir_out_classes,final_f_name+'_missed.npy' ), class_type)
                            
                            cv2.imwrite(os.path.join(dir_out_modal_image,final_f_name+'_missed.png' ), input_multi_visual_modal)
                            indexer = indexer+1
                            
                    else:
                        input_multi_visual_modal = np.zeros((input_height,input_width,3)).astype(np.int8)
                        final_f_name = f_name+'_'+str(indexer+indexer_start)
                        order_class_condition = texts_corr_order_index_int[i]-texts_corr_order_index_int[j]
                        if order_class_condition<0:
                            class_type = 1
                        else:
                            class_type = 0

                        input_multi_visual_modal[:,:,0] = labels_con[:,:,i]
                        input_multi_visual_modal[:,:,1] = img_poly[:,:,0]
                        input_multi_visual_modal[:,:,2] = labels_con[:,:,j]

                        np.save(os.path.join(dir_out_classes,final_f_name+'.npy' ), class_type)
                        
                        cv2.imwrite(os.path.join(dir_out_modal_image,final_f_name+'.png' ), input_multi_visual_modal)
                        indexer = indexer+1
                    
                    
@main.command()
@click.option(
    "--xml_file",
    "-xml",
    help="xml filename",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--dir_xml",
    "-dx",
    help="directory of GT page-xml files",
    type=click.Path(exists=True, file_okay=False),
)

@click.option(
    "--dir_out",
    "-o",
    help="directory where plots will be written",
    type=click.Path(exists=True, file_okay=False),
)

@click.option(
    "--dir_imgs",
    "-di",
    help="directory where the overlayed plots will be written", )

def visualize_reading_order(xml_file, dir_xml, dir_out, dir_imgs):
    assert xml_file or dir_xml, "A single xml file -xml or a dir of xml files -dx is required not both of them"

    if dir_xml:
        xml_files_ind = os.listdir(dir_xml)
        xml_files_ind = [ind_xml for ind_xml in xml_files_ind if ind_xml.endswith('.xml')]
    else:
        xml_files_ind = [xml_file]
        
    indexer_start= 0#55166
    #min_area = 0.0001

    for ind_xml in tqdm(xml_files_ind):
        indexer = 0
        #print(ind_xml)
        #print('########################')
        #xml_file = os.path.join(dir_xml,ind_xml )
        
        if dir_xml:
            xml_file = os.path.join(dir_xml,ind_xml )
            f_name = Path(ind_xml).stem
        else:
            xml_file = os.path.join(ind_xml )
            f_name = Path(ind_xml).stem
        print(f_name, 'f_name')
        
        #f_name = ind_xml.split('.')[0]
        _, _, _, file_name, id_paragraph, id_header,co_text_paragraph,co_text_header,tot_region_ref,x_len, y_len,index_tot_regions,img_poly = read_xml(xml_file)
        
        id_all_text = id_paragraph + id_header
        co_text_all = co_text_paragraph + co_text_header
        
        
        cx_main, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, _ = find_new_features_of_contours(co_text_all)

        texts_corr_order_index  = [int(index_tot_regions[tot_region_ref.index(i)]) for i in id_all_text ]
        #texts_corr_order_index_int = [int(x) for x in texts_corr_order_index]
        
        
        #cx_ordered = np.array(cx_main)[np.array(texts_corr_order_index)]
        #cx_ordered = cx_ordered.astype(np.int32)
        
        cx_ordered = [int(val) for (_, val) in sorted(zip(texts_corr_order_index, cx_main), key=lambda x: \
          x[0], reverse=False)]
        #cx_ordered = cx_ordered.astype(np.int32)
        
        cy_ordered = [int(val) for (_, val) in sorted(zip(texts_corr_order_index, cy_main), key=lambda x: \
          x[0], reverse=False)]
        #cy_ordered = cy_ordered.astype(np.int32)
        

        color = (0, 0, 255)
        thickness = 20
        if dir_imgs:
            layout = np.zeros( (y_len,x_len,3) ) 
            layout = cv2.fillPoly(layout, pts =co_text_all, color=(1,1,1))
                
            img_file_name_with_format = find_format_of_given_filename_in_dir(dir_imgs, f_name)
            img = cv2.imread(os.path.join(dir_imgs, img_file_name_with_format))
            
            overlayed = overlay_layout_on_image(layout, img, cx_ordered, cy_ordered, color, thickness)
            cv2.imwrite(os.path.join(dir_out, f_name+'.png'), overlayed)
            
        else:
            img = np.zeros( (y_len,x_len,3) ) 
            img = cv2.fillPoly(img, pts =co_text_all, color=(255,0,0))
            for i in range(len(cx_ordered)-1):
                start_point = (int(cx_ordered[i]), int(cy_ordered[i]))
                end_point = (int(cx_ordered[i+1]), int(cy_ordered[i+1]))
                img = cv2.arrowedLine(img, start_point, end_point, 
                                            color, thickness, tipLength = 0.03)
            
            cv2.imwrite(os.path.join(dir_out, f_name+'.png'), img)

    
@main.command()
@click.option(
    "--xml_file",
    "-xml",
    help="xml filename",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--dir_xml",
    "-dx",
    help="directory of GT page-xml files",
    type=click.Path(exists=True, file_okay=False),
)

@click.option(
    "--dir_out",
    "-o",
    help="directory where plots will be written",
    type=click.Path(exists=True, file_okay=False),
)

@click.option(
    "--dir_imgs",
    "-di",
    help="directory of images where textline segmentation will be overlayed", )

def visualize_textline_segmentation(xml_file, dir_xml, dir_out, dir_imgs):
    assert xml_file or dir_xml, "A single xml file -xml or a dir of xml files -dx is required not both of them"
    if dir_xml:
        xml_files_ind = os.listdir(dir_xml)
        xml_files_ind = [ind_xml for ind_xml in xml_files_ind if ind_xml.endswith('.xml')]
    else:
        xml_files_ind = [xml_file]
        
    for ind_xml in tqdm(xml_files_ind):
        indexer = 0
        #print(ind_xml)
        #print('########################')
        xml_file = os.path.join(dir_xml,ind_xml )
        f_name = Path(ind_xml).stem
        
        img_file_name_with_format = find_format_of_given_filename_in_dir(dir_imgs, f_name)
        img = cv2.imread(os.path.join(dir_imgs, img_file_name_with_format))
            
        co_tetxlines, y_len, x_len = get_textline_contours_for_visualization(xml_file)
        
        added_image = visualize_image_from_contours(co_tetxlines, img)
        
        cv2.imwrite(os.path.join(dir_out, f_name+'.png'), added_image)

        
        
@main.command()
@click.option(
    "--xml_file",
    "-xml",
    help="xml filename",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--dir_xml",
    "-dx",
    help="directory of GT page-xml files",
    type=click.Path(exists=True, file_okay=False),
)

@click.option(
    "--dir_out",
    "-o",
    help="directory where plots will be written",
    type=click.Path(exists=True, file_okay=False),
)

@click.option(
    "--dir_imgs",
    "-di",
    help="directory of images where textline segmentation will be overlayed", )

def visualize_layout_segmentation(xml_file, dir_xml, dir_out, dir_imgs):
    assert xml_file or dir_xml, "A single xml file -xml or a dir of xml files -dx is required not both of them"
    if dir_xml:
        xml_files_ind = os.listdir(dir_xml)
        xml_files_ind = [ind_xml for ind_xml in xml_files_ind if ind_xml.endswith('.xml')]
    else:
        xml_files_ind = [xml_file]
        
    for ind_xml in tqdm(xml_files_ind):
        indexer = 0
        #print(ind_xml)
        #print('########################')
        if dir_xml:
            xml_file = os.path.join(dir_xml,ind_xml )
            f_name = Path(ind_xml).stem
        else:
            xml_file = os.path.join(ind_xml )
            f_name = Path(ind_xml).stem
        print(f_name, 'f_name')
        
        img_file_name_with_format = find_format_of_given_filename_in_dir(dir_imgs, f_name)
        img = cv2.imread(os.path.join(dir_imgs, img_file_name_with_format))
            
        co_text, co_graphic, co_sep, co_img, co_table, co_map, co_noise, y_len, x_len = get_layout_contours_for_visualization(xml_file)
        
        
        added_image = visualize_image_from_contours_layout(co_text['paragraph'], co_text['header']+co_text['heading'], co_text['drop-capital'], co_sep, co_img, co_text['marginalia'], co_table, img)

        cv2.imwrite(os.path.join(dir_out, f_name+'.png'), added_image)




@main.command()
@click.option(
    "--xml_file",
    "-xml",
    help="xml filename",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--dir_xml",
    "-dx",
    help="directory of GT page-xml files",
    type=click.Path(exists=True, file_okay=False),
)

@click.option(
    "--dir_out",
    "-o",
    help="directory where plots will be written",
    type=click.Path(exists=True, file_okay=False),
)


def visualize_ocr_text(xml_file, dir_xml, dir_out):
    assert xml_file or dir_xml, "A single xml file -xml or a dir of xml files -dx is required not both of them"
    if dir_xml:
        xml_files_ind = os.listdir(dir_xml)
        xml_files_ind = [ind_xml for ind_xml in xml_files_ind if ind_xml.endswith('.xml')]
    else:
        xml_files_ind = [xml_file]
        
    font_path = "Charis-7.000/Charis-Regular.ttf"  # Make sure this file exists!
    font = ImageFont.truetype(font_path, 40)
        
    for ind_xml in tqdm(xml_files_ind):
        indexer = 0
        #print(ind_xml)
        #print('########################')
        if dir_xml:
            xml_file = os.path.join(dir_xml,ind_xml )
            f_name = Path(ind_xml).stem
        else:
            xml_file = os.path.join(ind_xml )
            f_name = Path(ind_xml).stem
        print(f_name, 'f_name')
            
        co_tetxlines, y_len, x_len, ocr_texts = get_textline_contours_and_ocr_text(xml_file)
    
        total_bb_coordinates = []
        
        image_text = Image.new("RGB", (x_len, y_len), "white")
        draw = ImageDraw.Draw(image_text)
        
        
        
        for index, cnt in enumerate(co_tetxlines):
            x,y,w,h = cv2.boundingRect(cnt)
            #total_bb_coordinates.append([x,y,w,h])
            
            #fit_text_single_line
            
            #x_bb = bb_ind[0]
            #y_bb = bb_ind[1]
            #w_bb = bb_ind[2]
            #h_bb = bb_ind[3]
            if ocr_texts[index]:
                
                
                is_vertical = h > 2*w  # Check orientation
                font = fit_text_single_line(draw, ocr_texts[index], font_path, w, int(h*0.4) )
                
                if is_vertical:
                    
                    vertical_font = fit_text_single_line(draw, ocr_texts[index], font_path, h, int(w * 0.8))

                    text_img = Image.new("RGBA", (h, w), (255, 255, 255, 0))  # Note: dimensions are swapped
                    text_draw = ImageDraw.Draw(text_img)
                    text_draw.text((0, 0), ocr_texts[index], font=vertical_font, fill="black")

                    # Rotate text image by 90 degrees
                    rotated_text = text_img.rotate(90, expand=1)

                    # Calculate paste position (centered in bbox)
                    paste_x = x + (w - rotated_text.width) // 2
                    paste_y = y + (h - rotated_text.height) // 2

                    image_text.paste(rotated_text, (paste_x, paste_y), rotated_text)  # Use rotated image as mask
                else:
                    text_bbox = draw.textbbox((0, 0), ocr_texts[index], font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    text_x = x + (w - text_width) // 2  # Center horizontally
                    text_y = y + (h - text_height) // 2  # Center vertically

                    # Draw the text
                    draw.text((text_x, text_y), ocr_texts[index], fill="black", font=font)
        image_text.save(os.path.join(dir_out, f_name+'.png'))
