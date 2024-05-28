import click
import json
from gt_gen_utils import *
from tqdm import tqdm

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
    "--dir_out",
    "-do",
    help="directory where ground truth images would be written",
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

def pagexml2label(dir_xml,dir_out,type_output,config):
    if config:
        with open(config) as f:
            config_params = json.load(f)
    else:
        print("passed")
        config_params = None
    gt_list = get_content_of_dir(dir_xml)
    get_images_of_ground_truth(gt_list,dir_xml,dir_out,type_output, config, config_params)
    
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
    help="input_height",
)
@click.option(
    "--input_width",
    "-iw",
    help="input_width",
)

def machine_based_reading_order(dir_xml, dir_out_modal_image, dir_out_classes, input_height, input_width):
    xml_files_ind = os.listdir(dir_xml)
    input_height = int(input_height)
    input_width = int(input_width)

    indexer_start= 0#55166
    max_area = 1
    min_area = 0.0001

    for ind_xml in tqdm(xml_files_ind):
        indexer = 0
        #print(ind_xml)
        #print('########################')
        xml_file = os.path.join(dir_xml,ind_xml )
        f_name = ind_xml.split('.')[0]
        file_name, id_paragraph, id_header,co_text_paragraph,\
        co_text_header,tot_region_ref,x_len, y_len,index_tot_regions,img_poly = read_xml(xml_file)
        
        id_all_text = id_paragraph + id_header
        co_text_all = co_text_paragraph + co_text_header
        
        
        _, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, _ = find_new_features_of_contours(co_text_header)
        
        img_header_and_sep = np.zeros((y_len,x_len), dtype='uint8')

        for j in range(len(cy_main)):
            img_header_and_sep[int(y_max_main[j]):int(y_max_main[j])+12,int(x_min_main[j]):int(x_max_main[j]) ] = 1


        texts_corr_order_index  = [index_tot_regions[tot_region_ref.index(i)] for i in id_all_text ]
        texts_corr_order_index_int = [int(x) for x in texts_corr_order_index]
        

        co_text_all, texts_corr_order_index_int = filter_contours_area_of_image(img_poly, co_text_all, texts_corr_order_index_int, max_area, min_area)
        
        arg_array = np.array(range(len(texts_corr_order_index_int)))
        
        labels_con = np.zeros((y_len,x_len,len(arg_array)),dtype='uint8')
        for i in range(len(co_text_all)):
            img_label = np.zeros((y_len,x_len,3),dtype='uint8')
            img_label=cv2.fillPoly(img_label, pts =[co_text_all[i]], color=(1,1,1))
            
            img_label[:,:,0][img_poly[:,:,0]==5] = 2
            img_label[:,:,0][img_header_and_sep[:,:]==1] = 3
            
            labels_con[:,:,i] = img_label[:,:,0]
        
        for i in range(len(texts_corr_order_index_int)):
            for j in range(len(texts_corr_order_index_int)):
                if i!=j:
                    input_matrix = np.zeros((input_height,input_width,3)).astype(np.int8)
                    final_f_name = f_name+'_'+str(indexer+indexer_start)
                    order_class_condition = texts_corr_order_index_int[i]-texts_corr_order_index_int[j]
                    if order_class_condition<0:
                        class_type = 1
                    else:
                        class_type = 0

                    input_matrix[:,:,0] = resize_image(labels_con[:,:,i], input_height, input_width)
                    input_matrix[:,:,1] = resize_image(img_poly[:,:,0], input_height, input_width)
                    input_matrix[:,:,2] = resize_image(labels_con[:,:,j], input_height, input_width)

                    np.save(os.path.join(dir_out_classes,final_f_name+'.npy' ), class_type)
                    
                    cv2.imwrite(os.path.join(dir_out_modal_image,final_f_name+'.png' ), input_matrix)
                    indexer = indexer+1

    
    
if __name__ == "__main__":
    main()
