import cv2
import os

def resize_image(seg_in, input_height, input_width):
    return cv2.resize(seg_in, (input_width, input_height), interpolation=cv2.INTER_NEAREST)


dir_imgs = './training_data_sample_enhancement/images'
dir_out_imgs = './training_data_sample_enhancement/images_gt'
dir_out_labs = './training_data_sample_enhancement/labels_gt'

ls_imgs = os.listdir(dir_imgs)


ls_scales = [ 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,  0.8, 0.85,  0.9]


for img in ls_imgs:
    img_name = img.split('.')[0]
    img_type = img.split('.')[1]
    image = cv2.imread(os.path.join(dir_imgs, img))
    for i, scale in enumerate(ls_scales):
        height_sc = int(image.shape[0]*scale)
        width_sc = int(image.shape[1]*scale)
        
        image_down_scaled = resize_image(image, height_sc, width_sc)
        image_back_to_org_scale = resize_image(image_down_scaled, image.shape[0], image.shape[1])
        
        cv2.imwrite(os.path.join(dir_out_imgs, img_name+'_'+str(i)+'.'+img_type), image_back_to_org_scale)
        cv2.imwrite(os.path.join(dir_out_labs, img_name+'_'+str(i)+'.'+img_type), image)
        
