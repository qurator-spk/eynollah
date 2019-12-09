import os
import cv2
import numpy as np
import seaborn as sns
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import random
from tqdm import tqdm




def bluring(img_in,kind):
    if kind=='guass':
        img_blur = cv2.GaussianBlur(img_in,(5,5),0)
    elif kind=="median":
        img_blur = cv2.medianBlur(img_in,5)
    elif kind=='blur':
        img_blur=cv2.blur(img_in,(5,5))
    return img_blur

def color_images(seg, n_classes):
    ann_u=range(n_classes)
    if len(np.shape(seg))==3:
        seg=seg[:,:,0]
        
    seg_img=np.zeros((np.shape(seg)[0],np.shape(seg)[1],3)).astype(float)
    colors=sns.color_palette("hls", n_classes)
    
    for c in ann_u:
        c=int(c)
        segl=(seg==c)
        seg_img[:,:,0]+=segl*(colors[c][0])
        seg_img[:,:,1]+=segl*(colors[c][1])
        seg_img[:,:,2]+=segl*(colors[c][2])
    return seg_img

   
def resize_image(seg_in,input_height,input_width):
    return cv2.resize(seg_in,(input_width,input_height),interpolation=cv2.INTER_NEAREST)
def get_one_hot(seg,input_height,input_width,n_classes):
    seg=seg[:,:,0]
    seg_f=np.zeros((input_height, input_width,n_classes))
    for j in range(n_classes):
        seg_f[:,:,j]=(seg==j).astype(int)
    return seg_f

    
def IoU(Yi,y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    IoUs = []
    classes_true=np.unique(Yi)
    for c in classes_true:
        TP = np.sum( (Yi == c)&(y_predi==c) )
        FP = np.sum( (Yi != c)&(y_predi==c) )
        FN = np.sum( (Yi == c)&(y_predi != c)) 
        IoU = TP/float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c,TP,FP,FN,IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))
    return mIoU
def data_gen(img_folder, mask_folder, batch_size,input_height, input_width,n_classes):
    c = 0
    n = os.listdir(img_folder) #List of training images
    random.shuffle(n)
    while True:
        img = np.zeros((batch_size, input_height, input_width, 3)).astype('float')
        mask = np.zeros((batch_size, input_height, input_width, n_classes)).astype('float')
    
        for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 
            #print(img_folder+'/'+n[i])
            filename=n[i].split('.')[0]
            train_img = cv2.imread(img_folder+'/'+n[i])/255.
            train_img =  cv2.resize(train_img, (input_width, input_height),interpolation=cv2.INTER_NEAREST)# Read an image from folder and resize
          
            img[i-c] = train_img #add to array - img[0], img[1], and so on.
            train_mask = cv2.imread(mask_folder+'/'+filename+'.png')
            #print(mask_folder+'/'+filename+'.png')
            #print(train_mask.shape)
            train_mask = get_one_hot( resize_image(train_mask,input_height,input_width),input_height,input_width,n_classes)
            #train_mask = train_mask.reshape(224, 224, 1) # Add extra dimension for parity with train_img size [512 * 512 * 3]
        
            mask[i-c] = train_mask
    
        c+=batch_size
        if(c+batch_size>=len(os.listdir(img_folder))):
            c=0
            random.shuffle(n)
        yield img, mask
        
def otsu_copy(img):
    img_r=np.zeros(img.shape)
    img1=img[:,:,0]
    img2=img[:,:,1]
    img3=img[:,:,2]
    _, threshold1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, threshold2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, threshold3 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_r[:,:,0]=threshold1
    img_r[:,:,1]=threshold1
    img_r[:,:,2]=threshold1
    return img_r

def rotation_90(img):
    img_rot=np.zeros((img.shape[1],img.shape[0],img.shape[2]))
    img_rot[:,:,0]=img[:,:,0].T
    img_rot[:,:,1]=img[:,:,1].T
    img_rot[:,:,2]=img[:,:,2].T
    return img_rot
    
def get_patches(dir_img_f,dir_seg_f,img,label,height,width,indexer):

    
    img_h=img.shape[0]
    img_w=img.shape[1]
    
    nxf=img_w/float(width)
    nyf=img_h/float(height)
    
    if nxf>int(nxf):
        nxf=int(nxf)+1
    if nyf>int(nyf):
        nyf=int(nyf)+1
        
    nxf=int(nxf)
    nyf=int(nyf)
        
    for i in range(nxf):
        for j in range(nyf):
            index_x_d=i*width
            index_x_u=(i+1)*width
            
            index_y_d=j*height
            index_y_u=(j+1)*height
            
            if index_x_u>img_w:
                index_x_u=img_w
                index_x_d=img_w-width
            if index_y_u>img_h:
                index_y_u=img_h
                index_y_d=img_h-height
                
            
            img_patch=img[index_y_d:index_y_u,index_x_d:index_x_u,:]
            label_patch=label[index_y_d:index_y_u,index_x_d:index_x_u,:]
            
            cv2.imwrite(dir_img_f+'/img_'+str(indexer)+'.png', img_patch )
            cv2.imwrite(dir_seg_f+'/img_'+str(indexer)+'.png' ,   label_patch )
            indexer+=1
    return indexer


        
def get_patches_num_scale(dir_img_f,dir_seg_f,img,label,height,width,indexer,scaler):

    
    img_h=img.shape[0]
    img_w=img.shape[1]
    
    height_scale=int(height*scaler)
    width_scale=int(width*scaler)
    
    
    nxf=img_w/float(width_scale)
    nyf=img_h/float(height_scale)
    
    if nxf>int(nxf):
        nxf=int(nxf)+1
    if nyf>int(nyf):
        nyf=int(nyf)+1
        
    nxf=int(nxf)
    nyf=int(nyf)
        
    for i in range(nxf):
        for j in range(nyf):
            index_x_d=i*width_scale
            index_x_u=(i+1)*width_scale
            
            index_y_d=j*height_scale
            index_y_u=(j+1)*height_scale
            
            if index_x_u>img_w:
                index_x_u=img_w
                index_x_d=img_w-width_scale
            if index_y_u>img_h:
                index_y_u=img_h
                index_y_d=img_h-height_scale
                
            
            img_patch=img[index_y_d:index_y_u,index_x_d:index_x_u,:]
            label_patch=label[index_y_d:index_y_u,index_x_d:index_x_u,:]
            
            img_patch=resize_image(img_patch,height,width)
            label_patch=resize_image(label_patch,height,width)
            
            cv2.imwrite(dir_img_f+'/img_'+str(indexer)+'.png', img_patch )
            cv2.imwrite(dir_seg_f+'/img_'+str(indexer)+'.png' ,   label_patch )
            indexer+=1

    return indexer



def provide_patches(dir_img,dir_seg,dir_flow_train_imgs,
                    dir_flow_train_labels,
                    input_height,input_width,blur_k,blur_aug,
                    flip_aug,binarization,scaling,scales,flip_index,
                    scaling_bluring,scaling_binarization,rotation,
                    augmentation=False,patches=False):
    
    imgs_cv_train=np.array(os.listdir(dir_img))
    segs_cv_train=np.array(os.listdir(dir_seg))
    
    indexer=0
    for im, seg_i in tqdm(zip(imgs_cv_train,segs_cv_train)):
        img_name=im.split('.')[0]
    
        if not patches:
            cv2.imwrite(dir_flow_train_imgs+'/img_'+str(indexer)+'.png', resize_image(cv2.imread(dir_img+'/'+im),input_height,input_width ) )
            cv2.imwrite(dir_flow_train_labels+'/img_'+str(indexer)+'.png' ,  resize_image(cv2.imread(dir_seg+'/'+img_name+'.png'),input_height,input_width ) )
            indexer+=1
            
            if augmentation:
                if rotation:
                    cv2.imwrite(dir_flow_train_imgs+'/img_'+str(indexer)+'.png',
                                rotation_90(  resize_image(cv2.imread(dir_img+'/'+im),
                                                                input_height,input_width) ) )
                    
                    
                    cv2.imwrite(dir_flow_train_labels+'/img_'+str(indexer)+'.png',
                                rotation_90 ( resize_image(cv2.imread(dir_seg+'/'+img_name+'.png'),
                                                               input_height,input_width) )  )
                    indexer+=1
                    
                if flip_aug:
                    for f_i in flip_index:
                        cv2.imwrite(dir_flow_train_imgs+'/img_'+str(indexer)+'.png',
                                    resize_image(cv2.flip(cv2.imread(dir_img+'/'+im),f_i),input_height,input_width) )
                        
                        cv2.imwrite(dir_flow_train_labels+'/img_'+str(indexer)+'.png' , 
                                    resize_image(cv2.flip(cv2.imread(dir_seg+'/'+img_name+'.png'),f_i),input_height,input_width) ) 
                        indexer+=1
                        
                if blur_aug:   
                    for blur_i in blur_k:
                        cv2.imwrite(dir_flow_train_imgs+'/img_'+str(indexer)+'.png',
                                    (resize_image(bluring(cv2.imread(dir_img+'/'+im),blur_i),input_height,input_width) ) )
                        
                        cv2.imwrite(dir_flow_train_labels+'/img_'+str(indexer)+'.png' , 
                                    resize_image(cv2.imread(dir_seg+'/'+img_name+'.png'),input_height,input_width)  ) 
                        indexer+=1
                        
                    
                if binarization:
                    cv2.imwrite(dir_flow_train_imgs+'/img_'+str(indexer)+'.png',
                                            resize_image(otsu_copy( cv2.imread(dir_img+'/'+im)),input_height,input_width ))
                    
                    cv2.imwrite(dir_flow_train_labels+'/img_'+str(indexer)+'.png',
                                            resize_image( cv2.imread(dir_seg+'/'+img_name+'.png'),input_height,input_width ))
                    indexer+=1
                    
                        
                    

            
            
        if patches:

            indexer=get_patches(dir_flow_train_imgs,dir_flow_train_labels,
                         cv2.imread(dir_img+'/'+im),cv2.imread(dir_seg+'/'+img_name+'.png'),
                          input_height,input_width,indexer=indexer)
            
            if augmentation:
                
                if rotation:
                    
                
                    indexer=get_patches(dir_flow_train_imgs,dir_flow_train_labels,
                                            rotation_90( cv2.imread(dir_img+'/'+im) ),
                                            rotation_90( cv2.imread(dir_seg+'/'+img_name+'.png') ),
                                            input_height,input_width,indexer=indexer)
                if flip_aug:
                    for f_i in flip_index:

                        indexer=get_patches(dir_flow_train_imgs,dir_flow_train_labels,
                                                cv2.flip( cv2.imread(dir_img+'/'+im) , f_i),
                                                cv2.flip( cv2.imread(dir_seg+'/'+img_name+'.png') ,f_i),
                                                input_height,input_width,indexer=indexer)
                if blur_aug:   
                    for blur_i in blur_k:
                        indexer=get_patches(dir_flow_train_imgs,dir_flow_train_labels,
                                                bluring( cv2.imread(dir_img+'/'+im) , blur_i),
                                                cv2.imread(dir_seg+'/'+img_name+'.png'),
                                                input_height,input_width,indexer=indexer)
         

                if scaling:  
                    for sc_ind in scales:
                        indexer=get_patches_num_scale(dir_flow_train_imgs,dir_flow_train_labels,
                                                 cv2.imread(dir_img+'/'+im) ,
                                                 cv2.imread(dir_seg+'/'+img_name+'.png'),
                                                input_height,input_width,indexer=indexer,scaler=sc_ind)
                if binarization:

                    indexer=get_patches(dir_flow_train_imgs,dir_flow_train_labels,
                                            otsu_copy( cv2.imread(dir_img+'/'+im)),
                                            cv2.imread(dir_seg+'/'+img_name+'.png'),
                                            input_height,input_width,indexer=indexer)
    
    
                        
                if scaling_bluring:  
                    for sc_ind in scales:
                        for blur_i in blur_k:
                            indexer=get_patches_num_scale(dir_flow_train_imgs,dir_flow_train_labels,
                                                    bluring( cv2.imread(dir_img+'/'+im) , blur_i) ,
                                                    cv2.imread(dir_seg+'/'+img_name+'.png') ,
                                                    input_height,input_width,indexer=indexer,scaler=sc_ind)

                if scaling_binarization:  
                    for sc_ind in scales:
                        indexer=get_patches_num_scale(dir_flow_train_imgs,dir_flow_train_labels,
                                                 otsu_copy( cv2.imread(dir_img+'/'+im)) ,
                                                 cv2.imread(dir_seg+'/'+img_name+'.png'),
                                                input_height,input_width,indexer=indexer,scaler=sc_ind)
    
    
    
    
    

