import os
import cv2
import numpy as np
import seaborn as sns
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import random
from tqdm import tqdm
import imutils
import math



def bluring(img_in,kind):
    if kind=='guass':
        img_blur = cv2.GaussianBlur(img_in,(5,5),0)
    elif kind=="median":
        img_blur = cv2.medianBlur(img_in,5)
    elif kind=='blur':
        img_blur=cv2.blur(img_in,(5,5))
    return img_blur

def elastic_transform(image, alpha, sigma,seedj, random_state=None):
    
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(seedj)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def rotation_90(img):
    img_rot=np.zeros((img.shape[1],img.shape[0],img.shape[2]))
    img_rot[:,:,0]=img[:,:,0].T
    img_rot[:,:,1]=img[:,:,1].T
    img_rot[:,:,2]=img[:,:,2].T
    return img_rot

def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr

def rotate_max_area(image,rotated, rotated_label,angle):
    """ image: cv2 image matrix object
        angle: in degree
    """
    wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0],
                                    math.radians(angle))
    h, w, _ = rotated.shape
    y1 = h//2 - int(hr/2)
    y2 = y1 + int(hr)
    x1 = w//2 - int(wr/2)
    x2 = x1 + int(wr)
    return rotated[y1:y2, x1:x2],rotated_label[y1:y2, x1:x2]
def rotation_not_90_func(img,label,thetha):
    rotated=imutils.rotate(img,thetha)
    rotated_label=imutils.rotate(label,thetha)
    return rotate_max_area(img, rotated,rotated_label,thetha)

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
    n = [f for f in os.listdir(img_folder) if not f.startswith('.')]# os.listdir(img_folder) #List of training images
    random.shuffle(n)
    while True:
        img = np.zeros((batch_size, input_height, input_width, 3)).astype('float')
        mask = np.zeros((batch_size, input_height, input_width, n_classes)).astype('float')
    
        for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 
            #print(img_folder+'/'+n[i])
            
            try:
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
            except:
                img[i-c] = np.ones((input_height, input_width, 3)).astype('float')
                mask[i-c] = np.zeros((input_height, input_width, n_classes)).astype('float')
                

    
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
def get_patches(dir_img_f,dir_seg_f,img,label,height,width,indexer):

    if img.shape[0]<height or img.shape[1]<width:
        img,label=do_padding(img,label,height,width)
    
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

def do_padding(img,label,height,width):
    
    height_new=img.shape[0]
    width_new=img.shape[1]
    
    h_start=0
    w_start=0
    
    if img.shape[0]<height:
        h_start=int( abs(height-img.shape[0])/2. )
        height_new=height
        
    if img.shape[1]<width:
        w_start=int( abs(width-img.shape[1])/2. )
        width_new=width
    
    img_new=np.ones((height_new,width_new,img.shape[2])).astype(float)*255
    label_new=np.zeros((height_new,width_new,label.shape[2])).astype(float)
    
    img_new[h_start:h_start+img.shape[0],w_start:w_start+img.shape[1],:]=np.copy(img[:,:,:])
    label_new[h_start:h_start+label.shape[0],w_start:w_start+label.shape[1],:]=np.copy(label[:,:,:])
    
    return img_new,label_new
    
        
def get_patches_num_scale(dir_img_f,dir_seg_f,img,label,height,width,indexer,n_patches,scaler):
    
    
    if img.shape[0]<height or img.shape[1]<width:
        img,label=do_padding(img,label,height,width)
    
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

def get_patches_num_scale_new(dir_img_f,dir_seg_f,img,label,height,width,indexer,scaler):
    img=resize_image(img,int(img.shape[0]*scaler),int(img.shape[1]*scaler))
    label=resize_image(label,int(label.shape[0]*scaler),int(label.shape[1]*scaler))
    
    if img.shape[0]<height or img.shape[1]<width:
        img,label=do_padding(img,label,height,width)
    
    img_h=img.shape[0]
    img_w=img.shape[1]
    
    height_scale=int(height*1)
    width_scale=int(width*1)
    
    
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
            
            #img_patch=resize_image(img_patch,height,width)
            #label_patch=resize_image(label_patch,height,width)
            
            cv2.imwrite(dir_img_f+'/img_'+str(indexer)+'.png', img_patch )
            cv2.imwrite(dir_seg_f+'/img_'+str(indexer)+'.png' ,   label_patch )
            indexer+=1

    return indexer


def provide_patches(dir_img,dir_seg,dir_flow_train_imgs,
                    dir_flow_train_labels,
                    input_height,input_width,blur_k,blur_aug,
                    flip_aug,binarization,scaling,scales,flip_index,
                    scaling_bluring,scaling_binarization,rotation,
                    rotation_not_90,thetha,scaling_flip,
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
                
                if rotation_not_90:
                    
                    for thetha_i in thetha:
                        img_max_rotated,label_max_rotated=rotation_not_90_func(cv2.imread(dir_img+'/'+im),cv2.imread(dir_seg+'/'+img_name+'.png'),thetha_i)
                        indexer=get_patches(dir_flow_train_imgs,dir_flow_train_labels,
                                                img_max_rotated,
                                                label_max_rotated,
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
                        indexer=get_patches_num_scale_new(dir_flow_train_imgs,dir_flow_train_labels,
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
                            indexer=get_patches_num_scale_new(dir_flow_train_imgs,dir_flow_train_labels,
                                                    bluring( cv2.imread(dir_img+'/'+im) , blur_i) ,
                                                    cv2.imread(dir_seg+'/'+img_name+'.png') ,
                                                    input_height,input_width,indexer=indexer,scaler=sc_ind)

                if scaling_binarization:  
                    for sc_ind in scales:
                        indexer=get_patches_num_scale_new(dir_flow_train_imgs,dir_flow_train_labels,
                                                otsu_copy( cv2.imread(dir_img+'/'+im)) ,
                                                cv2.imread(dir_seg+'/'+img_name+'.png'),
                                                input_height,input_width,indexer=indexer,scaler=sc_ind)
                        
                if scaling_flip:  
                    for sc_ind in scales:
                        for f_i in flip_index:
                            indexer=get_patches_num_scale_new(dir_flow_train_imgs,dir_flow_train_labels,
                                                    cv2.flip( cv2.imread(dir_img+'/'+im) , f_i) ,
                                                    cv2.flip(cv2.imread(dir_seg+'/'+img_name+'.png') ,f_i) ,
                                                    input_height,input_width,indexer=indexer,scaler=sc_ind)

    
    
    
    
    

