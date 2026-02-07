import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from .contour import find_new_features_of_contours, return_contours_of_interested_region
from .resize import resize_image
from .rotate import rotate_image

def get_marginals(text_with_lines, text_regions, num_col, slope_deskew, kernel=None):
    mask_marginals=np.zeros((text_with_lines.shape[0],text_with_lines.shape[1]))
    mask_marginals=mask_marginals.astype(np.uint8)

    text_with_lines=text_with_lines.astype(np.uint8)
    ##text_with_lines=cv2.erode(text_with_lines,self.kernel,iterations=3)

    text_with_lines_eroded=cv2.erode(text_with_lines,kernel,iterations=5)

    if text_with_lines.shape[0]<=1500:
        pass
    elif text_with_lines.shape[0]>1500 and text_with_lines.shape[0]<=1800:
        text_with_lines=resize_image(text_with_lines,int(text_with_lines.shape[0]*1.5),text_with_lines.shape[1])
        text_with_lines=cv2.erode(text_with_lines,kernel,iterations=5)
        text_with_lines=resize_image(text_with_lines,text_with_lines_eroded.shape[0],text_with_lines_eroded.shape[1])
    else:
        text_with_lines=resize_image(text_with_lines,int(text_with_lines.shape[0]*1.8),text_with_lines.shape[1])
        text_with_lines=cv2.erode(text_with_lines,kernel,iterations=7)
        text_with_lines=resize_image(text_with_lines,text_with_lines_eroded.shape[0],text_with_lines_eroded.shape[1])
        
        
    kernel_hor = np.ones((1, 5), dtype=np.uint8)
    text_with_lines = cv2.erode(text_with_lines,kernel_hor,iterations=6)
    
    text_with_lines_y=text_with_lines.sum(axis=0)
    text_with_lines_y_eroded=text_with_lines_eroded.sum(axis=0)

    thickness_along_y_percent=text_with_lines_y_eroded.max()/(float(text_with_lines.shape[0]))*100

    #print(thickness_along_y_percent,'thickness_along_y_percent')

    if thickness_along_y_percent<30:
        min_textline_thickness=8
    elif thickness_along_y_percent>=30 and thickness_along_y_percent<50:
        min_textline_thickness=20
    else:
        min_textline_thickness=45


    if thickness_along_y_percent>=14:

        text_with_lines_y_rev=-1*text_with_lines_y[:]

        text_with_lines_y_rev=text_with_lines_y_rev-np.min(text_with_lines_y_rev)

        sigma_gaus=1
        region_sum_0= gaussian_filter1d(text_with_lines_y, sigma_gaus)

        region_sum_0_rev=gaussian_filter1d(text_with_lines_y_rev, sigma_gaus)

        region_sum_0_updown=region_sum_0[len(region_sum_0)::-1]

        first_nonzero=(next((i for i, x in enumerate(region_sum_0) if x), None))
        last_nonzero=(next((i for i, x in enumerate(region_sum_0_updown) if x), None))


        last_nonzero=len(region_sum_0)-last_nonzero

        mid_point=(last_nonzero+first_nonzero)/2.


        one_third_right=(last_nonzero-mid_point)/3.0
        one_third_left=(mid_point-first_nonzero)/3.0

        peaks, _ = find_peaks(text_with_lines_y_rev, height=0)
        peaks=np.array(peaks)
        peaks=peaks[(peaks>first_nonzero) & (peaks < last_nonzero)]
        peaks=peaks[region_sum_0[peaks]<min_textline_thickness ]


        if num_col==1:
            peaks_right=peaks[peaks>mid_point]
            peaks_left=peaks[peaks<mid_point]
        if num_col==2:
            peaks_right=peaks[peaks>(mid_point+one_third_right)]
            peaks_left=peaks[peaks<(mid_point-one_third_left)]


        try:
            point_right=np.min(peaks_right)
        except:
            point_right=last_nonzero


        try:
            point_left=np.max(peaks_left)
        except:
            point_left=first_nonzero

        if point_left == first_nonzero and point_right == last_nonzero:
            return text_regions


        if point_right>=mask_marginals.shape[1]:
            point_right=mask_marginals.shape[1]-1

        try:
            mask_marginals[:,point_left:point_right]=1
        except:
            mask_marginals[:,:]=1

        mask_marginals_rotated=rotate_image(mask_marginals,-slope_deskew)

        mask_marginals_rotated_sum=mask_marginals_rotated.sum(axis=0)

        mask_marginals_rotated_sum[mask_marginals_rotated_sum!=0]=1
        index_x=np.array(range(len(mask_marginals_rotated_sum)))+1

        index_x_interest=index_x[mask_marginals_rotated_sum==1]

        min_point_of_left_marginal=np.min(index_x_interest)-16
        max_point_of_right_marginal=np.max(index_x_interest)+16

        if min_point_of_left_marginal<0:
            min_point_of_left_marginal=0
        if max_point_of_right_marginal>=text_regions.shape[1]:
            max_point_of_right_marginal=text_regions.shape[1]-1

        text_regions_org = np.copy(text_regions)
        text_regions[text_regions[:,:]==1]=4
        
        pixel_img=4
        min_area_text=0.00001
        
        polygon_mask_marginals_rotated = return_contours_of_interested_region(mask_marginals,1,min_area_text)
        
        polygon_mask_marginals_rotated = polygon_mask_marginals_rotated[0]

        polygons_of_marginals=return_contours_of_interested_region(text_regions,pixel_img,min_area_text)

        cx_text_only,cy_text_only ,x_min_text_only,x_max_text_only, y_min_text_only ,y_max_text_only,y_cor_x_min_main=find_new_features_of_contours(polygons_of_marginals)

        text_regions[(text_regions[:,:]==4)]=1

        marginlas_should_be_main_text=[]

        x_min_marginals_left=[]
        x_min_marginals_right=[]

        for i in range(len(cx_text_only)):
            results = cv2.pointPolygonTest(polygon_mask_marginals_rotated, (cx_text_only[i], cy_text_only[i]), False)

            if results == -1:
                marginlas_should_be_main_text.append(polygons_of_marginals[i])



        text_regions_org=cv2.fillPoly(text_regions_org, pts =marginlas_should_be_main_text, color=(4,4))
        text_regions = np.copy(text_regions_org)
            


        ###text_regions[:,0:point_left][text_regions[:,0:point_left]==1]=4

        ###text_regions[:,point_right:][ text_regions[:,point_right:]==1]=4
        #plt.plot(region_sum_0)
        #plt.plot(peaks,region_sum_0[peaks],'*')
        #plt.show()


        #plt.imshow(text_regions)
        #plt.show()

        #sys.exit()
    else:
        pass
    return text_regions
