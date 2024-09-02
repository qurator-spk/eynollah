import cv2
import numpy as np
from shapely import geometry

from .rotate import rotate_image, rotation_image_new
from multiprocessing import Process, Queue, cpu_count
from multiprocessing import Pool
def contours_in_same_horizon(cy_main_hor):
    X1 = np.zeros((len(cy_main_hor), len(cy_main_hor)))
    X2 = np.zeros((len(cy_main_hor), len(cy_main_hor)))

    X1[0::1, :] = cy_main_hor[:]
    X2 = X1.T

    X_dif = np.abs(X2 - X1)
    args_help = np.array(range(len(cy_main_hor)))
    all_args = []
    for i in range(len(cy_main_hor)):
        list_h = list(args_help[X_dif[i, :] <= 20])
        list_h.append(i)
        if len(list_h) > 1:
            all_args.append(list(set(list_h)))
    return np.unique(np.array(all_args, dtype=object))

def find_contours_mean_y_diff(contours_main):
    M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
    cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
    return np.mean(np.diff(np.sort(np.array(cy_main))))


def get_text_region_boxes_by_given_contours(contours):

    kernel = np.ones((5, 5), np.uint8)
    boxes = []
    contours_new = []
    for jj in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[jj])

        boxes.append([x, y, w, h])
        contours_new.append(contours[jj])

    del contours
    return boxes, contours_new

def filter_contours_area_of_image(image, contours, hierarchy, max_area, min_area):
    found_polygons_early = list()

    for jv,c in enumerate(contours):
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue

        polygon = geometry.Polygon([point[0] for point in c])
        area = polygon.area
        if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]) and hierarchy[0][jv][3] == -1:  # and hierarchy[0][jv][3]==-1 :
            found_polygons_early.append(np.array([[point] for point in polygon.exterior.coords], dtype=np.uint))
    return found_polygons_early

def filter_contours_area_of_image_tables(image, contours, hierarchy, max_area, min_area):
    found_polygons_early = list()

    for jv,c in enumerate(contours):
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue

        polygon = geometry.Polygon([point[0] for point in c])
        # area = cv2.contourArea(c)
        area = polygon.area
        ##print(np.prod(thresh.shape[:2]))
        # Check that polygon has area greater than minimal area
        # print(hierarchy[0][jv][3],hierarchy )
        if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]):  # and hierarchy[0][jv][3]==-1 :
            # print(c[0][0][1])
            found_polygons_early.append(np.array([[point] for point in polygon.exterior.coords], dtype=np.int32))
    return found_polygons_early

def find_new_features_of_contours(contours_main):

    areas_main = np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
    M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
    cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
    cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
    try:
        x_min_main = np.array([np.min(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

        argmin_x_main = np.array([np.argmin(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

        x_min_from_argmin = np.array([contours_main[j][argmin_x_main[j], 0, 0] for j in range(len(contours_main))])
        y_corr_x_min_from_argmin = np.array([contours_main[j][argmin_x_main[j], 0, 1] for j in range(len(contours_main))])

        x_max_main = np.array([np.max(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

        y_min_main = np.array([np.min(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
        y_max_main = np.array([np.max(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
    except:
        x_min_main = np.array([np.min(contours_main[j][:, 0]) for j in range(len(contours_main))])

        argmin_x_main = np.array([np.argmin(contours_main[j][:, 0]) for j in range(len(contours_main))])

        x_min_from_argmin = np.array([contours_main[j][argmin_x_main[j], 0] for j in range(len(contours_main))])
        y_corr_x_min_from_argmin = np.array([contours_main[j][argmin_x_main[j], 1] for j in range(len(contours_main))])

        x_max_main = np.array([np.max(contours_main[j][:, 0]) for j in range(len(contours_main))])

        y_min_main = np.array([np.min(contours_main[j][:, 1]) for j in range(len(contours_main))])
        y_max_main = np.array([np.max(contours_main[j][:, 1]) for j in range(len(contours_main))])

    # dis_x=np.abs(x_max_main-x_min_main)

    return cx_main, cy_main, x_min_main, x_max_main, y_min_main, y_max_main, y_corr_x_min_from_argmin
def find_features_of_contours(contours_main):
    

    areas_main=np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
    M_main=[cv2.moments(contours_main[j]) for j in range(len(contours_main))]
    cx_main=[(M_main[j]['m10']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
    cy_main=[(M_main[j]['m01']/(M_main[j]['m00']+1e-32)) for j in range(len(M_main))]
    x_min_main=np.array([np.min(contours_main[j][:,0,0]) for j in range(len(contours_main))])
    x_max_main=np.array([np.max(contours_main[j][:,0,0]) for j in range(len(contours_main))])

    y_min_main=np.array([np.min(contours_main[j][:,0,1]) for j in range(len(contours_main))])
    y_max_main=np.array([np.max(contours_main[j][:,0,1]) for j in range(len(contours_main))])

    
    return y_min_main, y_max_main
def return_parent_contours(contours, hierarchy):
    contours_parent = [contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1]
    return contours_parent

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

    contours_imgs = return_parent_contours(contours_imgs, hierarchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hierarchy, max_area=1, min_area=min_area)

    return contours_imgs

def do_work_of_contours_in_image(queue_of_all_params, contours_per_process, indexes_r_con_per_pro, img, slope_first):
    cnts_org_per_each_subprocess = []
    index_by_text_region_contours = []
    for mv in range(len(contours_per_process)):
        index_by_text_region_contours.append(indexes_r_con_per_pro[mv])
        
        img_copy = np.zeros(img.shape)
        img_copy = cv2.fillPoly(img_copy, pts=[contours_per_process[mv]], color=(1, 1, 1))

        img_copy = rotation_image_new(img_copy, -slope_first)

        img_copy = img_copy.astype(np.uint8)
        imgray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        cont_int, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cont_int[0][:, 0, 0] = cont_int[0][:, 0, 0] + np.abs(img_copy.shape[1] - img.shape[1])
        cont_int[0][:, 0, 1] = cont_int[0][:, 0, 1] + np.abs(img_copy.shape[0] - img.shape[0])


        cnts_org_per_each_subprocess.append(cont_int[0])

    queue_of_all_params.put([ cnts_org_per_each_subprocess, index_by_text_region_contours])


def get_textregion_contours_in_org_image_multi(cnts, img, slope_first):
    
    num_cores = cpu_count()
    queue_of_all_params = Queue()

    processes = []
    nh = np.linspace(0, len(cnts), num_cores + 1)
    indexes_by_text_con = np.array(range(len(cnts)))
    for i in range(num_cores):
        contours_per_process = cnts[int(nh[i]) : int(nh[i + 1])]
        indexes_text_con_per_process = indexes_by_text_con[int(nh[i]) : int(nh[i + 1])]

        processes.append(Process(target=do_work_of_contours_in_image, args=(queue_of_all_params, contours_per_process, indexes_text_con_per_process, img,slope_first )))
    for i in range(num_cores):
        processes[i].start()
    cnts_org = []
    all_index_text_con = []
    for i in range(num_cores):
        list_all_par = queue_of_all_params.get(True)
        contours_for_sub_process = list_all_par[0]
        indexes_for_sub_process = list_all_par[1]
        for j in range(len(contours_for_sub_process)):
            cnts_org.append(contours_for_sub_process[j])
            all_index_text_con.append(indexes_for_sub_process[j])
    for i in range(num_cores):
        processes[i].join()

    print(all_index_text_con)
    return cnts_org
def loop_contour_image(index_l, cnts,img, slope_first):
    img_copy = np.zeros(img.shape)
    img_copy = cv2.fillPoly(img_copy, pts=[cnts[index_l]], color=(1, 1, 1))

    # plt.imshow(img_copy)
    # plt.show()

    # print(img.shape,'img')
    img_copy = rotation_image_new(img_copy, -slope_first)
    ##print(img_copy.shape,'img_copy')
    # plt.imshow(img_copy)
    # plt.show()

    img_copy = img_copy.astype(np.uint8)
    imgray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    cont_int, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cont_int[0][:, 0, 0] = cont_int[0][:, 0, 0] + np.abs(img_copy.shape[1] - img.shape[1])
    cont_int[0][:, 0, 1] = cont_int[0][:, 0, 1] + np.abs(img_copy.shape[0] - img.shape[0])
    # print(np.shape(cont_int[0]))
    return cont_int[0]

def get_textregion_contours_in_org_image_multi2(cnts, img, slope_first):

    cnts_org = []
    # print(cnts,'cnts')
    with Pool(cpu_count()) as p:
        cnts_org = p.starmap(loop_contour_image, [(index_l,cnts, img,slope_first) for index_l in range(len(cnts))])
        
    return cnts_org

def get_textregion_contours_in_org_image(cnts, img, slope_first):

    cnts_org = []
    # print(cnts,'cnts')
    for i in range(len(cnts)):
        img_copy = np.zeros(img.shape)
        img_copy = cv2.fillPoly(img_copy, pts=[cnts[i]], color=(1, 1, 1))

        # plt.imshow(img_copy)
        # plt.show()

        # print(img.shape,'img')
        img_copy = rotation_image_new(img_copy, -slope_first)
        ##print(img_copy.shape,'img_copy')
        # plt.imshow(img_copy)
        # plt.show()

        img_copy = img_copy.astype(np.uint8)
        imgray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        cont_int, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cont_int[0][:, 0, 0] = cont_int[0][:, 0, 0] + np.abs(img_copy.shape[1] - img.shape[1])
        cont_int[0][:, 0, 1] = cont_int[0][:, 0, 1] + np.abs(img_copy.shape[0] - img.shape[0])
        # print(np.shape(cont_int[0]))
        cnts_org.append(cont_int[0])

    return cnts_org

def get_textregion_contours_in_org_image_light(cnts, img, slope_first):
    
    h_o = img.shape[0]
    w_o = img.shape[1]
    
    img = cv2.resize(img, (int(img.shape[1]/3.), int(img.shape[0]/3.)), interpolation=cv2.INTER_NEAREST)
    ##cnts = list( (np.array(cnts)/2).astype(np.int16) )
    #cnts = cnts/2
    cnts = [(i/ 3).astype(np.int32) for i in cnts]
    cnts_org = []
    #print(cnts,'cnts')
    for i in range(len(cnts)):
        img_copy = np.zeros(img.shape)
        img_copy = cv2.fillPoly(img_copy, pts=[cnts[i]], color=(1, 1, 1))

        # plt.imshow(img_copy)
        # plt.show()

        # print(img.shape,'img')
        img_copy = rotation_image_new(img_copy, -slope_first)
        ##print(img_copy.shape,'img_copy')
        # plt.imshow(img_copy)
        # plt.show()

        img_copy = img_copy.astype(np.uint8)
        imgray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, 0)

        cont_int, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cont_int[0][:, 0, 0] = cont_int[0][:, 0, 0] + np.abs(img_copy.shape[1] - img.shape[1])
        cont_int[0][:, 0, 1] = cont_int[0][:, 0, 1] + np.abs(img_copy.shape[0] - img.shape[0])
        # print(np.shape(cont_int[0]))
        cnts_org.append(cont_int[0]*3)

    return cnts_org

def return_contours_of_interested_textline(region_pre_p, pixel):

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

    contours_imgs = return_parent_contours(contours_imgs, hierarchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hierarchy, max_area=1, min_area=0.000000003)
    return contours_imgs

def return_contours_of_image(image):

    if len(image.shape) == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image = image.astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def return_contours_of_interested_region_by_min_size(region_pre_p, pixel, min_size=0.00003):

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

    contours_imgs = return_parent_contours(contours_imgs, hierarchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hierarchy, max_area=1, min_area=min_size)

    return contours_imgs

def return_contours_of_interested_region_by_size(region_pre_p, pixel, min_area, max_area):

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

    contours_imgs = return_parent_contours(contours_imgs, hierarchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hierarchy, max_area=max_area, min_area=min_area)

    img_ret = np.zeros((region_pre_p.shape[0], region_pre_p.shape[1], 3))
    img_ret = cv2.fillPoly(img_ret, pts=contours_imgs, color=(1, 1, 1))
    return img_ret[:, :, 0]

