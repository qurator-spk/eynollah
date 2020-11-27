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
    return np.unique(all_args)

def find_contours_mean_y_diff(contours_main):
    M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
    cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
    return np.mean(np.diff(np.sort(np.array(cy_main))))

def find_features_of_contours(contours_main):

    areas_main = np.array([cv2.contourArea(contours_main[j]) for j in range(len(contours_main))])
    M_main = [cv2.moments(contours_main[j]) for j in range(len(contours_main))]
    cx_main = [(M_main[j]["m10"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
    cy_main = [(M_main[j]["m01"] / (M_main[j]["m00"] + 1e-32)) for j in range(len(M_main))]
    x_min_main = np.array([np.min(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])
    x_max_main = np.array([np.max(contours_main[j][:, 0, 0]) for j in range(len(contours_main))])

    y_min_main = np.array([np.min(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])
    y_max_main = np.array([np.max(contours_main[j][:, 0, 1]) for j in range(len(contours_main))])

    return y_min_main, y_max_main, areas_main

def return_contours_of_interested_region_and_bounding_box(region_pre_p, pixel):

    # pixels of images are identified by 5
    cnts_images = (region_pre_p[:, :, 0] == pixel) * 1
    cnts_images = cnts_images.astype(np.uint8)
    cnts_images = np.repeat(cnts_images[:, :, np.newaxis], 3, axis=2)
    imgray = cv2.cvtColor(cnts_images, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hiearchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=1, min_area=0.0003)

    boxes = []

    for jj in range(len(contours_imgs)):
        x, y, w, h = cv2.boundingRect(contours_imgs[jj])
        boxes.append([int(x), int(y), int(w), int(h)])
    return contours_imgs, boxes

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

def filter_contours_area_of_image(image, contours, hirarchy, max_area, min_area):
    found_polygons_early = list()

    jv = 0
    for c in contours:
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue

        polygon = geometry.Polygon([point[0] for point in c])
        area = polygon.area
        if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]) and hirarchy[0][jv][3] == -1:  # and hirarchy[0][jv][3]==-1 :
            found_polygons_early.append(np.array([[point] for point in polygon.exterior.coords], dtype=np.uint))
        jv += 1
    return found_polygons_early

def filter_contours_area_of_image_interiors(image, contours, hirarchy, max_area, min_area):
    found_polygons_early = list()

    jv = 0
    for c in contours:
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue

        polygon = geometry.Polygon([point[0] for point in c])
        area = polygon.area
        if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]) and hirarchy[0][jv][3] != -1:
            # print(c[0][0][1])
            found_polygons_early.append(np.array([point for point in polygon.exterior.coords], dtype=np.uint))
        jv += 1
    return found_polygons_early


def filter_contours_area_of_image_tables(image, contours, hirarchy, max_area, min_area):
    found_polygons_early = list()

    jv = 0
    for c in contours:
        if len(c) < 3:  # A polygon cannot have less than 3 points
            continue

        polygon = geometry.Polygon([point[0] for point in c])
        # area = cv2.contourArea(c)
        area = polygon.area
        ##print(np.prod(thresh.shape[:2]))
        # Check that polygon has area greater than minimal area
        # print(hirarchy[0][jv][3],hirarchy )
        if area >= min_area * np.prod(image.shape[:2]) and area <= max_area * np.prod(image.shape[:2]):  # and hirarchy[0][jv][3]==-1 :
            # print(c[0][0][1])
            found_polygons_early.append(np.array([[point] for point in polygon.exterior.coords], dtype=np.int32))
        jv += 1
    return found_polygons_early

def find_new_features_of_contoures(contours_main):

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

    contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hiearchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=1, min_area=min_area)

    return contours_imgs

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

    # print(cnts_org,'cnts_org')

    # sys.exit()
    # self.y_shift = np.abs(img_copy.shape[0] - img.shape[0])
    # self.x_shift = np.abs(img_copy.shape[1] - img.shape[1])
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
    contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hiearchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=1, min_area=0.000000003)
    return contours_imgs

def return_bonding_box_of_contours(cnts):
    boxes_tot = []
    for i in range(len(cnts)):
        x, y, w, h = cv2.boundingRect(cnts[i])

        box = [x, y, w, h]
        boxes_tot.append(box)
    return boxes_tot

def return_contours_of_image(image):

    if len(image.shape) == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image = image.astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0)
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierachy

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

    contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hiearchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=1, min_area=min_size)

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
    contours_imgs, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_imgs = return_parent_contours(contours_imgs, hiearchy)
    contours_imgs = filter_contours_area_of_image_tables(thresh, contours_imgs, hiearchy, max_area=max_area, min_area=min_area)

    img_ret = np.zeros((region_pre_p.shape[0], region_pre_p.shape[1], 3))
    img_ret = cv2.fillPoly(img_ret, pts=contours_imgs, color=(1, 1, 1))
    return img_ret[:, :, 0]

def textline_contours_postprocessing(textline_mask, slope, contour_text_interest, box_ind, slope_first, add_boxes_coor_into_textlines=False):

    textline_mask = np.repeat(textline_mask[:, :, np.newaxis], 3, axis=2) * 255
    textline_mask = textline_mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_OPEN, kernel)
    textline_mask = cv2.morphologyEx(textline_mask, cv2.MORPH_CLOSE, kernel)
    textline_mask = cv2.erode(textline_mask, kernel, iterations=2)
    # textline_mask = cv2.erode(textline_mask, kernel, iterations=1)

    # print(textline_mask.shape[0]/float(textline_mask.shape[1]),'miz')
    try:
        # if np.abs(slope)>.5 and textline_mask.shape[0]/float(textline_mask.shape[1])>3:
        # plt.imshow(textline_mask)
        # plt.show()

        # if abs(slope)>1:
        # x_help=30
        # y_help=2
        # else:
        # x_help=2
        # y_help=2

        x_help = 30
        y_help = 2

        textline_mask_help = np.zeros((textline_mask.shape[0] + int(2 * y_help), textline_mask.shape[1] + int(2 * x_help), 3))
        textline_mask_help[y_help : y_help + textline_mask.shape[0], x_help : x_help + textline_mask.shape[1], :] = np.copy(textline_mask[:, :, :])

        dst = rotate_image(textline_mask_help, slope)
        dst = dst[:, :, 0]
        dst[dst != 0] = 1

        # if np.abs(slope)>.5 and textline_mask.shape[0]/float(textline_mask.shape[1])>3:
        # plt.imshow(dst)
        # plt.show()

        contour_text_copy = contour_text_interest.copy()

        contour_text_copy[:, 0, 0] = contour_text_copy[:, 0, 0] - box_ind[0]
        contour_text_copy[:, 0, 1] = contour_text_copy[:, 0, 1] - box_ind[1]

        img_contour = np.zeros((box_ind[3], box_ind[2], 3))
        img_contour = cv2.fillPoly(img_contour, pts=[contour_text_copy], color=(255, 255, 255))

        # if np.abs(slope)>.5 and textline_mask.shape[0]/float(textline_mask.shape[1])>3:
        # plt.imshow(img_contour)
        # plt.show()

        img_contour_help = np.zeros((img_contour.shape[0] + int(2 * y_help), img_contour.shape[1] + int(2 * x_help), 3))

        img_contour_help[y_help : y_help + img_contour.shape[0], x_help : x_help + img_contour.shape[1], :] = np.copy(img_contour[:, :, :])

        img_contour_rot = rotate_image(img_contour_help, slope)

        # plt.imshow(img_contour_rot_help)
        # plt.show()

        # plt.imshow(dst_help)
        # plt.show()

        # if np.abs(slope)>.5 and textline_mask.shape[0]/float(textline_mask.shape[1])>3:
        # plt.imshow(img_contour_rot_help)
        # plt.show()

        # plt.imshow(dst_help)
        # plt.show()

        img_contour_rot = img_contour_rot.astype(np.uint8)
        # dst_help = dst_help.astype(np.uint8)
        imgrayrot = cv2.cvtColor(img_contour_rot, cv2.COLOR_BGR2GRAY)
        _, threshrot = cv2.threshold(imgrayrot, 0, 255, 0)
        contours_text_rot, _ = cv2.findContours(threshrot.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        len_con_text_rot = [len(contours_text_rot[ib]) for ib in range(len(contours_text_rot))]
        ind_big_con = np.argmax(len_con_text_rot)

        # print('juzaa')
        if abs(slope) > 45:
            # print(add_boxes_coor_into_textlines,'avval')
            _, contours_rotated_clean = seperate_lines_vertical_cont(textline_mask, contours_text_rot[ind_big_con], box_ind, slope, add_boxes_coor_into_textlines=add_boxes_coor_into_textlines)
        else:
            _, contours_rotated_clean = seperate_lines(dst, contours_text_rot[ind_big_con], slope, x_help, y_help)

    except:

        contours_rotated_clean = []

    return contours_rotated_clean

