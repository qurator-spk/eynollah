import numpy as np
import cv2
from .contour import (
    find_new_features_of_contours,
    return_contours_of_image,
    return_parent_contours,
)

def adhere_drop_capital_region_into_corresponding_textline(
    text_regions_p,
    polygons_of_drop_capitals,
    contours_only_text_parent,
    contours_only_text_parent_h,
    all_box_coord,
    all_box_coord_h,
    all_found_texline_polygons,
    all_found_texline_polygons_h,
    kernel=None,
    curved_line=False,
):
    # print(np.shape(all_found_texline_polygons),np.shape(all_found_texline_polygons[3]),'all_found_texline_polygonsshape')
    # print(all_found_texline_polygons[3])
    cx_m, cy_m, _, _, _, _, _ = find_new_features_of_contours(contours_only_text_parent)
    cx_h, cy_h, _, _, _, _, _ = find_new_features_of_contours(contours_only_text_parent_h)
    cx_d, cy_d, _, _, y_min_d, y_max_d, _ = find_new_features_of_contours(polygons_of_drop_capitals)

    img_con_all = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
    for j_cont in range(len(contours_only_text_parent)):
        img_con_all[all_box_coord[j_cont][0] : all_box_coord[j_cont][1], all_box_coord[j_cont][2] : all_box_coord[j_cont][3], 0] = (j_cont + 1) * 3
        # img_con_all=cv2.fillPoly(img_con_all,pts=[contours_only_text_parent[j_cont]],color=((j_cont+1)*3,(j_cont+1)*3,(j_cont+1)*3))

    # plt.imshow(img_con_all[:,:,0])
    # plt.show()
    # img_con_all=cv2.dilate(img_con_all, kernel, iterations=3)

    # plt.imshow(img_con_all[:,:,0])
    # plt.show()
    # print(np.unique(img_con_all[:,:,0]))
    for i_drop in range(len(polygons_of_drop_capitals)):
        # print(i_drop,'i_drop')
        img_con_all_copy = np.copy(img_con_all)
        img_con = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
        img_con = cv2.fillPoly(img_con, pts=[polygons_of_drop_capitals[i_drop]], color=(1, 1, 1))

        # plt.imshow(img_con[:,:,0])
        # plt.show()
        ##img_con=cv2.dilate(img_con, kernel, iterations=30)

        # plt.imshow(img_con[:,:,0])
        # plt.show()

        # print(np.unique(img_con[:,:,0]))

        img_con_all_copy[:, :, 0] = img_con_all_copy[:, :, 0] + img_con[:, :, 0]

        img_con_all_copy[:, :, 0][img_con_all_copy[:, :, 0] == 1] = 0

        kherej_ghesmat = np.unique(img_con_all_copy[:, :, 0]) / 3
        res_summed_pixels = np.unique(img_con_all_copy[:, :, 0]) % 3
        region_with_intersected_drop = kherej_ghesmat[res_summed_pixels == 1]
        # region_with_intersected_drop=region_with_intersected_drop/3
        region_with_intersected_drop = region_with_intersected_drop.astype(np.uint8)

        # print(len(region_with_intersected_drop),'region_with_intersected_drop1')
        if len(region_with_intersected_drop) == 0:
            img_con_all_copy = np.copy(img_con_all)
            img_con = cv2.dilate(img_con, kernel, iterations=4)

            img_con_all_copy[:, :, 0] = img_con_all_copy[:, :, 0] + img_con[:, :, 0]

            img_con_all_copy[:, :, 0][img_con_all_copy[:, :, 0] == 1] = 0

            kherej_ghesmat = np.unique(img_con_all_copy[:, :, 0]) / 3
            res_summed_pixels = np.unique(img_con_all_copy[:, :, 0]) % 3
            region_with_intersected_drop = kherej_ghesmat[res_summed_pixels == 1]
            # region_with_intersected_drop=region_with_intersected_drop/3
            region_with_intersected_drop = region_with_intersected_drop.astype(np.uint8)
        # print(np.unique(img_con_all_copy[:,:,0]))
        if curved_line:

            if len(region_with_intersected_drop) > 1:
                sum_pixels_of_intersection = []
                for i in range(len(region_with_intersected_drop)):
                    # print((region_with_intersected_drop[i]*3+1))
                    sum_pixels_of_intersection.append(((img_con_all_copy[:, :, 0] == (region_with_intersected_drop[i] * 3 + 1)) * 1).sum())
                # print(sum_pixels_of_intersection)
                region_final = region_with_intersected_drop[np.argmax(sum_pixels_of_intersection)] - 1

                # print(region_final,'region_final')
                # cx_t,cy_t ,_, _, _ ,_,_= find_new_features_of_contours(all_found_texline_polygons[int(region_final)])
                try:
                    cx_t, cy_t, _, _, _, _, _ = find_new_features_of_contours(all_found_texline_polygons[int(region_final)])
                    # print(all_box_coord[j_cont])
                    # print(cx_t)
                    # print(cy_t)
                    # print(cx_d[i_drop])
                    # print(cy_d[i_drop])
                    y_lines = np.array(cy_t)  # all_box_coord[int(region_final)][0]+np.array(cy_t)

                    # print(y_lines)

                    y_lines[y_lines < y_min_d[i_drop]] = 0
                    # print(y_lines)

                    arg_min = np.argmin(np.abs(y_lines - y_min_d[i_drop]))
                    # print(arg_min)

                    cnt_nearest = np.copy(all_found_texline_polygons[int(region_final)][arg_min])
                    cnt_nearest[:, 0, 0] = all_found_texline_polygons[int(region_final)][arg_min][:, 0, 0]  # +all_box_coord[int(region_final)][2]
                    cnt_nearest[:, 0, 1] = all_found_texline_polygons[int(region_final)][arg_min][:, 0, 1]  # +all_box_coord[int(region_final)][0]

                    img_textlines = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
                    img_textlines = cv2.fillPoly(img_textlines, pts=[cnt_nearest], color=(255, 255, 255))
                    img_textlines = cv2.fillPoly(img_textlines, pts=[polygons_of_drop_capitals[i_drop]], color=(255, 255, 255))

                    img_textlines = img_textlines.astype(np.uint8)
                    imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                    contours_combined, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # print(len(contours_combined),'len textlines mixed')
                    areas_cnt_text = np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

                    contours_biggest = contours_combined[np.argmax(areas_cnt_text)]

                    # print(np.shape(contours_biggest))
                    # print(contours_biggest[:])
                    # contours_biggest[:,0,0]=contours_biggest[:,0,0]#-all_box_coord[int(region_final)][2]
                    # contours_biggest[:,0,1]=contours_biggest[:,0,1]#-all_box_coord[int(region_final)][0]

                    # contours_biggest=contours_biggest.reshape(np.shape(contours_biggest)[0],np.shape(contours_biggest)[2])

                    all_found_texline_polygons[int(region_final)][arg_min] = contours_biggest

                except:
                    # print('gordun1')
                    pass
            elif len(region_with_intersected_drop) == 1:
                region_final = region_with_intersected_drop[0] - 1

                # areas_main=np.array([cv2.contourArea(all_found_texline_polygons[int(region_final)][0][j] ) for j in range(len(all_found_texline_polygons[int(region_final)]))])

                # cx_t,cy_t ,_, _, _ ,_,_= find_new_features_of_contours(all_found_texline_polygons[int(region_final)])

                cx_t, cy_t, _, _, _, _, _ = find_new_features_of_contours(all_found_texline_polygons[int(region_final)])
                # print(all_box_coord[j_cont])
                # print(cx_t)
                # print(cy_t)
                # print(cx_d[i_drop])
                # print(cy_d[i_drop])
                y_lines = np.array(cy_t)  # all_box_coord[int(region_final)][0]+np.array(cy_t)

                y_lines[y_lines < y_min_d[i_drop]] = 0
                # print(y_lines)

                arg_min = np.argmin(np.abs(y_lines - y_min_d[i_drop]))
                # print(arg_min)

                cnt_nearest = np.copy(all_found_texline_polygons[int(region_final)][arg_min])
                cnt_nearest[:, 0, 0] = all_found_texline_polygons[int(region_final)][arg_min][:, 0, 0]  # +all_box_coord[int(region_final)][2]
                cnt_nearest[:, 0, 1] = all_found_texline_polygons[int(region_final)][arg_min][:, 0, 1]  # +all_box_coord[int(region_final)][0]

                img_textlines = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
                img_textlines = cv2.fillPoly(img_textlines, pts=[cnt_nearest], color=(255, 255, 255))
                img_textlines = cv2.fillPoly(img_textlines, pts=[polygons_of_drop_capitals[i_drop]], color=(255, 255, 255))

                img_textlines = img_textlines.astype(np.uint8)

                # plt.imshow(img_textlines)
                # plt.show()
                imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                contours_combined, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # print(len(contours_combined),'len textlines mixed')
                areas_cnt_text = np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

                contours_biggest = contours_combined[np.argmax(areas_cnt_text)]

                # print(np.shape(contours_biggest))
                # print(contours_biggest[:])
                # contours_biggest[:,0,0]=contours_biggest[:,0,0]#-all_box_coord[int(region_final)][2]
                # contours_biggest[:,0,1]=contours_biggest[:,0,1]#-all_box_coord[int(region_final)][0]
                # print(np.shape(contours_biggest),'contours_biggest')
                # print(np.shape(all_found_texline_polygons[int(region_final)][arg_min]))
                ##contours_biggest=contours_biggest.reshape(np.shape(contours_biggest)[0],np.shape(contours_biggest)[2])
                all_found_texline_polygons[int(region_final)][arg_min] = contours_biggest

                # print(cx_t,'print')
                try:
                    # print(all_found_texline_polygons[j_cont][0])
                    cx_t, cy_t, _, _, _, _, _ = find_new_features_of_contours(all_found_texline_polygons[int(region_final)])
                    # print(all_box_coord[j_cont])
                    # print(cx_t)
                    # print(cy_t)
                    # print(cx_d[i_drop])
                    # print(cy_d[i_drop])
                    y_lines = all_box_coord[int(region_final)][0] + np.array(cy_t)

                    y_lines[y_lines < y_min_d[i_drop]] = 0
                    # print(y_lines)

                    arg_min = np.argmin(np.abs(y_lines - y_min_d[i_drop]))
                    # print(arg_min)

                    cnt_nearest = np.copy(all_found_texline_polygons[int(region_final)][arg_min])
                    cnt_nearest[:, 0, 0] = all_found_texline_polygons[int(region_final)][arg_min][:, 0, 0]  # +all_box_coord[int(region_final)][2]
                    cnt_nearest[:, 0, 1] = all_found_texline_polygons[int(region_final)][arg_min][:, 0, 1]  # +all_box_coord[int(region_final)][0]

                    img_textlines = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
                    img_textlines = cv2.fillPoly(img_textlines, pts=[cnt_nearest], color=(255, 255, 255))
                    img_textlines = cv2.fillPoly(img_textlines, pts=[polygons_of_drop_capitals[i_drop]], color=(255, 255, 255))

                    img_textlines = img_textlines.astype(np.uint8)
                    imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                    contours_combined, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # print(len(contours_combined),'len textlines mixed')
                    areas_cnt_text = np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

                    contours_biggest = contours_combined[np.argmax(areas_cnt_text)]

                    # print(np.shape(contours_biggest))
                    # print(contours_biggest[:])
                    contours_biggest[:, 0, 0] = contours_biggest[:, 0, 0]  # -all_box_coord[int(region_final)][2]
                    contours_biggest[:, 0, 1] = contours_biggest[:, 0, 1]  # -all_box_coord[int(region_final)][0]

                    ##contours_biggest=contours_biggest.reshape(np.shape(contours_biggest)[0],np.shape(contours_biggest)[2])
                    all_found_texline_polygons[int(region_final)][arg_min] = contours_biggest
                    # all_found_texline_polygons[int(region_final)][arg_min]=contours_biggest

                except:
                    pass
            else:
                pass

            ##cx_t,cy_t ,_, _, _ ,_,_= find_new_features_of_contours(all_found_texline_polygons[int(region_final)])
            ###print(all_box_coord[j_cont])
            ###print(cx_t)
            ###print(cy_t)
            ###print(cx_d[i_drop])
            ###print(cy_d[i_drop])
            ##y_lines=all_box_coord[int(region_final)][0]+np.array(cy_t)

            ##y_lines[y_lines<y_min_d[i_drop]]=0
            ###print(y_lines)

            ##arg_min=np.argmin(np.abs(y_lines-y_min_d[i_drop])  )
            ###print(arg_min)

            ##cnt_nearest=np.copy(all_found_texline_polygons[int(region_final)][arg_min])
            ##cnt_nearest[:,0,0]=all_found_texline_polygons[int(region_final)][arg_min][:,0,0]#+all_box_coord[int(region_final)][2]
            ##cnt_nearest[:,0,1]=all_found_texline_polygons[int(region_final)][arg_min][:,0,1]#+all_box_coord[int(region_final)][0]

            ##img_textlines=np.zeros((text_regions_p.shape[0],text_regions_p.shape[1],3))
            ##img_textlines=cv2.fillPoly(img_textlines,pts=[cnt_nearest],color=(255,255,255))
            ##img_textlines=cv2.fillPoly(img_textlines,pts=[polygons_of_drop_capitals[i_drop] ],color=(255,255,255))

            ##img_textlines=img_textlines.astype(np.uint8)

            ##plt.imshow(img_textlines)
            ##plt.show()
            ##imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
            ##ret, thresh = cv2.threshold(imgray, 0, 255, 0)

            ##contours_combined,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            ##print(len(contours_combined),'len textlines mixed')
            ##areas_cnt_text=np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

            ##contours_biggest=contours_combined[np.argmax(areas_cnt_text)]

            ###print(np.shape(contours_biggest))
            ###print(contours_biggest[:])
            ##contours_biggest[:,0,0]=contours_biggest[:,0,0]#-all_box_coord[int(region_final)][2]
            ##contours_biggest[:,0,1]=contours_biggest[:,0,1]#-all_box_coord[int(region_final)][0]

            ##contours_biggest=contours_biggest.reshape(np.shape(contours_biggest)[0],np.shape(contours_biggest)[2])
            ##all_found_texline_polygons[int(region_final)][arg_min]=contours_biggest

        else:
            if len(region_with_intersected_drop) > 1:
                sum_pixels_of_intersection = []
                for i in range(len(region_with_intersected_drop)):
                    # print((region_with_intersected_drop[i]*3+1))
                    sum_pixels_of_intersection.append(((img_con_all_copy[:, :, 0] == (region_with_intersected_drop[i] * 3 + 1)) * 1).sum())
                # print(sum_pixels_of_intersection)
                region_final = region_with_intersected_drop[np.argmax(sum_pixels_of_intersection)] - 1

                # print(region_final,'region_final')
                # cx_t,cy_t ,_, _, _ ,_,_= find_new_features_of_contours(all_found_texline_polygons[int(region_final)])
                try:
                    cx_t, cy_t, _, _, _, _, _ = find_new_features_of_contours(all_found_texline_polygons[int(region_final)])
                    # print(all_box_coord[j_cont])
                    # print(cx_t)
                    # print(cy_t)
                    # print(cx_d[i_drop])
                    # print(cy_d[i_drop])
                    y_lines = all_box_coord[int(region_final)][0] + np.array(cy_t)

                    # print(y_lines)

                    y_lines[y_lines < y_min_d[i_drop]] = 0
                    # print(y_lines)

                    arg_min = np.argmin(np.abs(y_lines - y_min_d[i_drop]))
                    # print(arg_min)

                    cnt_nearest = np.copy(all_found_texline_polygons[int(region_final)][arg_min])
                    cnt_nearest[:, 0] = all_found_texline_polygons[int(region_final)][arg_min][:, 0] + all_box_coord[int(region_final)][2]
                    cnt_nearest[:, 1] = all_found_texline_polygons[int(region_final)][arg_min][:, 1] + all_box_coord[int(region_final)][0]

                    img_textlines = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
                    img_textlines = cv2.fillPoly(img_textlines, pts=[cnt_nearest], color=(255, 255, 255))
                    img_textlines = cv2.fillPoly(img_textlines, pts=[polygons_of_drop_capitals[i_drop]], color=(255, 255, 255))

                    img_textlines = img_textlines.astype(np.uint8)
                    imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                    contours_combined, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # print(len(contours_combined),'len textlines mixed')
                    areas_cnt_text = np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

                    contours_biggest = contours_combined[np.argmax(areas_cnt_text)]

                    # print(np.shape(contours_biggest))
                    # print(contours_biggest[:])
                    contours_biggest[:, 0, 0] = contours_biggest[:, 0, 0] - all_box_coord[int(region_final)][2]
                    contours_biggest[:, 0, 1] = contours_biggest[:, 0, 1] - all_box_coord[int(region_final)][0]

                    contours_biggest = contours_biggest.reshape(np.shape(contours_biggest)[0], np.shape(contours_biggest)[2])

                    all_found_texline_polygons[int(region_final)][arg_min] = contours_biggest

                except:
                    # print('gordun1')
                    pass
            elif len(region_with_intersected_drop) == 1:
                region_final = region_with_intersected_drop[0] - 1

                # areas_main=np.array([cv2.contourArea(all_found_texline_polygons[int(region_final)][0][j] ) for j in range(len(all_found_texline_polygons[int(region_final)]))])

                # cx_t,cy_t ,_, _, _ ,_,_= find_new_features_of_contours(all_found_texline_polygons[int(region_final)])

                # print(cx_t,'print')
                try:
                    # print(all_found_texline_polygons[j_cont][0])
                    cx_t, cy_t, _, _, _, _, _ = find_new_features_of_contours(all_found_texline_polygons[int(region_final)])
                    # print(all_box_coord[j_cont])
                    # print(cx_t)
                    # print(cy_t)
                    # print(cx_d[i_drop])
                    # print(cy_d[i_drop])
                    y_lines = all_box_coord[int(region_final)][0] + np.array(cy_t)

                    y_lines[y_lines < y_min_d[i_drop]] = 0
                    # print(y_lines)

                    arg_min = np.argmin(np.abs(y_lines - y_min_d[i_drop]))
                    # print(arg_min)

                    cnt_nearest = np.copy(all_found_texline_polygons[int(region_final)][arg_min])
                    cnt_nearest[:, 0] = all_found_texline_polygons[int(region_final)][arg_min][:, 0] + all_box_coord[int(region_final)][2]
                    cnt_nearest[:, 1] = all_found_texline_polygons[int(region_final)][arg_min][:, 1] + all_box_coord[int(region_final)][0]

                    img_textlines = np.zeros((text_regions_p.shape[0], text_regions_p.shape[1], 3))
                    img_textlines = cv2.fillPoly(img_textlines, pts=[cnt_nearest], color=(255, 255, 255))
                    img_textlines = cv2.fillPoly(img_textlines, pts=[polygons_of_drop_capitals[i_drop]], color=(255, 255, 255))

                    img_textlines = img_textlines.astype(np.uint8)
                    imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(imgray, 0, 255, 0)

                    contours_combined, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # print(len(contours_combined),'len textlines mixed')
                    areas_cnt_text = np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

                    contours_biggest = contours_combined[np.argmax(areas_cnt_text)]

                    # print(np.shape(contours_biggest))
                    # print(contours_biggest[:])
                    contours_biggest[:, 0, 0] = contours_biggest[:, 0, 0] - all_box_coord[int(region_final)][2]
                    contours_biggest[:, 0, 1] = contours_biggest[:, 0, 1] - all_box_coord[int(region_final)][0]

                    contours_biggest = contours_biggest.reshape(np.shape(contours_biggest)[0], np.shape(contours_biggest)[2])
                    all_found_texline_polygons[int(region_final)][arg_min] = contours_biggest
                    # all_found_texline_polygons[int(region_final)][arg_min]=contours_biggest

                except:
                    pass
            else:
                pass

    #####for i_drop in range(len(polygons_of_drop_capitals)):
    #####for j_cont in range(len(contours_only_text_parent)):
    #####img_con=np.zeros((text_regions_p.shape[0],text_regions_p.shape[1],3))
    #####img_con=cv2.fillPoly(img_con,pts=[polygons_of_drop_capitals[i_drop] ],color=(255,255,255))
    #####img_con=cv2.fillPoly(img_con,pts=[contours_only_text_parent[j_cont]],color=(255,255,255))

    #####img_con=img_con.astype(np.uint8)
    ######imgray = cv2.cvtColor(img_con, cv2.COLOR_BGR2GRAY)
    ######ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    ######contours_new,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #####contours_new,hir_new=return_contours_of_image(img_con)
    #####contours_new_parent=return_parent_contours( contours_new,hir_new)
    ######plt.imshow(img_con)
    ######plt.show()
    #####try:
    #####if len(contours_new_parent)==1:
    ######print(all_found_texline_polygons[j_cont][0])
    #####cx_t,cy_t ,_, _, _ ,_,_= find_new_features_of_contours(all_found_texline_polygons[j_cont])
    ######print(all_box_coord[j_cont])
    ######print(cx_t)
    ######print(cy_t)
    ######print(cx_d[i_drop])
    ######print(cy_d[i_drop])
    #####y_lines=all_box_coord[j_cont][0]+np.array(cy_t)

    ######print(y_lines)

    #####arg_min=np.argmin(np.abs(y_lines-y_min_d[i_drop])  )
    ######print(arg_min)

    #####cnt_nearest=np.copy(all_found_texline_polygons[j_cont][arg_min])
    #####cnt_nearest[:,0]=all_found_texline_polygons[j_cont][arg_min][:,0]+all_box_coord[j_cont][2]
    #####cnt_nearest[:,1]=all_found_texline_polygons[j_cont][arg_min][:,1]+all_box_coord[j_cont][0]

    #####img_textlines=np.zeros((text_regions_p.shape[0],text_regions_p.shape[1],3))
    #####img_textlines=cv2.fillPoly(img_textlines,pts=[cnt_nearest],color=(255,255,255))
    #####img_textlines=cv2.fillPoly(img_textlines,pts=[polygons_of_drop_capitals[i_drop] ],color=(255,255,255))

    #####img_textlines=img_textlines.astype(np.uint8)
    #####imgray = cv2.cvtColor(img_textlines, cv2.COLOR_BGR2GRAY)
    #####ret, thresh = cv2.threshold(imgray, 0, 255, 0)

    #####contours_combined,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #####areas_cnt_text=np.array([cv2.contourArea(contours_combined[j]) for j in range(len(contours_combined))])

    #####contours_biggest=contours_combined[np.argmax(areas_cnt_text)]

    ######print(np.shape(contours_biggest))
    ######print(contours_biggest[:])
    #####contours_biggest[:,0,0]=contours_biggest[:,0,0]-all_box_coord[j_cont][2]
    #####contours_biggest[:,0,1]=contours_biggest[:,0,1]-all_box_coord[j_cont][0]

    #####all_found_texline_polygons[j_cont][arg_min]=contours_biggest
    ######print(contours_biggest)
    ######plt.imshow(img_textlines[:,:,0])
    ######plt.show()
    #####else:
    #####pass
    #####except:
    #####pass
    return all_found_texline_polygons

def filter_small_drop_capitals_from_no_patch_layout(layout_no_patch, layout1):

    drop_only = (layout_no_patch[:, :, 0] == 4) * 1
    contours_drop, hir_on_drop = return_contours_of_image(drop_only)
    contours_drop_parent = return_parent_contours(contours_drop, hir_on_drop)

    areas_cnt_text = np.array([cv2.contourArea(contours_drop_parent[j]) for j in range(len(contours_drop_parent))])
    areas_cnt_text = areas_cnt_text / float(drop_only.shape[0] * drop_only.shape[1])

    contours_drop_parent = [contours_drop_parent[jz] for jz in range(len(contours_drop_parent)) if areas_cnt_text[jz] > 0.001]

    areas_cnt_text = [areas_cnt_text[jz] for jz in range(len(areas_cnt_text)) if areas_cnt_text[jz] > 0.001]

    contours_drop_parent_final = []

    for jj in range(len(contours_drop_parent)):
        x, y, w, h = cv2.boundingRect(contours_drop_parent[jj])
        # boxes.append([int(x), int(y), int(w), int(h)])

        iou_of_box_and_contoure = float(drop_only.shape[0] * drop_only.shape[1]) * areas_cnt_text[jj] / float(w * h) * 100
        height_to_weight_ratio = h / float(w)
        weigh_to_height_ratio = w / float(h)

        if iou_of_box_and_contoure > 60 and weigh_to_height_ratio < 1.2 and height_to_weight_ratio < 2:
            map_of_drop_contour_bb = np.zeros((layout1.shape[0], layout1.shape[1]))
            map_of_drop_contour_bb[y : y + h, x : x + w] = layout1[y : y + h, x : x + w]

            if (((map_of_drop_contour_bb == 1) * 1).sum() / float(((map_of_drop_contour_bb == 5) * 1).sum()) * 100) >= 15:
                contours_drop_parent_final.append(contours_drop_parent[jj])

    layout_no_patch[:, :, 0][layout_no_patch[:, :, 0] == 4] = 0

    layout_no_patch = cv2.fillPoly(layout_no_patch, pts=contours_drop_parent_final, color=(4, 4, 4))

    return layout_no_patch

