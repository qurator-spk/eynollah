import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os.path
import cv2
from scipy.ndimage import gaussian_filter1d

from .utils import crop_image_inside_box
from .utils.rotate import rotate_image_different
from .utils.resize import resize_image

class EynollahPlotter():
    """
    Class collecting all the plotting and image writing methods
    """

    def __init__(
        self,
        *,
        dir_of_all,
        dir_of_deskewed,
        dir_of_layout,
        dir_of_cropped_images,
        image_filename_stem,
        image_org=None,
        scale_x=1,
        scale_y=1,
    ):
        self.dir_of_all = dir_of_all
        self.dir_of_layout = dir_of_layout
        self.dir_of_cropped_images = dir_of_cropped_images
        self.dir_of_deskewed = dir_of_deskewed
        self.image_filename_stem = image_filename_stem
        # XXX TODO hacky these cannot be set at init time
        self.image_org = image_org
        self.scale_x = scale_x
        self.scale_y = scale_y

    def save_plot_of_layout_main(self, text_regions_p, image_page):
        if self.dir_of_layout is not None:
            values = np.unique(text_regions_p[:, :])
            # pixels=['Background' , 'Main text' , 'Heading' , 'Marginalia' ,'Drop capitals' , 'Images' , 'Seperators' , 'Tables', 'Graphics']
            pixels=['Background' , 'Main text'  , 'Image' , 'Separator','Marginalia']
            values_indexes = [0, 1, 2, 3, 4]
            plt.figure(figsize=(40, 40))
            plt.rcParams["font.size"] = "40"
            im = plt.imshow(text_regions_p[:, :])
            colors = [im.cmap(im.norm(value)) for value in values]
            patches = [mpatches.Patch(color=colors[np.where(values == i)[0][0]], label="{l}".format(l=pixels[int(np.where(values_indexes == i)[0][0])])) for i in values]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=40)
            plt.savefig(os.path.join(self.dir_of_layout, self.image_filename_stem + "_layout_main.png"))
        

    def save_plot_of_layout_main_all(self, text_regions_p, image_page):
        if self.dir_of_all is not None:
            values = np.unique(text_regions_p[:, :])
            # pixels=['Background' , 'Main text' , 'Heading' , 'Marginalia' ,'Drop capitals' , 'Images' , 'Seperators' , 'Tables', 'Graphics']
            pixels=['Background' , 'Main text'  , 'Image' , 'Separator','Marginalia']
            values_indexes = [0, 1, 2, 3, 4]
            plt.figure(figsize=(80, 40))
            plt.rcParams["font.size"] = "40"
            plt.subplot(1, 2, 1)
            plt.imshow(image_page)
            plt.subplot(1, 2, 2)
            im = plt.imshow(text_regions_p[:, :])
            colors = [im.cmap(im.norm(value)) for value in values]
            patches = [mpatches.Patch(color=colors[np.where(values == i)[0][0]], label="{l}".format(l=pixels[int(np.where(values_indexes == i)[0][0])])) for i in values]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=60)
            plt.savefig(os.path.join(self.dir_of_all, self.image_filename_stem + "_layout_main_and_page.png"))

    def save_plot_of_layout(self, text_regions_p, image_page):
        if self.dir_of_layout is not None:
            values = np.unique(text_regions_p[:, :])
            # pixels=['Background' , 'Main text' , 'Heading' , 'Marginalia' ,'Drop capitals' , 'Images' , 'Seperators' , 'Tables', 'Graphics']
            pixels = ["Background", "Main text", "Header", "Marginalia", "Drop capital", "Image", "Separator"]
            values_indexes = [0, 1, 2, 8, 4, 5, 6]
            plt.figure(figsize=(40, 40))
            plt.rcParams["font.size"] = "40"
            im = plt.imshow(text_regions_p[:, :])
            colors = [im.cmap(im.norm(value)) for value in values]
            patches = [mpatches.Patch(color=colors[np.where(values == i)[0][0]], label="{l}".format(l=pixels[int(np.where(values_indexes == i)[0][0])])) for i in values]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=40)
            plt.savefig(os.path.join(self.dir_of_layout, self.image_filename_stem + "_layout.png"))

    def save_plot_of_layout_all(self, text_regions_p, image_page):
        if self.dir_of_all is not None:
            values = np.unique(text_regions_p[:, :])
            # pixels=['Background' , 'Main text' , 'Heading' , 'Marginalia' ,'Drop capitals' , 'Images' , 'Seperators' , 'Tables', 'Graphics']
            pixels = ["Background", "Main text", "Header", "Marginalia", "Drop capital", "Image", "Separator"]
            values_indexes = [0, 1, 2, 8, 4, 5, 6]
            plt.figure(figsize=(80, 40))
            plt.rcParams["font.size"] = "40"
            plt.subplot(1, 2, 1)
            plt.imshow(image_page)
            plt.subplot(1, 2, 2)
            im = plt.imshow(text_regions_p[:, :])
            colors = [im.cmap(im.norm(value)) for value in values]
            patches = [mpatches.Patch(color=colors[np.where(values == i)[0][0]], label="{l}".format(l=pixels[int(np.where(values_indexes == i)[0][0])])) for i in values]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=60)
            plt.savefig(os.path.join(self.dir_of_all, self.image_filename_stem + "_layout_and_page.png"))

    def save_plot_of_textlines(self, textline_mask_tot_ea, image_page):
        if self.dir_of_all is not None:
            values = np.unique(textline_mask_tot_ea[:, :])
            pixels = ["Background", "Textlines"]
            values_indexes = [0, 1]
            plt.figure(figsize=(80, 40))
            plt.rcParams["font.size"] = "40"
            plt.subplot(1, 2, 1)
            plt.imshow(image_page)
            plt.subplot(1, 2, 2)
            im = plt.imshow(textline_mask_tot_ea[:, :])
            colors = [im.cmap(im.norm(value)) for value in values]
            patches = [mpatches.Patch(color=colors[np.where(values == i)[0][0]], label="{l}".format(l=pixels[int(np.where(values_indexes == i)[0][0])])) for i in values]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize=60)
            plt.savefig(os.path.join(self.dir_of_all, self.image_filename_stem + "_textline_and_page.png"))

    def save_deskewed_image(self, slope_deskew):
        if self.dir_of_all is not None:
            cv2.imwrite(os.path.join(self.dir_of_all, self.image_filename_stem + "_org.png"), self.image_org)
        if self.dir_of_deskewed is not None:
            img_rotated = rotate_image_different(self.image_org, slope_deskew)
            cv2.imwrite(os.path.join(self.dir_of_deskewed, self.image_filename_stem + "_deskewed.png"), img_rotated)

    def save_page_image(self, image_page):
        if self.dir_of_all is not None:
            cv2.imwrite(os.path.join(self.dir_of_all, self.image_filename_stem + "_page.png"), image_page)

    def save_plot_of_textline_density(self, img_patch_org):
        if self.dir_of_all is not None:
            plt.figure(figsize=(80,40))
            plt.rcParams['font.size']='50'
            plt.subplot(1,2,1)
            plt.imshow(img_patch_org)
            plt.subplot(1,2,2)
            plt.plot(gaussian_filter1d(img_patch_org.sum(axis=1), 3),np.array(range(len(gaussian_filter1d(img_patch_org.sum(axis=1), 3)))),linewidth=8)
            plt.xlabel('Density of textline prediction in direction of X axis',fontsize=60)
            plt.ylabel('Height',fontsize=60)
            plt.yticks([0,len(gaussian_filter1d(img_patch_org.sum(axis=1), 3))])
            plt.gca().invert_yaxis()
            plt.savefig(os.path.join(self.dir_of_all, self.image_filename_stem+'_density_of_textline.png'))

    def save_plot_of_rotation_angle(self, angels, var_res):
        if self.dir_of_all is not None:
            plt.figure(figsize=(60,30))
            plt.rcParams['font.size']='50'
            plt.plot(angels,np.array(var_res),'-o',markersize=25,linewidth=4)
            plt.xlabel('angle',fontsize=50)
            plt.ylabel('variance of sum of rotated textline in direction of x axis',fontsize=50)
            plt.plot(angels[np.argmax(var_res)],var_res[np.argmax(np.array(var_res))]  ,'*',markersize=50,label='Angle of deskewing=' +str("{:.2f}".format(angels[np.argmax(var_res)]))+r'$\degree$')
            plt.legend(loc='best')
            plt.savefig(os.path.join(self.dir_of_all, self.image_filename_stem+'_rotation_angle.png'))

    def write_images_into_directory(self, img_contours, image_page):
        if self.dir_of_cropped_images is not None:
            index = 0
            for cont_ind in img_contours:
                x, y, w, h = cv2.boundingRect(cont_ind)
                box = [x, y, w, h]
                croped_page, page_coord = crop_image_inside_box(box, image_page)

                croped_page = resize_image(croped_page, int(croped_page.shape[0] / self.scale_y), int(croped_page.shape[1] / self.scale_x))

                path = os.path.join(self.dir_of_cropped_images, self.image_filename_stem + "_" + str(index) + ".jpg")
                cv2.imwrite(path, croped_page)
                index += 1

