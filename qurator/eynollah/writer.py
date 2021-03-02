# pylint: disable=too-many-locals,wrong-import-position,too-many-lines,too-many-statements,chained-comparison,fixme,broad-except,c-extension-no-member
from pathlib import Path
import os.path

from .utils.xml import create_page_xml, add_textequiv, xml_reading_order

from ocrd_utils import getLogger
from lxml import etree as ET
import numpy as np

class EynollahXmlWriter():

    def __init__(self, *, dir_out, image_filename, curved_line):
        self.logger = getLogger('eynollah.writer')
        self.dir_out = dir_out
        self.image_filename = image_filename
        self.image_filename_stem = Path(Path(image_filename).name).stem
        self.curved_line = curved_line
        self.scale_x = None # XXX set outside __init__
        self.scale_y = None # XXX set outside __init__
        self.height_org = None # XXX set outside __init__
        self.width_org = None # XXX set outside __init__

    def calculate_page_coords(self, cont_page):
        self.logger.debug('enter calculate_page_coords')
        points_page_print = ""
        for _, contour in enumerate(cont_page[0]):
            if len(contour) == 2:
                points_page_print += str(int((contour[0]) / self.scale_x))
                points_page_print += ','
                points_page_print += str(int((contour[1]) / self.scale_y))
            else:
                points_page_print += str(int((contour[0][0]) / self.scale_x))
                points_page_print += ','
                points_page_print += str(int((contour[0][1] ) / self.scale_y))
            points_page_print = points_page_print + ' '
        return points_page_print[:-1]

    def serialize_lines_in_marginal(self, marginal, all_found_texline_polygons_marginals, marginal_idx, page_coord, all_box_coord_marginals, slopes_marginals, id_indexer_l):
        for j in range(len(all_found_texline_polygons_marginals[marginal_idx])):
            textline = ET.SubElement(marginal, 'TextLine')
            textline.set('id', 'l%s' % id_indexer_l)
            id_indexer_l += 1
            coord = ET.SubElement(textline, 'Coords')
            add_textequiv(textline)
            points_co = ''
            for l in range(len(all_found_texline_polygons_marginals[marginal_idx][j])):
                if not self.curved_line:
                    if len(all_found_texline_polygons_marginals[marginal_idx][j][l]) == 2:
                        points_co += str(int((all_found_texline_polygons_marginals[marginal_idx][j][l][0] + all_box_coord_marginals[marginal_idx][2] + page_coord[2]) / self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons_marginals[marginal_idx][j][l][1] + all_box_coord_marginals[marginal_idx][0] + page_coord[0]) / self.scale_y))
                    else:
                        points_co += str(int((all_found_texline_polygons_marginals[marginal_idx][j][l][0][0] + all_box_coord_marginals[marginal_idx][2] + page_coord[2]) / self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons_marginals[marginal_idx][j][l][0][1] + all_box_coord_marginals[marginal_idx][0] + page_coord[0])/self.scale_y))
                if self.curved_line and np.abs(slopes_marginals[marginal_idx]) <= 45:
                    if len(all_found_texline_polygons_marginals[marginal_idx][j][l]) == 2:
                        points_co += str(int((all_found_texline_polygons_marginals[marginal_idx][j][l][0] + page_coord[2]) / self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons_marginals[marginal_idx][j][l][1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co += str(int((all_found_texline_polygons_marginals[marginal_idx][j][l][0][0] + page_coord[2]) / self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons_marginals[marginal_idx][j][l][0][1] + page_coord[0]) / self.scale_y))
                        
                elif self.curved_line and np.abs(slopes_marginals[marginal_idx]) > 45:
                    if len(all_found_texline_polygons_marginals[marginal_idx][j][l]) == 2:
                        points_co += str(int((all_found_texline_polygons_marginals[marginal_idx][j][l][0] + all_box_coord_marginals[marginal_idx][2] + page_coord[2]) / self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons_marginals[marginal_idx][j][l][1] + all_box_coord_marginals[marginal_idx][0] + page_coord[0]) / self.scale_y))
                    else:
                        points_co += str(int((all_found_texline_polygons_marginals[marginal_idx][j][l][0][0] + all_box_coord_marginals[marginal_idx][2] + page_coord[2]) / self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons_marginals[marginal_idx][j][l][0][1] + all_box_coord_marginals[marginal_idx][0] + page_coord[0]) / self.scale_y))

                if l < len(all_found_texline_polygons_marginals[marginal_idx][j]) - 1:
                    points_co += ' '
            coord.set('points',points_co)
        return id_indexer_l

    def serialize_lines_in_region(self, textregion, all_found_texline_polygons, region_idx, page_coord, all_box_coord, slopes, id_indexer_l):
        self.logger.debug('enter serialize_lines_in_region')
        for j in range(len(all_found_texline_polygons[region_idx])):
            textline = ET.SubElement(textregion, 'TextLine')
            textline.set('id', 'l%s' % id_indexer_l)
            id_indexer_l += 1
            coord = ET.SubElement(textline, 'Coords')
            add_textequiv(textline)

            points_co = ''
            for l in range(len(all_found_texline_polygons[region_idx][j])):
                if not self.curved_line:
                    if len(all_found_texline_polygons[region_idx][j][l])==2:
                        textline_x_coord = max(0, int((all_found_texline_polygons[region_idx][j][l][0] + all_box_coord[region_idx][2] + page_coord[2]) / self.scale_x))
                        textline_y_coord = max(0, int((all_found_texline_polygons[region_idx][j][l][1] + all_box_coord[region_idx][0] + page_coord[0]) / self.scale_y))
                    else:
                        textline_x_coord = max(0, int((all_found_texline_polygons[region_idx][j][l][0][0] + all_box_coord[region_idx][2] + page_coord[2]) / self.scale_x))
                        textline_y_coord = max(0, int((all_found_texline_polygons[region_idx][j][l][0][1] + all_box_coord[region_idx][0] + page_coord[0]) / self.scale_y))
                    points_co += str(textline_x_coord)
                    points_co += ','
                    points_co += str(textline_y_coord)

                if self.curved_line and np.abs(slopes[region_idx]) <= 45:
                    if len(all_found_texline_polygons[region_idx][j][l]) == 2:
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][0] + page_coord[2]) / self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][0][0] + page_coord[2]) / self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][0][1] + page_coord[0])/self.scale_y))
                elif self.curved_line and np.abs(slopes[region_idx]) > 45:
                    if len(all_found_texline_polygons[region_idx][j][l])==2:
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][0] + all_box_coord[region_idx][2]+page_coord[2])/self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][1] + all_box_coord[region_idx][0]+page_coord[0])/self.scale_y))
                    else:
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][0][0] + all_box_coord[region_idx][2]+page_coord[2])/self.scale_x))
                        points_co += ','
                        points_co += str(int((all_found_texline_polygons[region_idx][j][l][0][1] + all_box_coord[region_idx][0]+page_coord[0])/self.scale_y))

                if l < len(all_found_texline_polygons[region_idx][j]) - 1:
                    points_co += ' '
            coord.set('points',points_co)
        return id_indexer_l

    def write_pagexml(self, pcgts):
        self.logger.info("filename stem: '%s'", self.image_filename_stem)
        tree = ET.ElementTree(pcgts)
        tree.write(os.path.join(self.dir_out, self.image_filename_stem) + ".xml")

    def build_pagexml_no_full_layout(self, found_polygons_text_region, page_coord, order_of_texts, id_of_texts, all_found_texline_polygons, all_box_coord, found_polygons_text_region_img, found_polygons_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals, cont_page):
        self.logger.debug('enter build_pagexml_no_full_layout')

        # create the file structure
        pcgts, page = create_page_xml(self.image_filename, self.height_org, self.width_org)
        page_print_sub = ET.SubElement(page, "Border")
        coord_page = ET.SubElement(page_print_sub, "Coords")
        coord_page.set('points', self.calculate_page_coords(cont_page))

        id_of_marginalia = []
        for  idx_marginal, _ in enumerate(found_polygons_marginals):
            id_of_marginalia.append('r%s' % len(order_of_texts) + idx_marginal)

        id_indexer = 0
        id_indexer_l = 0

        if len(found_polygons_text_region) > 0:
            xml_reading_order(page, order_of_texts, id_of_texts, found_polygons_marginals)
            for mm in range(len(found_polygons_text_region)):
                textregion = ET.SubElement(page, 'TextRegion')
                textregion.set('id', 'r%s' % id_indexer)
                id_indexer += 1
                textregion.set('type', 'paragraph')
                coord_text = ET.SubElement(textregion, 'Coords')
                coord_text.set('points', self.calculate_polygon_coords(found_polygons_text_region, mm, page_coord))
                id_indexer_l = self.serialize_lines_in_region(textregion, all_found_texline_polygons, mm, page_coord, all_box_coord, slopes, id_indexer_l)
                add_textequiv(textregion)

        for mm in range(len(found_polygons_marginals)):
            marginal = ET.SubElement(page, 'TextRegion')
            marginal.set('id', id_of_marginalia[mm])
            marginal.set('type', 'marginalia')
            coord_text = ET.SubElement(marginal, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_marginals, mm, page_coord))
            id_indexer_l = self.serialize_lines_in_marginal(marginal, all_found_texline_polygons_marginals, mm, page_coord, all_box_coord_marginals, slopes_marginals, id_indexer_l)

        id_indexer = len(found_polygons_text_region) + len(found_polygons_marginals)
        for mm in range(len(found_polygons_text_region_img)):
            textregion = ET.SubElement(page, 'ImageRegion')
            textregion.set('id', 'r%s' % id_indexer)
            id_indexer += 1
            coord_text = ET.SubElement(textregion, 'Coords')
            points_co = ''
            for lmm in range(len(found_polygons_text_region_img[mm])):
                points_co += str(int((found_polygons_text_region_img[mm][lmm,0,0] + page_coord[2]) / self.scale_x))
                points_co += ','
                points_co += str(int((found_polygons_text_region_img[mm][lmm,0,1] + page_coord[0]) / self.scale_y))
                if lmm < len(found_polygons_text_region_img[mm]) - 1:
                    points_co += ' '
            coord_text.set('points', points_co)

        return pcgts

    def build_pagexml_full_layout(self, found_polygons_text_region, found_polygons_text_region_h, page_coord, order_of_texts, id_of_texts, all_found_texline_polygons, all_found_texline_polygons_h, all_box_coord, all_box_coord_h, found_polygons_text_region_img, found_polygons_tables, found_polygons_drop_capitals, found_polygons_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals, cont_page):
        self.logger.debug('enter build_pagexml_full_layout')

        # create the file structure
        pcgts, page = create_page_xml(self.image_filename, self.height_org, self.width_org)
        page_print_sub = ET.SubElement(page, "Border")
        coord_page = ET.SubElement(page_print_sub, "Coords")
        coord_page.set('points', self.calculate_page_coords(cont_page))

        id_indexer = 0
        id_indexer_l = 0
        id_of_marginalia = []
        for  idx_marginal, _ in enumerate(found_polygons_marginals):
            id_of_marginalia.append('r%s' % len(order_of_texts) + idx_marginal)

        if len(found_polygons_text_region) > 0:
            xml_reading_order(page, order_of_texts, id_of_texts, found_polygons_marginals)
            for mm in range(len(found_polygons_text_region)):
                textregion=ET.SubElement(page, 'TextRegion')
                textregion.set('id', 'r%s' % id_indexer)
                id_indexer += 1
                textregion.set('type', 'paragraph')
                coord_text = ET.SubElement(textregion, 'Coords')
                coord_text.set('points', self.calculate_polygon_coords(found_polygons_text_region, mm, page_coord))
                id_indexer_l = self.serialize_lines_in_region(textregion, all_found_texline_polygons, mm, page_coord, all_box_coord, slopes, id_indexer_l)
                add_textequiv(textregion)

        self.logger.debug('len(found_polygons_text_region_h) %s', len(found_polygons_text_region_h))
        if len(found_polygons_text_region_h) > 0:
            for mm in range(len(found_polygons_text_region_h)):
                textregion=ET.SubElement(page, 'TextRegion')
                textregion.set('id', 'r%s' % id_indexer)
                id_indexer += 1
                textregion.set('type','header')
                coord_text = ET.SubElement(textregion, 'Coords')
                coord_text.set('points', self.calculate_polygon_coords(found_polygons_text_region_h, mm, page_coord))
                id_indexer_l = self.serialize_lines_in_region(textregion, all_found_texline_polygons_h, mm, page_coord, all_box_coord_h, slopes, id_indexer_l)
                add_textequiv(textregion)

        if len(found_polygons_drop_capitals) > 0:
            id_indexer = len(found_polygons_text_region) + len(found_polygons_text_region_h) + len(found_polygons_marginals)
            for mm in range(len(found_polygons_drop_capitals)):
                textregion=ET.SubElement(page, 'TextRegion')
                textregion.set('id',' r%s' % id_indexer)
                id_indexer += 1
                textregion.set('type', 'drop-capital')
                coord_text = ET.SubElement(textregion, 'Coords')
                coord_text.set('points', self.calculate_polygon_coords(found_polygons_drop_capitals, mm, page_coord))
                add_textequiv(textregion)

        for mm in range(len(found_polygons_marginals)):
            marginal = ET.SubElement(page, 'TextRegion')
            add_textequiv(textregion)
            marginal.set('id', id_of_marginalia[mm])
            marginal.set('type', 'marginalia')
            coord_text = ET.SubElement(marginal, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_marginals, mm, page_coord))
            id_indexer_l = self.serialize_lines_in_marginal(marginal, all_found_texline_polygons_marginals, mm, page_coord, all_box_coord_marginals, slopes_marginals, id_indexer_l)

        id_indexer = len(found_polygons_text_region) + len(found_polygons_text_region_h) + len(found_polygons_marginals) + len(found_polygons_drop_capitals)
        for mm in range(len(found_polygons_text_region_img)):
            textregion=ET.SubElement(page, 'ImageRegion')
            textregion.set('id', 'r%s' % id_indexer)
            id_indexer += 1
            coord_text = ET.SubElement(textregion, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_text_region_img, mm, page_coord))

        for mm in range(len(found_polygons_tables)):
            textregion = ET.SubElement(page, 'TableRegion')
            textregion.set('id', 'r%s' %id_indexer)
            id_indexer += 1
            coord_text = ET.SubElement(textregion, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_tables, mm, page_coord))

        return pcgts

    def calculate_polygon_coords(self, contour_list, i, page_coord):
        self.logger.debug('enter calculate_polygon_coords')
        coords = ''
        for j in range(len(contour_list[i])):
            if len(contour_list[i][j]) == 2:
                coords += str(int((contour_list[i][j][0] + page_coord[2]) / self.scale_x))
                coords += ','
                coords += str(int((contour_list[i][j][1] + page_coord[0]) / self.scale_y))
            else:
                coords += str(int((contour_list[i][j][0][0] + page_coord[2]) / self.scale_x))
                coords += ','
                coords += str(int((contour_list[i][j][0][1] + page_coord[0]) / self.scale_y))

            if j < len(contour_list[i]) - 1:
                coords=coords + ' '
        return coords

