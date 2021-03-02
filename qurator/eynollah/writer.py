# pylint: disable=too-many-locals,wrong-import-position,too-many-lines,too-many-statements,chained-comparison,fixme,broad-except,c-extension-no-member
from pathlib import Path
import os.path

from .utils.xml import create_page_xml, add_textequiv, xml_reading_order
from .utils.counter import EynollahIdCounter

from ocrd_utils import getLogger
from lxml import etree as ET
import numpy as np

class EynollahXmlWriter():

    def __init__(self, *, dir_out, image_filename, curved_line):
        self.logger = getLogger('eynollah.writer')
        self.counter = EynollahIdCounter()
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

    def serialize_lines_in_marginal(self, marginal, all_found_texline_polygons_marginals, marginal_idx, page_coord, all_box_coord_marginals, slopes_marginals, counter):
        for j in range(len(all_found_texline_polygons_marginals[marginal_idx])):
            textline = ET.SubElement(marginal, 'TextLine')
            textline.set('id', counter.next_line_id)
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

    def serialize_lines_in_region(self, textregion, all_found_texline_polygons, region_idx, page_coord, all_box_coord, slopes, counter):
        self.logger.debug('enter serialize_lines_in_region')
        for j in range(len(all_found_texline_polygons[region_idx])):
            textline = ET.SubElement(textregion, 'TextLine')
            textline.set('id', counter.next_line_id)
            coord = ET.SubElement(textline, 'Coords')
            add_textequiv(textline)
            region_bboxes = all_box_coord[region_idx]

            points_co = ''
            for idx_contour_textline, contour_textline in all_found_texline_polygons[region_idx][j]:
                if not self.curved_line:
                    if len(contour_textline) == 2:
                        textline_x_coord = max(0, int((contour_textline[0] + region_bboxes[2] + page_coord[2]) / self.scale_x))
                        textline_y_coord = max(0, int((contour_textline[1] + region_bboxes[0] + page_coord[0]) / self.scale_y))
                    else:
                        textline_x_coord = max(0, int((contour_textline[0][0] + region_bboxes[2] + page_coord[2]) / self.scale_x))
                        textline_y_coord = max(0, int((contour_textline[0][1] + region_bboxes[0] + page_coord[0]) / self.scale_y))
                    points_co += str(textline_x_coord)
                    points_co += ','
                    points_co += str(textline_y_coord)

                if self.curved_line and np.abs(slopes[region_idx]) <= 45:
                    if len(contour_textline) == 2:
                        points_co += str(int((contour_textline[0] + page_coord[2]) / self.scale_x))
                        points_co += ','
                        points_co += str(int((contour_textline[1] + page_coord[0]) / self.scale_y))
                    else:
                        points_co += str(int((contour_textline[0][0] + page_coord[2]) / self.scale_x))
                        points_co += ','
                        points_co += str(int((contour_textline[0][1] + page_coord[0])/self.scale_y))
                elif self.curved_line and np.abs(slopes[region_idx]) > 45:
                    if len(contour_textline)==2:
                        points_co += str(int((contour_textline[0] + region_bboxes[2] + page_coord[2])/self.scale_x))
                        points_co += ','
                        points_co += str(int((contour_textline[1] + region_bboxes[0] + page_coord[0])/self.scale_y))
                    else:
                        points_co += str(int((contour_textline[0][0] + region_bboxes[2]+page_coord[2])/self.scale_x))
                        points_co += ','
                        points_co += str(int((contour_textline[0][1] + region_bboxes[0]+page_coord[0])/self.scale_y))

                points_co += ' '
            coord.set('points', points_co[:-1])

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

        counter_textregions = EynollahIdCounter()
        counter_marginals = EynollahIdCounter(region_idx=len(order_of_texts))

        id_of_marginalia = [counter_marginals.next_region_id for _ in found_polygons_marginals]
        if len(found_polygons_text_region) > 0:
            xml_reading_order(page, order_of_texts, id_of_texts, id_of_marginalia)

        for mm in range(len(found_polygons_text_region)):
            textregion = ET.SubElement(page, 'TextRegion')
            textregion.set('id', counter_textregions.next_region_id)
            textregion.set('type', 'paragraph')
            coord_text = ET.SubElement(textregion, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_text_region[mm], page_coord))
            self.serialize_lines_in_region(textregion, all_found_texline_polygons, mm, page_coord, all_box_coord, slopes, counter_textregions)
            add_textequiv(textregion)

        for idx_marginal, marginal_polygon in enumerate(found_polygons_marginals):
            marginal = ET.SubElement(page, 'TextRegion')
            marginal.set('id', id_of_marginalia[idx_marginal])
            marginal.set('type', 'marginalia')
            coord_text = ET.SubElement(marginal, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(marginal_polygon, page_coord))
            self.serialize_lines_in_marginal(marginal, all_found_texline_polygons_marginals, mm, page_coord, all_box_coord_marginals, slopes_marginals, counter_textregions)

        for mm in range(len(found_polygons_text_region_img)):
            textregion = ET.SubElement(page, 'ImageRegion')
            textregion.set('id', counter_textregions.next_region_id)
            coord_text = ET.SubElement(textregion, 'Coords')
            points_co = ''
            for lmm in range(len(found_polygons_text_region_img[mm])):
                points_co += str(int((found_polygons_text_region_img[mm][lmm,0,0] + page_coord[2]) / self.scale_x))
                points_co += ','
                points_co += str(int((found_polygons_text_region_img[mm][lmm,0,1] + page_coord[0]) / self.scale_y))
                points_co += ' '
            coord_text.set('points', points_co[:-1])

        return pcgts

    def build_pagexml_full_layout(self, found_polygons_text_region, found_polygons_text_region_h, page_coord, order_of_texts, id_of_texts, all_found_texline_polygons, all_found_texline_polygons_h, all_box_coord, all_box_coord_h, found_polygons_text_region_img, found_polygons_tables, found_polygons_drop_capitals, found_polygons_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals, cont_page):
        self.logger.debug('enter build_pagexml_full_layout')

        # create the file structure
        pcgts, page = create_page_xml(self.image_filename, self.height_org, self.width_org)
        page_print_sub = ET.SubElement(page, "Border")
        coord_page = ET.SubElement(page_print_sub, "Coords")
        coord_page.set('points', self.calculate_page_coords(cont_page))

        counter_textregions = EynollahIdCounter()
        counter_marginals = EynollahIdCounter(region_idx=len(order_of_texts))

        id_of_marginalia = [counter_marginals.next_region_id for _ in found_polygons_marginals]
        xml_reading_order(page, order_of_texts, id_of_texts, id_of_marginalia)

        for mm in range(len(found_polygons_text_region)):
            textregion=ET.SubElement(page, 'TextRegion')
            textregion.set('id', counter_textregions.next_region_id)
            textregion.set('type', 'paragraph')
            coord_text = ET.SubElement(textregion, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_text_region[mm], page_coord))
            self.serialize_lines_in_region(textregion, all_found_texline_polygons, mm, page_coord, all_box_coord, slopes, counter_textregions)
            add_textequiv(textregion)

        self.logger.debug('len(found_polygons_text_region_h) %s', len(found_polygons_text_region_h))
        for mm in range(len(found_polygons_text_region_h)):
            textregion=ET.SubElement(page, 'TextRegion')
            textregion.set('id', counter_textregions.next_region_id)
            textregion.set('type','header')
            coord_text = ET.SubElement(textregion, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_text_region_h[mm], page_coord))
            self.serialize_lines_in_region(textregion, all_found_texline_polygons_h, mm, page_coord, all_box_coord_h, slopes, counter_textregions)
            add_textequiv(textregion)

        for mm in range(len(found_polygons_drop_capitals)):
            textregion=ET.SubElement(page, 'TextRegion')
            textregion.set('id', counter_textregions.next_region_id)
            textregion.set('type', 'drop-capital')
            coord_text = ET.SubElement(textregion, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_drop_capitals[mm], page_coord))
            add_textequiv(textregion)

        for mm in range(len(found_polygons_marginals)):
            marginal = ET.SubElement(page, 'TextRegion')
            add_textequiv(textregion)
            marginal.set('id', id_of_marginalia[mm])
            marginal.set('type', 'marginalia')
            coord_text = ET.SubElement(marginal, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_marginals[mm], page_coord))
            self.serialize_lines_in_marginal(marginal, all_found_texline_polygons_marginals, mm, page_coord, all_box_coord_marginals, slopes_marginals, counter_textregions)
        counter_textregions.inc('region', counter_marginals.get('region'))

        for mm in range(len(found_polygons_text_region_img)):
            textregion=ET.SubElement(page, 'ImageRegion')
            textregion.set('id', counter_textregions.next_region_id)
            coord_text = ET.SubElement(textregion, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_text_region_img[mm], page_coord))

        for mm in range(len(found_polygons_tables)):
            textregion = ET.SubElement(page, 'TableRegion')
            textregion.set('id', counter_textregions.next_region_id)
            coord_text = ET.SubElement(textregion, 'Coords')
            coord_text.set('points', self.calculate_polygon_coords(found_polygons_tables[mm], page_coord))

        return pcgts

    def calculate_polygon_coords(self, contour, page_coord):
        self.logger.debug('enter calculate_polygon_coords')
        coords = ''
        for value_bbox in contour:
            if len(value_bbox) == 2:
                coords += str(int((value_bbox[0] + page_coord[2]) / self.scale_x))
                coords += ','
                coords += str(int((value_bbox[1] + page_coord[0]) / self.scale_y))
            else:
                coords += str(int((value_bbox[0][0] + page_coord[2]) / self.scale_x))
                coords += ','
                coords += str(int((value_bbox[0][1] + page_coord[0]) / self.scale_y))
            coords=coords + ' '
        return coords[:-1]

