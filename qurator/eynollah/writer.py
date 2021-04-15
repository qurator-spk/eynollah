# pylint: disable=too-many-locals,wrong-import-position,too-many-lines,too-many-statements,chained-comparison,fixme,broad-except,c-extension-no-member
# pylint: disable=import-error
from pathlib import Path
import os.path

from .utils.xml import create_page_xml, xml_reading_order
from .utils.counter import EynollahIdCounter

from ocrd_utils import getLogger
from ocrd_models.ocrd_page import (
        BorderType,
        CoordsType,
        TextEquivType,
        PcGtsType,
        TextLineType,
        TextRegionType,
        ImageRegionType,
        TableRegionType,

        to_xml
        )
import numpy as np

class EynollahXmlWriter():

    def __init__(self, *, dir_out, image_filename, curved_line, pcgts=None):
        self.logger = getLogger('eynollah.writer')
        self.counter = EynollahIdCounter()
        self.dir_out = dir_out
        self.image_filename = image_filename
        self.curved_line = curved_line
        self.pcgts = pcgts
        self.scale_x = None # XXX set outside __init__
        self.scale_y = None # XXX set outside __init__
        self.height_org = None # XXX set outside __init__
        self.width_org = None # XXX set outside __init__

    @property
    def image_filename_stem(self):
        return Path(Path(self.image_filename).name).stem

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

    def serialize_lines_in_marginal(self, marginal_region, all_found_texline_polygons_marginals, marginal_idx, page_coord, all_box_coord_marginals, slopes_marginals, counter):
        for j in range(len(all_found_texline_polygons_marginals[marginal_idx])):
            coords = CoordsType()
            textline = TextLineType(id=counter.next_line_id, Coords=coords)
            marginal_region.add_TextLine(textline)
            textline.add_TextEquiv(TextEquivType(Unicode=''))
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
                points_co += ' '
            coords.set_points(points_co[:-1])

    def serialize_lines_in_region(self, text_region, all_found_texline_polygons, region_idx, page_coord, all_box_coord, slopes, counter):
        self.logger.debug('enter serialize_lines_in_region')
        for j in range(len(all_found_texline_polygons[region_idx])):
            coords = CoordsType()
            textline = TextLineType(id=counter.next_line_id, Coords=coords, TextEquiv=[TextEquivType(index=0, Unicode='')])
            text_region.add_TextLine(textline)
            region_bboxes = all_box_coord[region_idx]
            points_co = ''
            for idx_contour_textline, contour_textline in enumerate(all_found_texline_polygons[region_idx][j]):
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
            coords.set_points(points_co[:-1])

    def write_pagexml(self, pcgts):
        out_fname = os.path.join(self.dir_out, self.image_filename_stem) + ".xml"
        self.logger.info("output filename: '%s'", out_fname)
        with open(out_fname, 'w') as f:
            f.write(to_xml(pcgts))

    def build_pagexml_no_full_layout(self, found_polygons_text_region, page_coord, order_of_texts, id_of_texts, all_found_texline_polygons, all_box_coord, found_polygons_text_region_img, found_polygons_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals, cont_page):
        self.logger.debug('enter build_pagexml_no_full_layout')

        # create the file structure
        pcgts = self.pcgts if self.pcgts else create_page_xml(self.image_filename, self.height_org, self.width_org)
        page = pcgts.get_Page()
        page.set_Border(BorderType(Coords=CoordsType(points=self.calculate_page_coords(cont_page))))

        counter = EynollahIdCounter()
        if len(found_polygons_text_region) > 0:
            _counter_marginals = EynollahIdCounter(region_idx=len(order_of_texts))
            id_of_marginalia = [_counter_marginals.next_region_id for _ in found_polygons_marginals]
            xml_reading_order(page, order_of_texts, id_of_marginalia)

        for mm in range(len(found_polygons_text_region)):
            textregion = TextRegionType(id=counter.next_region_id, type_='paragraph',
                    Coords=CoordsType(points=self.calculate_polygon_coords(found_polygons_text_region[mm], page_coord)),
                    TextEquiv=[TextEquivType(index=0, Unicode='')])
            page.add_TextRegion(textregion)
            self.serialize_lines_in_region(textregion, all_found_texline_polygons, mm, page_coord, all_box_coord, slopes, counter)

        for mm in range(len(found_polygons_marginals)):
            marginal = TextRegionType(id=counter.next_region_id, type_='marginalia',
                    Coords=CoordsType(points=self.calculate_polygon_coords(found_polygons_marginals[mm], page_coord)))
            page.add_TextRegion(marginal)
            self.serialize_lines_in_marginal(marginal, all_found_texline_polygons_marginals, mm, page_coord, all_box_coord_marginals, slopes_marginals, counter)

        for mm in range(len(found_polygons_text_region_img)):
            img_region = ImageRegionType(id=counter.next_region_id, Coords=CoordsType())
            page.add_ImageRegion(img_region)
            points_co = ''
            for lmm in range(len(found_polygons_text_region_img[mm])):
                points_co += str(int((found_polygons_text_region_img[mm][lmm,0,0] + page_coord[2]) / self.scale_x))
                points_co += ','
                points_co += str(int((found_polygons_text_region_img[mm][lmm,0,1] + page_coord[0]) / self.scale_y))
                points_co += ' '
            img_region.get_Coords().set_points(points_co[:-1])

        return pcgts

    def build_pagexml_full_layout(self, found_polygons_text_region, found_polygons_text_region_h, page_coord, order_of_texts, id_of_texts, all_found_texline_polygons, all_found_texline_polygons_h, all_box_coord, all_box_coord_h, found_polygons_text_region_img, found_polygons_tables, found_polygons_drop_capitals, found_polygons_marginals, all_found_texline_polygons_marginals, all_box_coord_marginals, slopes, slopes_marginals, cont_page):
        self.logger.debug('enter build_pagexml_full_layout')

        # create the file structure
        pcgts = self.pcgts if self.pcgts else create_page_xml(self.image_filename, self.height_org, self.width_org)
        page = pcgts.get_Page()
        page.set_Border(BorderType(Coords=CoordsType(points=self.calculate_page_coords(cont_page))))

        counter = EynollahIdCounter()
        _counter_marginals = EynollahIdCounter(region_idx=len(order_of_texts))
        id_of_marginalia = [_counter_marginals.next_region_id for _ in found_polygons_marginals]
        xml_reading_order(page, order_of_texts, id_of_marginalia)

        for mm in range(len(found_polygons_text_region)):
            textregion = TextRegionType(id=counter.next_region_id, type_='paragraph',
                    TextEquiv=[TextEquivType(index=0, Unicode='')],
                    Coords=CoordsType(points=self.calculate_polygon_coords(found_polygons_text_region[mm], page_coord)))
            page.add_TextRegion(textregion)
            self.serialize_lines_in_region(textregion, all_found_texline_polygons, mm, page_coord, all_box_coord, slopes, counter)

        self.logger.debug('len(found_polygons_text_region_h) %s', len(found_polygons_text_region_h))
        for mm in range(len(found_polygons_text_region_h)):
            textregion = TextRegionType(id=counter.next_region_id, type_='header',
                    TextEquiv=[TextEquivType(index=0, Unicode='')],
                    Coords=CoordsType(points=self.calculate_polygon_coords(found_polygons_text_region_h[mm], page_coord)))
            page.add_TextRegion(textregion)
            self.serialize_lines_in_region(textregion, all_found_texline_polygons_h, mm, page_coord, all_box_coord_h, slopes, counter)

        for mm in range(len(found_polygons_marginals)):
            marginal = TextRegionType(id=counter.next_region_id, type_='marginalia',
                    TextEquiv=[TextEquivType(index=0, Unicode='')],
                    Coords=CoordsType(points=self.calculate_polygon_coords(found_polygons_marginals[mm], page_coord)))
            page.add_TextRegion(marginal)
            self.serialize_lines_in_marginal(marginal, all_found_texline_polygons_marginals, mm, page_coord, all_box_coord_marginals, slopes_marginals, counter)

        for mm in range(len(found_polygons_drop_capitals)):
            page.add_TextRegion(TextRegionType(id=counter.next_region_id, type_='drop-capital',
                    TextEquiv=[TextEquivType(index=0, Unicode='')],
                    Coords=CoordsType(points=self.calculate_polygon_coords(found_polygons_drop_capitals[mm], page_coord))))

        for mm in range(len(found_polygons_text_region_img)):
            page.add_ImageRegion(ImageRegionType(id=counter.next_region_id, Coords=CoordsType(points=self.calculate_polygon_coords(found_polygons_text_region_img[mm], page_coord))))

        for mm in range(len(found_polygons_tables)):
            page.add_TableRegion(TableRegionType(id=counter.next_region_id, Coords=CoordsType(points=self.calculate_polygon_coords(found_polygons_tables[mm], page_coord))))

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

