# pylint: disable=too-many-locals,wrong-import-position,too-many-lines,too-many-statements,chained-comparison,fixme,broad-except,c-extension-no-member
# pylint: disable=import-error
from pathlib import Path
import os.path
from typing import Optional
import logging
from .utils.xml import create_page_xml, xml_reading_order
from .utils.counter import EynollahIdCounter

from ocrd_models.ocrd_page import (
        BorderType,
        CoordsType,
        TextLineType,
        TextEquivType,
        TextRegionType,
        ImageRegionType,
        TableRegionType,
        SeparatorRegionType,
        to_xml
        )

class EynollahXmlWriter:

    def __init__(self, *, dir_out, image_filename, curved_line, pcgts=None):
        self.logger = logging.getLogger('eynollah.writer')
        self.counter = EynollahIdCounter()
        self.dir_out = dir_out
        self.image_filename = image_filename
        self.output_filename = os.path.join(self.dir_out or "", self.image_filename_stem) + ".xml"
        self.curved_line = curved_line
        self.pcgts = pcgts
        self.scale_x: Optional[float] = None # XXX set outside __init__
        self.scale_y: Optional[float] = None # XXX set outside __init__
        self.height_org: Optional[int] = None # XXX set outside __init__
        self.width_org: Optional[int] = None # XXX set outside __init__

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

    def serialize_lines_in_region(self, text_region, all_found_textline_polygons, region_idx, page_coord, all_box_coord, slopes, counter, ocr_all_textlines_textregion):
        self.logger.debug('enter serialize_lines_in_region')
        for j, polygon_textline in enumerate(all_found_textline_polygons[region_idx]):
            coords = CoordsType()
            textline = TextLineType(id=counter.next_line_id, Coords=coords)
            if ocr_all_textlines_textregion:
                # FIXME: add OCR confidence
                textline.set_TextEquiv([TextEquivType(Unicode=ocr_all_textlines_textregion[j])])
            text_region.add_TextLine(textline)
            text_region.set_orientation(-slopes[region_idx])
            region_bboxes = all_box_coord[region_idx]
            points_co = ''
            for point in polygon_textline:
                if len(point) != 2:
                    point = point[0]
                point_x = point[0] + page_coord[2]
                point_y = point[1] + page_coord[0]
                point_x = max(0, int(point_x / self.scale_x))
                point_y = max(0, int(point_y / self.scale_y))
                points_co += f'{point_x},{point_y} '
            coords.set_points(points_co[:-1])

    def write_pagexml(self, pcgts):
        self.logger.info("output filename: '%s'", self.output_filename)
        with open(self.output_filename, 'w') as f:
            f.write(to_xml(pcgts))

    def build_pagexml_no_full_layout(
        self,
        *,
        found_polygons_text_region,
        page_coord,
        order_of_texts,
        all_found_textline_polygons,
        all_box_coord,
        found_polygons_text_region_img,
        found_polygons_marginals_left,
        found_polygons_marginals_right,
        all_found_textline_polygons_marginals_left,
        all_found_textline_polygons_marginals_right,
        all_box_coord_marginals_left,
        all_box_coord_marginals_right,
        slopes,
        slopes_marginals_left,
        slopes_marginals_right,
        cont_page,
        polygons_seplines,
        found_polygons_tables,
    ):
        return self.build_pagexml_full_layout(
            found_polygons_text_region=found_polygons_text_region,
            found_polygons_text_region_h=[],
            page_coord=page_coord,
            order_of_texts=order_of_texts,
            all_found_textline_polygons=all_found_textline_polygons,
            all_found_textline_polygons_h=[],
            all_box_coord=all_box_coord,
            all_box_coord_h=[],
            found_polygons_text_region_img=found_polygons_text_region_img,
            found_polygons_tables=found_polygons_tables,
            found_polygons_drop_capitals=[],
            found_polygons_marginals_left=found_polygons_marginals_left,
            found_polygons_marginals_right=found_polygons_marginals_right,
            all_found_textline_polygons_marginals_left=all_found_textline_polygons_marginals_left,
            all_found_textline_polygons_marginals_right=all_found_textline_polygons_marginals_right,
            all_box_coord_marginals_left=all_box_coord_marginals_left,
            all_box_coord_marginals_right=all_box_coord_marginals_right,
            slopes=slopes,
            slopes_h=[],
            slopes_marginals_left=slopes_marginals_left,
            slopes_marginals_right=slopes_marginals_right,
            cont_page=cont_page,
            polygons_seplines=polygons_seplines,
        )

    def build_pagexml_full_layout(
        self,
        *,
        found_polygons_text_region,
        found_polygons_text_region_h,
        page_coord,
        order_of_texts,
        all_found_textline_polygons,
        all_found_textline_polygons_h,
        all_box_coord,
        all_box_coord_h,
        found_polygons_text_region_img,
        found_polygons_tables,
        found_polygons_drop_capitals,
        found_polygons_marginals_left,
        found_polygons_marginals_right,
        all_found_textline_polygons_marginals_left,
        all_found_textline_polygons_marginals_right,
        all_box_coord_marginals_left,
        all_box_coord_marginals_right,
        slopes,
        slopes_h,
        slopes_marginals_left,
        slopes_marginals_right,
        cont_page,
        polygons_seplines,
        ocr_all_textlines=None,
        ocr_all_textlines_h=None,
        ocr_all_textlines_marginals_left=None,
        ocr_all_textlines_marginals_right=None,
        ocr_all_textlines_drop=None,
        conf_contours_textregions=None,
        conf_contours_textregions_h=None,
        skip_layout_reading_order=False,
    ):
        self.logger.debug('enter build_pagexml')

        # create the file structure
        pcgts = self.pcgts if self.pcgts else create_page_xml(self.image_filename, self.height_org, self.width_org)
        page = pcgts.get_Page()
        assert page
        page.set_Border(BorderType(Coords=CoordsType(points=self.calculate_page_coords(cont_page))))

        counter = EynollahIdCounter()
        if len(order_of_texts):
            _counter_marginals = EynollahIdCounter(region_idx=len(order_of_texts))
            id_of_marginalia_left = [_counter_marginals.next_region_id
                                     for _ in found_polygons_marginals_left]
            id_of_marginalia_right = [_counter_marginals.next_region_id
                                      for _ in found_polygons_marginals_right]
            xml_reading_order(page, order_of_texts, id_of_marginalia_left, id_of_marginalia_right)

        for mm, region_contour in enumerate(found_polygons_text_region):
            textregion = TextRegionType(
                id=counter.next_region_id, type_='paragraph',
                Coords=CoordsType(points=self.calculate_polygon_coords(region_contour, page_coord,
                                                                       skip_layout_reading_order))
            )
            assert textregion.Coords
            if conf_contours_textregions:
                textregion.Coords.set_conf(conf_contours_textregions[mm])
            page.add_TextRegion(textregion)
            if ocr_all_textlines:
                ocr_textlines = ocr_all_textlines[mm]
            else:
                ocr_textlines = None
            self.serialize_lines_in_region(textregion, all_found_textline_polygons, mm, page_coord,
                                           all_box_coord, slopes, counter, ocr_textlines)

        self.logger.debug('len(found_polygons_text_region_h) %s', len(found_polygons_text_region_h))
        for mm, region_contour in enumerate(found_polygons_text_region_h):
            textregion = TextRegionType(
                id=counter.next_region_id, type_='heading',
                Coords=CoordsType(points=self.calculate_polygon_coords(region_contour, page_coord))
            )
            assert textregion.Coords
            if conf_contours_textregions_h:
                textregion.Coords.set_conf(conf_contours_textregions_h[mm])
            page.add_TextRegion(textregion)
            if ocr_all_textlines_h:
                ocr_textlines = ocr_all_textlines_h[mm]
            else:
                ocr_textlines = None
            self.serialize_lines_in_region(textregion, all_found_textline_polygons_h, mm, page_coord,
                                           all_box_coord_h, slopes_h, counter, ocr_textlines)

        for mm, region_contour in enumerate(found_polygons_marginals_left):
            marginal = TextRegionType(
                id=counter.next_region_id, type_='marginalia',
                Coords=CoordsType(points=self.calculate_polygon_coords(region_contour, page_coord))
            )
            page.add_TextRegion(marginal)
            if ocr_all_textlines_marginals_left:
                ocr_textlines = ocr_all_textlines_marginals_left[mm]
            else:
                ocr_textlines = None
            self.serialize_lines_in_region(marginal, all_found_textline_polygons_marginals_left, mm, page_coord, all_box_coord_marginals_left, slopes_marginals_left, counter, ocr_textlines)

        for mm, region_contour in enumerate(found_polygons_marginals_right):
            marginal = TextRegionType(
                id=counter.next_region_id, type_='marginalia',
                Coords=CoordsType(points=self.calculate_polygon_coords(region_contour, page_coord))
            )
            page.add_TextRegion(marginal)
            if ocr_all_textlines_marginals_right:
                ocr_textlines = ocr_all_textlines_marginals_right[mm]
            else:
                ocr_textlines = None
            self.serialize_lines_in_region(marginal, all_found_textline_polygons_marginals_right, mm, page_coord,
                                             all_box_coord_marginals_right, slopes_marginals_right, counter, ocr_textlines)

        for mm, region_contour in enumerate(found_polygons_drop_capitals):
            dropcapital = TextRegionType(
                id=counter.next_region_id, type_='drop-capital',
                Coords=CoordsType(points=self.calculate_polygon_coords(region_contour, page_coord))
            )
            page.add_TextRegion(dropcapital)
            all_box_coord_drop = [[0, 0, 0, 0]]
            slopes_drop = [0]
            if ocr_all_textlines_drop:
                ocr_textlines = ocr_all_textlines_drop[mm]
            else:
                ocr_textlines = None
            self.serialize_lines_in_region(dropcapital, [[found_polygons_drop_capitals[mm]]], 0, page_coord,
                                           all_box_coord_drop, slopes_drop, counter, ocr_textlines)

        for region_contour in found_polygons_text_region_img:
            page.add_ImageRegion(
                ImageRegionType(id=counter.next_region_id,
                                Coords=CoordsType(points=self.calculate_polygon_coords(region_contour, page_coord))))

        for region_contour in polygons_seplines:
            page.add_SeparatorRegion(
                SeparatorRegionType(id=counter.next_region_id,
                                    Coords=CoordsType(points=self.calculate_polygon_coords(region_contour, [0, 0, 0, 0]))))

        for region_contour in found_polygons_tables:
            page.add_TableRegion(
                TableRegionType(id=counter.next_region_id,
                                Coords=CoordsType(points=self.calculate_polygon_coords(region_contour, page_coord))))

        return pcgts

    def calculate_polygon_coords(self, contour, page_coord, skip_layout_reading_order=False):
        self.logger.debug('enter calculate_polygon_coords')
        coords = ''
        for point in contour:
            if len(point) != 2:
                point = point[0]
            point_x = point[0]
            point_y = point[1]
            if not skip_layout_reading_order:
                point_x += page_coord[2]
                point_y += page_coord[0]
            point_x = int(point_x  / self.scale_x)
            point_y = int(point_y  / self.scale_y)
            coords += str(point_x) + ',' + str(point_y) + ' '
        return coords[:-1]

