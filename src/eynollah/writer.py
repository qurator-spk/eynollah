# pylint: disable=too-many-locals,wrong-import-position,too-many-lines,too-many-statements,chained-comparison,fixme,broad-except,c-extension-no-member
# pylint: disable=import-error
from pathlib import Path
import os.path
import logging
from typing import Optional
import numpy as np
from shapely import affinity, clip_by_rect

from ocrd_utils import points_from_polygon
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

from .utils.xml import create_page_xml, xml_reading_order
from .utils.counter import EynollahIdCounter
from .utils.contour import contour2polygon, make_valid

class EynollahXmlWriter:

    def __init__(self, *, dir_out, image_filename, image_width, image_height, curved_line, pcgts=None):
        self.logger = logging.getLogger('eynollah.writer')
        self.counter = EynollahIdCounter()
        self.dir_out = dir_out
        self.image_filename = image_filename
        self.output_filename = os.path.join(self.dir_out or "", self.image_filename_stem) + ".xml"
        self.curved_line = curved_line
        self.pcgts = pcgts
        self.image_height = image_height
        self.image_width = image_width
        self.scale_x = 1.0
        self.scale_y = 1.0

    @property
    def image_filename_stem(self):
        return Path(Path(self.image_filename).name).stem

    def calculate_points(self, contour, offset=None):
        poly = contour2polygon(contour)
        if offset is not None:
            poly = affinity.translate(poly, *offset)
        poly = affinity.scale(poly, xfact=1 / self.scale_x, yfact=1 / self.scale_y, origin=(0, 0))
        poly = make_valid(clip_by_rect(poly, 0, 0, self.image_width, self.image_height))
        return points_from_polygon(poly.exterior.coords[:-1])

    def serialize_lines_in_region(self, text_region, all_found_textline_polygons, region_idx, page_coord, slopes, counter, ocr_all_textlines_textregion):
        for j, polygon_textline in enumerate(all_found_textline_polygons[region_idx]):
            coords = CoordsType()
            textline = TextLineType(id=counter.next_line_id, Coords=coords)
            if ocr_all_textlines_textregion:
                # FIXME: add OCR confidence
                textline.set_TextEquiv([TextEquivType(Unicode=ocr_all_textlines_textregion[j])])
            text_region.add_TextLine(textline)
            text_region.set_orientation(-slopes[region_idx])
            offset = [page_coord[2], page_coord[0]]
            coords.set_points(self.calculate_points(polygon_textline, offset))

    def write_pagexml(self, pcgts):
        self.logger.info("output filename: '%s'", self.output_filename)
        with open(self.output_filename, 'w') as f:
            f.write(to_xml(pcgts))

    def build_pagexml_no_full_layout(
        self,
        *,
        num_col,
        found_polygons_text_region,
        page_coord,
        page_slope,
        order_of_texts,
        all_found_textline_polygons,
        found_polygons_images,
        found_polygons_tables,
        found_polygons_marginals_left,
        found_polygons_marginals_right,
        all_found_textline_polygons_marginals_left,
        all_found_textline_polygons_marginals_right,
        slopes,
        slopes_marginals_left,
        slopes_marginals_right,
        cont_page,
        polygons_seplines,
        ocr_all_textlines=None,
        ocr_all_textlines_marginals_left=None,
        ocr_all_textlines_marginals_right=None,
        ocr_all_textlines_drop=None,
        conf_textregions=None,
        conf_marginals_left=None,
        conf_marginals_right=None,
        conf_images=None,
        conf_tables=None,
    ):
        return self.build_pagexml_full_layout(
            num_col=num_col,
            found_polygons_text_region=found_polygons_text_region,
            found_polygons_text_region_h=[],
            page_coord=page_coord,
            page_slope=page_slope,
            order_of_texts=order_of_texts,
            all_found_textline_polygons=all_found_textline_polygons,
            all_found_textline_polygons_h=[],
            found_polygons_images=found_polygons_images,
            found_polygons_tables=found_polygons_tables,
            found_polygons_drop_capitals=[],
            found_polygons_marginals_left=found_polygons_marginals_left,
            found_polygons_marginals_right=found_polygons_marginals_right,
            all_found_textline_polygons_marginals_left=all_found_textline_polygons_marginals_left,
            all_found_textline_polygons_marginals_right=all_found_textline_polygons_marginals_right,
            slopes=slopes,
            slopes_h=[],
            slopes_marginals_left=slopes_marginals_left,
            slopes_marginals_right=slopes_marginals_right,
            cont_page=cont_page,
            polygons_seplines=polygons_seplines,
            ocr_all_textlines=ocr_all_textlines,
            ocr_all_textlines_marginals_left=ocr_all_textlines_marginals_left,
            ocr_all_textlines_marginals_right=ocr_all_textlines_marginals_right,
            conf_textregions=conf_textregions,
            conf_marginals_left=conf_marginals_left,
            conf_marginals_right=conf_marginals_right,
            conf_images=conf_images,
            conf_tables=conf_tables,
        )

    def build_pagexml_full_layout(
        self,
        *,
        num_col,
        found_polygons_text_region,
        found_polygons_text_region_h,
        page_coord,
        page_slope,
        order_of_texts,
        all_found_textline_polygons,
        all_found_textline_polygons_h,
        found_polygons_images,
        found_polygons_tables,
        found_polygons_drop_capitals,
        found_polygons_marginals_left,
        found_polygons_marginals_right,
        all_found_textline_polygons_marginals_left,
        all_found_textline_polygons_marginals_right,
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
        conf_textregions=None,
        conf_textregions_h=None,
        conf_marginals_left=None,
        conf_marginals_right=None,
        conf_images=None,
        conf_tables=None,
        conf_drops=None,
        skip_layout_reading_order=False,
    ):
        self.logger.debug('enter build_pagexml')

        # create the file structure
        pcgts = self.pcgts if self.pcgts else create_page_xml(
            self.image_filename, self.image_height, self.image_width)
        page = pcgts.get_Page()
        pcgts.Metadata.Comments = "num_col %d" % num_col
        page.set_custom('layout {num_col:%d;} ' % num_col)
        page.set_orientation(-page_slope)
        if len(cont_page):
            page.set_Border(BorderType(Coords=CoordsType(points=self.calculate_points(cont_page[0]))))

        if skip_layout_reading_order:
            offset = None
        else:
            offset = [page_coord[2], page_coord[0]]
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
                Coords=CoordsType(points=self.calculate_points(region_contour, offset))
            )
            assert textregion.Coords
            if conf_textregions:
                textregion.Coords.set_conf(conf_textregions[mm])
            page.add_TextRegion(textregion)
            if ocr_all_textlines:
                ocr_textlines = ocr_all_textlines[mm]
            else:
                ocr_textlines = None
            self.serialize_lines_in_region(textregion, all_found_textline_polygons, mm, page_coord,
                                           slopes, counter, ocr_textlines)

        self.logger.debug('len(found_polygons_text_region_h) %s', len(found_polygons_text_region_h))
        for mm, region_contour in enumerate(found_polygons_text_region_h):
            textregion = TextRegionType(
                id=counter.next_region_id, type_='heading',
                Coords=CoordsType(points=self.calculate_points(region_contour, offset))
            )
            assert textregion.Coords
            if conf_textregions_h:
                textregion.Coords.set_conf(conf_textregions_h[mm])
            page.add_TextRegion(textregion)
            if ocr_all_textlines_h:
                ocr_textlines = ocr_all_textlines_h[mm]
            else:
                ocr_textlines = None
            self.serialize_lines_in_region(textregion, all_found_textline_polygons_h, mm, page_coord,
                                           slopes_h, counter, ocr_textlines)

        for mm, region_contour in enumerate(found_polygons_drop_capitals):
            dropcapital = TextRegionType(
                id=counter.next_region_id, type_='drop-capital',
                Coords=CoordsType(points=self.calculate_points(region_contour, offset))
            )
            if conf_drops:
                dropcapital.Coords.set_conf(conf_drops[mm])
            page.add_TextRegion(dropcapital)
            slopes_drop = [0]
            if ocr_all_textlines_drop:
                ocr_textlines = ocr_all_textlines_drop[mm]
            else:
                ocr_textlines = None
            self.serialize_lines_in_region(dropcapital, [[found_polygons_drop_capitals[mm]]], 0, page_coord,
                                           slopes_drop, counter, ocr_textlines)

        for mm, region_contour in enumerate(found_polygons_marginals_left):
            marginal = TextRegionType(
                id=counter.next_region_id, type_='marginalia',
                Coords=CoordsType(points=self.calculate_points(region_contour, offset))
            )
            if conf_marginals_left:
                marginal.Coords.set_conf(conf_marginals_left[mm])
            page.add_TextRegion(marginal)
            if ocr_all_textlines_marginals_left:
                ocr_textlines = ocr_all_textlines_marginals_left[mm]
            else:
                ocr_textlines = None
            self.serialize_lines_in_region(marginal, all_found_textline_polygons_marginals_left, mm, page_coord,
                                           slopes_marginals_left, counter, ocr_textlines)

        for mm, region_contour in enumerate(found_polygons_marginals_right):
            marginal = TextRegionType(
                id=counter.next_region_id, type_='marginalia',
                Coords=CoordsType(points=self.calculate_points(region_contour, offset))
            )
            if conf_marginals_right:
                marginal.Coords.set_conf(conf_marginals_right[mm])
            page.add_TextRegion(marginal)
            if ocr_all_textlines_marginals_right:
                ocr_textlines = ocr_all_textlines_marginals_right[mm]
            else:
                ocr_textlines = None
            self.serialize_lines_in_region(marginal, all_found_textline_polygons_marginals_right, mm, page_coord,
                                             slopes_marginals_right, counter, ocr_textlines)

        for mm, region_contour in enumerate(found_polygons_images):
            image = ImageRegionType(
                id=counter.next_region_id,
                Coords=CoordsType(points=self.calculate_points(region_contour, offset)))
            if conf_images:
                image.Coords.set_conf(conf_images[mm])
            page.add_ImageRegion(image)

        for region_contour in polygons_seplines:
            page.add_SeparatorRegion(
                SeparatorRegionType(id=counter.next_region_id,
                                    Coords=CoordsType(points=self.calculate_points(region_contour, None))))

        for mm, region_contour in enumerate(found_polygons_tables):
            table = TableRegionType(
                id=counter.next_region_id,
                Coords=CoordsType(points=self.calculate_points(region_contour, offset)))
            if conf_tables:
                table.Coords.set_conf(conf_tables[mm])
            page.add_TableRegion(table)

        return pcgts

