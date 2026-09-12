# pylint: disable=chained-comparison,c-extension-no-member
# pylint: disable=import-error
from __future__ import annotations
from pathlib import Path
import os.path
import logging

import numpy as np
import cv2
from shapely import affinity, clip_by_rect

from ocrd_utils import points_from_polygon
from ocrd_models.ocrd_page import (
    AlternativeImageType,
    BorderType,
    CoordsType,
    TextLineType,
    TextEquivType,
    TextRegionType,
    ImageRegionType,
    TableRegionType,
    SeparatorRegionType,
    PcGtsType,
    to_xml
)

from .utils import Region, TextRegion
from .utils.xml import create_page_xml, xml_reading_order
from .utils.counter import EynollahIdCounter
from .utils.contour import (
    contour2polygon,
    make_valid,
    ensure_polygon,
    join_polygons,
    bridge_polygons,
)

class EynollahXmlWriter:

    def __init__(
            self, *,
            dir_out: str | None,
            image_filename: str,
            image_width: int,
            image_height: int,
            pcgts: PcGtsType | None = None,
    ):
        self.logger = logging.getLogger('eynollah.writer')
        self.counter = EynollahIdCounter()
        self.dir_out = dir_out
        self.image_filename = image_filename
        self.output_filename = os.path.join(self.dir_out or "", self.image_filename_stem) + ".xml"
        self.pcgts = pcgts
        self.image_height = image_height
        self.image_width = image_width
        self.scale_x = 1.0
        self.scale_y = 1.0

    @property
    def image_filename_stem(self) -> str:
        return Path(Path(self.image_filename).name).stem

    def calculate_points(
            self, contour: np.ndarray,
            offset: list[int] | None = None,
            dilate: int = 0,
            open_holes: bool = False,
    ) -> str:
        poly = contour2polygon(contour, dilate=dilate, holes=open_holes)
        if offset is not None:
            poly = affinity.translate(poly, *offset)
        poly = affinity.scale(poly, xfact=1 / self.scale_x, yfact=1 / self.scale_y, origin=(0, 0))
        poly = make_valid(ensure_polygon(clip_by_rect(poly, 0, 0, self.image_width, self.image_height)))
        return points_from_polygon(poly.exterior.coords[:-1])

    def serialize_lines_in_region(
            self, text_region: TextRegionType,
            offset: list[int],
            counter: EynollahIdCounter,
            lines: list[Region],
    ) -> None:
        for line in lines:
            textline = TextLineType(
                id=counter.next_line_id,
                Coords=CoordsType(points=self.calculate_points(line.contour, offset, 5),
                                  conf=line.conf)
            )
            text_region.add_TextLine(textline)

    def write_pagexml(self, pcgts):
        self.logger.info("output filename: '%s'", self.output_filename)
        if img_alt := next(
                (img for img in pcgts.Page.AlternativeImage
                 if img.comments == "binarized"
                 and isinstance(img.filename, np.ndarray)), None):
            img_alt_filename = self.output_filename[:-4] + '.bin.png'
            cv2.imwrite(img_alt_filename, img_alt.filename)
            img_alt.filename = os.path.basename(img_alt_filename)
        with open(self.output_filename, 'w') as f:
            f.write(to_xml(pcgts))

    def build_pagexml(
        self,
        *,
        page: Region,
        img_bin: np.ndarray | None = None,
        num_col: int = 1,
        order_of_texts: list[int] = [], # ruff: ignore[B006] (safe here)
        textregions: list[TextRegion] = [], # ruff: ignore[B006] (safe here)
        textregions_h: list[TextRegion] = [], # ruff: ignore[B006] (safe here)
        images: list[Region] = [], # ruff: ignore[B006] (safe here)
        tables: list[Region] = [], # ruff: ignore[B006] (safe here)
        drop_caps: list[Region] = [], # ruff: ignore[B006] (safe here)
        marginals_left: list[TextRegion] = [], # ruff: ignore[B006] (safe here)
        marginals_right: list[TextRegion] = [], # ruff: ignore[B006] (safe here)
        seplines: list[Region] = [], # ruff: ignore[B006] (safe here)
    ):
        self.logger.debug('enter build_pagexml')

        # create the file structure
        pcgts = self.pcgts if self.pcgts else create_page_xml(
            self.image_filename, self.image_height, self.image_width)
        pcgts.Metadata.Comments = "num_col %d" % num_col
        if img_bin is not None:
            img_alt = AlternativeImageType(filename=img_bin, # will be replaced later
                                           comments="binarized")
            pcgts.Page.add_AlternativeImage(img_alt)
        pcgts.Page.set_custom('layout {num_col:%d;} ' % num_col)
        pcgts.Page.set_orientation(-page.skew)
        pcgts.Page.set_Border(BorderType(Coords=CoordsType(
            points=self.calculate_points(page.contour))))
        x, y, _, _ = cv2.boundingRect(page.contour)
        offset = [x, y]
        counter = EynollahIdCounter()
        if len(order_of_texts):
            _counter_marginals = EynollahIdCounter(region_idx=len(order_of_texts))
            id_of_marginalia_left = [_counter_marginals.next_region_id
                                     for _ in marginals_left]
            id_of_marginalia_right = [_counter_marginals.next_region_id
                                      for _ in marginals_right]
            xml_reading_order(pcgts.Page, order_of_texts, id_of_marginalia_left, id_of_marginalia_right)

        for region in textregions:
            textregion = TextRegionType(
                id=counter.next_region_id, type_='paragraph',
                Coords=CoordsType(points=self.calculate_points(region.contour, offset, 6),
                                  conf=region.conf),
                orientation=-region.skew
            )
            self.serialize_lines_in_region(textregion, offset, counter, region.lines)
            pcgts.Page.add_TextRegion(textregion)

        self.logger.debug('len(textregions_h) %s', len(textregions_h))
        for region in textregions_h:
            textregion = TextRegionType(
                id=counter.next_region_id, type_='heading',
                Coords=CoordsType(points=self.calculate_points(region.contour, offset, 6),
                                  conf=region.conf),
                orientation=-region.skew
            )
            self.serialize_lines_in_region(textregion, offset, counter, region.lines)
            pcgts.Page.add_TextRegion(textregion)

        for region in drop_caps:
            textregion = TextRegionType(
                id=counter.next_region_id, type_='drop-capital',
                Coords=CoordsType(points=self.calculate_points(region.contour, offset, 6),
                                  conf=region.conf),
                orientation=-region.skew
            )
            self.serialize_lines_in_region(textregion, offset, counter, [region])
            pcgts.Page.add_TextRegion(textregion)

        for region in marginals_left:
            textregion = TextRegionType(
                id=counter.next_region_id, type_='marginalia',
                Coords=CoordsType(points=self.calculate_points(region.contour, offset, 6),
                                  conf=region.conf),
                orientation=-region.skew
            )
            self.serialize_lines_in_region(textregion, offset, counter, region.lines)
            pcgts.Page.add_TextRegion(textregion)

        for region in marginals_right:
            textregion = TextRegionType(
                id=counter.next_region_id, type_='marginalia',
                Coords=CoordsType(points=self.calculate_points(region.contour, offset, 6),
                                  conf=region.conf),
                orientation=-region.skew
            )
            self.serialize_lines_in_region(textregion, offset, counter, region.lines)
            pcgts.Page.add_TextRegion(textregion)

        for region in images:
            image = ImageRegionType(
                id=counter.next_region_id,
                Coords=CoordsType(points=self.calculate_points(region.contour, offset, 2),
                                  conf=region.conf))
            pcgts.Page.add_ImageRegion(image)

        for region in seplines:
            pcgts.Page.add_SeparatorRegion(
                SeparatorRegionType(
                    id=counter.next_region_id,
                    Coords=CoordsType(points=self.calculate_points(region.contour, offset, 2,
                                                                   open_holes=True),
                                      conf=region.conf)))

        for region in tables:
            table = TableRegionType(
                id=counter.next_region_id,
                Coords=CoordsType(points=self.calculate_points(region.contour, offset, 6),
                                  conf=region.conf))
            pcgts.Page.add_TableRegion(table)

        return pcgts

