# pylint: disable=too-many-locals,wrong-import-position,too-many-lines,too-many-statements,chained-comparison,fixme,broad-except,c-extension-no-member
# pylint: disable=invalid-name
from .counter import EynollahIdCounter
import numpy as np
from datetime import datetime

from ocrd_models.ocrd_page import (
    CoordsType,
    GlyphType,
    ImageRegionType,
    MathsRegionType,
    MetadataType,
    MetadataItemType,
    NoiseRegionType,
    OrderedGroupIndexedType,
    OrderedGroupType,
    PcGtsType,
    PageType,
    ReadingOrderType,
    RegionRefIndexedType,
    RegionRefType,
    SeparatorRegionType,
    TableRegionType,
    TextEquivType,
    TextLineType,
    TextRegionType,
    UnorderedGroupIndexedType,
    UnorderedGroupType,
    WordType,

    to_xml)

def create_page_xml(imageFilename, height, width):
    now = datetime.now()
    pcgts = PcGtsType(
        Metadata=MetadataType(
            Creator='SBB_QURATOR',
            Created=now,
            LastChange=now
        ),
        Page=PageType(
            imageWidth=str(width),
            imageHeight=str(height),
            imageFilename=imageFilename,
            readingDirection='left-to-right',
            textLineOrder='top-to-bottom'
        ))
    return pcgts

def xml_reading_order(page, order_of_texts, id_of_marginalia):
    region_order = ReadingOrderType()
    og = OrderedGroupType(id="ro357564684568544579089")
    page.set_ReadingOrder(region_order)
    region_order.set_OrderedGroup(og)
    region_counter = EynollahIdCounter()
    for idx_textregion, _ in enumerate(order_of_texts):
        og.add_RegionRefIndexed(RegionRefIndexedType(index=str(region_counter.get('region')), regionRef=region_counter.region_id(order_of_texts[idx_textregion] + 1)))
        region_counter.inc('region')
    for id_marginal in id_of_marginalia:
        og.add_RegionRefIndexed(RegionRefIndexedType(index=str(region_counter.get('region')), regionRef=id_marginal))
        region_counter.inc('region')

def order_and_id_of_texts(found_polygons_text_region, found_polygons_text_region_h, matrix_of_orders, indexes_sorted, index_of_types, kind_of_texts, ref_point):
    indexes_sorted = np.array(indexes_sorted)
    index_of_types = np.array(index_of_types)
    kind_of_texts = np.array(kind_of_texts)

    id_of_texts = []
    order_of_texts = []

    index_of_types_1 = index_of_types[kind_of_texts == 1]
    indexes_sorted_1 = indexes_sorted[kind_of_texts == 1]

    index_of_types_2 = index_of_types[kind_of_texts == 2]
    indexes_sorted_2 = indexes_sorted[kind_of_texts == 2]

    counter = EynollahIdCounter(region_idx=ref_point)
    for idx_textregion, _ in enumerate(found_polygons_text_region):
        id_of_texts.append(counter.next_region_id)
        interest = indexes_sorted_1[indexes_sorted_1 == index_of_types_1[idx_textregion]]
        if len(interest) > 0:
            order_of_texts.append(interest[0])

    for idx_headerregion, _ in enumerate(found_polygons_text_region_h):
        id_of_texts.append(counter.next_region_id)
        interest = indexes_sorted_2[index_of_types_2[idx_headerregion]]
        order_of_texts.append(interest)

    return order_of_texts, id_of_texts
