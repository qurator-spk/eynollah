# pylint: disable=too-many-locals,wrong-import-position,too-many-lines,too-many-statements,chained-comparison,fixme,broad-except,c-extension-no-member
# pylint: disable=invalid-name
from lxml import etree as ET
from .counter import EynollahIdCounter
import numpy as np

NAMESPACES = {}
NAMESPACES['page'] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
NAMESPACES['xsi'] = "http://www.w3.org/2001/XMLSchema-instance"
NAMESPACES[None] = NAMESPACES['page']

def create_page_xml(imageFilename, height, width):
    pcgts = ET.Element("PcGts", nsmap=NAMESPACES)

    pcgts.set("{%s}schemaLocation" % NAMESPACES['xsi'], NAMESPACES['page'])

    metadata = ET.SubElement(pcgts, "Metadata")

    author = ET.SubElement(metadata, "Creator")
    author.text = "SBB_QURATOR"

    created = ET.SubElement(metadata, "Created")
    created.text = "2019-06-17T18:15:12"

    changetime = ET.SubElement(metadata, "LastChange")
    changetime.text = "2019-06-17T18:15:12"

    page = ET.SubElement(pcgts, "Page")

    page.set("imageFilename", imageFilename)
    page.set("imageHeight", str(height))
    page.set("imageWidth", str(width))
    page.set("type", "content")
    page.set("readingDirection", "left-to-right")
    page.set("textLineOrder", "top-to-bottom")

    return pcgts, page

def add_textequiv(parent, text=''):
    textequiv = ET.SubElement(parent, 'TextEquiv')
    unireg = ET.SubElement(textequiv, 'Unicode')
    unireg.text = text

def xml_reading_order(page, order_of_texts, id_of_texts, found_polygons_marginals):
    region_order = ET.SubElement(page, 'ReadingOrder')
    region_order_sub = ET.SubElement(region_order, 'OrderedGroup')
    region_order_sub.set('id', "ro357564684568544579089")
    indexer_region = 0
    for idx_text in order_of_texts:
        name = ET.SubElement(region_order_sub, 'RegionRefIndexed')
        name.set('index', str(indexer_region))
        name.set('regionRef', id_of_texts[idx_text])
        indexer_region += 1
    for  _ in found_polygons_marginals:
        name = ET.SubElement(region_order_sub, 'RegionRefIndexed')
        name.set('index', str(indexer_region))
        name.set('regionRef', 'r%s' % indexer_region)
        indexer_region += 1

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
