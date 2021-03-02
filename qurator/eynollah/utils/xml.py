# pylint: disable=too-many-locals,wrong-import-position,too-many-lines,too-many-statements,chained-comparison,fixme,broad-except,c-extension-no-member
from lxml import etree as ET
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

def xml_reading_order(page, order_of_texts, id_of_texts, id_of_marginalia, found_polygons_marginals):
    """
    XXX side-effect: extends id_of_marginalia
    """
    region_order = ET.SubElement(page, 'ReadingOrder')
    region_order_sub = ET.SubElement(region_order, 'OrderedGroup')
    region_order_sub.set('id', "ro357564684568544579089")
    indexer_region = 0
    for vj in order_of_texts:
        name = "coord_text_%s" % vj
        name = ET.SubElement(region_order_sub, 'RegionRefIndexed')
        name.set('index', str(indexer_region))
        name.set('regionRef', id_of_texts[vj])
        indexer_region += 1
    for vm in range(len(found_polygons_marginals)):
        id_of_marginalia.append('r%s' % indexer_region)
        name = "coord_text_%s" % indexer_region
        name = ET.SubElement(region_order_sub, 'RegionRefIndexed')
        name.set('index', str(indexer_region))
        name.set('regionRef', 'r%s' % indexer_region)
        indexer_region += 1
    return id_of_marginalia

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

    index_b = 0 + ref_point
    for mm, _ in enumerate(found_polygons_text_region):
        id_of_texts.append("r" + str(index_b))
        interest = indexes_sorted_1[indexes_sorted_1 == index_of_types_1[mm]]
        if len(interest) > 0:
            order_of_texts.append(interest[0])
            index_b += 1

    for mm, _ in enumerate(found_polygons_text_region_h):
        id_of_texts.append("r" + str(index_b))
        interest = indexes_sorted_2[index_of_types_2[mm]]
        order_of_texts.append(interest)
        index_b += 1

    return order_of_texts, id_of_texts

