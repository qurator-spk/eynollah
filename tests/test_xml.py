from lxml import etree as ET
from sbb_newspapers_org_image.utils.xml import create_page_xml, NAMESPACES

def tostring(el):
    return ET.tostring(el).decode('utf-8')

def test_create_xml():
    pcgts, page = create_page_xml('/path/to/img.tif', 100, 100)
    xmlstr = tostring(pcgts)
    assert 'xmlns="%s"' % NAMESPACES[None] in xmlstr
    assert 'Metadata' in xmlstr
