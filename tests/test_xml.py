from eynollah.utils.xml import create_page_xml
from ocrd_models.ocrd_page import to_xml

PAGE_2019 = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'

def test_create_xml():
    pcgts = create_page_xml('/path/to/img.tif', 100, 100)
    xmlstr = to_xml(pcgts)
    assert 'xmlns:pc="%s"' % PAGE_2019 in xmlstr
    assert 'Metadata' in xmlstr
