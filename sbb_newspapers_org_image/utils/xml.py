from lxml import etree as ET

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

