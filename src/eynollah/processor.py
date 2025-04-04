from typing import Optional
from ocrd_models import OcrdPage
from ocrd import Processor, OcrdPageResult

from .eynollah import Eynollah, EynollahXmlWriter

class EynollahProcessor(Processor):
    # already employs background CPU multiprocessing per page
    # already employs GPU (without singleton process atm)
    max_workers = 1

    @property
    def executable(self):
        return 'ocrd-eynollah-segment'

    def setup(self) -> None:
        if self.parameter['textline_light'] and not self.parameter['light_version']:
            raise ValueError("Error: You set parameter 'textline_light' to enable light textline detection, "
                             "but parameter 'light_version' is not enabled")
        self.eynollah = Eynollah(
            self.resolve_resource(self.parameter['models']),
            logger=self.logger,
            allow_enhancement=self.parameter['allow_enhancement'],
            curved_line=self.parameter['curved_line'],
            right2left=self.parameter['right_to_left'],
            ignore_page_extraction=self.parameter['ignore_page_extraction'],
            light_version=self.parameter['light_version'],
            textline_light=self.parameter['textline_light'],
            full_layout=self.parameter['full_layout'],
            allow_scaling=self.parameter['allow_scaling'],
            headers_off=self.parameter['headers_off'],
            tables=self.parameter['tables'],
        )
        self.eynollah.plotter = None

    def shutdown(self):
        if hasattr(self, 'eynollah'):
            del self.eynollah

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """
        Performs cropping, region and line segmentation with Eynollah.

        For each page, open and deserialize PAGE input file (from existing
        PAGE file in the input fileGrp, or generated from image file).
        Retrieve its respective page-level image (ignoring annotation that
        already added `binarized`, `cropped` or `deskewed` features).

        Set up Eynollah to detect regions and lines, and add each one to the
        page, respectively.

        \b
        - If ``tables``, try to detect table blocks and add them as TableRegion.
        - If ``full_layout``, then in addition to paragraphs and marginals, also
          try to detect drop capitals and headings.
        - If ``ignore_page_extraction``, then attempt no cropping of the page.
        - If ``curved_line``, then compute contour polygons for text lines
          instead of simple bounding boxes.

        Produce a new output file by serialising the resulting hierarchy.
        """
        assert input_pcgts
        assert input_pcgts[0]
        assert self.parameter
        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        page = pcgts.get_Page()
        page_image, _, _ = self.workspace.image_from_page(
            page, page_id,
            # avoid any features that would change the coordinate system: cropped,deskewed
            # (the PAGE builder merely adds regions, so afterwards we would not know which to transform)
            # also avoid binarization as models usually fare better on grayscale/RGB
            feature_filter='cropped,deskewed,binarized')
        if hasattr(page_image, 'filename'):
            image_filename = page_image.filename
        else:
            image_filename = "dummy" # will be replaced by ocrd.Processor.process_page_file
            result.images.append(OcrdPageResultImage(page_image, '.IMG', page)) # mark as new original
        # FIXME: mask out already existing regions (incremental segmentation)
        self.eynollah.cache_images(
            image_pil=page_image,
            dpi=self.parameter['dpi'],
        )
        self.eynollah.writer = EynollahXmlWriter(
            dir_out=None,
            image_filename=image_filename,
            curved_line=self.eynollah.curved_line,
            textline_light=self.eynollah.textline_light,
            pcgts=pcgts)
        self.eynollah.run_single()
        return result
