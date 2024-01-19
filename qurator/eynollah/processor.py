from typing import Optional
from ocrd.processor.ocrd_page_result import OcrdPageResult
from ocrd_models import OcrdPage
from ocrd import Processor

from .eynollah import Eynollah

class EynollahProcessor(Processor):

    @property
    def metadata_filename(self) -> str:
        return 'eynollah/ocrd-tool.json'

    def setup(self) -> None:
        if self.parameter['textline_light'] and not self.parameter['light_mode']:
            raise ValueError("Error: You set parameter 'textline_light' to enable light textline detection but parameter 'light_mode' is not enabled")

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        assert input_pcgts
        assert input_pcgts[0]
        assert self.parameter
        pcgts = input_pcgts[0]
        page = pcgts.get_Page()
        # XXX loses DPI information
        # page_image, _, _ = self.workspace.image_from_page(page, page_id, feature_filter='binarized')
        if not('://' in page.imageFilename):
            image_filename = next(self.workspace.mets.find_files(local_filename=page.imageFilename)).local_filename
        else:
            # could be a URL with file:// or truly remote
            image_filename = self.workspace.download_file(next(self.workspace.mets.find_files(url=page.imageFilename))).local_filename
        Eynollah(
            self.resolve_resource(self.parameter['models']),
            self.logger,
            allow_enhancement=self.parameter['allow_enhancement'],
            curved_line=self.parameter['curved_line'],
            light_version=self.parameter['light_mode'],
            right2left=self.parameter['right_to_left'],
            ignore_page_extraction=self.parameter['ignore_page_extraction'],
            textline_light=self.parameter['textline_light'],
            full_layout=self.parameter['full_layout'],
            allow_scaling=self.parameter['allow_scaling'],
            headers_off=self.parameter['headers_off'],
            tables=self.parameter['tables'],
            override_dpi=self.parameter['dpi'],
            pcgts=pcgts,
            image_filename=image_filename
        ).run()
        return OcrdPageResult(pcgts)
