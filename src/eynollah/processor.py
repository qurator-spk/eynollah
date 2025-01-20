from typing import Optional
from ocrd_models import OcrdPage
from ocrd import Processor, OcrdPageResult

from .eynollah import Eynollah

class EynollahProcessor(Processor):

    def setup(self) -> None:
        # for caching models
        self.models = None
        if self.parameter['textline_light'] and not self.parameter['light_mode']:
            raise ValueError("Error: You set parameter 'textline_light' to enable light textline detection but parameter 'light_mode' is not enabled")

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        assert input_pcgts
        assert input_pcgts[0]
        assert self.parameter
        pcgts = input_pcgts[0]
        page = pcgts.get_Page()
        # if not('://' in page.imageFilename):
        #     image_filename = next(self.workspace.mets.find_files(local_filename=page.imageFilename)).local_filename
        # else:
        #     # could be a URL with file:// or truly remote
        #     image_filename = self.workspace.download_file(next(self.workspace.mets.find_files(url=page.imageFilename))).local_filename
        page_image, _, _ = self.workspace.image_from_page(
            page, page_id,
            # avoid any features that would change the coordinate system: cropped,deskewed
            # (the PAGE builder merely adds regions, so afterwards we would not know which to transform)
            # also avoid binarization as models usually fare better on grayscale/RGB
            feature_filter='cropped,deskewed,binarized')
        eynollah = Eynollah(
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
            image_filename=page.imageFilename,
            image_pil=page_image
        )
        if self.models is not None:
            # reuse loaded models from previous page
            eynollah.models = self.models
        eynollah.run()
        self.models = eynollah.models
        return OcrdPageResult(pcgts)
