from functools import cached_property
from pathlib import Path
from typing import Optional
from click import command
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd.workspace import page_from_file
from ocrd_models import OcrdFileType, OcrdPage

from ocrd import Processor
from ocrd_utils import (
    make_file_id,
)

from eynollah.eynollah_ocr import Eynollah_ocr
from eynollah.model_zoo.model_zoo import EynollahModelZoo
from eynollah.utils.pil_cv2 import pil2cv
from eynollah.utils.xml import etree_namespace_for_element_tag


class EynollahRecognizeProcessor(Processor):

    @cached_property
    def executable(self):
        return 'ocrd-eynollah-recognize'

    def setup(self):
        """
        Load model, set predict function
        """
        assert self.parameter
        model_zoo = EynollahModelZoo(basedir=self.parameter['models'])
        assert self.parameter
        self.eynollah_ocr = Eynollah_ocr(
            model_zoo=model_zoo,
            tr_ocr=self.parameter['tr_ocr'],
            do_not_mask_with_textline_contour=self.parameter['do_not_mask_with_textline_contour'],
            batch_size=self.parameter['batch_size'] if self.parameter['batch_size'] >= 0 else 2 if self.parameter['tr_ocr'] else 8,
            min_conf_value_of_textline_text=0)

    # FIXME: This is just a proof-of-concept, very inefficient and non-conformant
    # TODO: OCR writing should use PAGE API once result dataclass mechanism is settled,
    #       then simplify/port to proces_page_pcgts
    def process_page_file(self, *input_files: Optional[OcrdFileType]) -> None:
        assert self.workspace
        page_file = input_files[0]
        assert page_file
        page = page_from_file(page_file)
        assert page
        page_image, page_coords, _ = self.workspace.image_from_page(
            page.get_Page(), page_file.pageId,
            feature_selector="")
        page_ns = etree_namespace_for_element_tag(page.etree.tag)

        img = pil2cv(page_image)
        if self.eynollah_ocr.tr_ocr:
            result = self.eynollah_ocr.run_trocr(
                img=img,
                page_tree=page.etree,
                page_ns=page_ns,

                tr_ocr_input_height_and_width = 384
            )
        else:
            page_image_bin, _, _ = self.workspace.image_from_page(
                page.get_Page(), page_file.pageId,
                feature_selector="binarized")
            result = self.eynollah_ocr.run_cnn( 
                img=img,
                page_tree=page.etree,
                page_ns=page_ns,

                img_bin=pil2cv(page_image_bin),
                image_width=512,
                image_height=32,
            )
        output_file_id = make_file_id(page_file, self.output_file_grp)
        output_filename = Path(self.output_file_grp, output_file_id + '.xml')
        output_filename.parent.mkdir()
        self.eynollah_ocr.write_ocr(
            result=result,
            img=img,
            page_tree=page.etree,
            page_ns=page_ns,
            out_file_ocr=str(output_filename),
            out_image_with_text=None,
        )
        self.workspace.add_file(
            file_id=output_file_id,
            file_grp=self.output_file_grp,
            page_id=page_file.pageId,
            local_filename=output_filename,
            mimetype=page_ns,
        )

@command()
@ocrd_cli_options
def main(*args, **kwargs):
    return ocrd_cli_wrap_processor(EynollahRecognizeProcessor, *args, **kwargs)
