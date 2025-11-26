from functools import cached_property
from typing import Optional

from PIL import Image
from frozendict import frozendict
import numpy as np
import cv2
from click import command

from ocrd import Processor, OcrdPageResult, OcrdPageResultImage
from ocrd_models.ocrd_page import OcrdPage, AlternativeImageType
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from eynollah.model_zoo.model_zoo import EynollahModelZoo

from .sbb_binarize import SbbBinarizer


def cv2pil(img):
    return Image.fromarray(img.astype('uint8'))

def pil2cv(img):
    # from ocrd/workspace.py
    color_conversion = cv2.COLOR_GRAY2BGR if img.mode in ('1', 'L') else  cv2.COLOR_RGB2BGR
    pil_as_np_array = np.array(img).astype('uint8') if img.mode == '1' else np.array(img)
    return cv2.cvtColor(pil_as_np_array, color_conversion)

class SbbBinarizeProcessor(Processor):
    # already employs GPU (without singleton process atm)
    max_workers = 1

    @cached_property
    def executable(self):
        return 'ocrd-sbb-binarize'

    def setup(self):
        """
        Set up the model prior to processing.
        """
        # resolve relative path via OCR-D ResourceManager
        assert isinstance(self.parameter, frozendict)
        model_zoo = EynollahModelZoo(basedir=self.parameter['model'])
        self.binarizer = SbbBinarizer(model_zoo=model_zoo, logger=self.logger)

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """
        Binarize images with sbb_binarization (based on selectional auto-encoders).

        For each page of the input file group, open and deserialize input PAGE-XML
        and its respective images. Then iterate over the element hierarchy down to
        the requested ``operation_level``.

        For each segment element, retrieve a raw (non-binarized) segment image
        according to the layout  annotation (from an existing ``AlternativeImage``,
        or by cropping into the higher-level images, and deskewing when applicable).

        Pass the image to the binarizer (which runs in fixed-size windows/patches
        across the image and stitches the results together).

        Serialize the resulting bilevel image as PNG file and add it to the output
        file group (with file ID suffix ``.IMG-BIN``) along with the output PAGE-XML
        (referencing it as new ``AlternativeImage`` for the segment element).

        Produce a new PAGE output file by serialising the resulting hierarchy.
        """
        assert input_pcgts
        assert input_pcgts[0]
        assert self.parameter
        oplevel = self.parameter['operation_level']
        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        page = pcgts.get_Page()
        page_image, page_xywh, _ = self.workspace.image_from_page(
            page, page_id, feature_filter='binarized')

        if oplevel == 'page':
            self.logger.info("Binarizing on 'page' level in page '%s'", page_id)
            page_image_bin = cv2pil(self.binarizer.run(image=pil2cv(page_image), use_patches=True))
            # update PAGE (reference the image file):
            page_image_ref = AlternativeImageType(comments=page_xywh['features'] + ',binarized,clipped')
            page.add_AlternativeImage(page_image_ref)
            result.images.append(OcrdPageResultImage(page_image_bin, '.IMG-BIN', page_image_ref))

        elif oplevel == 'region':
            regions = page.get_AllRegions(['Text', 'Table'], depth=1)
            if not regions:
                self.logger.warning("Page '%s' contains no text/table regions", page_id)
            for region in regions:
                region_image, region_xywh = self.workspace.image_from_segment(
                    region, page_image, page_xywh, feature_filter='binarized')
                region_image_bin = cv2pil(self.binarizer.run(image=pil2cv(region_image), use_patches=True))
                # update PAGE (reference the image file):
                region_image_ref = AlternativeImageType(comments=region_xywh['features'] + ',binarized')
                region.add_AlternativeImage(region_image_ref)
                result.images.append(OcrdPageResultImage(region_image_bin, region.id + '.IMG-BIN', region_image_ref))

        elif oplevel == 'line':
            lines = page.get_AllTextLines()
            if not lines:
                self.logger.warning("Page '%s' contains no text lines", page_id)
            for line in lines:
                line_image, line_xywh = self.workspace.image_from_segment(line, page_image, page_xywh, feature_filter='binarized')
                line_image_bin = cv2pil(self.binarizer.run(image=pil2cv(line_image), use_patches=True))
                # update PAGE (reference the image file):
                line_image_ref = AlternativeImageType(comments=line_xywh['features'] + ',binarized')
                line.add_AlternativeImage(line_image_ref)
                result.images.append(OcrdPageResultImage(line_image_bin, line.id + '.IMG-BIN', line_image_ref))

        return result

@command()
@ocrd_cli_options
def main(*args, **kwargs):
    return ocrd_cli_wrap_processor(SbbBinarizeProcessor, *args, **kwargs)
