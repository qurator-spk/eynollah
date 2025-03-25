from os import environ
from os.path import join
from pathlib import Path
from pkg_resources import resource_string
from json import loads

from PIL import Image
import numpy as np
import cv2
from click import command

from ocrd_utils import (
    getLogger,
    assert_file_grp_cardinality,
    make_file_id,
    MIMETYPE_PAGE
)
from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import AlternativeImageType, to_xml
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from .sbb_binarize import SbbBinarizer

OCRD_TOOL = loads(resource_string(__name__, 'ocrd-tool-binarization.json').decode('utf8'))
TOOL = 'ocrd-sbb-binarize'

def cv2pil(img):
    return Image.fromarray(img.astype('uint8'))

def pil2cv(img):
    # from ocrd/workspace.py
    color_conversion = cv2.COLOR_GRAY2BGR if img.mode in ('1', 'L') else  cv2.COLOR_RGB2BGR
    pil_as_np_array = np.array(img).astype('uint8') if img.mode == '1' else np.array(img)
    return cv2.cvtColor(pil_as_np_array, color_conversion)

class SbbBinarizeProcessor(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super().__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            # processing context
            self.setup()

    def setup(self):
        """
        Set up the model prior to processing.
        """
        LOG = getLogger('processor.SbbBinarize.__init__')
        if not 'model' in self.parameter:
            raise ValueError("'model' parameter is required")
        # resolve relative path via environment variable
        model_path = Path(self.parameter['model'])
        if not model_path.is_absolute():
            if 'SBB_BINARIZE_DATA' in environ and environ['SBB_BINARIZE_DATA']:
                LOG.info("Environment variable SBB_BINARIZE_DATA is set to '%s'" \
                         " - prepending to model value '%s'. If you don't want this mechanism," \
                         " unset the SBB_BINARIZE_DATA environment variable.",
                         environ['SBB_BINARIZE_DATA'], model_path)
                model_path = Path(environ['SBB_BINARIZE_DATA']).joinpath(model_path)
                model_path = model_path.resolve()
                if not model_path.is_dir():
                    raise FileNotFoundError("Does not exist or is not a directory: %s" % model_path)
        # resolve relative path via OCR-D ResourceManager
        model_path = self.resolve_resource(str(model_path))
        self.binarizer = SbbBinarizer(model_dir=model_path, logger=LOG)

    def process(self):
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
        LOG = getLogger('processor.SbbBinarize')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        oplevel = self.parameter['operation_level']

        for n, input_file in enumerate(self.input_files):
            file_id = make_file_id(input_file, self.output_file_grp)
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            pcgts.set_pcGtsId(file_id)
            page = pcgts.get_Page()
            page_image, page_xywh, _ = self.workspace.image_from_page(page, page_id, feature_filter='binarized')

            if oplevel == 'page':
                LOG.info("Binarizing on 'page' level in page '%s'", page_id)
                bin_image = cv2pil(self.binarizer.run(image=pil2cv(page_image), use_patches=True))
                # update METS (add the image file):
                bin_image_path = self.workspace.save_image_file(bin_image,
                        file_id + '.IMG-BIN',
                        page_id=input_file.pageId,
                        file_grp=self.output_file_grp)
                page.add_AlternativeImage(AlternativeImageType(filename=bin_image_path, comments='%s,binarized' % page_xywh['features']))

            elif oplevel == 'region':
                regions = page.get_AllRegions(['Text', 'Table'], depth=1)
                if not regions:
                    LOG.warning("Page '%s' contains no text/table regions", page_id)
                for region in regions:
                    region_image, region_xywh = self.workspace.image_from_segment(region, page_image, page_xywh, feature_filter='binarized')
                    region_image_bin = cv2pil(binarizer.run(image=pil2cv(region_image), use_patches=True))
                    region_image_bin_path = self.workspace.save_image_file(
                            region_image_bin,
                            "%s_%s.IMG-BIN" % (file_id, region.id),
                            page_id=input_file.pageId,
                            file_grp=self.output_file_grp)
                    region.add_AlternativeImage(
                        AlternativeImageType(filename=region_image_bin_path, comments='%s,binarized' % region_xywh['features']))

            elif oplevel == 'line':
                region_line_tuples = [(r.id, r.get_TextLine()) for r in page.get_AllRegions(['Text'], depth=0)]
                if not region_line_tuples:
                    LOG.warning("Page '%s' contains no text lines", page_id)
                for region_id, line in region_line_tuples:
                    line_image, line_xywh = self.workspace.image_from_segment(line, page_image, page_xywh, feature_filter='binarized')
                    line_image_bin = cv2pil(binarizer.run(image=pil2cv(line_image), use_patches=True))
                    line_image_bin_path = self.workspace.save_image_file(
                            line_image_bin,
                            "%s_%s_%s.IMG-BIN" % (file_id, region_id, line.id),
                            page_id=input_file.pageId,
                            file_grp=self.output_file_grp)
                    line.add_AlternativeImage(
                        AlternativeImageType(filename=line_image_bin_path, comments='%s,binarized' % line_xywh['features']))

            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=join(self.output_file_grp, file_id + '.xml'),
                content=to_xml(pcgts))

@command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(SbbBinarizeProcessor, *args, **kwargs)
