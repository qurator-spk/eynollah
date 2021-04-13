from json import loads
from pkg_resources import resource_string
from tempfile import NamedTemporaryFile
from os.path import join

from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import to_xml
from ocrd_utils import (
    getLogger,
    MIMETYPE_PAGE,
    assert_file_grp_cardinality,
    make_file_id
)

from .eynollah import Eynollah

OCRD_TOOL = loads(resource_string(__name__, 'ocrd-tool.json').decode('utf8'))

class EynollahProcessor(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-eynollah-segment']
        kwargs['version'] = OCRD_TOOL['version']
        super().__init__(*args, **kwargs)

    def process(self):
        LOG = getLogger('eynollah')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %s / %s ", page_id, len(self.input_files))
            pcgts_in = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts_in)
            page = pcgts_in.get_Page()
            page_image, _, _ = self.workspace.image_from_page(page, page_id, feature_filter='binarized')
            file_id = make_file_id(input_file, self.output_file_grp)
            with NamedTemporaryFile(buffering=0, suffix='.tif') as f:
                page_image.save(f.name)
                eynollah_kwargs = {
                    'dir_models': self.resolve_resource(self.parameter['models']),
                    'allow_enhancement': self.parameter['allow_enhancement'],
                    'curved_line': self.parameter['curved_line'],
                    'full_layout': self.parameter['full_layout'],
                    'allow_scaling': self.parameter['allow_scaling'],
                    'headers_off': self.parameter['headers_off'],
                    'override_dpi': self.parameter['dpi'] if self.parameter['dpi'] > 0 else None,
                    'logger': LOG,
                    'image_filename': f.name}
                pcgts_out = Eynollah(**eynollah_kwargs).run()
                pcgts_out.get_Page().imageFilename = pcgts_in.get_Page().imageFilename
                self.workspace.add_file(
                    ID=file_id,
                    file_grp=self.output_file_grp,
                    pageId=page_id,
                    mimetype=MIMETYPE_PAGE,
                    local_filename=join(self.output_file_grp, file_id) + '.xml',
                    content=to_xml(pcgts_out))
