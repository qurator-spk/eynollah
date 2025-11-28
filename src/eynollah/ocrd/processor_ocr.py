
from functools import cached_property
from frozendict import frozendict
from ocrd import Processor

from ..model_zoo.model_zoo import EynollahModelZoo


class EynollahOcrProcessor(Processor):
    # already employs GPU (without singleton process atm)
    max_workers = 1

    @cached_property
    def executable(self):
        return 'ocrd-eynollah-ocr'

    def setup(self):
        """
        Set up the model prior to processing.
        """
        # resolve relative path via OCR-D ResourceManager
        assert isinstance(self.parameter, frozendict)
        model_zoo = EynollahModelZoo(basedir=self.parameter['model'])
        raise NotImplementedError()
