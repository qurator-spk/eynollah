from typing import TypeVar

# NOTE: Creating an actual union type requires loading transformers which is expensive and error-prone
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# AnyModel = Union[VisionEncoderDecoderModel, TrOCRProcessor, KerasModel, List]
AnyModel = object
T = TypeVar('T')
