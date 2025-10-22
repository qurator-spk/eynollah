from typing import List, TypeVar, Union
from keras.models import Model as KerasModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

AnyModel = Union[VisionEncoderDecoderModel, TrOCRProcessor, KerasModel, List]
T = TypeVar('T')
