import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

from ocrd_utils import tf_disable_interactive_logs
tf_disable_interactive_logs()

from keras.layers import StringLookup
from keras.models import Model as KerasModel
from keras.models import load_model
from tabulate import tabulate
from ..patch_encoder import PatchEncoder, Patches
from .specs import EynollahModelSpecSet
from .default_specs import DEFAULT_MODEL_SPECS
from .types import AnyModel, T


class EynollahModelZoo:
    """
    Wrapper class that handles storage and loading of models for all eynollah runners.
    """

    model_basedir: Path
    specs: EynollahModelSpecSet

    def __init__(
        self,
        basedir: str,
        model_overrides: Optional[List[Tuple[str, str, str]]] = None,
    ) -> None:
        self.model_basedir = Path(basedir)
        self.logger = logging.getLogger('eynollah.model_zoo')
        if not self.model_basedir.exists():
            self.logger.warning(f"Model basedir does not exist: {basedir}. Set eynollah --model-basedir to the correct directory.")
        self.specs = deepcopy(DEFAULT_MODEL_SPECS)
        self._overrides = []
        if model_overrides:
            self.override_models(*model_overrides)
        self._loaded: Dict[str, AnyModel] = {}

    @property
    def model_overrides(self):
        return self._overrides

    def override_models(
        self,
        *model_overrides: Tuple[str, str, str],
    ):
        """
        Override the default model versions
        """
        for model_category, model_variant, model_filename in model_overrides:
            spec = self.specs.get(model_category, model_variant)
            self.logger.warning("Overriding filename for model spec %s to %s", spec, model_filename)
            self.specs.get(model_category, model_variant).filename = model_filename
        self._overrides += model_overrides

    def model_path(
        self,
        model_category: str,
        model_variant: str = '',
        absolute: bool = True,
    ) -> Path:
        """
        Translate model_{type,variant} tuple into an absolute (or relative) Path
        """
        spec = self.specs.get(model_category, model_variant)
        if spec.category in ('characters', 'num_to_char'):
            return self.model_path('ocr') / spec.filename
        if not Path(spec.filename).is_absolute() and absolute:
            model_path = Path(self.model_basedir).joinpath(spec.filename)
        else:
            model_path = Path(spec.filename)
        return model_path

    def load_models(
        self,
        *all_load_args: Union[str, Tuple[str], Tuple[str, str], Tuple[str, str, str]],
    ) -> Dict:
        """
        Load all models by calling load_model and return a dictionary mapping model_category to loaded model
        """
        ret = {}
        for load_args in all_load_args:
            if isinstance(load_args, str):
                ret[load_args] = self.load_model(load_args)
            else:
                ret[load_args[0]] = self.load_model(*load_args)
        return ret

    def load_model(
        self,
        model_category: str,
        model_variant: str = '',
        model_path_override: Optional[str] = None,
    ) -> AnyModel:
        """
        Load any model
        """
        if model_path_override:
            self.override_models((model_category, model_variant, model_path_override))
        model_path = self.model_path(model_category, model_variant)
        if model_path.suffix == '.h5' and Path(model_path.stem).exists():
            # prefer SavedModel over HDF5 format if it exists
            model_path = Path(model_path.stem)
        if model_category == 'ocr':
            model = self._load_ocr_model(variant=model_variant)
        elif model_category == 'num_to_char':
            model = self._load_num_to_char()
        elif model_category == 'characters':
            model = self._load_characters()
        elif model_category == 'trocr_processor':
            from transformers import TrOCRProcessor
            model = TrOCRProcessor.from_pretrained(model_path)
        else:
            try:
                model = load_model(model_path, compile=False)
            except Exception as e:
                self.logger.exception(e)
                model = load_model(
                    model_path, compile=False, custom_objects={"PatchEncoder": PatchEncoder, "Patches": Patches}
                )
        self._loaded[model_category] = model
        return model  # type: ignore

    def get(self, model_category: str, model_type: Optional[Type[T]] = None) -> T:
        if model_category not in self._loaded:
            raise ValueError(f'Model "{model_category} not previously loaded with "load_model(..)"')
        ret = self._loaded[model_category]
        if model_type:
            assert isinstance(ret, model_type)
        return ret  # type: ignore # FIXME: convince typing that we're returning generic type

    def _load_ocr_model(self, variant: str) -> AnyModel:
        """
        Load OCR model
        """
        ocr_model_dir = self.model_path('ocr', variant)
        if variant == 'tr':
            from transformers import VisionEncoderDecoderModel
            ret = VisionEncoderDecoderModel.from_pretrained(ocr_model_dir)
            assert isinstance(ret, VisionEncoderDecoderModel)
            return ret
        else:
            ocr_model = load_model(ocr_model_dir, compile=False)
            assert isinstance(ocr_model, KerasModel)
            return KerasModel(
                ocr_model.get_layer(name="image").input,   # type: ignore
                ocr_model.get_layer(name="dense2").output, # type: ignore
            )

    def _load_characters(self) -> List[str]:
        """
        Load encoding for OCR
        """
        with open(self.model_path('num_to_char'), "r") as config_file:
            return json.load(config_file)

    def _load_num_to_char(self) -> StringLookup:
        """
        Load decoder for OCR
        """
        characters = self._load_characters()
        # Mapping characters to integers.
        char_to_num = StringLookup(vocabulary=characters, mask_token=None)
        # Mapping integers back to original characters.
        return StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

    def __str__(self):
        return tabulate(
            [
                [
                    spec.type,
                    spec.category,
                    spec.variant,
                    spec.help,
                    f'Yes, at {self.model_path(spec.category, spec.variant)}'
                    if self.model_path(spec.category, spec.variant).exists()
                    else f'No, download {spec.dist_url}',
                    # self.model_path(spec.category, spec.variant),
                ]
                for spec in sorted(self.specs.specs, key=lambda x: x.dist_url)
            ],
            headers=[
                'Type',
                'Category',
                'Variant',
                'Help',
                'Used in',
                'Installed',
            ],
            tablefmt='github',
        )

    def shutdown(self):
        """
        Ensure that a loaded models is not referenced by ``self._loaded`` anymore
        """
        if hasattr(self, '_loaded') and getattr(self, '_loaded'):
            for needle in list(self._loaded.keys()):
                del self._loaded[needle]
