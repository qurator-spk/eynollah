import os
import json
import logging
from copy import deepcopy
from pathlib import Path
from fnmatch import fnmatchcase
from typing import Dict, List, Optional, Tuple, Type, Union

from tabulate import tabulate

from ..predictor import Predictor
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
        self.model_basedir = Path(basedir).resolve()
        self.logger = logging.getLogger('eynollah.model_zoo')
        if not self.model_basedir.exists():
            self.logger.warning(f"Model basedir does not exist: {basedir}. Set eynollah --model-basedir to the correct directory.")
        self.specs = deepcopy(DEFAULT_MODEL_SPECS)
        self._overrides = []
        if model_overrides:
            self.override_models(*model_overrides)
        self._loaded: Dict[str, Predictor] = {}

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
            self.specs.get(model_category, model_variant).filename = str(Path(model_filename).resolve())
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
        device: str = '',
    ) -> Dict:
        """
        Load all models by calling load_model and return a dictionary mapping model_category to loaded model
        """
        ret = {} # cannot use self._loaded here, yet – first spawn all predictors
        for load_args in all_load_args:
            if isinstance(load_args, str):
                model_category = load_args
                load_args = [model_category]
            else:
                model_category = load_args[0]
            load_kwargs = {}
            if model_category.endswith('_resized'):
                load_args[0] = model_category[:-8]
                load_kwargs["resized"] = True
            elif model_category.endswith('_patched'):
                load_args[0] = model_category[:-8]
                load_kwargs["patched"] = True
            ret[model_category] = Predictor(self.logger, self)
            ret[model_category].load_model(*load_args, **load_kwargs, device=device)
        self._loaded.update(ret)
        return self._loaded

    def load_model(
        self,
        model_category: str,
        model_variant: str = '',
        model_path_override: Optional[str] = None,
            patched: bool = False,
            resized: bool = False,
            device: str = '',
    ) -> AnyModel:
        """
        Load any model
        """
        os.environ['TF_USE_LEGACY_KERAS'] = '1' # avoid Keras 3 after TF 2.15
        from ocrd_utils import tf_disable_interactive_logs
        tf_disable_interactive_logs()

        import tensorflow as tf
        from tensorflow.keras.models import load_model

        from ..patch_encoder import (
            PatchEncoder,
            Patches,
            wrap_layout_model_patched,
            wrap_layout_model_resized,
        )
        cuda = False
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if device:
                if ',' in device:
                    for spec in device.split(','):
                        cat, dev = spec.split(':')
                        if fnmatchcase(model_category, cat):
                            device = dev
                            break
                if device == 'CPU':
                    gpus = []
                else:
                    assert device.startswith('GPU')
                    gpus = [gpus[int(device[3:])]]
            else:
                gpus = gpus[:1] # TF will always use first allowable
            tf.config.set_visible_devices(gpus, 'GPU')
            for device in gpus:
                tf.config.experimental.set_memory_growth(device, True)
                vendor_name = (
                    tf.config.experimental.get_device_details(device)
                    .get('device_name', 'unknown'))
                cuda = True
                self.logger.info("using GPU %s (%s) for model %s",
                                 device.name,
                                 vendor_name,
                                 model_category + (
                                     "_patched" if patched else
                                     "_resized" if resized else ""))
        except RuntimeError:
            self.logger.exception("cannot configure GPU devices")
        if not cuda:
            self.logger.warning("no GPU device available")

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
                # avoid wasting VRAM on non-transformer models
                model = load_model(model_path, compile=False)
            except Exception as e:
                self.logger.error(e)
                model = load_model(
                    model_path, compile=False,
                    custom_objects=dict(PatchEncoder=PatchEncoder,
                                        Patches=Patches))
            model._name = model_category
            if resized:
                model = wrap_layout_model_resized(model)
                model._name = model_category + '_resized'
            elif patched:
                model = wrap_layout_model_patched(model)
                model._name = model_category + '_patched'
            else:
                model.jit_compile = True
            model.make_predict_function()
        return model

    def get(self, model_category: str) -> Predictor:
        # if model_category not in self._loaded:
        #     raise ValueError(f'Model "{model_category}" not previously loaded with "load_model(..)"')
        if model_category in self._loaded:
            return self._loaded[model_category]
        else:
            return self.load_model(model_category)

    def _load_ocr_model(self, variant: str) -> AnyModel:
        """
        Load OCR model
        """
        from tensorflow.keras.models import Model as KerasModel
        from tensorflow.keras.models import load_model

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

    def _load_num_to_char(self) -> 'StringLookup':
        """
        Load decoder for OCR
        """
        from tensorflow.keras.layers import StringLookup

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
                self._loaded[needle].shutdown()
                del self._loaded[needle]
