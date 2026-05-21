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
        self._loaded: Dict[str, Union[Predictor, AnyModel]] = {}

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
        if model_path.suffix == '.h5' and Path(model_path.stem).exists():
            # prefer SavedModel over HDF5 format if it exists
            model_path = Path(model_path.stem)
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
            load_kwargs = dict(device=device)
            if isinstance(load_args, str):
                model_category, model_variant = load_args, ""
            elif len(load_args) > 2:
                # for calls to self.model_path
                self.override_models(load_args)
                # for calls to Predictor.load_model
                model_category, model_variant, model_path = load_args
                load_kwargs["model_variant"] = model_variant
                load_kwargs["model_path_override"] = model_path
            else:
                model_category, model_variant = load_args
                load_kwargs["model_variant"] = model_variant

            if model_category.endswith('_resized'):
                model_category = model_category[:-8]
                load_kwargs["resized"] = True
            elif model_category.endswith('_patched'):
                model_category = model_category[:-8]
                load_kwargs["patched"] = True

            if model_category == 'ocr':
                model = self._load_ocr_model(variant=model_variant, device=device)
            elif model_category == 'num_to_char':
                model = self._load_num_to_char()
            elif model_category == 'characters':
                model = self._load_characters()
            elif model_category == 'trocr_processor':
                from transformers import TrOCRProcessor
                model_path = self.model_path(model_category, model_variant)
                model = TrOCRProcessor.from_pretrained(model_path)
            else:
                model = Predictor(self.logger, self)
                model.load_model(model_category, **load_kwargs)

            ret[model_category] = model
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
        from tensorflow.keras.models import Model as KerasModel

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
                if ':' in device:
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
                # tf.config.experimental.set_memory_growth(device, True)
                # dynamic growth never frees memory (to avoid fragmentation),
                # so the VRAM requirements end up much larger than feasible
                # (for small GPUs); so try hard (calibrated) limits instead:
                tf.config.set_logical_device_configuration(
                    device,
                    [tf.config.LogicalDeviceConfiguration(memory_limit={
                        "binarization": 868, # due to bs 5
                        "enhancement": 980, # due to bs 3
                        "col_classifier": 210,
                        "page": 618,
                        "textline": 1680, # 954 for bs 1
                        "region_1_2": 1580,
                        "region_fl_np": 1756,
                        "table": 1818,
                        "reading_order": 632,
                        "ocr": 850,
                    }[model_category])])
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
        try:
            if model_path.is_dir() and not (model_path / "keras_metadata.pb").exists():
                raise ValueError()
            model = load_model(model_path, compile=False)
            model.make_predict_function()
        except (AttributeError, ValueError):
            model = tf.saved_model.load(model_path)
            model.predict_on_batch = model.serve
            model.input_shape = model.signatures.get('serving_default').inputs[0].shape
        model._name = model_category
        if resized:
            model = wrap_layout_model_resized(model)
            model._name = model_category + '_resized'
        elif patched:
            model = wrap_layout_model_patched(model)
            model._name = model_category + '_patched'
        else:
            # increases required VRAM, does not always work
            # (depending on CUDA/libcudnn/TF version):
            #model.jit_compile = True
            pass

        if model_category == 'ocr':
            model = KerasModel(
                model.get_layer(name="image").input,   # type: ignore
                model.get_layer(name="dense2").output, # type: ignore
        )

        return model

    def get(self, model_category: str) -> Union[Predictor, AnyModel]:
        if model_category not in self._loaded:
            raise ValueError(f'Model "{model_category}" not previously loaded with "load_model(..)"')
        return self._loaded[model_category]

    def _load_ocr_model(self, variant: str, device: str = "") -> AnyModel:
        """
        Load OCR model
        """
        model_dir = self.model_path('ocr', variant)
        if variant == 'tr':
            from transformers import VisionEncoderDecoderModel
            import torch
            model = VisionEncoderDecoderModel.from_pretrained(model_dir)
            assert isinstance(model, VisionEncoderDecoderModel)
            device0 = torch.device('cpu')
            if not device and torch.cuda.is_available():
                device = 'GPU' # try
            if device and ':' in device:
                for spec in device.split(','):
                    cat, dev = spec.split(':')
                    if fnmatchcase('ocr', cat):
                        device = dev
                        break
            if device and device.startswith('GPU'):
                try:
                    device0 = torch.device('cuda', int(device[3:] or 0))
                    name = torch.cuda.get_device_name(device0)
                    self.logger.info("using GPU %s (%s) for model ocr:tr", device0, name)
                except:
                    self.logger.exception("cannot configure GPU device")
                    device0 = torch.device('cpu')
            if device0.type == 'cuda':
                model.to(device0)
            else:
                self.logger.warning("no GPU device available")
            return model

        return self.load_model('ocr', model_variant=variant, device=device)

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
        os.environ['TF_USE_LEGACY_KERAS'] = '1' # avoid Keras 3 after TF 2.15
        from ocrd_utils import tf_disable_interactive_logs
        tf_disable_interactive_logs()

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
                if isinstance(self._loaded[needle], Predictor):
                    self._loaded[needle].shutdown()
