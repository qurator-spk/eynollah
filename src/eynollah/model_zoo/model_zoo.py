from __future__ import annotations
import os
import json
import logging
from copy import deepcopy
from pathlib import Path
from fnmatch import fnmatchcase

from tabulate import tabulate

from ..predictor import Predictor
from .specs import EynollahModelSpecSet
from .default_specs import DEFAULT_MODEL_SPECS
from .types import AnyModel, T


MODEL_VRAM_LIMITS = {
    "binarization": 868, # due to bs 5
    "enhancement": 980, # due to bs 3
    "col_classifier": 210,
    "page": 618,
    "textline": 1880, # 954 for bs 1
    "region_1_2": 1580,
    "region_fl_np": 1756,
    "table": 1818,
    "reading_order": 632,
    "ocr": 2600, # 850 for bs 8
    "extract_images": 954,
}

class EynollahModelZoo:
    """
    Wrapper class that handles storage and loading of models for all eynollah runners.
    """

    model_basedir: Path
    specs: EynollahModelSpecSet

    def __init__(
        self,
        basedir: str,
        model_overrides: list[tuple[str, str, str]] | None = None,
    ) -> None:
        self.model_basedir = Path(basedir).resolve()
        self.logger = logging.getLogger('eynollah.model_zoo')
        if not self.model_basedir.exists():
            self.logger.warning(f"Model basedir does not exist: {basedir}. Set eynollah --model-basedir to the correct directory.")
        self.specs = deepcopy(DEFAULT_MODEL_SPECS)
        self._overrides = []
        if model_overrides:
            self.override_models(*model_overrides)
        self._loaded: dict[str, Predictor] = {}

    @property
    def model_overrides(self):
        return self._overrides

    def override_models(
        self,
        *model_overrides: tuple[str, str, str],
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
        if model_path.with_suffix('.onnx').exists():
            # prefer ONNX over SavedModel format if it exists
            model_path = model_path.with_suffix('.onnx')

        return model_path

    def load_models(
        self,
        *all_load_args: str | tuple[str] | tuple[str, str] | tuple[str, str, str],
        device: str = '',
    ) -> dict:
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

            # if model_category.endswith('_resized'):
            #     model_category = model_category[:-8]
            #     load_kwargs["resized"] = True
            # elif model_category.endswith('_patched'):
            #     model_category = model_category[:-8]
            #     load_kwargs["patched"] = True

            model = Predictor(self.logger, self)
            model.load_model(model_category, **load_kwargs)

            ret[model_category] = model
        self._loaded.update(ret)
        return self._loaded

    def load_model(
            self,
            model_category: str,
            model_variant: str = '',
            model_path_override: str | None = None,
            # patched: bool = False,
            # resized: bool = False,
            device: str = '',
    ) -> AnyModel:
        """
        Load any model
        """
        if model_path_override:
            self.override_models((model_category, model_variant, model_path_override))
        model_path = self.model_path(model_category, model_variant)

        if model_category == 'ocr' and model_variant == 'tr':
            model = self._load_trocr_model(model_path, device=device)
        elif model_path.is_dir() and (model_path / "keras_metadata.pb").exists():
            # Keras model
            model = self._load_keras_model(model_category, model_path, device=device)
        elif model_path.is_dir():
            # TF-Serving model
            model = self._load_serving_model(model_category, model_path, device=device)
        elif model_path.suffix == '.onnx':
            # ONNX model
            model = self._load_onnx_model(model_category, model_path, device=device)
        else:
            raise ValueError("unknown model type for '%s'" % str(model_path))
        model._name = model_category
        return model

    def get(self, model_category: str) -> Predictor:
        if model_category not in self._loaded:
            raise ValueError(f'Model "{model_category}" not previously loaded with "load_model(..)"')
        return self._loaded[model_category]

    def _configure_tf_device(self, model_category, device=''):
        from ocrd_utils import tf_disable_interactive_logs
        tf_disable_interactive_logs()
        import tensorflow as tf

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
                if ':' in device:
                    self.logger.warning("missing device specification for model type %s", model_category)
                    gpus = gpus[:1]
                elif device == 'CPU':
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
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=MODEL_VRAM_LIMITS[model_category])])
                vendor_name = (
                    tf.config.experimental.get_device_details(device)
                    .get('device_name', 'unknown'))
                cuda = True
                self.logger.info("using GPU %s (%s) for model %s",
                                 device.name,
                                 vendor_name,
                                 model_category # + (
                                     # "_patched" if patched else
                                     # "_resized" if resized else "")
                )
        except RuntimeError:
            self.logger.exception("cannot configure GPU devices")
        if not cuda:
            self.logger.warning("no GPU device available")

    def _configure_torch_device(self, model_category, device=''):
        import torch

        device0 = torch.device('cpu')
        if not device and torch.cuda.is_available():
            device = 'GPU' # try
        if device and ':' in device:
            for spec in device.split(','):
                cat, dev = spec.split(':')
                if fnmatchcase('ocr', cat):
                    device = dev
                    break
            if ':' in device:
                self.logger.warning("missing device specification for model type %s", model_category)
                device = 'GPU'
        if device and device.startswith('GPU'):
            try:
                device0 = torch.device('cuda', int(device[3:] or 0))
                name = torch.cuda.get_device_name(device0)
                self.logger.info("using GPU %s (%s) for model ocr:tr", device0, name)
            except:
                self.logger.exception("cannot configure GPU device")
                device0 = torch.device('cpu')
        if device0.type != 'cuda':
            self.logger.warning("no GPU device available")
        return device0

    def _load_keras_model(self, model_category, model_path, device='') -> AnyModel:
        os.environ['TF_USE_LEGACY_KERAS'] = '1' # avoid Keras 3 after TF 2.15
        from ocrd_utils import tf_disable_interactive_logs
        tf_disable_interactive_logs()

        from tensorflow.keras.models import load_model
        from tensorflow.keras.models import Model as KerasModel

        from ..training.models import cnn_rnn_ocr_model4inference

        self._configure_tf_device(model_category, device=device)

        model = load_model(model_path, compile=False)
        assert isinstance(model, KerasModel)

        # from ..patch_encoder import (
        #     wrap_layout_model_patched,
        #     wrap_layout_model_resized,
        # )
        # if resized:
        #     model = wrap_layout_model_resized(model)
        #     model._name = model_category + '_resized'
        # elif patched:
        #     model = wrap_layout_model_patched(model)
        #     model._name = model_category + '_patched'

        if model_category == 'ocr':
            # cnn-rnn-ocr task model may not be in inference mode, yet
            model = cnn_rnn_ocr_model4inference(model, model_path)

        model.make_predict_function()

        return model

    def _load_serving_model(self, model_category, model_path, device='') -> AnyModel:
        from ocrd_utils import tf_disable_interactive_logs
        tf_disable_interactive_logs()
        import tensorflow as tf

        self._configure_tf_device(model_category, device=device)
        model = tf.saved_model.load(model_path)
        model.predict_on_batch = model.serve
        spec = model.signatures['serving_default']
        # some models receive lots of additional/internal
        # (unknown) captured inputs polluting .inputs
        # TF>=2.16 has spec.function_type.flat_inputs
        # this non-public API works:
        # input_spec = spec.inputs[:len(spec._arg_keywords)]
        # but perhaps this is most reliable:
        input_spec = tf.nest.flatten(spec.structured_input_signature, True)
        input_spec = [tuple(i.shape) for i in input_spec]
        if len(input_spec) > 1:
            model.input_shape = tuple(input_spec)
        else:
            model.input_shape = input_spec[0]

        return model

    def _load_onnx_model(self, model_category, model_path, device='') -> AnyModel:
        import onnxruntime as ort
        import numpy as np
        from ocrd_utils import config

        ort.set_default_logger_severity(3)

        providers = ort.get_available_providers()
        if device:
            if ':' in device:
                for spec in device.split(','):
                    cat, dev = spec.split(':')
                    if fnmatchcase(model_category, cat):
                        device = dev
                        break
            if ':' in device:
                self.logger.warning("missing device specification for model type %s", model_category)
                gpu = 0
            elif device == 'CPU':
                gpu = -1
            else:
                assert device.startswith('GPU')
                gpu = int(device[3:] or "0")
        else:
            gpu = 0 # try first allowable
        # make runtime-configurable
        if override_providers := os.environ.get('EYNOLLAH_ONNX_EP', ''):
            override_providers = override_providers.split(',')
            providers = [provider for provider in providers
                         if provider[:-17] in override_providers]
        # configure and prioritise
        if 'AzureExecutionProvider' in providers:
            providers.remove('AzureExecutionProvider')
        if 'CUDAExecutionProvider' in providers:
            providers.remove('CUDAExecutionProvider')
            if gpu >= 0:
                providers = [('CUDAExecutionProvider', {
                    'device_id': gpu,
                    # 'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': MODEL_VRAM_LIMITS[model_category] * 1024 * 1024,
                    # 'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    #'cudnn_conv_use_max_workspace': 0,
                    # 'do_copy_in_default_stream': True,
                    # enable_cuda_graph
                    # cudnn_conv1d_pad_to_nc1d
                    # prefer_nhwc
                    # tunable_op_enable
                    # tunable_op_tuning_enable
                    # tunable_op_max_tuning_duration_ms
                    # use_ep_level_unified_stream
                    # enable_skip_layer_norm_strict_mode
                    # ...
                })] + providers
        if 'TensorrtExecutionProvider' in providers:
            providers.remove('TensorrtExecutionProvider')
            if gpu >= 0:
                providers = [('TensorrtExecutionProvider', {
                    'device_id': gpu,
                    'trt_max_workspace_size': MODEL_VRAM_LIMITS[model_category] * 1024 * 1024,
                    # 'trt_fp16_enable': True,
                    # trt_bf16_enable
                    'trt_engine_cache_enable': True,
                    'trt_timing_cache_enable': True,
                    'trt_engine_cache_path': config.XDG_CONFIG_HOME,
                    'trt_timing_cache_path': config.XDG_CONFIG_HOME,
                    # ...
                    # trt_engine_hw_compatible
                    # trt_engine_cache_prefix
                    # trt_onnx_model_folder_path
                    # trt_ep_context_file_path
                    # trt_cuda_graph_enable
                    # trt_profile_opt_shapes
                    # trt_profile_min_shapes
                    # trt_profile_max_shapes
                    # trt_builder_optimization_level
                    # trt_build_heuristics_enable
                    # trt_sparsity_enable
                    # trt_weight_stripped_engine_enable
                    # trt_dla_core
                    # trt_dla_enable
                    # trt_min_subgraph_size
                    # trt_ep_context_embed_mode
                })] + providers
        provider0 = providers[0]
        if isinstance(provider0, tuple):
            provider0 = provider0[0]
        self.logger.info("using %s with ONNX provider %s for model %s",
                         "GPU %d" % gpu if (gpu >= 0 and not
                                            provider0.startswith("CPU"))
                         else "CPU",
                         provider0[:-17], model_category)
        model = ort.InferenceSession(
            model_path,
            providers=providers)
        model_inputs = [model_input.name
                        for model_input in model.get_inputs()]
        model_outputs = [model_output.name
                         for model_output in model.get_outputs()]
        def predict_onnx(inputs):
            if len(model_inputs) == 1:
                inputs = [inputs]
            outputs = model.run(model_outputs, {
                model_input:
                input_data.astype(
                    # models expect data_type() == 'tensor(float)', but np.float16 is 'tensor(float16)'
                    # FIXME: do this dynamically (but how to convert .type to np.dtype?)
                    np.float32 if input_data.dtype in [np.float16, np.float64] else
                    input_data.dtype)
                for model_input, input_data in zip(model_inputs, inputs)
            })
            if len(model_outputs) == 1:
                outputs = outputs[0]
            return outputs
        model.predict_on_batch = predict_onnx
        input_spec = model.get_inputs()
        input_spec = [i.shape for i in input_spec]
        if len(input_spec) > 1:
            model.input_shape = tuple(input_spec)
        else:
            model.input_shape = input_spec[0]

        return model

    def _load_trocr_model(self, model_path, device: str = "") -> AnyModel:
        """
        Load OCR model
        """
        from transformers import VisionEncoderDecoderModel, TrOCRProcessor
        import numpy as np

        device = self._configure_torch_device('ocr', device=device)
        proc = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        assert isinstance(model, VisionEncoderDecoderModel)

        model.to(device)
        def predict_torch(inputs):
            output = model.generate(
                proc(inputs, return_tensors="pt").pixel_values.to(device),
                # beam search instead of greedy decoding:
                num_beams=4,
                # also return probability
                output_scores=True,
                return_dict_in_generate=True)
            if output.sequences_scores is not None:
                # log-prob averaged over length
                conf = output.sequences_scores.exp().clamp(0.0, 1.0).cpu().numpy()
            else:
                conf = np.ones(len(output.sequences), dtype=float)
            text = proc.batch_decode(
                output.sequences,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            # we must convert to ndarray for Predictor resultq to work
            text = np.array(text)
            return text, conf
        model.predict_on_batch = predict_torch
        # not actually needed (image processor does resize itself)
        # no batch dimension (images passed as list w/ varying shapes)
        model.input_shape = (None,
                             None,
                             len(proc.image_processor.image_mean))
        return model

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
        if hasattr(self, '_loaded') and self._loaded:
            for needle in list(self._loaded.keys()):
                if isinstance(self._loaded[needle], Predictor):
                    self._loaded[needle].shutdown()
