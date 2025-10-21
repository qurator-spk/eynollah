from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple,  List, Type, TypeVar, Union
from copy import deepcopy

from keras.layers import StringLookup
from keras.models import Model, load_model
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from eynollah.patch_encoder import PatchEncoder, Patches

SomeEynollahModel = Union[VisionEncoderDecoderModel, TrOCRProcessor, Model, List]
T = TypeVar('T')

# Dict mapping model_category to dict mapping variant (default is '') to Path
DEFAULT_MODEL_VERSIONS: Dict[str, Dict[str, str]] = {

    "enhancement": {
        '': "eynollah-enhancement_20210425"
    },

    "binarization": {
        '': "eynollah-binarization_20210425"
    },

    "binarization_multi_1": {
        '': "saved_model_2020_01_16/model_bin1",
    },
    "binarization_multi_2": {
        '': "saved_model_2020_01_16/model_bin2",
    },
    "binarization_multi_3": {
        '': "saved_model_2020_01_16/model_bin3",
    },
    "binarization_multi_4": {
        '': "saved_model_2020_01_16/model_bin4",
    },

    "col_classifier": {
        '': "eynollah-column-classifier_20210425",
    },

    "page": {
        '': "model_eynollah_page_extraction_20250915",
    },

    # TODO: What is this commented out model?
    #?: "eynollah-main-regions-aug-scaling_20210425",

    # early layout
    "region": {
        '': "eynollah-main-regions-ensembled_20210425",
        'extract_only_images': "eynollah-main-regions_20231127_672_org_ens_11_13_16_17_18",
        'light': "eynollah-main-regions_20220314",
    },

    # early layout, non-light, 2nd part
    "region_p2": {
        '': "eynollah-main-regions-aug-rotation_20210425",
    },

    # early layout, light, 1-or-2-column
    "region_1_2": { 
        #'': "modelens_12sp_elay_0_3_4__3_6_n"
        #'': "modelens_earlylayout_12spaltige_2_3_5_6_7_8"
        #'': "modelens_early12_sp_2_3_5_6_7_8_9_10_12_14_15_16_18"
        #'': "modelens_1_2_4_5_early_lay_1_2_spaltige"
        #'': "model_3_eraly_layout_no_patches_1_2_spaltige"
        '': "modelens_e_l_all_sp_0_1_2_3_4_171024"
    },

    # full layout / no patches
    "region_fl_np": {
        #'': "modelens_full_lay_1_3_031124"
        #'': "modelens_full_lay_13__3_19_241024"
        #'': "model_full_lay_13_241024"
        #'': "modelens_full_lay_13_17_231024"
        #'': "modelens_full_lay_1_2_221024"
        #'': "eynollah-full-regions-1column_20210425"
        '': "modelens_full_lay_1__4_3_091124"
    },

    # full layout / with patches
    "region_fl": {
        #'': "eynollah-full-regions-3+column_20210425"
        #'': #"model_2_full_layout_new_trans"
        #'': "modelens_full_lay_1_3_031124"
        #'': "modelens_full_lay_13__3_19_241024"
        #'': "model_full_lay_13_241024"
        #'': "modelens_full_lay_13_17_231024"
        #'': "modelens_full_lay_1_2_221024"
        #'': "modelens_full_layout_24_till_28"
        #'': "model_2_full_layout_new_trans"
        '': "modelens_full_lay_1__4_3_091124",
    },

    "reading_order": {
        #'': "model_mb_ro_aug_ens_11"
        #'': "model_step_3200000_mb_ro"
        #'': "model_ens_reading_order_machine_based"
        #'': "model_mb_ro_aug_ens_8"
        #'': "model_ens_reading_order_machine_based"
        '': "model_eynollah_reading_order_20250824"
    },

    "textline": {
        #'light': "eynollah-textline_light_20210425"
        'light': "modelens_textline_0_1__2_4_16092024",
        #'': "modelens_textline_1_4_16092024"
        #'': "model_textline_ens_3_4_5_6_artificial"
        #'': "modelens_textline_1_3_4_20240915"
        #'': "model_textline_ens_3_4_5_6_artificial"
        #'': "modelens_textline_9_12_13_14_15"
        #'': "eynollah-textline_20210425"
         '': "modelens_textline_0_1__2_4_16092024"
    },

    "table": {
        'light': "modelens_table_0t4_201124",
        '': "eynollah-tables_20210319",
    },

    "ocr": {
        'tr': "model_eynollah_ocr_trocr_20250919",
        '': "model_eynollah_ocr_cnnrnn_20250930",
    },

    'trocr_processor': {
        '': 'microsoft/trocr-base-printed',
        'htr': "microsoft/trocr-base-handwritten",
    },

    'num_to_char': {
        '': 'characters_org.txt'
    },

    'characters': {
        '': 'characters_org.txt'
    },

}


class EynollahModelZoo():
    """
    Wrapper class that handles storage and loading of models for all eynollah runners.
    """
    model_basedir: Path
    model_versions: dict

    def __init__(
        self,
        basedir: str,
        model_overrides: Optional[List[Tuple[str, str, str]]]=None,
    ) -> None:
        self.model_basedir = Path(basedir)
        self.logger = logging.getLogger('eynollah.model_zoo')
        self.model_versions = deepcopy(DEFAULT_MODEL_VERSIONS)
        if model_overrides:
            self.override_models(*model_overrides)
        self._loaded: Dict[str, SomeEynollahModel] = {}

    def override_models(
        self,
        *model_overrides: Tuple[str, str, str],
    ):
        """
        Override the default model versions
        """
        for model_category, model_variant, model_filename in model_overrides:
            if model_category not in DEFAULT_MODEL_VERSIONS:
                raise ValueError(f"Unknown model_category '{model_category}', must be one of {DEFAULT_MODEL_VERSIONS.keys()}")
            if model_variant not in DEFAULT_MODEL_VERSIONS[model_category]:
                raise ValueError(f"Unknown variant {model_variant} for {model_category}. Known variants: {DEFAULT_MODEL_VERSIONS[model_category].keys()}")
            self.logger.warning(
                "Overriding default model %s ('%s' variant) from %s to %s",
                model_category,
                model_variant,
                DEFAULT_MODEL_VERSIONS[model_category][model_variant],
                model_filename
            )
            self.model_versions[model_category][model_variant] = model_filename

    def model_path(
        self,
        model_category: str,
        model_variant: str = '',
        model_filename: str = '',
        absolute: bool = True,
    ) -> Path:
        """
        Translate model_{type,variant,filename} tuple into an absolute (or relative) Path
        """
        if model_category not in DEFAULT_MODEL_VERSIONS:
            raise ValueError(f"Unknown model_category '{model_category}', must be one of {DEFAULT_MODEL_VERSIONS.keys()}")
        if model_variant not in DEFAULT_MODEL_VERSIONS[model_category]:
            raise ValueError(f"Unknown variant {model_variant} for {model_category}. Known variants: {DEFAULT_MODEL_VERSIONS[model_category].keys()}")
        if not model_filename:
            model_filename = DEFAULT_MODEL_VERSIONS[model_category][model_variant]
        if not Path(model_filename).is_absolute() and absolute:
            model_path = Path(self.model_basedir).joinpath(model_filename)
        else:
            model_path = Path(model_filename)
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
        model_filename: str = '',
    ) -> SomeEynollahModel:
        """
        Load any model
        """
        model_path = self.model_path(model_category, model_variant, model_filename)
        if model_path.suffix  == '.h5' and Path(model_path.stem).exists():
            # prefer SavedModel over HDF5 format if it exists
            model_path = Path(model_path.stem)
        if model_category == 'ocr':
            model = self._load_ocr_model(variant=model_variant)
        elif model_category == 'num_to_char':
            model = self._load_num_to_char()
        elif model_category == 'characters':
            model = self._load_characters()
        elif model_category == 'trocr_processor':
            return TrOCRProcessor.from_pretrained(self.model_path(...))
        else:
            try:
                model = load_model(model_path, compile=False)
            except Exception as e:
                self.logger.exception(e)
                model = load_model(model_path, compile=False, custom_objects={
                    "PatchEncoder": PatchEncoder, "Patches": Patches})
        self._loaded[model_category] = model
        return model # type: ignore

    def get(self, model_category: str, model_type: Optional[Type[T]]=None) -> T:
        if model_category not in self._loaded:
            raise ValueError(f'Model "{model_category} not previously loaded with "load_model(..)"')
        ret = self._loaded[model_category]
        if model_type:
            assert isinstance(ret, model_type)
        return ret # type: ignore # FIXME: convince typing that we're returning generic type

    def _load_ocr_model(self, variant: str) -> SomeEynollahModel:
        """
        Load OCR model
        """
        ocr_model_dir = Path(self.model_basedir, self.model_versions["ocr"][variant])
        if variant == 'tr':
            return VisionEncoderDecoderModel.from_pretrained(ocr_model_dir)
        else:
            ocr_model = load_model(ocr_model_dir, compile=False)
            assert isinstance(ocr_model, Model)
            return Model(
                ocr_model.get_layer(name = "image").input,    # type: ignore
                ocr_model.get_layer(name = "dense2").output)  # type: ignore

    def _load_characters(self) -> List[str]:
        """
        Load encoding for OCR
        """
        with open(self.model_path('ocr') / self.model_path('num_to_char', absolute=False), "r") as config_file:
            return json.load(config_file)
                
    def _load_num_to_char(self) -> StringLookup:
        """
        Load decoder for OCR
        """
        characters = self._load_characters()
        # Mapping characters to integers.
        char_to_num = StringLookup(vocabulary=characters, mask_token=None)
        # Mapping integers back to original characters.
        return StringLookup(
            vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

    def __str__(self):
        return str(json.dumps({
            'basedir': str(self.model_basedir),
            'versions': self.model_versions,
        }, indent=2))

    def shutdown(self):
        """
        Ensure that a loaded models is not referenced by ``self._loaded`` anymore
        """
        if hasattr(self, '_loaded') and getattr(self, '_loaded'):
            for needle in self._loaded:
                if self._loaded[needle]:
                    del self._loaded[needle]

