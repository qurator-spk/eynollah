from copy import deepcopy
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple,  List, Type, TypeVar, Union

from keras.layers import StringLookup
from keras.models import Model as KerasModel, load_model
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from eynollah.patch_encoder import PatchEncoder, Patches

AnyModel = Union[VisionEncoderDecoderModel, TrOCRProcessor, KerasModel, List]
T = TypeVar('T')

# NOTE: This needs to change whenever models change
ZENODO = "https://zenodo.org/records/17295988/files"
MODELS_VERSION = "v0_7_0"

def dist_url(dist_name: str) -> str:
    return f'{ZENODO}/models_{dist_name}_${MODELS_VERSION}.zip'

@dataclass
class EynollahModelSpec():
    """
    Describing a single model abstractly.
    """
    category: str
    # Relative filename to the models_eynollah directory in the dists
    filename: str
    # The smallest model distribution containing this model (link to Zenodo)
    dist: str
    type: Type[AnyModel]
    variant: str = ''
    help: str = ''

class EynollahModelSpecSet():
    """
    List of all used models for eynollah.
    """
    specs: List[EynollahModelSpec]

    def __init__(self, specs: List[EynollahModelSpec]) -> None:
        self.specs = specs
        self.categories: Set[str] = set([spec.category for spec in self.specs])
        self.variants: Dict[str, Set[str]] = {
            spec.category: set([x.variant for x in self.specs if x.category == spec.category])
            for spec in self.specs
        }
        self._index_category_variant: Dict[Tuple[str, str], EynollahModelSpec] = {
            (spec.category, spec.variant): spec
            for spec in self.specs
        }

    def asdict(self) -> Dict[str, Dict[str, str]]:
        return {
            spec.category: {
                spec.variant: spec.filename
            }
            for spec in self.specs
        }

    def get(self, category: str, variant: str) -> EynollahModelSpec:
        if category not in self.categories:
            raise ValueError(f"Unknown category '{category}', must be one of {self.categories}")
        if variant not in self.variants[category]:
            raise ValueError(f"Unknown variant {variant} for {category}. Known variants: {self.variants[category]}")
        return self._index_category_variant[(category, variant)]

DEFAULT_MODEL_SPECS = EynollahModelSpecSet([# {{{

    EynollahModelSpec(
        category="enhancement",
        variant='',
        filename="models_eynollah/eynollah-enhancement_20210425",
        dist=dist_url("enhancement"),
        type=KerasModel,
    ),
    
    EynollahModelSpec(
        category="binarization",
        variant='',
        filename="models_eynollah/eynollah-binarization_20210425",
        dist=dist_url("binarization"),
        type=KerasModel,
    ),
    
    EynollahModelSpec(
        category="binarization_multi_1",
        variant='',
        filename="models_eynollah/saved_model_2020_01_16/model_bin1",
        dist=dist_url("binarization"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="binarization_multi_2",
        variant='',
        filename="models_eynollah/saved_model_2020_01_16/model_bin2",
        dist=dist_url("binarization"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="binarization_multi_3",
        variant='',
        filename="models_eynollah/saved_model_2020_01_16/model_bin3",
        dist=dist_url("binarization"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="binarization_multi_4",
        variant='',
        filename="models_eynollah/saved_model_2020_01_16/model_bin4",
        dist=dist_url("binarization"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="col_classifier",
        variant='',
        filename="models_eynollah/eynollah-column-classifier_20210425",
        dist=dist_url("layout"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="page",
        variant='',
        filename="models_eynollah/model_eynollah_page_extraction_20250915",
        dist=dist_url("layout"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="region",
        variant='',
        filename="models_eynollah/eynollah-main-regions-ensembled_20210425",
        dist=dist_url("layout"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="region",
        variant='extract_only_images',
        filename="models_eynollah/eynollah-main-regions_20231127_672_org_ens_11_13_16_17_18",
        dist=dist_url("layout"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="region",
        variant='light',
        filename="models_eynollah/eynollah-main-regions_20220314",
        dist=dist_url("layout"),
        help="early layout",
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="region_p2",
        variant='',
        filename="models_eynollah/eynollah-main-regions-aug-rotation_20210425",
        dist=dist_url("layout"),
        help="early layout, non-light, 2nd part",
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="region_1_2",
        variant='',
        #filename="models_eynollah/modelens_12sp_elay_0_3_4__3_6_n",
        #filename="models_eynollah/modelens_earlylayout_12spaltige_2_3_5_6_7_8",
        #filename="models_eynollah/modelens_early12_sp_2_3_5_6_7_8_9_10_12_14_15_16_18",
        #filename="models_eynollah/modelens_1_2_4_5_early_lay_1_2_spaltige",
        #filename="models_eynollah/model_3_eraly_layout_no_patches_1_2_spaltige",
        filename="models_eynollah/modelens_e_l_all_sp_0_1_2_3_4_171024",
        dist=dist_url("layout"),
        help="early layout, light, 1-or-2-column",
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="region_fl_np",
        variant='',
        #'filename="models_eynollah/modelens_full_lay_1_3_031124",
        #'filename="models_eynollah/modelens_full_lay_13__3_19_241024",
        #'filename="models_eynollah/model_full_lay_13_241024",
        #'filename="models_eynollah/modelens_full_lay_13_17_231024",
        #'filename="models_eynollah/modelens_full_lay_1_2_221024",
        #'filename="models_eynollah/eynollah-full-regions-1column_20210425",
        filename="models_eynollah/modelens_full_lay_1__4_3_091124",
        dist=dist_url("layout"),
        help="full layout / no patches",
        type=KerasModel,
    ),

    # FIXME: Why is region_fl and region_fl_np the same model?
    EynollahModelSpec(
        category="region_fl",
        variant='',
        # filename="models_eynollah/eynollah-full-regions-3+column_20210425",
        # filename="models_eynollah/model_2_full_layout_new_trans",
        # filename="models_eynollah/modelens_full_lay_1_3_031124",
        # filename="models_eynollah/modelens_full_lay_13__3_19_241024",
        # filename="models_eynollah/model_full_lay_13_241024",
        # filename="models_eynollah/modelens_full_lay_13_17_231024",
        # filename="models_eynollah/modelens_full_lay_1_2_221024",
        # filename="models_eynollah/modelens_full_layout_24_till_28",
        # filename="models_eynollah/model_2_full_layout_new_trans",
        filename="models_eynollah/modelens_full_lay_1__4_3_091124",
        dist=dist_url("layout"),
        help="full layout / with patches",
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="reading_order",
        variant='',
        #filename="models_eynollah/model_mb_ro_aug_ens_11",
        #filename="models_eynollah/model_step_3200000_mb_ro",
        #filename="models_eynollah/model_ens_reading_order_machine_based",
        #filename="models_eynollah/model_mb_ro_aug_ens_8",
        #filename="models_eynollah/model_ens_reading_order_machine_based",
        filename="models_eynollah/model_eynollah_reading_order_20250824",
        dist=dist_url("layout"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="textline",
        variant='',
        #filename="models_eynollah/modelens_textline_1_4_16092024",
        #filename="models_eynollah/model_textline_ens_3_4_5_6_artificial",
        #filename="models_eynollah/modelens_textline_1_3_4_20240915",
        #filename="models_eynollah/model_textline_ens_3_4_5_6_artificial",
        #filename="models_eynollah/modelens_textline_9_12_13_14_15",
        #filename="models_eynollah/eynollah-textline_20210425",
        filename="models_eynollah/modelens_textline_0_1__2_4_16092024",
        dist=dist_url("layout"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="textline",
        variant='light',
        #filename="models_eynollah/eynollah-textline_light_20210425",
        filename="models_eynollah/modelens_textline_0_1__2_4_16092024",
        dist=dist_url("layout"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="table",
        variant='',
        filename="models_eynollah/eynollah-tables_20210319",
        dist=dist_url("layout"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="table",
        variant='light',
        filename="models_eynollah/modelens_table_0t4_201124",
        dist=dist_url("layout"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="ocr",
        variant='',
        filename="models_eynollah/model_eynollah_ocr_cnnrnn_20250930",
        dist=dist_url("ocr"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="num_to_char",
        variant='',
        filename="models_eynollah/characters_org.txt",
        dist=dist_url("ocr"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="characters",
        variant='',
        filename="models_eynollah/characters_org.txt",
        dist=dist_url("ocr"),
        type=List,
    ),

    EynollahModelSpec(
        category="ocr",
        variant='tr',
        filename="models_eynollah/model_eynollah_ocr_trocr_20250919",
        dist=dist_url("trocr"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="trocr_processor",
        variant='',
        filename="models_eynollah/microsoft/trocr-base-printed",
        dist=dist_url("trocr"),
        type=KerasModel,
    ),

    EynollahModelSpec(
        category="trocr_processor",
        variant='htr',
        filename="models_eynollah/microsoft/trocr-base-handwritten",
        dist=dist_url("trocr"),
        type=TrOCRProcessor,
    ),

])# }}}

class EynollahModelZoo():
    """
    Wrapper class that handles storage and loading of models for all eynollah runners.
    """
    model_basedir: Path
    specs: EynollahModelSpecSet

    def __init__(
        self,
        basedir: str,
        model_overrides: Optional[List[Tuple[str, str, str]]]=None,
    ) -> None:
        self.model_basedir = Path(basedir)
        self.logger = logging.getLogger('eynollah.model_zoo')
        self.specs = deepcopy(DEFAULT_MODEL_SPECS)
        if model_overrides:
            self.override_models(*model_overrides)
        self._loaded: Dict[str, AnyModel] = {}

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
    ) -> AnyModel:
        """
        Load any model
        """
        model_path = self.model_path(model_category, model_variant)
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

    def _load_ocr_model(self, variant: str) -> AnyModel:
        """
        Load OCR model
        """
        ocr_model_dir = self.model_path('ocr', variant)
        if variant == 'tr':
            return VisionEncoderDecoderModel.from_pretrained(ocr_model_dir)
        else:
            ocr_model = load_model(ocr_model_dir, compile=False)
            assert isinstance(ocr_model, KerasModel)
            return KerasModel(
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
            'versions': self.specs,
        }, indent=2))

    def shutdown(self):
        """
        Ensure that a loaded models is not referenced by ``self._loaded`` anymore
        """
        if hasattr(self, '_loaded') and getattr(self, '_loaded'):
            for needle in self._loaded:
                if self._loaded[needle]:
                    del self._loaded[needle]

