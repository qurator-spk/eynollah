from .specs import EynollahModelSpec, EynollahModelSpecSet

# NOTE: This needs to change whenever models/versions change
ZENODO = "https://zenodo.org/records/17295988/files"
MODELS_VERSION = "v0_7_0"

def dist_url(dist_name: str="layout") -> str:
    return f'{ZENODO}/models_{dist_name}_{MODELS_VERSION}.zip'

DEFAULT_MODEL_SPECS = EynollahModelSpecSet([

    EynollahModelSpec(
        category="enhancement",
        variant='',
        filename="models_eynollah/eynollah-enhancement_20210425",
        dist_url=dist_url(),
        type='Keras',
    ),

    EynollahModelSpec(
        category="binarization",
        variant='hybrid',
        filename="models_eynollah/eynollah-binarization-hybrid_20230504/model_bin_hybrid_trans_cnn_sbb_ens",
        dist_url=dist_url(),
        type='Keras',
    ),
    
    EynollahModelSpec(
        category="binarization",
        variant='20210309',
        filename="models_eynollah/eynollah-binarization_20210309",
        dist_url=dist_url("extra"),
        type='Keras',
    ),
    
    EynollahModelSpec(
        category="binarization",
        variant='',
        filename="models_eynollah/eynollah-binarization_20210425",
        dist_url=dist_url("extra"),
        type='Keras',
    ),

    EynollahModelSpec(
        category="col_classifier",
        variant='',
        filename="models_eynollah/eynollah-column-classifier_20210425",
        dist_url=dist_url(),
        type='Keras',
    ),

    EynollahModelSpec(
        category="page",
        variant='',
        filename="models_eynollah/model_eynollah_page_extraction_20250915",
        dist_url=dist_url(),
        type='Keras',
    ),

    EynollahModelSpec(
        category="region",
        variant='',
        filename="models_eynollah/eynollah-main-regions-ensembled_20210425",
        dist_url=dist_url(),
        type='Keras',
    ),

    EynollahModelSpec(
        category="extract_images",
        variant='',
        filename="models_eynollah/eynollah-main-regions_20231127_672_org_ens_11_13_16_17_18",
        dist_url=dist_url(),
        type='Keras',
    ),

    EynollahModelSpec(
        category="region",
        variant='',
        filename="models_eynollah/eynollah-main-regions_20220314",
        dist_url=dist_url(),
        help="early layout",
        type='Keras',
    ),

    EynollahModelSpec(
        category="region_p2",
        variant='non-light',
        filename="models_eynollah/eynollah-main-regions-aug-rotation_20210425",
        dist_url=dist_url('extra'),
        help="early layout, non-light, 2nd part",
        type='Keras',
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
        dist_url=dist_url("layout"),
        help="early layout, light, 1-or-2-column",
        type='Keras',
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
        dist_url=dist_url(),
        help="full layout / no patches",
        type='Keras',
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
        dist_url=dist_url(),
        help="full layout / with patches",
        type='Keras',
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
        dist_url=dist_url(),
        type='Keras',
    ),

    EynollahModelSpec(
        category="textline",
        variant='non-light',
        #filename="models_eynollah/modelens_textline_1_4_16092024",
        #filename="models_eynollah/model_textline_ens_3_4_5_6_artificial",
        #filename="models_eynollah/modelens_textline_1_3_4_20240915",
        #filename="models_eynollah/model_textline_ens_3_4_5_6_artificial",
        #filename="models_eynollah/modelens_textline_9_12_13_14_15",
        #filename="models_eynollah/eynollah-textline_20210425",
        filename="models_eynollah/modelens_textline_0_1__2_4_16092024",
        dist_url=dist_url('extra'),
        type='Keras',
    ),

    EynollahModelSpec(
        category="textline",
        variant='',
        #filename="models_eynollah/eynollah-textline_light_20210425",
        filename="models_eynollah/modelens_textline_0_1__2_4_16092024",
        dist_url=dist_url(),
        type='Keras',
    ),

    EynollahModelSpec(
        category="table",
        variant='non-light',
        filename="models_eynollah/eynollah-tables_20210319",
        dist_url=dist_url('extra'),
        type='Keras',
    ),

    EynollahModelSpec(
        category="table",
        variant='',
        filename="models_eynollah/modelens_table_0t4_201124",
        dist_url=dist_url(),
        type='Keras',
    ),

    EynollahModelSpec(
        category="ocr",
        variant='',
        filename="models_eynollah/model_eynollah_ocr_cnnrnn_20250930",
        dist_url=dist_url("ocr"),
        type='Keras',
    ),

    EynollahModelSpec(
        category="ocr",
        variant='degraded',
        filename="models_eynollah/model_eynollah_ocr_cnnrnn__degraded_20250805/",
        help="slightly better at degraded Fraktur",
        dist_url=dist_url("ocr"),
        type='Keras',
    ),

    EynollahModelSpec(
        category="num_to_char",
        variant='',
        filename="characters_org.txt",
        dist_url=dist_url("ocr"),
        type='decoder',
    ),

    EynollahModelSpec(
        category="characters",
        variant='',
        filename="characters_org.txt",
        dist_url=dist_url("ocr"),
        type='List[str]',
    ),

    EynollahModelSpec(
        category="ocr",
        variant='tr',
        filename="models_eynollah/model_eynollah_ocr_trocr_20250919",
        dist_url=dist_url("ocr"),
        help='much slower transformer-based',
        type='Keras',
    ),

    EynollahModelSpec(
        category="trocr_processor",
        variant='',
        filename="models_eynollah/model_eynollah_ocr_trocr_20250919",
        dist_url=dist_url("ocr"),
        type='TrOCRProcessor',
    ),

    EynollahModelSpec(
        category="trocr_processor",
        variant='htr',
        filename="models_eynollah/microsoft/trocr-base-handwritten",
        dist_url=dist_url("extra"),
        type='TrOCRProcessor',
    ),

])
