from eynollah.model_zoo import EynollahModelZoo

def test_trocr1(
    model_dir,
):
    model_zoo = EynollahModelZoo(model_dir)
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        model_zoo.load_models('trocr_processor',
                              ('ocr', 'tr'))
        proc = model_zoo.get('trocr_processor')
        assert isinstance(proc, TrOCRProcessor)
        model = model_zoo.get('ocr')
        assert isinstance(model, VisionEncoderDecoderModel)
    except ImportError:
        pass
